# Vast.ai CLI Reference for MedNLA

Use the CLI for normal MedNLA eval work. It keeps instance lifecycle actions
visible and easy to rerun.

## Setup

```bash
python3 -m pip install --user vastai
vastai --help
set -a && source .env && set +a
vastai set api-key "$VAST_API_KEY"
vastai show user
vastai create ssh-key ~/.ssh/id_ed25519.pub
```

If the SSH key already exists, the create step can fail harmlessly. Verify keys
in the Vast UI or proceed to instance creation if SSH is already configured.

## Search Offers

Primary Qwen7B pilot search:

```bash
vastai search offers \
  'reliability>0.98 gpu_name in ["RTX 4090", "RTX 3090"] num_gpus=1 gpu_ram>=20 cuda_vers>=12.1 verified=true direct_port_count>=1 rentable=true rented=false' \
  --storage 120 \
  -o 'dlperf_usd-'
```

Broader fallback if no 24 GB class card is available:

```bash
vastai search offers \
  'reliability>0.98 num_gpus=1 gpu_ram>=20 cuda_vers>=12.1 verified=true direct_port_count>=1 rentable=true rented=false' \
  --storage 120 \
  -o 'dlperf_usd-'
```

Use `--storage 120` in searches so displayed pricing reflects the disk size we
actually intend to rent.

## Create Instance

```bash
vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk 120 \
  --onstart-cmd "nvidia-smi" \
  --ssh \
  --direct
```

Save the returned `new_contract` as `INSTANCE_ID`.

## State and SSH

```bash
vastai show instance INSTANCE_ID
vastai show instances
vastai ssh-url INSTANCE_ID
ssh -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 \
  root@HOST -p PORT 'echo ssh_ready && nvidia-smi'
```

Poll every 15 seconds after creation. If direct SSH is not reachable within 120
seconds, destroy the instance and try another offer. Do not sync files, install
packages, or wait on a `loading` instance past this gate.

Remote smoke checks:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Instance state policy:

| State | Action |
| --- | --- |
| `loading` | Poll every 15 seconds; destroy if SSH is unreachable at 120 seconds. |
| `running` | SSH and run smoke checks. |
| `exited` | Destroy and retry with another offer. |
| `unknown` | Destroy unless it recovers quickly. |
| `offline` | Destroy and retry. |

## Sync Project Files

For a pushed branch, clone directly on the remote. For local/unpushed work,
sync the current workspace without secrets or run artifacts. Use `rsync` or
`scp` only after the direct SSH gate succeeds:

```bash
rsync -az --delete \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '.env.*' \
  --exclude 'env/' \
  --exclude 'runs/' \
  ./ root@HOST:/workspace/medical_language_autoencoders/
```

If using Windows/WSL and OpenSSH rejects a key on `/mnt/c`, copy it into WSL
first:

```bash
mkdir -p ~/.ssh
cp /mnt/c/Users/<you>/.ssh/id_ed25519 ~/.ssh/vast_mednla_ed25519
chmod 600 ~/.ssh/vast_mednla_ed25519
ssh -i ~/.ssh/vast_mednla_ed25519 root@HOST -p PORT
```

Download artifacts:

```bash
rsync -az root@HOST:/workspace/medical_language_autoencoders/runs/ ./runs/
```

## Cleanup

Destroy by default:

```bash
vastai destroy instance INSTANCE_ID
vastai show instances
```

Use `stop instance` only when intentionally preserving disk for a short pause:

```bash
vastai stop instance INSTANCE_ID
```

Always verify teardown with `vastai show instances`; failed instances can still
incur disk charges.
