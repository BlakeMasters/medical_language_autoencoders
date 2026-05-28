# Vast.ai Rehearsal Checklist

Goal: prove a rented GPU instance can run this repo's MedNLA smoke path and
return artifacts before we rely on Vast for full evals.

## Local Preflight

```bash
git status --short
python -m pytest tests/mednla -q
vastai show user
vastai show instances
```

Confirm `.env` and `env/` are ignored before any sync or commit:

```bash
git check-ignore -v .env env/
```

## Rent a Test Instance

Use the Qwen7B pilot profile from [vast_cli_reference.md](vast_cli_reference.md):

```bash
vastai search offers \
  'reliability>0.98 gpu_name in ["RTX 4090", "RTX 3090"] num_gpus=1 gpu_ram>=20 cuda_vers>=12.1 verified=true direct_port_count>=1 rentable=true rented=false' \
  --storage 120 \
  -o 'dlperf_usd-'

vastai create instance OFFER_ID \
  --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime \
  --disk 120 \
  --onstart-cmd "nvidia-smi" \
  --ssh \
  --direct
```

Poll until `running`. Destroy and retry if it reaches `exited`, `unknown`, or
`offline`.

## Remote Bootstrap

After SSH:

```bash
apt-get update
apt-get install -y git rsync tmux htop
cd /workspace/medical_language_autoencoders
python -m pip install -U pip
python -m pip install -e .
python -m pip install "sglang[all]>=0.5.6"
```

If the workspace was not cloned on the remote, sync it from local first using
the `rsync` command in [vast_cli_reference.md](vast_cli_reference.md).

## Project Smoke

```bash
cd /workspace/medical_language_autoencoders
mkdir -p runs/vast_rehearsal
python -m pytest tests/mednla -q
python scripts/mednla/prepare_items.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --output runs/vast_rehearsal/items.jsonl
head -n 2 runs/vast_rehearsal/items.jsonl
```

Expected result:

- MedNLA tests pass.
- `items.jsonl` has 10 rows for the pinned MedQA pilot config.
- Each row records the pinned Hugging Face revision in `source_metadata`.

## Optional NLA Inference Smoke

Run this in `tmux` because model download and SGLang startup can take time:

```bash
tmux new -s mednla
cd /workspace/medical_language_autoencoders
python -m sglang.launch_server \
  --model-path kitft/nla-qwen2.5-7b-L20-av \
  --port 30000 \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  > runs/vast_rehearsal/sglang.log 2>&1
```

In a second SSH shell:

```bash
cd /workspace/medical_language_autoencoders
python nla_inference.py kitft/nla-qwen2.5-7b-L20-av \
  --sglang-url http://localhost:30000 \
  --n 1 \
  > runs/vast_rehearsal/nla_inference_smoke.txt
```

If this OOMs on a 24 GB card, keep the rehearsal scoped to tests plus
`prepare_items.py` and rent a larger GPU for full evals.

## Artifact Return and Teardown

```bash
rsync -az root@HOST:/workspace/medical_language_autoencoders/runs/ ./runs/
vastai destroy instance INSTANCE_ID
vastai show instances
```

Success criteria:

- Local secrets were never uploaded.
- Remote CUDA is available.
- MedNLA tests and item prep succeed remotely.
- Artifacts are synced back under local `runs/`.
- The instance is destroyed and no unwanted rental remains active.
