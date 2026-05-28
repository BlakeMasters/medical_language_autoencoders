# Vast.ai Runbook for MedNLA Evals

Use Vast as an ephemeral GPU control plane for MedNLA pilot/evaluation runs. Keep
all credentials local, write experiment artifacts under `runs/`, sync artifacts
back, then destroy the instance unless there is a deliberate reason to preserve
the disk.

## Local Secrets

The repo ignores `env/`, `.env`, and `.env.*`. Keep secrets there or in the Vast
CLI config, never in tracked docs, logs, or writeups.

Expected local `.env` entries:

```bash
VAST_API_KEY=...
HF_TOKEN=...              # only needed for gated HF models
ANTHROPIC_API_KEY=...     # only needed for datagen stage2/provider runs
```

Load them in a local shell without printing values:

```bash
set -a
source .env
set +a
vastai set api-key "$VAST_API_KEY"
vastai show user
```

## Instance Profile

Default for the current MedNLA pilot:

- GPU: one RTX 4090 or RTX 3090 class card, `gpu_ram>=20`.
- Disk: 120 GB. Qwen7B, AV/AR checkpoints, SGLang, Hugging Face cache, and run
  artifacts can exceed a small default disk quickly.
- CUDA: `cuda_vers>=12.1`.
- Reliability: prefer `reliability>0.98`, `verified=true`, and direct SSH.

Use A100/H100-class instances only for larger datagen/training configs or if
the eval runner grows beyond the Qwen7B pilot footprint.

## SGLang Template Settings

Prefer the Vast SGLang Inference Engine template over a plain PyTorch runtime
image for T4 decode. The PyTorch runtime image can run T3, but it does not ship
the SGLang/CUDA toolchain in a compatible state for FlashInfer runtime kernels.

Use these template environment variables for the Qwen7B actor smoke:

```bash
SGLANG_MODEL=kitft/nla-qwen2.5-7b-L20-av
AUTO_PARALLEL=false
APT_PACKAGES=git rsync tmux htop
PIP_PACKAGES=pytest
SGLANG_ARGS=--trust-remote-code --disable-radix-cache --mem-fraction-static 0.75
```

For RTX 4090 hosts, prefer a CUDA 12.x SGLang image unless the host driver
clearly supports the selected CUDA major version. If SGLang startup hits CUDA
graph or JIT issues, append these flags to `SGLANG_ARGS`:

```bash
--disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule
```

The template serves SGLang on internal port `18000`. T4 currently defaults to
`http://localhost:30000`, so either point `nla_decode.sglang_url` at
`http://127.0.0.1:18000` on the remote config or launch a separate local SGLang
server on port `30000`.

## Standard Flow

1. Search and create an instance with [vast_cli_reference.md](vast_cli_reference.md).
2. Sync this repo or clone the pushed branch onto `/workspace/medical_language_autoencoders`.
3. Bootstrap Python dependencies and run the smoke checks in [vast_rehearsal.md](vast_rehearsal.md).
4. Run project eval commands from `tmux`; write outputs below `runs/mednla/...`.
5. Sync `runs/` back locally.
6. Destroy the instance and verify no unwanted rentals remain.

## Current Project Commands

Run these first on any new Vast instance:

```bash
cd /workspace/medical_language_autoencoders
python -m pytest tests/mednla -q
python scripts/mednla/prepare_items.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --output runs/mednla/pilot_qwen7b_medqa/items.jsonl
```

For NLA actor inference smoke testing:

```bash
python -m sglang.launch_server \
  --model-path kitft/nla-qwen2.5-7b-L20-av \
  --port 30000 \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --trust-remote-code

python nla_inference.py kitft/nla-qwen2.5-7b-L20-av \
  --sglang-url http://localhost:30000 \
  --n 1
```

When a full MedNLA eval runner is added, keep the same contract: checked-in
config under `configs/mednla/`, outputs under `runs/mednla/<run_id>/`, and a
small smoke command before scaling up.

## Operational Rules

- Do not paste API keys into issue comments, writeups, shell transcripts, or
  model prompts.
- Keep long remote commands inside `tmux`.
- Prefer `rsync`/`scp` after SSH is known; use `vastai copy` only when it is
  simpler.
- For failed `loading`, `unknown`, `offline`, or `exited` instances, destroy and
  retry with another offer.
- For CUDA OOM, reduce batch size or model scope before renting a larger class.
- Default cleanup is `destroy instance`, not `stop instance`.
