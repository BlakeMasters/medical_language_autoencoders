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

## Runtime Profiles

Use a plain CUDA/PyTorch runtime for T3 base probes and the Vast SGLang
Inference Engine image for T4 decode. Keep them separate unless a specific image
has already been proven to support both paths; installing `sglang[all]` into the
T3 environment can replace the torch/transformers stack.

For T3 on a PyTorch runtime:

```bash
cd /workspace/medical_language_autoencoders
python -m pip install -r requirements/mednla-vast-t3.txt
```

For T4 inside a SGLang-ready image:

```bash
cd /workspace/medical_language_autoencoders
python -m pip install -r requirements/mednla-vast-t4-sglang.txt
```

For a deliberate future T5 MedGemma judge run, use
`requirements/mednla-vast-t5.txt` in a separate process after accepting the
MedGemma Hugging Face terms. The immediate budget smoke validates live T4 decode
plus deterministic `heuristic_v1` scoring, not the live MedGemma judge.

## SGLang Template Settings

Prefer a two-phase SGLang launch for smoke tests: rent an SSH-ready SGLang image
first, then start the NLA actor server manually after repo sync. Avoid making
model download/server startup part of the Vast boot path; if it stalls, the
instance can remain in `loading` without reachable SSH.

If using the managed SGLang template anyway, use these environment variables for
the Qwen7B actor smoke:

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
`http://localhost:30000`; pass `--sglang-url http://127.0.0.1:18000` when using
the template. If the template requires a bearer token, pass it through an
environment variable with `--auth-token-env`; do not write tokens into YAML,
tracked docs, or logs.

## SSH Gate

Every Vast rental must pass an actual SSH test before any file sync or setup.
Poll at 15-second intervals and destroy the instance if SSH is not reachable
within 120 seconds of creation. Treat `loading` plus unreachable SSH as a boot
failure, not as a reason to wait indefinitely.

```bash
ssh -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 \
  root@HOST -p PORT 'echo ssh_ready && nvidia-smi'
```

Use `scp` or `rsync` only after this direct SSH check succeeds.

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
python scripts/mednla/check_sglang_ready.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --sglang-url http://127.0.0.1:18000
```

Then run T4 decode only after readiness passes:

```bash
python scripts/mednla/run_nla_decode.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --activations runs/mednla/pilot_qwen7b_medqa/activations.parquet \
  --out runs/mednla/pilot_qwen7b_medqa/decodes.jsonl \
  --no-critic \
  --sglang-url http://127.0.0.1:18000 \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/decode_manifest.json
```

For the minimal T5 smoke, score the returned decodes with the deterministic
scorer:

```bash
python scripts/mednla/score_explanations.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --items runs/mednla/pilot_qwen7b_medqa/items.jsonl \
  --predictions runs/mednla/pilot_qwen7b_medqa/predictions.jsonl \
  --decodes runs/mednla/pilot_qwen7b_medqa/decodes.jsonl \
  --out runs/mednla/pilot_qwen7b_medqa/scores_heuristic.jsonl \
  --scorer heuristic_v1 \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/score_manifest.json
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
- Destroy any instance that does not pass direct SSH within 120 seconds.
- For failed `loading`, `unknown`, `offline`, or `exited` instances, destroy and
  retry with another offer.
- For CUDA OOM, reduce batch size or model scope before renting a larger class.
- Default cleanup is `destroy instance`, not `stop instance`.
