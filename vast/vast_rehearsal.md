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

For the minimal T4/T5 smoke, prefer a two-phase SGLang launch:

1. Rent a SGLang-ready image with a trivial startup command.
2. Prove direct SSH within 120 seconds.
3. Sync the repo and artifacts.
4. Start the NLA actor SGLang server manually in `tmux`.

This avoids a boot path where model download/server startup keeps the instance
in `loading` with no SSH access.

If using the managed Vast SGLang template instead, configure it with:

```bash
SGLANG_MODEL=kitft/nla-qwen2.5-7b-L20-av
AUTO_PARALLEL=false
APT_PACKAGES=git rsync tmux htop
PIP_PACKAGES=pytest
SGLANG_ARGS=--trust-remote-code --disable-radix-cache --mem-fraction-static 0.75
```

If using an RTX 4090 host and SGLang reports CUDA graph or JIT failures, append
these flags to `SGLANG_ARGS` and restart the service:

```bash
--disable-cuda-graph --disable-piecewise-cuda-graph --disable-overlap-schedule
```

Use the Qwen7B pilot profile from [vast_cli_reference.md](vast_cli_reference.md)
when renting the GPU. If running only T3, a PyTorch runtime image is acceptable:

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

Poll every 15 seconds. Destroy and retry if it reaches `exited`, `unknown`, or
`offline`.

Hard SSH gate: destroy the instance if this command does not succeed within 120
seconds of creation, even if the Vast state still says `loading`:

```bash
ssh -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null \
  -o ConnectTimeout=10 \
  root@HOST -p PORT 'echo ssh_ready && nvidia-smi'
```

Only use `scp` or `rsync` after this direct SSH check succeeds.

## Remote Bootstrap

After SSH on a T3 PyTorch runtime:

```bash
apt-get update
apt-get install -y git rsync tmux htop
cd /workspace/medical_language_autoencoders
python -m pip install -U pip
python -m pip install -r requirements/mednla-vast-t3.txt
```

After SSH on the T4 SGLang image, install only this repo's bounded client/test
dependencies:

```bash
cd /workspace/medical_language_autoencoders
python scripts/mednla/vast_bootstrap.py --profile t4-sglang --require-existing-torch
```

The minimal budget smoke stops at live T4 decode plus deterministic T5
`heuristic_v1` scoring. Use `requirements/mednla-vast-t5.txt` only for a
separate, deliberate live MedGemma judge run.

Cost guardrails:

- Capture `vastai show user` before launch and after teardown.
- Do not run model download or SGLang startup in the instance boot command.
- Do not run broad T4 `pip install -r` commands that can reinstall CUDA/Torch.
- Destroy any instance that misses the 120-second direct SSH gate.

If the workspace was not cloned on the remote, sync it from local first using
the `rsync` command in [vast_cli_reference.md](vast_cli_reference.md).

## Project Smoke

```bash
cd /workspace/medical_language_autoencoders
mkdir -p runs/vast_rehearsal
python scripts/mednla/preflight_runtime.py --json-out runs/vast_rehearsal/preflight_runtime.json --require-torch
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

## SGLang Readiness

Start the NLA actor server manually in `tmux` if it is not already running:

```bash
tmux new -s sglang
python -m sglang.launch_server \
  --model-path kitft/nla-qwen2.5-7b-L20-av \
  --host 127.0.0.1 \
  --port 18000 \
  --trust-remote-code \
  --disable-radix-cache \
  --mem-fraction-static 0.72
```

Then verify the service is up before T4:

```bash
curl -sf http://127.0.0.1:18000/v1/models
curl -sf http://127.0.0.1:18000/model_info
```

The project T4 client calls native SGLang `/generate` with `input_embeds`, not
the OpenAI-compatible chat endpoint. Use the project readiness check before
running decode:

```bash
cd /workspace/medical_language_autoencoders
python scripts/mednla/check_sglang_ready.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --sglang-url http://127.0.0.1:18000
```

If the template requires API auth, read the token from the instance environment
and pass it without printing the value:

```bash
python scripts/mednla/check_sglang_ready.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --sglang-url http://127.0.0.1:18000 \
  --auth-token-env OPEN_BUTTON_TOKEN
```

## T4 Decode Smoke

Only run decode after readiness succeeds and a T3 run has produced
`activations.parquet`:

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

Then run deterministic T5 scoring:

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
