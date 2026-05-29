# MedNLA evaluation harness

End-to-end pipeline for measuring whether base-model latent activations on medical QA items decode into medically grounded NLA explanations. The harness lives under `nla/mednla/`, `scripts/mednla/`, and `configs/mednla/`; it does not modify upstream training code.

## Prerequisites

- Python environment with the repo installed, plus the MedNLA runtime requirements for the stage being run.
- One CUDA-capable GPU with at least 24 GB VRAM for the Qwen2.5-7B pilot probe/decode path.
- Hugging Face access for:
  - `kitft/nla-qwen2.5-7b-L20-av`
  - `kitft/nla-qwen2.5-7b-L20-ar`
  - `google/medgemma-4b-it`, only for the optional MedGemma judge path.
- `huggingface-cli login` before gated model use. Accept the MedGemma terms at `https://huggingface.co/google/medgemma-4b-it` before running the judge.
- For Vast runs, follow `vast/README.md`. It contains the SGLang-template path, the 120-second SSH reachability gate, and the rule that auth tokens are passed by environment variable rather than config or logs.

## Artifact Layout

Use one output root per run:

```text
runs/mednla/{run_id}/
  config.yaml
  items.jsonl
  predictions.jsonl
  activations.parquet
  decodes.jsonl
  scores_heuristic.jsonl
  scores_judge.jsonl
  tables/
  figures_data/
  logs/
```

`runs/` is gitignored. Do not commit run artifacts.

## End-To-End Pipeline

The current pilot config is `configs/mednla/pilot_qwen7b_medqa.yaml`.

```bash
# 1) Normalize items.
python scripts/mednla/prepare_items.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --output runs/mednla/pilot_qwen7b_medqa/items.jsonl
```

```bash
# 2) Probe the base model for answers and activations.
python scripts/mednla/run_base_probes.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --items runs/mednla/pilot_qwen7b_medqa/items.jsonl \
  --model qwen7b \
  --predictions-out runs/mednla/pilot_qwen7b_medqa/predictions.jsonl \
  --activations-out runs/mednla/pilot_qwen7b_medqa/activations.parquet \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/probe_manifest.json
```

```bash
# 3) Start SGLang for the released AV checkpoint in a separate shell.
python -m sglang.launch_server \
  --model-path kitft/nla-qwen2.5-7b-L20-av \
  --port 30000 \
  --disable-radix-cache \
  --trust-remote-code
```

```bash
# 4) Confirm SGLang readiness before decoding.
python scripts/mednla/check_sglang_ready.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --sglang-url http://127.0.0.1:30000
```

```bash
# 5) Decode activations with the NLA actor. Add --no-critic on tight 24 GB hosts.
python scripts/mednla/run_nla_decode.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --activations runs/mednla/pilot_qwen7b_medqa/activations.parquet \
  --out runs/mednla/pilot_qwen7b_medqa/decodes.jsonl \
  --sglang-url http://127.0.0.1:30000 \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/decode_manifest.json
```

```bash
# 6a) Fast deterministic scoring.
python scripts/mednla/score_explanations.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --items runs/mednla/pilot_qwen7b_medqa/items.jsonl \
  --predictions runs/mednla/pilot_qwen7b_medqa/predictions.jsonl \
  --decodes runs/mednla/pilot_qwen7b_medqa/decodes.jsonl \
  --out runs/mednla/pilot_qwen7b_medqa/scores_heuristic.jsonl \
  --scorer heuristic_v1 \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/score_heuristic_manifest.json
```

```bash
# 6b) Optional MedGemma judge scoring.
python scripts/mednla/score_explanations.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --items runs/mednla/pilot_qwen7b_medqa/items.jsonl \
  --predictions runs/mednla/pilot_qwen7b_medqa/predictions.jsonl \
  --decodes runs/mednla/pilot_qwen7b_medqa/decodes.jsonl \
  --out runs/mednla/pilot_qwen7b_medqa/scores_judge.jsonl \
  --scorer medgemma_judge_v1 \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/score_judge_manifest.json
```

```bash
# 7) Analysis tables and figure data.
python scripts/mednla/analyze_results.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --items runs/mednla/pilot_qwen7b_medqa/items.jsonl \
  --predictions runs/mednla/pilot_qwen7b_medqa/predictions.jsonl \
  --decodes runs/mednla/pilot_qwen7b_medqa/decodes.jsonl \
  --scores runs/mednla/pilot_qwen7b_medqa/scores_heuristic.jsonl runs/mednla/pilot_qwen7b_medqa/scores_judge.jsonl \
  --out-dir runs/mednla/pilot_qwen7b_medqa \
  --manifest-out runs/mednla/pilot_qwen7b_medqa/analysis_manifest.json
```

For a heuristic-only smoke, pass only `scores_heuristic.jsonl` to `--scores`.

## Eval Runner

`run_eval_pipeline.py` is a local orchestrator for the same commands above. It
does not rent Vast instances, start SGLang, or manage schedulers. Use it after
the environment is ready and, for decode stages, after the operator has started
SGLang separately.

Dry-run the full GPU path before executing it:

```bash
python scripts/mednla/run_eval_pipeline.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --run-dir runs/mednla/pilot_qwen7b_medqa \
  --stages preflight,prepare,probe,check_sglang,decode,score_heuristic,analysis,validate \
  --sglang-url http://127.0.0.1:30000 \
  --quick-analysis \
  --dry-run
```

Run a CPU-safe local continuation after artifacts already exist:

```bash
python scripts/mednla/run_eval_pipeline.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --model qwen7b \
  --run-dir runs/mednla/pilot_qwen7b_medqa \
  --stages score_heuristic,analysis,validate \
  --quick-analysis \
  --resume
```

Validate a run directory directly:

```bash
python scripts/mednla/validate_run.py \
  --config configs/mednla/pilot_qwen7b_medqa.yaml \
  --run-dir runs/mednla/pilot_qwen7b_medqa \
  --model qwen7b \
  --require-analysis
```

The runner writes per-stage logs under `logs/`, passes `--manifest-out` where
the underlying stage supports it, and writes `run_validation.json` during the
validation stage. The optional `preflight` stage writes `preflight_runtime.json`
with Python, package, CUDA, command, and disk-space information. Use
`--auth-token-env`, not a raw token, for SGLang endpoints that require
authentication.

## Analysis Outputs

T6 writes:

- `tables/summary_by_model_dataset.csv`
- `tables/taxonomy_by_model_dataset.csv`
- `tables/prompt_stability.csv`
- `tables/rationale_alignment_by_correctness.csv`
- `tables/failure_cases.jsonl`
- `figures_data/taxonomy_stack.jsonl`
- `figures_data/accuracy_vs_aligned.jsonl`
- `figures_data/reconstruction_by_cell.jsonl`
- `_summary.json`

The CSVs and JSONL figure-data files are the analysis deliverable. Plot rendering is intentionally left to the report-writing workflow.

## Pilot Checklist

Before scaling:

- Three hand-written toy medical questions run end to end.
- Ten real items from one dataset run end to end for Qwen.
- Manual inspection confirms prompt text, answer parsing, activation shape, AV output, and joins.
- One intentionally shuffled option variant confirms answer-key remapping.
- One random activation decode confirms the AV server is not silently serving stale cache.
- T6 tables include bootstrap confidence intervals and at least two observed taxonomy cells.

## Scaling To 200-500 Items

- Increase `sample_size` in the YAML and keep the seed fixed for reproducibility.
- Do not assume the first 10 pilot items remain a prefix of a larger stratified sample.
- Use `--quick` only for code validation; final tables should use `analysis.bootstrap_resamples`, currently 1000.
- Estimated wall clock per 100 items x 3 variants on 1x H100: probe about 15 minutes, decode about 30 minutes, judge about 5 hours, analysis about 2 minutes.
- If `accuracy_ci_lo == accuracy_ci_hi` on the 10-item pilot, that can be degenerate pilot data. If it happens at full scale, investigate the input rows and bootstrap grouping.

## Claims To Support

Appropriate claims:

- "Accuracy and NLA explanation quality diverged in X% of correct answers."
- "Prompt variants changed NLA quality more or less often than final answer correctness."
- "Reconstruction quality was or was not associated with medically aligned explanations."
- "These findings suggest accuracy alone may be incomplete as a measure of medically grounded reasoning."

Claims to avoid:

- "The NLA reveals the model's true reasoning."
- "A weak NLA explanation proves shortcut behavior."
- "Gold-rationale mismatch proves medical invalidity."
- "The released general-domain NLA is validated for clinical reasoning."

## Known Failure Modes

- CJK output: the activation may be from the wrong layer, or SGLang may be routing token IDs instead of embeddings. Stop and inspect the T3 layer index, AV sidecar `d_model`, and SGLang readiness output.
- Heuristic and judge disagreement: expected. The heuristic is a sanity check, not a medical authority. Audit high-disagreement cases manually.
- `gold_rationale=None` for MedQA: MedQA has no rationale column, so `rationale_alignment` can be `null`.
- Subject sparsity: subject-level plots can be noisy or unavailable when too many rows lack a subject.
- Low reconstruction quality: treat aligned explanations with low reconstruction scores cautiously; the decode may not preserve enough activation information.
