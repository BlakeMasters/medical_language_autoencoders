# Clean Report Run Artifacts

This directory is the cleaned release copy of the final report run for:

> Do Medical QA Benchmarks Measure Medically Grounded Reasoning? A Natural Language Autoencoder Study of Latent Explanations

It contains the artifacts needed to inspect and reproduce the reported tables,
figures, validation checks, and audit queue for the Qwen2.5-7B-Instruct MedQA
200-item run.

Included:

- `items.jsonl`, `predictions.jsonl`, `activations.parquet`, and `decodes.jsonl`
- `scores_heuristic.jsonl` and `scores_judge.jsonl`
- stage manifests for probe, decode, scoring, and analysis
- `tables/` and `figures_data/` analysis outputs
- `report/` summary, audit queue, claims checklist, and artifact index
- `run_validation.json`

Deliberately omitted from this clean release:

- remote install logs and SGLang server logs
- transient PID files
- local machine preflight details
- optional MedGemma judge canary outputs
- secrets or local `.env` files

The original raw runtime directory used to assemble this clean copy was
`runs/vast_final_report/38347984_final/report_qwen_medqa_n200/`, which remains
local-only and gitignored.
