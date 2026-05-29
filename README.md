# Medical Natural Language Autoencoders

This repository contains the code and artifacts for **Do Medical QA Benchmarks
Measure Medically Grounded Reasoning? A Natural Language Autoencoder Study of
Latent Explanations**.

- Paper: [ResearchGate preprint](https://www.researchgate.net/publication/405436619_Do_Medical_QA_Benchmarks_Measure_Medically_Grounded_Reasoning_A_Natural_Language_Autoencoder_Study_of_Latent_Explanations)
- DOI: [10.13140/RG.2.2.22314.79044](https://doi.org/10.13140/RG.2.2.22314.79044)

## Overview

Medical QA benchmarks are often reported as evidence that a language model has
medical knowledge or clinical reasoning ability. This project treats that as a
measurement problem: final-answer accuracy is useful, but it may not tell us
whether the model's internal representation contains medically grounded
reasoning.

The project applies **Natural Language Autoencoders (NLAs)** to medical
multiple-choice QA. NLAs were introduced in the original Natural Language
Autoencoders work as a way to decode model activations into natural language and
reconstruct activations from that text. This repository builds on that released
NLA infrastructure and asks whether decoded latent explanations provide
auxiliary validity evidence for medical QA evaluation.

The completed report run evaluates:

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- NLA checkpoint: `kitft/nla-qwen2.5-7b-L20-av` with `kitft/nla-qwen2.5-7b-L20-ar`
- Dataset: 200 MedQA test items from `GBaker/MedQA-USMLE-4-options`
- Prompt variants: 3 per item, for 600 predictions and decoded explanations
- Scorers: deterministic `heuristic_v1` and `medgemma_judge_v1`

## Main Result

The final validated run produced:

| Metric | Value |
| --- | ---: |
| Items | 200 |
| Prompt-variant predictions | 600 |
| Decode parse-ok rate | 100% |
| CJK injection warnings | 0 |
| Answer accuracy | 57.5% |
| Mean reconstruction cosine | 0.828 |
| Heuristic aligned-explanation rate | 5.5% |
| MedGemma judge aligned-explanation rate | 77.7% |

The central finding is that final-answer accuracy and NLA explanation quality
are different measurements, and the estimated aligned-explanation rate depends
strongly on the scoring instrument. The report treats NLA explanations as
auxiliary measurements, not literal model thoughts or clinical explanations.

## Repository Layout

```text
nla/mednla/          MedNLA data schemas, probing, decoding, scoring, analysis,
                     validation, pipeline, and reporting helpers
scripts/mednla/      CLI entrypoints for T1-T9 pipeline stages
configs/mednla/      MedNLA pilot and report-run configs
docs/mednla.md       End-to-end local pipeline documentation
vast/                Vast/SGLang runtime notes and cost-control workflow
requirements/        Stage-specific runtime constraints
.tex/                LaTeX report source and Overleaf-ready bundle
```

The original upstream NLA training and inference code remains in the repository
because this project builds on it. The MedNLA additions live primarily under
`nla/mednla/`, `scripts/mednla/`, `configs/mednla/`, and `docs/mednla.md`.

## Quick Start

Install the package in editable mode:

```bash
python -m pip install -e .
```

For the full pipeline, use the detailed recipe in
[`docs/mednla.md`](docs/mednla.md). The main stage commands are:

```bash
python scripts/mednla/prepare_items.py --config configs/mednla/pilot_qwen7b_medqa.yaml --output runs/mednla/pilot_qwen7b_medqa/items.jsonl
python scripts/mednla/run_base_probes.py --config configs/mednla/pilot_qwen7b_medqa.yaml --items runs/mednla/pilot_qwen7b_medqa/items.jsonl --model qwen7b --predictions-out runs/mednla/pilot_qwen7b_medqa/predictions.jsonl --activations-out runs/mednla/pilot_qwen7b_medqa/activations.parquet
python scripts/mednla/run_nla_decode.py --config configs/mednla/pilot_qwen7b_medqa.yaml --model qwen7b --activations runs/mednla/pilot_qwen7b_medqa/activations.parquet --out runs/mednla/pilot_qwen7b_medqa/decodes.jsonl --sglang-url http://127.0.0.1:30000
python scripts/mednla/score_explanations.py --config configs/mednla/pilot_qwen7b_medqa.yaml --items runs/mednla/pilot_qwen7b_medqa/items.jsonl --predictions runs/mednla/pilot_qwen7b_medqa/predictions.jsonl --decodes runs/mednla/pilot_qwen7b_medqa/decodes.jsonl --out runs/mednla/pilot_qwen7b_medqa/scores_heuristic.jsonl --scorer heuristic_v1
python scripts/mednla/analyze_results.py --config configs/mednla/pilot_qwen7b_medqa.yaml --items runs/mednla/pilot_qwen7b_medqa/items.jsonl --predictions runs/mednla/pilot_qwen7b_medqa/predictions.jsonl --decodes runs/mednla/pilot_qwen7b_medqa/decodes.jsonl --scores runs/mednla/pilot_qwen7b_medqa/scores_heuristic.jsonl --out-dir runs/mednla/pilot_qwen7b_medqa
```

For GPU/Vast execution, see [`vast/README.md`](vast/README.md). The runner
`scripts/mednla/run_eval_pipeline.py` can orchestrate prepared stages, but it
does not rent cloud instances or start SGLang automatically.

## Report Artifacts

The LaTeX report source and Overleaf upload bundle are in:

```text
.tex/Formatting_Instructions_For_NeurIPS_2025/
```

The generated report bundle includes summary tables, figure data, an audit
queue, and a claims checklist. Run artifacts are stored under `runs/`, which is
intentionally gitignored.

## Attribution

This project depends on the released Natural Language Autoencoders method,
checkpoints, and inference tooling:

- Anthropic, "Natural Language Autoencoders: Turning Claude's thoughts into text"
- `kitft/natural_language_autoencoders`
- `kitft/nla-inference`

For the medical QA study, cite:

```bibtex
@misc{masters2026medicalnla,
  author = {Masters, Blake},
  title = {Do Medical QA Benchmarks Measure Medically Grounded Reasoning? A Natural Language Autoencoder Study of Latent Explanations},
  year = {2026},
  doi = {10.13140/RG.2.2.22314.79044},
  url = {https://www.researchgate.net/publication/405436619_Do_Medical_QA_Benchmarks_Measure_Medically_Grounded_Reasoning_A_Natural_Language_Autoencoder_Study_of_Latent_Explanations}
}
```

For the original NLA work, cite:

```bibtex
@article{frasertaliente2026nla,
  author = {Fraser-Taliente, Kit and Kantamneni, Subhash and Ong, Euan and Mossing, Dan and Lu, Christina and Bogdan, Paul C. and Ameisen, Emmanuel and Chen, James and Kishylau, Dzmitry and Pearce, Adam and Tarng, Julius and Wu, Alex and Wu, Jeff and Zhang, Yang and Ziegler, Daniel M. and Hubinger, Evan and Batson, Joshua and Lindsey, Jack and Zimmerman, Samuel and Marks, Samuel},
  title = {Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  journal = {Transformer Circuits Thread},
  year = {2026},
  url = {https://transformer-circuits.pub/2026/nla/index.html}
}
```

## License

This repository preserves the upstream Apache-2.0 license. Released checkpoints
also inherit the licenses and use restrictions of their base models.
