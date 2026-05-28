"""Analyze MedNLA score artifacts into tables and figure data."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.analysis import (
    AnalysisJoinError,
    accuracy_vs_aligned,
    failure_cases,
    join_rows,
    prompt_stability,
    rationale_alignment_by_correctness,
    reconstruction_by_cell,
    summary_by_model_dataset,
    taxonomy_by_model_dataset,
)
from nla.mednla.run_manifest import build_run_manifest, write_run_manifest
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, from_dict

logger = logging.getLogger("nla.mednla.analyze_results")

SUMMARY_COLUMNS = (
    "model_short_name",
    "dataset",
    "n_items",
    "n_predictions",
    "accuracy",
    "accuracy_ci_lo",
    "accuracy_ci_hi",
    "aligned_rate",
    "aligned_rate_ci_lo",
    "aligned_rate_ci_hi",
    "mean_reconstruction_cos",
    "mean_reconstruction_mse",
    "scorer",
)
TAXONOMY_COLUMNS = (
    "model_short_name",
    "dataset",
    "taxonomy_cell",
    "count",
    "proportion",
    "proportion_ci_lo",
    "proportion_ci_hi",
    "scorer",
)
PROMPT_STABILITY_COLUMNS = (
    "item_id",
    "model_short_name",
    "scorer",
    "n_variants",
    "answer_agreement",
    "correct_agreement",
    "quality_agreement",
    "medical_relevance_var",
    "shuffle_changed_answer",
)
RATIONALE_COLUMNS = (
    "correct",
    "scorer",
    "n_with_rationale",
    "mean_rationale_alignment",
    "sd_rationale_alignment",
    "fraction_alignment_ge_1",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _load_jsonl(path: Path, cls: type[Any]) -> list[Any]:
    rows: list[Any] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(from_dict(cls, orjson.loads(line)))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid {cls.__name__}: {exc}") from exc
    return rows


def _bootstrap_resamples(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    analysis_cfg = cfg.get("analysis", {})
    if analysis_cfg is None:
        analysis_cfg = {}
    if not isinstance(analysis_cfg, dict):
        raise ValueError("config analysis must be a mapping")
    if args.bootstrap_resamples is not None:
        return int(args.bootstrap_resamples)
    if args.quick:
        return int(analysis_cfg.get("quick_bootstrap_resamples", 200))
    return int(analysis_cfg.get("bootstrap_resamples", 1000))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row) + b"\n")


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


def _run(args: argparse.Namespace) -> int:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures_data"
    cfg = _load_yaml(config_path)
    n_resamples = _bootstrap_resamples(args, cfg)
    if n_resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")

    items = _load_jsonl(Path(args.items), MedItem)
    predictions = _load_jsonl(Path(args.predictions), Prediction)
    decodes = _load_jsonl(Path(args.decodes), DecodeRecord)
    scores: list[ScoreRecord] = []
    for score_path in args.scores:
        scores.extend(_load_jsonl(Path(score_path), ScoreRecord))

    joined = join_rows(items, predictions, scores, decodes)
    summary_rows = summary_by_model_dataset(joined, n_resamples=n_resamples, seed=args.seed)
    taxonomy_rows = taxonomy_by_model_dataset(joined, n_resamples=n_resamples, seed=args.seed)
    prompt_rows = prompt_stability(joined)
    rationale_rows = rationale_alignment_by_correctness(joined)
    failure_rows = failure_cases(joined)
    accuracy_rows = accuracy_vs_aligned(joined)
    reconstruction_rows = reconstruction_by_cell(joined)

    _write_csv(tables_dir / "summary_by_model_dataset.csv", summary_rows, SUMMARY_COLUMNS)
    _write_csv(tables_dir / "taxonomy_by_model_dataset.csv", taxonomy_rows, TAXONOMY_COLUMNS)
    _write_csv(tables_dir / "prompt_stability.csv", prompt_rows, PROMPT_STABILITY_COLUMNS)
    _write_csv(tables_dir / "rationale_alignment_by_correctness.csv", rationale_rows, RATIONALE_COLUMNS)
    _write_jsonl(tables_dir / "failure_cases.jsonl", failure_rows)
    _write_jsonl(figures_dir / "taxonomy_stack.jsonl", taxonomy_rows)
    _write_jsonl(figures_dir / "accuracy_vs_aligned.jsonl", accuracy_rows)
    _write_jsonl(figures_dir / "reconstruction_by_cell.jsonl", reconstruction_rows)

    outputs = {
        "summary_by_model_dataset": str(tables_dir / "summary_by_model_dataset.csv"),
        "taxonomy_by_model_dataset": str(tables_dir / "taxonomy_by_model_dataset.csv"),
        "prompt_stability": str(tables_dir / "prompt_stability.csv"),
        "rationale_alignment_by_correctness": str(tables_dir / "rationale_alignment_by_correctness.csv"),
        "failure_cases": str(tables_dir / "failure_cases.jsonl"),
        "taxonomy_stack": str(figures_dir / "taxonomy_stack.jsonl"),
        "accuracy_vs_aligned": str(figures_dir / "accuracy_vs_aligned.jsonl"),
        "reconstruction_by_cell": str(figures_dir / "reconstruction_by_cell.jsonl"),
        "summary": str(out_dir / "_summary.json"),
    }
    elapsed = time.perf_counter() - started
    summary = {
        "n_items": len({row["item_id"] for row in joined}),
        "n_predictions": len({row["prediction_id"] for row in joined}),
        "n_joined_rows": len(joined),
        "models": sorted({row["model_short_name"] for row in joined}),
        "datasets": sorted({row["dataset"] for row in joined}),
        "scorers": sorted({row["scorer"] for row in joined}),
        "wall_seconds": elapsed,
        "bootstrap_resamples": n_resamples,
        "outputs": outputs,
    }
    _write_summary(out_dir / "_summary.json", summary)
    if args.manifest_out:
        ended_at_utc = datetime.now(timezone.utc).isoformat()
        model_short_name = ",".join(summary["models"]) if summary["models"] else "unknown"
        manifest = build_run_manifest(
            stage="mednla_analysis",
            started_at_utc=started_at_utc,
            ended_at_utc=ended_at_utc,
            duration_sec=elapsed,
            config_path=config_path,
            model_short_name=model_short_name,
            args=vars(args),
            outputs=outputs,
            extra=summary,
        )
        write_run_manifest(args.manifest_out, manifest)

    logger.info(
        "wrote_analysis rows=%d models=%s datasets=%s scorers=%s out_dir=%s",
        len(joined),
        ",".join(summary["models"]),
        ",".join(summary["datasets"]),
        ",".join(summary["scorers"]),
        out_dir,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Analyze MedNLA score artifacts.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--items", required=True, help="Prepared MedNLA items JSONL.")
    parser.add_argument("--predictions", required=True, help="T3 predictions JSONL.")
    parser.add_argument("--decodes", required=True, help="T4 decodes JSONL.")
    parser.add_argument("--scores", required=True, nargs="+", help="One or more T5 scores JSONL files.")
    parser.add_argument("--out-dir", required=True, help="Output run directory.")
    parser.add_argument("--bootstrap-resamples", type=int, default=None, help="Override bootstrap resample count.")
    parser.add_argument("--quick", action="store_true", help="Use config analysis.quick_bootstrap_resamples.")
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    parser.add_argument("--manifest-out", default=None, help="Optional output run manifest JSON.")
    args = parser.parse_args()

    try:
        return _run(args)
    except AnalysisJoinError as exc:
        logger.error("join_error error=%s", exc)
        return 3
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("config_or_io_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
