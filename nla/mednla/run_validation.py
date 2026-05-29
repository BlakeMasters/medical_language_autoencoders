"""Validate MedNLA run directories and emit compact QA summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, from_dict

CJK_WARNING = "cjk_injection_failure"
TAXONOMY_CELLS = {"correct_aligned", "correct_weak", "incorrect_aligned", "incorrect_weak"}
ANALYSIS_OUTPUTS = (
    "tables/summary_by_model_dataset.csv",
    "tables/taxonomy_by_model_dataset.csv",
    "tables/prompt_stability.csv",
    "tables/rationale_alignment_by_correctness.csv",
    "tables/failure_cases.jsonl",
    "figures_data/taxonomy_stack.jsonl",
    "figures_data/accuracy_vs_aligned.jsonl",
    "figures_data/reconstruction_by_cell.jsonl",
    "_summary.json",
)
MANIFEST_OUTPUTS = (
    "probe_manifest.json",
    "decode_manifest.json",
    "score_heuristic_manifest.json",
    "analysis_manifest.json",
)


class RunValidationIOError(RuntimeError):
    """Raised when run artifacts cannot be read or parsed."""


@dataclass(frozen=True, slots=True)
class ValidationOptions:
    config: dict[str, Any]
    run_dir: Path
    model_short_name: str | None = None
    parse_ok_threshold: float = 0.8
    require_analysis: bool = False
    require_judge_score: bool = False
    require_manifests: bool = False


def validate_run(options: ValidationOptions) -> dict[str, Any]:
    run = options.run_dir
    errors: list[str] = []
    warnings: list[str] = []
    artifacts = {
        "items": str(run / "items.jsonl"),
        "predictions": str(run / "predictions.jsonl"),
        "activations": str(run / "activations.parquet"),
        "decodes": str(run / "decodes.jsonl"),
        "scores_heuristic": str(run / "scores_heuristic.jsonl"),
        "scores_judge": str(run / "scores_judge.jsonl"),
    }

    items = _safe_load(run / "items.jsonl", MedItem, errors, required=True)
    predictions = _safe_load(run / "predictions.jsonl", Prediction, errors, required=True)
    decodes = _safe_load(run / "decodes.jsonl", DecodeRecord, errors, required=True)
    heuristic_scores = _safe_load(run / "scores_heuristic.jsonl", ScoreRecord, errors, required=True)
    judge_scores = _safe_load(
        run / "scores_judge.jsonl",
        ScoreRecord,
        errors,
        required=options.require_judge_score,
    )
    score_groups = {"heuristic_v1": heuristic_scores}
    if judge_scores:
        score_groups["medgemma_judge_v1"] = judge_scores

    if not (run / "activations.parquet").exists():
        errors.append("missing activations.parquet")
    elif (run / "activations.parquet").stat().st_size <= 0:
        errors.append("empty activations.parquet")

    item_index = _index_unique(items, "item_id", errors)
    prediction_index = _index_unique(predictions, "prediction_id", errors)
    decode_index = _index_unique(decodes, "prediction_id", errors)

    _check_predictions(options, predictions, item_index, errors)
    _check_decodes(predictions, decodes, decode_index, errors)
    _check_scores(prediction_index, score_groups, errors)
    _check_analysis_outputs(run, options.require_analysis, errors)
    _check_manifest_outputs(run, options.require_manifests, options.require_judge_score, errors)

    total_decodes = len(decodes)
    parse_ok = sum(1 for decode in decodes if decode.parse_ok)
    parse_ok_rate = (parse_ok / total_decodes) if total_decodes else 0.0
    cjk_count = sum(1 for decode in decodes if CJK_WARNING in decode.decode_warnings)
    if total_decodes and parse_ok_rate < options.parse_ok_threshold:
        errors.append(f"decode parse_ok_rate {parse_ok_rate:.3f} below threshold {options.parse_ok_threshold:.3f}")
    if cjk_count:
        errors.append(f"{CJK_WARNING} warnings present: {cjk_count}")

    scorer_counts = {scorer: len(rows) for scorer, rows in score_groups.items() if rows}
    taxonomy_counts = Counter(score.taxonomy_cell for rows in score_groups.values() for score in rows)
    if not sum(scorer_counts.values()):
        errors.append("no score rows found")

    analysis_present = all((run / rel_path).exists() and (run / rel_path).stat().st_size > 0 for rel_path in ANALYSIS_OUTPUTS)
    summary = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "thresholds": {
            "parse_ok_rate_min": options.parse_ok_threshold,
            "cjk_injection_failure_max": 0,
        },
        "counts": {
            "items": len(items),
            "predictions": len(predictions),
            "decodes": len(decodes),
            "parse_ok": parse_ok,
            "parse_ok_rate": parse_ok_rate,
            "cjk_injection_failure": cjk_count,
            "scores_by_scorer": scorer_counts,
            "taxonomy_cells": dict(sorted(taxonomy_counts.items())),
        },
        "artifacts": artifacts,
        "analysis_outputs_present": analysis_present,
    }
    return summary


def write_validation_summary(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))


def _safe_load(path: Path, cls: type[Any], errors: list[str], *, required: bool) -> list[Any]:
    if not path.exists():
        if required:
            errors.append(f"missing {path.name}")
        return []
    if path.stat().st_size <= 0:
        errors.append(f"empty {path.name}")
        return []
    try:
        return _load_jsonl(path, cls)
    except Exception as exc:
        raise RunValidationIOError(f"failed to read {path}: {exc}") from exc


def _load_jsonl(path: Path, cls: type[Any]) -> list[Any]:
    rows: list[Any] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(from_dict(cls, orjson.loads(line)))
            except Exception as exc:
                raise ValueError(f"line {line_number}: invalid {cls.__name__}: {exc}") from exc
    return rows


def _index_unique(rows: list[Any], key_name: str, errors: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        key = str(getattr(row, key_name))
        if key in out:
            errors.append(f"duplicate {key_name}: {key}")
        else:
            out[key] = row
    return out


def _check_predictions(
    options: ValidationOptions,
    predictions: list[Prediction],
    item_index: dict[str, MedItem],
    errors: list[str],
) -> None:
    expected_variants = tuple(options.config.get("prompt_variants") or ())
    if not expected_variants:
        errors.append("config prompt_variants must be non-empty")
    variants_by_item: dict[str, set[str]] = defaultdict(set)
    for prediction in predictions:
        if options.model_short_name and prediction.model_short_name != options.model_short_name:
            errors.append(
                f"prediction {prediction.prediction_id} model_short_name={prediction.model_short_name!r} "
                f"!= expected {options.model_short_name!r}"
            )
        if prediction.item_id not in item_index:
            errors.append(f"prediction {prediction.prediction_id} missing item {prediction.item_id}")
        variants_by_item[prediction.item_id].add(prediction.prompt_variant)
    expected = set(str(variant) for variant in expected_variants)
    for item_id in item_index:
        observed = variants_by_item.get(item_id, set())
        if observed != expected:
            errors.append(f"item {item_id} prompt variants {sorted(observed)} != expected {sorted(expected)}")


def _check_decodes(
    predictions: list[Prediction],
    decodes: list[DecodeRecord],
    decode_index: dict[str, DecodeRecord],
    errors: list[str],
) -> None:
    prediction_ids = {prediction.prediction_id for prediction in predictions}
    for decode in decodes:
        if decode.prediction_id not in prediction_ids:
            errors.append(f"decode missing prediction {decode.prediction_id}")
    for prediction in predictions:
        if prediction.prediction_id not in decode_index:
            errors.append(f"prediction {prediction.prediction_id} missing decode")


def _check_scores(
    prediction_index: dict[str, Prediction],
    score_groups: dict[str, list[ScoreRecord]],
    errors: list[str],
) -> None:
    prediction_ids = set(prediction_index)
    for expected_scorer, scores in score_groups.items():
        if not scores:
            continue
        seen: set[str] = set()
        for score in scores:
            if score.scorer != expected_scorer:
                errors.append(f"score {score.prediction_id} has scorer {score.scorer!r}, expected {expected_scorer!r}")
            if score.prediction_id not in prediction_index:
                errors.append(f"score missing prediction {score.prediction_id}")
            if score.prediction_id in seen:
                errors.append(f"duplicate score prediction_id for {expected_scorer}: {score.prediction_id}")
            seen.add(score.prediction_id)
            if score.taxonomy_cell not in TAXONOMY_CELLS:
                errors.append(f"invalid taxonomy_cell for {score.prediction_id}: {score.taxonomy_cell!r}")
        missing = prediction_ids - seen
        if missing:
            errors.append(f"{expected_scorer} missing scores for {len(missing)} prediction(s)")


def _check_analysis_outputs(run: Path, require_analysis: bool, errors: list[str]) -> None:
    if not require_analysis:
        return
    for rel_path in ANALYSIS_OUTPUTS:
        path = run / rel_path
        if not path.exists():
            errors.append(f"missing analysis output {rel_path}")
        elif path.stat().st_size <= 0:
            errors.append(f"empty analysis output {rel_path}")


def _check_manifest_outputs(
    run: Path,
    require_manifests: bool,
    require_judge_score: bool,
    errors: list[str],
) -> None:
    if not require_manifests:
        return
    expected = list(MANIFEST_OUTPUTS)
    if require_judge_score:
        expected.append("score_judge_manifest.json")
    for rel_path in expected:
        path = run / rel_path
        if not path.exists():
            errors.append(f"missing manifest {rel_path}")
        elif path.stat().st_size <= 0:
            errors.append(f"empty manifest {rel_path}")


__all__ = [
    "ANALYSIS_OUTPUTS",
    "CJK_WARNING",
    "MANIFEST_OUTPUTS",
    "RunValidationIOError",
    "ValidationOptions",
    "validate_run",
    "write_validation_summary",
]
