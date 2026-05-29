"""Report-bundle helpers for MedNLA analysis artifacts."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import orjson

from nla.mednla.analysis import TAXONOMY_CELLS, join_rows
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, from_dict

AUDIT_LABEL_COLUMNS = (
    "automated_score_reasonable",
    "clear_confabulation",
    "gold_rationale_incomplete",
    "highlight_candidate",
    "auditor_notes",
)
BASE_AUDIT_COLUMNS = (
    "audit_selection_reasons",
    "prediction_id",
    "item_id",
    "model_short_name",
    "dataset",
    "subject",
    "prompt_variant",
    "scorer",
    "taxonomy_cell",
    "correct",
    "selected_answer",
    "selected_answer_original",
    "answer_key",
    "question",
    "explanation",
    "scorer_notes",
    "scorer_evidence",
    "reconstruction_cos",
    "reconstruction_mse",
) + AUDIT_LABEL_COLUMNS

CLAIMS_TO_SUPPORT = (
    "Accuracy and NLA explanation quality diverged in X% of correct answers.",
    "Prompt variants changed NLA quality more or less often than final answer correctness.",
    "Reconstruction quality was or was not associated with medically aligned explanations.",
    "These findings suggest accuracy alone may be incomplete as a measure of medically grounded reasoning.",
)
CLAIMS_TO_AVOID = (
    "The NLA reveals the model's true reasoning.",
    "A weak NLA explanation proves shortcut behavior.",
    "Gold-rationale mismatch proves medical invalidity.",
    "The released general-domain NLA is validated for clinical reasoning.",
)


def export_report_bundle(
    *,
    run_dir: Path,
    out_dir: Path,
    items_path: Path | None = None,
    predictions_path: Path | None = None,
    decodes_path: Path | None = None,
    score_paths: list[Path] | None = None,
    per_cell: int = 10,
    max_cases: int = 80,
    seed: int = 42,
    include_raw_av_text: bool = False,
) -> dict[str, Any]:
    """Write a report-ready evidence bundle and return its artifact index."""

    if per_cell <= 0:
        raise ValueError("per_cell must be positive")
    if max_cases <= 0:
        raise ValueError("max_cases must be positive")

    paths = resolve_artifact_paths(
        run_dir=run_dir,
        items_path=items_path,
        predictions_path=predictions_path,
        decodes_path=decodes_path,
        score_paths=score_paths,
    )
    _require_analysis_outputs(run_dir)

    items = _load_jsonl(paths["items"], MedItem)
    predictions = _load_jsonl(paths["predictions"], Prediction)
    decodes = _load_jsonl(paths["decodes"], DecodeRecord)
    scores: list[ScoreRecord] = []
    for score_path in paths["scores"]:
        scores.extend(_load_jsonl(score_path, ScoreRecord))

    joined = join_rows(items, predictions, scores, decodes)
    summary_rows = _read_csv_rows(run_dir / "tables" / "summary_by_model_dataset.csv")
    taxonomy_rows = _read_csv_rows(run_dir / "tables" / "taxonomy_by_model_dataset.csv")
    prompt_rows = _read_csv_rows(run_dir / "tables" / "prompt_stability.csv")
    failure_rows = _read_jsonl_dicts(run_dir / "tables" / "failure_cases.jsonl")
    figure_paths = _figure_paths(run_dir)

    audit_rows, selection_notes = build_audit_queue(
        joined,
        failure_rows=failure_rows,
        prompt_stability_rows=prompt_rows,
        per_cell=per_cell,
        max_cases=max_cases,
        seed=seed,
        include_raw_av_text=include_raw_av_text,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": out_dir / "summary.md",
        "audit_queue_csv": out_dir / "audit_queue.csv",
        "audit_queue_jsonl": out_dir / "audit_queue.jsonl",
        "claims_checklist": out_dir / "claims_checklist.md",
        "artifact_index": out_dir / "artifact_index.json",
    }
    audit_columns = BASE_AUDIT_COLUMNS + (("raw_av_text",) if include_raw_av_text else ())
    _write_csv(outputs["audit_queue_csv"], audit_rows, audit_columns)
    _write_jsonl(outputs["audit_queue_jsonl"], audit_rows)

    index = build_artifact_index(
        run_dir=run_dir,
        out_dir=out_dir,
        paths=paths,
        source_counts={
            "items": len(items),
            "predictions": len(predictions),
            "decodes": len(decodes),
            "scores": len(scores),
        },
        joined=joined,
        audit_rows=audit_rows,
        summary_rows=summary_rows,
        taxonomy_rows=taxonomy_rows,
        figure_paths=figure_paths,
        selection_notes=selection_notes,
        include_raw_av_text=include_raw_av_text,
    )
    _write_summary_md(outputs["summary"], index, summary_rows, taxonomy_rows)
    _write_claims_checklist(outputs["claims_checklist"], index)
    outputs_text = {key: str(value) for key, value in outputs.items()}
    index["outputs"] = outputs_text
    outputs["artifact_index"].write_bytes(orjson.dumps(index, option=orjson.OPT_INDENT_2))
    return index


def resolve_artifact_paths(
    *,
    run_dir: Path,
    items_path: Path | None,
    predictions_path: Path | None,
    decodes_path: Path | None,
    score_paths: list[Path] | None,
) -> dict[str, Any]:
    scores = score_paths if score_paths is not None else sorted(run_dir.glob("scores_*.jsonl"))
    paths = {
        "items": items_path or run_dir / "items.jsonl",
        "predictions": predictions_path or run_dir / "predictions.jsonl",
        "decodes": decodes_path or run_dir / "decodes.jsonl",
        "scores": scores,
    }
    missing = [str(paths[name]) for name in ("items", "predictions", "decodes") if not paths[name].exists()]
    missing.extend(str(path) for path in scores if not path.exists())
    if not scores:
        missing.append(str(run_dir / "scores_*.jsonl"))
    if missing:
        raise FileNotFoundError("missing required report artifact(s): " + ", ".join(missing))
    return paths


def build_audit_queue(
    joined: list[dict[str, Any]],
    *,
    failure_rows: list[dict[str, Any]],
    prompt_stability_rows: list[dict[str, Any]],
    per_cell: int,
    max_cases: int,
    seed: int,
    include_raw_av_text: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: "OrderedDict[tuple[str, str], dict[str, Any]]" = OrderedDict()
    notes: dict[str, Any] = {
        "missing_taxonomy_cells": [],
        "reconstruction_cosine": "available" if any(row["reconstruction_cos"] is not None for row in joined) else "unavailable_all_null",
    }

    def add(row: dict[str, Any], reason: str) -> None:
        key = (str(row["scorer"]), str(row["prediction_id"]))
        if key in selected:
            selected[key]["_reasons"].add(reason)
            return
        if len(selected) >= max_cases:
            return
        selected[key] = {"row": row, "_reasons": {reason}}

    joined_by_key = {(str(row["scorer"]), str(row["prediction_id"])): row for row in joined}
    for failure in failure_rows:
        row = joined_by_key.get((str(failure.get("scorer")), str(failure.get("prediction_id"))))
        if row is not None:
            add(row, "correct_weak_failure_case")

    unstable_groups = _unstable_prompt_groups(prompt_stability_rows)
    for row in joined:
        if (str(row["item_id"]), str(row["model_short_name"]), str(row["scorer"])) in unstable_groups:
            add(row, "prompt_instability")

    if notes["reconstruction_cosine"] == "available":
        low_reconstruction = sorted(
            (row for row in joined if row["reconstruction_cos"] is not None),
            key=lambda row: (float(row["reconstruction_cos"]), str(row["scorer"]), str(row["prediction_id"])),
        )
        for row in low_reconstruction[:per_cell]:
            add(row, "low_reconstruction")

    for cell in TAXONOMY_CELLS:
        candidates = [row for row in joined if row["taxonomy_cell"] == cell]
        if not candidates:
            notes["missing_taxonomy_cells"].append(cell)
            continue
        ordered = sorted(
            candidates,
            key=lambda row: _stable_sample_key(seed, cell, str(row["scorer"]), str(row["prediction_id"])),
        )
        for row in ordered[:per_cell]:
            add(row, f"taxonomy_sample:{cell}")

    return [_audit_row(payload["row"], payload["_reasons"], include_raw_av_text) for payload in selected.values()], notes


def build_artifact_index(
    *,
    run_dir: Path,
    out_dir: Path,
    paths: dict[str, Any],
    source_counts: dict[str, int],
    joined: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, str]],
    taxonomy_rows: list[dict[str, str]],
    figure_paths: dict[str, Path],
    selection_notes: dict[str, Any],
    include_raw_av_text: bool,
) -> dict[str, Any]:
    scorers = sorted({str(row["scorer"]) for row in joined})
    taxonomy_counts = Counter(str(row["taxonomy_cell"]) for row in joined)
    models = sorted({str(row["model_short_name"]) for row in joined})
    datasets = sorted({str(row["dataset"]) for row in joined})
    unavailable: dict[str, str] = {}
    if selection_notes["reconstruction_cosine"] != "available":
        unavailable["reconstruction_cos"] = "all joined rows have null reconstruction_cos"
    if "medgemma_judge_v1" not in scorers:
        unavailable["medgemma_judge_v1"] = "no MedGemma judge score file included"
    return {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "artifact_paths": {
            "items": str(paths["items"]),
            "predictions": str(paths["predictions"]),
            "decodes": str(paths["decodes"]),
            "scores": [str(path) for path in paths["scores"]],
            "tables": {
                "summary_by_model_dataset": str(run_dir / "tables" / "summary_by_model_dataset.csv"),
                "taxonomy_by_model_dataset": str(run_dir / "tables" / "taxonomy_by_model_dataset.csv"),
                "prompt_stability": str(run_dir / "tables" / "prompt_stability.csv"),
                "failure_cases": str(run_dir / "tables" / "failure_cases.jsonl"),
            },
            "figures_data": {name: str(path) for name, path in figure_paths.items()},
        },
        "counts": {
            "items": source_counts["items"],
            "predictions": source_counts["predictions"],
            "decodes": source_counts["decodes"],
            "scores": source_counts["scores"],
            "joined_items": len({row["item_id"] for row in joined}),
            "joined_predictions": len({row["prediction_id"] for row in joined}),
            "joined_rows": len(joined),
            "audit_rows": len(audit_rows),
            "summary_rows": len(summary_rows),
            "taxonomy_rows": len(taxonomy_rows),
        },
        "models": models,
        "datasets": datasets,
        "scorers": scorers,
        "taxonomy_cell_counts": dict(sorted(taxonomy_counts.items())),
        "missing_taxonomy_cells": selection_notes["missing_taxonomy_cells"],
        "unavailable_fields": unavailable,
        "include_raw_av_text": include_raw_av_text,
        "heuristic_only": scorers == ["heuristic_v1"],
    }


def _require_analysis_outputs(run_dir: Path) -> None:
    required = (
        run_dir / "tables" / "summary_by_model_dataset.csv",
        run_dir / "tables" / "taxonomy_by_model_dataset.csv",
        run_dir / "tables" / "prompt_stability.csv",
        run_dir / "tables" / "failure_cases.jsonl",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required T6 output(s): " + ", ".join(missing))


def _figure_paths(run_dir: Path) -> dict[str, Path]:
    names = {
        "taxonomy_stack": run_dir / "figures_data" / "taxonomy_stack.jsonl",
        "accuracy_vs_aligned": run_dir / "figures_data" / "accuracy_vs_aligned.jsonl",
        "reconstruction_by_cell": run_dir / "figures_data" / "reconstruction_by_cell.jsonl",
    }
    return {name: path for name, path in names.items() if path.exists()}


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


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = orjson.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_summary_md(
    path: Path,
    index: dict[str, Any],
    summary_rows: list[dict[str, str]],
    taxonomy_rows: list[dict[str, str]],
) -> None:
    lines = [
        "# MedNLA Report Bundle",
        "",
        "## Run Summary",
        "",
        f"- Models: {', '.join(index['models']) or 'none'}",
        f"- Datasets: {', '.join(index['datasets']) or 'none'}",
        f"- Scorers: {', '.join(index['scorers']) or 'none'}",
        f"- Items: {index['counts']['items']}",
        f"- Predictions: {index['counts']['predictions']}",
        f"- Joined score rows: {index['counts']['joined_rows']}",
        f"- Manual audit rows: {index['counts']['audit_rows']}",
        "",
        "## Summary Table",
        "",
        _markdown_table(
            summary_rows,
            ("model_short_name", "dataset", "scorer", "n_items", "n_predictions", "accuracy", "aligned_rate"),
        ),
        "",
        "## Taxonomy Counts",
        "",
        _markdown_table(
            taxonomy_rows,
            ("model_short_name", "dataset", "scorer", "taxonomy_cell", "count", "proportion"),
        ),
        "",
        "## Artifact Notes",
        "",
    ]
    wrote_note = False
    if index["heuristic_only"]:
        lines.append("- This bundle contains heuristic scoring only; do not make MedGemma judge validation claims.")
        wrote_note = True
    if index["unavailable_fields"]:
        for key, reason in index["unavailable_fields"].items():
            lines.append(f"- `{key}` unavailable: {reason}.")
            wrote_note = True
    if index["missing_taxonomy_cells"]:
        lines.append("- Missing taxonomy cells in audit source rows: " + ", ".join(index["missing_taxonomy_cells"]) + ".")
        wrote_note = True
    if not wrote_note:
        lines.append("- No unavailable fields recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_claims_checklist(path: Path, index: dict[str, Any]) -> None:
    lines = [
        "# MedNLA Claims Checklist",
        "",
        "## Claims This Bundle Can Support",
        "",
    ]
    lines.extend(f"- [ ] {claim}" for claim in CLAIMS_TO_SUPPORT)
    lines.extend(["", "## Claims To Avoid", ""])
    lines.extend(f"- [ ] {claim}" for claim in CLAIMS_TO_AVOID)
    lines.extend(["", "## Bundle-Specific Reminders", ""])
    if index["heuristic_only"]:
        lines.append("- [ ] This bundle is heuristic-only; avoid claims about MedGemma judge agreement or validation.")
    if "reconstruction_cos" in index["unavailable_fields"]:
        lines.append("- [ ] Reconstruction cosine was unavailable; avoid reconstruction-quality claims from this bundle.")
    lines.append("- [ ] Treat manual audit labels as qualitative calibration, not threshold tuning.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
    if not rows:
        return "_No rows._"
    output = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        output.append("| " + " | ".join(_markdown_cell(row.get(column, "")) for column in columns) + " |")
    return "\n".join(output)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _unstable_prompt_groups(rows: list[dict[str, str]]) -> set[tuple[str, str, str]]:
    unstable: set[tuple[str, str, str]] = set()
    for row in rows:
        if (
            _as_float(row.get("answer_agreement")) < 1.0
            or _as_bool(row.get("shuffle_changed_answer"))
        ):
            unstable.add((str(row.get("item_id")), str(row.get("model_short_name")), str(row.get("scorer"))))
    return unstable


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def _stable_sample_key(seed: int, *parts: str) -> str:
    data = "\x1f".join([str(seed), *parts])
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _audit_row(row: dict[str, Any], reasons: set[str], include_raw_av_text: bool) -> dict[str, Any]:
    audit = {
        "audit_selection_reasons": ";".join(sorted(reasons)),
        "prediction_id": row["prediction_id"],
        "item_id": row["item_id"],
        "model_short_name": row["model_short_name"],
        "dataset": row["dataset"],
        "subject": row["subject"],
        "prompt_variant": row["prompt_variant"],
        "scorer": row["scorer"],
        "taxonomy_cell": row["taxonomy_cell"],
        "correct": row["correct"],
        "selected_answer": row["selected_answer"],
        "selected_answer_original": row["selected_answer_original"],
        "answer_key": row["answer_key"],
        "question": row["question"],
        "explanation": row["explanation"],
        "scorer_notes": row["scorer_notes"],
        "scorer_evidence": row["scorer_evidence"],
        "reconstruction_cos": row["reconstruction_cos"],
        "reconstruction_mse": row["reconstruction_mse"],
        "automated_score_reasonable": "",
        "clear_confabulation": "",
        "gold_rationale_incomplete": "",
        "highlight_candidate": "",
        "auditor_notes": "",
    }
    if include_raw_av_text:
        audit["raw_av_text"] = row["raw_av_text"]
    return audit


__all__ = [
    "AUDIT_LABEL_COLUMNS",
    "BASE_AUDIT_COLUMNS",
    "CLAIMS_TO_AVOID",
    "CLAIMS_TO_SUPPORT",
    "build_artifact_index",
    "build_audit_queue",
    "export_report_bundle",
    "resolve_artifact_paths",
]
