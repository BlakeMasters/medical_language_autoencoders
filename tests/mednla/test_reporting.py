"""Tests for MedNLA report-bundle export."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import orjson

import scripts.mednla.export_report_bundle as report_cli
from nla.mednla.analysis import (
    accuracy_vs_aligned,
    failure_cases,
    join_rows,
    prompt_stability,
    rationale_alignment_by_correctness,
    reconstruction_by_cell,
    summary_by_model_dataset,
    taxonomy_by_model_dataset,
)
from nla.mednla.reporting import AUDIT_LABEL_COLUMNS, build_audit_queue, export_report_bundle
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, to_dict

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
PROMPT_COLUMNS = (
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


def _item(index: int) -> MedItem:
    return MedItem(
        item_id=f"item:{index}",
        dataset="medqa",
        split="test",
        subject="cardiology" if index % 2 == 0 else "renal",
        question=f"Question {index}: What supports the diagnosis?",
        choices={"A": "Aspirin", "B": "Ibuprofen", "C": "Metformin", "D": "Warfarin"},
        answer_key="A",
        gold_rationale="Aspirin inhibits platelets.",
        source_metadata={"index": index},
    )


def _choice_map(variant: str) -> dict[str, str]:
    if variant == "option_shuffle":
        return {"A": "B", "B": "A", "C": "C", "D": "D"}
    return {"A": "A", "B": "B", "C": "C", "D": "D"}


def _prediction(item: MedItem, variant: str, original_answer: str) -> Prediction:
    choice_map = _choice_map(variant)
    selected = choice_map[original_answer]
    return Prediction(
        prediction_id=f"{item.item_id}:{variant}",
        item_id=item.item_id,
        model_id="fake/model",
        model_short_name="fake",
        layer_index=20,
        prompt_variant=variant,  # type: ignore[arg-type]
        variant_seed=42,
        prompt_text="prompt",
        generated_text=selected,
        selected_answer=selected,
        selected_answer_text=item.choices[original_answer],
        correct=original_answer == item.answer_key,
        probe="pre_answer_last_prompt_token",
        token_index=3,
        activation_id=f"act:{item.item_id}:{variant}",
        original_to_variant_choice_map=choice_map,
        source_metadata={},
    )


def _decode(pred: Prediction, index: int, *, reconstruction: bool = True) -> DecodeRecord:
    return DecodeRecord(
        activation_id=pred.activation_id,
        prediction_id=pred.prediction_id,
        model_short_name=pred.model_short_name,
        nla_actor="actor",
        nla_critic=None,
        raw_av_text=f"<explanation>Raw text {pred.prediction_id}</explanation>",
        explanation=f"Decoded explanation {pred.prediction_id}",
        parse_ok=True,
        reconstruction_mse=0.1 + index * 0.01 if reconstruction else None,
        reconstruction_cos=0.9 - index * 0.01 if reconstruction else None,
        decode_warnings=[],
    )


def _score(pred: Prediction, cell: str, *, scorer: str = "heuristic_v1") -> ScoreRecord:
    quality = "aligned" if cell.endswith("_aligned") else "weak"
    return ScoreRecord(
        prediction_id=pred.prediction_id,
        medical_relevance=2 if quality == "aligned" else 0,
        rationale_alignment=1 if quality == "aligned" else 0,
        answer_support="supports_selected" if quality == "aligned" else "unclear",
        medically_invalid=False,
        shortcut_suspected=quality == "weak",
        nla_quality_binary=quality,  # type: ignore[arg-type]
        taxonomy_cell=cell,  # type: ignore[arg-type]
        scorer=scorer,
        scorer_notes=f"{cell} notes",
        scorer_evidence=f"{cell} evidence",
    )


def _fixture(
    *,
    reconstruction: bool = True,
    cells: tuple[str, ...] = ("correct_aligned", "correct_weak", "incorrect_aligned", "incorrect_weak"),
) -> tuple[list[MedItem], list[Prediction], list[DecodeRecord], list[ScoreRecord]]:
    variants = ("canonical", "option_shuffle", "compact")
    items = [_item(index) for index in range(4)]
    predictions: list[Prediction] = []
    decodes: list[DecodeRecord] = []
    scores: list[ScoreRecord] = []
    counter = 0
    for item_index, item in enumerate(items):
        for variant in variants:
            cell = cells[counter % len(cells)]
            original_answer = "A" if cell.startswith("correct") else "B"
            if item_index == 0 and variant == "option_shuffle":
                original_answer = "B"
                cell = "incorrect_weak"
            pred = _prediction(item, variant, original_answer)
            predictions.append(pred)
            decodes.append(_decode(pred, counter, reconstruction=reconstruction))
            scores.append(_score(pred, cell))
            counter += 1
    return items, predictions, decodes, scores


def _write_run(
    run_dir: Path,
    items: list[MedItem],
    predictions: list[Prediction],
    decodes: list[DecodeRecord],
    scores: list[ScoreRecord],
    *,
    write_artifacts: bool = True,
) -> None:
    if write_artifacts:
        _write_jsonl(run_dir / "items.jsonl", items)
        _write_jsonl(run_dir / "predictions.jsonl", predictions)
        _write_jsonl(run_dir / "decodes.jsonl", decodes)
        _write_jsonl(run_dir / "scores_heuristic.jsonl", scores)

    joined = join_rows(items, predictions, scores, decodes)
    _write_csv(
        run_dir / "tables" / "summary_by_model_dataset.csv",
        summary_by_model_dataset(joined, n_resamples=5),
        SUMMARY_COLUMNS,
    )
    _write_csv(
        run_dir / "tables" / "taxonomy_by_model_dataset.csv",
        taxonomy_by_model_dataset(joined, n_resamples=5),
        TAXONOMY_COLUMNS,
    )
    _write_csv(run_dir / "tables" / "prompt_stability.csv", prompt_stability(joined), PROMPT_COLUMNS)
    _write_csv(
        run_dir / "tables" / "rationale_alignment_by_correctness.csv",
        rationale_alignment_by_correctness(joined),
        RATIONALE_COLUMNS,
    )
    _write_jsonl_dicts(run_dir / "tables" / "failure_cases.jsonl", failure_cases(joined))
    _write_jsonl_dicts(run_dir / "figures_data" / "taxonomy_stack.jsonl", taxonomy_by_model_dataset(joined, n_resamples=5))
    _write_jsonl_dicts(run_dir / "figures_data" / "accuracy_vs_aligned.jsonl", accuracy_vs_aligned(joined))
    _write_jsonl_dicts(run_dir / "figures_data" / "reconstruction_by_cell.jsonl", reconstruction_by_cell(joined))


def test_export_full_run_dir_bundle(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture()
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "report"
    _write_run(run_dir, items, predictions, decodes, scores)

    index = export_report_bundle(run_dir=run_dir, out_dir=out_dir, per_cell=2, max_cases=40, seed=7)

    assert index["counts"]["items"] == len(items)
    assert index["counts"]["predictions"] == len(predictions)
    assert index["counts"]["audit_rows"] > 0
    assert index["heuristic_only"] is True
    for name in ("summary.md", "audit_queue.csv", "audit_queue.jsonl", "claims_checklist.md", "artifact_index.json"):
        assert (out_dir / name).exists()
    summary = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "heuristic scoring only" in summary


def test_split_artifact_paths_are_supported(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture()
    run_dir = tmp_path / "analysis"
    artifacts = tmp_path / "artifacts"
    _write_run(run_dir, items, predictions, decodes, scores, write_artifacts=False)
    _write_jsonl(artifacts / "items.jsonl", items)
    _write_jsonl(artifacts / "predictions.jsonl", predictions)
    _write_jsonl(artifacts / "decodes.jsonl", decodes)
    _write_jsonl(artifacts / "scores_heuristic.jsonl", scores)

    index = export_report_bundle(
        run_dir=run_dir,
        out_dir=tmp_path / "report",
        items_path=artifacts / "items.jsonl",
        predictions_path=artifacts / "predictions.jsonl",
        decodes_path=artifacts / "decodes.jsonl",
        score_paths=[artifacts / "scores_heuristic.jsonl"],
        per_cell=2,
    )

    assert Path(index["artifact_paths"]["items"]).name == "items.jsonl"
    assert Path(index["artifact_paths"]["items"]).parent.name == "artifacts"
    assert index["counts"]["joined_rows"] == len(scores)


def test_audit_selection_is_deterministic_and_marks_reasons(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture()
    run_dir = tmp_path / "run"
    _write_run(run_dir, items, predictions, decodes, scores)
    joined = join_rows(items, predictions, scores, decodes)
    failures = _read_jsonl(run_dir / "tables" / "failure_cases.jsonl")
    prompt_rows = _read_csv(run_dir / "tables" / "prompt_stability.csv")

    rows_a, notes_a = build_audit_queue(
        joined,
        failure_rows=failures,
        prompt_stability_rows=prompt_rows,
        per_cell=2,
        max_cases=40,
        seed=123,
        include_raw_av_text=False,
    )
    rows_b, notes_b = build_audit_queue(
        joined,
        failure_rows=failures,
        prompt_stability_rows=prompt_rows,
        per_cell=2,
        max_cases=40,
        seed=123,
        include_raw_av_text=False,
    )

    assert rows_a == rows_b
    assert notes_a == notes_b
    reasons = ";".join(row["audit_selection_reasons"] for row in rows_a)
    assert "correct_weak_failure_case" in reasons
    assert "prompt_instability" in reasons
    assert "taxonomy_sample:correct_aligned" in reasons


def test_missing_taxonomy_cells_and_null_reconstruction_are_recorded(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture(
        reconstruction=False,
        cells=("correct_aligned", "correct_weak"),
    )
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "report"
    _write_run(run_dir, items, predictions, decodes, scores)

    index = export_report_bundle(run_dir=run_dir, out_dir=out_dir, per_cell=2)

    assert "incorrect_aligned" in index["missing_taxonomy_cells"]
    assert index["unavailable_fields"]["reconstruction_cos"] == "all joined rows have null reconstruction_cos"
    claims = (out_dir / "claims_checklist.md").read_text(encoding="utf-8")
    assert "Reconstruction cosine was unavailable" in claims


def test_audit_rows_have_blank_human_labels_and_raw_text_opt_in(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture()
    run_dir = tmp_path / "run"
    _write_run(run_dir, items, predictions, decodes, scores)

    export_report_bundle(run_dir=run_dir, out_dir=tmp_path / "no_raw", per_cell=1)
    no_raw_header = _csv_header(tmp_path / "no_raw" / "audit_queue.csv")
    no_raw_rows = _read_csv(tmp_path / "no_raw" / "audit_queue.csv")
    assert "raw_av_text" not in no_raw_header
    for column in AUDIT_LABEL_COLUMNS:
        assert no_raw_rows[0][column] == ""

    export_report_bundle(
        run_dir=run_dir,
        out_dir=tmp_path / "with_raw",
        per_cell=1,
        include_raw_av_text=True,
    )
    assert "raw_av_text" in _csv_header(tmp_path / "with_raw" / "audit_queue.csv")
    raw_json = _read_jsonl(tmp_path / "with_raw" / "audit_queue.jsonl")
    assert raw_json[0]["raw_av_text"].startswith("<explanation>")


def test_cli_writes_expected_bundle_files(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture()
    run_dir = tmp_path / "run"
    out_dir = tmp_path / "report"
    _write_run(run_dir, items, predictions, decodes, scores)

    rc = report_cli._run(
        argparse.Namespace(
            run_dir=str(run_dir),
            items=None,
            predictions=None,
            decodes=None,
            scores=None,
            out_dir=str(out_dir),
            per_cell=2,
            max_cases=40,
            seed=42,
            include_raw_av_text=False,
        )
    )

    assert rc == 0
    index = orjson.loads((out_dir / "artifact_index.json").read_bytes())
    assert index["outputs"]["audit_queue_csv"].endswith("audit_queue.csv")
    assert index["counts"]["audit_rows"] > 0


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(to_dict(row)) + b"\n")


def _write_jsonl_dicts(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row) + b"\n")


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))
