"""Tests for MedNLA run artifact validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import pytest

import scripts.mednla.validate_run as validate_cli
from nla.mednla.run_validation import ANALYSIS_OUTPUTS, ValidationOptions, validate_run
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, to_dict


def _config() -> dict:
    return {"prompt_variants": ["canonical", "option_shuffle", "compact"]}


def _item(index: int) -> MedItem:
    return MedItem(
        item_id=f"item:{index}",
        dataset="medqa",
        split="test",
        subject=None,
        question=f"Question {index}?",
        choices={"A": "Aspirin", "B": "Ibuprofen", "C": "Metformin", "D": "Warfarin"},
        answer_key="A",
        gold_rationale=None,
        source_metadata={},
    )


def _prediction(item: MedItem, variant: str) -> Prediction:
    return Prediction(
        prediction_id=f"{item.item_id}:{variant}",
        item_id=item.item_id,
        model_id="fake/model",
        model_short_name="qwen7b",
        layer_index=20,
        prompt_variant=variant,  # type: ignore[arg-type]
        variant_seed=0,
        prompt_text="prompt",
        generated_text="A",
        selected_answer="A",
        selected_answer_text="Aspirin",
        correct=True,
        probe="pre_answer_last_prompt_token",
        token_index=5,
        activation_id=f"act:{item.item_id}:{variant}",
        original_to_variant_choice_map={"A": "A", "B": "B", "C": "C", "D": "D"},
        source_metadata={},
    )


def _decode(pred: Prediction, *, parse_ok: bool = True, warnings: list[str] | None = None) -> DecodeRecord:
    return DecodeRecord(
        activation_id=pred.activation_id,
        prediction_id=pred.prediction_id,
        model_short_name=pred.model_short_name,
        nla_actor="actor",
        nla_critic=None,
        raw_av_text="<explanation>medical explanation</explanation>",
        explanation="medical explanation" if parse_ok else None,
        parse_ok=parse_ok,
        reconstruction_mse=None,
        reconstruction_cos=None,
        decode_warnings=warnings or [],
    )


def _score(pred: Prediction, *, scorer: str = "heuristic_v1", taxonomy_cell: str = "correct_aligned") -> ScoreRecord:
    return ScoreRecord(
        prediction_id=pred.prediction_id,
        medical_relevance=2,
        rationale_alignment=None,
        answer_support="supports_selected",
        medically_invalid=False,
        shortcut_suspected=False,
        nla_quality_binary="aligned",
        taxonomy_cell=taxonomy_cell,  # type: ignore[arg-type]
        scorer=scorer,
        scorer_notes="notes",
        scorer_evidence="evidence",
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(to_dict(row)) + b"\n")


def _make_run(
    run_dir: Path,
    *,
    parse_ok: bool = True,
    cjk: bool = False,
    taxonomy_cell: str = "correct_aligned",
    include_analysis: bool = True,
) -> tuple[list[MedItem], list[Prediction], list[DecodeRecord], list[ScoreRecord]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    items = [_item(0), _item(1)]
    variants = ("canonical", "option_shuffle", "compact")
    predictions = [_prediction(item, variant) for item in items for variant in variants]
    warnings = ["cjk_injection_failure"] if cjk else []
    decodes = [_decode(pred, parse_ok=parse_ok, warnings=warnings) for pred in predictions]
    scores = [_score(pred, taxonomy_cell=taxonomy_cell) for pred in predictions]
    _write_jsonl(run_dir / "items.jsonl", items)
    _write_jsonl(run_dir / "predictions.jsonl", predictions)
    _write_jsonl(run_dir / "decodes.jsonl", decodes)
    _write_jsonl(run_dir / "scores_heuristic.jsonl", scores)
    (run_dir / "activations.parquet").write_bytes(b"parquet")
    if include_analysis:
        for rel_path in ANALYSIS_OUTPUTS:
            path = run_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
    return items, predictions, decodes, scores


def test_happy_path_heuristic_only_passes_without_judge(tmp_path: Path) -> None:
    _make_run(tmp_path)

    summary = validate_run(ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b", require_analysis=True))

    assert summary["ok"] is True
    assert summary["counts"]["items"] == 2
    assert summary["counts"]["predictions"] == 6
    assert summary["counts"]["scores_by_scorer"] == {"heuristic_v1": 6}
    assert summary["analysis_outputs_present"] is True


def test_cli_writes_summary_and_exits_8_for_missing_artifacts(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("prompt_variants: [canonical, option_shuffle, compact]\n", encoding="utf-8")
    out_path = tmp_path / "run_validation.json"

    rc = validate_cli._run(
        argparse.Namespace(
            config=str(config_path),
            run_dir=str(tmp_path / "missing"),
            model="qwen7b",
            out=str(out_path),
            parse_ok_threshold=0.8,
            require_analysis=True,
            require_judge_score=False,
        )
    )

    assert rc == 8
    summary = orjson.loads(out_path.read_bytes())
    assert summary["ok"] is False
    assert any("missing items.jsonl" in error for error in summary["errors"])


def test_duplicate_ids_and_missing_joins_fail(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _make_run(tmp_path)
    _write_jsonl(tmp_path / "predictions.jsonl", predictions + [predictions[0]])
    _write_jsonl(tmp_path / "decodes.jsonl", decodes[:-1])
    _write_jsonl(tmp_path / "scores_heuristic.jsonl", scores[:-1])

    summary = validate_run(ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b"))

    assert summary["ok"] is False
    assert any("duplicate prediction_id" in error for error in summary["errors"])
    assert any("missing decode" in error for error in summary["errors"])
    assert any("missing scores" in error for error in summary["errors"])


def test_low_parse_ok_rate_fails(tmp_path: Path) -> None:
    _make_run(tmp_path, parse_ok=False)

    summary = validate_run(ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b"))

    assert summary["ok"] is False
    assert any("parse_ok_rate" in error for error in summary["errors"])


def test_cjk_warning_fails(tmp_path: Path) -> None:
    _make_run(tmp_path, cjk=True)

    summary = validate_run(ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b"))

    assert summary["ok"] is False
    assert any("cjk_injection_failure" in error for error in summary["errors"])


def test_invalid_taxonomy_cell_fails(tmp_path: Path) -> None:
    _make_run(tmp_path, taxonomy_cell="bad_cell")

    summary = validate_run(ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b"))

    assert summary["ok"] is False
    assert any("invalid taxonomy_cell" in error for error in summary["errors"])


def test_required_analysis_outputs_are_checked(tmp_path: Path) -> None:
    _make_run(tmp_path, include_analysis=False)

    summary = validate_run(
        ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b", require_analysis=True)
    )

    assert summary["ok"] is False
    assert any("missing analysis output" in error for error in summary["errors"])


def test_require_judge_score_fails_when_absent(tmp_path: Path) -> None:
    _make_run(tmp_path)

    summary = validate_run(
        ValidationOptions(config=_config(), run_dir=tmp_path, model_short_name="qwen7b", require_judge_score=True)
    )

    assert summary["ok"] is False
    assert any("missing scores_judge.jsonl" in error for error in summary["errors"])
