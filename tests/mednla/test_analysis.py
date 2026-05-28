"""Tests for MedNLA analysis tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import orjson
import pytest

import scripts.mednla.analyze_results as analysis_cli
from nla.mednla.analysis import (
    AnalysisJoinError,
    bootstrap_proportion,
    failure_cases,
    join_rows,
    prompt_stability,
    taxonomy_by_model_dataset,
)
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, to_dict


def _item(index: int) -> MedItem:
    return MedItem(
        item_id=f"item:{index:02d}",
        dataset="medqa",
        split="test",
        subject="cardiology" if index % 2 == 0 else "neurology",
        question=f"Question {index}: Which finding supports the diagnosis?",
        choices={"A": "Aspirin", "B": "Ibuprofen", "C": "Metformin", "D": "Warfarin"},
        answer_key="A",
        gold_rationale="Aspirin inhibits platelet aggregation." if index % 2 == 0 else None,
        source_metadata={"source_index": index},
    )


def _choice_map(variant: str) -> dict[str, str]:
    if variant == "option_shuffle":
        return {"A": "B", "B": "A", "C": "C", "D": "D"}
    return {"A": "A", "B": "B", "C": "C", "D": "D"}


def _prediction(item: MedItem, variant: str, original_answer: str) -> Prediction:
    choice_map = _choice_map(variant)
    selected_answer = choice_map[original_answer]
    return Prediction(
        prediction_id=f"{item.item_id}:{variant}",
        item_id=item.item_id,
        model_id="fake/model",
        model_short_name="fake",
        layer_index=1,
        prompt_variant=variant,  # type: ignore[arg-type]
        variant_seed=17,
        prompt_text="prompt",
        generated_text=selected_answer,
        selected_answer=selected_answer,
        selected_answer_text=item.choices[original_answer],
        correct=original_answer == item.answer_key,
        probe="pre_answer_last_prompt_token",
        token_index=7,
        activation_id=f"act:{item.item_id}:{variant}",
        original_to_variant_choice_map=choice_map,
        source_metadata={},
    )


def _decode(pred: Prediction, index: int, variant_index: int) -> DecodeRecord:
    return DecodeRecord(
        activation_id=pred.activation_id,
        prediction_id=pred.prediction_id,
        model_short_name=pred.model_short_name,
        nla_actor="actor",
        nla_critic=None,
        raw_av_text=f"<explanation>Explanation for {pred.prediction_id}</explanation>",
        explanation=f"Explanation for {pred.prediction_id}",
        parse_ok=True,
        reconstruction_mse=0.1 + index * 0.01 + variant_index * 0.001,
        reconstruction_cos=0.9 - index * 0.01 - variant_index * 0.001,
        decode_warnings=[],
    )


def _score(pred: Prediction, index: int, variant_index: int, *, scorer: str = "heuristic_v1") -> ScoreRecord:
    aligned = (index + variant_index) % 3 == 0
    quality = "aligned" if aligned else "weak"
    taxonomy = f"{'correct' if pred.correct else 'incorrect'}_{quality}"
    return ScoreRecord(
        prediction_id=pred.prediction_id,
        medical_relevance=2 if aligned else variant_index % 2,
        rationale_alignment=1 if aligned and index % 2 == 0 else (0 if index % 2 == 0 else None),
        answer_support="supports_selected" if aligned else "unclear",
        medically_invalid=False,
        shortcut_suspected=False,
        nla_quality_binary=quality,  # type: ignore[arg-type]
        taxonomy_cell=taxonomy,  # type: ignore[arg-type]
        scorer=scorer,
        scorer_notes=f"{scorer} notes",
        scorer_evidence=f"{scorer} evidence",
    )


def _fixture(n_items: int = 20, *, scorers: tuple[str, ...] = ("heuristic_v1",)) -> tuple[
    list[MedItem],
    list[Prediction],
    list[DecodeRecord],
    list[ScoreRecord],
]:
    items = [_item(index) for index in range(n_items)]
    predictions: list[Prediction] = []
    decodes: list[DecodeRecord] = []
    scores: list[ScoreRecord] = []
    variants = ("canonical", "option_shuffle", "compact")
    for index, item in enumerate(items):
        for variant_index, variant in enumerate(variants):
            original_answer = "A"
            if index == 0 and variant == "option_shuffle":
                original_answer = "B"
            pred = _prediction(item, variant, original_answer)
            predictions.append(pred)
            decodes.append(_decode(pred, index, variant_index))
            for scorer in scorers:
                scores.append(_score(pred, index, variant_index, scorer=scorer))
    return items, predictions, decodes, scores


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(to_dict(row)) + b"\n")


def test_join_rows_success_and_join_failures() -> None:
    items, predictions, decodes, scores = _fixture(3)

    joined = join_rows(items, predictions, scores, decodes)

    assert len(joined) == len(scores)
    assert joined[0]["question"].startswith("Question")
    assert joined[0]["raw_av_text"].startswith("<explanation>")
    assert joined[0]["selected_answer_original"] in {"A", "B"}
    with pytest.raises(AnalysisJoinError, match="duplicate score"):
        join_rows(items, predictions, scores + [scores[0]], decodes)
    with pytest.raises(AnalysisJoinError, match="missing decode"):
        join_rows(items, predictions, scores, decodes[1:])
    with pytest.raises(AnalysisJoinError, match="duplicate prediction_id"):
        join_rows(items, predictions + [predictions[0]], scores, decodes)


def test_bootstrap_proportion_estimate_and_ci() -> None:
    values = [True] * 10 + [False] * 10
    item_ids = [f"item:{index}" for index in range(20)]

    point, lo, hi = bootstrap_proportion(values, item_ids, n_resamples=1000, seed=3)

    assert point == pytest.approx(0.5)
    assert 0.25 < lo < 0.5
    assert 0.5 < hi < 0.75


def test_bootstrap_resamples_item_blocks_not_rows() -> None:
    values = [True, True, False, False, False]
    item_ids = ["a", "a", "b", "c", "d"]

    actual = bootstrap_proportion(values, item_ids, n_resamples=25, seed=9)
    expected = _manual_item_block_bootstrap(values, item_ids, n_resamples=25, seed=9)

    assert actual == pytest.approx(expected)


def test_taxonomy_counts_sum_to_total() -> None:
    items, predictions, decodes, scores = _fixture(6)
    joined = join_rows(items, predictions, scores, decodes)

    rows = taxonomy_by_model_dataset(joined, n_resamples=25, seed=11)

    assert {row["taxonomy_cell"] for row in rows} == {
        "correct_aligned",
        "correct_weak",
        "incorrect_aligned",
        "incorrect_weak",
    }
    assert sum(row["count"] for row in rows) == len(joined)


def test_prompt_stability_uses_original_choice_space() -> None:
    items, predictions, decodes, scores = _fixture(3)
    joined = join_rows(items, predictions, scores, decodes)

    rows = {row["item_id"]: row for row in prompt_stability(joined)}

    assert rows["item:01"]["answer_agreement"] == 1.0
    assert rows["item:00"]["shuffle_changed_answer"] is True
    assert rows["item:01"]["shuffle_changed_answer"] is False


def test_multiple_scorers_remain_distinguishable() -> None:
    items, predictions, decodes, scores = _fixture(2, scorers=("heuristic_v1", "medgemma_judge_v1"))
    joined = join_rows(items, predictions, scores, decodes)

    stability_rows = prompt_stability(joined)
    failure_rows = failure_cases(joined)

    assert {row["scorer"] for row in stability_rows} == {"heuristic_v1", "medgemma_judge_v1"}
    assert {row["scorer"] for row in failure_rows} <= {"heuristic_v1", "medgemma_judge_v1"}


def test_failure_cases_are_correct_weak_top_k_sorted_by_cosine() -> None:
    items, predictions, decodes, scores = _fixture(10)
    joined = join_rows(items, predictions, scores, decodes)

    rows = failure_cases(joined, top_k=5)
    cosines = [row["reconstruction_cos"] for row in rows]

    assert len(rows) <= 5
    assert {row["taxonomy_cell"] for row in rows} == {"correct_weak"}
    assert cosines == sorted(cosines, reverse=True)


def test_cli_writes_required_outputs_and_manifest(tmp_path: Path) -> None:
    items, predictions, decodes, scores = _fixture(5)
    config = tmp_path / "config.yaml"
    items_path = tmp_path / "items.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    decodes_path = tmp_path / "decodes.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    out_dir = tmp_path / "analysis"
    manifest_path = out_dir / "analysis_manifest.json"
    config.write_text(
        "analysis:\n  bootstrap_resamples: 25\n  quick_bootstrap_resamples: 7\n",
        encoding="utf-8",
    )
    _write_jsonl(items_path, items)
    _write_jsonl(predictions_path, predictions)
    _write_jsonl(decodes_path, decodes)
    _write_jsonl(scores_path, scores)

    rc = analysis_cli._run(
        argparse.Namespace(
            config=str(config),
            items=str(items_path),
            predictions=str(predictions_path),
            decodes=str(decodes_path),
            scores=[str(scores_path)],
            out_dir=str(out_dir),
            bootstrap_resamples=None,
            quick=True,
            seed=42,
            manifest_out=str(manifest_path),
        )
    )

    assert rc == 0
    required = [
        out_dir / "tables" / "summary_by_model_dataset.csv",
        out_dir / "tables" / "taxonomy_by_model_dataset.csv",
        out_dir / "tables" / "prompt_stability.csv",
        out_dir / "tables" / "rationale_alignment_by_correctness.csv",
        out_dir / "tables" / "failure_cases.jsonl",
        out_dir / "figures_data" / "taxonomy_stack.jsonl",
        out_dir / "figures_data" / "accuracy_vs_aligned.jsonl",
        out_dir / "figures_data" / "reconstruction_by_cell.jsonl",
        out_dir / "_summary.json",
        manifest_path,
    ]
    for path in required:
        assert path.exists()
        assert path.stat().st_size > 0
    assert _csv_header(out_dir / "tables" / "summary_by_model_dataset.csv") == list(analysis_cli.SUMMARY_COLUMNS)
    assert _csv_header(out_dir / "tables" / "taxonomy_by_model_dataset.csv") == list(analysis_cli.TAXONOMY_COLUMNS)
    assert _csv_header(out_dir / "tables" / "prompt_stability.csv") == list(analysis_cli.PROMPT_STABILITY_COLUMNS)
    assert _csv_header(out_dir / "tables" / "rationale_alignment_by_correctness.csv") == list(
        analysis_cli.RATIONALE_COLUMNS
    )
    summary = orjson.loads((out_dir / "_summary.json").read_bytes())
    assert summary["bootstrap_resamples"] == 7
    assert summary["n_predictions"] == len(predictions)
    manifest = orjson.loads(manifest_path.read_bytes())
    assert manifest["stage"] == "mednla_analysis"


def _csv_header(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return next(csv.reader(handle))


def _manual_item_block_bootstrap(
    values: list[bool],
    item_ids: list[str],
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    import numpy as np

    values_arr = np.asarray(values, dtype=np.float64)
    blocks: dict[str, list[int]] = {}
    for index, item_id in enumerate(item_ids):
        blocks.setdefault(item_id, []).append(index)
    unique = np.asarray(sorted(blocks), dtype=object)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = [index for item_id in sampled for index in blocks[str(item_id)]]
        means.append(float(values_arr[indices].mean()))
    return float(values_arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
