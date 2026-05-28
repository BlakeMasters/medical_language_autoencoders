"""Analysis helpers for MedNLA score artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np

from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord

TAXONOMY_CELLS = ("correct_aligned", "correct_weak", "incorrect_aligned", "incorrect_weak")


class AnalysisJoinError(ValueError):
    """Raised when MedNLA analysis artifacts cannot be joined losslessly."""


def join_rows(
    items: list[MedItem],
    predictions: list[Prediction],
    scores: list[ScoreRecord],
    decodes: list[DecodeRecord],
) -> list[dict[str, Any]]:
    items_by_id = _index_unique(items, "item_id")
    predictions_by_id = _index_unique(predictions, "prediction_id")
    decodes_by_pred = _index_unique(decodes, "prediction_id")
    score_keys: set[tuple[str, str]] = set()
    joined: list[dict[str, Any]] = []

    for score in scores:
        score_key = (score.scorer, score.prediction_id)
        if score_key in score_keys:
            raise AnalysisJoinError(f"duplicate score for scorer={score.scorer!r} prediction_id={score.prediction_id!r}")
        score_keys.add(score_key)

        pred = predictions_by_id.get(score.prediction_id)
        if pred is None:
            raise AnalysisJoinError(f"missing prediction for score prediction_id={score.prediction_id!r}")
        item = items_by_id.get(pred.item_id)
        if item is None:
            raise AnalysisJoinError(f"missing item for prediction_id={pred.prediction_id!r} item_id={pred.item_id!r}")
        decode = decodes_by_pred.get(pred.prediction_id)
        if decode is None:
            raise AnalysisJoinError(f"missing decode for prediction_id={pred.prediction_id!r}")

        selected_answer_original = _selected_answer_original(pred)
        joined.append(
            {
                "prediction_id": pred.prediction_id,
                "item_id": item.item_id,
                "dataset": item.dataset,
                "subject": item.subject,
                "model_short_name": pred.model_short_name,
                "prompt_variant": pred.prompt_variant,
                "correct": pred.correct,
                "selected_answer": pred.selected_answer,
                "selected_answer_original": selected_answer_original,
                "answer_key": item.answer_key,
                "nla_quality_binary": score.nla_quality_binary,
                "taxonomy_cell": score.taxonomy_cell,
                "medical_relevance": score.medical_relevance,
                "rationale_alignment": score.rationale_alignment,
                "answer_support": score.answer_support,
                "shortcut_suspected": score.shortcut_suspected,
                "medically_invalid": score.medically_invalid,
                "reconstruction_cos": decode.reconstruction_cos,
                "reconstruction_mse": decode.reconstruction_mse,
                "explanation": decode.explanation,
                "raw_av_text": decode.raw_av_text,
                "scorer": score.scorer,
                "scorer_notes": score.scorer_notes,
                "scorer_evidence": score.scorer_evidence,
                "question": item.question,
            }
        )
    return sorted(joined, key=lambda row: (row["scorer"], row["model_short_name"], row["dataset"], row["prediction_id"]))


def bootstrap_proportion(
    values: list[bool] | np.ndarray,
    item_ids: list[str],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.size == 0:
        raise ValueError("empty input")
    if values_arr.size != len(item_ids):
        raise ValueError("values and item_ids must have the same length")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")

    item_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, item_id in enumerate(item_ids):
        item_to_indices[str(item_id)].append(index)
    unique_items = np.asarray(sorted(item_to_indices), dtype=object)
    item_index_blocks = {item_id: np.asarray(indices, dtype=np.int64) for item_id, indices in item_to_indices.items()}
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)

    for resample_index in range(n_resamples):
        sampled_items = rng.choice(unique_items, size=len(unique_items), replace=True)
        sampled_indices = np.concatenate([item_index_blocks[str(item_id)] for item_id in sampled_items])
        boot_means[resample_index] = float(values_arr[sampled_indices].mean())

    return (
        float(values_arr.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def summary_by_model_dataset(
    joined: list[dict[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, dataset, scorer), group in _groups(joined, "model_short_name", "dataset", "scorer"):
        item_ids = [row["item_id"] for row in group]
        accuracy, accuracy_lo, accuracy_hi = bootstrap_proportion(
            [bool(row["correct"]) for row in group],
            item_ids,
            n_resamples=n_resamples,
            seed=seed,
        )
        aligned, aligned_lo, aligned_hi = bootstrap_proportion(
            [row["nla_quality_binary"] == "aligned" for row in group],
            item_ids,
            n_resamples=n_resamples,
            seed=seed + 1,
        )
        rows.append(
            {
                "model_short_name": model,
                "dataset": dataset,
                "n_items": len({row["item_id"] for row in group}),
                "n_predictions": len(group),
                "accuracy": accuracy,
                "accuracy_ci_lo": accuracy_lo,
                "accuracy_ci_hi": accuracy_hi,
                "aligned_rate": aligned,
                "aligned_rate_ci_lo": aligned_lo,
                "aligned_rate_ci_hi": aligned_hi,
                "mean_reconstruction_cos": _mean_present(row["reconstruction_cos"] for row in group),
                "mean_reconstruction_mse": _mean_present(row["reconstruction_mse"] for row in group),
                "scorer": scorer,
            }
        )
    return rows


def taxonomy_by_model_dataset(
    joined: list[dict[str, Any]],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, dataset, scorer), group in _groups(joined, "model_short_name", "dataset", "scorer"):
        item_ids = [row["item_id"] for row in group]
        for offset, cell in enumerate(TAXONOMY_CELLS):
            values = [row["taxonomy_cell"] == cell for row in group]
            proportion, lo, hi = bootstrap_proportion(
                values,
                item_ids,
                n_resamples=n_resamples,
                seed=seed + offset,
            )
            rows.append(
                {
                    "model_short_name": model,
                    "dataset": dataset,
                    "taxonomy_cell": cell,
                    "count": sum(values),
                    "proportion": proportion,
                    "proportion_ci_lo": lo,
                    "proportion_ci_hi": hi,
                    "scorer": scorer,
                }
            )
    return rows


def prompt_stability(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (item_id, model, scorer), group in _groups(joined, "item_id", "model_short_name", "scorer"):
        selected_originals = [row["selected_answer_original"] for row in group]
        canonical = [row for row in group if row["prompt_variant"] == "canonical"]
        shuffled = [row for row in group if row["prompt_variant"] == "option_shuffle"]
        shuffle_changed = False
        if canonical and shuffled:
            base = canonical[0]["selected_answer_original"]
            shuffle_changed = any(row["selected_answer_original"] != base for row in shuffled)
        rows.append(
            {
                "item_id": item_id,
                "model_short_name": model,
                "scorer": scorer,
                "n_variants": len(group),
                "answer_agreement": _all_equal_float(selected_originals),
                "correct_agreement": _all_equal_float([row["correct"] for row in group]),
                "quality_agreement": _all_equal_float([row["nla_quality_binary"] for row in group]),
                "medical_relevance_var": float(np.var([row["medical_relevance"] for row in group], dtype=np.float64)),
                "shuffle_changed_answer": shuffle_changed,
            }
        )
    return rows


def rationale_alignment_by_correctness(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (correct, scorer), group in _groups(joined, "correct", "scorer"):
        values = [row["rationale_alignment"] for row in group if row["rationale_alignment"] is not None]
        if values:
            arr = np.asarray(values, dtype=np.float64)
            mean_value: float | None = float(arr.mean())
            sd_value: float | None = float(arr.std())
            fraction_ge_1: float | None = float((arr >= 1).mean())
        else:
            mean_value = None
            sd_value = None
            fraction_ge_1 = None
        rows.append(
            {
                "correct": correct,
                "scorer": scorer,
                "n_with_rationale": len(values),
                "mean_rationale_alignment": mean_value,
                "sd_rationale_alignment": sd_value,
                "fraction_alignment_ge_1": fraction_ge_1,
            }
        )
    return rows


def failure_cases(joined: list[dict[str, Any]], *, top_k: int = 30) -> list[dict[str, Any]]:
    candidates = [row for row in joined if row["taxonomy_cell"] == "correct_weak"]
    sorted_rows = sorted(
        candidates,
        key=lambda row: (
            row["reconstruction_cos"] is None,
            -(float(row["reconstruction_cos"]) if row["reconstruction_cos"] is not None else 0.0),
            row["prediction_id"],
            row["scorer"],
        ),
    )
    return [
        {
            "prediction_id": row["prediction_id"],
            "item_id": row["item_id"],
            "model_short_name": row["model_short_name"],
            "prompt_variant": row["prompt_variant"],
            "scorer": row["scorer"],
            "correct": row["correct"],
            "taxonomy_cell": row["taxonomy_cell"],
            "question": row["question"],
            "answer_key": row["answer_key"],
            "selected_answer": row["selected_answer"],
            "explanation": row["explanation"],
            "scorer_notes": row["scorer_notes"],
            "reconstruction_cos": row["reconstruction_cos"],
        }
        for row in sorted_rows[:top_k]
    ]


def accuracy_vs_aligned(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, dataset, subject, scorer), group in _groups(
        joined,
        "model_short_name",
        "dataset",
        "subject",
        "scorer",
    ):
        rows.append(
            {
                "model_short_name": model,
                "dataset": dataset,
                "subject": subject,
                "scorer": scorer,
                "n_items": len({row["item_id"] for row in group}),
                "n_predictions": len(group),
                "accuracy": _mean_bool(row["correct"] for row in group),
                "aligned_rate": _mean_bool(row["nla_quality_binary"] == "aligned" for row in group),
                "correct_weak_rate": _mean_bool(row["taxonomy_cell"] == "correct_weak" for row in group),
            }
        )
    return rows


def reconstruction_by_cell(joined: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prediction_id": row["prediction_id"],
            "item_id": row["item_id"],
            "model_short_name": row["model_short_name"],
            "dataset": row["dataset"],
            "subject": row["subject"],
            "scorer": row["scorer"],
            "taxonomy_cell": row["taxonomy_cell"],
            "nla_quality_binary": row["nla_quality_binary"],
            "correct": row["correct"],
            "reconstruction_cos": row["reconstruction_cos"],
            "reconstruction_mse": row["reconstruction_mse"],
        }
        for row in joined
    ]


def _index_unique(rows: list[Any], key_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        key = str(getattr(row, key_name))
        if key in out:
            raise AnalysisJoinError(f"duplicate {key_name}: {key}")
        out[key] = row
    return out


def _selected_answer_original(pred: Prediction) -> str | None:
    if pred.selected_answer is None:
        return None
    inverse = {variant_key: original_key for original_key, variant_key in pred.original_to_variant_choice_map.items()}
    return inverse.get(pred.selected_answer, pred.selected_answer)


def _groups(rows: list[dict[str, Any]], *keys: str) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return [(key, grouped[key]) for key in sorted(grouped, key=lambda key: tuple("" if value is None else value for value in key))]


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return float(np.mean(np.asarray(present, dtype=np.float64)))


def _mean_bool(values: Iterable[bool]) -> float:
    values_list = [bool(value) for value in values]
    if not values_list:
        raise ValueError("empty input")
    return float(np.mean(np.asarray(values_list, dtype=np.float64)))


def _all_equal_float(values: list[Any]) -> float:
    if not values:
        return 1.0
    return 1.0 if len(set(values)) <= 1 else 0.0


__all__ = [
    "AnalysisJoinError",
    "TAXONOMY_CELLS",
    "accuracy_vs_aligned",
    "bootstrap_proportion",
    "failure_cases",
    "join_rows",
    "prompt_stability",
    "rationale_alignment_by_correctness",
    "reconstruction_by_cell",
    "summary_by_model_dataset",
    "taxonomy_by_model_dataset",
]
