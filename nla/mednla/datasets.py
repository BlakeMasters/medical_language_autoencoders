"""Load and normalize MedQA / MedMCQA items from Hugging Face datasets."""

from __future__ import annotations

import hashlib
import logging
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from datasets import load_dataset

from nla.mednla.schema import MedItem

logger = logging.getLogger("nla.mednla.datasets")

_REQUIRED_CFG_KEYS = ("run_id", "seed", "sample_size", "datasets")
_ANSWER_LETTERS = "ABCD"


def _stable_hash(value: tuple[Any, ...]) -> int:
    return int.from_bytes(hashlib.sha256(repr(value).encode()).digest()[:8], "big")


def _normalize_medqa(
    row: dict[str, Any],
    hf_index: int,
    split: str,
    revision: str | None,
    *,
    hf_path: str,
    hf_config: str | None,
) -> MedItem:
    item_id = f"medqa:{split}:{hf_index:06d}"
    try:
        options = row["options"]
        choices = dict(sorted(options.items()))
        answer_key = row["answer_idx"].strip().upper()
        if answer_key not in choices:
            raise ValueError(f"answer_idx {answer_key!r} not in choices")
        for key, text in choices.items():
            if not str(text).strip():
                raise ValueError(f"empty choice text for key {key!r}")
        subject_raw = row.get("meta_info")
        subject = (subject_raw or "").strip() or None
        return MedItem(
            item_id=item_id,
            dataset="medqa",
            split=split,
            subject=subject,
            question=str(row["question"]),
            choices=choices,
            answer_key=answer_key,
            gold_rationale=None,
            source_metadata={
                "hf_path": hf_path,
                "hf_config": hf_config,
                "hf_index": hf_index,
                "hf_revision": revision,
            },
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{item_id}: {exc}") from exc


def _normalize_medmcqa(
    row: dict[str, Any],
    hf_index: int,
    split: str,
    revision: str | None,
    *,
    hf_path: str,
    hf_config: str | None,
) -> MedItem:
    item_id = f"medmcqa:{split}:{hf_index:06d}"
    try:
        cop = int(row["cop"])
        if cop < 0 or cop > 3:
            raise ValueError(f"cop out of range: {cop}")
        answer_key = _ANSWER_LETTERS[cop]
        choices = {
            "A": str(row["opa"]),
            "B": str(row["opb"]),
            "C": str(row["opc"]),
            "D": str(row["opd"]),
        }
        subject_raw = row.get("subject_name")
        subject = (subject_raw or "").strip() or None
        rationale_raw = row.get("exp")
        gold_rationale = (rationale_raw or "").strip() or None
        source_metadata: dict[str, Any] = {
            "hf_path": hf_path,
            "hf_config": hf_config,
            "hf_index": hf_index,
            "hf_revision": revision,
            "medmcqa_id": row["id"],
            "topic_name": row.get("topic_name"),
        }
        return MedItem(
            item_id=item_id,
            dataset="medmcqa",
            split=split,
            subject=subject,
            question=str(row["question"]),
            choices=choices,
            answer_key=answer_key,
            gold_rationale=gold_rationale,
            source_metadata=source_metadata,
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{item_id}: {exc}") from exc


DATASET_ADAPTERS: dict[str, Callable[..., MedItem]] = {
    "medqa": _normalize_medqa,
    "medmcqa": _normalize_medmcqa,
}


def load_items(cfg: dict[str, Any], *, seed: int) -> list[MedItem]:
    for key in _REQUIRED_CFG_KEYS:
        if key not in cfg:
            raise ValueError(key)

    all_items: list[MedItem] = []
    seen_ids: set[str] = set()

    for dataset_entry in cfg["datasets"]:
        dataset_name = dataset_entry["name"]
        if dataset_name not in DATASET_ADAPTERS:
            raise ValueError(f"unknown dataset name: {dataset_name!r}")
        adapter = DATASET_ADAPTERS[dataset_name]
        hf_path = dataset_entry["hf_path"]
        hf_config = dataset_entry.get("hf_config")
        split = dataset_entry["split"]
        revision = dataset_entry.get("revision")

        ds = load_dataset(
            hf_path,
            name=hf_config,
            split=split,
            revision=revision,
        )
        for hf_index, row in enumerate(ds):
            item = adapter(
                row,
                hf_index,
                split,
                revision,
                hf_path=hf_path,
                hf_config=hf_config,
            )
            if item.item_id in seen_ids:
                raise ValueError(f"duplicate item_id: {item.item_id}")
            seen_ids.add(item.item_id)
            all_items.append(item)

    sample_size = int(cfg["sample_size"])
    if len(all_items) <= sample_size:
        return sorted(all_items, key=lambda item: item.item_id)

    subject_count = sum(1 for item in all_items if item.subject is not None)
    subject_rate = subject_count / len(all_items)
    if subject_rate >= 0.5:
        sampled = _stratified_sample(all_items, sample_size, seed)
    else:
        logger.warning(
            "subject coverage=%.1f%% below 50%%; using uniform sampling",
            subject_rate * 100.0,
        )
        sampled = _uniform_sample(all_items, sample_size, seed)

    return sorted(sampled, key=lambda item: item.item_id)


def _uniform_sample(items: list[MedItem], sample_size: int, seed: int) -> list[MedItem]:
    rng = random.Random(_stable_hash((seed, "items")))
    return rng.sample(items, sample_size)


def _stratified_sample(items: list[MedItem], sample_size: int, seed: int) -> list[MedItem]:
    buckets: dict[tuple[str, str | None], list[MedItem]] = defaultdict(list)
    for item in items:
        buckets[(item.dataset, item.subject)].append(item)

    total = len(items)
    strata = sorted(buckets.keys())
    quotas: dict[tuple[str, str | None], int] = {key: 0 for key in strata}
    remainders: list[tuple[float, tuple[str, str | None]]] = []
    allocated = 0
    for key in strata:
        exact = sample_size * len(buckets[key]) / total
        base = int(exact)
        quotas[key] = base
        allocated += base
        remainders.append((exact - base, key))

    remainder = sample_size - allocated
    remainders.sort(key=lambda pair: (-pair[0], pair[1]))
    for index in range(remainder):
        quotas[remainders[index][1]] += 1

    selected: list[MedItem] = []
    selected_ids: set[str] = set()
    for key in strata:
        bucket = buckets[key]
        draw_count = min(quotas[key], len(bucket))
        if draw_count == 0:
            continue
        rng = random.Random(_stable_hash((seed, "items", key[0], str(key[1]))))
        picks = rng.sample(bucket, draw_count)
        for item in picks:
            selected.append(item)
            selected_ids.add(item.item_id)

    if len(selected) < sample_size:
        remaining = [item for item in items if item.item_id not in selected_ids]
        need = sample_size - len(selected)
        top_up_rng = random.Random(_stable_hash((seed, "items")))
        selected.extend(top_up_rng.sample(remaining, min(need, len(remaining))))

    if len(selected) > sample_size:
        trim_rng = random.Random(_stable_hash((seed, "items", "trim")))
        selected = trim_rng.sample(selected, sample_size)

    return selected


__all__ = ["DATASET_ADAPTERS", "load_items"]
