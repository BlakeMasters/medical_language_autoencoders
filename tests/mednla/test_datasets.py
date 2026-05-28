"""Tests for MedNLA dataset loading and sampling."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import orjson
import pytest
import yaml
from datasets import Dataset

from nla.mednla.datasets import load_items
from nla.mednla.schema import to_dict

ITEM_ID_RE = re.compile(r"^[a-z_]+:[a-z]+:[0-9]{6}$")


def _medqa_rows() -> list[dict[str, Any]]:
    rows = []
    for index in range(6):
        meta = "step1" if index < 3 else "step2"
        rows.append(
            {
                "question": f"Question {index}?",
                "options": {
                    "A": f"Option A{index}",
                    "B": f"Option B{index}",
                    "C": f"Option C{index}",
                    "D": f"Option D{index}",
                },
                "answer": f"Option A{index}",
                "answer_idx": "A",
                "meta_info": meta,
            }
        )
    return rows


def _medmcqa_rows() -> list[dict[str, Any]]:
    return [
        {
            "id": "mcq-1",
            "question": "MedMCQA question?",
            "opa": "Alpha",
            "opb": "Beta",
            "opc": "Gamma",
            "opd": "Delta",
            "cop": 2,
            "exp": "Because gamma.",
            "subject_name": "pharmacology",
            "topic_name": "antibiotics",
        }
    ]


def _selection_hash(items: list) -> str:
    payload = b"".join(orjson.dumps(to_dict(item)) for item in items)
    return hashlib.sha256(payload).hexdigest()


def _fake_load_dataset(
    path: str,
    name: str | None = None,
    split: str | None = None,
    revision: str | None = None,
):
    del name, split, revision
    if path == "GBaker/MedQA-USMLE-4-options":
        return Dataset.from_list(_medqa_rows())
    if path == "openlifescienceai/medmcqa":
        return Dataset.from_list(_medmcqa_rows())
    raise ValueError(f"unexpected dataset path: {path}")


def _base_cfg(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "run_id": "test_run",
        "seed": 42,
        "sample_size": 4,
        "datasets": [
            {
                "name": "medqa",
                "hf_path": "GBaker/MedQA-USMLE-4-options",
                "hf_config": None,
                "revision": "main",
                "split": "test",
            }
        ],
    }
    cfg.update(overrides)
    return cfg


def test_load_items_sample_size_when_source_is_larger(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    items = load_items(_base_cfg(sample_size=4), seed=42)
    assert len(items) == 4


def test_load_items_returns_all_when_source_is_smaller(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    items = load_items(_base_cfg(sample_size=100), seed=42)
    assert len(items) == 6


def test_load_items_is_deterministic_for_same_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    first = load_items(_base_cfg(sample_size=4), seed=7)
    second = load_items(_base_cfg(sample_size=4), seed=7)
    assert _selection_hash(first) == _selection_hash(second)


def test_load_items_changes_when_seed_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    first = load_items(_base_cfg(sample_size=4), seed=7)
    second = load_items(_base_cfg(sample_size=4), seed=8)
    assert _selection_hash(first) != _selection_hash(second)


def test_item_id_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    items = load_items(_base_cfg(sample_size=4), seed=1)
    for item in items:
        assert ITEM_ID_RE.match(item.item_id)


def test_medqa_empty_choice_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)

    def bad_loader(path: str, name=None, split=None, revision=None):
        del name, split, revision
        rows = _medqa_rows()
        rows[0]["options"]["B"] = "   "
        return Dataset.from_list(rows)

    monkeypatch.setattr("nla.mednla.datasets.load_dataset", bad_loader)
    with pytest.raises(ValueError, match="medqa:test:000000"):
        load_items(_base_cfg(sample_size=1), seed=1)


def test_medmcqa_cop_maps_to_answer_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    cfg = _base_cfg(
        sample_size=1,
        datasets=[
            {
                "name": "medmcqa",
                "hf_path": "openlifescienceai/medmcqa",
                "hf_config": None,
                "revision": "main",
                "split": "validation",
            }
        ],
    )
    items = load_items(cfg, seed=1)
    assert len(items) == 1
    assert items[0].answer_key == "C"


def test_stratified_sampling_when_subject_coverage_is_high(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nla.mednla.datasets.load_dataset", _fake_load_dataset)
    items = load_items(_base_cfg(sample_size=4), seed=99)
    subjects = {item.subject for item in items}
    assert len(subjects) >= 2


def test_pilot_config_uses_pinned_hf_revision() -> None:
    cfg_path = Path("configs/mednla/pilot_qwen7b_medqa.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for dataset_entry in cfg["datasets"]:
        revision = dataset_entry["revision"]
        assert revision != "main"
        assert re.fullmatch(r"[0-9a-f]{40}", revision)
