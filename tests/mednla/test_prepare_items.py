"""Tests for the MedNLA prepare_items CLI."""

from __future__ import annotations

import sys

from scripts.mednla import prepare_items


def test_main_reports_dataset_load_failures(
    monkeypatch,
    tmp_path,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    output_path = tmp_path / "items.jsonl"
    cfg_path.write_text(
        "\n".join(
            [
                "run_id: test_run",
                "seed: 1",
                "sample_size: 1",
                "datasets: []",
            ]
        ),
        encoding="utf-8",
    )

    def fail_load_items(cfg, *, seed: int):
        del cfg, seed
        raise RuntimeError("huggingface timeout")

    monkeypatch.setattr(prepare_items, "load_items", fail_load_items)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_items.py",
            "--config",
            str(cfg_path),
            "--output",
            str(output_path),
        ],
    )

    assert prepare_items.main() == 4
    assert not output_path.exists()
