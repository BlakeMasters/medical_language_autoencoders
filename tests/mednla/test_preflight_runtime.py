"""Tests for MedNLA runtime preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson

import scripts.mednla.preflight_runtime as preflight


def _summary(*, cuda: bool = False, free_gb: float = 100.0, torch_importable: bool = True) -> dict:
    return {
        "ok": True,
        "errors": [],
        "python": {"version": "3.x", "executable": "python"},
        "packages": {"torch": "2.x", "orjson": "3.x"},
        "torch": {
            "importable": torch_importable,
            "error": None,
            "version": "2.x" if torch_importable else None,
            "torch_cuda": "12.4" if torch_importable else None,
            "cuda_available": cuda,
            "cuda_device_count": 1 if cuda else 0,
            "cuda_device_name": "GPU" if cuda else None,
        },
        "cuda": {
            "available": cuda,
            "device_count": 1 if cuda else 0,
            "device_name": "GPU" if cuda else None,
            "torch_cuda": "12.4" if torch_importable else None,
        },
        "disk": {"path": ".", "free_gb": free_gb, "total_gb": 200.0},
        "commands": {"git": "/usr/bin/git", "nvidia-smi": None},
    }


def test_inspect_runtime_summary_contains_expected_sections() -> None:
    summary = preflight.inspect_runtime()

    assert "executable" in summary["python"]
    assert "torch" in summary["packages"]
    assert "available" in summary["cuda"]
    assert "free_gb" in summary["disk"]
    assert "git" in summary["commands"]


def test_require_cuda_returns_8_when_cuda_unavailable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight, "inspect_runtime", lambda path=".": _summary(cuda=False))

    rc = preflight._run(
        argparse.Namespace(
            json_out=str(tmp_path / "preflight.json"),
            require_cuda=True,
            require_torch=False,
            min_free_gb=None,
            path=".",
        )
    )

    assert rc == 8
    data = orjson.loads((tmp_path / "preflight.json").read_bytes())
    assert data["ok"] is False
    assert "cuda is not available" in data["errors"]


def test_min_free_gb_threshold_failure_returns_8(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight, "inspect_runtime", lambda path=".": _summary(free_gb=2.0))

    rc = preflight._run(
        argparse.Namespace(
            json_out=str(tmp_path / "preflight.json"),
            require_cuda=False,
            require_torch=False,
            min_free_gb=10.0,
            path=".",
        )
    )

    assert rc == 8
    data = orjson.loads((tmp_path / "preflight.json").read_bytes())
    assert "below required" in data["errors"][0]


def test_json_out_writes_parseable_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(preflight, "inspect_runtime", lambda path=".": _summary(cuda=True))

    out = tmp_path / "preflight.json"
    rc = preflight._run(
        argparse.Namespace(
            json_out=str(out),
            require_cuda=True,
            require_torch=True,
            min_free_gb=1.0,
            path=".",
        )
    )

    assert rc == 0
    data = orjson.loads(out.read_bytes())
    assert data["ok"] is True
    assert data["python"]["executable"] == "python"
