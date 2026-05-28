"""Tests for the MedNLA SGLang readiness CLI."""

from __future__ import annotations

import argparse
import types
from pathlib import Path

import httpx
import orjson
import pytest

import scripts.mednla.check_sglang_ready as ready_cli

D_MODEL = 8


class FakeReadyClient:
    cfg = types.SimpleNamespace(d_model=D_MODEL)
    raw_text = "<explanation>ready</explanation>"
    error = None
    last = None

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self._http = types.SimpleNamespace(headers={}, timeout=None, closed=False, close=self._close)
        type(self).last = self

    def _close(self):
        self._http.closed = True

    def generate(self, activation, **kwargs):
        self.activation = activation
        self.generate_kwargs = kwargs
        if type(self).error is not None:
            raise type(self).error
        return type(self).raw_text


@pytest.fixture(autouse=True)
def reset_fake_client() -> None:
    FakeReadyClient.cfg = types.SimpleNamespace(d_model=D_MODEL)
    FakeReadyClient.raw_text = "<explanation>ready</explanation>"
    FakeReadyClient.error = None
    FakeReadyClient.last = None


def _config(path: Path, *, d_model: int = D_MODEL) -> None:
    path.write_text(
        "\n".join(
            [
                "models:",
                "  - short_name: fake",
                "    nla_actor: actor",
                f"    d_model: {d_model}",
                "nla_decode:",
                "  sglang_url: http://localhost:30000",
            ]
        ),
        encoding="utf-8",
    )


def _args(config: Path, **overrides) -> argparse.Namespace:
    defaults = {
        "config": str(config),
        "model": "fake",
        "sglang_url": "http://127.0.0.1:18000",
        "auth_token": None,
        "auth_token_env": None,
        "timeout": 1.0,
        "skip_generate": False,
        "activation_value": 0.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _summary(capsys) -> dict:
    return orjson.loads(capsys.readouterr().out)


def test_ready_success_with_auth_env(monkeypatch, tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    _config(config)
    monkeypatch.setattr(ready_cli, "NLAClient", FakeReadyClient)
    monkeypatch.setattr(ready_cli, "resolve_checkpoint_path", lambda ref: tmp_path / str(ref))
    monkeypatch.setenv("SGLANG_TOKEN", "secret-from-env")

    assert ready_cli._run(_args(config, auth_token_env="SGLANG_TOKEN", activation_value=1.25)) == 0

    summary = _summary(capsys)
    assert summary["ok"] is True
    assert summary["sglang_url"] == "http://127.0.0.1:18000"
    assert summary["d_model"] == D_MODEL
    assert summary["raw_text_returned"] is True
    assert FakeReadyClient.last._http.headers["Authorization"] == "Bearer secret-from-env"
    assert FakeReadyClient.last.activation.shape == (D_MODEL,)
    assert FakeReadyClient.last.activation[0] == 1.25
    assert FakeReadyClient.last._http.closed is True


def test_ready_empty_output_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    _config(config)
    FakeReadyClient.raw_text = ""
    FakeReadyClient.error = None
    monkeypatch.setattr(ready_cli, "NLAClient", FakeReadyClient)
    monkeypatch.setattr(ready_cli, "resolve_checkpoint_path", lambda ref: tmp_path / str(ref))

    assert ready_cli._run(_args(config)) == 5

    summary = _summary(capsys)
    assert summary["ok"] is False
    assert summary["error_type"] == "EmptyGenerateOutput"
    FakeReadyClient.raw_text = "<explanation>ready</explanation>"


def test_ready_http_failure_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    _config(config)
    request = httpx.Request("POST", "http://127.0.0.1:18000/generate")
    response = httpx.Response(401, request=request)
    FakeReadyClient.error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
    monkeypatch.setattr(ready_cli, "NLAClient", FakeReadyClient)
    monkeypatch.setattr(ready_cli, "resolve_checkpoint_path", lambda ref: tmp_path / str(ref))

    assert ready_cli._run(_args(config)) == 4

    summary = _summary(capsys)
    assert summary["error_type"] == "HTTPStatusError"
    assert summary["response_status"] == 401
    FakeReadyClient.error = None


def test_ready_checkpoint_load_failure_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    _config(config)
    monkeypatch.setattr(ready_cli, "resolve_checkpoint_path", lambda ref: (_ for _ in ()).throw(RuntimeError("boom")))

    assert ready_cli._run(_args(config)) == 3

    summary = _summary(capsys)
    assert summary["error_type"] == "RuntimeError"
    assert "boom" in summary["error"]


def test_ready_d_model_mismatch_fails(monkeypatch, tmp_path: Path, capsys) -> None:
    config = tmp_path / "config.yaml"
    _config(config, d_model=D_MODEL + 1)
    monkeypatch.setattr(ready_cli, "NLAClient", FakeReadyClient)
    monkeypatch.setattr(ready_cli, "resolve_checkpoint_path", lambda ref: tmp_path / str(ref))

    assert ready_cli._run(_args(config, skip_generate=True)) == 5

    summary = _summary(capsys)
    assert summary["error_type"] == "DModelMismatch"
