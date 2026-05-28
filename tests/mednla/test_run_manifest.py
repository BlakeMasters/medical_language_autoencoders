"""Tests for MedNLA run manifest helpers."""

from __future__ import annotations

import orjson

import nla.mednla.run_manifest as run_manifest


def test_build_run_manifest_redacts_secrets_and_writes_json(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        run_manifest,
        "_git_info",
        lambda repo_root: {
            "branch": "scopechange",
            "sha": "abc123",
            "dirty": True,
            "status_short": " M file.py",
        },
    )
    monkeypatch.setattr(run_manifest, "_package_versions", lambda: {"torch": "2.x"})
    monkeypatch.setattr(run_manifest, "_vast_env", lambda: {"CONTAINER_ID": "container"})

    manifest = run_manifest.build_run_manifest(
        stage="mednla_decode",
        started_at_utc="2026-05-28T00:00:00+00:00",
        ended_at_utc="2026-05-28T00:00:03+00:00",
        duration_sec=3.0,
        config_path=tmp_path / "config.yaml",
        model_short_name="qwen7b",
        args={
            "config": tmp_path / "config.yaml",
            "auth_token": "secret",
            "auth_token_env": "OPEN_BUTTON_TOKEN",
            "sglang_url": "http://127.0.0.1:18000",
        },
        outputs={"decodes": tmp_path / "decodes.jsonl"},
        repo_root=tmp_path,
        extra={"total": 30},
    )
    out = tmp_path / "manifest.json"
    run_manifest.write_run_manifest(out, manifest)

    data = orjson.loads(out.read_bytes())
    assert data["stage"] == "mednla_decode"
    assert data["args"]["auth_token"] == "<redacted>"
    assert data["args"]["auth_token_env"] == "<redacted>"
    assert data["args"]["config"].endswith("config.yaml")
    assert data["git"]["branch"] == "scopechange"
    assert data["packages"] == {"torch": "2.x"}
    assert data["vast"] == {"CONTAINER_ID": "container"}
    assert data["extra"] == {"total": 30}
