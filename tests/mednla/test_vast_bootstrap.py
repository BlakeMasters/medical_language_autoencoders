"""Tests for Vast dependency bootstrap helper."""

from __future__ import annotations

import argparse
import builtins
import subprocess

import pytest

import scripts.mednla.vast_bootstrap as vast_bootstrap


def _args(
    *,
    profile: str,
    dry_run: bool = False,
    skip_pip_upgrade: bool = True,
    require_existing_torch: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        profile=profile,
        dry_run=dry_run,
        skip_pip_upgrade=skip_pip_upgrade,
        require_existing_torch=require_existing_torch,
    )


def test_t4_sglang_dry_run_uses_no_deps_and_never_installs_torch_or_sglang_all(capsys) -> None:
    rc = vast_bootstrap._run(_args(profile="t4-sglang", dry_run=True))

    out = capsys.readouterr().out
    assert rc == 0
    assert "pip install --no-deps -e ." in out
    assert "sglang[all]" not in out
    tokens = out.replace('"', "").split()
    assert "torch" not in tokens


def test_require_existing_torch_fails_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="torch is not importable"):
        vast_bootstrap.ensure_existing_torch()


def test_command_failure_returns_9_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 1)

    monkeypatch.setattr(vast_bootstrap.subprocess, "run", fake_run)

    rc = vast_bootstrap._run(_args(profile="t3", skip_pip_upgrade=False))

    assert rc == 9
    assert len(calls) == 1
    assert calls[0][-2:] == ("-U", "pip")


def test_unknown_profile_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        vast_bootstrap.build_bootstrap_commands(profile="unknown")
