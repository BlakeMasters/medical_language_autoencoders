"""Small JSON manifests for MedNLA run artifacts."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import orjson

_SECRET_MARKERS = ("token", "key", "secret", "password", "credential")
_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "sglang",
    "numpy",
    "pyarrow",
    "httpx",
    "orjson",
    "pyyaml",
    "datasets",
)
_VAST_ENV_KEYS = (
    "CONTAINER_ID",
    "CUDA_VISIBLE_DEVICES",
    "GPU_COUNT",
    "VAST_CONTAINERLABEL",
    "VAST_ID",
    "VAST_INSTANCE_ID",
    "VAST_BUNDLE_ID",
)


def build_run_manifest(
    *,
    stage: str,
    started_at_utc: str,
    ended_at_utc: str,
    duration_sec: float,
    config_path: str | Path,
    model_short_name: str,
    args: Mapping[str, Any],
    outputs: Mapping[str, Any],
    repo_root: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_sec": float(duration_sec),
        "config_path": str(config_path),
        "model_short_name": model_short_name,
        "args": _redact_mapping(args),
        "outputs": _jsonable(outputs),
        "git": _git_info(root),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "vast": _vast_env(),
    }
    if extra:
        manifest["extra"] = _jsonable(dict(extra))
    return manifest


def write_run_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(orjson.dumps(_jsonable(dict(manifest)), option=orjson.OPT_INDENT_2))


def _redact_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in mapping.items():
        lowered = str(key).lower()
        if any(marker in lowered for marker in _SECRET_MARKERS) and value not in (None, ""):
            redacted[str(key)] = "<redacted>"
        else:
            redacted[str(key)] = _jsonable(value)
    return redacted


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in _PACKAGE_NAMES:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _git_info(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    status = git("status", "--short")
    return {
        "branch": git("branch", "--show-current"),
        "sha": git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status,
    }


def _vast_env() -> dict[str, str]:
    return {
        key: value
        for key in _VAST_ENV_KEYS
        if (value := os.environ.get(key))
    }


__all__ = ["build_run_manifest", "write_run_manifest"]
