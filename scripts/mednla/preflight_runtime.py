"""Inspect local runtime readiness for MedNLA eval commands."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
from typing import Any

_PACKAGE_NAMES = (
    "torch",
    "transformers",
    "sglang",
    "numpy",
    "pyarrow",
    "httpx",
    "orjson",
    "pyyaml",
    "safetensors",
    "huggingface_hub",
    "datasets",
)
_COMMAND_NAMES = ("git", "nvidia-smi", "vastai", "rsync", "scp", "ssh")


def inspect_runtime(*, path: str | Path = ".") -> dict[str, Any]:
    torch_info = _torch_info()
    disk = shutil.disk_usage(path)
    return {
        "ok": True,
        "errors": [],
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "packages": _package_versions(),
        "torch": torch_info,
        "cuda": {
            "available": bool(torch_info.get("cuda_available")),
            "device_count": int(torch_info.get("cuda_device_count") or 0),
            "device_name": torch_info.get("cuda_device_name"),
            "torch_cuda": torch_info.get("torch_cuda"),
        },
        "disk": {
            "path": str(path),
            "free_gb": disk.free / (1024**3),
            "total_gb": disk.total / (1024**3),
        },
        "commands": {name: shutil.which(name) for name in _COMMAND_NAMES},
    }


def apply_requirements(
    summary: dict[str, Any],
    *,
    require_cuda: bool = False,
    require_torch: bool = False,
    min_free_gb: float | None = None,
) -> dict[str, Any]:
    errors = list(summary.get("errors", []))
    if require_torch and not summary["torch"]["importable"]:
        errors.append("torch is not importable")
    if require_cuda and not summary["cuda"]["available"]:
        errors.append("cuda is not available")
    if min_free_gb is not None and summary["disk"]["free_gb"] < min_free_gb:
        errors.append(
            f"disk free {summary['disk']['free_gb']:.2f} GiB below required {min_free_gb:.2f} GiB"
        )
    summary = dict(summary)
    summary["errors"] = errors
    summary["ok"] = not errors
    return summary


def write_json(path: str | Path, summary: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _run(args: argparse.Namespace) -> int:
    summary = inspect_runtime(path=args.path)
    summary = apply_requirements(
        summary,
        require_cuda=args.require_cuda,
        require_torch=args.require_torch,
        min_free_gb=args.min_free_gb,
    )
    if args.json_out:
        write_json(args.json_out, summary)
    else:
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True))
        sys.stdout.write("\n")
    return 0 if summary["ok"] else 8


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect MedNLA runtime readiness.")
    parser.add_argument("--json-out", default=None, help="Optional output JSON path.")
    parser.add_argument("--require-cuda", action="store_true", help="Fail if torch CUDA is unavailable.")
    parser.add_argument("--require-torch", action="store_true", help="Fail if torch cannot be imported.")
    parser.add_argument("--min-free-gb", type=float, default=None, help="Fail if disk free space is below this GiB.")
    parser.add_argument("--path", default=".", help="Path whose filesystem should be checked for free space.")
    args = parser.parse_args()
    try:
        return _run(args)
    except OSError as exc:
        sys.stderr.write(f"preflight_io_error: {exc}\n")
        return 2


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in _PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "importable": False,
            "error": str(exc),
            "version": None,
            "torch_cuda": None,
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if device_count else None
    return {
        "importable": True,
        "error": None,
        "version": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_device_name": device_name,
    }


if __name__ == "__main__":
    sys.exit(main())
