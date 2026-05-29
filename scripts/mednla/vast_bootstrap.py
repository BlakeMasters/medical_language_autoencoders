"""Bootstrap bounded MedNLA dependencies on Vast runtimes."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger("nla.mednla.vast_bootstrap")

_T4_LIGHTWEIGHT_DEPS = (
    "pytest",
    "orjson",
    "pyyaml",
    "numpy",
    "pyarrow",
    "httpx",
    "safetensors",
    "huggingface_hub",
    "transformers>=4.46,<5",
)


def build_bootstrap_commands(
    *,
    profile: str,
    python_executable: str = sys.executable,
    skip_pip_upgrade: bool = False,
) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    if not skip_pip_upgrade:
        commands.append((python_executable, "-m", "pip", "install", "-U", "pip"))

    if profile == "t3":
        commands.append(
            (python_executable, "-m", "pip", "install", "-r", "requirements/mednla-vast-t3.txt")
        )
    elif profile == "t4-sglang":
        commands.extend(
            [
                (python_executable, "-m", "pip", "install", "--no-deps", "-e", "."),
                (python_executable, "-m", "pip", "install", *_T4_LIGHTWEIGHT_DEPS),
            ]
        )
    elif profile == "t5-judge":
        commands.append(
            (python_executable, "-m", "pip", "install", "-r", "requirements/mednla-vast-t5.txt")
        )
    else:
        raise ValueError(f"unknown profile: {profile!r}")
    return commands


def ensure_existing_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "torch is not importable in this runtime; refusing to install a replacement CUDA stack"
        ) from exc


def _run(args: argparse.Namespace) -> int:
    if args.require_existing_torch and args.profile != "t4-sglang":
        raise ValueError("--require-existing-torch is only valid for --profile t4-sglang")
    if args.require_existing_torch:
        ensure_existing_torch()

    commands = build_bootstrap_commands(
        profile=args.profile,
        python_executable=sys.executable,
        skip_pip_upgrade=args.skip_pip_upgrade,
    )
    if args.dry_run:
        for command in commands:
            print(format_command(command))
        return 0

    for command in commands:
        logger.info("run_bootstrap_command command=%s", format_command(command))
        result = subprocess.run(command, cwd=_REPO_ROOT, check=False)
        if result.returncode != 0:
            logger.error("bootstrap_command_failed returncode=%d", result.returncode)
            return 9
    return 0


def format_command(command: tuple[str, ...]) -> str:
    return shlex.join(command)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Bootstrap bounded MedNLA dependencies on Vast.")
    parser.add_argument("--profile", required=True, choices=["t3", "t4-sglang", "t5-judge"])
    parser.add_argument("--dry-run", action="store_true", help="Print pip commands without executing them.")
    parser.add_argument("--skip-pip-upgrade", action="store_true", help="Do not run pip install -U pip first.")
    parser.add_argument(
        "--require-existing-torch",
        action="store_true",
        help="For t4-sglang, fail if torch is not already importable.",
    )
    args = parser.parse_args()
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("bootstrap_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
