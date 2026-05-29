"""Validate a MedNLA run directory."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.run_validation import RunValidationIOError, ValidationOptions, validate_run, write_validation_summary

logger = logging.getLogger("nla.mednla.validate_run")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _run(args: argparse.Namespace) -> int:
    cfg = _load_yaml(Path(args.config))
    run_dir = Path(args.run_dir)
    out_path = Path(args.out) if args.out else run_dir / "run_validation.json"
    summary = validate_run(
        ValidationOptions(
            config=cfg,
            run_dir=run_dir,
            model_short_name=args.model,
            parse_ok_threshold=args.parse_ok_threshold,
            require_analysis=args.require_analysis,
            require_judge_score=args.require_judge_score,
            require_manifests=args.require_manifests,
        )
    )
    write_validation_summary(out_path, summary)
    if summary["ok"]:
        logger.info("validation_passed out=%s", out_path)
        return 0
    logger.error("validation_failed errors=%s out=%s", summary["errors"], out_path)
    return 8


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Validate MedNLA run artifacts.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-dir", required=True, help="MedNLA run artifact directory.")
    parser.add_argument("--model", default=None, help="Expected model short_name for predictions.")
    parser.add_argument("--out", default=None, help="Output validation JSON. Default: <run-dir>/run_validation.json.")
    parser.add_argument("--parse-ok-threshold", type=float, default=0.8)
    parser.add_argument("--require-analysis", action="store_true", help="Require T6 analysis table/figure outputs.")
    parser.add_argument("--require-judge-score", action="store_true", help="Require scores_judge.jsonl.")
    parser.add_argument("--require-manifests", action="store_true", help="Require stage manifest JSON files.")
    args = parser.parse_args()
    try:
        return _run(args)
    except (OSError, ValueError, yaml.YAMLError, RunValidationIOError) as exc:
        logger.error("config_or_io_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
