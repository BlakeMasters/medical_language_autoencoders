"""Run selected MedNLA eval pipeline stages with per-stage logs."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.pipeline import (
    PipelineOptions,
    build_stage_commands,
    default_run_dir,
    format_command,
    parse_stage_list,
    stage_outputs_valid,
)

logger = logging.getLogger("nla.mednla.run_eval_pipeline")


def _run(args: argparse.Namespace) -> int:
    stages = parse_stage_list(args.stages, include_judge=args.score_judge)
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.config)
    options = PipelineOptions(
        config=Path(args.config),
        model=args.model,
        run_dir=run_dir,
        stages=stages,
        quick_analysis=args.quick_analysis,
        sglang_url=args.sglang_url,
        no_critic=args.no_critic,
        allow_cjk_warnings=args.allow_cjk_warnings,
        include_judge=args.score_judge,
        auth_token_env=args.auth_token_env,
    )
    commands = build_stage_commands(options)

    if args.dry_run:
        for command in commands:
            print(format_command(command.argv))
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    for command in commands:
        if args.resume and stage_outputs_valid(command):
            logger.info("skip_stage stage=%s reason=valid_outputs", command.name)
            continue
        command.log_path.parent.mkdir(parents=True, exist_ok=True)
        with command.log_path.open("wb") as log_handle:
            log_handle.write(f"$ {format_command(command.argv)}\n".encode("utf-8"))
            log_handle.flush()
            result = subprocess.run(
                command.argv,
                cwd=_REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            logger.error(
                "stage_failed stage=%s returncode=%d log=%s",
                command.name,
                result.returncode,
                command.log_path,
            )
            return int(result.returncode)
        logger.info("stage_completed stage=%s log=%s", command.name, command.log_path)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run selected MedNLA eval pipeline stages.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Model short_name from config.")
    parser.add_argument("--run-dir", default=None, help="Run directory. Default: runs/mednla/<config.run_id>.")
    parser.add_argument("--stages", default=None, help="Comma-separated stages. Default: prepare,score_heuristic,analysis,validate.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--resume", action="store_true", help="Skip stages whose expected outputs already validate lightly.")
    parser.add_argument("--quick-analysis", action="store_true", help="Pass --quick to analyze_results.py.")
    parser.add_argument("--sglang-url", default=None, help="Override config SGLang URL for readiness/decode stages.")
    parser.add_argument("--no-critic", action="store_true", help="Pass --no-critic to decode stage.")
    parser.add_argument("--allow-cjk-warnings", action="store_true", help="Allow CJK warnings in decode stage.")
    parser.add_argument("--score-judge", action="store_true", help="Include/expect scores_judge.jsonl.")
    parser.add_argument("--auth-token-env", default=None, help="Environment variable containing SGLang bearer token.")
    args = parser.parse_args()

    try:
        return _run(args)
    except (OSError, ValueError) as exc:
        logger.error("pipeline_config_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
