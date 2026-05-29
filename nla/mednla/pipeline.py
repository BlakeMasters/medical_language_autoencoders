"""Command construction helpers for MedNLA eval orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import subprocess
import sys

import orjson
import yaml

STAGES = (
    "preflight",
    "prepare",
    "probe",
    "check_sglang",
    "decode",
    "score_heuristic",
    "score_judge",
    "analysis",
    "validate",
)
DEFAULT_STAGES = ("prepare", "score_heuristic", "analysis", "validate")


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    config: Path
    model: str
    run_dir: Path
    stages: tuple[str, ...] = DEFAULT_STAGES
    quick_analysis: bool = False
    sglang_url: str | None = None
    no_critic: bool = False
    allow_cjk_warnings: bool = False
    include_judge: bool = False
    auth_token_env: str | None = None
    python_executable: str = sys.executable


@dataclass(frozen=True, slots=True)
class StageCommand:
    name: str
    argv: tuple[str, ...]
    outputs: tuple[Path, ...]
    log_path: Path


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def default_run_dir(config_path: str | Path) -> Path:
    cfg = load_config(config_path)
    run_id = cfg.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("config run_id must be a non-empty string")
    return Path("runs") / "mednla" / run_id


def parse_stage_list(stages_arg: str | None, *, include_judge: bool = False) -> tuple[str, ...]:
    stages = DEFAULT_STAGES if stages_arg is None else tuple(stage.strip() for stage in stages_arg.split(",") if stage.strip())
    if include_judge and stages_arg is None:
        stages = ("prepare", "score_heuristic", "score_judge", "analysis", "validate")
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError(f"unknown stage(s): {', '.join(unknown)}")
    if len(set(stages)) != len(stages):
        raise ValueError("duplicate stages are not supported")
    return stages


def build_stage_commands(options: PipelineOptions) -> list[StageCommand]:
    run = options.run_dir
    logs = run / "logs"
    items = run / "items.jsonl"
    predictions = run / "predictions.jsonl"
    activations = run / "activations.parquet"
    decodes = run / "decodes.jsonl"
    scores_heuristic = run / "scores_heuristic.jsonl"
    scores_judge = run / "scores_judge.jsonl"
    analysis_outputs = _analysis_outputs(run)
    commands: list[StageCommand] = []

    for stage in options.stages:
        if stage == "preflight":
            argv = (
                options.python_executable,
                "scripts/mednla/preflight_runtime.py",
                "--json-out",
                str(run / "preflight_runtime.json"),
            )
            outputs = (run / "preflight_runtime.json",)
        elif stage == "prepare":
            argv = (
                options.python_executable,
                "scripts/mednla/prepare_items.py",
                "--config",
                str(options.config),
                "--output",
                str(items),
            )
            outputs = (items,)
        elif stage == "probe":
            argv = (
                options.python_executable,
                "scripts/mednla/run_base_probes.py",
                "--config",
                str(options.config),
                "--items",
                str(items),
                "--model",
                options.model,
                "--predictions-out",
                str(predictions),
                "--activations-out",
                str(activations),
                "--manifest-out",
                str(run / "probe_manifest.json"),
            )
            outputs = (predictions, activations)
        elif stage == "check_sglang":
            argv_list = [
                options.python_executable,
                "scripts/mednla/check_sglang_ready.py",
                "--config",
                str(options.config),
                "--model",
                options.model,
            ]
            if options.sglang_url:
                argv_list.extend(["--sglang-url", options.sglang_url])
            if options.auth_token_env:
                argv_list.extend(["--auth-token-env", options.auth_token_env])
            argv = tuple(argv_list)
            outputs = ()
        elif stage == "decode":
            argv_list = [
                options.python_executable,
                "scripts/mednla/run_nla_decode.py",
                "--config",
                str(options.config),
                "--model",
                options.model,
                "--activations",
                str(activations),
                "--out",
                str(decodes),
                "--manifest-out",
                str(run / "decode_manifest.json"),
            ]
            if options.no_critic:
                argv_list.append("--no-critic")
            if options.allow_cjk_warnings:
                argv_list.append("--allow-cjk-warnings")
            if options.sglang_url:
                argv_list.extend(["--sglang-url", options.sglang_url])
            if options.auth_token_env:
                argv_list.extend(["--auth-token-env", options.auth_token_env])
            argv = tuple(argv_list)
            outputs = (decodes,)
        elif stage == "score_heuristic":
            argv = _score_argv(
                options,
                items=items,
                predictions=predictions,
                decodes=decodes,
                out=scores_heuristic,
                scorer="heuristic_v1",
                manifest=run / "score_heuristic_manifest.json",
            )
            outputs = (scores_heuristic,)
        elif stage == "score_judge":
            argv = _score_argv(
                options,
                items=items,
                predictions=predictions,
                decodes=decodes,
                out=scores_judge,
                scorer="medgemma_judge_v1",
                manifest=run / "score_judge_manifest.json",
            )
            outputs = (scores_judge,)
        elif stage == "analysis":
            score_paths = [str(scores_heuristic)]
            if options.include_judge or "score_judge" in options.stages:
                score_paths.append(str(scores_judge))
            argv_list = [
                options.python_executable,
                "scripts/mednla/analyze_results.py",
                "--config",
                str(options.config),
                "--items",
                str(items),
                "--predictions",
                str(predictions),
                "--decodes",
                str(decodes),
                "--scores",
                *score_paths,
                "--out-dir",
                str(run),
                "--manifest-out",
                str(run / "analysis_manifest.json"),
            ]
            if options.quick_analysis:
                argv_list.append("--quick")
            argv = tuple(argv_list)
            outputs = tuple(analysis_outputs)
        elif stage == "validate":
            argv_list = [
                options.python_executable,
                "scripts/mednla/validate_run.py",
                "--config",
                str(options.config),
                "--run-dir",
                str(run),
                "--model",
                options.model,
                "--out",
                str(run / "run_validation.json"),
            ]
            if "analysis" in options.stages:
                argv_list.append("--require-analysis")
            if options.include_judge or "score_judge" in options.stages:
                argv_list.append("--require-judge-score")
            argv = tuple(argv_list)
            outputs = (run / "run_validation.json",)
        else:  # pragma: no cover - parse_stage_list prevents this.
            raise ValueError(f"unsupported stage: {stage}")

        commands.append(StageCommand(stage, argv, outputs, logs / f"{stage}.log"))
    return commands


def stage_outputs_valid(command: StageCommand) -> bool:
    if command.name == "check_sglang":
        return False
    if not command.outputs:
        return False
    if not all(_nonempty(path) for path in command.outputs):
        return False
    if command.name == "validate":
        try:
            data = orjson.loads(command.outputs[0].read_bytes())
        except Exception:
            return False
        return bool(data.get("ok"))
    return True


def format_command(argv: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(argv))


def _score_argv(
    options: PipelineOptions,
    *,
    items: Path,
    predictions: Path,
    decodes: Path,
    out: Path,
    scorer: str,
    manifest: Path,
) -> tuple[str, ...]:
    return (
        options.python_executable,
        "scripts/mednla/score_explanations.py",
        "--config",
        str(options.config),
        "--items",
        str(items),
        "--predictions",
        str(predictions),
        "--decodes",
        str(decodes),
        "--out",
        str(out),
        "--scorer",
        scorer,
        "--manifest-out",
        str(manifest),
    )


def _analysis_outputs(run: Path) -> list[Path]:
    return [
        run / "tables" / "summary_by_model_dataset.csv",
        run / "tables" / "taxonomy_by_model_dataset.csv",
        run / "tables" / "prompt_stability.csv",
        run / "tables" / "rationale_alignment_by_correctness.csv",
        run / "tables" / "failure_cases.jsonl",
        run / "figures_data" / "taxonomy_stack.jsonl",
        run / "figures_data" / "accuracy_vs_aligned.jsonl",
        run / "figures_data" / "reconstruction_by_cell.jsonl",
        run / "_summary.json",
    ]


def _nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


__all__ = [
    "DEFAULT_STAGES",
    "PipelineOptions",
    "STAGES",
    "StageCommand",
    "build_stage_commands",
    "default_run_dir",
    "format_command",
    "load_config",
    "parse_stage_list",
    "stage_outputs_valid",
]
