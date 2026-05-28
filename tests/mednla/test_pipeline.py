"""Tests for MedNLA eval pipeline orchestration."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

import scripts.mednla.run_eval_pipeline as pipeline_cli
from nla.mednla.pipeline import PipelineOptions, build_stage_commands, parse_stage_list


def _config(path: Path) -> Path:
    path.write_text("run_id: test_run\nprompt_variants: [canonical]\n", encoding="utf-8")
    return path


def _args(config: Path, run_dir: Path, *, stages: str | None, dry_run: bool = False, resume: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config),
        model="qwen7b",
        run_dir=str(run_dir),
        stages=stages,
        dry_run=dry_run,
        resume=resume,
        quick_analysis=False,
        sglang_url="http://127.0.0.1:18000",
        no_critic=True,
        allow_cjk_warnings=False,
        score_judge=False,
        auth_token_env=None,
    )


def test_build_stage_commands_use_current_scripts_and_artifacts(tmp_path: Path) -> None:
    options = PipelineOptions(
        config=Path("configs/mednla/pilot_qwen7b_medqa.yaml"),
        model="qwen7b",
        run_dir=tmp_path / "run",
        stages=parse_stage_list("prepare,probe,check_sglang,decode,score_heuristic,analysis,validate"),
        quick_analysis=True,
        sglang_url="http://127.0.0.1:18000",
        no_critic=True,
        python_executable="python",
    )

    commands = build_stage_commands(options)

    assert [command.name for command in commands] == [
        "prepare",
        "probe",
        "check_sglang",
        "decode",
        "score_heuristic",
        "analysis",
        "validate",
    ]
    by_name = {command.name: command for command in commands}
    assert "scripts/mednla/prepare_items.py" in by_name["prepare"].argv
    assert "scripts/mednla/run_base_probes.py" in by_name["probe"].argv
    assert "scripts/mednla/check_sglang_ready.py" in by_name["check_sglang"].argv
    assert "scripts/mednla/run_nla_decode.py" in by_name["decode"].argv
    assert "scripts/mednla/score_explanations.py" in by_name["score_heuristic"].argv
    assert "scripts/mednla/analyze_results.py" in by_name["analysis"].argv
    assert "scripts/mednla/validate_run.py" in by_name["validate"].argv
    assert by_name["prepare"].outputs == (tmp_path / "run" / "items.jsonl",)
    assert tmp_path / "run" / "predictions.jsonl" in by_name["probe"].outputs
    assert tmp_path / "run" / "activations.parquet" in by_name["probe"].outputs
    assert "--quick" in by_name["analysis"].argv
    assert "--require-analysis" in by_name["validate"].argv


def test_stage_selection_preserves_order_and_rejects_unknown() -> None:
    assert parse_stage_list("decode,score_heuristic,validate") == ("decode", "score_heuristic", "validate")
    assert parse_stage_list(None, include_judge=True) == (
        "prepare",
        "score_heuristic",
        "score_judge",
        "analysis",
        "validate",
    )
    with pytest.raises(ValueError, match="unknown stage"):
        parse_stage_list("prepare,nope")
    with pytest.raises(ValueError, match="duplicate"):
        parse_stage_list("prepare,prepare")


def test_dry_run_does_not_execute_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    config = _config(tmp_path / "config.yaml")
    calls: list[tuple[str, ...]] = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(pipeline_cli.subprocess, "run", fake_run)

    rc = pipeline_cli._run(_args(config, tmp_path / "run", stages="prepare,validate", dry_run=True))

    assert rc == 0
    assert calls == []
    out = capsys.readouterr().out
    assert "scripts/mednla/prepare_items.py" in out
    assert "scripts/mednla/validate_run.py" in out


def test_resume_skips_only_valid_existing_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path / "config.yaml")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "items.jsonl").write_text("{}\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(pipeline_cli.subprocess, "run", fake_run)

    rc = pipeline_cli._run(_args(config, run_dir, stages="prepare,score_heuristic", resume=True))

    assert rc == 0
    assert len(calls) == 1
    assert "scripts/mednla/score_explanations.py" in calls[0]


def test_auth_env_name_is_in_commands_but_value_is_not(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SGLANG_TOKEN", "secret-value")
    options = PipelineOptions(
        config=Path("config.yaml"),
        model="qwen7b",
        run_dir=tmp_path / "run",
        stages=("check_sglang", "decode"),
        auth_token_env="SGLANG_TOKEN",
        python_executable="python",
    )

    rendered = "\n".join(" ".join(command.argv) for command in build_stage_commands(options))

    assert "--auth-token-env SGLANG_TOKEN" in rendered
    assert "secret-value" not in rendered


def test_nonzero_stage_exit_stops_later_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _config(tmp_path / "config.yaml")
    calls: list[tuple[str, ...]] = []

    def fake_run(argv, **kwargs):
        calls.append(tuple(argv))
        return subprocess.CompletedProcess(argv, 9)

    monkeypatch.setattr(pipeline_cli.subprocess, "run", fake_run)

    rc = pipeline_cli._run(_args(config, tmp_path / "run", stages="prepare,score_heuristic"))

    assert rc == 9
    assert len(calls) == 1
    assert "scripts/mednla/prepare_items.py" in calls[0]
