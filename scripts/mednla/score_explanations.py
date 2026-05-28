"""Score MedNLA decoded explanations into ScoreRecord JSONL."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.run_manifest import build_run_manifest, write_run_manifest
from nla.mednla.schema import DecodeRecord, MedItem, Prediction, ScoreRecord, from_dict, to_dict
from nla.mednla.scoring import HeuristicScorer, MedGemmaJudge

logger = logging.getLogger("nla.mednla.score_explanations")

_HF_REMEDIATION = "huggingface-cli login + accept https://huggingface.co/google/medgemma-4b-it terms"


class ModelLoadError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _load_jsonl(path: Path, cls: type[Any]) -> list[Any]:
    rows: list[Any] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(from_dict(cls, orjson.loads(line)))
    return rows


def _index_unique(rows: list[Any], key_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        key = str(getattr(row, key_name))
        if key in out:
            raise ValueError(f"duplicate {key_name}: {key}")
        out[key] = row
    return out


def _join_rows(
    items: list[MedItem],
    predictions: list[Prediction],
    decodes: list[DecodeRecord],
    *,
    limit: int | None,
) -> list[tuple[MedItem, Prediction, DecodeRecord]]:
    items_by_id = _index_unique(items, "item_id")
    _index_unique(predictions, "prediction_id")
    decodes_by_pred = _index_unique(decodes, "prediction_id")

    joined: list[tuple[MedItem, Prediction, DecodeRecord]] = []
    for pred in predictions:
        item = items_by_id.get(pred.item_id)
        decode = decodes_by_pred.get(pred.prediction_id)
        if item is None or decode is None:
            raise ValueError(
                f"missing join: prediction_id={pred.prediction_id} item={item is not None} decode={decode is not None}"
            )
        joined.append((item, pred, decode))
    return joined[:limit] if limit is not None else joined


def _summary_path(out_path: Path) -> Path:
    return out_path.parent / "_summary.json"


def _partial_path(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".partial")


def _mean_medical_relevance(scores: list[ScoreRecord]) -> float | None:
    if not scores:
        return None
    return float(np.mean([score.medical_relevance for score in scores]))


def _cuda_mem(label: str) -> None:
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    logger.info("cuda_mem label=%s free=%d total=%d", label, free, total)


def _make_scorer(args: argparse.Namespace, cfg: dict[str, Any]) -> Any:
    if args.scorer == HeuristicScorer.name:
        return HeuristicScorer()
    if args.scorer != MedGemmaJudge.name:
        raise ValueError(f"unknown scorer: {args.scorer!r}")

    scoring_cfg = cfg.get("scoring", {})
    if not isinstance(scoring_cfg, dict):
        raise ValueError("config scoring must be a mapping")
    judge_model = args.judge_model or str(scoring_cfg.get("judge_model", "google/medgemma-4b-it"))
    judge_loader = args.judge_loader or str(scoring_cfg.get("judge_loader", "auto"))
    max_new_tokens = int(scoring_cfg.get("judge_max_new_tokens", 400))
    try:
        return MedGemmaJudge(
            judge_model,
            device=args.device,
            dtype=args.dtype,
            max_new_tokens=max_new_tokens,
            loader=judge_loader,
        )
    except Exception as exc:
        raise ModelLoadError(f"{exc}. {_HF_REMEDIATION}") from exc


def _write_summary(out_path: Path, scorer_name: str, scores: list[ScoreRecord], elapsed_sec: float) -> dict[str, Any]:
    parse_errors = sum(1 for score in scores if score.scorer_notes.startswith("judge_parse_error:"))
    taxonomy_counts = Counter(score.taxonomy_cell for score in scores)
    summary = {
        "scorer": scorer_name,
        "n_records": len(scores),
        "n_judge_parse_errors": parse_errors,
        "mean_medical_relevance": _mean_medical_relevance(scores),
        "taxonomy_cell_counts": dict(sorted(taxonomy_counts.items())),
        "elapsed_sec": elapsed_sec,
        "out": str(out_path),
    }
    _summary_path(out_path).write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    return summary


def _run(args: argparse.Namespace) -> int:
    started_at_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    config_path = Path(args.config)
    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    joined = _join_rows(
        _load_jsonl(Path(args.items), MedItem),
        _load_jsonl(Path(args.predictions), Prediction),
        _load_jsonl(Path(args.decodes), DecodeRecord),
        limit=args.limit,
    )
    model_short_name = joined[0][1].model_short_name if joined else "unknown"
    _cuda_mem("start")
    scorer = _make_scorer(args, cfg)

    scores: list[ScoreRecord] = []
    manager = scorer if hasattr(scorer, "__enter__") else nullcontext(scorer)
    try:
        with manager as active_scorer, tmp_path.open("wb") as handle:
            for item, pred, decode in joined:
                score = active_scorer.score(item, pred, decode)
                scores.append(score)
                handle.write(orjson.dumps(to_dict(score)) + b"\n")
                if len(scores) % 16 == 0:
                    handle.flush()
            handle.flush()
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            os.replace(tmp_path, _partial_path(out_path))
        raise
    finally:
        close = getattr(scorer, "close", None)
        if callable(close):
            close()
        _cuda_mem("end")

    elapsed_sec = time.perf_counter() - started
    summary = _write_summary(out_path, args.scorer, scores, elapsed_sec)
    if args.manifest_out:
        ended_at_utc = datetime.now(timezone.utc).isoformat()
        manifest = build_run_manifest(
            stage="mednla_score",
            started_at_utc=started_at_utc,
            ended_at_utc=ended_at_utc,
            duration_sec=elapsed_sec,
            config_path=config_path,
            model_short_name=model_short_name,
            args=vars(args),
            outputs={"scores": str(out_path), "summary": str(_summary_path(out_path))},
            extra=summary,
        )
        write_run_manifest(args.manifest_out, manifest)

    parse_error_rate = (summary["n_judge_parse_errors"] / summary["n_records"]) if summary["n_records"] else 0.0
    logger.info(
        "wrote_scores scorer=%s n=%d parse_errors=%d out=%s",
        args.scorer,
        summary["n_records"],
        summary["n_judge_parse_errors"],
        out_path,
    )
    if args.scorer == MedGemmaJudge.name and parse_error_rate >= 0.25:
        logger.error("judge_parse_error_rate_too_high rate=%.3f", parse_error_rate)
        return 7
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Score MedNLA decoded explanations.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--items", required=True, help="Prepared MedNLA items JSONL.")
    parser.add_argument("--predictions", required=True, help="T3 predictions JSONL.")
    parser.add_argument("--decodes", required=True, help="T4 decodes JSONL.")
    parser.add_argument("--out", required=True, help="Output scores JSONL.")
    parser.add_argument("--scorer", default="heuristic_v1", choices=["heuristic_v1", "medgemma_judge_v1"])
    parser.add_argument("--judge-model", default=None, help="Override config scoring.judge_model.")
    parser.add_argument("--judge-loader", default=None, choices=["auto", "image_text", "causal_lm"])
    parser.add_argument("--device", default="cuda", help="Torch device or device_map for MedGemma judge.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--limit", type=int, default=None, help="Limit joined rows after validation.")
    parser.add_argument("--manifest-out", default=None, help="Optional output run manifest JSON.")
    args = parser.parse_args()

    try:
        return _run(args)
    except ModelLoadError as exc:
        logger.error("model_load_error error=%s", exc)
        return 3
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("config_or_io_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
