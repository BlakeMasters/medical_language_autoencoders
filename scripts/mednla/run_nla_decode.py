"""Run MedNLA AV/AR decoding over a T3 activations parquet."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.decode import (
    WARNING_CJK_INJECTION_FAILURE,
    DecodeSchemaError,
    MedNLADecoder,
)
from nla.mednla.schema import to_dict

logger = logging.getLogger("nla.mednla.run_nla_decode")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _resolve_model(cfg: dict[str, Any], short_name: str) -> dict[str, Any]:
    matches = [
        model
        for model in cfg.get("models", [])
        if isinstance(model, dict) and model.get("short_name") == short_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one model with short_name={short_name!r}, got {len(matches)}")
    return matches[0]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _run(args: argparse.Namespace) -> int:
    cfg = _load_yaml(Path(args.config))
    model_cfg = _resolve_model(cfg, args.model)
    decode_cfg = cfg.get("nla_decode", {})
    if not isinstance(decode_cfg, dict):
        raise ValueError("config nla_decode must be a mapping")

    sglang_url = str(decode_cfg.get("sglang_url", "http://localhost:30000"))
    temperature = float(decode_cfg.get("temperature", 0.7))
    max_new_tokens = int(decode_cfg.get("max_new_tokens", 200))
    batch_size = args.batch_size if args.batch_size is not None else int(decode_cfg.get("batch_size", 16))
    ar_path = None if args.no_critic else model_cfg.get("nla_critic")

    out_path = Path(args.out)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        decoder = MedNLADecoder(
            av_path=str(model_cfg["nla_actor"]),
            ar_path=str(ar_path) if ar_path else None,
            sglang_url=sglang_url,
            ar_device=args.ar_device,
            ar_dtype=args.ar_dtype,
        )
    except Exception as exc:
        logger.error("decoder_load_error error=%s", exc)
        return 3

    total = 0
    parse_ok = 0
    warning_counts: Counter[str] = Counter()
    reconstruction_cosines: list[float] = []
    try:
        with decoder, tmp_path.open("wb") as handle:
            for record in decoder.decode_batch(
                args.activations,
                batch_size=batch_size,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                limit=args.limit,
            ):
                handle.write(orjson.dumps(to_dict(record)) + b"\n")
                total += 1
                parse_ok += int(record.parse_ok)
                warning_counts.update(record.decode_warnings)
                if record.reconstruction_cos is not None:
                    reconstruction_cosines.append(record.reconstruction_cos)
                if total % 32 == 0:
                    handle.flush()

                if WARNING_CJK_INJECTION_FAILURE in record.decode_warnings:
                    if args.allow_cjk_warnings:
                        logger.warning(
                            "cjk_warning prediction_id=%s raw_prefix=%r",
                            record.prediction_id,
                            record.raw_av_text[:200],
                        )
                    else:
                        handle.flush()
                        logger.error(
                            "cjk_injection_failure prediction_id=%s raw_prefix=%r",
                            record.prediction_id,
                            record.raw_av_text[:200],
                        )
                        return 6
            handle.flush()
        os.replace(tmp_path, out_path)
    except DecodeSchemaError as exc:
        logger.error("decode_schema_error error=%s", exc)
        return 4

    mean_cos = _mean(reconstruction_cosines)
    parse_ok_rate = (parse_ok / total) if total else 0.0
    logger.info(
        "wrote_decodes total=%d parse_ok_rate=%.3f mean_reconstruction_cos=%s warnings=%s out=%s",
        total,
        parse_ok_rate,
        f"{mean_cos:.4f}" if mean_cos is not None else "none",
        dict(sorted(warning_counts.items())),
        out_path,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Decode MedNLA activation vectors with released AV/AR checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="Model short_name from config.")
    parser.add_argument("--activations", required=True, help="T3 activations parquet.")
    parser.add_argument("--out", required=True, help="Output decodes JSONL.")
    parser.add_argument("--no-critic", action="store_true", help="Skip AR reconstruction scoring.")
    parser.add_argument("--allow-cjk-warnings", action="store_true", help="Continue after CJK injection warnings.")
    parser.add_argument("--limit", type=int, default=None, help="Limit decoded rows.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config nla_decode.batch_size.")
    parser.add_argument("--ar-device", default="cuda", help="Torch device for NLACritic.")
    parser.add_argument("--ar-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    try:
        return _run(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("config_or_io_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
