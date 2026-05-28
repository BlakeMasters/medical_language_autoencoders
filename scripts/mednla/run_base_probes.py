"""Run base-model MedNLA probes and write predictions plus activations."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import orjson
import torch
import yaml

from nla.mednla.probe import (
    ActivationParquetWriter,
    BaseModelProbe,
    ProbeValidationError,
    assert_activation_parquet,
    load_nla_sidecar_d_model,
)
from nla.mednla.schema import MedItem, VariantName, from_dict, to_dict

logger = logging.getLogger("nla.mednla.run_base_probes")


class ModelLoadError(RuntimeError):
    pass


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def _load_items(path: Path, *, limit: int | None) -> list[MedItem]:
    items: list[MedItem] = []
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(from_dict(MedItem, orjson.loads(line)))
            if limit is not None and len(items) >= limit:
                break
    return items


def _resolve_model(cfg: dict[str, Any], short_name: str) -> dict[str, Any]:
    matches = [
        model
        for model in cfg.get("models", [])
        if isinstance(model, dict) and model.get("short_name") == short_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one model with short_name={short_name!r}, got {len(matches)}")
    return matches[0]


def _resolve_variants(cfg: dict[str, Any], variants_arg: str | None) -> list[VariantName]:
    configured = cfg.get("prompt_variants", [])
    if not isinstance(configured, list) or not configured:
        raise ValueError("config prompt_variants must be a non-empty list")
    variants = variants_arg.split(",") if variants_arg else configured
    allowed = set(configured)
    resolved: list[VariantName] = []
    for raw_variant in variants:
        variant = raw_variant.strip()
        if variant not in allowed:
            raise ValueError(f"variant {variant!r} not in configured variants {sorted(allowed)!r}")
        if variant not in {"canonical", "option_shuffle", "compact"}:
            raise ValueError(f"unsupported variant {variant!r}")
        resolved.append(variant)  # type: ignore[arg-type]
    return resolved


def _summary_path(predictions_out: Path) -> Path:
    return predictions_out.parent / "_summary.json"


def _partial_path(predictions_out: Path) -> Path:
    return predictions_out.with_suffix(predictions_out.suffix + ".partial")


def _validate_sidecar_d_model(model_cfg: dict[str, Any], d_model: int) -> None:
    actor_path = model_cfg.get("nla_actor")
    if not actor_path:
        raise ValueError("model config missing nla_actor")
    sidecar_d_model = load_nla_sidecar_d_model(str(actor_path))
    if sidecar_d_model != d_model:
        raise ProbeValidationError(
            f"d_model mismatch: sidecar={sidecar_d_model} config={d_model}"
        )


def _write_partial_marker(path: Path, exc: BaseException) -> None:
    payload = {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def _run(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config_path = Path(args.config)
    items_path = Path(args.items)
    predictions_out = Path(args.predictions_out)
    activations_out = Path(args.activations_out)
    predictions_tmp = predictions_out.with_suffix(predictions_out.suffix + ".tmp")
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    activations_out.parent.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    model_cfg = _resolve_model(cfg, args.model)
    variants = _resolve_variants(cfg, args.variants)
    items = _load_items(items_path, limit=args.limit)
    generation = cfg.get("generation", {})
    max_new_tokens = int(generation.get("max_new_tokens", 8))

    d_model = int(model_cfg["d_model"])
    _validate_sidecar_d_model(model_cfg, d_model)

    try:
        probe = BaseModelProbe(
            str(model_cfg["model_id"]),
            int(model_cfg["layer_index"]),
            d_model,
            dtype=args.dtype,
            device=args.device,
            seed=int(cfg.get("seed", 42)),
        )
    except ProbeValidationError:
        raise
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        raise ModelLoadError(str(exc)) from exc

    n_predictions = 0
    n_correct = 0
    parquet_writer: ActivationParquetWriter | None = None
    try:
        parquet_writer = ActivationParquetWriter(activations_out, d_model)
        with (
            predictions_tmp.open("wb") as pred_handle,
            probe,
        ):
            for item in items:
                for variant in variants:
                    prediction, vector = probe.probe_one(
                        item,
                        variant,
                        args.variant_seed,
                        max_new_tokens=max_new_tokens,
                        model_short_name=str(model_cfg["short_name"]),
                    )
                    pred_handle.write(orjson.dumps(to_dict(prediction)) + b"\n")
                    parquet_writer.append(prediction, vector)
                    n_predictions += 1
                    n_correct += int(prediction.correct)
                    if n_predictions % 32 == 0:
                        pred_handle.flush()
            pred_handle.flush()
        parquet_writer.commit()
        os.replace(predictions_tmp, predictions_out)
    except torch.cuda.OutOfMemoryError:
        raise
    except ProbeValidationError:
        if parquet_writer is not None:
            parquet_writer.close()
        raise
    except BaseException as exc:
        if parquet_writer is not None:
            parquet_writer.close()
        _write_partial_marker(_partial_path(predictions_out), exc)
        logger.error("partial_probe_run error_type=%s error=%s", exc.__class__.__name__, exc)
        return 5

    assert_activation_parquet(activations_out, d_model=d_model, expected_rows=n_predictions)
    elapsed_sec = time.perf_counter() - started
    summary = {
        "n_predictions": n_predictions,
        "n_correct": n_correct,
        "elapsed_sec": elapsed_sec,
        "d_model": d_model,
        "model_short_name": model_cfg["short_name"],
        "variants": variants,
        "variant_seed": args.variant_seed,
    }
    _summary_path(predictions_out).write_bytes(orjson.dumps(summary, option=orjson.OPT_INDENT_2))
    logger.info(
        "wrote_probes predictions=%d correct=%d activations=%s predictions_out=%s elapsed_sec=%.2f",
        n_predictions,
        n_correct,
        activations_out,
        predictions_out,
        elapsed_sec,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Run MedNLA base-model answer/activation probes.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--items", required=True, help="Prepared items JSONL.")
    parser.add_argument("--model", required=True, help="Model short_name from config.")
    parser.add_argument("--predictions-out", required=True, help="Output predictions JSONL.")
    parser.add_argument("--activations-out", required=True, help="Output activations parquet.")
    parser.add_argument("--variants", default=None, help="Comma-separated variant names. Default: config order.")
    parser.add_argument("--variant-seed", type=int, default=0, help="Prompt variant seed.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of input items.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda, cuda:0, cpu.")
    args = parser.parse_args()

    try:
        return _run(args)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        logger.error("config_or_io_error error=%s", exc)
        return 2
    except torch.cuda.OutOfMemoryError:
        raise
    except ProbeValidationError as exc:
        logger.error("shape_or_probe_error error=%s", exc)
        return 4
    except ModelLoadError as exc:
        logger.error("model_load_error error=%s", exc)
        return 3
    except Exception as exc:
        logger.error("model_load_or_runtime_error error=%s", exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
