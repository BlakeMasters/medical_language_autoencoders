"""Prepare normalized MedNLA items JSONL from a YAML config."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import orjson
import yaml

from nla.mednla.datasets import load_items
from nla.mednla.schema import to_dict

logger = logging.getLogger("nla.mednla.prepare_items")

def _write_items_jsonl(items: list, output_path: Path) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("wb") as handle:
        for item in items:
            line = orjson.dumps(to_dict(item)) + b"\n"
            handle.write(line)
    os.replace(tmp_path, output_path)

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Prepare normalized MedNLA items JSONL.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)

    try:
        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except OSError as exc:
        logger.error("failed to read config %s: %s", config_path, exc)
        return 2
    except yaml.YAMLError as exc:
        logger.error("invalid YAML in %s: %s", config_path, exc)
        return 2

    if not isinstance(cfg, dict):
        logger.error("config root must be a mapping")
        return 2

    seed = args.seed if args.seed is not None else cfg.get("seed")
    if seed is None:
        logger.error("seed must be provided via --seed or config")
        return 2

    try:
        items = load_items(cfg, seed=int(seed))
    except ValueError as exc:
        logger.error("validation error: %s", exc)
        return 3

    try:
        _write_items_jsonl(items, output_path)
    except OSError as exc:
        logger.error("failed to write %s: %s", output_path, exc)
        return 2

    counts = Counter(item.dataset for item in items)
    counts_repr = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    logger.info("wrote %d items to %s (%s)", len(items), output_path, counts_repr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
