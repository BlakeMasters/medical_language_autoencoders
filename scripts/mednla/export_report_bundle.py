"""Export a writeup-ready MedNLA report bundle and manual audit queue."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nla.mednla.reporting import export_report_bundle

logger = logging.getLogger("nla.mednla.export_report_bundle")


def _run(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "report"
    score_paths = [Path(path) for path in args.scores] if args.scores else None
    index = export_report_bundle(
        run_dir=run_dir,
        out_dir=out_dir,
        items_path=Path(args.items) if args.items else None,
        predictions_path=Path(args.predictions) if args.predictions else None,
        decodes_path=Path(args.decodes) if args.decodes else None,
        score_paths=score_paths,
        per_cell=args.per_cell,
        max_cases=args.max_cases,
        seed=args.seed,
        include_raw_av_text=args.include_raw_av_text,
    )
    logger.info(
        "wrote_report_bundle out_dir=%s audit_rows=%d scorers=%s",
        out_dir,
        index["counts"]["audit_rows"],
        ",".join(index["scorers"]),
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Export MedNLA report bundle and manual audit queue.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing T6 tables/figures.")
    parser.add_argument("--items", default=None, help="Optional explicit items JSONL path.")
    parser.add_argument("--predictions", default=None, help="Optional explicit predictions JSONL path.")
    parser.add_argument("--decodes", default=None, help="Optional explicit decodes JSONL path.")
    parser.add_argument("--scores", nargs="+", default=None, help="Optional explicit score JSONL path(s).")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run-dir>/report.")
    parser.add_argument("--per-cell", type=int, default=10, help="Maximum audit examples per taxonomy cell.")
    parser.add_argument("--max-cases", type=int, default=80, help="Maximum total audit rows.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic audit sampling seed.")
    parser.add_argument(
        "--include-raw-av-text",
        action="store_true",
        help="Include raw AV text in audit rows. Omitted by default to keep files small.",
    )
    args = parser.parse_args()
    try:
        return _run(args)
    except (OSError, ValueError) as exc:
        logger.error("report_bundle_error error=%s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
