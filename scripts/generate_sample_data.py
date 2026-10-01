"""
scripts/generate_sample_data.py — Regenerate the CIC-IDS2017 / UNSW-NB15
stand-in datasets with realistic native column schemas.

Usage:
    python scripts/generate_sample_data.py                # default 200 flows/class
    python scripts/generate_sample_data.py --per-class 400 --seed 7
    python scripts/generate_sample_data.py --no-backup

Existing files are moved to ``data/raw/_original_samples`` first, so the original
Phase 1 placeholder samples stay recoverable.

These are SYNTHETIC stand-ins, not the real benchmarks. See the module docstring
of ``src/data/sample_datasets.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_setup import setup_logger
from src.data.sample_datasets import generate_all


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=200,
                        help="Flows generated per class per dataset (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--no-backup", action="store_true",
                        help="Overwrite existing stand-ins without backing them up")
    args = parser.parse_args()

    logger = setup_logger("NIDS-Phase2-GenerateSamples")
    logger.info("=" * 68)
    logger.info("Generating synthetic stand-ins with native column schemas")
    logger.info("=" * 68)
    logger.warning(
        "These are NOT CIC-IDS2017 / UNSW-NB15. They exist so the Phase 2 "
        "pipeline can be validated end to end."
    )

    summary = generate_all(per_class=args.per_class, seed=args.seed,
                           backup=not args.no_backup)

    logger.info(f"CIC-IDS2017 stand-in : {summary['cic']['path']} "
                f"({summary['cic']['rows']:,} rows)")
    logger.info(f"UNSW-NB15 stand-in    : {summary['unsw']['path']} "
                f"({summary['unsw']['rows']:,} rows)")
    if not args.no_backup:
        logger.info(f"Previous samples moved to: {summary['backup_dir']}")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
