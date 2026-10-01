"""
scripts/generate_lab_capture.py — Generate and extract the home-lab dataset.

Usage:
    python scripts/generate_lab_capture.py
    python scripts/generate_lab_capture.py --flows 1200 --seed 7
    python scripts/generate_lab_capture.py --mode both        # pcap + conn.log
    python scripts/generate_lab_capture.py --extract-only     # no simulation

Steps:

1. Simulate one capture artefact per row of ``lab/capture_logs/session_labels.csv``
   (plan section 2.1): ``*.pcap`` and/or Zeek ``conn.log``.
2. Extract flow features with ``src/data/feature_extractor.py`` (plan section 2.2).
3. Label flows by capture-session time window.
4. Write ``data/lab_captures/lab_flows.csv`` in the canonical feature schema.

To use REAL captures instead, drop your own ``*.pcap`` / ``conn.log`` into
``data/lab_captures`` and run with ``--extract-only``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_setup import setup_logger
from src.data.lab_simulator import build_lab_dataset
from src.data.feature_extractor import extract_lab_features

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DIR = _PROJECT_ROOT / "data" / "lab_captures"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_DIR,
                        help="Capture directory (default: data/lab_captures)")
    parser.add_argument("--flows", type=int, default=900,
                        help="Flows simulated per capture session (default: 900)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument("--mode", choices=["pcap", "zeek", "both"], default="pcap",
                        help="Artefact type to simulate (default: pcap)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip simulation; only extract features from existing captures")
    parser.add_argument("--csv-name", default="lab_flows.csv",
                        help="Extracted dataset filename (default: lab_flows.csv)")
    args = parser.parse_args()

    logger = setup_logger("NIDS-Phase2-LabCapture")
    logger.info("=" * 68)
    logger.info("Home-lab capture dataset")
    logger.info("=" * 68)

    output_dir = Path(args.output_dir)
    csv_path = output_dir / args.csv_name

    if args.extract_only:
        logger.info("Extraction-only mode: reading existing captures.")
        frame = extract_lab_features(output_dir, save_to=csv_path)
        source = frame.attrs.get("source", "unknown")
        logger.info(f"Source: {source}")
    else:
        logger.warning(
            "Simulating capture artefacts. They are VALID pcap / Zeek log files "
            "containing SYNTHETIC packets, not recordings from a physical lab. "
            "See data/lab_captures/PROVENANCE.md."
        )
        summary, frame = build_lab_dataset(
            output_dir, flows_per_session=args.flows, seed=args.seed,
            csv_name=args.csv_name, mode=args.mode,
        )
        logger.info(f"Sessions simulated : {summary['sessions']}")
        logger.info(f"Flows per session  : {summary['flows_per_session']}")
        logger.info(f"Artefacts written  : {len(summary['artifacts'])}")
        logger.info(f"Dataset written    : {summary['csv']}")

    logger.info(f"\nRows extracted : {len(frame):,}")
    logger.info(f"Columns        : {len(frame.columns)}")
    logger.info("\nLabel distribution:")
    for label, count in frame["label"].value_counts().items():
        logger.info(f"  {label:<14s} {count:>7,d}")

    unlabelled = int((frame["label"] == "Other").sum())
    if unlabelled:
        logger.warning(
            f"{unlabelled:,} flows fell outside every session window and were "
            "labelled 'Other'."
        )

    logger.info("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
