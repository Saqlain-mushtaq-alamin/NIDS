"""
scripts/run_cross_dataset.py — Phase 2 experiment driver.

Runs the complete cross-dataset generalization study (PHASE_2 section 2.3-2.4,
FR5 + FR6) and writes every artifact the summary document cites:

* ``results/cross_dataset_<mode>_<model>.csv``      — per-seed full matrix
* ``results/cross_dataset_<mode>_<model>_agg.csv``  — mean ± std per pair
* ``results/cross_dataset_<mode>_<model>_stats.csv``— per-pair significance
* ``results/cross_dataset_<mode>_<model>_report.md``— human-readable report
* ``results/cross_dataset_<mode>_<model>_summary.json`` — machine-readable headline

Both feature modes are run by default (``full`` then ``behavior_only``) because
the central research question is whether stripping identity columns recovers
transfer performance.

Examples::

    python scripts/run_cross_dataset.py
    python scripts/run_cross_dataset.py --modes full behavior_only --models xgb rf
    python scripts/run_cross_dataset.py --seeds 42 1337 2024 --datasets unsw-nb15
    python scripts/run_cross_dataset.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.cross_dataset import CrossDatasetEvaluator, RESULT_COLUMNS
from src.evaluation.statistics import analyze
from src.models.baseline_rf import RFBaseline
from src.models.baseline_xgb import XGBBaseline
from src.utils.logging_setup import setup_logger
from src.utils.pandas_compat import apply_pandas_compat

apply_pandas_compat()

RESULTS_DIR = _PROJECT_ROOT / "results"

DEFAULT_DATASETS = ["cic-ids-2017", "unsw-nb15", "lab-captured"]
DEFAULT_MODES = ["full", "behavior_only"]
DEFAULT_SEEDS = [42, 1337, 2024, 7, 99]

MODEL_FACTORIES = {
    "xgb": lambda seed: XGBBaseline(
        n_estimators=150, max_depth=6, learning_rate=0.05, random_state=seed,
        n_jobs=4,
    ),
    "rf": lambda seed: RFBaseline(
        n_estimators=200, max_depth=12, random_state=seed, n_jobs=4,
    ),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 cross-dataset generalization experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Registry keys to evaluate.")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        choices=["full", "behavior_only", "behavior_extended"],
                        help="Feature modes to compare.")
    parser.add_argument("--models", nargs="+", default=["xgb"],
                        choices=sorted(MODEL_FACTORIES),
                        help="Model families to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Seeds; each is a full pass over the matrix.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--balanced", action="store_true",
                        help="Pass class_weight='balanced' to the estimators.")
    parser.add_argument("--no-normalization", action="store_true",
                        help="Disable the robust scaler.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true",
                        help="One seed, full mode only, XGB (smoke test).")
    return parser.parse_args(argv)


def _make_evaluator(args, model_key: str, mode: str, seed: int) -> CrossDatasetEvaluator:
    factory = MODEL_FACTORIES[model_key]
    return CrossDatasetEvaluator(
        datasets=list(args.datasets),
        model_factory=lambda: factory(seed),
        feature_mode=mode,
        normalization="none" if args.no_normalization else "robust",
        seed=seed,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        model_name=model_key,
        class_weight="balanced" if args.balanced else None,
    )


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return str(value)


def run_configuration(args, model_key: str, mode: str, logger) -> dict:
    """Run every seed for one (model, feature_mode) combination."""
    frames = []
    for seed in args.seeds:
        logger.info(
            "=== %s | mode=%s | seed=%d ===", model_key, mode, seed
        )
        evaluator = _make_evaluator(args, model_key, mode, seed)
        matrix = evaluator.run_full_matrix()
        frames.append(matrix)
        for _, row in matrix.iterrows():
            logger.info(
                "  %-14s -> %-14s  ACC=%.4f  macroF1=%.4f  cov=%.0f%%  feat=%d",
                row["train_dataset"], row["test_dataset"], row["accuracy"],
                row["macro_f1"], 100 * row["test_coverage"], row["n_train_features"],
            )

    results = pd.concat(frames, ignore_index=True)[RESULT_COLUMNS]
    stats_bundle = analyze(results)

    stem = f"cross_dataset_{mode}_{model_key}"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_path = out / f"{stem}.csv"
    agg_path = out / f"{stem}_agg.csv"
    stats_path = out / f"{stem}_stats.csv"
    report_path = out / f"{stem}_report.md"
    json_path = out / f"{stem}_summary.json"

    results.to_csv(results_path, index=False)
    stats_bundle["aggregate"].to_csv(agg_path, index=False)
    stats_bundle["comparisons"].to_csv(stats_path, index=False)

    evaluator = _make_evaluator(args, model_key, mode, args.seeds[0])
    notes = [
        f"Seeds: {', '.join(str(s) for s in args.seeds)} "
        f"({len(args.seeds)} passes; every cell repeated per seed).",
        "`macro-F1` is the headline metric because the unified taxonomy is "
        "imbalanced; weighted F1 is retained in the CSV as `f1`.",
        "ΔF1 for a pair `A→B` is measured against the diagonal cell `A→A`, i.e. "
        "the same feature space and model family evaluated on held-out data from "
        "the training domain.",
        "Test rows whose label never appears in the training set are excluded "
        "from scoring and reported through `test_coverage`; predictions that "
        "fall outside the trained label set are counted as errors.",
    ]
    evaluator.generate_report(results, report_path, extra_notes=notes)

    in_mean = stats_bundle["in_domain_mean"]
    cross_mean = stats_bundle["cross_domain_mean"]
    overall_t = stats_bundle["overall_ttest"]
    overall_w = stats_bundle["overall_wilcoxon"]
    comparisons = stats_bundle["comparisons"]

    summary = {
        "model": model_key,
        "feature_mode": mode,
        "seeds": list(args.seeds),
        "datasets": list(args.datasets),
        "n_cells": int(len(results) / max(len(args.seeds), 1)),
        "in_domain_macro_f1_mean": in_mean,
        "cross_domain_macro_f1_mean": cross_mean,
        "mean_f1_delta": in_mean - cross_mean,
        "overall_delta_ci95": list(stats_bundle["overall_delta_ci"]),
        "overall_ttest_p": None if overall_t is None else overall_t.p_value,
        "overall_wilcoxon_p": None if overall_w is None else overall_w.p_value,
        "cross_worse_pairs": stats_bundle["cross_worse_pairs"],
        "total_pairs": stats_bundle["total_pairs"],
        "significant_pairs": stats_bundle["significant_pairs"],
        "mean_test_coverage": float(results["test_coverage"].mean()),
        "per_pair": json.loads(comparisons.to_json(orient="records")),
        "artifacts": {
            "matrix": str(results_path),
            "aggregate": str(agg_path),
            "comparisons": str(stats_path),
            "report": str(report_path),
        },
    }
    json_path.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )

    logger.info("--- %s | %s headline ---", model_key, mode)
    logger.info("  in-domain  macro-F1 : %.4f", in_mean)
    logger.info("  cross-domain macro-F1: %.4f", cross_mean)
    logger.info("  mean dF1 (in-cross) : %+.4f", in_mean - cross_mean)
    if overall_t is not None:
        logger.info("  %s", overall_t.format())
    if overall_w is not None:
        logger.info("  %s", overall_w.format())
    logger.info("  cross worse in %d/%d pairs (%d significant)",
                stats_bundle["cross_worse_pairs"], stats_bundle["total_pairs"],
                stats_bundle["significant_pairs"])
    logger.info("  report: %s", report_path)

    return summary


def compare_modes(summaries: list[dict], logger) -> None:
    """
    Log the full-vs-behavior comparison, the mitigation check of PHASE_2 G2.
    """
    by_mode: dict[str, dict[str, dict]] = {}
    for summary in summaries:
        by_mode.setdefault(summary["model"], {})[summary["feature_mode"]] = summary

    for model, modes in by_mode.items():
        if "full" not in modes or "behavior_only" not in modes:
            continue
        full = modes["full"]
        behavior = modes["behavior_only"]
        gain = behavior["cross_domain_macro_f1_mean"] - full["cross_domain_macro_f1_mean"]
        logger.info(
            "=== mitigation check (%s): behavior_only cross-domain macro-F1 "
            "%.4f vs full %.4f -> %+.4f (%s)",
            model,
            behavior["cross_domain_macro_f1_mean"],
            full["cross_domain_macro_f1_mean"],
            gain,
            "behavior-only recovers transfer" if gain > 0 else "no recovery",
        )


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.quick:
        args.seeds = [args.seeds[0]]
        args.modes = ["full"]
        args.models = ["xgb"]

    logger = setup_logger("NIDS-Phase2-CrossDataset", level=20)
    logger.info("datasets=%s modes=%s models=%s seeds=%s",
                args.datasets, args.modes, args.models, args.seeds)

    summaries = []
    for model_key in args.models:
        for mode in args.modes:
            summaries.append(run_configuration(args, model_key, mode, logger))

    compare_modes(summaries, logger)
    logger.info("done — %d configuration(s), artifacts in %s",
                len(summaries), Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
