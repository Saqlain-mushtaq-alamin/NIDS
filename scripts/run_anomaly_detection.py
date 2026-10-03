"""
scripts/run_anomaly_detection.py — Phase 3 autoencoder experiment driver.

Trains the flow autoencoder on benign-only traffic, calibrates its threshold on
held-out benign validation flows, and evaluates it against attack traffic and
against a second dataset.

Two rules the script enforces rather than leaving to the caller
-------------------------------------------------------------
1. **The autoencoder never sees attacks during fitting.** Fitting on a mixed set
   teaches the network to reconstruct attacks too, which silently destroys its
   ability to score them. ``fit_from_frame`` filters to benign rows first.
2. **The threshold is calibrated on benign validation flows only**, for the same
   reason: a threshold fitted on data containing the attacks it must catch is
   biased upward.

Examples::

    python scripts/run_anomaly_detection.py
    python scripts/run_anomaly_detection.py --dataset lab_flows --latent-dim 16
    python scripts/run_anomaly_detection.py --quick
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

from src.data.preprocessor import Preprocessor
from src.models.autoencoder import AnomalyDetector


def _load_flows(dataset: str) -> pd.DataFrame:
    """Load a labelled flow frame from the Phase 2 sample datasets."""
    candidates = [
        _PROJECT_ROOT / "data" / f"{dataset}.csv",
        _PROJECT_ROOT / "data" / "lab_captures" / f"{dataset}.csv",
        _PROJECT_ROOT / "data" / "raw" / f"{dataset}.csv",
    ]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path)
            if "label" in frame.columns:
                return frame
    raise FileNotFoundError(
        f"No labelled CSV for dataset '{dataset}'. Looked in: "
        + ", ".join(str(p) for p in candidates)
    )


def run(dataset: str = "lab_flows", latent_dim: int = 16, epochs: int = 100,
        percentile: float = 99.0, seed: int = 42,
        test_ratio: float = 0.2, val_ratio: float = 0.1) -> dict:
    """Train, calibrate and evaluate the autoencoder for one seed."""
    frame = _load_flows(dataset)

    # The scaler is fitted on the training rows only; val/test are transformed
    # with it, so no test statistic leaks into training.
    preprocessor = Preprocessor(drop_ip_port=True, normalization="robust")
    splits = preprocessor.fit_and_split(frame, test_ratio=test_ratio,
                                        val_ratio=val_ratio, seed=seed)

    train_x, train_y = splits["train"]["X"], splits["train"]["y"]
    val_x, val_y = splits["val"]["X"], splits["val"]["y"]
    test_x, test_y = splits["test"]["X"], splits["test"]["y"]

    # Rule 1: the autoencoder is fitted on benign traffic ONLY. Training on a
    # mixed set teaches it to reconstruct attacks too, which destroys its ability
    # to score them as anomalous.
    benign_train = np.asarray([str(v).strip().lower() == "benign"
                               for v in train_y])
    if not benign_train.any():
        raise ValueError("No benign rows in the training split")

    detector = AnomalyDetector(latent_dim=latent_dim, epochs=epochs,
                               percentile=percentile, seed=seed)
    detector.fit(train_x[benign_train],
                 X_val=val_x, y_val=val_y)

    evaluation = detector.evaluate(test_x, test_y, dataset=dataset)
    per_feature = detector.per_feature_error(test_x)
    worst = pd.DataFrame({
        "feature": list(preprocessor.get_feature_names()),
        "mean_error": per_feature.mean(axis=0),
    }).sort_values("mean_error", ascending=False)

    predictions = detector.predict(test_x)
    benign_test = np.asarray([str(v).strip().lower() == "benign" for v in test_y])

    return {
        "seed": seed,
        "dataset": dataset,
        "n_features": int(preprocessor.n_features_),
        "latent_dim": latent_dim,
        "threshold": float(detector.threshold),
        "percentile": percentile,
        "n_train_benign": int(benign_train.sum()),
        "n_train_attack_excluded": int((~benign_train).sum()),
        "n_val": int(len(val_y)),
        "n_test": int(len(test_y)),
        "n_test_attack": int((~benign_test).sum()),
        "metrics": evaluation.metrics,
        "per_class_detection": evaluation.per_class_detection,
        "error_summary": evaluation.error_summary,
        "notes": evaluation.notes,
        "worst_features": worst.head(10).to_dict("records"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="lab_flows")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--output-dir", type=Path,
                        default=_PROJECT_ROOT / "results" / "phase3" / "autoencoder")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    seeds = [42] if args.quick else args.seeds
    epochs = 20 if args.quick else args.epochs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run(dataset=args.dataset, latent_dim=args.latent_dim,
                   epochs=epochs, percentile=args.percentile, seed=seed)
               for seed in seeds]
    primary = results[0]

    def _mean(key, field):
        return float(np.mean([r[key][field] for r in results
                              if isinstance(r[key], dict) and field in r[key]]))

    summary = {
        "dataset": args.dataset,
        "seeds": seeds,
        "n_features": primary["n_features"],
        "threshold_mean": float(np.mean([r["threshold"] for r in results])),
        "detection_rate_mean": _mean("metrics", "detection_rate"),
        "false_alarm_rate_mean": _mean("metrics", "false_alarm_rate"),
        "roc_auc_mean": _mean("metrics", "roc_auc"),
        "average_precision_mean": _mean("metrics", "average_precision"),
        "per_seed": results,
        "limitations": [
            "The underlying flow data is a synthetic stand-in, so these numbers "
            "validate the pipeline, not real-world detection quality.",
            "The reconstruction threshold is calibrated on benign validation "
            "flows only; raising the percentile trades detection rate for a "
            "lower false-alarm rate.",
        ],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(primary["worst_features"]).to_csv(
        args.output_dir / "worst_features.csv", index=False)

    print("=" * 78)
    print("PHASE 3 — autoencoder anomaly detection (SYNTHETIC DATA)")
    print("=" * 78)
    for result in results:
        metrics = result["metrics"]
        print(f"\nseed {result['seed']}  dataset={result['dataset']}  "
              f"features={result['n_features']}  latent={result['latent_dim']}")
        print(f"  threshold {result['threshold']:.4f} "
              f"(validation-benign percentile {result['percentile']:g})")
        print(f"  detection rate   {metrics['detection_rate']}")
        print(f"  false-alarm rate {metrics['false_alarm_rate']}")
        print(f"  ROC-AUC {metrics['roc_auc']}  "
              f"average precision {metrics['average_precision']}")
        if result["notes"]:
            print(f"  notes: {result['notes']}")
    print(f"\nmean detection rate   {summary['detection_rate_mean']}")
    print(f"mean false-alarm rate {summary['false_alarm_rate_mean']}")
    print(f"\nArtifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())