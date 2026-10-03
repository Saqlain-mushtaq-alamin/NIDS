"""
scripts/run_drift_monitoring.py — Phase 3 concept-drift experiment driver.

Fits the drift monitors on a stable baseline stream, then replays a stream whose
distribution shifts partway through, and reports what fired and what did not.

The script also runs a **negative control**: an unshifted stream monitored by the
same detector. Without it, a monitor that flags everything would look identical
to one that works. ADWIN's ``delta`` is a false-positive-rate bound, so some
false alarms on the control stream are expected and are reported as a rate.

Examples::

    python scripts/run_drift_monitoring.py
    python scripts/run_drift_monitoring.py --seeds 42 1337
    python scripts/run_drift_monitoring.py --quick
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

from src.drift.drift_detector import (DriftDetector, _psi,
                                       simulate_concept_drift)


def run(seed: int = 42, n_samples: int = 1200, n_features: int = 8,
        top_k: int = 6, psi_threshold: float = 0.25,
        delta: float = 0.002) -> dict:
    """Fit on a stable baseline, then monitor a shifting stream plus a control."""
    features, _ = simulate_concept_drift(n_samples=n_samples,
                                         n_features=n_features, seed=seed)
    split = n_samples // 2
    baseline = features.iloc[:split].reset_index(drop=True)
    drifting = features.iloc[split:].reset_index(drop=True)

    detector = DriftDetector(top_k=top_k, psi_threshold=psi_threshold,
                             delta=delta, seed=seed).fit(baseline)

    rng = np.random.default_rng(seed + 1)
    confidences_stable = rng.uniform(0.80, 0.95, size=len(drifting))
    confidences_low = rng.uniform(0.00, 0.20, size=len(drifting))
    stable_detections = np.zeros(len(drifting), dtype=int)
    high_detections = np.ones(len(drifting), dtype=int)

    detector.monitor(drifting, confidences=confidences_stable,
                     detections=stable_detections, window_size=200)
    report = detector.monitor(drifting, confidences=confidences_low,
                              detections=high_detections, window_size=200)

    # Negative control: same detector, no shift at all.
    control_detector = DriftDetector(top_k=top_k, psi_threshold=psi_threshold,
                                     delta=delta, seed=seed).fit(baseline)
    control = control_detector.monitor(
        pd.DataFrame(rng.normal(0, 1, size=(len(baseline), n_features)),
                     columns=list(features.columns)),
        window_size=200)

    drifted_features = {c for c in features.columns
                        if abs(drifting[c].mean() - baseline[c].mean()) > 1.0}
    monitored = set(detector.monitored_features_)
    alerted_features = {alert.feature for alert in report.alerts}
    # A feature that was never selected for monitoring cannot be alerted on, so
    # recall is measured over the monitored subset and selection coverage is
    # reported separately rather than folded into a single misleading number.
    shifted_monitored = drifted_features & monitored

    return {
        "seed": seed,
        "monitored_features": detector.monitored_features_,
        "n_alerts": report.n_alerts,
        "alert_kinds": (pd.Series([a.kind for a in report.alerts]
                                  ).value_counts().to_dict()
                        if report.alerts else {}),
        "alerts": [a.to_dict() for a in report.alerts],
        "feature_stats": report.feature_stats.to_dict("records"),
        "detection_rate": report.detection_rate,
        "drifted_features": sorted(drifted_features),
        "monitored_features": sorted(monitored),
        "shifted_and_monitored": sorted(shifted_monitored),
        "alerted_features": sorted(alerted_features),
        "monitoring_coverage": (len(shifted_monitored) / len(drifted_features)
                                if drifted_features else float("nan")),
        "recall_on_monitored_shifted_features": (
            len(shifted_monitored & alerted_features) / len(shifted_monitored)
            if shifted_monitored else float("nan")),
        "control_alerts": control.n_alerts,
        "control_features_alerted": sorted({a.feature for a in control.alerts}),
        "psi_sanity": {
            "identical": float(_psi(np.random.default_rng(0).normal(0, 1, 800),
                                    np.random.default_rng(1).normal(0, 1, 800))),
            "shifted": float(_psi(np.random.default_rng(0).normal(0, 1, 800),
                                  np.random.default_rng(1).normal(3, 1, 800))),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--delta", type=float, default=0.002)
    parser.add_argument("--output-dir", type=Path,
                        default=_PROJECT_ROOT / "results" / "phase3" / "drift")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    seeds = [42] if args.quick else args.seeds
    samples = 400 if args.quick else args.samples
    top_k = 3 if args.quick else args.top_k

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run(seed=seed, n_samples=samples, n_features=args.features,
                   top_k=top_k, delta=args.delta) for seed in seeds]
    primary = results[0]

    summary = {
        "seeds": seeds,
        "adwin_delta": args.delta,
        "delta_is_false_positive_rate_bound": True,
        "mean_alerts_on_shift": float(np.mean([r["n_alerts"] for r in results])),
        "mean_alerts_on_control": float(np.mean([r["control_alerts"]
                                                 for r in results])),
        "mean_monitoring_coverage": float(np.mean(
            [r["monitoring_coverage"] for r in results])),
        "mean_recall_on_monitored_shifted_features": float(np.mean(
            [r["recall_on_monitored_shifted_features"] for r in results])),
        "psi_sanity": primary["psi_sanity"],
        "per_seed": results,
        "limitations": [
            "The drift is synthetic: it demonstrates that the monitors fire and "
            "attribute the change to the right features, not that they will "
            "detect drift in real traffic.",
            "ADWIN's delta bounds the false-positive rate, so the control "
            "stream is expected to raise some alerts; it is reported rather "
            "than tuned away.",
            "Nothing here retrains a model or blocks traffic. Drift output is "
            "evidence for a human to adjudicate.",
        ],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if primary["feature_stats"]:
        pd.DataFrame(primary["feature_stats"]).to_csv(
            args.output_dir / "feature_stats.csv", index=False)
    if primary["alerts"]:
        pd.DataFrame(primary["alerts"]).to_csv(
            args.output_dir / "alerts.csv", index=False)

    print("=" * 78)
    print("PHASE 3 — concept-drift monitoring (SYNTHETIC SHIFT)")
    print("=" * 78)
    for result in results:
        print(f"\nseed {result['seed']}: {result['n_alerts']} alerts on the "
              f"shifted stream, {result['control_alerts']} on the control")
        print(f"  alert kinds      {result['alert_kinds']}")
        print(f"  shifted features     {result['drifted_features']}")
        print(f"  monitored features   {result['monitored_features']}")
        print(f"  alerted features     {result['alerted_features']}")
        print(f"  monitoring coverage  "
              f"{result['monitoring_coverage']:.3f} of shifted features watched")
        print(f"  recall (monitored)   "
              f"{result['recall_on_monitored_shifted_features']:.3f}")
        print(f"  PSI sanity: identical "
              f"{result['psi_sanity']['identical']:.4f} (expect ~0), "
              f"shifted {result['psi_sanity']['shifted']:.3f} (expect >0.25)")
    print(f"\nmean alerts on shift  {summary['mean_alerts_on_shift']:.1f}")
    print(f"mean alerts on control {summary['mean_alerts_on_control']:.1f}")
    print(f"\nArtifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())