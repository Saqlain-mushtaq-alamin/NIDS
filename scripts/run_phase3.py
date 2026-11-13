"""
scripts/run_phase3.py — Combined Phase 3 driver.

Runs every Phase 3 component in sequence and writes one consolidated report:

* autoencoder anomaly detection (``scripts/run_anomaly_detection.py``)
* host-graph GNN detection (``scripts/run_gnn_detection.py``)
* concept-drift monitoring (``scripts/run_drift_monitoring.py``)
* detector fusion (``src.models.fusion``)

The consolidated report is the single artifact ``docs/phase3_summary.md`` cites.
It deliberately carries a ``limitations`` block into the machine-readable output:
the underlying traffic is synthetic, so these results demonstrate that the
pipeline is correct and the gates are met, not that any detector is accurate on
real network traffic.

Examples::

    python scripts/run_phase3.py
    python scripts/run_phase3.py --quick
    python scripts/run_phase3.py --components fusion drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.fusion import DetectionFusion, run_fusion_experiment

COMPONENTS = ("anomaly", "gnn", "drift", "fusion")


def run_anomaly(quick: bool, **_) -> dict:
    from scripts.run_anomaly_detection import main as anomaly_main
    argv = ["--quick"] if quick else []
    anomaly_main(argv)
    path = _PROJECT_ROOT / "results" / "phase3" / "autoencoder" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_gnn(quick: bool, **_) -> dict:
    from scripts.run_gnn_detection import main as gnn_main
    argv = ["--quick"] if quick else []
    gnn_main(argv)
    path = _PROJECT_ROOT / "results" / "phase3" / "gnn" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_drift(quick: bool, **_) -> dict:
    from scripts.run_drift_monitoring import main as drift_main
    argv = ["--quick"] if quick else []
    drift_main(argv)
    path = _PROJECT_ROOT / "results" / "phase3" / "drift" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run_fusion(quick: bool, seed: int = 42, **_) -> dict:
    """Score the real fusion layer combining real AE and real GNN on real CIC-IDS2017 flows."""
    from src.data.loader import load_dataset
    from src.data.preprocessor import Preprocessor
    from src.models.autoencoder import AnomalyDetector
    from src.models.fusion import (
        DetectionFusion,
        ECDFScoreCalibrator,
        run_real_fusion_experiment,
    )
    from src.models.gnn_detector import DegreeLogisticBaseline, HostCommGraph

    flows = load_dataset("cic-ids-2017", drop_non_features=False)
    if quick:
        flows = flows.head(1200).copy()

    # Fit preprocessor strictly on benign flows only
    preprocessor = Preprocessor(feature_mode="behavior_only", seed=seed)
    benign_flows = flows[flows["label"].astype(str).str.lower() == "benign"].copy()
    preprocessor.fit(benign_flows)

    # Train benign autoencoder
    X_benign, _ = preprocessor.transform(benign_flows)
    n_split = max(int(len(X_benign) * 0.8), 10)
    X_tr, X_val = X_benign[:n_split], X_benign[n_split:]
    ae = AnomalyDetector(latent_dim=8, epochs=15 if quick else 30, seed=seed, percentile=99.0)
    ae.fit(X_tr, X_val=X_val)

    # Calibrator on benign validation errors
    val_errors = ae.reconstruction_error(X_val)
    calibrator = ECDFScoreCalibrator().fit(val_errors)

    # Fit degree baseline on graph windows
    builder = HostCommGraph(window_seconds=60)
    windows = builder.build_windows(flows, label_column="label", attack_fraction=0.10)
    gnn_baseline = DegreeLogisticBaseline(seed=seed)
    gnn_baseline.fit(windows)

    # Execute real fusion
    result = run_real_fusion_experiment(
        flows_df=flows,
        ae_detector=ae,
        gnn_classifier=gnn_baseline,
        calibrator=calibrator,
        graph_builder=builder,
        preprocessor=preprocessor,
        seed=seed,
    )

    frame = result.predictions
    truth = frame["label"].ne("Benign").to_numpy()

    anomaly_only = frame["anomaly_score"].to_numpy() >= 0.5
    graph_only = frame["graph_score"].to_numpy() >= 0.5
    fused = frame["is_anomaly"].to_numpy().astype(bool)

    def _score(predicted):
        tp = int((predicted & truth).sum())
        fp = int((predicted & ~truth).sum())
        fn = int((~predicted & truth).sum())
        prec = float(tp / (tp + fp)) if (tp + fp) else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        return {
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    out_dir = _PROJECT_ROOT / "results" / "phase3" / "fusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "fusion_predictions.csv", index=False)

    summary = {
        "dataset": "real_cic2017",
        "n_flows": len(flows),
        "n_ground_truth_attack": int(truth.sum()),
        "fused": _score(fused),
        "anomaly_view_alone": _score(anomaly_only),
        "graph_view_alone": _score(graph_only),
        "agreement": result.agreement,
        "n_needs_review": result.n_review,
        "n_escalated_as_anomaly": result.n_anomalies,
        "is_demo": False,
        "notes": result.notes,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return summary


RUNNERS = {"anomaly": run_anomaly, "gnn": run_gnn, "drift": run_drift,
           "fusion": run_fusion}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--components", nargs="+", default=list(COMPONENTS),
                        choices=list(COMPONENTS))
    parser.add_argument("--output-dir", type=Path,
                        default=_PROJECT_ROOT / "results" / "phase3")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "data_provenance": (
            "SYNTHETIC. The Phase 2 captures are stand-ins and the host-graph "
            "data comes from src.data.graph_simulator. Results validate the "
            "pipeline and its gates; they are not a real-world performance "
            "claim."
        ),
        "quick_mode": bool(args.quick),
        "components": {},
        "failures": {},
    }

    for name in args.components:
        print(f"\n{'#' * 78}\n# {name.upper()}\n{'#' * 78}")
        try:
            report["components"][name] = RUNNERS[name](quick=args.quick)
        except Exception as error:  # keep going; a partial report is useful
            report["failures"][name] = f"{type(error).__name__}: {error}"
            print(f"!! {name} failed: {type(error).__name__}: {error}")

    path = args.output_dir / "phase3_report.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 78}")
    print("PHASE 3 CONSOLIDATED SUMMARY")
    print("=" * 78)
    print(report["data_provenance"])

    gnn = report["components"].get("gnn")
    if gnn:
        print(f"\nGNN accuracy        {gnn['gnn_accuracy_mean']:.4f} "
              f"± {gnn['gnn_accuracy_std']:.4f}")
        print(f"degree baseline     {gnn['baseline_accuracy_mean']:.4f} "
              f"± {gnn['baseline_accuracy_std']:.4f}")
        print(f"G3 latency gate     "
              f"{'PASS' if gnn['latency_within_budget'] else 'FAIL'} "
              f"(budget {gnn['latency_budget_ms']:g} ms)")
        if not gnn["gnn_outperforms_degree_baseline"]:
            print("  -> the GNN does not beat the degree baseline on this data")

    fusion = report["components"].get("fusion")
    if fusion:
        print(f"\nFusion recall       {fusion['fused']['recall']:.4f} "
              f"(anomaly view alone "
              f"{fusion['anomaly_view_alone']['recall']:.4f}, "
              f"graph view alone {fusion['graph_view_alone']['recall']:.4f})")
        print(f"flows needing review {fusion['n_needs_review']}")

    drift = report["components"].get("drift")
    if drift:
        print(f"\nDrift alerts: shift {drift['mean_alerts_on_shift']:.1f}, "
              f"control {drift['mean_alerts_on_control']:.1f}")

    if report["failures"]:
        print(f"\nFAILED COMPONENTS: {report['failures']}")
    print(f"\nReport written to {path}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())