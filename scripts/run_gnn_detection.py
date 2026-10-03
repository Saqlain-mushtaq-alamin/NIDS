"""
scripts/run_gnn_detection.py — Phase 3 GNN experiment driver.

Builds host-communication graphs from the synthetic multi-host simulator, trains
the GraphSAGE window classifier, evaluates it on held-out windows, and writes
every artifact ``docs/phase3_summary.md`` cites.

Honesty constraint enforced by this script
-----------------------------------------
The script always runs a **degree-only logistic-regression baseline** over the
same split and prints it next to the GNN score. On this synthetic data the
baseline matches the GNN, and reporting only the GNN number would misrepresent
the GNN as adding value. The comparison is written to the results directory so
the summary document cannot quietly omit it.

Examples::

    python scripts/run_gnn_detection.py
    python scripts/run_gnn_detection.py --windows 40 --seeds 42 1337
    python scripts/run_gnn_detection.py --quick
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

from src.data.graph_simulator import GraphScenarioConfig, simulate_graph_windows
from src.models.gnn_detector import (LATENCY_BUDGET_MS, HostCommGraph,
                                     WindowGNNClassifier, split_windows)


def _degree_baseline(train_records, test_records) -> dict:
    """
    Fit a logistic regression on degree summaries of the same graphs.

    This is the honest comparison point: if a handful of degree statistics
    already separate the classes, the GNN is not earning its complexity and the
    result must say so.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def featurise(records):
        rows = []
        for record in records:
            x = record.data.x.numpy()
            e = record.data.edge_attr.numpy()
            rows.append(np.concatenate([x.max(axis=0), x.mean(axis=0),
                                        e.max(axis=0), e.mean(axis=0)]))
        return np.asarray(rows, dtype=np.float64)

    x_train = featurise(train_records)
    x_test = featurise(test_records)
    y_train = np.array([r.is_attack for r in train_records], dtype=int)
    y_test = np.array([r.is_attack for r in test_records], dtype=int)

    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(x_train), y_train)
    predicted = model.predict(scaler.transform(x_test))
    return {
        "name": "degree_logistic_regression",
        "accuracy": float((predicted == y_test).mean()),
        "n_features": int(x_train.shape[1]),
    }


def run(seed: int = 42, windows_per_scenario: int = 40, epochs: int = 120,
        hidden_dim: int = 48, latency_iterations: int = 200,
        device: str | None = None) -> dict:
    """Train and evaluate the GNN for one seed, returning the full result."""
    config = GraphScenarioConfig(seed=seed)
    flows = simulate_graph_windows(config,
                                   n_windows_per_scenario=windows_per_scenario)

    builder = HostCommGraph(window_seconds=config.window_seconds)
    records = builder.build_windows(flows, label_column="label")
    train_records, test_records = split_windows(records, test_fraction=0.25,
                                                seed=seed)

    # Balanced class weights: attack windows outnumber benign windows 2:1 by
    # construction, and an unweighted loss collapses onto the majority class.
    targets = np.array([r.is_attack for r in train_records], dtype=int)
    n_benign = max(int((targets == 0).sum()), 1)
    n_attack = max(int((targets == 1).sum()), 1)
    class_weight = [len(targets) / (2 * n_benign), len(targets) / (2 * n_attack)]

    classifier = WindowGNNClassifier(hidden_dim=hidden_dim, epochs=epochs,
                                    seed=seed, class_weight=class_weight,
                                    device=device)
    classifier.n_train_windows_ = len(train_records)
    classifier.fit(train_records)

    latency = classifier.benchmark_latency(n_iterations=latency_iterations,
                                           warmup=min(20, latency_iterations // 5))
    evaluation = classifier.evaluate(test_records, latency=latency)
    baseline = _degree_baseline(train_records, test_records)

    window_table = pd.DataFrame([r.to_row() for r in records])

    return {
        "seed": seed,
        "n_windows": len(records),
        "n_flows": int(len(flows)),
        "n_train_windows": len(train_records),
        "n_test_windows": len(test_records),
        "window_id_overlap": len(
            set(r.window_id for r in train_records)
            & set(r.window_id for r in test_records)),
        "window_label_counts": window_table["label"].value_counts().to_dict(),
        "gnn_metrics": evaluation.metrics,
        "gnn_per_class": (evaluation.per_class.to_dict("records")
                          if evaluation.per_class is not None else None),
        "baseline": baseline,
        "latency": latency.to_dict(),
        "notes": evaluation.notes,
        "window_table": window_table,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024])
    parser.add_argument("--windows", type=int, default=40,
                        help="windows generated per scenario")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--latency-iterations", type=int, default=200)
    parser.add_argument("--output-dir", type=Path,
                        default=_PROJECT_ROOT / "results" / "phase3" / "gnn")
    parser.add_argument("--quick", action="store_true",
                        help="one seed and few windows, for smoke testing")
    args = parser.parse_args(argv)

    seeds = [42] if args.quick else args.seeds
    windows = 8 if args.quick else args.windows
    epochs = 20 if args.quick else args.epochs
    iterations = 30 if args.quick else args.latency_iterations

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run(seed=seed, windows_per_scenario=windows, epochs=epochs,
                   hidden_dim=args.hidden_dim, latency_iterations=iterations)
               for seed in seeds]

    primary = results[0]
    accuracies = [r["gnn_metrics"]["accuracy"] for r in results]
    baseline_accuracies = [r["baseline"]["accuracy"] for r in results]

    summary = {
        "data_source": "SYNTHETIC (src.data.graph_simulator) — not a real capture",
        "seeds": seeds,
        "windows_per_scenario": windows,
        "gnn_accuracy_mean": float(np.mean(accuracies)),
        "gnn_accuracy_std": float(np.std(accuracies)),
        "baseline_accuracy_mean": float(np.mean(baseline_accuracies)),
        "baseline_accuracy_std": float(np.std(baseline_accuracies)),
        "gnn_outperforms_degree_baseline":
            float(np.mean(accuracies)) > float(np.mean(baseline_accuracies)),
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "latency_within_budget": bool(primary["latency"]["within_budget"]),
        "window_id_overlap": primary["window_id_overlap"],
        "per_seed": [
            {k: v for k, v in r.items() if k not in ("window_table",)}
            for r in results
        ],
        "limitations": [
            "The four scenarios are synthetic and structurally distinct, so the "
            "task is easier than real network traffic.",
            "A degree-only logistic regression reaches comparable accuracy, so "
            "this experiment does NOT demonstrate that the GNN adds value over "
            "a simple graph-statistics baseline.",
            "Results validate that the pipeline runs and the gates are met; "
            "they are not a performance claim about real-world detection.",
        ],
    }

    primary["window_table"].to_csv(args.output_dir / "windows.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for index, result in enumerate(results):
        (args.output_dir / f"per_seed_{index}.json").write_text(
            json.dumps({k: v for k, v in result.items()
                        if k != "window_table"}, indent=2, default=str),
            encoding="utf-8")

    print("=" * 78)
    print("PHASE 3 — GNN host-communication-graph detection (SYNTHETIC DATA)")
    print("=" * 78)
    for result in results:
        metrics = result["gnn_metrics"]
        print(f"\nseed {result['seed']}: "
              f"{result['n_windows']} windows "
              f"({result['n_train_windows']} train / "
              f"{result['n_test_windows']} test)")
        print(f"  GNN    accuracy {metrics['accuracy']:.4f}  "
              f"precision {metrics['precision']:.4f}  "
              f"recall {metrics['recall']:.4f}  "
              f"macro-F1 {metrics['macro_f1']:.4f}  "
              f"MCC {metrics['mcc']:.4f}")
        print(f"  BASELINE (degree logreg) accuracy "
              f"{result['baseline']['accuracy']:.4f}")
        latency = result["latency"]
        print(f"  latency mean {latency['mean_ms']:.3f} ms / "
              f"p95 {latency['p95_ms']:.3f} ms on {latency['device']} — "
              f"{'PASS' if latency['within_budget'] else 'FAIL'} "
              f"(budget {latency['budget_ms']:g} ms)")
    print(f"\nGNN mean accuracy        {np.mean(accuracies):.4f} "
          f"± {np.std(accuracies):.4f}")
    print(f"degree baseline accuracy {np.mean(baseline_accuracies):.4f} "
          f"± {np.std(baseline_accuracies):.4f}")
    if not summary["gnn_outperforms_degree_baseline"]:
        print("\nNOTE: the GNN does NOT beat the degree-only baseline on this "
              "data. Reported as-is; see docs/phase3_summary.md.")
    print(f"\nArtifacts written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())