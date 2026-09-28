"""
src/evaluation/metrics.py — Comprehensive metrics computation for NIDS evaluation.
Generates per-class reports, confusion matrices, ROC curves, and summary statistics.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_fscore_support, matthews_corrcoef
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RESULTS_DIR = _PROJECT_ROOT / "results"
_FIGURES_DIR = _RESULTS_DIR / "figures"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    average: str = "macro") -> dict:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        average: Averaging strategy for precision/recall/f1.

    Returns:
        Dict with accuracy, precision, recall, f1, macro_f1, mcc.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    return metrics


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Generate per-class precision, recall, F1 report as DataFrame."""
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report_dict).T
    return df


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def compute_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: list[str]) -> dict:
    """
    Compute one-vs-rest ROC curves and AUC for each class.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities (n_samples, n_classes).
        class_names: List of class names.

    Returns:
        Dict with 'fpr', 'tpr', 'auc' per class, and 'macro_auc'.
    """
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=class_names)

    results = {}
    aucs = []

    for i, cls in enumerate(class_names):
        if i >= y_proba.shape[1]:
            break
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        results[cls] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc),
        }
        aucs.append(roc_auc)

    results["macro_auc"] = float(np.mean(aucs)) if aucs else 0.0
    return results


def latency_benchmark(model, X_test: np.ndarray, batch_sizes: list[int] = None) -> dict:
    """
    Benchmark inference latency.

    Args:
        model: Fitted model with predict() method.
        X_test: Test feature matrix.
        batch_sizes: List of batch sizes to benchmark.

    Returns:
        Dict with per_batch and per_flow latency metrics.
    """
    import time

    if batch_sizes is None:
        batch_sizes = [1, 32, 128, 1024]

    results = {}
    for bs in batch_sizes:
        if bs > len(X_test):
            bs = len(X_test)
        X_batch = X_test[:bs]

        # Warmup
        model.predict(X_batch[:min(10, bs)])

        start = time.perf_counter()
        for _ in range(10):
            model.predict(X_batch)
        elapsed = (time.perf_counter() - start) / 10

        throughput = bs / elapsed if elapsed > 0 else float("inf")
        results[f"batch_{bs}"] = {
            "latency_sec": float(elapsed),
            "throughput_flows_sec": float(throughput),
            "per_flow_usec": float(elapsed / bs * 1_000_000) if bs > 0 else 0,
        }

    # Overall throughput on full test set
    start = time.perf_counter()
    model.predict(X_test)
    total_elapsed = time.perf_counter() - start
    results["full_test"] = {
        "total_sec": float(total_elapsed),
        "throughput_flows_sec": float(len(X_test) / total_elapsed) if total_elapsed > 0 else float("inf"),
        "per_flow_usec": float(total_elapsed / len(X_test) * 1_000_000),
    }

    return results


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                save_dir: Optional[Path] = None) -> Path:
    """Save per-class classification report as CSV."""
    if save_dir is None:
        save_dir = _RESULTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    report_df = per_class_report(y_true, y_pred)
    path = save_dir / "baseline_classification_report.csv"
    report_df.to_csv(path)
    return path


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           save_dir: Optional[Path] = None) -> Path:
    """Save confusion matrix as CSV and optionally as PNG."""
    if save_dir is None:
        save_dir = _RESULTS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = compute_confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    csv_path = save_dir / "baseline_confusion_matrix.csv"
    cm_df.to_csv(csv_path)

    # Try to generate PNG
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig_dir = save_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix — Baseline")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        png_path = fig_dir / "baseline_confusion_matrix.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
    except Exception:
        pass

    return csv_path


def save_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                     class_names: list[str],
                     save_dir: Optional[Path] = None) -> Optional[Path]:
    """Save ROC curves plot as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if save_dir is None:
            save_dir = _RESULTS_DIR
        save_dir = Path(save_dir)
        fig_dir = save_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        roc_data = compute_roc_curves(y_true, y_proba, class_names)

        plt.figure(figsize=(10, 8))
        for cls in class_names:
            if cls in roc_data and isinstance(roc_data[cls], dict):
                plt.plot(roc_data[cls]["fpr"], roc_data[cls]["tpr"],
                         label=f"{cls} (AUC = {roc_data[cls]['auc']:.3f})")

        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves — Baseline (Macro AUC = {roc_data['macro_auc']:.3f})")
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        png_path = fig_dir / "baseline_roc_curves.png"
        plt.savefig(png_path, dpi=150)
        plt.close()
        return png_path
    except Exception:
        return None


def full_evaluation_report(model, X_test: np.ndarray, y_test: np.ndarray,
                            class_names: Optional[list[str]] = None,
                            save_results: bool = True) -> dict:
    """
    Generate a complete evaluation report.

    Args:
        model: Fitted model with predict() and optionally predict_proba().
        X_test: Test features.
        y_test: Test labels.
        class_names: List of class names for ROC curves.
        save_results: If True, save reports to results/ directory.

    Returns:
        Dict with summary metrics, per-class report, confusion matrix, etc.
    """
    y_pred = model.predict(X_test)
    summary = compute_metrics(y_test, y_pred)

    report = {
        "summary": summary,
        "per_class": per_class_report(y_test, y_pred).to_dict(),
        "confusion_matrix": compute_confusion_matrix(y_test, y_pred).tolist(),
    }

    # ROC curves if predict_proba available
    try:
        y_proba = model.predict_proba(X_test)
        if class_names is None:
            class_names = sorted(set(y_test))
        roc_data = compute_roc_curves(y_test, y_proba, class_names)
        report["roc_auc"] = roc_data
        report["summary"]["macro_auc"] = roc_data.get("macro_auc", 0.0)
    except Exception:
        pass

    # Latency
    report["latency"] = latency_benchmark(model, X_test)

    # Save results
    if save_results:
        save_classification_report(y_test, y_pred)
        save_confusion_matrix(y_test, y_pred)

        try:
            y_proba = model.predict_proba(X_test)
            if class_names is None:
                class_names = sorted(set(y_test))
            save_roc_curves(y_test, y_proba, class_names)
        except Exception:
            pass

    return report
