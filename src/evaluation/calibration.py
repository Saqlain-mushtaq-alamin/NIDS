"""
src/evaluation/calibration.py — Confidence calibration and temperature scaling.

Detector confidence estimates (e.g. from neural network softmax or gradient-boosted
tree probability estimates) are often miscalibrated: a detector that outputs 0.95
confidence might only be correct 75% of the time, or vice versa.

In automated response systems (Phase 4), confidence triggers high-impact actions
such as firewall blocks (threshold 0.9) or subnet rate-limiting (threshold 0.7).
Uncalibrated confidences lead to excessive false-positive blocks or missed attacks.

This module provides:
1. `expected_calibration_error`: ECE and Maximum Calibration Error (MCE) over
   reliability bins.
2. `reliability_diagram_data`: Binned confidence, observed accuracy, and counts.
3. `TemperatureScaler`: Post-hoc temperature scaling ($p_i = \text{softmax}(z_i / T)$)
   fitted via cross-entropy minimization on validation logits/probabilities.
"""
from __future__ import annotations

from typing import Optional, Union
import numpy as np

__all__ = [
    "expected_calibration_error",
    "reliability_diagram_data",
    "TemperatureScaler",
]


def _to_probs_and_preds(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalise true labels and predicted probabilities.

    Returns:
        (y_true_indices, confidences, predictions)
    """
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        # Binary probabilities for positive class (class 1)
        probs_1 = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
        probs_0 = 1.0 - probs_1
        y_prob = np.column_stack([probs_0, probs_1])

    if y_prob.ndim != 2:
        raise ValueError(f"y_prob must be 1D or 2D, got shape {y_prob.shape}")

    # Row-normalize to guarantee valid distribution
    row_sums = y_prob.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    y_prob = y_prob / row_sums

    predictions = y_prob.argmax(axis=1)
    confidences = y_prob.max(axis=1)

    y_arr = np.asarray(y_true)
    if y_arr.ndim != 1 or y_arr.shape[0] != y_prob.shape[0]:
        raise ValueError(
            f"y_true shape {y_arr.shape} does not match y_prob rows {y_prob.shape[0]}"
        )

    # Convert y_arr to integer class indices if not already
    if np.issubdtype(y_arr.dtype, np.integer):
        y_true_idx = y_arr.astype(int)
    else:
        # Map distinct labels to integers in order of occurrence or sorted
        classes = sorted(list(set(y_arr)))
        mapping = {c: i for i, c in enumerate(classes)}
        y_true_idx = np.array([mapping[v] for v in y_arr], dtype=int)

    return y_true_idx, confidences, predictions


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> dict:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE partitions predictions into `n_bins` according to predicted confidence.
    Within each bin $B_m$, it compares average confidence to empirical accuracy:
    $$\\text{ECE} = \\sum_{m=1}^{M} \\frac{|B_m|}{N} |\\text{acc}(B_m) - \\text{conf}(B_m)|$$

    Args:
        y_true: True class labels or indices (n,).
        y_prob: Predicted class probabilities (n,) or (n, k).
        n_bins: Number of confidence bins (default 10).
        strategy: 'uniform' (equal-width bins in [0, 1]) or 'quantile' (equal counts).

    Returns:
        dict containing:
            ece: Expected Calibration Error in [0, 1].
            mce: Maximum Calibration Error in [0, 1].
            bin_edges: Boundaries of the bins.
            bin_accs: Empirical accuracy per non-empty bin.
            bin_confs: Average confidence per non-empty bin.
            bin_counts: Number of samples per non-empty bin.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    y_true_idx, confidences, predictions = _to_probs_and_preds(y_true, y_prob)
    accuracies = (y_true_idx == predictions).astype(float)
    n_samples = len(confidences)

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(confidences, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_accs = []
    bin_confs = []
    bin_counts = []
    weighted_diffs = []
    abs_diffs = []

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            diff = abs(bin_acc - bin_conf)

            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(count)
            weighted_diffs.append((count / n_samples) * diff)
            abs_diffs.append(diff)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.5 * (low + high))
            bin_counts.append(0)

    ece = float(np.sum(weighted_diffs))
    mce = float(np.max(abs_diffs)) if abs_diffs else 0.0

    return {
        "ece": ece,
        "mce": mce,
        "n_samples": n_samples,
        "n_bins": n_bins,
        "bin_edges": bin_edges.tolist(),
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
    }


def reliability_diagram_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Convenience wrapper around expected_calibration_error for diagram plotting."""
    return expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy="uniform")


class TemperatureScaler:
    """
    Post-hoc temperature scaling for probability calibration.

    Fits a single positive scalar $T$ on validation logits or probabilities by
    minimising negative log-likelihood (cross-entropy):
    $$\\hat{p}_i(T) = \\text{softmax}(z_i / T)$$

    For tree models (e.g. XGBoost) that produce predicted probabilities rather
    than raw logits, pseudo-logits are computed via log-odds: $z = \\log(P + \\epsilon)$.
    """

    def __init__(self, bounds: tuple[float, float] = (0.05, 20.0)):
        self.bounds = bounds
        self.temperature_: float = 1.0
        self.is_fitted: bool = False

    def _to_logits(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            p1 = np.clip(arr, 1e-12, 1.0 - 1e-12)
            arr = np.column_stack([1.0 - p1, p1])

        # If values are bounded in [0, 1] and rows sum ~ 1, treat as probabilities
        if np.all(arr >= 0.0) and np.allclose(arr.sum(axis=1), 1.0, atol=1e-2):
            clipped = np.clip(arr, 1e-12, 1.0)
            return np.log(clipped)
        return arr

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        """
        Fit temperature parameter $T > 0$ on validation data.

        Args:
            X_val: Validation logits or probabilities of shape (n, k) or (n,).
            y_val: Validation integer class labels (n,).

        Returns:
            self
        """
        from scipy.optimize import minimize_scalar

        logits = self._to_logits(X_val)
        y_val_idx, _, _ = _to_probs_and_preds(y_val, logits)
        n_samples = logits.shape[0]

        def _nll(temp: float) -> float:
            scaled = logits / max(temp, 1e-6)
            # Numerically stable log-softmax
            max_s = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = max_s + np.log(np.sum(np.exp(scaled - max_s), axis=1, keepdims=True))
            log_probs = scaled - log_sum_exp
            # Gather log_prob for true class
            true_log_probs = log_probs[np.arange(n_samples), y_val_idx]
            return -float(np.mean(true_log_probs))

        res = minimize_scalar(_nll, bounds=self.bounds, method="bounded")
        self.temperature_ = float(res.x)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Produce calibrated probabilities: $\\text{softmax}(z / T)$.
        """
        logits = self._to_logits(X)
        temp = self.temperature_ if self.is_fitted else 1.0
        scaled = logits / max(temp, 1e-6)
        max_s = np.max(scaled, axis=1, keepdims=True)
        exp_s = np.exp(scaled - max_s)
        probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        return probs

    def calibrate(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict_proba."""
        return self.predict_proba(X)

