"""
tests/test_oracle_mcc.py — Unit tests for vectorized ROC MCC computation and Oracle thresholding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef, roc_curve

from src.evaluation.metrics import compute_vectorized_mcc_roc, find_oracle_mcc


def test_vectorized_mcc_matches_sklearn_exactly():
    rng = np.random.default_rng(42)
    n = 1000
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(0, 1, size=n)

    fpr, tpr, thresholds, mcc_vec = compute_vectorized_mcc_roc(y_true, y_proba)

    # Test 20 random points across the thresholds
    indices = rng.choice(len(thresholds), size=min(20, len(thresholds)), replace=False)
    for idx in indices:
        th = thresholds[idx]
        pred = (y_proba >= th).astype(int)
        expected_mcc = matthews_corrcoef(y_true, pred)
        np.testing.assert_allclose(mcc_vec[idx], expected_mcc, atol=1e-5)


def test_find_oracle_mcc_exceeds_or_equals_arbitrary_thresholds():
    rng = np.random.default_rng(123)
    n = 2000
    y_true = rng.integers(0, 2, size=n)
    # Give some signal
    y_proba = np.clip(y_true * 0.4 + rng.uniform(0, 0.6, size=n), 0, 1)

    source_th = 0.75
    best_th, best_mcc, best_tpr, best_fpr = find_oracle_mcc(y_true, y_proba, source_th=source_th)

    # Test that oracle is >= MCC of every candidate tested
    pred_src = (y_proba >= source_th).astype(int)
    mcc_src = matthews_corrcoef(y_true, pred_src)
    assert best_mcc >= mcc_src

    # Test 50 random thresholds
    test_thresholds = rng.uniform(0.01, 0.99, size=50)
    for th in test_thresholds:
        p = (y_proba >= th).astype(int)
        m = matthews_corrcoef(y_true, p)
        assert best_mcc >= m - 1e-6, f"Oracle MCC {best_mcc} was exceeded by threshold {th} with MCC {m}"


def test_oracle_mcc_all_printed_roc_points():
    rng = np.random.default_rng(999)
    n = 1500
    y_true = rng.integers(0, 2, size=n)
    y_proba = rng.uniform(0, 1, size=n)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    step = max(1, len(fpr) // 10)
    sample_indices = list(range(0, len(fpr), step))[:10]

    best_th, best_mcc, _, _ = find_oracle_mcc(y_true, y_proba)

    for idx in sample_indices:
        pred = (y_proba >= thresholds[idx]).astype(int)
        mcc_pt = matthews_corrcoef(y_true, pred)
        assert best_mcc >= mcc_pt - 1e-6
