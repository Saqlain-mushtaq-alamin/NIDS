"""
tests/test_calibration.py — Tests for confidence calibration and temperature scaling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.evaluation.calibration import (
    expected_calibration_error,
    reliability_diagram_data,
    TemperatureScaler,
)


class TestExpectedCalibrationError:
    def test_perfectly_calibrated_probabilities(self):
        # 1000 samples, predicted prob = 0.8, exactly 80% correct
        n = 1000
        rng = np.random.default_rng(42)
        y_true = (rng.uniform(0, 1, size=n) < 0.8).astype(int)
        y_prob = np.column_stack([np.full(n, 0.2), np.full(n, 0.8)])

        res = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert res["ece"] < 0.05
        assert res["n_samples"] == n

    def test_severely_miscalibrated_probabilities(self):
        # Always predicts 0.99 confidence, but accuracy is only 0.2
        n = 500
        y_true = np.zeros(n, dtype=int)
        y_prob = np.column_stack([np.full(n, 0.01), np.full(n, 0.99)])

        res = expected_calibration_error(y_true, y_prob, n_bins=10)
        # Expected error: |0.0 - 0.99| = 0.99
        assert res["ece"] > 0.9

    def test_reliability_diagram_data_structure(self):
        y_true = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.8, 0.4, 0.7])
        data = reliability_diagram_data(y_true, y_prob, n_bins=5)

        assert "bin_edges" in data
        assert "bin_confs" in data
        assert "bin_accs" in data
        assert "bin_counts" in data
        assert len(data["bin_edges"]) == 6


class TestTemperatureScaler:
    def test_temperature_scaling_improves_overconfident_probabilities(self):
        rng = np.random.default_rng(42)
        n = 600
        # Overconfident logits (magnitude too large)
        y_true = rng.integers(0, 3, size=n)
        logits = rng.normal(0, 1, size=(n, 3))
        # Add moderate signal to true class
        logits[np.arange(n), y_true] += 1.5
        # Multiply by 4 to artificially create extreme overconfidence
        overconfident_logits = logits * 4.0

        ts = TemperatureScaler()
        ts.fit(overconfident_logits, y_true)

        assert ts.is_fitted
        # Temperature should be > 1 to soften the overconfident logits
        assert ts.temperature_ > 1.2

        raw_probs = ts._to_logits(overconfident_logits)
        from scipy.special import softmax
        p_raw = softmax(overconfident_logits, axis=1)
        p_cal = ts.predict_proba(overconfident_logits)

        ece_raw = expected_calibration_error(y_true, p_raw)["ece"]
        ece_cal = expected_calibration_error(y_true, p_cal)["ece"]

        assert ece_cal < ece_raw

    def test_temperature_scaling_preserves_argmax(self):
        rng = np.random.default_rng(123)
        logits = rng.normal(0, 2, size=(50, 4))
        ts = TemperatureScaler()
        ts.temperature_ = 2.5
        ts.is_fitted = True

        p_cal = ts.predict_proba(logits)
        assert np.all(p_cal.argmax(axis=1) == logits.argmax(axis=1))
        assert np.allclose(p_cal.sum(axis=1), 1.0)
