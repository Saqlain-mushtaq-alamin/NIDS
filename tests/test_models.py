"""
tests/test_models.py — Tests for Phase 1 baseline models (Section 1.5).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import time
import pytest


class TestXGBBaseline:
    """Tests for src/models/baseline_xgb.py."""

    def test_xgb_trains(self):
        from src.models.baseline_xgb import XGBBaseline
        X = np.random.randn(500, 20)
        y = np.random.choice(["Benign", "DoS", "PortScan"], size=500)
        model = XGBBaseline(n_estimators=10, max_depth=3)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10

    def test_xgb_predict_proba(self):
        from src.models.baseline_xgb import XGBBaseline
        X = np.random.randn(500, 20)
        y = np.random.choice(["Benign", "DoS"], size=500)
        model = XGBBaseline(n_estimators=10, max_depth=3)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape[0] == 5
        assert proba.shape[1] == 2
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_xgb_feature_importance(self):
        from src.models.baseline_xgb import XGBBaseline
        X = np.random.randn(500, 20)
        y = np.random.choice(["Benign", "DoS"], size=500)
        model = XGBBaseline(n_estimators=10, max_depth=3)
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert "importances" in imp
        assert len(imp["importances"]) == 20


class TestRFBaseline:
    """Tests for src/models/baseline_rf.py."""

    def test_rf_trains(self):
        from src.models.baseline_rf import RFBaseline
        X = np.random.randn(500, 20)
        y = np.random.choice(["Benign", "DoS", "PortScan"], size=500)
        model = RFBaseline(n_estimators=10, max_depth=5)
        model.fit(X, y)
        preds = model.predict(X[:10])
        assert len(preds) == 10

    def test_rf_predict_proba(self):
        from src.models.baseline_rf import RFBaseline
        X = np.random.randn(500, 20)
        y = np.random.choice(["Benign", "DoS"], size=500)
        model = RFBaseline(n_estimators=10, max_depth=5)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
        assert proba.shape[0] == 5


class TestMetrics:
    """Tests for src/evaluation/metrics.py (Section 1.6)."""

    def test_metrics_computation(self):
        from src.evaluation.metrics import compute_metrics
        y_true = np.array(["Benign", "DoS", "Benign", "PortScan", "DoS"])
        y_pred = np.array(["Benign", "DoS", "DoS", "PortScan", "DoS"])
        metrics = compute_metrics(y_true, y_pred)
        required_keys = {"accuracy", "precision", "recall", "f1", "macro_f1"}
        assert required_keys.issubset(set(metrics.keys())), \
            f"Missing metrics: {required_keys - set(metrics.keys())}"

    def test_perfect_predictions(self):
        from src.evaluation.metrics import compute_metrics
        y = np.array(["A", "B", "C", "A", "B"])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0

    def test_per_class_report(self):
        from src.evaluation.metrics import per_class_report
        y_true = np.array(["A", "A", "B", "B", "C"])
        y_pred = np.array(["A", "B", "B", "C", "C"])
        report = per_class_report(y_true, y_pred)
        assert "precision" in report.columns
        assert "recall" in report.columns
        assert "f1-score" in report.columns


class TestInferenceLatency:
    """Performance tests for inference throughput."""

    def test_inference_throughput(self):
        from src.models.baseline_xgb import XGBBaseline
        X = np.random.randn(1000, 40)
        y = np.random.choice(["Benign", "DoS"], size=1000)
        model = XGBBaseline(n_estimators=10, max_depth=3)
        model.fit(X, y)

        X_test = np.random.randn(10000, 40)
        start = time.perf_counter()
        model.predict(X_test)
        elapsed = time.perf_counter() - start
        throughput = 10000 / elapsed
        # Should be fast (at least 1K flows/sec on synthetic data)
        assert throughput >= 1000, f"Throughput {throughput:.0f} flows/sec < 1K minimum"
