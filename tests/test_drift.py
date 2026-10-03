"""
tests/test_drift.py — Tests for concept-drift monitoring (PHASE_3 section 3.3).

:class:`TestPsi` includes a regression test for a severe arithmetic bug: the
percentages were divided by the sum of the raw feature *values* rather than the
sum of the bin *counts*.  For any feature averaging near zero — which is every
standardised flow feature — that denominator is ~0, so the percentages stopped
being proportions and PSI returned values in the thousands instead of the
conventional 0-4 range.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def stable_frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0, 1, size=(400, 6)),
                        columns=list("abcdef"))


class TestPsi:
    """Tests for the PSI helper."""

    def test_identical_distributions_are_stable(self):
        from src.drift.drift_detector import _psi
        rng = np.random.default_rng(0)
        assert _psi(rng.normal(0, 1, 800), rng.normal(0, 1, 800)) < 0.1

    def test_large_shift_is_significant(self):
        from src.drift.drift_detector import _psi
        rng = np.random.default_rng(1)
        assert _psi(rng.normal(0, 1, 800), rng.normal(3, 1, 800)) > 0.25

    def test_percentages_are_normalised(self):
        """
        Regression: PSI must not blow up for near-zero-mean features.

        The previous code divided bin counts by ``expected.sum()`` — the sum of
        the raw values, near zero for centred features — instead of by
        ``expected_counts.sum()``. That produced PSI values in the thousands.
        """
        from src.drift.drift_detector import _psi
        rng = np.random.default_rng(2)
        psi = _psi(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        assert psi < 1.0, f"PSI for two identical samples was {psi}"

    def test_degenerate_reference_returns_nan(self):
        from src.drift.drift_detector import _psi
        assert np.isnan(_psi(np.ones(50), np.random.randn(50)))
        assert np.isnan(_psi(np.array([1.0]), np.array([1.0])))

    def test_constant_actual_is_flagged(self):
        from src.drift.drift_detector import _psi
        rng = np.random.default_rng(3)
        assert _psi(rng.normal(0, 1, 500), np.full(500, 9.0)) > 0.25


class TestDriftDetector:
    """Tests for src/drift/drift_detector.DriftDetector."""

    def test_selects_high_variance_features(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        frame = stable_frame.copy()
        frame["z_quiet"] = 0.01
        frame["z_loud"] = frame["a"] * 25.0

        detector = DriftDetector(top_k=3).fit(frame)
        assert len(detector.monitored_features_) == 3
        assert "z_quiet" not in detector.monitored_features_
        assert detector.monitored_features_[0] == "z_loud"

    def test_ignores_constant_features(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        frame = stable_frame.copy()
        frame["constant"] = 5.0
        detector = DriftDetector(top_k=6).fit(frame)
        assert "constant" not in detector.monitored_features_

    def test_monitor_requires_fit(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        with pytest.raises(RuntimeError):
            DriftDetector().monitor(stable_frame)

    def test_detects_distribution_shift(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        rng = np.random.default_rng(5)
        detector = DriftDetector(top_k=6).fit(stable_frame)
        shifted = pd.DataFrame(rng.normal(4, 1, size=(400, 6)),
                               columns=list("abcdef"))
        report = detector.monitor(shifted)

        kinds = {alert.kind for alert in report.alerts}
        assert "feature_drift" in kinds or "psi_shift" in kinds
        flagged = {alert.feature for alert in report.alerts}
        assert len(flagged) == 6
        assert report.n_samples == 400

    def test_adwin_is_primed_with_the_reference(self, stable_frame):
        """
        Regression: ADWIN must be able to see a shift that starts *after* fit().

        An unprimed detector handed an already-shifted window observes a
        stationary stream and correctly stays silent, while PSI fires. Priming
        in fit() gives both monitors the same baseline.
        """
        from src.drift.drift_detector import DriftDetector
        rng = np.random.default_rng(6)
        detector = DriftDetector(top_k=2).fit(stable_frame)
        name = detector.monitored_features_[0]
        # ``total`` is the sum of observed values and is ~0 for centred data
        # (and may be negative); ``width`` is the number of samples consumed.
        assert detector._detectors[name].width > 0

        shifted = pd.DataFrame(rng.normal(4, 1, size=(400, 6)),
                               columns=list("abcdef"))
        report = detector.monitor(shifted)
        assert report.n_alerts > 0
        # Counted by the wrapper: River 0.22's C-backed ``n_detections`` stays
        # at 0 even when drift is raised.
        assert detector._detection_counts[name] > 0

    def test_alerts_avoid_percent_of_a_near_zero_mean(self, stable_frame):
        """Severity must be reported as a z-score, not a percentage of the mean."""
        from src.drift.drift_detector import DriftDetector
        rng = np.random.default_rng(7)
        detector = DriftDetector(top_k=3).fit(stable_frame)
        shifted = pd.DataFrame(rng.normal(3, 1, size=(400, 6)),
                               columns=list("abcdef"))
        for alert in detector.monitor(shifted).alerts:
            if alert.kind == "feature_drift":
                assert alert.z_score is not None
                assert abs(alert.z_score) < 1e4
                assert "%" not in alert.message

    def test_alert_serialisation(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        rng = np.random.default_rng(8)
        detector = DriftDetector(top_k=3).fit(stable_frame)
        shifted = pd.DataFrame(rng.normal(4, 1, size=(400, 6)),
                               columns=list("abcdef"))
        report = detector.monitor(shifted)
        assert report.alerts
        payload = report.to_dict()
        assert payload["n_alerts"] == len(report.alerts)
        assert {"feature", "kind", "severity", "message"} <= set(
            payload["alerts"][0])

    def test_confidence_and_detection_monitors(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        rng = np.random.default_rng(9)
        detector = DriftDetector(top_k=2).fit(stable_frame)

        stable = pd.DataFrame(rng.normal(0, 1, size=(300, 6)),
                              columns=list("abcdef"))
        confidences = rng.uniform(0.8, 0.95, size=300)
        detector.monitor(stable, confidences=confidences,
                         detections=np.zeros(300, dtype=int))

        low_confidence = rng.uniform(0.0, 0.2, size=300)
        report = detector.monitor(stable, confidences=low_confidence,
                                  detections=np.ones(300, dtype=int))
        kinds = {alert.kind for alert in report.alerts}
        assert kinds & {"prediction_drift", "detection_rate_drift"}
        assert report.detection_rate == pytest.approx(1.0)

    def test_mismatched_optional_sequence_raises(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        detector = DriftDetector(top_k=2).fit(stable_frame)
        with pytest.raises(ValueError):
            detector.monitor(stable_frame, confidences=[0.5, 0.5])

    def test_missing_monitored_column_raises(self, stable_frame):
        from src.drift.drift_detector import DriftDetector
        detector = DriftDetector(top_k=3).fit(stable_frame)
        with pytest.raises(ValueError):
            detector.monitor(stable_frame[["a", "b"]])

    def test_rejects_invalid_hyperparameters(self):
        from src.drift.drift_detector import DriftDetector
        with pytest.raises(ValueError):
            DriftDetector(top_k=0)
        with pytest.raises(ValueError):
            DriftDetector(delta=0.0)
        with pytest.raises(ValueError):
            DriftDetector(delta=1.5)
        with pytest.raises(ValueError):
            DriftDetector(psi_threshold=-1)

    def test_fit_rejects_empty_frame(self):
        from src.drift.drift_detector import DriftDetector
        with pytest.raises(ValueError):
            DriftDetector().fit(pd.DataFrame())

    def test_round_trip_save_and_load(self, stable_frame, tmp_path):
        from src.drift.drift_detector import DriftDetector
        detector = DriftDetector(top_k=3).fit(stable_frame)
        rng = np.random.default_rng(10)
        detector.monitor(pd.DataFrame(rng.normal(1, 1, size=(200, 6)),
                                      columns=list("abcdef")))

        path = detector.save(tmp_path / "drift.pkl")
        restored = DriftDetector.load(path)
        assert restored.monitored_features_ == detector.monitored_features_
        assert restored._fitted is True


class TestSimulateConceptDrift:
    """Tests for the synthetic drift generator."""

    def test_shift_is_applied_to_only_some_features(self):
        from src.drift.drift_detector import simulate_concept_drift
        features, _ = simulate_concept_drift(n_samples=600, n_features=8, seed=0)
        before = features.iloc[:300]
        after = features.iloc[300:]
        shifts = (after.mean() - before.mean()).abs()
        assert (shifts > 1.0).sum() == 4
        # Unshifted features still carry sampling noise around zero; they must
        # stay small relative to the 4.0 shift, not exactly zero.
        assert (shifts < 0.5).sum() == 4

    def test_run_drift_experiment_reports_alerts(self):
        from src.drift.drift_detector import run_drift_experiment
        report = run_drift_experiment(seed=0)
        assert report.n_alerts > 0
        assert report.feature_stats is not None