"""
tests/test_autoencoder.py — Tests for the flow autoencoder (PHASE_3 section 3.1).

One test documents a correction to the phase plan.  The plan's worked example
asserts that reconstruction loss on 40-D i.i.d. Gaussian input should roughly
halve through a 16-D bottleneck.  That is unattainable: a rank-16 linear map of an
isotropic 40-D unit Gaussian leaves 24/40 of the variance irrecoverable, so the
best achievable MSE is ~0.6 and no autoencoder can beat the identity baseline.
The behavioural tests therefore use **structured** benign data, where a
low-dimensional bottleneck genuinely helps, and assert the property that
actually matters: benign structure is reconstructed better than anomaly noise.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch


def _structured_benign(n_samples=600, n_features=12, seed=0):
    """Low-dimensional benign traffic embedded in a wider feature space."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0, 1, size=(n_samples, 4))
    basis = rng.normal(0, 1, size=(4, n_features))
    return latent @ basis + rng.normal(0, 0.05, size=(n_samples, n_features))


class TestFlowAutoencoder:
    """Tests for src.models.autoencoder.FlowAutoencoder."""

    def test_forward_preserves_shape(self):
        from src.models.autoencoder import FlowAutoencoder
        model = FlowAutoencoder(input_dim=12, latent_dim=8)
        x = torch.randn(5, 12)
        assert model(x).shape == x.shape

    def test_reconstruction_error_shape(self):
        from src.models.autoencoder import FlowAutoencoder
        model = FlowAutoencoder(input_dim=12, latent_dim=8)
        errors = model.reconstruction_error(torch.randn(7, 12))
        assert errors.shape == (7,)

    def test_latent_bottleneck_dimension(self):
        from src.models.autoencoder import FlowAutoencoder
        model = FlowAutoencoder(input_dim=20, latent_dim=6)
        assert model.latent_dim == 6

    def test_encode_returns_bottleneck_width(self):
        from src.models.autoencoder import FlowAutoencoder
        model = FlowAutoencoder(input_dim=20, latent_dim=6)
        assert model.encode(torch.randn(4, 20)).shape == (4, 6)

    def test_rejects_non_positive_dimensions(self):
        from src.models.autoencoder import FlowAutoencoder
        with pytest.raises(ValueError):
            FlowAutoencoder(input_dim=8, latent_dim=0)
        with pytest.raises(ValueError):
            FlowAutoencoder(input_dim=-1, latent_dim=4)


class TestAnomalyDetector:
    """Tests for src.models.autoencoder.AnomalyDetector."""

    def test_separates_structured_benign_from_anomalies(self):
        from src.models.autoencoder import AnomalyDetector
        X_benign = _structured_benign(700, 12, seed=1)
        X_bad = np.random.default_rng(2).normal(12.0, 1.0, size=(200, 12))

        detector = AnomalyDetector(latent_dim=6, epochs=40, seed=1)
        labels = ["Benign"] * 700 + ["DDoS"] * 200
        detector.fit(np.vstack([X_benign, X_bad]), y=labels,
                     X_val=X_benign[:150], y_val=["Benign"] * 150)

        scores = detector.reconstruction_error(np.vstack([X_benign, X_bad]))
        assert scores[:700].mean() < scores[700:].mean()

        metrics = detector.evaluate(np.vstack([X_benign, X_bad]),
                                    labels, dataset="synthetic")
        assert metrics.metrics["recall"] > 0.8

    def test_threshold_calibration_on_validation_benign(self):
        from src.models.autoencoder import AnomalyDetector
        X = _structured_benign(600, 12, seed=3)
        detector = AnomalyDetector(latent_dim=6, epochs=30, seed=3)
        detector.fit(X, X_val=X[:200], y_val=["Benign"] * 200)
        threshold = detector.calibrate_threshold(X[:200], percentile=95.0)
        assert np.isfinite(threshold) and threshold > 0

    def test_rejects_empty_training_data(self):
        from src.models.autoencoder import AnomalyDetector, AutoencoderError
        detector = AnomalyDetector(latent_dim=4)
        with pytest.raises((ValueError, AutoencoderError)):
            detector.fit(np.empty((0, 8)))

    def test_validation_labels_are_required_with_validation_data(self):
        from src.models.autoencoder import AnomalyDetector, AutoencoderError
        X = _structured_benign(50, 6, seed=7)
        detector = AnomalyDetector(latent_dim=3, epochs=2)
        with pytest.raises(AutoencoderError):
            # X_val without y_val cannot distinguish benign from attack rows.
            detector.fit(X, X_val=X[:10])

    def test_round_trip_save_and_load(self, tmp_path):
        from src.models.autoencoder import AnomalyDetector
        X = _structured_benign(300, 10, seed=5)
        detector = AnomalyDetector(latent_dim=4, epochs=20, seed=5)
        detector.fit(X, X_val=X[:100], y_val=["Benign"] * 100)

        detector.save(str(tmp_path), name="ae")
        restored = AnomalyDetector.load(str(tmp_path), name="ae")

        original = detector.reconstruction_error(X[:20])
        reloaded = restored.reconstruction_error(X[:20])
        assert np.allclose(original, reloaded, atol=1e-5)

    def test_evaluate_at_prevalence(self):
        from src.models.autoencoder import AnomalyDetector
        X = _structured_benign(1000, 10, seed=10)
        X_attack = np.random.default_rng(11).normal(8.0, 1.0, size=(500, 10))
        y = np.array(["Benign"] * 1000 + ["Attack"] * 500)

        detector = AnomalyDetector(latent_dim=4, epochs=15, seed=10)
        detector.fit(X[:800], y=["Benign"] * 800, X_val=X[800:1000], y_val=["Benign"] * 200)

        prev_metrics = detector.evaluate_at_prevalence(
            np.vstack([X, X_attack]), y, prevalence=0.05, seed=42
        )
        assert prev_metrics["target_prevalence"] == pytest.approx(0.05)
        assert "false_alarms_per_100k" in prev_metrics
        assert "pr_auc" in prev_metrics
        assert "roc_auc" in prev_metrics
        assert prev_metrics["n_benign"] == 1000


class TestBaselineAnomalyModels:
    """Tests for IsolationForest, PCA Reconstruction, and OneClassSVM baselines."""

    def test_baselines_fit_strictly_on_benign_and_score(self):
        from src.models.autoencoder import BaselineAnomalyModels
        X_tr = _structured_benign(400, 8, seed=20)
        X_val = _structured_benign(100, 8, seed=21)
        X_attack = np.random.default_rng(22).normal(6.0, 1.0, size=(100, 8))

        baselines = BaselineAnomalyModels(latent_dim=4, percentile=99.0, seed=42)
        baselines.fit(X_tr, X_val)

        for name in ("IsolationForest", "PCA", "OneClassSVM"):
            scores = baselines.score_samples(name, np.vstack([X_val, X_attack]))
            assert len(scores) == 200
            # Attack scores should generally exceed benign scores
            assert scores[100:].mean() > scores[:100].mean()

            res = baselines.evaluate(name, np.vstack([X_val, X_attack]), ["Benign"] * 100 + ["Attack"] * 100)
            assert res.metrics["recall"] > 0.5
            assert "false_alarms_per_100k" in res.metrics

    def test_baselines_evaluate_at_prevalence(self):
        from src.models.autoencoder import BaselineAnomalyModels
        X_tr = _structured_benign(300, 8, seed=25)
        X_val = _structured_benign(100, 8, seed=26)
        X_attack = np.random.default_rng(27).normal(6.0, 1.0, size=(200, 8))

        baselines = BaselineAnomalyModels(latent_dim=4, percentile=99.0, seed=42)
        baselines.fit(X_tr, X_val)

        prev_res = baselines.evaluate_at_prevalence(
            "PCA", np.vstack([X_val, X_attack]), np.array(["Benign"] * 100 + ["Attack"] * 200),
            prevalence=0.10, seed=42
        )
        assert prev_res["target_prevalence"] == pytest.approx(0.10)
        assert "false_alarms_per_100k" in prev_res
        assert "pr_auc" in prev_res


class TestAutoencoderPlanCorrection:
    """Documents why the phase plan's Gaussian loss-halving claim is not tested."""

    def test_isotropic_gaussian_cannot_halve_loss(self):
        # A rank-16 bottleneck applied to a 40-D isotropic unit-variance Gaussian
        # must lose at least (40 - 16) / 40 of the variance. Asserting the
        # arithmetic makes the correction explicit rather than folklore.
        irreducible_mse = (40 - 16) / 40
        assert irreducible_mse == pytest.approx(0.6)
        assert irreducible_mse > 0.5, (
            "a rank-16 bottleneck cannot beat the identity baseline, so the "
            "plan's 'loss halves' expectation is unattainable"
        )