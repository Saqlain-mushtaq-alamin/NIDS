"""
tests/test_shap_explainer.py — Tests for SHAP attribution (PHASE_4 section 4.1).

The plan's Test 4.1 is reproduced verbatim at the top of each class so the
phase gate is checkable against the plan itself; the remaining tests pin the
behaviours a SOC console depends on: class selection for multi-class models,
per-alert latency accounting, and failure modes that must raise rather than
return silently-wrong attributions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import xgboost as xgb

from src.explainability.shap_explainer import SHAPExplainer

FEATURES = [f"feat_{i}" for i in range(10)]


def _binary_model():
    """The plan's Test 4.1 model, trained exactly as written."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False,
                              eval_metric="logloss")
    model.fit(X, y)
    return model, X


def _multiclass_model():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 8))
    y = (X[:, 0] > 0).astype(int) + 2 * (X[:, 1] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=20, eval_metric="mlogloss")
    model.fit(X, y)
    return model, X


class TestPlanTest41:
    """✅ Test 4.1 — SHAP Explainer Validation (verbatim from the plan)."""

    def test_shap_tree_explainer_runs(self):
        """SHAP TreeExplainer produces valid explanations for XGBoost."""
        model, X = _binary_model()

        explainer = SHAPExplainer(model, model_type="tree")
        result = explainer.explain(X[:5], feature_names=FEATURES)

        assert result["shap_values"] is not None
        assert len(result["shap_values"]) == 5 or result["shap_values"].shape[0] == 5

    def test_shap_top_features(self):
        """Top features correctly identified."""
        shap_values = np.array([0.1, -0.5, 0.3, 0.01, -0.8])
        feature_names = ["duration", "packets", "bytes", "flags", "iat_std"]

        explainer = SHAPExplainer.__new__(SHAPExplainer)
        top = explainer.top_features(shap_values, feature_names, top_k=3)

        assert top[0]["feature"] == "iat_std"        # Largest absolute value
        assert top[0]["direction"] == "decreases risk"  # Negative SHAP
        assert len(top) == 3


class TestExplainContract:
    """Shape, class selection and bookkeeping of :meth:`explain`."""

    def test_binary_model_returns_one_vector_per_alert(self):
        model, X = _binary_model()
        explainer = SHAPExplainer(model, model_type="tree")
        result = explainer.explain(X[:7], feature_names=FEATURES)

        assert result["shap_values"].shape == (7, 10)
        assert result["feature_names"] == FEATURES
        assert result["n_samples"] == 7
        assert isinstance(result["base_value"], float)

    def test_multiclass_cube_and_class_selection(self):
        """A 6/4-class model yields (n, f, k) until a class is requested."""
        model, X = _multiclass_model()
        explainer = SHAPExplainer(model, model_type="tree")

        cube = explainer.explain(X[:4], feature_names=[f"f{i}" for i in range(8)])
        assert cube["shap_values"].shape == (4, 8, 4)

        single = explainer.explain(X[:4], feature_names=[f"f{i}" for i in range(8)],
                                   class_index=2)
        assert single["shap_values"].shape == (4, 8)
        assert single["class_index"] == 2
        np.testing.assert_allclose(single["shap_values"],
                                   cube["shap_values"][:, :, 2])

    def test_explanation_latency_is_recorded(self):
        """Per-alert timing is part of the contract (plan step 4.1.6)."""
        model, X = _binary_model()
        explainer = SHAPExplainer(model, model_type="tree")

        result = explainer.explain(X[:5], feature_names=FEATURES)
        timing = result["timing_ms"]
        assert timing["n_alerts"] == 5
        assert timing["total_ms"] >= 0.0
        assert timing["ms_per_alert"] == pytest.approx(
            timing["total_ms"] / 5, rel=1e-6)

        explainer.explain(X[5:8], feature_names=FEATURES)
        summary = explainer.benchmark()
        assert summary["n_calls"] == 2
        assert summary["n_alerts"] == 8
        assert summary["mean_ms_per_alert"] is not None

    def test_global_importance_ranks_the_driving_feature(self):
        model, X = _binary_model()
        explainer = SHAPExplainer(model, model_type="tree")
        result = explainer.explain(X[:50], feature_names=FEATURES)

        ranking = explainer.global_importance(result["shap_values"], FEATURES,
                                              top_k=10)
        assert ranking[0]["feature"] in {"feat_0", "feat_1"}
        assert ranking[0]["mean_abs_shap"] >= ranking[-1]["mean_abs_shap"]
        assert 0.0 < ranking[0]["share_of_total"] <= 1.0

    def test_mismatched_feature_names_raise(self):
        model, X = _binary_model()
        explainer = SHAPExplainer(model, model_type="tree")
        with pytest.raises(ValueError):
            explainer.explain(X[:3], feature_names=["only_one"])

    def test_uninitialised_tree_explainer_raises(self):
        explainer = SHAPExplainer(None, model_type="kernel")
        with pytest.raises(RuntimeError):
            explainer.explain(np.zeros((1, 10)), feature_names=FEATURES)

    def test_summary_plot_is_written(self, tmp_path):
        model, X = _binary_model()
        explainer = SHAPExplainer(model, model_type="tree")
        result = explainer.explain(X[:30], feature_names=FEATURES)

        out = explainer.save_summary_plot(result["shap_values"], FEATURES,
                                          tmp_path / "shap.png", top_k=5)
        if out is not None:  # a plotting backend is present in this project
            assert out.is_file() and out.stat().st_size > 0


class TestKernelExplainer:
    """KernelSHAP path used for the neural models."""

    def test_background_is_capped(self):
        class Stub:
            @staticmethod
            def predict_proba(Z):
                Z = np.atleast_2d(Z)
                p = 1 / (1 + np.exp(-Z.sum(axis=1)))
                return np.column_stack([1 - p, p])

        background = np.random.default_rng(0).normal(size=(500, 6))
        explainer = SHAPExplainer.for_neural(background, Stub().predict_proba,
                                             max_background_samples=25)
        assert explainer.background_.shape[0] == 25
        assert explainer.model_type == "kernel"

    def test_kernel_matches_tree_on_a_tree_model(self):
        """KernelSHAP on a tree model must land near the exact values."""
        model, X = _binary_model()
        background = X[:40]
        sample = X[:20]

        exact = SHAPExplainer(model, model_type="tree").explain(
            sample, feature_names=FEATURES)["shap_values"]
        approx = SHAPExplainer.for_neural(
            background, lambda Z: np.asarray(model.predict_proba(Z), dtype=float)
        ).explain(sample, feature_names=FEATURES)["shap_values"]

        # A 2-column predict_proba must collapse to the positive class, the
        # same shape the tree path returns for a binary model.
        assert approx.shape == exact.shape == (20, 10)

        # The two paths explain different output scales (tree margin vs.
        # probability), so equality is not expected — but they must tell the
        # same story: the same curve over the batch and the same top features.
        assert np.corrcoef(exact.ravel(), approx.ravel())[0, 1] > 0.9
        top_exact = set(np.argsort(np.abs(exact).mean(axis=0))[::-1][:3])
        top_approx = set(np.argsort(np.abs(approx).mean(axis=0))[::-1][:3])
        assert len(top_exact & top_approx) >= 2
        assert (np.sign(exact) == np.sign(approx)).mean() > 0.8

    def test_background_is_required(self):
        with pytest.raises(ValueError):
            SHAPExplainer(None, model_type="kernel").set_background(
                np.zeros((3, 4)))
