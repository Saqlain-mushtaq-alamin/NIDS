"""
tests/test_deeplift_explainer.py — DeepLIFT attribution via Captum
(PHASE_4 section 4.2).

✅ Test 4.2 from the plan is reproduced verbatim, followed by the cases a
production explainer has to survive: single-output models (where a class index
is meaningless), out-of-range class indices, per-alert latency accounting, and
the requirement that attributions be non-zero — a silently-zero attribution is
worse than an exception because it renders as a confident "no contributors".
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.deeplift_explainer import DeepLIFTExplainer

N_FEATURES = 10
FEATURES = [f"feat_{i}" for i in range(N_FEATURES)]


def _classifier(width: int = 6) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(N_FEATURES, 32),
        nn.ReLU(),
        nn.Linear(32, width),
    )
    model.eval()
    return model


class TestPlanTest42:
    """✅ Test 4.2 — DeepLIFT Validation (verbatim from the plan)."""

    def test_deeplift_runs(self):
        """DeepLIFT produces valid attributions for a neural network."""
        from src.explainability.deeplift_explainer import DeepLIFTExplainer

        model = torch.nn.Sequential(
            torch.nn.Linear(10, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

        explainer = DeepLIFTExplainer(model)
        X = torch.randn(5, 10)
        result = explainer.explain(X, target_class=1)

        assert result["attributions"].shape == (5, 10)
        assert not all(result["attributions"].flatten() == 0), \
            "Attributions should be non-zero"


class TestDeepLIFTContract:
    """Shapes, targets, baselines and latency."""

    def test_multiclass_target_selection_changes_attribution(self):
        model = _classifier(width=6)
        explainer = DeepLIFTExplainer(model)
        X = np.random.default_rng(0).normal(size=(4, N_FEATURES)).astype(np.float32)

        class0 = explainer.explain(X, target_class=0, feature_names=FEATURES)
        class3 = explainer.explain(X, target_class=3, feature_names=FEATURES)

        assert class0["attributions"].shape == (4, N_FEATURES)
        assert class0["target_class"] == 0
        assert class3["target_class"] == 3
        assert not np.allclose(class0["attributions"], class3["attributions"])

    def test_out_of_range_target_raises(self):
        explainer = DeepLIFTExplainer(_classifier(width=6))
        X = np.zeros((2, N_FEATURES), dtype=np.float32)
        with pytest.raises(ValueError):
            explainer.explain(X, target_class=9)

    def test_multi_output_model_requires_a_target(self):
        explainer = DeepLIFTExplainer(_classifier(width=6))
        X = np.zeros((2, N_FEATURES), dtype=np.float32)
        with pytest.raises(ValueError):
            explainer.explain(X, target_class=None)

    def test_single_output_model_ignores_class_index(self):
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(N_FEATURES, 8), nn.ReLU(), nn.Linear(8, 1))
        model.eval()
        explainer = DeepLIFTExplainer(model)

        X = np.random.default_rng(3).normal(size=(3, N_FEATURES)).astype(np.float32)
        result = explainer.explain(X, target_class=None)

        assert result["attributions"].shape == (3, N_FEATURES)
        assert result["target_class"] is None
        assert not np.allclose(result["attributions"], 0.0)

    def test_attribution_is_wrt_a_baseline(self):
        """A different baseline must produce a different attribution."""
        model = _classifier(width=6)
        X = np.random.default_rng(5).normal(size=(3, N_FEATURES)).astype(np.float32)

        zero = DeepLIFTExplainer(model).explain(X, target_class=1)
        shifted = DeepLIFTExplainer(model, baseline=np.ones(N_FEATURES)
                                    ).explain(X, target_class=1)

        assert not np.allclose(zero["attributions"], shifted["attributions"])

    def test_latency_is_recorded_per_alert(self):
        explainer = DeepLIFTExplainer(_classifier(width=6))
        X = np.random.default_rng(7).normal(size=(6, N_FEATURES)).astype(np.float32)

        result = explainer.explain(X, target_class=1)
        assert result["timing_ms"]["n_alerts"] == 6
        assert result["timing_ms"]["ms_per_alert"] >= 0.0
        assert explainer.benchmark()["n_calls"] == 1
        assert explainer.benchmark()["method"] == "captum/deeplift"

    def test_top_features_ranks_by_absolute_attribution(self):
        explainer = DeepLIFTExplainer.__new__(DeepLIFTExplainer)
        values = np.array([0.02, -0.9, 0.4, 0.1])
        top = explainer.top_features(values, ["a", "b", "c", "d"], top_k=2)
        assert [t["feature"] for t in top] == ["b", "c"]
        assert top[0]["direction"] == "decreases risk"

    def test_numpy_and_tensors_are_both_accepted(self):
        explainer = DeepLIFTExplainer(_classifier(width=6))
        array = np.random.default_rng(9).normal(
            size=(2, N_FEATURES)).astype(np.float32)

        from_array = explainer.explain(array, target_class=0)
        from_tensor = explainer.explain(torch.from_numpy(array), target_class=0)

        np.testing.assert_allclose(from_array["attributions"],
                                   from_tensor["attributions"], atol=1e-6)

    def test_baseline_shape_mismatch_raises(self):
        explainer = DeepLIFTExplainer(_classifier(width=6),
                                      baseline=np.zeros(N_FEATURES + 3))
        with pytest.raises(ValueError):
            explainer.explain(np.zeros((1, N_FEATURES), dtype=np.float32),
                              target_class=1)
