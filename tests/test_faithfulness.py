"""
tests/test_faithfulness.py — Deletion-AUC faithfulness metric
(PHASE_4 section 4.3).

✅ Test 4.3 from the plan is reproduced first.  The second class is a
regression suite for the masking logic: the first implementation selected
"top-k features" by comparing feature *indices* against k and then cross-multi
plied the row index against the column index, which masked every row with
*other* rows' features.  Both bugs produce a plausible-looking curve, and both
are invisible unless the expected mask is asserted directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.explainability.faithfulness import (
    as_predict_proba,
    compare_explainers,
    deletion_auc,
    deletion_curve,
)


class SumModel:
    """The plan's mock scorer (outputs are not constrained to [0, 1])."""

    def predict_proba(self, X):
        total = X.sum(axis=1)
        return np.column_stack([1 - total, total])


class IdentityModel:
    """Class-1 probability is driven by feature 0 alone.

    The fixtures feed feature 0 values that are mostly positive, so masking it
    to the 0 baseline visibly *lowers* the probability — which is exactly the
    behaviour the deletion curve is supposed to reward.
    """

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.column_stack([1 - p, p])


def _driven_inputs(n: int, seed: int) -> np.ndarray:
    """(n, 6) inputs where only column 0 carries signal (mean 4, so p ~ 0.98)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 0] = rng.normal(4.0, 1.0, size=n)
    return X


class TestPlanTest43:
    """✅ Test 4.3 — Faithfulness Metric (verbatim from the plan)."""

    def test_faithfulness_metric_computed(self):
        from src.explainability.faithfulness import deletion_auc
        import numpy as np

        class MockModel:
            def predict_proba(self, X):
                return np.column_stack([1 - X.sum(axis=1), X.sum(axis=1)])

        model = MockModel()
        X = np.random.randn(10, 5)
        attributions = np.array([0.5, 0.1, 0.3, 0.05, 0.02])

        auc = deletion_auc(model, X, attributions)
        assert 0 <= auc, "Deletion AUC should be non-negative"


class TestMasking:
    """The mask must hit the right features of the right rows."""

    def test_masks_each_rows_own_top_feature_first(self):
        # Row 0 is driven by column 0, row 1 by column 2.  A correct
        # implementation masks column 0 in row 0 and column 2 in row 1.
        X = np.array([[10.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 10.0, 0.0]])
        attributions = np.array([[5.0, 1.0, 0.5, 0.1],
                                 [0.1, 0.5, 5.0, 1.0]])

        curve = deletion_curve(SumModel(), X, attributions, steps=4,
                               class_index=1)

        # k=1: both rows lose their own dominant feature -> sum is 0.
        assert curve["k_masked"] == [0, 1, 2, 3, 4]
        assert curve["mean_probability"][1] == pytest.approx(0.0)
        # k=0: untouched input; the sum clips to the top of [0, 1].
        assert curve["mean_probability"][0] == pytest.approx(1.0)

    def test_shared_ranking_is_used_for_a_single_vector(self):
        X = np.array([[0.1, 0.2, 0.3]])
        attributions = np.array([0.9, 0.5, 0.1])   # order: col 0, 1, 2
        curve = deletion_curve(SumModel(), X, attributions, steps=3,
                               class_index=1)

        # Progressive removal of col0, then col1, then col2.
        assert curve["mean_probability"] == pytest.approx([0.6, 0.5, 0.3, 0.0])

    def test_baseline_fill_value_is_honoured(self):
        X = np.array([[0.4, 0.4]])
        attributions = np.array([1.0, 0.5])
        curve = deletion_curve(SumModel(), X, attributions, steps=2,
                               class_index=1, baseline=0.2)
        assert curve["mean_probability"] == pytest.approx([0.8, 0.6, 0.4])

    def test_probabilities_are_clipped_before_integration(self):
        # SumModel can score far outside [0, 1]; the AUC must still be bounded.
        X = np.full((5, 4), 50.0)
        auc = deletion_auc(SumModel(), X, np.arange(4.0), steps=4,
                           class_index=1)
        assert 0.0 <= auc <= 1.0

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            deletion_curve(SumModel(), np.zeros((2, 4)), np.zeros(3))
        with pytest.raises(ValueError):
            deletion_auc(SumModel(), np.zeros((2, 4)), np.zeros(4), steps=0)


class TestOrdering:
    """A metric that cannot tell a right explanation from a wrong one is noise."""

    def test_correct_attribution_scores_lower_auc_than_reversed(self):
        X = _driven_inputs(40, seed=0)
        model = IdentityModel()          # only column 0 matters

        faithful = np.array([5.0, 0.1, 0.1, 0.1, 0.1, 0.1])
        unfaithful = faithful[::-1].copy()

        auc_faithful = deletion_auc(model, X, faithful, steps=6, class_index=1)
        auc_unfaithful = deletion_auc(model, X, unfaithful, steps=6,
                                      class_index=1)

        assert auc_faithful < auc_unfaithful

    def test_per_row_class_indices_are_respected(self):
        class TwoClassPerRow:
            def predict_proba(self, X):
                p = 1 / (1 + np.exp(-X[:, 0]))
                return np.column_stack([1 - p, p])

        # Row 0 is strongly class 1 (so explaining it as class 0 scores low),
        # row 1 sits on the decision boundary.
        X = np.array([[5.0, 0.0], [0.0, 0.0]])
        p0 = 1 / (1 + np.exp(-5.0))
        curve = deletion_curve(TwoClassPerRow(), X, np.array([1.0, 0.0]),
                               steps=2, class_index=[0, 1])

        assert curve["mean_probability"][0] == pytest.approx(((1 - p0) + 0.5) / 2)
        # Masking feature 0 puts both rows on the boundary.
        assert curve["mean_probability"][1] == pytest.approx(0.5)
        # ...and that starting point is only reachable with per-row classes:
        # a single class index would have given (p0 + 0.5) / 2 instead.
        assert curve["mean_probability"][0] != pytest.approx((p0 + 0.5) / 2)


class TestCompareExplainers:
    """Paired comparison output contract (PHASE_4 section 4.3.3)."""

    def test_returns_both_score_vectors_and_a_verdict(self):
        X = _driven_inputs(30, seed=2)
        model = IdentityModel()

        faithful = np.tile([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], (30, 1))
        unfaithful = np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 5.0], (30, 1))

        result = compare_explainers(model, X, faithful, unfaithful, steps=6,
                                    class_index=1)

        assert result["n_alerts"] == 30
        assert len(result["shap_scores"]) == 30
        assert len(result["deeplift_scores"]) == 30
        assert result["lower_auc_is_better"] is True
        assert result["shap_mean_deletion_auc"] < result["deeplift_mean_deletion_auc"]
        assert result["more_faithful"] == "shap"
        assert result["p_value"] is not None and 0.0 <= result["p_value"] <= 1.0
        assert "shap" in result["verdict"]

    def test_identical_methods_report_a_tie(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(20, 4))
        attrs = np.tile([1.0, 0.5, 0.2, 0.1], (20, 1))

        result = compare_explainers(IdentityModel(), X, attrs, attrs, steps=4,
                                    class_index=1)
        assert result["more_faithful"] == "tie"
        assert result["mean_difference"] == pytest.approx(0.0)
        # Zero-variance differences make SciPy answer with nan; the report
        # must degrade to None rather than carry nan into JSON.
        p_value = result["p_value"]
        assert p_value is None or 0.0 <= p_value <= 1.0
        assert "nan" not in result["verdict"]

    def test_torch_module_is_accepted_as_scorer(self):
        torch = pytest.importorskip("torch")
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(),
                                    torch.nn.Linear(8, 2))
        model.eval()
        X = np.random.default_rng(4).normal(size=(10, 4))
        attrs = np.abs(np.random.default_rng(5).normal(size=(10, 4)))

        result = compare_explainers(model, X, attrs, attrs, steps=3,
                                    class_index=1)
        assert result["n_alerts"] == 10
        assert all(0.0 <= v <= 1.0 for v in result["shap_scores"])

    def test_predict_proba_wrapper_selects_the_positive_class(self):
        wrapped = as_predict_proba(SumModel())
        out = wrapped(np.array([[1.0, 2.0]]))
        assert out.shape == (1,)
        assert out[0] == pytest.approx(3.0)
