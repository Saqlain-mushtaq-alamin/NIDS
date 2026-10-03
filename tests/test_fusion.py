"""
tests/test_fusion.py — Tests for multi-detector fusion (PHASE_3 section 3.4).

The tests pin the *output contract* rather than a headline metric, because the
contract is what downstream consumers depend on: a fused verdict must always be
traceable, must always expose both the escalation label and its ``final_label``
alias, and must never silently resolve a disputed case.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


class TestOutputContract:
    """The fused frame must satisfy the documented column contract."""

    def test_emits_every_contract_column(self):
        from src.models.fusion import FUSION_COLUMNS, DetectionFusion
        result = DetectionFusion().fuse(np.array([0.1, 0.9]),
                                        np.array([0.2, 0.8]))
        assert set(FUSION_COLUMNS) <= set(result.predictions.columns)

    def test_label_and_final_label_agree(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse(np.array([0.1, 0.5, 0.9]),
                                        np.array([0.2, 0.5, 0.8]))
        assert (result.predictions["label"]
                == result.predictions["final_label"]).all()

    def test_every_row_records_a_detection_source(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse(np.array([0.1, 0.9, 0.5, 0.2]),
                                        np.array([0.9, 0.1, 0.5, 0.5]))
        sources = set(result.predictions["detection_source"])
        assert sources
        assert sources <= {"both", "anomaly_detector", "graph_detector",
                           "both_disputed"}

    def test_labels_use_the_escalation_taxonomy(self):
        from src.models.fusion import (LABEL_ATTACK, LABEL_BENIGN, LABEL_REVIEW,
                                       DetectionFusion)
        result = DetectionFusion().fuse(np.array([0.05, 0.95, 0.5]),
                                        np.array([0.05, 0.95, 0.5]))
        assert set(result.predictions["label"]) <= {LABEL_BENIGN, LABEL_ATTACK,
                                                    LABEL_REVIEW}


class TestFusionBehaviour:
    """Behavioural properties of the weighted combination."""

    def test_needs_review_rows_are_flagged_as_anomalies(self):
        from src.models.fusion import LABEL_REVIEW, DetectionFusion
        result = DetectionFusion().fuse(np.array([0.95, 0.05]),
                                        np.array([0.05, 0.95]))
        review = result.predictions[result.predictions["label"] == LABEL_REVIEW]
        assert len(review) == 2
        assert review["needs_review"].all()
        # Escalation is the safe default: a disputed row counts as an anomaly.
        assert review["is_anomaly"].all()

    def test_unanimous_benign_and_attack_are_not_reviewed(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse(np.array([0.01, 0.01]),
                                        np.array([0.01, 0.01]))
        assert not result.predictions["needs_review"].any()
        assert not result.predictions["is_anomaly"].any()

    def test_review_band_can_exclude_escalation(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion(escalate_review=False).fuse(
            np.array([0.95, 0.05]), np.array([0.05, 0.95]))
        review = result.predictions[result.predictions["needs_review"]]
        assert len(review) == 2
        assert not review["is_anomaly"].any()
        assert result.notes

    def test_agreement_is_reported(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse(np.array([0.1, 0.1, 0.9, 0.9]),
                                        np.array([0.1, 0.1, 0.9, 0.9]))
        assert result.agreement["rate"] == pytest.approx(1.0)
        assert result.agreement["n_disputed"] == 0

    def test_weights_are_normalised_and_reflected_in_the_result(self):
        from src.models.fusion import DetectionFusion
        fusion = DetectionFusion(anomaly_weight=3.0, graph_weight=1.0)
        assert fusion.weights == {"anomaly": pytest.approx(0.75),
                                  "graph": pytest.approx(0.25)}
        result = fusion.fuse(np.array([0.5]), np.array([0.5]))
        assert result.weights == fusion.weights

    def test_anomaly_only_view_is_weighted_more_heavily(self):
        from src.models.fusion import DetectionFusion
        fusion = DetectionFusion(anomaly_weight=0.9, graph_weight=0.1)
        # Only the graph view says "attack"; with 90% weight on the per-flow
        # view the fused score must stay below the 0.5 boundary.
        score = fusion.fused_score([0.0], [1.0])
        assert score[0] < 0.5

    def test_graph_only_view_still_contributes(self):
        from src.models.fusion import DetectionFusion
        fusion = DetectionFusion(anomaly_weight=0.5, graph_weight=0.5)
        score = fusion.fused_score([0.0], [1.0])
        assert score[0] == pytest.approx(0.5)

    def test_scores_are_normalised_into_unit_range(self):
        from src.models.fusion import DetectionFusion
        scores = np.array([-1000.0, 0.0, 1000.0])
        # Both views scaled: the graph view is calibrated by default, so the
        # flag is needed to exercise normalisation of an uncalibrated view.
        fused = DetectionFusion().fused_score(scores, scores,
                                              normalise_graph=True)
        assert fused.min() == pytest.approx(0.0)
        assert fused.max() == pytest.approx(1.0)

    def test_absent_graph_view_gets_the_full_weight(self):
        """A missing view must not cap the surviving view below the boundary."""
        from src.models.fusion import DetectionFusion
        fusion = DetectionFusion(anomaly_weight=0.6, graph_weight=0.4)
        assert fusion.effective_weights(graph_available=False) == {
            "anomaly": 1.0, "graph": 0.0}
        assert fusion.fused_score([0.0, 1.0], [0.0, 0.0],
                                  graph_available=False)[1] == pytest.approx(1.0)

    def test_constant_scores_map_to_undecided(self):
        """A flat score carries no ranking information and must not be rescaled."""
        from src.models.fusion import DetectionFusion
        fused = DetectionFusion().fused_score([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert np.allclose(fused, 0.5)

    def test_graph_view_is_optional(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse([0.1, 0.95])
        assert result.notes
        assert list(result.predictions["label"]) == ["Benign", "Attack"]

    def test_audit_columns_carry_raw_values(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse(
            anomaly_score=[0.2, 0.8], graph_score=[0.3, 0.7],
            reconstruction_error=[1.5, 9.0], gnn_probability=[0.3, 0.7])
        assert list(result.predictions["reconstruction_error"]) == [1.5, 9.0]
        assert list(result.predictions["gnn_attack_probability"]) == [0.3, 0.7]


class TestValidation:
    """Invalid inputs must be rejected rather than silently mishandled."""

    def test_rejects_empty_input(self):
        from src.models.fusion import DetectionFusion
        with pytest.raises(ValueError):
            DetectionFusion().fuse([])

    def test_rejects_all_nan_scores(self):
        from src.models.fusion import DetectionFusion
        with pytest.raises(ValueError):
            DetectionFusion().fuse([float("nan"), float("nan")])

    def test_rejects_length_mismatch(self):
        from src.models.fusion import DetectionFusion
        with pytest.raises(ValueError):
            DetectionFusion().fuse([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_rejects_invalid_weights(self):
        from src.models.fusion import DetectionFusion
        with pytest.raises(ValueError):
            DetectionFusion(anomaly_weight=-1.0, graph_weight=1.0)
        with pytest.raises(ValueError):
            DetectionFusion(anomaly_weight=0.0, graph_weight=0.0)

    def test_rejects_invalid_review_band(self):
        from src.models.fusion import DetectionFusion
        with pytest.raises(ValueError):
            DetectionFusion(review_band=0.9)


class TestScoring:
    """Ground truth is used only to score, never to decide."""

    def test_scoring_returns_nothing_without_ground_truth(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse([0.1, 0.9], [0.1, 0.9])
        assert DetectionFusion.score(result) == {}

    def test_scoring_counts_confusion_matrix_terms(self):
        from src.models.fusion import DetectionFusion
        result = DetectionFusion().fuse(
            anomaly_score=[0.9, 0.9, 0.1, 0.1],
            graph_score=[0.9, 0.1, 0.1, 0.1],
            labels=["Attack", "Attack", "Benign", "Attack"])
        scores = DetectionFusion.score(result)
        assert scores["n_ground_truth"] == 4
        assert set(scores) >= {"accuracy", "precision", "recall", "f1",
                               "tp", "fp", "fn", "tn"}
        assert 0.0 <= scores["f1"] <= 1.0


class TestFusionExperiment:
    """The worked example must recover both attack regimes."""

    def test_experiment_separates_both_attack_types(self):
        from src.models.fusion import DetectionFusion, run_fusion_experiment
        result = run_fusion_experiment(seed=42, n_flows=2000)
        scores = DetectionFusion.score(result)
        assert scores["recall"] > 0.9
        assert scores["n_ground_truth"] == 2000
        assert len(result) == 2000

    def test_per_flow_view_alone_misses_exfiltration(self):
        """
        The premise of fusion: no single view sees everything.

        Individually-normal exfiltration flows look benign to the per-flow view,
        and a volumetric attack spread across many peers looks unremarkable to
        the structural view. Fusing them is therefore not redundant.
        """
        from src.models.fusion import DetectionFusion, run_fusion_experiment
        result = run_fusion_experiment(seed=42, n_flows=2000)
        frame = result.predictions

        exfil = frame.iloc[:300]           # per-flow view sees nothing
        volumetric = frame.iloc[300:600]    # structural view is lukewarm
        assert exfil["anomaly_score"].max() < 0.5
        # The structural view rates the volumetric regime far below the
        # exfiltration regime it genuinely recognises.
        assert volumetric["graph_score"].mean() < 0.6 * exfil["graph_score"].mean()
        assert frame["is_anomaly"].sum() > 500