"""
tests/test_tiered_explanation.py — Two-tier analyst-facing explanations
(PHASE_4 section 4.4, requirements FR6/FR7).

✅ Test 4.4 from the plan comes first.  The rest pins the properties that make
Tier 1 trustworthy in a SOC console: every taxonomy label maps to a severity
and an action, a partial alert degrades to a placeholder instead of an
exception, the feature name is always visible on an indicator line, and Tier 2
reports how far SHAP and DeepLIFT agree.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.explainability import TieredExplanation, build_alert
from src.explainability.attribution import feature_description

FEATURES = ["syn_flag_count", "flow_duration", "total_fwd_packets",
            "flow_bytes_per_sec", "flow_iat_std", "rst_flag_count"]
SHAP = np.array([0.82, -0.31, 0.24, 0.11, -0.05, 0.02])
DEEPLIFT = np.array([0.75, -0.28, 0.19, 0.14, -0.06, 0.03])


def _alert(**overrides) -> dict:
    alert = {
        "label": "DoS",
        "confidence": 0.97,
        "top_features": [
            {"feature": "syn_flag_count", "contribution": 0.82,
             "direction": "increases risk",
             "description": "syn_flag_count: raised risk (SHAP +0.820)"},
            {"feature": "flow_duration", "contribution": -0.31,
             "direction": "decreases risk",
             "description": "flow_duration: lowered risk (SHAP -0.310)"},
            {"feature": "total_fwd_packets", "contribution": 0.24,
             "direction": "increases risk"},
        ],
        "source_ip": "10.0.0.5",
        "feature_names": FEATURES,
        "shap_values": SHAP.tolist(),
        "deeplift_values": DEEPLIFT.tolist(),
    }
    alert.update(overrides)
    return alert


class TestPlanTest44:
    """✅ Test 4.4 — Tier-1/Tier-2 Validation (verbatim from the plan)."""

    def test_tier1_summary_generated(self):
        from src.explainability import TieredExplanation

        explainer = TieredExplanation()
        alert = {
            "label": "DoS",
            "confidence": 0.94,
            "top_features": [
                {"feature": "syn_flag_count", "contribution": 0.8,
                 "direction": "increases risk"},
                {"feature": "flow_duration", "contribution": 0.5,
                 "direction": "increases risk"},
                {"feature": "total_fwd_packets", "contribution": 0.3,
                 "direction": "increases risk"},
            ],
        }
        summary = explainer.tier1_summary(alert)
        assert len(summary) < 500  # FR6: concise
        assert "DoS" in summary
        assert "syn_flag_count" in summary  # Top feature named

    def test_tier1_all_attack_types_covered(self):
        from src.explainability import TieredExplanation

        explainer = TieredExplanation()
        for label in ["DoS", "DDoS", "PortScan", "BruteForce", "WebAttack",
                      "Botnet", "Infiltration"]:
            alert = {"label": label, "confidence": 0.9,
                     "top_features": [{"feature": "f1", "contribution": 0.5}]}
            summary = explainer.tier1_summary(alert)
            assert len(summary) > 0
            assert label in summary


class TestTier1:
    """Triage text: complete, short, and readable without the drill-down."""

    @pytest.mark.parametrize("label,severity", [
        ("DoS", "HIGH"), ("DDoS", "HIGH"), ("Botnet", "HIGH"),
        ("Infiltration", "HIGH"), ("BruteForce", "MEDIUM"),
        ("WebAttack", "MEDIUM"), ("PortScan", "LOW"), ("Benign", "LOW"),
        ("SomethingUnmapped", "MEDIUM"),
    ])
    def test_severity_table(self, label, severity):
        assert TieredExplanation().severity_for(label) == severity

    @pytest.mark.parametrize("label", [
        "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet",
        "Infiltration", "Anomaly", "Benign",
    ])
    def test_every_taxonomy_label_has_an_action(self, label):
        explainer = TieredExplanation()
        assert explainer.action_for(label) != explainer.DEFAULT_ACTION

    def test_unknown_label_falls_back_to_safe_defaults(self):
        explainer = TieredExplanation()
        assert explainer.severity_for("NotALabel") == "MEDIUM"
        assert explainer.action_for("NotALabel") == explainer.DEFAULT_ACTION

    def test_summary_states_confidence_severity_and_action(self):
        explainer = TieredExplanation()
        summary = explainer.tier1_summary(_alert())
        assert "HIGH CONFIDENCE ALERT" in summary
        assert "97.0% confidence" in summary
        assert "DoS Detected" in summary
        assert summary.count("Recommended Action:") == 1
        assert explainer.action_for("DoS") in summary
        assert "(source 10.0.0.5)" in summary

    def test_only_the_top_three_indicators_are_listed(self):
        summary = TieredExplanation().tier1_summary(_alert())
        assert summary.count("1.") == 1 and summary.count("3.") == 1
        assert "4." not in summary

    def test_indicator_lines_always_name_the_feature(self):
        """An analyst must never see a bare ratio with no field attached."""
        summary = TieredExplanation().tier1_summary(_alert())
        for feature in ("syn_flag_count", "flow_duration", "total_fwd_packets"):
            assert feature in summary

    def test_indicator_lines_stay_within_the_scan_budget(self):
        explainer = TieredExplanation()
        line = explainer._indicator_line({
            "feature": "flow_bytes_per_sec",
            "contribution": 0.5,
            "value": 12345.6789,
            "baseline": 0.0001,
        })
        assert len(line) <= TieredExplanation.MAX_DESCRIPTION_CHARS

    def test_partial_alert_degrades_instead_of_raising(self):
        explainer = TieredExplanation()
        summary = explainer.tier1_summary({})
        assert "Anomaly Detected" in summary
        assert "no attribution available" in summary

        summary = explainer.tier1_summary({"label": "DoS", "confidence": None})
        assert "0.0% confidence" in summary

    def test_summary_survives_a_cp1252_console(self):
        """Windows consoles default to cp1252, which cannot encode emoji.

        The phase plan's example prefixed severity with a coloured emoji;
        rendering one raised UnicodeEncodeError mid-alert, so severity is
        plain text and the colour lives in the HTML dashboard instead.
        """
        summary = TieredExplanation().tier1_summary(_alert())
        summary.encode("cp1252")   # raises UnicodeEncodeError if this regresses
        assert not any(ord(ch) > 0x2022 for ch in summary)


class TestTier2:
    """Drill-down payload: both methods, ranked, with an agreement signal."""

    def test_contains_both_attribution_methods_and_the_global_ranking(self):
        alert = _alert()
        tier2 = TieredExplanation().tier2_full(alert)

        for key in ("label", "confidence", "severity", "recommended_action",
                    "shap_values", "deeplift_values", "shap_ranked",
                    "deeplift_ranked", "method_agreement"):
            assert key in tier2

        assert [row["feature"] for row in tier2["shap_ranked"]] == \
            sorted(FEATURES, key=lambda f: -abs(SHAP[FEATURES.index(f)]))
        assert tier2["shap_ranked"][0]["feature"] == "syn_flag_count"
        assert tier2["shap_ranked"][0]["value"] == pytest.approx(0.82)

    def test_raw_vectors_are_excluded_from_the_alert_copy(self):
        tier2 = TieredExplanation().tier2_full(_alert())
        assert "shap_values" not in tier2["alert"]

    def test_method_agreement_scores_top_k_overlap(self):
        agreement = TieredExplanation().tier2_full(_alert())["method_agreement"]
        assert agreement["top_k"] == 3
        assert set(agreement["shared_features"]) == {
            "syn_flag_count", "flow_duration", "total_fwd_packets"}
        assert agreement["jaccard"] == pytest.approx(1.0)

    def test_disagreement_lowers_the_jaccard_score(self):
        opposing = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 9.0])
        alert = _alert(deeplift_values=opposing.tolist())
        agreement = TieredExplanation().tier2_full(alert)["method_agreement"]
        assert agreement["jaccard"] < 1.0
        assert agreement["shared_features"] != agreement["shap_top"]

    def test_missing_attribution_returns_empty_rankings(self):
        tier2 = TieredExplanation().tier2_full({"label": "DoS"})
        assert tier2["shap_ranked"] == [] and tier2["deeplift_ranked"] == []
        assert tier2["method_agreement"]["jaccard"] is None


class TestBuildAlert:
    """The bridge from a numeric attribution to an analyst-facing object."""

    def test_derives_severity_action_and_tier1_from_the_label(self):
        alert = build_alert("BruteForce", 0.88, FEATURES, SHAP,
                            source_ip="10.0.0.7", alert_id="A-1")
        explainer = TieredExplanation()

        assert alert["alert_id"] == "A-1"
        assert alert["severity"] == explainer.severity_for("BruteForce")
        assert alert["recommended_action"] == explainer.action_for("BruteForce")
        assert len(alert["top_features"]) == 3
        assert alert["top_features"][0]["feature"] == "syn_flag_count"
        assert alert["shap_values"] == pytest.approx(SHAP.tolist())

    def test_tier1_and_tier2_read_the_same_numbers(self):
        alert = build_alert("DoS", 0.95, FEATURES, SHAP, values=SHAP,
                            baseline=np.zeros(len(FEATURES)),
                            deeplift_values=DEEPLIFT)
        explainer = TieredExplanation()

        tier1 = explainer.tier1_summary(alert)
        tier2 = explainer.tier2_full(alert)
        assert tier1.startswith(f"{alert['severity']} CONFIDENCE ALERT")
        assert tier2["confidence"] == pytest.approx(alert["confidence"])
        assert tier2["recommended_action"] == alert["recommended_action"]

    def test_values_and_baselines_flow_into_the_descriptions(self):
        alert = build_alert("DoS", 0.9, FEATURES, SHAP,
                            values=np.array([12.0] + [0.0] * 5),
                            baseline=np.array([1.0] + [0.0] * 5))
        description = alert["top_features"][0]["description"]
        assert "syn_flag_count" in description
        assert "12.0x baseline" in description

    def test_extra_keys_override_the_defaults(self):
        alert = build_alert("DoS", 0.9, FEATURES, SHAP,
                            extra={"session_id": "sess-3"})
        assert alert["session_id"] == "sess-3"

    def test_feature_description_states_direction_and_ratio(self):
        text = feature_description("flow_duration", -0.4, value=0.01,
                                   baseline=2.0, unit="s")
        assert text.startswith("flow_duration: lowered risk")
        # A 200x *drop* must not be rounded into "0.0x", which reads like no
        # change at all.
        assert "0.005x baseline" in text
        assert "(0.01 vs 2s)" in text
