"""
src/explainability/tiered_explanation.py — Two-tier analyst-facing explanations.

Two audiences, two depths (PHASE_4 section 4.4, requirements FR6/FR7/NFR7):

**Tier 1 — SOC triage.**  A 2–3 sentence summary an analyst reads while the
alert is still live: what it probably is, how confident the model is, the three
features that drove the verdict, and the recommended action.  It is generated
from string templates over the attribution data — no LLM, so it is deterministic
and testable.

**Tier 2 — full attribution.**  Every feature contribution from SHAP and
DeepLIFT side by side, the global importance ranking, and the raw values, for
the analyst who drills in or writes the post-incident report.

Severity and recommended actions are table-driven
(:attr:`TieredExplanation.SEVERITY_MAP`, :attr:`TieredExplanation.ACTION_MAP`)
so the mapping can be reviewed by a human without reading code.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.explainability.attribution import feature_description, rank_features

__all__ = ["TieredExplanation", "build_alert"]


class TieredExplanation:
    """Generate analyst-friendly tiered explanations for NIDS alerts."""

    SEVERITY_MAP = {
        "DoS": "HIGH", "DDoS": "HIGH", "Botnet": "HIGH",
        "BruteForce": "MEDIUM", "WebAttack": "MEDIUM",
        "PortScan": "LOW", "Infiltration": "HIGH",
        "Anomaly": "MEDIUM", "Benign": "LOW",
    }

    ACTION_MAP = {
        "DoS": "BLOCK source IP and rate-limit subnet",
        "DDoS": "BLOCK source IP and escalate to Tier-2",
        "BruteForce": "Lock targeted account, BLOCK source after 5 attempts",
        "PortScan": "Log and monitor — escalate if followed by exploitation",
        "WebAttack": "BLOCK request pattern, review WAF rules",
        "Botnet": "Isolate affected host, initiate incident response",
        "Infiltration": "Isolate host, escalate to Tier-2 immediately",
        "Anomaly": "Review manually — autoencoder flagged unusual pattern",
        "Benign": "No action — flow scored as benign",
    }

    DEFAULT_ACTION = "Review manually"
    MAX_DESCRIPTION_CHARS = 90
    # Severity is rendered as text, not as colour emoji. The phase plan's
    # example used emoji, but this project runs on Windows consoles whose
    # default codec (cp1252 here) raises UnicodeEncodeError on them, which
    # would take the triage console down mid-alert. The colour coding belongs
    # to the dashboard, which renders HTML and can use CSS safely.

    # -- lookups -----------------------------------------------------------
    def severity_for(self, label: str) -> str:
        """Severity tier for a taxonomy label (defaults to MEDIUM)."""
        return self.SEVERITY_MAP.get(str(label), "MEDIUM")

    def action_for(self, label: str) -> str:
        """Recommended response for a taxonomy label."""
        return self.ACTION_MAP.get(str(label), self.DEFAULT_ACTION)

    # -- Tier 1 ------------------------------------------------------------
    @staticmethod
    def _indicator_line(feature: dict) -> str:
        """Render one ``top_features`` entry as a single triage line."""
        name = str(feature.get("feature") or "")
        description = feature.get("description")
        if not description:
            description = feature_description(
                name or "unknown",
                float(feature.get("contribution", 0.0)),
                value=feature.get("value"),
                baseline=feature.get("baseline"),
            )
        description = str(description)
        # The feature name must always be visible: an analyst who sees
        # "12x above baseline" needs to know *which* field that refers to.
        if name and not description.startswith(name):
            description = f"{name}: {description}"
        if len(description) > TieredExplanation.MAX_DESCRIPTION_CHARS:
            description = description[:TieredExplanation.MAX_DESCRIPTION_CHARS - 3] + "..."
        return description

    def tier1_summary(self, alert: dict) -> str:
        """
        Short, actionable Tier-1 triage summary.

        Required alert keys: ``label``, ``confidence``, ``top_features``.
        Missing keys degrade to safe placeholders rather than raising — a
        triage console must render *something* for a partial alert.
        """
        label = str(alert.get("label", "Anomaly"))
        confidence = float(alert.get("confidence", 0.0) or 0.0)
        top_features = list(alert.get("top_features") or [])
        severity = alert.get("severity") or self.severity_for(label)
        action = alert.get("recommended_action") or self.action_for(label)

        header = (
            f"{severity} CONFIDENCE ALERT — {label} Detected "
            f"({confidence:.1%} confidence)"
        )

        if top_features:
            lines = "\n".join(
                f"  {i + 1}. {self._indicator_line(feature)}"
                for i, feature in enumerate(top_features[:3])
            )
            indicators = f"Key indicators:\n{lines}"
        else:
            indicators = "Key indicators: no attribution available for this alert."

        source_ip = alert.get("source_ip")
        context = f" (source {source_ip})" if source_ip else ""

        return (
            f"{header}{context}\n\n"
            f"{indicators}\n\n"
            f"Recommended Action: {action}"
        )

    # -- Tier 2 ------------------------------------------------------------
    def tier2_full(self, alert: dict) -> dict:
        """
        Complete attribution payload for Tier-2 drill-down.

        Everything Tier 1 compressed away is here: both attribution methods,
        the global ranking, the per-feature values, and the metadata needed to
        reproduce the decision.
        """
        shap_values = alert.get("shap_values")
        deeplift_values = alert.get("deeplift_values")
        feature_names = list(alert.get("feature_names") or [])

        def _rows(values, limit: Optional[int] = None) -> list[dict]:
            if values is None or not feature_names:
                return []
            array = np.asarray(values, dtype=float).reshape(-1)
            if array.shape[0] != len(feature_names):
                return []
            ranked = rank_features(array, feature_names, top_k=limit or len(feature_names))
            values_by_name = {name: float(v) for name, v in zip(feature_names, array)}
            for row in ranked:
                row["value"] = values_by_name.get(row["feature"])
                row["description"] = self._indicator_line(row)
            return ranked

        ranked_shap = _rows(shap_values)
        deeplift_ranked = _rows(deeplift_values)

        return {
            "alert": {k: v for k, v in alert.items()
                      if k not in {"shap_values", "deeplift_values"}},
            "label": alert.get("label", "Anomaly"),
            "confidence": float(alert.get("confidence", 0.0) or 0.0),
            "severity": alert.get("severity") or self.severity_for(alert.get("label", "Anomaly")),
            "recommended_action": alert.get("recommended_action")
                                  or self.action_for(alert.get("label", "Anomaly")),
            "shap_values": shap_values,
            "deeplift_values": deeplift_values,
            "shap_ranked": ranked_shap,
            "deeplift_ranked": deeplift_ranked,
            "method_agreement": self._agreement(ranked_shap, deeplift_ranked),
            "feature_importance_plot_data": alert.get("feature_importances"),
            "similar_historical_alerts": alert.get("similar_historical_alerts", []),
        }

    @staticmethod
    def _agreement(shap_ranked: Sequence[dict],
                   deeplift_ranked: Sequence[dict],
                   top_k: int = 3) -> dict:
        """Top-K overlap between the two methods — a cheap sanity signal."""
        a = [row["feature"] for row in shap_ranked[:top_k]]
        b = [row["feature"] for row in deeplift_ranked[:top_k]]
        if not a or not b:
            return {"top_k": top_k, "shared_features": [], "jaccard": None}
        shared = sorted(set(a) & set(b))
        union = set(a) | set(b)
        return {
            "top_k": top_k,
            "shap_top": a,
            "deeplift_top": b,
            "shared_features": shared,
            "jaccard": len(shared) / len(union) if union else 0.0,
        }


def build_alert(label: str, confidence: float,
                feature_names: Sequence[str],
                shap_values: np.ndarray,
                values: Optional[np.ndarray] = None,
                baseline: Optional[np.ndarray] = None,
                deeplift_values: Optional[np.ndarray] = None,
                source_ip: Optional[str] = None,
                alert_id: Optional[str] = None,
                top_k: int = 3,
                severity: Optional[str] = None,
                extra: Optional[dict] = None) -> dict:
    """
    Assemble one alert dict from raw attributions.

    This is the single place where a numeric attribution vector becomes an
    analyst-facing object, so the descriptions Tier 1 renders are derived from
    the same numbers Tier 2 shows — the two tiers can never disagree.
    """
    explainer_stub = TieredExplanation()
    ranked = rank_features(shap_values, feature_names, top_k=top_k)

    values_by_feature = dict(zip(feature_names, values)) if values is not None else {}
    baseline_by_feature = dict(zip(feature_names, baseline)) if baseline is not None else {}

    top_features = []
    for row in ranked:
        row = dict(row)
        row["value"] = values_by_feature.get(row["feature"])
        row["baseline"] = baseline_by_feature.get(row["feature"])
        row["description"] = explainer_stub._indicator_line(row)
        top_features.append(row)

    alert = {
        "alert_id": alert_id,
        "label": str(label),
        "confidence": float(confidence),
        "severity": severity or explainer_stub.severity_for(label),
        "recommended_action": explainer_stub.action_for(label),
        "source_ip": source_ip,
        "feature_names": list(feature_names),
        "shap_values": [float(v) for v in np.asarray(shap_values, dtype=float).reshape(-1)],
        "top_features": top_features,
    }
    if deeplift_values is not None:
        alert["deeplift_values"] = [float(v) for v in
                                    np.asarray(deeplift_values, dtype=float).reshape(-1)]
    if extra:
        alert.update(extra)
    return alert
