"""
src/dashboard/store.py — Data access for the Phase 4 dashboard.

The dashboard is a *viewer*: it renders the artifacts written by
``scripts/run_phase4.py`` rather than re-running detectors.  Two consequences
are deliberate:

* It starts in under a second and cannot perturb a detection run.
* When no report exists yet (fresh clone, CI) it falls back to
  :func:`demo_report`, so the layout can be reviewed before the experiment has
  been executed.  The demo state is clearly labelled as such by the UI.

Everything read here is JSON/CSV on disk, and every reader tolerates a missing
or malformed file: a monitoring screen must degrade to "no data" rather than
raise.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = _PROJECT_ROOT / "results" / "phase4" / "phase4_report.json"
DRIFT_DIR = _PROJECT_ROOT / "results" / "phase3" / "drift"

__all__ = ["REPORT_PATH", "alerts_frame", "bundle", "demo_report",
           "load_drift", "load_report", "severity_color"]


def bundle(report: dict, name: str) -> dict:
    """
    Fetch one bundle (``dashboard``, ``response``, ``faithfulness`` …) from a
    report.

    The measured report nests every bundle under ``components`` because that is
    the per-component execution log the phase gates read, while the demo report
    (which defines the layout the UI was written against) keeps them at the top
    level.  Reading both is the viewer's job, so the reader — not the two
    writers — carries that difference.
    """
    components = report.get("components")
    if isinstance(components, dict):
        nested = components.get(name)
        if isinstance(nested, dict) and nested:
            return nested
    top_level = report.get(name)
    return top_level if isinstance(top_level, dict) else {}


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None
    return None


def load_report(path: Optional[Path] = None) -> tuple[dict, bool]:
    """
    Load the Phase 4 report.

    Returns ``(report, is_demo)``.  ``is_demo`` is True when the fallback
    synthetic report is being served because no real report exists yet.
    """
    report = _safe_read_json(Path(path) if path else REPORT_PATH)
    if report:
        return report, False
    return demo_report(), True


def load_drift() -> dict:
    """Drift panel data, reusing the Phase 3 monitors when present."""
    summary = _safe_read_json(DRIFT_DIR / "summary.json")
    alerts_frame = None
    alerts_path = DRIFT_DIR / "alerts.csv"
    try:
        if alerts_path.is_file():
            alerts_frame = pd.read_csv(alerts_path)
    except Exception:
        alerts_frame = None

    payload = {"summary": summary or {}, "alerts": []}
    if alerts_frame is not None and not alerts_frame.empty:
        payload["alerts"] = alerts_frame.tail(100).to_dict(orient="records")
    return payload


def severity_color(severity: str) -> str:
    """Hex colour used for severity badges (red/amber/green)."""
    return {"HIGH": "#d9534f", "MEDIUM": "#f0ad4e",
            "LOW": "#5cb85c"}.get(str(severity).upper(), "#777777")


def _demo_alert(alert_id: str, label: str, confidence: float,
                feature_names: list[str], top_features: list[dict],
                source_ip: str) -> dict:
    from src.explainability import TieredExplanation

    explainer = TieredExplanation()
    alert = {
        "alert_id": alert_id,
        "label": label,
        "confidence": confidence,
        "severity": explainer.severity_for(label),
        "recommended_action": explainer.action_for(label),
        "source_ip": source_ip,
        "feature_names": feature_names,
        "top_features": top_features,
        "shap_values": [float(f.get("contribution", 0.0)) for f in top_features]
                       + [0.0] * (len(feature_names) - len(top_features)),
        "deeplift_values": [float(f.get("contribution", 0.0)) * 0.9
                            for f in top_features]
                           + [0.0] * (len(feature_names) - len(top_features)),
    }
    alert["tier1"] = explainer.tier1_summary(alert)
    return alert


def demo_report() -> dict:
    """
    Synthetic report used when ``results/phase4/phase4_report.json`` is absent.

    Clearly flagged with ``"is_demo": true``; the UI shows a warning banner for
    it so demo data can never be mistaken for a measurement.
    """
    feature_names = ["syn_flag_count", "flow_duration", "total_fwd_packets",
                     "flow_bytes_per_sec", "flow_iat_std", "rst_flag_count",
                     "fwd_packet_length_mean", "down_up_ratio"]
    template = [
        ("DoS", 0.972, [("syn_flag_count", "12x above baseline"),
                        ("flow_duration", "0.003s vs 2.4s baseline"),
                        ("total_fwd_packets", "1 packet per direction")]),
        ("PortScan", 0.934, [("rst_flag_count", "1 per probe, 65k ports"),
                             ("flow_duration", "0.001s per flow"),
                             ("fwd_packet_length_mean", "40 bytes (SYN only)")]),
        ("BruteForce", 0.881, [("flow_iat_std", "0.4s uniform retry cadence"),
                               ("total_fwd_packets", "12 credential attempts"),
                               ("down_up_ratio", "0.9 (echo-like)")]),
        ("WebAttack", 0.764, [("flow_bytes_per_sec", "18 kB/s payload burst"),
                              ("down_up_ratio", "4.1 asymmetric"),
                              ("fwd_packet_length_mean", "1420 bytes")]),
        ("Botnet", 0.958, [("flow_duration", "30s periodic beacon"),
                           ("flow_iat_std", "0.02s (highly regular)"),
                           ("flow_bytes_per_sec", "1.1 kB/s steady")]),
        ("Benign", 0.512, [("fwd_packet_length_mean", "812 bytes"),
                           ("flow_duration", "4.2s interactive session"),
                           ("down_up_ratio", "1.3 balanced")]),
    ]
    alerts = [
        _demo_alert(f"DEMO-{i:03d}", label, conf, feature_names,
                    [{"feature": name, "description": text,
                      "contribution": 0.4 - 0.1 * i, "direction": "increases risk"}
                     for name, text in feats],
                    f"10.0.0.{5 + i}")
        for i, (label, conf, feats) in enumerate(template)
    ]

    timeline = [
        {"offset_s": i * 30, "benign": 40 + (i % 5) * 6,
         "attack": 12 if i % 3 else 0}
        for i in range(40)
    ]

    return {
        "is_demo": True,
        "note": ("Synthetic placeholder shown because no phase 4 report was "
                 "found. Run `python scripts/run_phase4.py` to populate it."),
        "data_provenance": "DEMO DATA — not a measurement.",
        "dashboard": {"alerts": alerts, "timeline": timeline},
        "components": {},
        "response": {"n_replayed": 0, "action_counts": {}, "log": [],
                     "rollbacks": [], "burst_test": {}},
        "faithfulness": {},
        "drift": None,
    }


def alerts_frame(report: dict) -> pd.DataFrame:
    """Tabular view of the alert feed for ``st.dataframe``."""
    alerts = bundle(report, "dashboard").get("alerts") or []
    rows = []
    for alert in alerts:
        rows.append({
            "alert_id": alert.get("alert_id"),
            "label": alert.get("label"),
            "confidence": round(float(alert.get("confidence", 0.0)), 4),
            "severity": alert.get("severity"),
            "source_ip": alert.get("source_ip"),
            "session_id": alert.get("session_id"),
            "action": alert.get("action") or alert.get("recommended_action"),
        })
    return pd.DataFrame(rows)
