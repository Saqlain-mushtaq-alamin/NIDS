"""
src/dashboard/app.py — Live SOC dashboard for explainability & response (FR8).

Launch::

    streamlit run src/dashboard/app.py --server.headless=true

Panels (PHASE_4 section 4.6):

1. **Alert feed** — every explained alert, severity-coded, with headline counts.
2. **Confidence** — distribution of prediction confidence overall and per class.
3. **Explanations** — Tier-1 SOC triage summary for the selected alert and the
   Tier-2 drill-down (SHAP vs DeepLIFT attribution charts).
4. **Response actions** — what the response engine did, plus per-IP rollback.
5. **Drift** — concept-drift monitor status from the Phase 3 detectors.
6. **Traffic timeline** — flows per bucket, split benign vs attack.

The dashboard renders ``results/phase4/phase4_report.json`` written by
``scripts/run_phase4.py``; it never runs a detector itself, so it cannot
perturb a measurement. With no report present it falls back to clearly
labelled demo data so the layout is reviewable out of the box.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.dashboard.components.panels import (
    render_alert_feed,
    render_confidence_distribution,
    render_drift_monitoring,
    render_explanations,
    render_response_actions,
    render_traffic_timeline,
)
from src.dashboard.store import bundle, load_drift, load_report

st.set_page_config(page_title="NIDS — Explainability & Response",
                   page_icon=":shield:", layout="wide")

st.title("NIDS — Explainability & Automated Response")
report, is_demo = load_report()

if is_demo:
    st.warning(
        "**Demo data.** No phase 4 report found, so this page shows a "
        "synthetic placeholder. Run `python scripts/run_phase4.py` to populate "
        "it with measured results."
    )
else:
    st.caption(report.get("data_provenance", ""))

    generated = report.get("generated_at")
    if generated:
        st.caption(f"Report generated: {generated}")

fa = bundle(report, "faithfulness")
if fa.get("verdict"):
    st.info(f"SHAP vs DeepLIFT faithfulness: {fa['verdict']}")

tabs = st.tabs([
    "Alert feed",
    "Confidence distribution",
    "Explanations (Tier 1 / Tier 2)",
    "Response actions",
    "Drift monitoring",
    "Traffic timeline",
])

with tabs[0]:
    render_alert_feed(report)
with tabs[1]:
    render_confidence_distribution(report)
with tabs[2]:
    render_explanations(report)
with tabs[3]:
    render_response_actions(report)
with tabs[4]:
    render_drift_monitoring(load_drift())
with tabs[5]:
    render_traffic_timeline(report)

st.divider()
st.caption(
    "Phase 4 — explainability (SHAP / DeepLIFT), faithfulness benchmarking, "
    "tiered triage explanations, and the tiered response engine with rollback."
)
