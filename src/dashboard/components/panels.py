"""
src/dashboard/components/panels.py — Individual dashboard panels.

Each function takes the Phase 4 report (and, where relevant, the drift payload)
and renders exactly one panel.  Keeping them separate means the panel set can
be unit-tested without a browser and each panel fails independently: a missing
section in the report yields ``st.info`` for that panel, not a blank screen.
"""
from __future__ import annotations

import inspect
from typing import Optional

import pandas as pd
import streamlit as st

from src.dashboard.store import alerts_frame, bundle, severity_color
from src.explainability import TieredExplanation
from src.response import ResponseEngine

__all__ = [
    "render_alert_feed",
    "render_confidence_distribution",
    "render_drift_monitoring",
    "render_explanations",
    "render_response_actions",
    "render_traffic_timeline",
]

_SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
_STRETCH_CACHE: dict = {}


def _stretch(target) -> dict:
    """
    Keyword arguments that make a chart/table span the page width.

    Streamlit renamed ``use_container_width=True`` to ``width="stretch"``;
    the old spelling is deprecated (removal announced for 1.6x) while the new
    one does not exist before 1.49.  ``requirements.txt`` allows 1.28+, so
    probe the installed signature once instead of pinning the dashboard to a
    single release.
    """
    key = getattr(target, "__name__", str(target))
    if key not in _STRETCH_CACHE:
        params = inspect.signature(target).parameters
        if "width" in params:
            _STRETCH_CACHE[key] = {"width": "stretch"}
        elif "use_container_width" in params:
            _STRETCH_CACHE[key] = {"use_container_width": True}
        else:  # pragma: no cover - future streamlit
            _STRETCH_CACHE[key] = {}
    return dict(_STRETCH_CACHE[key])



def _sorted_alerts(alerts: list[dict]) -> list[dict]:
    return sorted(
        alerts,
        key=lambda a: (_SEVERITY_ORDER.get(a.get("severity"), 3),
                       -float(a.get("confidence") or 0.0)),
    )


def render_alert_feed(report: dict, limit: int = 50) -> None:
    """Scrolling alert feed with severity colour-coding and headline counts."""
    alerts = _sorted_alerts(bundle(report, "dashboard").get("alerts") or [])
    if not alerts:
        st.info("No alerts in this run.")
        return

    high = sum(1 for a in alerts if a.get("severity") == "HIGH")
    medium = sum(1 for a in alerts if a.get("severity") == "MEDIUM")
    low = sum(1 for a in alerts if a.get("severity") == "LOW")
    mean_conf = sum(float(a.get("confidence") or 0.0) for a in alerts) / len(alerts)

    left, middle, right, extra = st.columns(4)
    left.metric("Alerts explained", len(alerts))
    middle.metric("HIGH severity", high)
    right.metric("MEDIUM / LOW", f"{medium} / {low}")
    extra.metric("Mean confidence", f"{mean_conf:.1%}")

    st.dataframe(alerts_frame(report).head(limit), **_stretch(st.dataframe),
                 height=320)

    st.markdown("**Highest-priority alerts**")
    for alert in alerts[:3]:
        colour = severity_color(alert.get("severity"))
        badge = (f":red[{alert.get('severity')}]" if alert.get("severity") == "HIGH"
                 else f":orange[{alert.get('severity')}]"
                 if alert.get("severity") == "MEDIUM"
                 else f":green[{alert.get('severity')}]")
        st.markdown(
            f"<span style='color:{colour};font-weight:600'>"
            f"{badge} — {alert.get('label')} "
            f"({float(alert.get('confidence') or 0):.1%})</span>",
            unsafe_allow_html=True,
        )
        st.text(alert.get("tier1") or TieredExplanation().tier1_summary(alert))
        st.divider()


def render_confidence_distribution(report: dict) -> None:
    """Histogram of prediction confidences, split by predicted class."""
    import plotly.express as px

    alerts = bundle(report, "dashboard").get("alerts") or []
    if not alerts:
        st.info("No confidence data in this run.")
        return

    frame = pd.DataFrame({
        "confidence": [float(a.get("confidence") or 0.0) for a in alerts],
        "label": [a.get("label") for a in alerts],
        "severity": [a.get("severity") for a in alerts],
    })

    fig = px.histogram(frame, x="confidence", color="severity", nbins=20,
                       color_discrete_map={"HIGH": "#d9534f",
                                           "MEDIUM": "#f0ad4e",
                                           "LOW": "#5cb85c"},
                       title="Alert confidence distribution")
    fig.update_layout(bargap=0.1, xaxis_title="Confidence",
                      yaxis_title="Alerts", legend_title="Severity")
    st.plotly_chart(fig, **_stretch(st.plotly_chart))

    fig_box = px.box(frame, x="label", y="confidence", color="severity",
                     color_discrete_map={"HIGH": "#d9534f",
                                         "MEDIUM": "#f0ad4e",
                                         "LOW": "#5cb85c"},
                     title="Confidence by predicted class")
    fig_box.update_layout(xaxis_title="Predicted class", yaxis_title="Confidence")
    st.plotly_chart(fig_box, **_stretch(st.plotly_chart))


def _contribution_chart(values: list, feature_names: list, title: str,
                        top_k: int = 12):
    import plotly.express as px

    pairs = list(zip(feature_names, values))
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    pairs = pairs[:top_k]
    frame = pd.DataFrame({
        "feature": [p[0] for p in pairs],
        "contribution": [float(p[1]) for p in pairs],
    })
    frame["direction"] = frame["contribution"].apply(
        lambda v: "raises risk" if v > 0 else "lowers risk")
    fig = px.bar(frame, x="contribution", y="feature", color="direction",
                 orientation="h", title=title,
                 color_discrete_map={"raises risk": "#d9534f",
                                     "lowers risk": "#5cb85c"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"},
                      xaxis_title="Attribution (signed)")
    return fig


def render_explanations(report: dict) -> None:
    """Tier-1 summary for the selected alert, with the Tier-2 drill-down."""
    alerts = bundle(report, "dashboard").get("alerts") or []
    if not alerts:
        st.info("No explanations in this run.")
        return

    options = [f"{a.get('alert_id')} — {a.get('label')} "
               f"({float(a.get('confidence') or 0):.1%})" for a in alerts]
    choice = st.selectbox("Alert", options)
    index = options.index(choice)
    alert = alerts[index]

    explainer = TieredExplanation()
    tier1 = alert.get("tier1") or explainer.tier1_summary(alert)
    st.subheader("Tier 1 — SOC triage summary")
    st.text_area("Tier-1 summary", tier1, height=180, disabled=True)
    st.caption(f"Severity: {alert.get('severity')} · "
               f"Recommended action: {alert.get('recommended_action')} · "
               f"Source: {alert.get('source_ip')} · "
               f"Session: {alert.get('session_id')}")

    st.subheader("Tier 2 — full feature attribution")
    tier2 = explainer.tier2_full(alert)
    feature_names = list(alert.get("feature_names") or [])
    shap_values = alert.get("shap_values")
    deeplift_values = alert.get("deeplift_values")

    if shap_values and feature_names:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(_contribution_chart(shap_values, feature_names,
                                                "SHAP attribution"),
                            **_stretch(st.plotly_chart))
        if deeplift_values:
            with right:
                st.plotly_chart(_contribution_chart(deeplift_values, feature_names,
                                                    "DeepLIFT attribution"),
                                **_stretch(st.plotly_chart))
        agreement = tier2.get("method_agreement") or {}
        if agreement.get("jaccard") is not None:
            st.caption(
                f"Top-{agreement.get('top_k')} agreement between SHAP and "
                f"DeepLIFT: Jaccard {agreement['jaccard']:.2f} "
                f"(shared: {', '.join(agreement.get('shared_features') or []) or 'none'})"
            )
    else:
        st.info("This alert carries no per-feature attributions.")

    with st.expander("Ranked contributions (all features)"):
        rows = tier2.get("shap_ranked") or []
        if rows:
            st.dataframe(pd.DataFrame(rows), **_stretch(st.dataframe))
        else:
            st.caption("No ranked contributions available.")


def render_response_actions(report: dict) -> None:
    """Response action log with per-alert rollback controls (FR7)."""
    response = bundle(report, "response")
    stats = response.get("stats") or {}
    mode = "DRY-RUN (no firewall rules were applied)" if stats.get(
        "dry_run", True) else "LIVE"
    st.caption(
        f"Engine mode: {mode} · "
        f"max {stats.get('max_blocks_per_minute', '-')} blocks/minute "
        f"(NFR10) · auto-unblock after "
        f"{stats.get('auto_unblock_seconds', '-')}s"
    )

    counts = stats.get("action_counts") or response.get("action_counts") or {}
    if counts:
        cols = st.columns(max(len(counts), 1))
        for column, (name, count) in zip(cols, counts.items()):
            column.metric(name, count)

    log = response.get("log") or []
    if not log:
        st.info("No response actions were recorded in this run.")
        return

    frame = pd.DataFrame([{
        "time": pd.to_datetime(entry.get("timestamp"), unit="s",
                               errors="coerce"),
        "action": entry.get("action"),
        "source_ip": entry.get("source_ip"),
        "label": entry.get("label"),
        "confidence": round(float(entry.get("confidence") or 0.0), 4),
        "reason": entry.get("reason"),
        "command": entry.get("command"),
    } for entry in log])
    st.dataframe(frame, **_stretch(st.dataframe), height=300)

    st.subheader("Rollback")
    rolled_back = st.session_state.setdefault("rolled_back_ips", set())
    # The same host can be blocked by several alerts (one noisy session is
    # enough), and a widget key must still be unique per rendered button —
    # so the list is de-duplicated and the key carries the position.
    blocked_ips = list(dict.fromkeys(
        entry.get("source_ip") for entry in log
        if entry.get("action") == "BLOCK"
        and entry.get("source_ip")
        and entry.get("source_ip") not in rolled_back))
    if not blocked_ips:
        st.caption("No active block is waiting for rollback.")
        return

    st.caption(
        "Rollback acts on this session's in-memory dry-run engine; the "
        "recorded run on disk stays untouched."
    )
    engine = st.session_state.setdefault("response_engine", ResponseEngine())
    for position, ip in enumerate(blocked_ips[:20]):
        left, right = st.columns([3, 1])
        left.write(ip)
        if right.button("Unblock", key=f"unblock_{position}_{ip}"):
            engine.rollback(ip)
            rolled_back.add(ip)
            st.session_state["rollback_notice"] = f"Rollback issued for {ip}"
    notice = st.session_state.get("rollback_notice")
    if notice:
        st.success(notice)

    st.subheader("Live engine state")
    st.json({"active_blocks": list(engine.active_blocks),
             "stats": engine.stats()})


def render_drift_monitoring(drift: dict) -> None:
    """Drift detector status, reusing the Phase 3 monitor output."""
    summary = drift.get("summary") or {}
    alerts = drift.get("alerts") or []
    if not summary and not alerts:
        st.info("No drift report found. Run `python scripts/run_drift_monitoring.py`.")
        return

    if summary:
        st.json(summary)
    if alerts:
        st.markdown(f"**Recent drift alerts ({len(alerts)})**")
        st.dataframe(pd.DataFrame(alerts), **_stretch(st.dataframe),
                     height=300)
    else:
        st.success("No drift alerts recorded.")


def render_traffic_timeline(report: dict, bucket_seconds: Optional[int] = None) -> None:
    """Flows per time bucket, split benign vs attack."""
    import plotly.express as px

    timeline = bundle(report, "dashboard").get("timeline") or []
    if not timeline:
        st.info("No traffic timeline in this run.")
        return

    frame = pd.DataFrame(timeline)
    frame["attack"] = frame.get("attack", 0)
    frame["benign"] = frame.get("benign", 0)
    if bucket_seconds is None:
        bucket_seconds = frame["offset_s"].diff().dropna().median() or 30

    long = frame.melt(id_vars="offset_s", value_vars=["benign", "attack"],
                      var_name="class", value_name="flows")
    fig = px.bar(long, x="offset_s", y="flows", color="class",
                 barmode="stack",
                 color_discrete_map={"benign": "#5cb85c", "attack": "#d9534f"},
                 title=f"Traffic volume ({bucket_seconds:g}s buckets)")
    fig.update_layout(xaxis_title="Seconds from capture start",
                      yaxis_title="Flows", legend_title="Class")
    st.plotly_chart(fig, **_stretch(st.plotly_chart))
