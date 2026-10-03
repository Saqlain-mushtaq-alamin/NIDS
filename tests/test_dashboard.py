"""
tests/test_dashboard.py — Live SOC dashboard (PHASE_4 section 4.6, requirement
FR8).

✅ Test 4.6 from the plan checks that the app exists and starts.  The rest
covers what actually breaks a monitoring screen: the report layout the viewer
reads (nested vs. demo), the empty-report path (a fresh clone has no
measurements), and the deprecation warnings that would otherwise flood the
console on every run.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.dashboard import store
from src.dashboard.store import bundle, demo_report, load_report, severity_color

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "dashboard" / "app.py"


class TestPlanTest46:
    """✅ Test 4.6 — Dashboard Validation (verbatim from the plan)."""

    def test_dashboard_components_exist(self):
        content = APP.read_text(encoding="utf-8")
        assert "st.title" in content
        assert "render_alert_feed" in content
        assert "render_explanations" in content
        assert "render_response_actions" in content

    def test_dashboard_launches_as_a_streamlit_app(self):
        exe = shutil.which("streamlit")
        if exe is None:
            pytest.skip("streamlit CLI is not on PATH")

        port = 8600 + os.getpid() % 300
        process = subprocess.Popen(
            [exe, "run", str(APP), "--server.headless=true",
             f"--server.port={port}", "--server.address=127.0.0.1"],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            time.sleep(8)
            assert process.poll() is None, "streamlit exited during startup"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover
                process.kill()


class TestReportReading:
    """The viewer reads two layouts: measured (nested) and demo (top level)."""

    def test_bundle_reads_the_measured_layout(self):
        report = {"components": {"dashboard": {"alerts": [1]},
                                 "response": {"log": [2]},
                                 "faithfulness": {"verdict": "shap"}}}
        assert bundle(report, "dashboard")["alerts"] == [1]
        assert bundle(report, "response")["log"] == [2]
        assert bundle(report, "faithfulness")["verdict"] == "shap"

    def test_bundle_reads_the_demo_layout(self):
        report = demo_report()
        assert bundle(report, "dashboard")["alerts"]
        assert bundle(report, "dashboard")["timeline"]
        assert isinstance(bundle(report, "response"), dict)

    def test_bundle_degrades_to_empty(self):
        assert bundle({}, "dashboard") == {}
        assert bundle({"components": None}, "response") == {}
        assert bundle({"components": {"dashboard": []}}, "dashboard") == {}

    def test_demo_report_is_clearly_labelled(self):
        report = demo_report()
        assert report["is_demo"] is True
        assert "DEMO DATA" in report["data_provenance"]
        assert "Run `python scripts/run_phase4.py`" in report["note"]

    def test_load_report_falls_back_to_demo(self, tmp_path):
        missing = tmp_path / "phase4_report.json"
        report, is_demo = load_report(missing)
        assert is_demo is True
        assert report["is_demo"] is True

        measured = {"components": {"dashboard": {"alerts": []}},
                    "generated_at": "2026-01-01T00:00:00+00:00"}
        missing.write_text(json.dumps(measured), encoding="utf-8")
        report, is_demo = load_report(missing)
        assert is_demo is False
        assert report["generated_at"].startswith("2026")

    def test_malformed_report_falls_back_instead_of_raising(self, tmp_path):
        broken = tmp_path / "phase4_report.json"
        broken.write_text("{not json", encoding="utf-8")
        report, is_demo = load_report(broken)
        assert is_demo is True
        assert report["components"] == {}

    @pytest.mark.parametrize("severity,colour", [
        ("HIGH", "#d9534f"), ("MEDIUM", "#f0ad4e"), ("LOW", "#5cb85c"),
        ("unknown", "#777777"),
    ])
    def test_severity_colours(self, severity, colour):
        assert severity_color(severity) == colour

    def test_alerts_frame_extracts_the_feed_columns(self):
        frame = store.alerts_frame(demo_report())
        assert not frame.empty
        for column in ("alert_id", "label", "confidence", "severity",
                       "source_ip", "action"):
            assert column in frame.columns


class TestAppRender:
    """The app must run headlessly, for every report shape it can be given."""

    def _run(self):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP), default_timeout=60)
        at.run()
        return at

    def test_tabs_match_the_plan(self):
        at = self._run()
        assert not at.exception, [str(e.value) for e in at.exception]
        assert [tab.label for tab in at.tabs] == [
            "Alert feed",
            "Confidence distribution",
            "Explanations (Tier 1 / Tier 2)",
            "Response actions",
            "Drift monitoring",
            "Traffic timeline",
        ]

    def test_charts_render_without_deprecation_warnings(self):
        at = self._run()
        assert not at.exception, [str(e.value) for e in at.exception]

        report, _ = load_report()
        n_alerts = len(bundle(report, "dashboard").get("alerts") or [])
        charts = len(at.get("plotly_chart"))
        if n_alerts:
            assert charts >= 4          # confidence (2), tier-2 (1-2), timeline
        else:
            assert any("No alerts" in w.value for w in at.info)

        warnings = [w.value for w in at.warning]
        assert not any("use_container_width" in w for w in warnings)
        assert not any("deprecat" in w.lower() for w in warnings)

    def test_headline_metrics_are_shown_for_a_populated_report(self):
        report, _ = load_report()
        if not bundle(report, "dashboard").get("alerts"):
            pytest.skip("no populated report on disk")

        at = self._run()
        assert not at.exception
        assert len(at.get("metric")) >= 4
        assert any("Alerts explained" in m.label for m in at.get("metric"))

    def test_empty_report_renders_info_instead_of_crashing(self, tmp_path,
                                                            monkeypatch):
        empty = tmp_path / "phase4_report.json"
        empty.write_text(json.dumps({
            "generated_at": "2026-01-01T00:00:00+00:00",
            "data_provenance": "synthetic lab captures",
            "components": {"dashboard": {"alerts": [], "timeline": []},
                           "response": {"log": [], "stats": {}},
                           "faithfulness": {}},
        }), encoding="utf-8")
        monkeypatch.setattr(store, "REPORT_PATH", empty)

        at = self._run()
        assert not at.exception, [str(e.value) for e in at.exception]
        messages = [w.value for w in at.info]
        assert any("No alerts" in m for m in messages)
        assert any("No response actions" in m for m in messages)
