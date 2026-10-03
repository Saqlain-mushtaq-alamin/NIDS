"""
tests/test_phase4_pipeline.py — End-to-end smoke test for the Phase 4 driver.

``scripts/run_phase4.py`` is the artefact the plan asks for (section 4.7): one
command that explains alerts, measures faithfulness, builds tiered
explanations, replays them through the response engine and writes the report
the dashboard reads.  This test runs that whole chain in quick mode against a
temporary output directory — pointing it at ``results/phase4`` would overwrite
the measured figures the report is supposed to contain.

The run is expensive (it trains both classifiers), so it happens once for the
whole module.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

from scripts.run_phase4 import main as run_phase4
from src.dashboard.store import bundle

N_ALERTS = 10


@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    """(report, output_dir) for a single quick run of the phase 4 driver."""
    output_dir = tmp_path_factory.mktemp("phase4")
    started = time.perf_counter()
    code = run_phase4(["--quick", "--n-alerts", str(N_ALERTS),
                       "--output-dir", str(output_dir)])
    elapsed = time.perf_counter() - started

    assert code == 0, f"phase 4 driver failed after {elapsed:.1f}s"
    report = json.loads((output_dir / "phase4_report.json").read_text(
        encoding="utf-8"))
    report["_elapsed_s"] = elapsed
    return report, output_dir


def test_writes_every_artifact(quick_run):
    _, output_dir = quick_run

    for name in ("phase4_report.json", "alerts.csv", "faithfulness_scores.csv",
                 "tier1_summaries.md", "response_log.csv"):
        path = output_dir / name
        assert path.is_file() and path.stat().st_size > 0, name


def test_every_component_ran_without_failure(quick_run):
    report, _ = quick_run

    assert report["failures"] == {}, report["failures"]
    assert report["quick_mode"] is True
    assert sorted(report["components"]) == [
        "dashboard", "deeplift", "faithfulness", "response", "shap", "tiered"]


def test_explainability_results(quick_run):
    report, _ = quick_run

    shap = report["components"]["shap"]
    assert shap["n_alerts"] == N_ALERTS
    assert shap["benchmark"]["mean_ms_per_alert"] > 0
    assert shap["global_top_features"]

    deeplift = report["components"]["deeplift"]
    assert deeplift["n_alerts"] == N_ALERTS
    assert len(deeplift["target_classes"]) == N_ALERTS

    faithfulness = report["components"]["faithfulness"]
    assert len(faithfulness["shap_scores"]) == N_ALERTS
    assert len(faithfulness["deeplift_scores"]) == N_ALERTS
    assert faithfulness["verdict"]
    assert 0.0 <= faithfulness["shap_mean_deletion_auc"] <= 1.0
    assert 0.0 <= faithfulness["deeplift_mean_deletion_auc"] <= 1.0


def test_tiered_explanations_and_response(quick_run):
    report, _ = quick_run

    tiered = report["components"]["tiered"]
    assert tiered["n_alerts"] == N_ALERTS
    assert tiered["coverage"]["all_covered"] is True
    assert tiered["tier1_max_chars"] < 500
    assert "method_agreement" in tiered["tier2_keys"]

    response = report["components"]["response"]
    assert set(response["tiers_exercised"]) == {
        "LOG_ONLY", "RATE_LIMIT", "BLOCK"}
    assert response["stats"]["dry_run"] is True
    assert response["rollback_demo"]["succeeded"] is True
    assert response["auto_unblock_demo"]["succeeded"] is True


def test_gates(quick_run):
    gates = quick_run[0]["gates"]

    assert gates["G4.3_tier1_attack_type_coverage"]["pass"] is True
    assert gates["G4.4_tier1_concise"]["pass"] is True
    assert gates["G4.5_response_tiers_exercised"]["pass"] is True
    assert gates["G4.6_rate_limiter_bounds_blocks_NFR10"]["pass"] is True
    assert gates["G4.7_rollback_and_ttl_unblock"]["pass"] is True
    # Volume-dependent gates are deliberately unanswered in quick mode.
    assert gates["G4.1_tree_shap_latency"]["pass"] is None
    assert gates["G4.2_faithfulness_sample_size"]["pass"] is None


def test_report_layout_is_the_one_the_dashboard_reads(quick_run):
    report, output_dir = quick_run

    alerts = bundle(report, "dashboard")["alerts"]
    assert len(alerts) == N_ALERTS

    # Tier 2 needs both attribution vectors side by side.
    assert alerts[0]["shap_values"]
    assert alerts[0]["deeplift_values"]
    assert alerts[0]["tier1"]
    assert alerts[0]["action"] in {"LOG_ONLY", "RATE_LIMIT", "BLOCK"}

    response = bundle(report, "response")
    assert response["log"]
    assert bundle(report, "faithfulness")["verdict"]

    frame = pd.read_csv(output_dir / "alerts.csv")
    assert len(frame) == N_ALERTS
    assert frame["alert_id"].notna().all()
    assert frame["top_feature_1"].notna().all()
