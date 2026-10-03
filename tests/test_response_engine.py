"""
tests/test_response_engine.py — Tiered automated response (PHASE_4 section 4.5,
requirements FR7 / NFR10).

✅ Test 4.5 from the plan covers the three tiers, the per-minute block budget
and manual rollback.  The rest exercises the safety properties that only show
up over time or under abuse: TTL expiry (both polled and backgrounded), a
refresh not spending budget twice, dry-run never reaching the host firewall,
and configuration typos being rejected instead of silently disabling a limit.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from src.response import BLOCK, LOG_ONLY, RATE_LIMIT, ResponseEngine


def alert(confidence: float, severity: str, source_ip: str = "10.0.0.5",
          alert_id: str = "A-1") -> dict:
    return {"confidence": confidence, "severity": severity,
            "source_ip": source_ip, "alert_id": alert_id,
            "label": "DoS"}


class TestPlanTest45:
    """✅ Test 4.5 — Response Engine Validation (verbatim from the plan)."""

    def test_response_tiers(self):
        from src.response import ResponseEngine

        engine = ResponseEngine()
        assert engine.decide_action({"confidence": 0.95,
                                     "severity": "HIGH"}) == "BLOCK"
        assert engine.decide_action({"confidence": 0.95,
                                     "severity": "MEDIUM"}) == "RATE_LIMIT"
        assert engine.decide_action({"confidence": 0.6,
                                     "severity": "HIGH"}) == "LOG_ONLY"

    def test_rate_limit_prevents_self_dos(self):
        """NFR10: a false-positive burst must degrade, not lock out the network."""
        from src.response import ResponseEngine

        engine = ResponseEngine(max_blocks_per_minute=3, dry_run=True)
        results = [
            engine.execute({"confidence": 0.99, "severity": "HIGH",
                            "source_ip": f"10.0.0.{i}"})
            for i in range(6)
        ]
        blocked = [r for r in results if r["action"] == "BLOCK"]
        assert len(blocked) <= 3
        # The remainder is still *recorded*, just not enforced.
        assert any(r["action"] == "LOG_ONLY" for r in results)

    def test_rollback_works(self):
        from src.response import ResponseEngine

        engine = ResponseEngine(dry_run=True)
        engine.execute({"confidence": 0.95, "severity": "HIGH",
                        "source_ip": "10.0.0.5"})
        assert engine.rollback("10.0.0.5") is True
        assert "10.0.0.5" not in engine.active_blocks
        assert engine.rollback("10.0.0.5") is False  # Already gone


class TestDecisionTable:
    """Severity and confidence interact the way the plan's table says."""

    @pytest.mark.parametrize("confidence,severity,expected", [
        (0.60, "HIGH", LOG_ONLY),      # below the confidence floor
        (0.69, "MEDIUM", LOG_ONLY),
        (0.70, "HIGH", RATE_LIMIT),    # floor reached, below the block bar
        (0.89, "HIGH", RATE_LIMIT),
        (0.90, "MEDIUM", RATE_LIMIT),  # MEDIUM never blocks outright
        (0.90, "HIGH", BLOCK),
        (0.99, "HIGH", BLOCK),
        (0.99, "LOW", LOG_ONLY),       # LOW always stays log-only
    ])
    def test_tier_boundaries(self, confidence, severity, expected):
        engine = ResponseEngine()
        assert engine.decide_action(alert(confidence, severity)) == expected

    @pytest.mark.parametrize("missing", [{}, {"confidence": None},
                                         {"severity": None}])
    def test_partial_alerts_default_to_the_safe_direction(self, missing):
        """A triage console must not crash on an under-filled alert."""
        engine = ResponseEngine()
        assert engine.decide_action(missing) == LOG_ONLY

    def test_unmapped_severity_is_treated_as_medium(self):
        engine = ResponseEngine()
        assert engine.decide_action({"confidence": 0.99,
                                     "severity": "CATASTROPHIC"}) == RATE_LIMIT


class TestBudgetAndTTL:
    """NFR10 budget accounting and automatic expiry."""

    def test_a_refresh_does_not_spend_budget(self):
        """Re-blocking a host refreshes one rule; it must not count as new."""
        engine = ResponseEngine(max_blocks_per_minute=2, dry_run=True)
        first = engine.execute(alert(0.99, "HIGH", "10.0.0.1"))
        refresh = engine.execute(alert(0.99, "HIGH", "10.0.0.1"))
        second = engine.execute(alert(0.99, "HIGH", "10.0.0.2"))
        refused = engine.execute(alert(0.99, "HIGH", "10.0.0.3"))

        assert first["action"] == BLOCK
        assert second["action"] == BLOCK
        assert refresh["refreshed_existing_block"] is True
        # The third *new* host is refused even though only two rules exist.
        assert refused["action"] == LOG_ONLY
        assert len(engine.recent_blocks) == 2
        assert engine.stats()["blocks_remaining_this_minute"] == 0
        assert engine.stats()["rate_limited_downgrades"] == 1
        assert set(engine.active_blocks) == {"10.0.0.1", "10.0.0.2"}

    def test_downgraded_block_still_records_the_alert(self):
        engine = ResponseEngine(max_blocks_per_minute=1, dry_run=True)
        engine.execute(alert(0.99, "HIGH", "10.0.0.1"))
        refused = engine.execute(alert(0.99, "HIGH", "10.0.0.2"))

        assert refused["action"] == LOG_ONLY
        assert refused["requested_action"] == BLOCK
        assert refused["success"] is False
        assert "Rate limit" in refused["reason"]
        assert "10.0.0.2" not in engine.active_blocks

    def test_expired_budget_refills_after_a_minute(self):
        engine = ResponseEngine(max_blocks_per_minute=1, dry_run=True)
        engine.recent_blocks.append(time.time())       # spent, this minute
        assert engine.stats()["blocks_remaining_this_minute"] == 0

        engine.recent_blocks[0] = time.time() - 61     # ...but a minute ago
        assert engine.stats()["blocks_remaining_this_minute"] == 1

    def test_block_expires_on_the_next_poll(self):
        engine = ResponseEngine(auto_unblock_seconds=0.15, dry_run=True)
        engine.execute(alert(0.99, "HIGH"))
        assert "10.0.0.5" in engine.active_blocks

        time.sleep(0.25)
        assert engine.active_blocks == {}

        actions = [entry["action"] for entry in engine.history()]
        assert "AUTO_UNBLOCK" in actions

    def test_background_sweeper_expires_without_new_alerts(self):
        """Expiry must not depend on another alert ever arriving."""
        engine = ResponseEngine(auto_unblock_seconds=0.1, dry_run=True)
        engine.execute(alert(0.99, "HIGH"))
        try:
            time.sleep(0.5)
            actions = [entry["action"] for entry in engine.history()]
            assert "AUTO_UNBLOCK" in actions
        finally:
            engine.shutdown()

    def test_rate_limits_also_expire(self):
        engine = ResponseEngine(rate_limit_seconds=0.1, dry_run=True)
        engine.execute(alert(0.8, "HIGH"))
        assert engine.active_rate_limits
        time.sleep(0.2)
        assert engine.active_rate_limits == {}

    def test_rollback_all_clears_every_block(self):
        engine = ResponseEngine(dry_run=True)
        for i in range(3):
            engine.execute(alert(0.99, "HIGH", f"10.0.0.{i}"))
        assert len(engine.active_blocks) == 3

        assert engine.rollback_all() == 3
        assert engine.active_blocks == {}
        assert engine.rollback_all() == 0


class TestDryRunSafety:
    """Nothing in this repository turns dry-run off."""

    @pytest.fixture()
    def spy(self, monkeypatch):
        calls = []

        def _record(command, *args, **kwargs):
            calls.append(command)
            raise AssertionError("firewall command executed")

        monkeypatch.setattr("src.response.action_engine.subprocess.run", _record)
        return calls

    def test_dry_run_never_reaches_the_firewall(self, spy):
        engine = ResponseEngine(dry_run=True)
        engine.execute(alert(0.99, "HIGH", "10.0.0.9"))
        engine.execute(alert(0.8, "HIGH", "10.0.0.9"))
        engine.rollback("10.0.0.9")

        assert spy == []

    def test_the_command_is_still_recorded_for_audit(self, spy):
        engine = ResponseEngine(dry_run=True)
        result = engine.execute(alert(0.99, "HIGH", "10.0.0.9"))

        assert "netsh" in result["command"] or "iptables" in result["command"]
        assert result["applied"] is False
        assert result["dry_run"] is True
        assert "dry_run" in result["reason"]

    def test_live_mode_would_execute(self, monkeypatch):
        executed = []

        class _Result:
            returncode = 0
            stderr = ""

        monkeypatch.setattr(
            "src.response.action_engine.subprocess.run",
            lambda command, **kwargs: executed.append(command) or _Result())

        engine = ResponseEngine(dry_run=False)
        try:
            engine.execute(alert(0.99, "HIGH", "10.0.0.9"))
        finally:
            engine.shutdown()

        assert len(executed) == 1
        assert executed[0][0] in {"netsh", "iptables"}

    def test_default_construction_is_dry_run(self):
        assert ResponseEngine().dry_run is True


class TestConfiguration:
    def test_loads_the_project_config(self):
        engine = ResponseEngine.from_config()
        assert engine.max_blocks_per_minute == 10
        assert engine.auto_unblock_seconds == 300
        assert engine.dry_run is True

    def test_an_unknown_key_is_rejected(self, tmp_path):
        path = tmp_path / "response.yaml"
        path.write_text("response:\n  max_block_per_minute: 5\n",
                        encoding="utf-8")
        with pytest.raises(ValueError, match="max_block_per_minute"):
            ResponseEngine.from_config(path)

    @pytest.mark.parametrize("kwargs", [
        {"max_blocks_per_minute": 0},
        {"auto_unblock_seconds": 0},
    ])
    def test_unsafe_limits_are_rejected(self, kwargs):
        with pytest.raises(ValueError):
            ResponseEngine(**kwargs)


class TestBookkeeping:
    def test_stats_summarise_the_run_for_the_report(self):
        engine = ResponseEngine(max_blocks_per_minute=5, dry_run=True)
        engine.execute(alert(0.99, "HIGH", "10.0.0.1"))
        engine.execute(alert(0.80, "HIGH", "10.0.0.2"))
        engine.execute(alert(0.50, "HIGH", "10.0.0.3"))

        stats = engine.stats()
        assert stats["total_actions"] == 3
        assert stats["action_counts"] == {BLOCK: 1, RATE_LIMIT: 1, LOG_ONLY: 1}
        assert stats["active_blocks"] == 1
        assert stats["active_rate_limits"] == 1
        assert stats["max_blocks_per_minute"] == 5
        assert stats["dry_run"] is True

    def test_history_returns_detached_copies(self):
        engine = ResponseEngine(dry_run=True)
        engine.execute(alert(0.99, "HIGH"))
        history = engine.history()
        history[0]["label"] = "tampered"
        assert engine.history()[0]["label"] == "DoS"

    def test_context_manager_stops_the_sweeper(self):
        with ResponseEngine(auto_unblock_seconds=0.1) as engine:
            engine.execute(alert(0.99, "HIGH"))
            assert engine._sweeper is not None
        assert engine._sweeper is None

    def test_command_builders_are_symmetric(self):
        block = ResponseEngine.build_block_command("10.0.0.9")
        unblock = ResponseEngine.build_unblock_command("10.0.0.9")
        assert "10.0.0.9" in " ".join(block)
        assert "10.0.0.9" in " ".join(unblock)
        assert unblock[0] == block[0]
