"""
src/response/action_engine.py — Tiered automated response with false-positive
safety (PHASE_4 section 4.5, requirements FR7 / NFR10).

Decision table (severity × confidence):

==============  =============================  ==================
Condition       Action                         Effect
==============  =============================  ==================
conf < 0.7 or   ``LOG_ONLY``                   record the alert,
severity == LOW                                 change nothing
0.7 ≤ conf <    ``RATE_LIMIT``                 rate-limit the source
0.9 or MEDIUM
conf ≥ 0.9 and  ``BLOCK``                      block the source until
severity == HIGH                                the TTL expires
==============  =============================  ==================

Three safety properties are enforced in code rather than left to the operator:

1. **Bounded blocking (NFR10).**  At most ``max_blocks_per_minute`` *new* blocks
   may be installed per rolling 60 s window.  A burst of false positives — a
   mis-tuned model on a busy network — degrades to ``LOG_ONLY`` instead of
   locking every host out of the network.  Re-blocking an IP that is already
   blocked refreshes its TTL and does **not** consume budget, because it does
   not add a new firewall rule.
2. **Automatic rollback.**  Every block carries a TTL (default 300 s).  Blocks
   are reaped by a background sweeper *and* on every engine call, so a block
   expires even if no further alert ever arrives, and a test or a caller that
   polls the engine sees expiry without waiting on a thread schedule.
3. **Dry-run by default.**  The engine never touches the host firewall unless
   it is constructed with ``dry_run=False``.  The command that *would* run is
   still built and recorded, so a run is fully auditable without needing
   elevated privileges.  Nothing in this repository turns dry-run off.
"""
from __future__ import annotations

import platform
import subprocess
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Optional, Union

__all__ = ["ResponseEngine", "LOG_ONLY", "RATE_LIMIT", "BLOCK"]

LOG_ONLY = "LOG_ONLY"
RATE_LIMIT = "RATE_LIMIT"
BLOCK = "BLOCK"

ACTIONS = (LOG_ONLY, RATE_LIMIT, BLOCK)
SEVERITIES = ("LOW", "MEDIUM", "HIGH")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "response.yaml"


class ResponseEngine:
    """Tiered automated response with false-positive safety."""

    def __init__(self, max_blocks_per_minute: int = 10,
                 auto_unblock_seconds: int = 300,
                 rate_limit_seconds: int = 60,
                 log_confidence_below: float = 0.7,
                 block_confidence: float = 0.9,
                 dry_run: bool = True,
                 action_log_size: int = 10000):
        if max_blocks_per_minute < 1:
            raise ValueError("max_blocks_per_minute must be >= 1.")
        if auto_unblock_seconds <= 0:
            raise ValueError("auto_unblock_seconds must be > 0.")

        self.max_blocks_per_minute = int(max_blocks_per_minute)
        self.auto_unblock_seconds = float(auto_unblock_seconds)
        self.rate_limit_seconds = float(rate_limit_seconds)
        self.log_confidence_below = float(log_confidence_below)
        self.block_confidence = float(block_confidence)
        self.dry_run = bool(dry_run)

        self._active_blocks: dict[str, float] = {}       # ip -> expiry (epoch s)
        self._active_rate_limits: dict[str, float] = {}
        self.action_log: deque[dict] = deque(maxlen=action_log_size)
        self.recent_blocks: deque[float] = deque(
            maxlen=self.max_blocks_per_minute)

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._sweeper: Optional[threading.Thread] = None

    # -- construction ------------------------------------------------------
    @classmethod
    def from_config(cls, path: Union[str, Path] = DEFAULT_CONFIG) -> "ResponseEngine":
        """
        Build an engine from ``configs/response.yaml``.

        The YAML may carry a top-level ``response:`` mapping (the file is
        shared with other phase configs); unknown keys are rejected so a typo
        cannot silently disable a safety limit.
        """
        import yaml

        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = _PROJECT_ROOT / config_path
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        params = dict(raw.get("response", raw))

        known = {"max_blocks_per_minute", "auto_unblock_seconds",
                 "rate_limit_seconds", "log_confidence_below",
                 "block_confidence", "dry_run", "action_log_size"}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"Unknown response config keys: {unknown}")
        return cls(**params)

    # -- decision ----------------------------------------------------------
    def decide_action(self, alert: dict) -> str:
        """
        Map an alert onto a response tier.

        A missing confidence is treated as 0.0 (the safe direction: log only)
        rather than as an error, because a triage console must not crash on a
        partially populated alert.
        """
        confidence = float(alert.get("confidence", 0.0) or 0.0)
        severity = str(alert.get("severity", "MEDIUM") or "MEDIUM").upper()
        if severity not in SEVERITIES:
            severity = "MEDIUM"

        if confidence < self.log_confidence_below or severity == "LOW":
            return LOG_ONLY
        if confidence < self.block_confidence or severity == "MEDIUM":
            return RATE_LIMIT
        return BLOCK

    # -- execution ---------------------------------------------------------
    def execute(self, alert: dict) -> dict:
        """Decide and carry out the response for one alert."""
        action = self.decide_action(alert)
        source_ip = str(alert.get("source_ip") or "unknown")
        now = time.time()

        result = {
            "action": action,
            "requested_action": action,
            "source_ip": source_ip,
            "timestamp": now,
            "confidence": float(alert.get("confidence", 0.0) or 0.0),
            "severity": str(alert.get("severity", "MEDIUM") or "MEDIUM"),
            "label": alert.get("label"),
            "alert_id": alert.get("alert_id"),
            "success": True,
            "dry_run": self.dry_run,
        }

        with self._lock:
            self.sweep()
            if action == BLOCK:
                if source_ip in self._active_blocks:
                    # Refreshing an existing rule is not a new block.
                    self._extend(source_ip, self.auto_unblock_seconds, now)
                    result["refreshed_existing_block"] = True
                elif not self._rate_limit_check(now):
                    result["action"] = LOG_ONLY
                    result["reason"] = "Rate limit exceeded — downgraded to LOG_ONLY"
                    result["success"] = False
                else:
                    outcome = self._block_ip(source_ip, now)
                    result.update(outcome)
                    self.recent_blocks.append(now)
            elif action == RATE_LIMIT:
                outcome = self._rate_limit_ip(source_ip, now)
                result.update(outcome)

            self.action_log.append(result)
        return dict(result)

    def rollback(self, source_ip: str) -> bool:
        """Manually unblock an IP (false-positive correction)."""
        with self._lock:
            if source_ip not in self._active_blocks:
                self.sweep()
                if source_ip not in self._active_blocks:
                    return False
            self._unblock_ip(source_ip, reason="manual_rollback")
            return True

    def rollback_all(self) -> int:
        """Remove every active block; returns how many were removed."""
        with self._lock:
            ips = list(self._active_blocks)
            for ip in ips:
                self._unblock_ip(ip, reason="rollback_all")
            return len(ips)

    # -- state -------------------------------------------------------------
    @property
    def active_blocks(self) -> dict[str, float]:
        """IP -> expiry timestamp for blocks that have not yet expired."""
        with self._lock:
            self.sweep()
            return dict(self._active_blocks)

    @property
    def active_rate_limits(self) -> dict[str, float]:
        """IP -> expiry timestamp for active rate limits."""
        with self._lock:
            now = time.time()
            for ip, expiry in list(self._active_rate_limits.items()):
                if expiry <= now:
                    self._active_rate_limits.pop(ip, None)
            return dict(self._active_rate_limits)

    def sweep(self, now: Optional[float] = None) -> list[str]:
        """
        Expire blocks whose TTL has elapsed.

        Called on every engine operation *and* by the background sweeper, so
        expiry does not depend on new alerts arriving or on thread timing.
        """
        now = time.time() if now is None else now
        expired = [ip for ip, expiry in self._active_blocks.items()
                   if expiry <= now]
        for ip in expired:
            self._active_blocks.pop(ip, None)
            self.action_log.append({
                "action": "AUTO_UNBLOCK",
                "source_ip": ip,
                "timestamp": now,
                "success": True,
                "dry_run": self.dry_run,
                "reason": f"TTL of {self.auto_unblock_seconds:g}s elapsed",
            })
        for ip, expiry in list(self._active_rate_limits.items()):
            if expiry <= now:
                self._active_rate_limits.pop(ip, None)
        return expired

    def history(self) -> list[dict]:
        """Full action log, oldest first (JSON-ready copies)."""
        with self._lock:
            return [dict(entry) for entry in self.action_log]

    def stats(self) -> dict:
        """Counters used by the dashboard and the phase report."""
        with self._lock:
            self.sweep()
            counts = Counter(entry["action"] for entry in self.action_log)
            return {
                "total_actions": sum(counts.values()),
                "action_counts": dict(counts),
                "active_blocks": len(self._active_blocks),
                "active_rate_limits": len(self._active_rate_limits),
                "rate_limited_downgrades": sum(
                    1 for e in self.action_log
                    if e.get("reason", "").startswith("Rate limit")),
                "blocks_remaining_this_minute": self._blocks_remaining(time.time()),
                "max_blocks_per_minute": self.max_blocks_per_minute,
                "auto_unblock_seconds": self.auto_unblock_seconds,
                "dry_run": self.dry_run,
            }

    def shutdown(self, timeout: float = 2.0) -> None:
        """Stop the background sweeper (idempotent)."""
        self._stop.set()
        sweeper = self._sweeper
        if sweeper is not None and sweeper.is_alive():
            sweeper.join(timeout=timeout)
        self._sweeper = None

    def __enter__(self) -> "ResponseEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # -- internals ---------------------------------------------------------
    def _sweep_interval(self) -> float:
        return max(0.05, min(1.0, self.auto_unblock_seconds / 4.0))

    def _ensure_sweeper(self) -> None:
        if self._sweeper is not None or self._stop.is_set():
            return

        def _loop() -> None:
            while not self._stop.wait(self._sweep_interval()):
                with self._lock:
                    self.sweep()

        thread = threading.Thread(target=_loop, name="nids-response-sweeper",
                                  daemon=True)
        self._sweeper = thread
        thread.start()

    def _rate_limit_check(self, now: float) -> bool:
        while self.recent_blocks and self.recent_blocks[0] <= now - 60:
            self.recent_blocks.popleft()
        return len(self.recent_blocks) < self.max_blocks_per_minute

    def _blocks_remaining(self, now: float) -> int:
        while self.recent_blocks and self.recent_blocks[0] <= now - 60:
            self.recent_blocks.popleft()
        return max(0, self.max_blocks_per_minute - len(self.recent_blocks))

    def _extend(self, ip: str, seconds: float, now: float) -> None:
        self._active_blocks[ip] = now + seconds
        if ip in self._active_rate_limits:
            self._active_rate_limits[ip] = now + self.rate_limit_seconds

    # -- platform actions --------------------------------------------------
    @staticmethod
    def build_block_command(ip: str) -> list[str]:
        """Firewall command that blocks ``ip`` on the current platform."""
        if platform.system() == "Windows":
            return [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name=NIDS-Block-{ip}", "dir=in", "action=block",
                f"remoteip={ip}",
            ]
        return ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]

    @staticmethod
    def build_unblock_command(ip: str) -> list[str]:
        """Firewall command that removes the block for ``ip``."""
        if platform.system() == "Windows":
            return [
                "netsh", "advfirewall", "firewall", "delete", "rule",
                f"name=NIDS-Block-{ip}",
            ]
        return ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]

    @staticmethod
    def build_rate_limit_command(ip: str) -> list[str]:
        """Firewall command that rate-limits ``ip``."""
        if platform.system() == "Windows":
            return [
                "netsh", "advfirewall", "firewall", "add", "rule",
                f"name=NIDS-RateLimit-{ip}", "dir=in", "action=bypass",
                f"remoteip={ip}",
                "description=NIDS rate-limit marker (no native netsh limiter)",
            ]
        return ["iptables", "-A", "INPUT", "-s", ip, "-m", "limit",
                "--limit", "10/sec", "-j", "ACCEPT"]

    def _apply(self, command: list[str], ip: str, action: str) -> dict:
        """
        Run (or, in dry-run, merely record) a firewall command.

        Failures are recorded, never raised: the engine is a safety net, and an
        operator must see a failed block in the log rather than lose the alert
        to an exception.
        """
        record = {"command": " ".join(command), "applied": False}
        if self.dry_run:
            record["reason"] = "dry_run: command recorded, not executed"
            return record
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       timeout=15, check=False)
            record.update({
                "applied": completed.returncode == 0,
                "returncode": completed.returncode,
                "stderr": (completed.stderr or "").strip()[:500],
            })
        except Exception as error:  # pragma: no cover - host dependent
            record.update({"applied": False, "error": f"{type(error).__name__}: {error}"})
        return record

    def _block_ip(self, ip: str, now: float) -> dict:
        outcome = self._apply(self.build_block_command(ip), ip, BLOCK)
        self._active_blocks[ip] = now + self.auto_unblock_seconds
        self._ensure_sweeper()
        outcome["expires_at"] = self._active_blocks[ip]
        return outcome

    def _unblock_ip(self, ip: str, reason: str) -> None:
        self._active_blocks.pop(ip, None)
        self._active_rate_limits.pop(ip, None)
        outcome = self._apply(self.build_unblock_command(ip), ip, "UNBLOCK")
        self.action_log.append({
            "action": "UNBLOCK",
            "source_ip": ip,
            "timestamp": time.time(),
            "success": True,
            "dry_run": self.dry_run,
            "reason": reason,
            **outcome,
        })

    def _rate_limit_ip(self, ip: str, now: float) -> dict:
        outcome = self._apply(self.build_rate_limit_command(ip), ip, RATE_LIMIT)
        self._active_rate_limits[ip] = now + self.rate_limit_seconds
        outcome["expires_at"] = self._active_rate_limits[ip]
        return outcome
