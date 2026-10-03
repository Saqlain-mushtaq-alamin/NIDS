"""
src.response — Automated response for NIDS alerts (Phase 4).

Public API::

    from src.response import ResponseEngine, LOG_ONLY, RATE_LIMIT, BLOCK
"""
from src.response.action_engine import BLOCK, LOG_ONLY, RATE_LIMIT, ResponseEngine

__all__ = ["BLOCK", "LOG_ONLY", "RATE_LIMIT", "ResponseEngine"]
