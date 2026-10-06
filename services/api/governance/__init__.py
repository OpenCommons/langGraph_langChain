"""OEE completeness governance layer (see docs/GOVERNANCE.md for guarantees and limits)."""

from __future__ import annotations

import threading

from governance.capability import CapabilityAuthority
from governance.engine import GovernanceEngine, load_klines, load_policy
from governance.merkle import ProvenanceChain

_engine: GovernanceEngine | None = None
_lock = threading.Lock()


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def get_engine() -> GovernanceEngine:
    """Process-wide engine built from settings."""
    global _engine
    with _lock:
        if _engine is None:
            from config import get_settings

            s = get_settings()
            _engine = GovernanceEngine(
                policy=load_policy(s.governance_policy_path or None),
                authority=CapabilityAuthority(s.governance_token_secret)
                if s.governance_token_secret
                else None,
                chain=ProvenanceChain(s.governance_chain_path or None),
                seed_klines=load_klines(s.governance_klines_path or None),
                require_token=s.governance_require_token,
                default_tools=_csv(s.governance_default_tools) or ["*"],
                default_scopes=_csv(s.governance_default_scopes)
                or ["agent:run", "handoff:dispatch"],
                max_false_positives=s.governance_max_false_positives,
                kline_max=s.governance_kline_max,
                predictive=s.governance_predictive_enabled,
                predictive_threshold=s.governance_predictive_threshold,
                predictive_horizon=s.governance_predictive_horizon,
                predictive_margin=s.governance_predictive_margin,
            )
        return _engine


def reset_engine() -> None:
    global _engine
    with _lock:
        _engine = None
