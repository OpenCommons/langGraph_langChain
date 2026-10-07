"""
GovernanceEngine: the Atomic Triad.

Every action the graph admits is *verified* (aDFA over the full thread context),
*executed* (only if verified) and *recorded* (Merkle chain + K-line) inside one
``admit`` call. There is no code path that executes without verifying, and none
that returns without recording — a blocked action, a capability denial and a
tool that raises are all recorded.

Limits (see docs/GOVERNANCE.md): single process, in-memory state unless a chain
path is configured, Python-runtime latency, and the record cannot undo a tool's
external side effects.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import structlog

from governance.adfa import RuleSet
from governance.capability import CapabilityAuthority, CapabilityError
from governance.clock import LamportClock
from governance.features import ContextEncoder
from governance.klines import HAZARD, NOMINAL, KLine, KLineStore
from governance.merkle import ProvenanceChain
from governance.predictive import MarkovHazardEstimator
from governance.swap import RuleManager, RuleSetRejected

log = structlog.get_logger()

_DATA_DIR = pathlib.Path(__file__).parent / "data"
MAX_THREAD_FRAMES = 64
_ARGS_LIMIT = 500


@dataclass(frozen=True)
class Admission:
    allowed: bool
    executed: bool
    result: Any
    reason: str | None
    rule: str | None
    lamport: int
    record_hash: str
    rules_version: str

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "result"}


def load_policy(path: str | None = None) -> dict:
    return json.loads(pathlib.Path(path or _DATA_DIR / "default_policy.json").read_text())


def load_klines(path: str | None = None) -> list[KLine]:
    data = json.loads(pathlib.Path(path or _DATA_DIR / "klines.json").read_text())
    return [KLine.from_dict(k) for k in data["klines"]]


def _bounded_args(args: Any) -> Any:
    text = json.dumps(args, default=str, sort_keys=True)
    return json.loads(text) if len(text) <= _ARGS_LIMIT else text[:_ARGS_LIMIT] + "…"


class GovernanceEngine:
    def __init__(
        self,
        *,
        policy: Mapping | None = None,
        authority: CapabilityAuthority | None = None,
        chain: ProvenanceChain | None = None,
        seed_klines: Sequence[KLine] | None = None,
        require_token: bool = False,
        default_tools: Sequence[str] = ("*",),
        default_scopes: Sequence[str] = ("agent:run", "handoff:dispatch"),
        max_false_positives: int = 0,
        kline_max: int = 1000,
        predictive: bool = False,
        predictive_threshold: float = 0.5,
        predictive_horizon: int = 3,
        predictive_margin: float = 0.1,
    ) -> None:
        policy = policy or load_policy()
        self.encoder = ContextEncoder(policy)
        dim = len(self.encoder.features)
        self.authority = authority or CapabilityAuthority(secrets.token_hex(32))
        self.chain = chain or ProvenanceChain()
        self.clock = LamportClock()
        self.klines = KLineStore(
            load_klines() if seed_klines is None else seed_klines, max_observed=kline_max
        )
        initial = RuleSet.from_dict({**policy["rules"], "dim": dim}, self.encoder.features)
        self.rules = RuleManager(initial, self.klines.all, max_false_positives)
        self.require_token = require_token
        self._default_token = self.authority.mint("default-agent", default_tools, default_scopes)
        self.predictive = predictive
        self.estimator = MarkovHazardEstimator()
        self._threshold, self._horizon, self._margin = (
            predictive_threshold,
            predictive_horizon,
            predictive_margin,
        )
        self._tightened_from: set[str] = set()
        self._threads: dict[str, dict] = {}
        self._threads_lock = threading.Lock()

    # ── helpers ────────────────────────────────────────────────────────────
    def _thread(self, thread_id: str) -> dict:
        with self._threads_lock:
            return self._threads.setdefault(
                thread_id,
                {"s0": {"mode": "default", "thread_id": thread_id}, "frames": [], "cls": None},
            )

    def _classify(self, rules: RuleSet, vector: list[float], s0: Mapping, blocked: bool) -> str:
        if blocked:
            return "hazard"
        return (
            "elevated"
            if rules.widened(self._margin, "probe").step(vector, s0).blocked
            else "nominal"
        )

    def _maybe_tighten(self, rules: RuleSet, cls: str) -> None:
        if rules.version in self._tightened_from:
            return
        if self.estimator.p_hazard(cls, self._horizon) <= self._threshold:
            return
        self._tightened_from.add(rules.version)
        try:
            self.rules.stage(rules.widened(self._margin, f"{rules.version}+tight"))
            log.warning("governance.rules_tightened", base=rules.version)
        except RuleSetRejected as exc:
            log.info("governance.tighten_rejected", reason=str(exc))

    # ── the Atomic Triad ───────────────────────────────────────────────────
    def admit(
        self,
        *,
        thread_id: str,
        tool: str,
        args: Mapping | str | None,
        context: Sequence[str],
        execute: Callable[[], Any] | None,
        token: str | None = None,
        scope: str = "agent:run",
        message_lamport: int | None = None,
        lamport: int | None = None,
    ) -> Admission:
        if lamport is None:
            lamport = self.clock.receive(message_lamport) if message_lamport else self.clock.tick()
        rules = self.rules.active()  # one reference per event
        th = self._thread(thread_id)
        s0 = th["s0"]

        # identity & capability
        token_id, reason = "none", None
        try:
            if token is None and not self.require_token:
                token = self._default_token
            token_id = self.authority.authorize(token, tool, scope).id
        except CapabilityError as exc:
            reason = f"capability: {exc}"

        # verify over the full thread context (O(|w|) single transition)
        vector = self.encoder.encode(context, tool, args)
        verdict = rules.step(vector, s0)
        rule = verdict.rule
        if reason is None and verdict.blocked:
            reason = f"hazard: {rule}"
        allowed = reason is None

        # execute — only if verified
        executed, result, error, failure = False, None, None, None
        if allowed and execute is not None:
            try:
                result = execute()
                executed = True
            except Exception as exc:
                failure = exc
                error = f"{type(exc).__name__}: {exc}"
                log.warning("governance.tool_error", tool=tool, error=error)

        # record — always
        frame = {
            "lamport": lamport,
            "thread_id": thread_id,
            "tool": tool,
            "args": _bounded_args(args),
            "vector": vector,
            "output": list(verdict.output),
            "allowed": allowed,
            "reason": reason,
            "executed": executed,
            "error": error,
            "result_sha256": hashlib.sha256(str(result).encode()).hexdigest() if executed else None,
            "rules_version": rules.version,
        }
        rec = self.chain.append(s0, frame, token_id)
        if verdict.blocked or reason is None:
            self._observe(th, rules, vector, s0, verdict.blocked)

        if failure is not None:
            raise failure
        return Admission(
            allowed, executed, result, reason, rule, lamport, rec["hash"], rules.version
        )

    def _observe(self, th: dict, rules: RuleSet, vector, s0, blocked: bool) -> None:
        th["frames"] = (th["frames"] + [vector])[-MAX_THREAD_FRAMES:]
        self.klines.add(s0, th["frames"], HAZARD if blocked else NOMINAL)
        if blocked:
            th["frames"] = []  # the hazard trace is sealed; the next one starts fresh
        if self.predictive:
            cls = self._classify(rules, vector, s0, blocked)
            if th["cls"] is not None:
                self.estimator.observe(th["cls"], cls)
            th["cls"] = cls
            self._maybe_tighten(rules, cls)

    # ── read-only inspection (no clock tick, no record, no mutation) ───────
    def snapshot(self) -> dict:
        active = self.rules.peek()
        ok, bad = self.chain.verify()
        return {
            "lamport": self.clock.value,
            "rules_version": active.version,
            "pending_rules_version": self.rules.pending_version,
            "rule_names": [r.name for r in active.rules],
            "features": list(self.encoder.features),
            "chain": {
                "length": len(self.chain),
                "head": self.chain.head,
                "valid": ok,
                "first_bad": bad,
            },
            "klines": self.klines.counts(),
            "predictive": {
                "enabled": self.predictive,
                "threshold": self._threshold,
                "horizon": self._horizon,
                "p_hazard": {
                    c: self.estimator.p_hazard(c, self._horizon) for c in self.estimator.classes
                }
                if self.predictive
                else None,
            },
        }
