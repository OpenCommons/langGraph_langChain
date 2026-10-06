"""
/governance endpoints — OEE completeness layer (docs/GOVERNANCE.md).

Read-only (never tick the clock, append a record, or change rules):
GET  /governance/state           clock, rule version, chain head, k-line counts
GET  /governance/chain           records (newest ``limit``) + full-chain verification
POST /governance/replay          counterfactual replay of a candidate rule set (no staging)

Mutating / dispatching:
POST /governance/dispatch        fire-and-record handoff event -> 202 Accepted
POST /governance/rules           stage a replay-gated rule set (needs scope governance:admin)
"""

from __future__ import annotations

import uuid
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from governance import get_engine
from governance.adfa import RuleSet
from governance.capability import CapabilityError
from governance.swap import RuleSetRejected

log = structlog.get_logger()
router = APIRouter(prefix="/governance", tags=["governance"])

ADMIN_SCOPE = "governance:admin"
DISPATCH_SCOPE = "handoff:dispatch"


def _compile(raw: dict) -> RuleSet:
    engine = get_engine()
    try:
        return RuleSet.from_dict(
            {**raw, "dim": len(engine.encoder.features)}, engine.encoder.features
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, f"invalid rule set: {exc}") from exc


@router.get("/state")
def state():
    return get_engine().snapshot()


@router.get("/chain")
def chain(limit: int = Query(50, ge=1, le=1000)):
    engine = get_engine()
    ok, first_bad = engine.chain.verify()
    return {
        "valid": ok,
        "first_bad_index": first_bad,
        "length": len(engine.chain),
        "head": engine.chain.head,
        "lamport": engine.clock.value,
        "records": engine.chain.records(limit),
    }


class RuleSetBody(BaseModel):
    rules: dict[str, Any]


@router.post("/replay")
def replay_candidate(body: RuleSetBody):
    engine = get_engine()
    report = engine.rules.check(_compile(body.rules))
    return {**report.to_dict(), "eligible": engine.rules.eligible(report)}


@router.post("/rules")
def stage_rules(body: RuleSetBody, x_capability_token: str | None = Header(default=None)):
    engine = get_engine()
    try:
        cap = engine.authority.verify(x_capability_token)
    except CapabilityError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, str(exc)) from exc
    if not cap.allows_scope(ADMIN_SCOPE):
        raise HTTPException(status.HTTP_403_FORBIDDEN, f"scope {ADMIN_SCOPE!r} required")
    candidate = _compile(body.rules)
    try:
        report = engine.rules.stage(candidate)
    except RuleSetRejected as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, {"error": str(exc), "replay": exc.report.to_dict()}
        ) from exc
    return {"staged": candidate.version, "takes_effect": "next event", "replay": report.to_dict()}


class DispatchBody(BaseModel):
    thread_id: str
    target: str = Field(description="Receiving agent / handoff name")
    payload: dict[str, Any] = Field(default_factory=dict)
    context: list[str] = Field(default_factory=list)
    lamport: int | None = Field(default=None, description="Sender's Lamport timestamp")


def _record_dispatch(event_id: str, body: DispatchBody, token: str | None, lamport: int) -> None:
    try:
        get_engine().admit(
            thread_id=body.thread_id,
            tool=body.target,
            args={"event_id": event_id, **body.payload},
            context=body.context,
            execute=None,  # fire-and-record: the receiver acts, we verify and record
            token=token,
            scope=DISPATCH_SCOPE,
            lamport=lamport,
        )
    except Exception as exc:
        log.error("governance.dispatch_record_failed", event_id=event_id, error=str(exc))


@router.post("/dispatch", status_code=status.HTTP_202_ACCEPTED)
def dispatch(
    body: DispatchBody,
    background: BackgroundTasks,
    x_capability_token: str | None = Header(default=None),
):
    engine = get_engine()
    event_id = uuid.uuid4().hex
    lamport = engine.clock.receive(body.lamport) if body.lamport else engine.clock.tick()
    background.add_task(_record_dispatch, event_id, body, x_capability_token, lamport)
    return {"status": "accepted", "event_id": event_id, "lamport": lamport}
