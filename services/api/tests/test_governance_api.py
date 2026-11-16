"""Governance HTTP surface and its wiring into the LangGraph agent API (fake LLM)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import config
import governance
from governance.capability import CapabilityAuthority
from graphs import checkpoint
from routers import governance_api, langgraph_api
from tests.fakes import fake_llm, tool_call

SECRET = "api-test-secret"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DB_PATH", ":memory:")
    monkeypatch.setenv("GOVERNANCE_TOKEN_SECRET", SECRET)
    config.get_settings.cache_clear()
    checkpoint.reset_checkpointer()
    governance.reset_engine()
    app = FastAPI()
    app.include_router(langgraph_api.router)
    app.include_router(governance_api.router)
    app.dependency_overrides[langgraph_api.get_llm] = lambda: fake_llm("hello")
    yield TestClient(app)
    config.get_settings.cache_clear()
    checkpoint.reset_checkpointer()
    governance.reset_engine()


def _auth() -> CapabilityAuthority:
    return CapabilityAuthority(SECRET)


def _use(client, *replies):
    client.app.dependency_overrides[langgraph_api.get_llm] = lambda: fake_llm(*replies)


def test_agent_tool_call_is_admitted_executed_and_recorded(client):
    _use(client, tool_call("calculator", {"expression": "6*7"}), "42")
    r = client.post("/lg/agent/run", json={"message": "6*7?", "thread_id": "g1"})
    assert r.json()["answer"] == "42" and int(r.headers["X-Lamport"]) >= 1
    chain = client.get("/governance/chain").json()
    assert chain["valid"] and chain["length"] == 1
    frame = chain["records"][0]["frame"]
    assert frame["tool"] == "calculator" and frame["executed"] is True and frame["allowed"] is True


def test_agent_tool_with_hazard_thread_is_blocked_not_executed(client):
    from chains.tools import base_tools
    from governance import get_engine
    from governance.tools import govern_tools

    ran = []
    risky = base_tools()[0].model_copy(
        update={"name": "shell", "func": lambda expression: ran.append(1)}
    )
    wrapped = govern_tools(
        [risky], get_engine(), thread_id="g3", context=["dr", "op table x", "rm -", "rf /"]
    )[0]
    assert "BLOCKED" in wrapped.invoke({"expression": "1"}) and ran == []
    assert client.get("/governance/chain").json()["records"][-1]["frame"]["allowed"] is False


def test_agent_capability_scoping_via_header(client):
    tok = _auth().mint("limited", ["current_time"], ["agent:run"])
    _use(client, tool_call("calculator", {"expression": "2+2"}), "denied")
    client.post(
        "/lg/agent/run",
        json={"message": "2+2", "thread_id": "g4"},
        headers={"X-Capability-Token": tok},
    )
    frame = client.get("/governance/chain").json()["records"][-1]["frame"]
    assert (
        frame["allowed"] is False and frame["executed"] is False and "capability" in frame["reason"]
    )


def test_agent_lamport_merges_incoming_timestamp(client):
    r = client.post("/lg/agent/run", json={"message": "hi", "lamport": 100})
    assert r.headers["X-Lamport"] == "101"


def test_require_token_rejects_anonymous(client, monkeypatch):
    monkeypatch.setenv("GOVERNANCE_REQUIRE_TOKEN", "true")
    config.get_settings.cache_clear()
    governance.reset_engine()
    assert client.post("/lg/agent/run", json={"message": "hi"}).status_code == 401


def test_dispatch_returns_202_immediately_and_records(client):
    r = client.post(
        "/governance/dispatch",
        json={"thread_id": "d1", "target": "agent-b", "payload": {"task": "x"}, "lamport": 7},
    )
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "accepted" and body["lamport"] == 8
    chain = client.get("/governance/chain").json()
    frame = chain["records"][-1]["frame"]
    assert frame["tool"] == "agent-b" and frame["lamport"] == 8 and frame["executed"] is False


def test_state_and_chain_endpoints_are_read_only(client):
    client.post("/governance/dispatch", json={"thread_id": "d2", "target": "agent-b"})
    before = client.get("/governance/state").json()
    for _ in range(3):
        client.get("/governance/state")
        client.get("/governance/chain")
    assert client.get("/governance/state").json() == before
    for method in ("post", "put", "delete", "patch"):
        assert getattr(client, method)("/governance/state").status_code == 405
        assert getattr(client, method)("/governance/chain").status_code == 405


def test_chain_endpoint_reports_verification_and_limit(client):
    for i in range(3):
        client.post("/governance/dispatch", json={"thread_id": f"c{i}", "target": "b"})
    body = client.get("/governance/chain?limit=2").json()
    assert body["valid"] and body["length"] == 3 and len(body["records"]) == 2
    assert client.get("/governance/chain?limit=0").status_code == 422


def test_chain_endpoint_flags_tampering(client):
    client.post("/governance/dispatch", json={"thread_id": "t", "target": "b"})
    client.post("/governance/dispatch", json={"thread_id": "t", "target": "b"})
    governance.get_engine().chain._records[0]["frame"]["tool"] = "forged"
    body = client.get("/governance/chain").json()
    assert body["valid"] is False and body["first_bad_index"] == 0


NEW_RULES = {
    "version": "v2",
    "rules": [
        {"name": "destructive", "bounds": {"destructive": [0.5, 1], "tool_risk": [0.5, 1]}},
        {"name": "exfil", "bounds": {"exfiltration": [0.5, 1], "tool_risk": [0.3, 1]}},
        {"name": "inject", "bounds": {"injection": [0.5, 1], "tool_risk": [0.5, 1]}},
    ],
}
BLIND_RULES = {
    "version": "blind",
    "rules": [{"name": "n", "bounds": {"destructive": [1, 1], "tool_risk": [0, 0]}}],
}


def test_replay_endpoint_does_not_stage(client):
    ok = client.post("/governance/replay", json={"rules": NEW_RULES}).json()
    assert ok["eligible"] and ok["all_hazards_clamped"] and ok["false_positives"] == 0
    bad = client.post("/governance/replay", json={"rules": BLIND_RULES}).json()
    assert not bad["eligible"]
    assert client.get("/governance/state").json()["pending_rules_version"] is None


def test_rule_swap_requires_admin_scope_and_replay_gate(client):
    payload = {"rules": NEW_RULES}
    assert client.post("/governance/rules", json=payload).status_code == 401
    weak = _auth().mint("agent", ["*"], ["agent:run"])
    assert (
        client.post(
            "/governance/rules", json=payload, headers={"X-Capability-Token": weak}
        ).status_code
        == 403
    )
    admin = _auth().mint("ops", ["*"], ["governance:admin"])
    hdr = {"X-Capability-Token": admin}
    rejected = client.post("/governance/rules", json={"rules": BLIND_RULES}, headers=hdr)
    assert rejected.status_code == 409
    assert (
        client.post(
            "/governance/rules",
            json={"rules": {"rules": [{"name": "x", "bounds": {"nope": [0, 1]}}]}},
            headers=hdr,
        ).status_code
        == 422
    )
    ok = client.post("/governance/rules", json=payload, headers=hdr)
    assert ok.status_code == 200 and ok.json()["staged"] == "v2"
    assert client.get("/governance/state").json()["pending_rules_version"] == "v2"
    client.post("/governance/dispatch", json={"thread_id": "s", "target": "b"})  # next event
    assert client.get("/governance/state").json()["rules_version"] == "v2"
