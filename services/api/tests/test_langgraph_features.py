"""LangGraph features, exercised with a fake chat model (no live LLM)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import config
from chains.tools import base_tools
from graphs import checkpoint
from graphs.approval_graph import build_approval_graph
from graphs.react_agent import build_react_agent, route_after_agent
from graphs.supervisor_graph import build_supervisor_graph, initial_state, parse_route
from routers import langgraph_api
from tests.fakes import fake_llm, tool_call


def _cfg(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


# ── ReAct agent, routing, checkpointing ─────────────────────────────────────


def test_route_after_agent():
    from langchain_core.messages import AIMessage

    assert route_after_agent({"messages": [tool_call("calculator", {})]}) == "tools"
    assert route_after_agent({"messages": [AIMessage(content="done")]}) == "end"


def test_react_agent_calls_tool_then_answers():
    llm = fake_llm(tool_call("calculator", {"expression": "6*7"}), "It is 42")
    graph = build_react_agent(llm, base_tools())
    out = graph.invoke({"messages": [HumanMessage(content="6*7?")]})
    kinds = [m.type for m in out["messages"]]
    assert kinds == ["human", "ai", "tool", "ai"]
    assert out["messages"][2].content == "42"
    assert out["messages"][-1].content == "It is 42"


def test_checkpointer_persists_per_thread():
    saver = InMemorySaver()
    graph = build_react_agent(fake_llm("first", "second"), base_tools(), saver)
    graph.invoke({"messages": [HumanMessage(content="a")]}, _cfg("t1"))
    graph.invoke({"messages": [HumanMessage(content="b")]}, _cfg("t1"))
    assert len(graph.get_state(_cfg("t1")).values["messages"]) == 4
    assert graph.get_state(_cfg("t2")).values == {}


def test_graph_events_stream_updates():
    graph = build_react_agent(
        fake_llm(tool_call("calculator", {"expression": "1+1"}), "2"), base_tools()
    )
    nodes = [
        next(iter(u))
        for u in graph.stream({"messages": [HumanMessage(content="1+1")]}, stream_mode="updates")
    ]
    assert nodes == ["agent", "tools", "agent"]


# ── human in the loop ───────────────────────────────────────────────────────


def test_interrupt_pauses_then_resumes_approved():
    graph = build_approval_graph(fake_llm("step 1, step 2"), InMemorySaver())
    out = graph.invoke({"request": "deploy"}, _cfg("h1"))
    assert "result" not in out
    snapshot = graph.get_state(_cfg("h1"))
    assert snapshot.next == ("approval",)
    assert snapshot.tasks[0].interrupts[0].value["proposal"] == "step 1, step 2"
    out = graph.invoke(Command(resume={"approved": True}), _cfg("h1"))
    assert out["result"] == "executed: step 1, step 2"


def test_interrupt_resume_rejected():
    graph = build_approval_graph(fake_llm("plan"), InMemorySaver())
    graph.invoke({"request": "x"}, _cfg("h2"))
    out = graph.invoke(Command(resume={"approved": False, "feedback": "too risky"}), _cfg("h2"))
    assert out["result"] == "cancelled: too risky"


# ── supervisor ──────────────────────────────────────────────────────────────


def test_parse_route():
    assert parse_route(" Math. ") == "math"
    assert parse_route("writer") == "writer"
    assert parse_route("FINISH") == "FINISH"
    assert parse_route("???") == "FINISH"


def test_supervisor_routes_to_workers_then_finishes():
    # supervisor → math; math agent → tool call, answer; supervisor → writer;
    # writer; supervisor → FINISH
    llm = fake_llm(
        "math",
        tool_call("calculator", {"expression": "2+2"}),
        "4",
        "writer",
        "Four.",
        "FINISH",
    )
    out = build_supervisor_graph(llm).invoke(initial_state("add 2+2 and write it up"))
    texts = [m.content for m in out["messages"]]
    assert texts == ["add 2+2 and write it up", "[math] 4", "[writer] Four."]
    assert out["steps"] == 3


def test_supervisor_step_limit():
    out = build_supervisor_graph(fake_llm("writer"), max_steps=2).invoke(initial_state("go"))
    assert out["steps"] == 2


# ── HTTP + SQLite checkpointer ──────────────────────────────────────────────


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_DB_PATH", ":memory:")
    config.get_settings.cache_clear()
    checkpoint.reset_checkpointer()
    app = FastAPI()
    app.include_router(langgraph_api.router)
    app.dependency_overrides[langgraph_api.get_llm] = lambda: fake_llm("hello there")
    yield TestClient(app)
    checkpoint.reset_checkpointer()
    config.get_settings.cache_clear()


def test_agent_run_state_and_stream(client):
    body = client.post("/lg/agent/run", json={"message": "hi", "thread_id": "T"}).json()
    assert body == {"thread_id": "T", "answer": "hello there"}
    state = client.get("/lg/agent/state/T").json()
    assert [m["type"] for m in state["messages"]] == ["human", "ai"]
    r = client.post("/lg/agent/stream", json={"message": "again", "thread_id": "T"})
    assert '"agent"' in r.text and "[DONE]" in r.text


def test_agent_history_returns_checkpoint_snapshots(client):
    for message in ("first", "second"):
        client.post("/lg/agent/run", json={"message": message, "thread_id": "H"})

    history = client.get("/lg/agent/history/H").json()

    assert history["thread_id"] == "H"
    assert history["history"]
    assert history["history"][0]["checkpoint_id"]
    assert any(
        snapshot["values"]["messages"][-1]["content"] == "second"
        for snapshot in history["history"]
        if snapshot["values"].get("messages")
    )
    assert len(client.get("/lg/agent/history/H?limit=1").json()["history"]) == 1
    assert client.get("/lg/agent/history/H?limit=0").status_code == 422


def test_approval_http_flow(client):
    started = client.post("/lg/approval/start", json={"request": "r", "thread_id": "A"}).json()
    assert started["status"] == "awaiting_approval"
    assert started["interrupts"][0]["proposal"] == "hello there"
    done = client.post("/lg/approval/resume", json={"thread_id": "A", "approved": True}).json()
    assert done["status"] == "done" and done["result"].startswith("executed")


def test_supervisor_http(client):
    client.app.dependency_overrides[langgraph_api.get_llm] = lambda: fake_llm("FINISH")
    body = client.post("/lg/supervisor/run", json={"task": "nothing"}).json()
    assert body["steps"] == 1
