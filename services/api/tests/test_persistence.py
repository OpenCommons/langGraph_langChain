"""Checkpointer / store backend selection, Postgres wiring, and the health probe."""

import asyncio
import os
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore

import config
from graphs import checkpoint, store
from routers import health, langgraph_api


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for name in ("CHECKPOINT_BACKEND", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CHECKPOINT_DB_PATH", ":memory:")
    config.get_settings.cache_clear()
    checkpoint.reset_checkpointer()
    store.reset_store()
    yield
    config.get_settings.cache_clear()
    checkpoint.reset_checkpointer()
    store.reset_store()


def _postgres(monkeypatch, url="postgresql://db:5432/x"):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("DATABASE_URL", url)
    config.get_settings.cache_clear()


def test_default_backend_is_sqlite_with_in_memory_store():
    assert isinstance(checkpoint.get_checkpointer(), SqliteSaver)
    assert isinstance(store.get_store(), InMemoryStore)
    assert checkpoint.setup_persistence() == "sqlite"


def test_postgres_requires_database_url(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_BACKEND", "postgres")
    config.get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        checkpoint.get_checkpointer()


def test_postgres_backend_builds_postgres_saver_and_store_over_one_pool(monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore

    _postgres(monkeypatch)
    pool = MagicMock()
    monkeypatch.setattr(checkpoint, "_pool", pool)
    assert isinstance(checkpoint.get_checkpointer(), PostgresSaver)
    assert isinstance(store.get_store(), PostgresStore)
    assert checkpoint.get_checkpointer() is checkpoint.get_checkpointer()


def test_setup_persistence_runs_both_migrations(monkeypatch):
    _postgres(monkeypatch)
    saver, pg_store = MagicMock(), MagicMock()
    monkeypatch.setattr(checkpoint, "get_checkpointer", lambda: saver)
    monkeypatch.setattr(store, "get_store", lambda: pg_store)
    assert checkpoint.setup_persistence() == "postgres"
    saver.setup.assert_called_once()
    pg_store.setup.assert_called_once()


def test_health_probes_postgres_only_when_configured(monkeypatch):
    class _Boom:
        def connection(self, **_):
            raise OSError("no route to host")

        def close(self):
            pass

    async def run():
        monkeypatch.setattr(health, "_check_pe", _noop)
        monkeypatch.setattr(health, "_check_re", _noop)
        monkeypatch.setattr(health.engine_fanout, "interaction_targets", lambda: [])
        monkeypatch.setattr(
            health, "resolve_bridge_targets", lambda: {"pe_url": "http://x", "re_url": "http://y"}
        )
        return await health.health()

    async def _noop(*_a, **_k):
        return {"status": "ok"}

    assert "postgres" not in asyncio.run(run())["services"]
    _postgres(monkeypatch)
    monkeypatch.setattr(checkpoint, "_pool", _Boom())
    body = asyncio.run(run())
    assert body["services"]["postgres"].startswith("error") and body["status"] == "degraded"


def test_memory_endpoints_share_a_store_across_threads():
    app = FastAPI()
    app.include_router(langgraph_api.router)
    client = TestClient(app)
    client.put("/lg/memory/u1", json={"key": "tone", "value": {"v": "brief"}})
    assert client.get("/lg/memory/u1").json()["memories"] == {"tone": {"v": "brief"}}
    assert client.get("/lg/memory/u2").json()["memories"] == {}


@pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="set TEST_DATABASE_URL to run against a real Postgres",
)
def test_live_postgres_state_survives_a_restart(monkeypatch):
    from langchain_core.messages import HumanMessage

    from chains.tools import base_tools
    from graphs.react_agent import build_react_agent
    from tests.fakes import fake_llm

    _postgres(monkeypatch, os.environ["TEST_DATABASE_URL"])
    checkpoint.setup_persistence()
    cfg = {"configurable": {"thread_id": "live-pg-thread"}}
    graph = build_react_agent(fake_llm("ok"), base_tools(), checkpoint.get_checkpointer())
    graph.invoke({"messages": [HumanMessage(content="hi")]}, cfg)
    store.get_store().put(("memories", "live"), "k", {"v": 1})

    checkpoint.reset_checkpointer()  # a new process would start from here
    store.reset_store()
    graph = build_react_agent(fake_llm("ok"), base_tools(), checkpoint.get_checkpointer())
    assert len(graph.get_state(cfg).values["messages"]) >= 2
    assert store.get_store().get(("memories", "live"), "k").value == {"v": 1}
