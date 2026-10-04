"""LangChain features, exercised with a fake chat model (no live LLM)."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.vectorstores import InMemoryVectorStore

import config
from chains import memory, rag
from chains.lcel import Extraction, qa_chain, structured_chain, summarize_chain
from chains.tools import calculator, make_retrieval_tool, safe_calculate
from core import llm_factory
from routers import langchain_api
from tests.fakes import fake_llm, tool_call


def _store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(DeterministicFakeEmbedding(size=16))
    rag.ingest(
        store,
        [
            Document(page_content="Qdrant is the vector database.", metadata={"source": "a.md"}),
            Document(page_content="Redis caches things.", metadata={"source": "b.md"}),
        ],
    )
    return store


# ── default model ────────────────────────────────────────────────────────────


def test_default_model_is_llama31_8b(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    s = config.Settings(_env_file=None)
    assert s.default_model == "llama3.1-8b"
    assert s.llm_model == "llama3.1:8b"


def test_default_model_env_and_llm_override(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL", "llama3.2-3b")
    assert config.Settings(_env_file=None).llm_model == "llama3.2:3b"
    monkeypatch.setenv("LLM_MODEL", "custom:1b")
    assert config.Settings(_env_file=None).llm_model == "custom:1b"


def test_factory_builds_default_and_named_model(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    config.get_settings.cache_clear()
    assert llm_factory.get_chat_model().model == "llama3.1:8b"
    assert llm_factory.get_chat_model("mistral-7b").model == "mistral:7b-instruct-q4_K_M"
    assert llm_factory.get_chat_model("raw:tag").model == "raw:tag"
    config.get_settings.cache_clear()


# ── chains ───────────────────────────────────────────────────────────────────


def test_summarize_and_qa_chains():
    assert summarize_chain(fake_llm("short")).invoke({"text": "long text"}) == "short"
    assert qa_chain(fake_llm("42")).invoke({"context": "c", "question": "q"}) == "42"


def test_structured_chain_returns_pydantic():
    payload = json.dumps({"title": "T", "summary": "S", "keywords": ["a", "b"]})
    out = structured_chain(fake_llm(payload)).invoke({"text": "whatever"})
    assert out == Extraction(title="T", summary="S", keywords=["a", "b"])


# ── tools ────────────────────────────────────────────────────────────────────


def test_calculator_tool_and_safety():
    assert safe_calculate("(12 + 3) * 4 / 5") == 12
    assert calculator.invoke({"expression": "2 ** 5"}) == "32"
    assert calculator.invoke({"expression": "__import__('os')"}).startswith("Error")
    assert calculator.invoke({"expression": "1/0"}).startswith("Error")
    assert calculator.invoke({"expression": "9 ** 9999"}).startswith("Error")
    assert calculator.invoke({"expression": "(10**6+1) ** 1000"}).startswith("Error")


def test_retrieval_tool_returns_sources():
    tool = make_retrieval_tool(_store().as_retriever(search_kwargs={"k": 1}))
    assert "[Source:" in tool.invoke({"query": "vector database"})


# ── rag ──────────────────────────────────────────────────────────────────────


def test_split_and_load(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("word " * 500)
    docs = rag.load_documents(f)
    assert docs[0].metadata["source"] == "doc.txt"
    assert len(rag.split_documents(docs, chunk_size=200, chunk_overlap=20)) > 5


def test_rag_chain_answers_with_sources():
    out = rag.rag_chain(_store().as_retriever(search_kwargs={"k": 1}), fake_llm("Qdrant")).invoke(
        "What is the vector database?"
    )
    assert out["answer"] == "Qdrant"
    assert len(out["sources"]) == 1


# ── memory ───────────────────────────────────────────────────────────────────


def test_chat_memory_keeps_history_per_session():
    memory.clear_history("s1")
    chain = memory.chat_with_memory(fake_llm("hello", "again"))
    cfg = {"configurable": {"session_id": "s1"}}
    chain.invoke({"input": "hi"}, cfg)
    chain.invoke({"input": "more"}, cfg)
    assert [m.content for m in memory.get_history("s1").messages] == [
        "hi",
        "hello",
        "more",
        "again",
    ]
    assert memory.get_history("other").messages == []


# ── HTTP ─────────────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(langchain_api.router)
    llm = fake_llm("ok")
    app.dependency_overrides[langchain_api.get_llm] = lambda: llm
    app.dependency_overrides[langchain_api.get_retriever] = lambda: _store().as_retriever()
    app.state.llm = llm
    return TestClient(app)


def test_chat_endpoint_and_stream(client):
    memory.clear_history("api")
    r = client.post("/lc/chat", json={"message": "hi", "session_id": "api"})
    assert r.json() == {"session_id": "api", "reply": "ok"}
    r = client.post("/lc/chat", json={"message": "hi", "session_id": "api", "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    assert '"token"' in r.text and "[DONE]" in r.text


def test_tools_endpoint_runs_tool_call(client):
    client.app.dependency_overrides[langchain_api.get_llm] = lambda: fake_llm(
        tool_call("calculator", {"expression": "6*7"}), "The answer is 42"
    )
    body = client.post("/lc/tools", json={"prompt": "6*7?"}).json()
    assert body["tool_calls"][0]["output"] == "42"
    assert body["answer"] == "The answer is 42"


def test_rag_endpoint(client):
    body = client.post("/lc/rag", json={"question": "vector?"}).json()
    assert body["answer"] == "ok" and body["sources"]
