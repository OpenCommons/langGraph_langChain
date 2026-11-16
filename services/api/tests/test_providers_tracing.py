"""Pluggable chat-model providers and tracing backends, chosen by env vars."""

import pytest
from langchain_ollama import ChatOllama

import config
from core import llm_factory, tracing

_ENV = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "TRACING_BACKEND",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in (*_ENV, "LANGSMITH_PROJECT"):
        # setenv-then-delenv: tracing writes os.environ directly, so have
        # monkeypatch restore (unset) these on teardown.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(tracing, "_otel_configured", False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _set(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    config.get_settings.cache_clear()


# ── providers ────────────────────────────────────────────────────────────────


def test_default_provider_is_ollama():
    assert isinstance(llm_factory.get_chat_model(), ChatOllama)


def test_non_ollama_provider_uses_init_chat_model(monkeypatch):
    _set(monkeypatch, LLM_PROVIDER="openai", LLM_MODEL="some-model", LLM_BASE_URL="http://gw/v1")
    seen = {}

    def fake_init(name, **kw):
        seen.update(name=name, **kw)
        return "model"

    monkeypatch.setattr("langchain.chat_models.init_chat_model", fake_init)
    assert llm_factory.get_chat_model() == "model"
    assert seen["name"] == "some-model" and seen["model_provider"] == "openai"
    assert seen["base_url"] == "http://gw/v1"
    llm_factory.get_chat_model("other-model")
    assert seen["name"] == "other-model"


def test_non_ollama_provider_requires_explicit_model(monkeypatch):
    _set(monkeypatch, LLM_PROVIDER="anthropic")
    assert config.get_settings().llm_model == ""  # not derived from the Ollama registry
    with pytest.raises(ValueError, match="LLM_MODEL"):
        llm_factory.get_chat_model()


def test_missing_provider_package_gives_actionable_error(monkeypatch):
    _set(monkeypatch, LLM_PROVIDER="openai", LLM_MODEL="m")

    def boom(*a, **k):
        raise ImportError("No module named langchain_openai")

    monkeypatch.setattr("langchain.chat_models.init_chat_model", boom)
    with pytest.raises(RuntimeError, match="requirements-providers.txt"):
        llm_factory.get_chat_model()


# ── tracing ──────────────────────────────────────────────────────────────────


def test_tracing_is_off_by_default():
    assert tracing.selected_backend() == "none" and tracing.setup_tracing() is None


def test_legacy_langsmith_flag_selects_langsmith(monkeypatch):
    _set(monkeypatch, LANGSMITH_TRACING="true", LANGSMITH_API_KEY="k")
    assert tracing.selected_backend() == "langsmith"
    assert tracing.setup_tracing() == "langsmith"


def test_langsmith_backend_needs_a_key(monkeypatch):
    _set(monkeypatch, TRACING_BACKEND="langsmith")
    assert tracing.setup_tracing() is None
    _set(monkeypatch, TRACING_BACKEND="langsmith", LANGSMITH_API_KEY="k")
    assert tracing.setup_tracing() == "langsmith"


def test_otel_needs_an_endpoint_and_never_hard_codes_one(monkeypatch):
    _set(monkeypatch, TRACING_BACKEND="otel")
    assert tracing.setup_tracing() is None


def test_unknown_backend_is_ignored(monkeypatch):
    _set(monkeypatch, TRACING_BACKEND="nope")
    assert tracing.setup_tracing() is None


def test_otel_emits_spans_for_langchain_runs(monkeypatch):
    pytest.importorskip("openinference.instrumentation.langchain")
    sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
    from openinference.instrumentation.langchain import LangChainInstrumentor

    from tests.fakes import fake_llm

    _set(monkeypatch, TRACING_BACKEND="otel")
    exporter = sdk_export.InMemorySpanExporter()
    try:
        assert tracing.setup_tracing(span_exporter=exporter) == "otel"
        fake_llm("traced").invoke("hello")
        assert exporter.get_finished_spans()
    finally:
        LangChainInstrumentor().uninstrument()
