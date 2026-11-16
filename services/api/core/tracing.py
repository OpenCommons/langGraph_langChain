"""Configurable tracing, selected at runtime by ``TRACING_BACKEND``.

``langsmith``  exports LANGSMITH_* so LangChain/LangGraph report to LangSmith
               (needs LANGSMITH_API_KEY).
``otel``       OpenTelemetry spans via OpenInference's LangChain instrumentation,
               sent over OTLP/HTTP to ``OTEL_EXPORTER_OTLP_ENDPOINT`` (any
               collector, Langfuse, Phoenix, Tempo, ...). Optional packages:
               services/api/requirements-tracing.txt.
``none``/empty off. ``LANGSMITH_TRACING=true`` with no backend implies ``langsmith``.

No endpoint is hard-coded; a backend whose configuration or packages are missing
logs a warning and tracing stays off — it never stops the API from starting.
"""

from __future__ import annotations

import os

import structlog

from config import get_settings

log = structlog.get_logger()

_otel_configured = False


def selected_backend() -> str:
    s = get_settings()
    backend = s.tracing_backend.strip().lower()
    if backend in ("", "none"):
        return "langsmith" if s.langsmith_tracing else "none"
    return backend


def _setup_otel(span_exporter=None) -> bool:
    global _otel_configured
    if _otel_configured:
        return True
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_HEADERS"):
        if not os.environ.get(name, "x").strip():  # empty value would override SDK defaults
            os.environ.pop(name, None)
    if span_exporter is None and not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        log.warning("tracing.otel_disabled", reason="OTEL_EXPORTER_OTLP_ENDPOINT is not set")
        return False
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        if span_exporter is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            processor = BatchSpanProcessor(OTLPSpanExporter())
        else:
            processor = SimpleSpanProcessor(span_exporter)
    except ImportError as exc:
        log.warning("tracing.otel_disabled", reason=f"missing package: {exc}")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": "localaistack-api"}))
    provider.add_span_processor(processor)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    _otel_configured = True
    return True


def setup_tracing(span_exporter=None) -> str | None:
    """Enable the selected backend. Returns its name, or ``None`` if tracing is off."""
    backend = selected_backend()
    if backend == "none":
        return None
    if backend == "langsmith":
        from core.llm_factory import apply_langsmith_env

        if apply_langsmith_env():
            return "langsmith"
        log.warning("tracing.langsmith_disabled", reason="LANGSMITH_API_KEY is not set")
        return None
    if backend == "otel":
        return "otel" if _setup_otel(span_exporter) else None
    log.warning("tracing.unknown_backend", backend=backend)
    return None
