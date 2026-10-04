"""
Chat-model factory and tracing hooks for the LangChain / LangGraph modules.

``get_chat_model`` is the only place a ChatOllama is built for the new
``chains`` and ``graphs`` modules. With no arguments it returns the stack
default (``DEFAULT_MODEL`` → llama3.1-8b → ``llama3.1:8b``); any registry id or
raw Ollama tag can be passed instead.
"""

from __future__ import annotations

import os
from typing import Any

import structlog
from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama

from config import get_settings, resolve_model_tag

log = structlog.get_logger()


def get_chat_model(
    model: str | None = None,
    *,
    temperature: float = 0.1,
    **kwargs: Any,
) -> ChatOllama:
    """Build a chat model from configuration. ``model`` is a registry id or tag."""
    s = get_settings()
    tag = resolve_model_tag(model, s.models_registry_path) if model else s.llm_model
    return ChatOllama(base_url=s.ollama_base_url, model=tag, temperature=temperature, **kwargs)


class LoggingCallbackHandler(BaseCallbackHandler):
    """Structured log line per LLM call and tool call (cheap local tracing)."""

    def on_llm_start(self, serialized, prompts, **kwargs):
        log.info("llm.start", prompts=len(prompts))

    def on_chat_model_start(self, serialized, messages, **kwargs):
        log.info("llm.start", batches=len(messages))

    def on_llm_end(self, response, **kwargs):
        log.info("llm.end", generations=len(response.generations))

    def on_llm_error(self, error, **kwargs):
        log.warning("llm.error", error=str(error))

    def on_tool_start(self, serialized, input_str, **kwargs):
        log.info("tool.start", tool=(serialized or {}).get("name"), input=input_str[:200])

    def on_tool_end(self, output, **kwargs):
        log.info("tool.end")


def default_callbacks() -> list[BaseCallbackHandler]:
    return [LoggingCallbackHandler()]


def run_config(thread_id: str | None = None, **extra: Any) -> dict[str, Any]:
    """RunnableConfig with logging callbacks, and a thread id for checkpointed graphs."""
    config: dict[str, Any] = {"callbacks": default_callbacks(), **extra}
    if thread_id is not None:
        config["configurable"] = {"thread_id": thread_id}
    return config


def apply_langsmith_env() -> bool:
    """Export LangSmith settings to the environment when tracing is enabled."""
    s = get_settings()
    if not (s.langsmith_tracing and s.langsmith_api_key):
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = s.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    return True
