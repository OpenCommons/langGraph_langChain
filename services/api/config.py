import json
import os
import pathlib
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings

# Registry id of the stack-wide default chat model (config/models.registry.json).
DEFAULT_MODEL_ID = "llama3.1-8b"
# Ollama tag it maps to, used only if the registry cannot be read.
_DEFAULT_MODEL_FALLBACK_TAG = "llama3.1:8b"


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    redis_url: str = "redis://localhost:4379"

    # Single source of truth for the default chat model: a registry id (or a
    # raw Ollama tag). LLM_MODEL, when set, is an explicit Ollama-tag override;
    # left empty it is derived from DEFAULT_MODEL.
    default_model: str = DEFAULT_MODEL_ID
    llm_model: str = ""
    # Chat-model provider: "ollama" (default, local) or any langchain
    # init_chat_model provider — openai, anthropic, google_vertexai,
    # google_genai, azure_openai, ... For a non-ollama provider LLM_MODEL must
    # be that provider's model name; keys come from the provider's own env var
    # (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...). Nothing is hard-coded.
    llm_provider: str = "ollama"
    # Optional endpoint override for non-ollama providers (gateways, proxies).
    llm_base_url: str = ""
    embed_model: str = "ternary-bonsai:4"
    # Output dimension of embed_model. Must match the existing Qdrant collection;
    # recreate the collection if you swap to a model with a different dim.
    embed_dim: int = 768
    collection_name: str = "localai_docs"

    # Model registry (config/models.registry.json). Empty = auto-resolve; see
    # core.model_registry.registry_path for the search order.
    models_registry_path: str = ""

    # RAG retrieval
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.4

    # LangGraph
    graph_recursion_limit: int = 25
    # SQLite file holding thread-scoped graph checkpoints (":memory:" for tests).
    checkpoint_db_path: str = "state/checkpoints.sqlite"
    # Persistence backend for graph checkpoints and the cross-thread store:
    # "sqlite" (default; store is in-memory) or "postgres" (PostgresSaver +
    # PostgresStore, needs DATABASE_URL).
    checkpoint_backend: str = "sqlite"
    database_url: str = ""
    postgres_pool_size: int = 10
    # Create / migrate the Postgres tables at API startup. Turn off when a
    # one-shot `make db-setup` job owns migrations (e.g. several replicas).
    postgres_auto_setup: bool = True

    # LangSmith tracing (optional). Exported to the environment at startup,
    # where LangChain reads it; off unless langsmith_tracing is true.
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "localaistack"
    # Tracing backend: "" / "none", "langsmith" or "otel" (OpenTelemetry +
    # OpenInference; endpoint via OTEL_EXPORTER_OTLP_ENDPOINT). LANGSMITH_TRACING=true
    # with no TRACING_BACKEND keeps selecting langsmith.
    tracing_backend: str = ""

    # Governance (OEE completeness layer; docs/GOVERNANCE.md)
    governance_enabled: bool = True
    # HMAC secret for capability tokens. Empty = a random per-process secret
    # (tokens die with the process and cannot be shared between replicas).
    governance_token_secret: str = ""
    # When true a capability token is mandatory; otherwise requests without one
    # get the default token below.
    governance_require_token: bool = False
    governance_default_tools: str = "*"
    governance_default_scopes: str = "agent:run,handoff:dispatch"
    # JSONL file for the Merkle provenance chain ("" = in-memory only).
    governance_chain_path: str = ""
    governance_policy_path: str = ""
    governance_klines_path: str = ""
    governance_kline_max: int = 1000
    governance_max_false_positives: int = 0
    governance_predictive_enabled: bool = False
    governance_predictive_threshold: float = 0.5
    governance_predictive_horizon: int = 3
    governance_predictive_margin: float = 0.1

    # Reality Engine stack URLs (PE = Perception Engine, RE = Reality Engine)
    # Docker: set to http://host.docker.internal:<port>
    # Local dev: http://localhost:<port>
    pe_url: str = "http://localhost:3004"
    re_url: str = "http://localhost:3000"

    # Personal health domain
    # Separate Qdrant collection for health knowledge documents.
    health_collection_name: str = "health_docs"
    # Set HEALTH_CONTEXT_ENABLED=true to automatically inject the current health
    # state into every chat system prompt. Can also be enabled per-request via
    # ChatRequest.health_context=true or the X-Health-Context: enabled header.
    health_context_enabled: bool = False
    # Seconds between HealthKit scope reconciliations against each running PE
    # (core/health_scope.py). 0 disables the follower.
    health_scope_interval_s: float = 30.0

    log_level: str = "info"

    @model_validator(mode="after")
    def _resolve_llm_model(self) -> "Settings":
        if not self.llm_model and self.llm_provider.lower() == "ollama":
            self.llm_model = resolve_model_tag(self.default_model, self.models_registry_path)
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"  # tolerate env vars from other services (WEBUI_SECRET_KEY, etc.)


def resolve_model_tag(name: str, registry_path: str = "") -> str:
    """Map a registry id (``llama3.1-8b``) to its Ollama tag (``llama3.1:8b``).

    A name the registry does not know is assumed to already be a tag. Reads the
    registry file directly: this runs while Settings is being built, so it must
    not call get_settings().
    """
    try:
        from core.model_registry import _default_paths

        override = registry_path or os.getenv("MODELS_REGISTRY_PATH", "")
        candidates = (pathlib.Path(override),) if override else _default_paths()
        path = next((p for p in candidates if p.is_file()), None)
        if path is not None:
            for m in json.loads(path.read_text()).get("models", []):
                if m.get("id") == name:
                    return m["tag"]
    except (OSError, ValueError, KeyError):
        pass
    return _DEFAULT_MODEL_FALLBACK_TAG if name == DEFAULT_MODEL_ID else name


@lru_cache
def get_settings() -> Settings:
    return Settings()
