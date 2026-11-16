# Model providers and tracing

Everything is selected by environment variables; no endpoint or key is hard-coded.

## Chat-model providers (`core/llm_factory.get_chat_model`)

| Variable | Meaning |
|---|---|
| `LLM_PROVIDER` | `ollama` (default) or any `init_chat_model` provider: `openai`, `anthropic`, `google_genai`, `google_vertexai`, `azure_openai`, … |
| `LLM_MODEL` | `ollama`: optional tag override (else derived from `DEFAULT_MODEL` via the model registry). Other providers: **required**, the provider's model name |
| `LLM_BASE_URL` | optional endpoint override for non-ollama providers (gateway / proxy) |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, … | read by the provider's own library |

Install only the integrations you use: `pip install -r services/api/requirements.txt -r services/api/requirements-providers.txt`.
A missing package raises an error naming that file. Embeddings and the Reality-Engine
bridge are unaffected (they stay on Ollama). Per-request model selection through the
model registry applies to `ollama` only. Not exercised against any hosted provider (no keys).

## Tracing (`core/tracing.setup_tracing`, called at API startup)

| `TRACING_BACKEND` | Behaviour | Needs |
|---|---|---|
| empty / `none` | off (`LANGSMITH_TRACING=true` alone still selects `langsmith`) | — |
| `langsmith` | exports `LANGSMITH_*` for LangChain/LangGraph | `LANGSMITH_API_KEY`, optional `LANGSMITH_PROJECT` |
| `otel` | OpenTelemetry spans via OpenInference's LangChain instrumentation, OTLP/HTTP | `OTEL_EXPORTER_OTLP_ENDPOINT` (+ optional `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_SERVICE_NAME`); `pip install -r services/api/requirements-tracing.txt` |

A backend whose key, endpoint or packages are missing logs a warning and tracing stays off;
the API still starts. Langfuse, Phoenix, Tempo etc. are reached through their OTLP endpoint.
`tests/test_providers_tracing.py` checks span emission with an in-memory exporter (skipped
when the optional packages are absent); nothing was sent to a hosted backend.
