# LangChain & LangGraph facilities

Both live in the existing FastAPI service (`services/api`), which already runs in
`docker compose` with health checks and Qdrant/Redis dependencies — no extra
container is needed.

## Default model

`DEFAULT_MODEL=llama3.1-8b` (a `config/models.registry.json` id) is the single
source of truth. `config.Settings` resolves it to the Ollama tag `llama3.1:8b`
(`config.resolve_model_tag`) and every service reads `settings.llm_model`.

| Variable | Meaning |
|---|---|
| `DEFAULT_MODEL` | registry id (or raw tag) of the default chat model; default `llama3.1-8b` |
| `LLM_MODEL` | optional Ollama-tag override; empty → derived from `DEFAULT_MODEL` |
| `CHECKPOINT_DB_PATH` | SQLite file for LangGraph checkpoints (`/app/state/checkpoints.sqlite` in compose) |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | optional LangSmith tracing (exported at startup) |

`make pull-model` pulls the default; any registered model works per request via
`core.llm_factory.get_chat_model("<id or tag>")`.

## Layout

| Path | Contents |
|---|---|
| `core/llm_factory.py` | `get_chat_model()` factory (ChatOllama), logging callback, LangSmith env export |
| `chains/lcel.py` | prompt templates; summarize, Q&A and Pydantic structured-output chains |
| `chains/tools.py` | `calculator` (safe AST evaluator), `current_time`, retrieval-tool wrapper |
| `chains/rag.py` | load → split → ingest → retriever chain (Qdrant via `core/vector_store.py`) |
| `chains/memory.py` | per-session chat history (`RunnableWithMessageHistory`) |
| `graphs/react_agent.py` | ReAct `StateGraph`: agent ⇄ tools with a conditional edge |
| `graphs/checkpoint.py` | SQLite checkpointer; `thread_id` selects the conversation |
| `graphs/approval_graph.py` | human-in-the-loop: `interrupt()` before executing a plan |
| `graphs/supervisor_graph.py` | supervisor routing to `math` / `writer` workers, bounded steps |
| `graphs/state.py` | shared message state schema and deterministic reducers |
| `routers/langchain_api.py`, `routers/langgraph_api.py` | HTTP surface |

The pre-existing `/graph/rag` and `/graph/agent` (Reality Engine-bound graphs)
and `/rag/*`, `/chat` are unchanged.

## Shared graph state and reducers

Graphs use `BaseGraphState` for messages and extend `StateFactory` for
domain-specific fields. State fields that may be merged from parallel updates
should use reducers whose result is independent of update order: set union,
numeric maximum, or a naturally ordered value. These reducers are
commutative, associative, and idempotent, so retries and merge ordering do not
change their result. `reduce_latest` selects the greatest value; timestamps
therefore select the newest timestamp, while unordered status values need an
explicit version or timestamp to express chronology.

`add_messages` is intentionally different: it preserves message order and is
not commutative. It remains appropriate for conversational history, whose order
is part of the data rather than a merge-independent aggregate.

## Endpoints

```bash
# LangChain
curl -s localhost:4000/lc/chat -H 'Content-Type: application/json' \
  -d '{"message":"hi","session_id":"s1","stream":true}'          # SSE tokens
curl -s localhost:4000/lc/summarize  -d '{"text":"..."}' -H 'Content-Type: application/json'
curl -s localhost:4000/lc/structured -d '{"text":"..."}' -H 'Content-Type: application/json'
curl -s localhost:4000/lc/tools      -d '{"prompt":"What is 6*7?"}' -H 'Content-Type: application/json'
curl -s localhost:4000/lc/rag        -d '{"question":"..."}' -H 'Content-Type: application/json'

# LangGraph
curl -s localhost:4000/lg/agent/run    -d '{"message":"What is 17*23?","thread_id":"t1"}' -H 'Content-Type: application/json'
curl -sN localhost:4000/lg/agent/stream -d '{"message":"...","thread_id":"t1"}' -H 'Content-Type: application/json'  # graph events (SSE)
curl -s localhost:4000/lg/agent/state/t1
curl -s localhost:4000/lg/approval/start  -d '{"request":"rotate keys","thread_id":"a1"}' -H 'Content-Type: application/json'
curl -s localhost:4000/lg/approval/resume -d '{"thread_id":"a1","approved":true}' -H 'Content-Type: application/json'
curl -s localhost:4000/lg/supervisor/run  -d '{"task":"add 2+2 and write it up"}' -H 'Content-Type: application/json'
```

`/lc/rag` needs documents ingested first (`make ingest FILE=...`).

## Persistence, governance and long-term memory

- `CHECKPOINT_BACKEND=postgres` + `DATABASE_URL` swaps the SQLite checkpointer for
  `PostgresSaver` and adds a `PostgresStore`; `PUT/GET /lg/memory/{user_id}` read and
  write cross-thread memory. See [HA_DEPLOYMENT.md](HA_DEPLOYMENT.md).
- Every `/lg/agent/*` tool call is verified over the whole thread and recorded by the
  [governance layer](GOVERNANCE.md); `X-Capability-Token` scopes the tools available.
- `LLM_PROVIDER` / `TRACING_BACKEND`: [PROVIDERS_AND_TRACING.md](PROVIDERS_AND_TRACING.md).

## Tests

`tests/test_langchain_features.py` and `tests/test_langgraph_features.py` use the
scripted fake chat model in `tests/fakes.py`, so no Ollama is required:

```bash
make test
pytest services/api/tests --ignore=services/api/tests/e2e
```
