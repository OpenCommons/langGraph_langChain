# localAIStack

A local AI stack that runs alongside any active RealityEngine runtime (CPP,
Scala, LSP). It hosts the RAG / LangGraph orchestration API (FastAPI +
Strawberry GraphQL), a Qdrant vector store shared with the Reality Engine,
Redis for LangGraph checkpointing, Open WebUI for chatting with native Ollama
models, and a Loki + Prometheus + Grafana trio for centralized log and metrics
monitoring.

## Service URLs

| Service | URL | Description |
|---|---|---|
| API (REST + GraphQL) | http://localhost:4000 | RAG / LangChain (`/lc`) / LangGraph (`/lg`) orchestration |
| API Docs | http://localhost:4000/docs | OpenAPI explorer |
| Open WebUI | http://localhost:4080 | Chat UI against native Ollama |
| Qdrant | http://localhost:4333/dashboard | Vector DB (unified with Reality Engine) |
| Ollama | http://localhost:11434 | Native LLM runtime (Metal on macOS) |
| **Grafana** | **http://localhost:4002** | **localAIStack Overview dashboard** |
| **Loki** | **http://localhost:4100** | **Log aggregation API** |
| **Prometheus** | **http://localhost:4090** | **Metrics scraping (Loki + API)** |

## Pinned service versions

All observability and vector services use explicit, stable image pins to ensure
deterministic builds:

| Service | Image | Pinned version |
|---|---|---|
| Qdrant | `qdrant/qdrant` | `v1.14.1` |
| Loki | `grafana/loki` | `3.4.2` |
| Grafana | `grafana/grafana` | `11.6.1` |
| Prometheus | `prom/prometheus` | `v2.53.4` (LTS) |
| Redis | `redis` | `7.4-alpine` |

## Quick start

```bash
./scripts/setup.sh       # one-time: pull Ollama models, register ternary-bonsai:4
./scripts/start.sh       # start everything (installs Loki Docker plugin if missing)
./scripts/stop.sh        # stop everything
```

`start.sh` ensures the `loki` Docker plugin is installed+enabled before bringing
up the compose stack — the qdrant, redis, api, and open-webui containers all
use the Loki log driver and will fail to start without it.

Both `setup.sh` and `start.sh` (plus `make up`) enforce a pinned Ollama version:
**v0.32.0**. They parse `ollama --version` output (including formats like
`ollama version is 0.32.0`) and will auto-install/upgrade to v0.32.0 on
macOS/Linux via the official installer when needed. If automatic install is not
possible, they fail with a concise manual-install instruction.

## Model registry

`config/models.registry.json` is the single source of truth for every local
model the stack knows how to run. It is *declarative* — listing a model does not
download it. Three separate facts get joined at read time:

| Fact | Lives in | Means |
|---|---|---|
| available | `config/models.registry.json` | the stack has curated metadata for it |
| selected | `.env` (`DEFAULT_MODEL`, optional `LLM_MODEL` tag override / `EMBED_MODEL`) | the API will use it |
| installed | the Ollama host (`/api/tags`) | the weights are actually pulled |

```bash
make models                              # registry table, * marks installed
make model-pull ID=nemotron-3-nano-4b    # pull a registered model
make model-info ID=nemotron-3-nano-4b    # full entry (API must be up)
curl -s localhost:4000/models | python3 -m json.tool
curl -s 'localhost:4000/models?role=embedding' | python3 -m json.tool
```

`GET /models` returns the join, so the three common failure modes are
distinguishable in one response: `installed: false` (never pulled),
`unregistered_selections` (a `.env` tag no entry claims — usually a typo), and
`fits_host: false` (bigger than this machine's RAM). `setup.sh` runs the same
check against your `.env` selections and warns without blocking — overriding the
registry is legitimate while testing.

### Registered models

| ID | Tag | Role | Size | Context | Notes |
|---|---|---|---|---|---|
| `llama3.1-8b` | `llama3.1:8b` | llm | 4.9 GB | 128K | **Default** — general chat/RAG, tool calling |
| `llama3.2-3b` | `llama3.2:3b` | llm | 2.0 GB | 128K | Fast smoke tests |
| `mistral-7b` | `mistral:7b-instruct-q4_K_M` | llm | 4.4 GB | 32K | RAG grading A/B |
| `phi3.5-mini` | `phi3.5:3.8b` | llm | 2.2 GB | 128K | No tool-calling |
| `nemotron-3-nano-4b` | `nemotron-3-nano:4b` | llm | 2.8 GB | 256K | Alternative agentic model — tools + thinking |
| `nemotron-3-nano-30b` | `nemotron-3-nano:30b` | llm | 24 GB | 1M | Needs ≥48 GB RAM |
| `nomic-embed-text` | `nomic-embed-text` | embedding | 274 MB | 2K | Default embedder, 768-dim |
| `ternary-bonsai-4b` | `ternary-bonsai:4` | embedding | 546 MB | 32K | **broken** — see known issue below |

**Default model: `llama3.1-8b`.** `DEFAULT_MODEL` (a registry id, default
`llama3.1-8b`) is the single source of truth; every service resolves it to the
Ollama tag `llama3.1:8b` through the registry. Set `LLM_MODEL=<ollama tag>` only
to override with a tag directly. `make pull-model` (and `setup.sh`) pull it.

```bash
make pull-model                      # pulls DEFAULT_MODEL (llama3.1:8b)
DEFAULT_MODEL=nemotron-3-nano-4b make pull-model   # or switch the default
curl -s -X POST localhost:4000/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

**NVIDIA Nemotron 3 Nano 4B** remains registered as an alternative agentic
model (half the footprint, 256K context). The 30B variant is registered but not
for this baseline: `nemotron-3-nano:latest` resolves to it, which is why the
registry pins explicit `:4b` / `:30b` tags.

## LangChain & LangGraph

The API exposes real LangChain (`/lc/*`) and LangGraph (`/lg/*`) facilities on
top of the default model: LCEL chains, tool calling, RAG, chat memory, SSE
streaming, a checkpointed ReAct agent, human-in-the-loop interrupts and a
supervisor workflow.

```bash
make langchain-demo     # summarization chain + tool calling
make langgraph-demo     # checkpointed ReAct agent (thread "demo")
```

See [`docs/LANGCHAIN_LANGGRAPH.md`](docs/LANGCHAIN_LANGGRAPH.md) for the
architecture, configuration and endpoint examples.

### Endpoint reference

| Endpoint | Facility |
|---|---|
| `POST /lc/chat` | LangChain chat with per-session memory (`stream: true` → SSE) |
| `POST /lc/summarize` · `/lc/structured` | LCEL summarization · Pydantic structured output |
| `POST /lc/tools` | tool calling (calculator, current time) |
| `POST /lc/rag` | retriever chain over Qdrant |
| `POST /lg/agent/run` · `/lg/agent/stream` · `GET /lg/agent/state/{thread}` | ReAct agent, SQLite-checkpointed per `thread_id`, event streaming |
| `POST /lg/approval/start` · `/lg/approval/resume` | human-in-the-loop interrupt / resume |
| `POST /lg/supervisor/run` | supervisor → worker multi-agent workflow |
| `POST /graph/rag` · `/graph/agent` | existing Reality Engine-bound LangGraph graphs |

Configuration: `DEFAULT_MODEL`, `LLM_MODEL`, `CHECKPOINT_DB_PATH`,
`LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` (see `.env.example`).

### Adding a model

1. Append an entry to `config/models.registry.json` (stable kebab-case `id`,
   exact Ollama `tag`).
2. `make model-pull ID=<id>`
3. Point `.env` at it only if it should become the selected llm/embedder.

Swapping the *embedder* also means matching `EMBED_DIM` and recreating the
`localai_docs` Qdrant collection — the registry records each embedder's
`embed_dim` so the mismatch is visible before you hit it.

## Provider completion conformance

Validate the RealityEngine PE completion callback contract without a live
OpenAI key:

```bash
make provider-conformance
node scripts/pe-completion-conformance.mjs --mode native --stub --dry-run
node scripts/pe-completion-conformance.mjs --mode openai-chat --stub --dry-run
```

For live Ollama probing, omit `--stub`. Native mode calls `/api/chat`; OpenAI
compatible mode calls `/v1/chat/completions`. Omit `--dry-run` to POST the
validated completion body to `$PE_URL/api/integrations/completions`.

## Logging (Loki + Grafana)

All containerised services ship logs to Loki via the Docker `loki` log driver
with labels `app=localaistack` and `service=<name>`.  Grafana is
auto-provisioned with the Loki datasource and the **localAIStack Overview**
dashboard (panels: per-service log rate, error rate, RAG/LangGraph API logs,
GraphQL trigger events from the Reality Engine bridge, Qdrant, Redis, WebUI).

### Example LogQL queries

```logql
# All localAIStack logs
{app="localaistack"}

# API logs only
{app="localaistack", service="api"}

# Errors across the stack
{app="localaistack"} |~ "(?i)error|(?i)exception|(?i)traceback"

# Upstream triggers pushed from Reality Engine machines
{app="localaistack", service="api"} |~ "(?i)graphql|updateProcessState|ragStatusCode"

# Log rate per service (1-minute windows)
sum by (service) (count_over_time({app="localaistack"}[1m]))
```

### Ollama logs

Ollama runs **natively on the host** (Metal acceleration) rather than in a
container, so its logs are not shipped into Loki.  Tail them directly:

```bash
tail -f /tmp/ollama.log
```

## Architecture notes

- **Qdrant is unified** with the Reality Engine stack: both `localai_docs`
  (document embeddings) and `reality-events` (perceptual vectors) live in this
  instance.  Any active runtime (CPP, Scala, LSP) connects from its Docker
  network via `host.docker.internal:4333`.
- **Model metadata** lives in `config/models.registry.json`, mounted into the
  api container at `/app/config` and served at `GET /models`. See
  [Model registry](#model-registry).
- **Embedding model** defaults to `ternary-bonsai:4` (registered at setup time
  from `hf.co/prism-ml/Bonsai-4B-gguf`).  Override via `.env` `EMBED_MODEL`.
  > ⚠ **Known issue:** the prism-ml repo ships only a Q1_0 (BitNet ternary)
  > GGUF, which Ollama's bundled GGML does not yet recognize — the model
  > registers and appears in the WebUI dropdown, but inference and embeddings
  > return HTTP 500 (`file_type=unknown`) until upstream Ollama ships GGML
  > with TQ1_0/BitNet support.  For a working default, set
  > `EMBED_MODEL=nomic-embed-text` + `EMBED_DIM=768` in `.env`.
- **LangChain / LangGraph** modules (`services/api/chains/`, `graphs/`,
  `core/llm_factory.py`) run inside the same api container; LangGraph state is
  checkpointed to SQLite at `CHECKPOINT_DB_PATH` (`./volumes/state`).
  See [docs/LANGCHAIN_LANGGRAPH.md](docs/LANGCHAIN_LANGGRAPH.md).
- **GraphQL trigger receiver** (`services/api/routers/graphql_endpoint.py`)
  exposes `POST /graphql` for machine → AI upstream pushes.  Events are retained
  in a 128-entry ring buffer; verify with `curl http://localhost:4000/graphql/events`.
