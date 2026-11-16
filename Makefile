.PHONY: mlx-start mlx-stop mlx-status use-mlx use-ollama pull-model langchain-demo langgraph-demo setup start stop up down logs health query ingest models models-installed ollama-check model-pull model-info provider-conformance db-setup governance evals loadtest ha-up ha-down clean

# ── Lifecycle ─────────────────────────────────────────────────────────────────
setup:
	@bash scripts/setup.sh

start:
	@bash scripts/start.sh

stop:
	@bash scripts/stop.sh

restart: stop start

up:
	@bash scripts/lib/ollama_guard.sh --ensure
	@ollama serve &>/tmp/ollama.log & sleep 1
	@docker compose up -d

down:
	@docker compose down

# ── MLX (exclusive alternative to Ollama; host-native, Apple Silicon) ─────────
# Usage: make mlx-start ID=RadixArk/Muse-Glimmer-q4-MLX
mlx-start:
	@bash scripts/start-mlx.sh $(ID)

mlx-stop:
	@bash scripts/stop-mlx.sh

mlx-status:
	@curl -sf http://localhost:8081/v1/models >/dev/null 2>&1 \
		&& echo "MLX running on :8081" || { echo "MLX not running"; exit 1; }

# Switch the exclusive backend in .env and restart (stop first, with the old setting).
use-mlx:
	@bash scripts/stop.sh
	@bash scripts/lib/env_set.sh USE_MLX true
	@bash scripts/start.sh

use-ollama:
	@bash scripts/stop.sh
	@bash scripts/lib/env_set.sh USE_MLX false
	@bash scripts/start.sh

# ── Observability ─────────────────────────────────────────────────────────────
logs:
	@docker compose logs -f

logs-api:
	@docker compose logs -f api

# /health covers ollama, qdrant, redis, the RE/PE bridge and — when
# CHECKPOINT_BACKEND=postgres — postgres. The governance line is the Merkle
# chain verdict (valid=false means the provenance log was tampered with).
health:
	@curl -s http://localhost:4000/health | python3 -m json.tool
	@curl -s "http://localhost:4000/governance/chain?limit=1" | python3 -c \
		"import sys,json; d=json.load(sys.stdin); print('governance chain: valid=%s length=%s lamport=%s' % (d['valid'], d['length'], d['lamport']))"

governance:
	@curl -s http://localhost:4000/governance/state | python3 -m json.tool

# ── Persistence (PostgreSQL checkpointer + store) ─────────────────────────────
# Needs CHECKPOINT_BACKEND=postgres and DATABASE_URL (see .env.example).
db-setup:
	@set -a; [ -f .env ] && . ./.env; set +a; python3 scripts/db_setup.py

# ── Evaluation policies / load / HA ───────────────────────────────────────────
# Policy-driven OEE evaluation of the golden dataset (fake LLM, no network).
evals:
	@python3 scripts/run_evals.py

# Usage: make loadtest [BASE_URL=http://localhost:4000 CONCURRENCY=20 DURATION=30]
loadtest:
	@python3 scripts/loadtest/load_test.py --base-url $(or $(BASE_URL),http://localhost:4000) \
		--concurrency $(or $(CONCURRENCY),20) --duration $(or $(DURATION),30)

# HA stack: Postgres-backed state, 3 API replicas behind nginx (docs/HA_DEPLOYMENT.md)
ha-up:
	@docker compose -f docker-compose.yml -f docker-compose.ha.yml up -d --build

ha-down:
	@docker compose -f docker-compose.yml -f docker-compose.ha.yml down

# ── Model registry ────────────────────────────────────────────────────────────
# Registry = config/models.registry.json (available), .env = selected,
# Ollama = installed. `make models` shows all three; the API serves the same
# join at GET /models.
models:
	@bash scripts/lib/models_registry.sh --list

# Tags actually pulled on the Ollama host, registry or not.
models-installed:
	@curl -s http://localhost:11434/api/tags | python3 -c \
		"import sys,json; [print(' ', m['name']) for m in json.load(sys.stdin).get('models',[])]"

ollama-check:
	@bash scripts/lib/ollama_guard.sh --check

# Usage: make model-pull ID=nemotron-3-nano-4b
model-pull:
	@bash scripts/lib/models_registry.sh --pull $(ID)

# Pull the default model (DEFAULT_MODEL in .env, default llama3.1-8b → llama3.1:8b)
pull-model:
	@set -a; [ -f .env ] && . ./.env; set +a; \
		bash scripts/lib/models_registry.sh --pull "$${DEFAULT_MODEL:-llama3.1-8b}"

# Usage: make model-info ID=nemotron-3-nano-4b   (requires the API to be up)
model-info:
	@curl -s http://localhost:4000/models/$(ID) | python3 -m json.tool

provider-conformance:
	@node --test scripts/pe-completion-conformance.test.mjs

# ── RAG operations ────────────────────────────────────────────────────────────
# Usage: make ingest FILE=./data/<domain>/documents/spec.pdf
ingest:
	@python3 scripts/ingest.py $(FILE)

# Usage: make query Q="What is the reality engine?"
query:
	@curl -s -X POST http://localhost:4000/graph/rag \
		-H "Content-Type: application/json" \
		-d '{"question": "$(Q)"}' | python3 -m json.tool

# Usage: make agent Q="Search the knowledge base for X"
agent:
	@curl -s -X POST http://localhost:4000/graph/agent \
		-H "Content-Type: application/json" \
		-d '{"messages": [{"role": "user", "content": "$(Q)"}]}' | python3 -m json.tool

# ── LangChain / LangGraph demos (API must be up; default model llama3.1-8b) ──
# Usage: make langchain-demo [TEXT="..."]
langchain-demo:
	@curl -s -X POST http://localhost:4000/lc/summarize \
		-H "Content-Type: application/json" \
		-d '{"text": "$(or $(TEXT),LangChain composes prompts and models and parsers into chains with LCEL.)"}' | python3 -m json.tool
	@curl -s -X POST http://localhost:4000/lc/tools \
		-H "Content-Type: application/json" \
		-d '{"prompt": "What is (12 + 3) * 4?"}' | python3 -m json.tool

# Usage: make langgraph-demo [Q="..."]
langgraph-demo:
	@curl -s -X POST http://localhost:4000/lg/agent/run \
		-H "Content-Type: application/json" \
		-d '{"thread_id": "demo", "message": "$(or $(Q),What is 17 * 23?)"}' | python3 -m json.tool

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@docker compose down -v
	@rm -rf volumes/qdrant/* volumes/redis/* volumes/open-webui/*
	@echo "Volumes cleared. Run 'make setup' to reinitialize."
