# Load testing and high availability

## Load test

`scripts/loadtest/load_test.py` (async `httpx`, no extra dependencies):

```bash
make loadtest                                              # governance scenario, 20 workers, 30 s
make loadtest BASE_URL=http://localhost:4000 CONCURRENCY=100 DURATION=60
python3 scripts/loadtest/load_test.py --scenario threads   # 500 workers (workbook T8.1); needs a live LLM
```

| Scenario | Exercises | Needs LLM |
|---|---|---|
| `health` | `GET /health` | no |
| `governance` | `GET /governance/state` + `POST /governance/dispatch` | no |
| `agent` / `threads` | `POST /lg/agent/run` ×2 on a private thread, then `GET /lg/agent/state/{thread}` and check the thread holds exactly its own two turns (cross-thread isolation) | yes |

It prints requests, error rate, rps and p50/p95/p99, and exits non-zero if the error rate
exceeds `--max-error-rate` (0.01), p95 exceeds `--max-p95-ms` (2000) or any isolation
violation is seen — usable as a deploy gate. The unit tests run it against a mock transport
(`tests/test_loadtest_script.py`); a real run needs the stack up. Results depend entirely on
the host and model: no capacity numbers are claimed here.

## HA overlay

```bash
export GOVERNANCE_TOKEN_SECRET=$(openssl rand -hex 32)   # required by the overlay
make ha-up        # docker compose -f docker-compose.yml -f docker-compose.ha.yml up -d --build
make ha-down
```

Requires Docker Compose ≥ 2.24.

| Piece | What it does |
|---|---|
| `postgres` (base file) | durable checkpoints + store |
| `db-migrate` | one-shot job: `setup_persistence()`; replicas start only after it succeeds (`POSTGRES_AUTO_SETUP=false`) |
| `api` ×`API_REPLICAS` (3) | stateless replicas, `CHECKPOINT_BACKEND=postgres`, no source mount/`--reload`, a restart policy |
| `api-lb` (nginx) | publishes host `:4000`; re-resolves Docker DNS every 5 s, retries the next replica on error/timeout/5xx, no SSE buffering |

Any replica can resume any `thread_id`, so killing one loses nothing that was already
checkpointed. **Not covered:** Postgres itself is a single instance (use a managed/replicated
Postgres for HA of the database), nginx is a single instance, there is no TLS termination or
rate limiting, and Ollama runs on the host. Each replica has its **own** governance chain and
Lamport clock (a shared chain file would interleave writers and break the hash links), so
`/governance/chain` shows only the replica that served the request.

### Failover rehearsal (workbook T8.4 — not yet executed)

1. `make ha-up`; send `POST /lg/approval/start` with a `thread_id` so it pauses at the interrupt.
2. `docker kill` the replica that served it (`docker compose … ps`); repeat the request count
   until it lands elsewhere.
3. `POST /lg/approval/resume` for the same `thread_id` — expect `status: done` from another replica.
4. Repeat with `docker kill` **during** a `/lg/agent/run` to exercise a mid-superstep crash;
   the client should retry and the thread must contain no duplicated or missing turns.
5. `docker compose restart postgres` and confirm `GET /lg/agent/state/<thread>` is unchanged.

Step 5's durability is covered by `tests/test_persistence.py::test_live_postgres_state_survives_a_restart`
(`TEST_DATABASE_URL=postgresql://… pytest services/api/tests/test_persistence.py`); steps 1–4
have not been run.
