# Deployment plan status audit

Source: `docs/lang_framework_deployment_plan.xlsx` (commit `cd10dd6`). The workbook
marks **every task "Completed"** (its KPI sheet says "32 Tasks", but the roadmap
sheet lists only **30** — T1.1 … T8.4) and its integrity sheet reports "PASSED" for
persistence, observability and OEE safety. That was not true of the repository: the
`.xlsx` status column is a plan artefact, not evidence. This file is the audited
status. The workbook binary is intentionally **not edited** (openpyxl would drop its
cached formula values and styling); treat this document as authoritative.

Status vocabulary:

| Status | Meaning |
|---|---|
| **Done** | Implemented in this repo and covered by a test or a run recorded below |
| **Partial** | Part of the task is implemented; the gap is stated |
| **Blocked** | Needs something that cannot come from the repo (external account, key, live infra) |
| **Not started** | Nothing implemented |

"Done" never means "verified against a hosted service we have no credentials for".
Tests named below run with `pytest services/api/tests --ignore=services/api/tests/e2e`.

## Summary (workbook said 30/30 Completed)

| Status | Count |
|---|---|
| Done | 8 |
| Partial | 17 |
| Blocked | 2 |
| Not started | 3 |

## Task audit

| ID | Task | Workbook | True status | Evidence / remaining gap |
|---|---|---|---|---|
| T1.1 | Provision Postgres & Redis checkpointer storage | Completed | **Partial** | `postgres` (pgvector/pg16) service in `docker-compose.yml`; `graphs/checkpoint.py` `PostgresSaver`; `scripts/db_setup.py`; `/health` probe. Exercised against a local PostgreSQL 16 (`tests/test_persistence.py::test_live_postgres_state_survives_a_restart`, opt-in via `TEST_DATABASE_URL`). Redis is provisioned but used for cache/health only — **no `RedisSaver` is wired** (`langgraph-checkpoint-redis` is installed, unused). pgvector extension is provisioned but not used by any code. |
| T1.2 | Provision cross-thread long-term store | Completed | **Done** | `graphs/store.py` (`PostgresStore`, in-memory fallback), `PUT/GET /lg/memory/{user}`; live-Postgres round trip across a simulated restart; `test_memory_endpoints_share_a_store_across_threads`. |
| T1.3 | Multi-provider LLM gateway | Completed | **Partial** | `core/llm_factory.py` selects `LLM_PROVIDER` via `init_chat_model` (tests with a stub). Provider packages are optional (`requirements-providers.txt`). Never called against OpenAI / Anthropic / Vertex — no keys. |
| T1.4 | Provision observability accounts (LangSmith, Langfuse) | Completed | **Blocked** | Accounts/keys cannot be created from the repo. Config surface exists (`LANGSMITH_*`, `OTEL_EXPORTER_OTLP_*`); nothing is provisioned. Langfuse is reachable through its OTLP endpoint, not through a Langfuse SDK. |
| T2.1 | OpenTelemetry & OpenInference tracing | Completed | **Done** | `core/tracing.py` (`TRACING_BACKEND=otel`, OpenInference LangChain instrumentation); `tests/test_providers_tracing.py::test_otel_emits_spans_for_langchain_runs` (skips when the optional packages are absent). Spans are emitted for LangChain runs, which LangGraph steps ride on; no collector was available to confirm end-to-end export. |
| T2.2 | Custom callback handler library | Completed | **Partial** | `core/llm_factory.LoggingCallbackHandler` logs LLM/tool start/end. **No token-count aggregation or latency tracking.** |
| T2.3 | Unified state schema library (`omega_tau`) | Completed | **Done** | `graphs/state.py` provides `BaseGraphState`, `StateFactory`, and shared reducers; the ReAct, approval, and supervisor graphs use the shared contract. `tests/test_state_reducers.py` checks reducer laws with generated values. |
| T3.1 | ReAct agent core node | Completed | **Done** | `graphs/react_agent.py` (custom `StateGraph`, not `create_agent`); `tests/test_langgraph_features.py`. |
| T3.2 | Input/output guardrails middleware | Completed | **Partial** | `governance/tools.py` verifies every **tool call** over the whole thread before it runs (`tests/test_governance*.py`). Not `AgentMiddleware`; no model-call (`wrap_model_call`) screening or output-schema validation. |
| T3.3 | Verify Workflow 1 callback & tracing integration | Completed | **Blocked** | Needs a real LangSmith/Langfuse project. In-memory OTel span capture is tested; the hosted check is not possible here. |
| T4.1 | StateGraph with checkpointer binding | Completed | **Done** | SQLite default, PostgreSQL via `CHECKPOINT_BACKEND=postgres`; per-`thread_id` persistence tested for both. |
| T4.2 | Native suspension gate (`interrupt`) | Completed | **Done** | `graphs/approval_graph.py`; `test_approval_http_flow`. Generic plan approval, not the workbook's financial-transfer payload. |
| T4.3 | Resumption & edit API endpoint | Completed | **Partial** | `POST /lg/approval/resume` (`Command(resume=...)`) exists. **No `update_state()` edit endpoint.** |
| T4.4 | Time-travel & historical state replay | Completed | **Partial** | The bounded history endpoint is PR #4 and is **not on this branch** (untouched here to avoid conflicts). `get_state_history` is used by `evals/harness.py` for the monotonic-state check only. Rollback / fork-from-checkpoint is not implemented. |
| T5.1 | Specialised subgraphs (Research, Code, QA) | Completed | **Partial** | `graphs/supervisor_graph.py` has `math` and `writer` workers; no Research/Code/QA subgraphs, no private state schemas. |
| T5.2 | Central supervisor router | Completed | **Done** | `graphs/supervisor_graph.py` (`parse_route`, bounded `max_steps`); not the `langgraph-supervisor` package. |
| T5.3 | Cross-thread long-term memory | Completed | **Partial** | Store + `/lg/memory` API (see T1.2). **No memory-extraction node and no injection of stored preferences into prompts.** |
| T5.4 | Verify subgraph isolation & compression | Completed | **Not started** | No compression of subgraph scratchpads; nothing to verify. |
| T6.1 | Bounded join-semilattice state reducers | Completed | **Partial** | `graphs/state.py` adds tested set-union, max, and ordered-latest reducers. `add_messages` preserves conversational order and is intentionally not commutative; the quality-evaluation router remains missing (T6.2). |
| T6.2 | Conditional router edges for quality evaluation | Completed | **Partial** | Conditional edges exist (`route_after_agent`, supervisor). **No code-syntax / validation-criteria evaluation router.** |
| T6.3 | Deterministic recursion limits | Completed | **Done** | `GRAPH_RECURSION_LIMIT=25` applied in `routers/langgraph_api._config`; supervisor `max_steps`; `terminates` eval invariant (`tests/test_evals.py`). |
| T6.4 | Verify mathematical OEE runtime invariants | Completed | **Partial** | `governance/` + `evals/` check admissibility, termination and monotonic state **empirically** (tests, golden cases). Verification happens at the tool boundary, not before node scheduling. See `docs/GOVERNANCE.md` for the exact guarantees; no formal OEE-completeness claim is made. |
| T7.1 | Golden evaluation datasets | Completed | **Partial** | `data/evals/golden.json` (5 cases, local). Not curated in LangSmith/Langfuse. |
| T7.2 | LLM-as-a-judge evaluators | Completed | **Not started** | `evals/` is deterministic and policy-driven. No Faithfulness / Relevance / Context-Precision judge. |
| T7.3 | Production tracing dashboard & alert rules | Completed | **Partial** | `config/dashboards/localaistack-overview.json` (Loki log panels). **No latency/token-cost/error-rate metrics, no Slack/PagerDuty alerts.** |
| T7.4 | Continuous curation queue | Completed | **Not started** | Governance records blocked actions (`GET /governance/chain`) but there is no annotation/curation queue. |
| T8.1 | BSP stress test, 500 concurrent threads | Completed | **Partial** | `scripts/loadtest/load_test.py` (`--scenario threads`, thread-isolation check; `make loadtest`); unit-tested against a mock transport and run locally (20 workers, governance scenario: ~470 rps, 0 errors). **The 500-thread run against a live LLM has not been executed.** |
| T8.2 | CI/CD evaluation gate | Completed | **Partial** | The golden set runs through the `oee` policy in the Regression workflow (`tests/test_evals.py`; 0.9 pass rate + critical invariants). No LangSmith eval-run check. |
| T8.3 | Production API gateway | Completed | **Partial** | `docker-compose.ha.yml` + `config/nginx-ha.conf`: balancing, retry, SSE-safe. **No TLS termination or rate limiting**; CORS is `*`. |
| T8.4 | Disaster recovery / failover rehearsal | Completed | **Partial** | State surviving a process restart is tested against real Postgres. The mid-superstep pod-crash rehearsal has **not** been run; procedure in `docs/HA_DEPLOYMENT.md`. |

## Workbook integrity-sheet claims

| Claim | Reality |
|---|---|
| "PostgresSaver and PgStore endpoints active with zero loss" | Wired now (this change); not previously present. Not load-tested for loss. |
| "All 5 operational workflows generating OpenTelemetry spans" | Spans are available behind `TRACING_BACKEND=otel`; confirmed only with an in-memory exporter. Workflows WF-2/3/4/5 do not exist as specified. |
| "Verified omega_tau schema and tau conditional edge router present" | The shared schema and tested reducers now exist (T2.3); the conditional quality-evaluation router remains missing (T6.2). |
| "OEE Runtime Safety Compliant: YES" | Not supportable. See `docs/GOVERNANCE.md` for what is and is not guaranteed. |

## What this change implemented from the PR #4 follow-ups

1. PostgreSQL / PgStore persistence — T1.1, T1.2, T4.1
2. Configurable tracing and model providers — T1.3, T2.1
3. Policy-driven evaluation harness — T6.3, T6.4, T7.1, T8.2
4. Load test scripts and HA compose — T8.1, T8.3, T8.4
5. This audit
6. The OEE governance layer — T3.2, T6.4 (`docs/GOVERNANCE.md`)
