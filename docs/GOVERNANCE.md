# OEE completeness governance layer

`services/api/governance/` applies the argument of the manuscript *RealityEngine
V0.01* — one-dimensional external guardrails are OEE-incomplete; verification must
run **inline with execution and be recorded, as one step** — to the LangGraph agent
API. It is a minimal, testable Python module. **It does not make the system
"OEE-complete"** and nothing here should be described that way: see
[Guarantees and limits](#guarantees-and-limits).

## Architecture

```
POST /lg/agent/run ──► govern_tools(...) wraps each tool
                              │  every tool call:
                              ▼
        ┌───────────────── GovernanceEngine.admit ─────────────────┐
        │ 1 Lamport tick / receive          (clock.py)             │
        │ 2 capability check                (capability.py)        │
        │ 3 encode whole thread → [0,1]^N   (features.py)          │
        │ 4 aDFA single transition          (adfa.py)   hazard ⇒   │
        │      output clamped to 0, action blocked                 │
        │ 5 execute  — only if 2 and 4 allowed                     │
        │ 6 record   — Merkle chain + K-line (merkle.py, klines.py)│
        └───────────────────────────────────────────────────────────┘
   active RuleSet read once per event ◄── atomic swap (swap.py) ◄── replay gate (klines.py)
```

| Module | Role |
|---|---|
| `adfa.py` | `Rule` = interval (axis-aligned polytope) bounds per Critical Event Sequence; `RuleSet.step` is the single feed-forward transition. No recursion, no loopback, nothing carried between inputs. |
| `features.py` | `ContextEncoder`: whole-thread text (+ proposed action) → `[0,1]^N`. Text is squashed to `[a-z0-9]` and joined across turns, so a phrase split over messages is still seen. |
| `clock.py` | `LamportClock`: `tick()` on local events, `receive(t)` sets `L = max(L, t) + 1`. |
| `merkle.py` | `ProvenanceChain`: `h_L = SHA256(h_{L-1} ‖ S0 ‖ frame ‖ capability_token_id)`; each field length-prefixed; `verify()`. |
| `klines.py` | `(S0, frames)` store and `replay`, a deterministic counterfactual run reporting whether every hazard trace is clamped and how many nominal traces are false-positive clamps. |
| `swap.py` | `RuleManager`: candidate rule sets are replay-gated, staged by one reference assignment, adopted on the next event. No lock. |
| `predictive.py` | Optional Markov estimate `P(K_{t+k}=hazard ∣ K_t)` (`T^k`); above a threshold, stages a tightened rule set (still replay-gated). |
| `capability.py` | HMAC-signed capability tokens: allowed tools + scopes; a delegated child must be a subset of its parent. |
| `engine.py` | `GovernanceEngine.admit`, the Atomic Triad, and a read-only `snapshot()`. |
| `tools.py` | Wraps LangChain tools so every call goes through `admit`; a refused call returns a `BLOCKED …` message to the model. |

Defaults (features, lexicon, tool risk, rules) are in `governance/data/default_policy.json`;
the seed K-lines (hazard and nominal traces the replay gate checks every candidate
against) in `governance/data/klines.json`. Override with `GOVERNANCE_POLICY_PATH` /
`GOVERNANCE_KLINES_PATH`.

### 1. Atomic Triad admission

`admit` is the only path to execute a governed tool. It verifies first, executes only
if verification passed, and records **always** — a hazard block, a capability denial and
a tool that raises are all recorded (a raising tool is recorded, then re-raised).
Verification runs over the *full thread*: prior checkpointed messages + the new prompt
+ the proposed call's arguments, not just the latest prompt.

### 2. aDFA guard

Input is `[0,1]^N` (`ValueError` otherwise). A rule matches iff every bounded component is
inside its inclusive `[lo, hi]`. First matching active rule wins; the output is the zero
vector and the action is blocked. Rules may be scoped to `modes`, matched against
`S0["mode"]`. Cost is one pass over the rules' bounds — independent of history length.

### 3. Lamport clock

Per engine instance. Incremented on each local event (an admission, a dispatch), merged on
receipt (`"lamport": T` in the request body of `/lg/agent/*` and `/governance/dispatch`).
The timestamp is in every chain record's frame, in the `202` dispatch response, in the
`X-Lamport` response header of `/lg/agent/run`, and in the first SSE event of
`/lg/agent/stream`.

### 4. Merkle provenance chain

Append-only. `S0` — the initial active-state configuration for the thread
(`{"mode", "thread_id"}`) — and the whole frame (lamport, tool, args, input vector, clamped
output, allowed/reason, executed, result SHA-256, rule-set version) are inside each hash,
so editing, deleting or reordering any record breaks verification from that point.
`GOVERNANCE_CHAIN_PATH` makes it a JSONL file that is reloaded at start.

### 5. K-line replay

`klines.replay(rules, klines)` mirrors the manuscript's Appendix D Lisp pseudocode
(transcribed in the module docstring — **the manuscript is not in this repository, so
fidelity to Appendix D is by description, not by comparison**). Every admitted action adds
a K-line: the thread's frames since its last hazard, labelled `hazard` if a rule clamped
it, else `nominal`. A hazard K-line is *clamped* if any frame is blocked; a nominal one is a
*false positive* if any frame is blocked.

### 6. Atomic rule swap

`RuleManager.stage` replays the candidate; only a set that clamps **all** hazard K-lines
and has ≤ `GOVERNANCE_MAX_FALSE_POSITIVES` false positives is eligible
(`RuleSetRejected` otherwise). Staging is one reference assignment; each event calls
`active()` once, adopts the staged set by another reference assignment, and evaluates with
that immutable object — no global lock, no dropped request, no event sees a mixed set.

### 7. Predictive estimate (`GOVERNANCE_PREDICTIVE_ENABLED=true`, off by default)

Observed class transitions (`nominal` / `elevated` — would match rules widened by
`GOVERNANCE_PREDICTIVE_MARGIN` — / `hazard`) feed a Laplace-smoothed Markov matrix. When
`P(hazard within k)` exceeds `GOVERNANCE_PREDICTIVE_THRESHOLD`, the active rules are widened
and staged through the same replay gate (once per base version). Empirical frequencies, not
a proof; with little data the smoothing alone can cross a low threshold.

### 8. Identity and capability scoping

`X-Capability-Token` carries `base64url(JSON{id, sub, tools, scopes, parent, exp}).HMAC-SHA256`.
`authorize(token, tool, scope)` checks both. `delegate` mints a child whose tools and scopes
are subsets of the parent's (a `*` parent allows anything; a child may not widen to `*`) and
whose expiry cannot outlive the parent's. Without a header the engine uses a default token
(`GOVERNANCE_DEFAULT_TOOLS`, `GOVERNANCE_DEFAULT_SCOPES`) unless
`GOVERNANCE_REQUIRE_TOKEN=true` (then `401`). Scopes in use: `agent:run`, `handoff:dispatch`,
`governance:admin`.

**Future work — SPIFFE.** The shared HMAC secret is the stand-in for workload identity.
`CapabilityAuthority.verify/delegate/authorize` is the seam: a SPIFFE implementation would
validate an X.509/JWT SVID, map its SPIFFE ID to the same `Capability`, and keep the
subset rule. Not implemented.

Mint tokens with the secret you configured, e.g.:

```bash
GOVERNANCE_TOKEN_SECRET=… python3 -c "
import sys; sys.path.insert(0, 'services/api')
from governance.capability import CapabilityAuthority
a = CapabilityAuthority('$GOVERNANCE_TOKEN_SECRET')
print(a.mint('ops', ['*'], ['governance:admin']))"
```

### 9. ACP / MCP-style handoffs

`POST /governance/dispatch` returns **202 Accepted** at once with `event_id` and `lamport`;
verification and recording (`execute=None`: fire-and-record — the receiving agent acts) run
as a background task. Read-only inspection is separate: `GET /governance/state` and
`GET /governance/chain` never tick the clock, append, or change rules (tests assert state is
identical before/after, and that POST/PUT/DELETE/PATCH on them are `405`).

## HTTP API

| Method / path | Mutates? | Notes |
|---|---|---|
| `GET /governance/state` | no | clock, rule version, pending version, chain head/validity, K-line counts, predictive state |
| `GET /governance/chain?limit=50` | no | newest `limit` records + verification of the **whole** chain (`valid`, `first_bad_index`) |
| `POST /governance/replay` | no | counterfactual replay of `{"rules": {...}}`; reports `eligible` |
| `POST /governance/dispatch` | records | `202`; body `{thread_id, target, payload, context, lamport}` |
| `POST /governance/rules` | rules | needs scope `governance:admin` (`401`/`403`); replay-gated (`409`) |
| `POST /lg/agent/run` / `/stream` | records | every tool call governed; `X-Capability-Token`, optional `lamport` |

Rule sets use named bounds: `{"version": "v2", "rules": [{"name": "r", "bounds": {"destructive": [0.5, 1], "tool_risk": [0.5, 1]}}]}`.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GOVERNANCE_ENABLED` | `true` | wrap `/lg/agent/*` tools |
| `GOVERNANCE_TOKEN_SECRET` | random per process | HMAC secret. **Set it** to keep tokens across restarts / replicas; admin tokens are unusable without it |
| `GOVERNANCE_REQUIRE_TOKEN` | `false` | require `X-Capability-Token` on agent calls |
| `GOVERNANCE_DEFAULT_TOOLS` / `_SCOPES` | `*` / `agent:run,handoff:dispatch` | the default token |
| `GOVERNANCE_CHAIN_PATH` | empty (memory) | JSONL file for the chain |
| `GOVERNANCE_POLICY_PATH` / `GOVERNANCE_KLINES_PATH` | packaged defaults | feature/rule policy and seed K-lines |
| `GOVERNANCE_KLINE_MAX` | `1000` | observed K-lines kept (seeds always kept) |
| `GOVERNANCE_MAX_FALSE_POSITIVES` | `0` | replay-gate budget |
| `GOVERNANCE_PREDICTIVE_ENABLED` / `_THRESHOLD` / `_HORIZON` / `_MARGIN` | `false` / `0.5` / `3` / `0.1` | optional predictive estimate |

## Guarantees and limits

**What is actually provided** (each backed by a test in `tests/test_governance*.py`):

- A governed tool call is never executed without having been verified, and is never
  verified without a record being appended — within one `admit` call, in one process.
- Verification sees the whole checkpointed thread plus the new prompt, so a hazard phrase
  split across turns or padded with punctuation is seen by the lexicon.
- The aDFA is a single, stateless, feed-forward transition; cost does not grow with history.
- The provenance chain detects edit, deletion and reordering of records (verified from the
  genesis hash on every read).
- A rule set that fails the replay check is never adopted, and adoption never blocks or
  drops an in-flight request.
- A child capability cannot exceed its parent, and expired/forged tokens are refused.
- The read-only endpoints cannot mutate state.

**What is not provided:**

- **No formal OEE completeness.** This is a runtime guard with tests, not a proof of
  completeness or of the manuscript's theorem. A hazard not expressible in the lexicon
  (paraphrase, another language, an encoded payload) produces a nominal vector and is not
  blocked. Rules and the K-line corpus are only as good as their authors.
- **Single node.** Clock, chain, K-lines and rule state are per process. With several API
  replicas each has its own chain and Lamport clock (the HA overlay documents this);
  nothing is replicated or totally ordered across nodes.
- **Python runtime, not real time.** Latency is that of CPython under the GIL and of the
  HTTP stack — nowhere near sub-microsecond. "Atomic swap" relies on CPython reference
  assignment being atomic.
- **Tamper-evident, not tamper-proof.** Someone who can rewrite the log *and* its head can
  forge a consistent chain. Anchor `head` externally if that matters. The in-memory default
  is lost on restart.
- **The record cannot undo side effects.** If a tool performs an external effect and the
  process dies before the record is appended, the effect is unrecorded. The Triad is atomic
  with respect to the code path, not a distributed transaction.
- **Dispatch is not delivery.** A `202` means accepted for verification and recording;
  delivery to the receiver is out of scope. Dispatch records may land in the chain in a
  different order from their Lamport timestamps.
- **Only tool calls on `/lg/agent/*` are governed.** The other routers, model calls and the
  graph's own transitions are not.
- Capability tokens use a shared secret; there is no revocation list (use short expiries).
- `GET /governance/state` and `/chain` verify the entire chain each call, which is O(length).

## Running the tests

```bash
pytest services/api/tests/test_governance.py services/api/tests/test_governance_api.py
make health        # /health + the Merkle chain verdict (needs the stack up)
make governance    # GET /governance/state
```
