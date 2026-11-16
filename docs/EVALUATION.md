# Evaluation policies (OEE testing)

`services/api/evals/` is a deterministic, policy-driven harness. A *case* scripts the model's
replies (so no LLM is needed) and runs through the real ReAct graph, a checkpointer and the
governance layer. A *policy* says which invariants are enforced and what gates a build.

```bash
make evals                       # python3 scripts/run_evals.py; exit 1 if the gate fails
python3 scripts/run_evals.py --policy oee --cases data/evals/golden.json
pytest services/api/tests/test_evals.py
```

| File | Role |
|---|---|
| `data/evals/policies/oee.json` | the policy: `invariants`, `critical`, `min_pass_rate` (0.9), `max_steps` (25), `forbidden_tools` |
| `data/evals/golden.json` | golden cases: `prompt`, optional `history` turns, scripted `script`, `expected_tools`, `answer_contains`, `expect_blocked` |
| `evals/policy.py` / `evals/harness.py` | `EvalPolicy`, `load_cases`, `scripted_runner`, `check`, `evaluate` |

Invariants: `terminates` (steps ≤ `max_steps`), `monotonic_state` (each checkpoint extends the
previous), `admissibility` (a refused call is never executed, tool calls and governance records
agree, the chain verifies, blocking matches `expect_blocked`), `expected_tools`,
`forbidden_tools` (executed, not merely attempted), `answer_contains`. The gate passes when the
pass rate ≥ `min_pass_rate` **and** no `critical` invariant fails on any case. A run that raises
is a failed case.

The Regression workflow runs `tests/test_evals.py`, so a change that breaks the golden set fails CI.
This is rule-based: there is no LLM-as-judge, and `EVALS_DIR` can point at another dataset.
The harness imports the scripted fake model from `tests/fakes.py`.
