"""Evaluation-policy harness: the golden dataset is the OEE regression gate."""

import dataclasses

import pytest

from evals.harness import EvalCase, RunTrace, evaluate, load_cases, scripted_runner
from evals.policy import EvalPolicy, load_policy


def test_golden_dataset_passes_the_oee_policy():
    report = evaluate(load_policy("oee"), load_cases())
    assert report.passed, report.to_dict()
    assert report.pass_rate == 1.0 and not report.critical_violations


def test_policy_validation():
    with pytest.raises(ValueError, match="unknown invariants"):
        EvalPolicy("p", ("nope",))
    with pytest.raises(ValueError, match="critical"):
        EvalPolicy("p", ("terminates",), critical=("monotonic_state",))
    with pytest.raises(ValueError, match="min_pass_rate"):
        EvalPolicy("p", ("terminates",), min_pass_rate=1.5)


def _trace(**kw):
    base = {
        "answer": "ok",
        "steps": 1,
        "tools_called": [],
        "executed_tools": [],
        "snapshots": [["a"], ["a", "b"]],
        "records": [],
        "chain_valid": True,
    }
    return RunTrace(**{**base, **kw})


def _case(**kw):
    return EvalCase("c", "p", ("x",), **kw)


def _run(policy, case, trace):
    return evaluate(policy, [case], lambda _c: trace)


POLICY = EvalPolicy(
    "t",
    ("terminates", "monotonic_state", "admissibility", "forbidden_tools", "answer_contains"),
    critical=("monotonic_state", "admissibility"),
    max_steps=3,
    forbidden_tools=("shell",),
)


def test_each_invariant_is_enforced():
    assert _run(POLICY, _case(), _trace()).passed
    assert "terminates" in _run(POLICY, _case(), _trace(steps=9)).results[0].failures
    assert (
        "monotonic_state"
        in _run(POLICY, _case(), _trace(snapshots=[["a", "b"], ["b"]])).results[0].failures
    )
    assert (
        "forbidden_tools"
        in _run(POLICY, _case(), _trace(executed_tools=["shell"])).results[0].failures
    )
    assert (
        "answer_contains"
        in _run(POLICY, _case(answer_contains=("zzz",)), _trace()).results[0].failures
    )
    bad_chain = _run(POLICY, _case(), _trace(chain_valid=False))
    assert "admissibility" in bad_chain.results[0].failures
    refused_but_run = _trace(
        tools_called=["shell"],
        records=[{"executed": True, "allowed": False}],
        executed_tools=["shell"],
    )
    assert "admissibility" in _run(POLICY, _case(), refused_but_run).results[0].failures


def test_critical_violation_fails_gate_even_above_pass_rate():
    policy = dataclasses.replace(POLICY, min_pass_rate=0.5)
    cases = [_case(), dataclasses.replace(_case(), id="d")]
    traces = iter([_trace(), _trace(chain_valid=False)])
    report = evaluate(policy, cases, lambda _c: next(traces))
    assert (
        report.pass_rate == 0.5
        and report.critical_violations == ["d:admissibility"]
        and not report.passed
    )


def test_pass_rate_threshold_gates_non_critical_failures():
    policy = dataclasses.replace(POLICY, min_pass_rate=0.9)
    cases = [_case(answer_contains=("zzz",)), dataclasses.replace(_case(), id="d")]
    report = evaluate(policy, cases, lambda c: _trace())
    assert report.pass_rate == 0.5 and not report.passed


def test_crashing_run_is_a_failed_case():
    def boom(_c):
        raise RuntimeError("x")

    report = evaluate(POLICY, [_case()], boom)
    assert not report.passed and "RuntimeError" in report.results[0].failures["terminates"]


def test_scripted_runner_blocks_hazard_and_records_it():
    case = next(c for c in load_cases() if c.expect_blocked)
    trace = scripted_runner(case)
    assert trace.tools_called == ["shell"] and trace.executed_tools == []
    assert trace.records[0]["allowed"] is False and trace.chain_valid
