"""
Policy-driven evaluation harness.

A case scripts the model's replies, so a run is deterministic and needs no live
LLM; the agent, checkpointer and governance layer under test are the real ones.
``evaluate`` applies the invariants a policy enforces and gates on its pass rate.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from chains.tools import base_tools
from evals.policy import EvalPolicy, data_dir
from governance.capability import CapabilityAuthority
from governance.engine import GovernanceEngine
from governance.tools import govern_tools
from graphs.react_agent import build_react_agent


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    # Scripted model replies: a string, or {"tool": name, "args": {...}} for a tool call.
    script: tuple
    expected_tools: tuple[str, ...] = ()
    answer_contains: tuple[str, ...] = ()
    # Earlier turns of the thread (hazards can be spread across them).
    history: tuple[str, ...] = ()
    expect_blocked: bool = False

    @classmethod
    def from_dict(cls, d: Mapping) -> EvalCase:
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            script=tuple(d["script"]),
            expected_tools=tuple(d.get("expected_tools", ())),
            answer_contains=tuple(d.get("answer_contains", ())),
            history=tuple(d.get("history", ())),
            expect_blocked=bool(d.get("expect_blocked", False)),
        )


@dataclass
class RunTrace:
    answer: str
    steps: int
    tools_called: list[str]
    executed_tools: list[str]
    snapshots: list[list[str]]  # message ids per checkpoint, oldest first
    records: list[dict]  # governance chain frames for this run
    chain_valid: bool


@dataclass
class CaseResult:
    id: str
    passed: bool
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalReport:
    policy: str
    results: list[CaseResult]
    min_pass_rate: float
    critical_violations: list[str]

    @property
    def pass_rate(self) -> float:
        return sum(r.passed for r in self.results) / len(self.results) if self.results else 0.0

    @property
    def passed(self) -> bool:
        return self.pass_rate >= self.min_pass_rate and not self.critical_violations

    def to_dict(self) -> dict:
        return {
            "policy": self.policy,
            "passed": self.passed,
            "pass_rate": round(self.pass_rate, 4),
            "min_pass_rate": self.min_pass_rate,
            "critical_violations": self.critical_violations,
            "results": [asdict(r) for r in self.results],
        }


def load_cases(path: pathlib.Path | None = None) -> list[EvalCase]:
    path = path or data_dir() / "golden.json"
    return [EvalCase.from_dict(c) for c in json.loads(path.read_text())["cases"]]


@tool
def shell(command: str) -> str:
    """Run a shell command (eval stub: pretends to run it)."""
    return f"ran: {command}"


@tool
def send_email(to: str, body: str) -> str:
    """Send an email (eval stub: pretends to send it)."""
    return f"sent to {to}"


def scripted_runner(case: EvalCase, recursion_limit: int = 25) -> RunTrace:
    """Run one case through the real ReAct graph, governance layer and a checkpointer."""
    from tests.fakes import fake_llm, tool_call

    replies = [
        tool_call(s["tool"], s.get("args", {}), f"call-{i}") if isinstance(s, dict) else s
        for i, s in enumerate(case.script)
    ]
    engine = GovernanceEngine(authority=CapabilityAuthority("eval-secret"))
    saver = InMemorySaver()
    thread = f"eval-{case.id}"
    tools = govern_tools(
        [*base_tools(), shell, send_email],
        engine,
        thread_id=thread,
        context=[*case.history, case.prompt],
    )
    graph = build_react_agent(fake_llm(*replies), tools, saver)
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": recursion_limit}
    out = graph.invoke({"messages": [HumanMessage(content=case.prompt)]}, cfg)
    messages = out["messages"]
    snapshots = [
        [m.id for m in s.values.get("messages", [])]
        for s in reversed(list(graph.get_state_history(cfg)))
    ]
    records = [r["frame"] for r in engine.chain.records()]
    return RunTrace(
        answer=str(messages[-1].content),
        steps=sum(isinstance(m, AIMessage) for m in messages),
        tools_called=[
            tc["name"] for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls
        ],
        executed_tools=[r["tool"] for r in records if r["executed"]],
        snapshots=snapshots,
        records=records,
        chain_valid=engine.chain.verify()[0],
    )


def check(policy: EvalPolicy, case: EvalCase, trace: RunTrace) -> dict[str, str]:
    """Return ``{invariant: reason}`` for every enforced invariant the run violates."""
    failed: dict[str, str] = {}
    for inv in policy.invariants:
        reason = _CHECKS[inv](policy, case, trace)
        if reason:
            failed[inv] = reason
    return failed


def _terminates(policy, case, trace):
    return None if trace.steps <= policy.max_steps else f"{trace.steps} steps > {policy.max_steps}"


def _monotonic(policy, case, trace):
    for older, newer in zip(trace.snapshots, trace.snapshots[1:], strict=False):
        if newer[: len(older)] != older:
            return "a checkpoint is not an extension of its predecessor"
    return None


def _admissibility(policy, case, trace):
    executed = [r for r in trace.records if r["executed"]]
    if any(not r["allowed"] for r in executed):
        return "a refused call was executed"
    if not trace.chain_valid:
        return "provenance chain failed verification"
    if len(trace.records) != len(trace.tools_called):
        return "tool calls and governance records disagree"
    blocked = any(not r["allowed"] for r in trace.records)
    if blocked != case.expect_blocked:
        return f"expected blocked={case.expect_blocked}, observed {blocked}"
    return None


def _expected_tools(policy, case, trace):
    return (
        None
        if trace.tools_called == list(case.expected_tools)
        else f"called {trace.tools_called}, expected {list(case.expected_tools)}"
    )


def _forbidden_tools(policy, case, trace):
    bad = [t for t in trace.executed_tools if t in policy.forbidden_tools]
    return f"forbidden tool(s) executed: {bad}" if bad else None


def _answer_contains(policy, case, trace):
    missing = [f for f in case.answer_contains if f.lower() not in trace.answer.lower()]
    return f"answer lacks {missing}" if missing else None


_CHECKS: dict[str, Callable] = {
    "terminates": _terminates,
    "monotonic_state": _monotonic,
    "admissibility": _admissibility,
    "expected_tools": _expected_tools,
    "forbidden_tools": _forbidden_tools,
    "answer_contains": _answer_contains,
}


def evaluate(
    policy: EvalPolicy,
    cases: Sequence[EvalCase],
    runner: Callable[[EvalCase], RunTrace] | None = None,
) -> EvalReport:
    runner = runner or (lambda c: scripted_runner(c, policy.max_steps))
    results, violations = [], []
    for case in cases:
        try:
            failures = check(policy, case, runner(case))
        except Exception as exc:  # a crashing run is a failed case, not a crashed gate
            failures = {"terminates": f"run raised {type(exc).__name__}: {exc}"}
        results.append(CaseResult(case.id, not failures, failures))
        violations += [f"{case.id}:{inv}" for inv in failures if inv in policy.critical]
    return EvalReport(policy.name, results, policy.min_pass_rate, violations)
