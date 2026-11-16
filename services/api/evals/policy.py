"""Evaluation policy: which checks apply to a run, and what pass rate gates a build."""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass, field

# Invariants the harness knows how to check. A policy lists the ones it enforces.
INVARIANTS = (
    "terminates",  # steps stay within the recursion limit
    "monotonic_state",  # thread state only accumulates (join-semilattice of messages)
    "admissibility",  # a call governance refused is never executed; every executed call recorded
    "expected_tools",  # the case's expected tool sequence occurred
    "forbidden_tools",  # no tool outside the policy's / case's allow-list ran
    "answer_contains",  # the answer contains every required fragment
)


@dataclass(frozen=True)
class EvalPolicy:
    name: str
    invariants: tuple[str, ...]
    # Fraction of cases that must pass for the gate to pass (CI gate: 0.9).
    min_pass_rate: float = 0.9
    # Invariants that must hold on *every* case regardless of pass rate.
    critical: tuple[str, ...] = ()
    max_steps: int = 25
    forbidden_tools: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        unknown = (set(self.invariants) | set(self.critical)) - set(INVARIANTS)
        if unknown:
            raise ValueError(f"unknown invariants in policy {self.name!r}: {sorted(unknown)}")
        if not 0.0 <= self.min_pass_rate <= 1.0:
            raise ValueError("min_pass_rate must be within [0, 1]")
        if not set(self.critical) <= set(self.invariants):
            raise ValueError("critical invariants must also be listed in invariants")

    @classmethod
    def from_dict(cls, d: Mapping) -> EvalPolicy:
        return cls(
            name=d["name"],
            invariants=tuple(d["invariants"]),
            min_pass_rate=float(d.get("min_pass_rate", 0.9)),
            critical=tuple(d.get("critical", ())),
            max_steps=int(d.get("max_steps", 25)),
            forbidden_tools=tuple(d.get("forbidden_tools", ())),
        )


def data_dir() -> pathlib.Path:
    """``EVALS_DIR`` or the repo's ``data/evals`` (found by walking up from here)."""
    if os.environ.get("EVALS_DIR"):
        return pathlib.Path(os.environ["EVALS_DIR"])
    for parent in pathlib.Path(__file__).resolve().parents:
        if (parent / "data" / "evals").is_dir():
            return parent / "data" / "evals"
    raise FileNotFoundError("data/evals not found; set EVALS_DIR")


def load_policy(name: str = "oee", directory: pathlib.Path | None = None) -> EvalPolicy:
    path = (directory or data_dir()) / "policies" / f"{name}.json"
    return EvalPolicy.from_dict(json.loads(path.read_text()))
