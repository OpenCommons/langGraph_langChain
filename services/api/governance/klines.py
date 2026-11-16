"""
K-line store and deterministic counterfactual replay.

A K-line is ``(S0, frames)``: the initial active-state configuration and the
ordered input vectors observed from it, labelled ``hazard`` or ``nominal``.
Replay mirrors the manuscript's Appendix D Lisp pseudocode (the manuscript is
not in this repository, so the shape below is a transcription of its intent):

    (defun replay (rules s0 frames)            ; one pass, no side effects
      (mapcar (lambda (f) (adfa-step rules s0 f)) frames))

    (defun counterfactual (rules klines)
      (let ((hazards (remove-if-not #'hazard-p klines))
            (nominal (remove-if     #'hazard-p klines)))
        (list :all-hazards-clamped (every (lambda (k) (some #'blocked (replay-kline rules k))) hazards)
              :false-positives     (count-if (lambda (k) (some #'blocked (replay-kline rules k))) nominal))))
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

from governance.adfa import RuleSet, Verdict

HAZARD = "hazard"
NOMINAL = "nominal"


@dataclass(frozen=True)
class KLine:
    s0: Mapping
    frames: tuple[tuple[float, ...], ...]
    label: str = NOMINAL

    @classmethod
    def from_dict(cls, d: Mapping) -> KLine:
        label = d.get("label", NOMINAL)
        if label not in (HAZARD, NOMINAL):
            raise ValueError(f"bad k-line label {label!r}")
        return cls(dict(d.get("s0", {})), tuple(tuple(map(float, f)) for f in d["frames"]), label)


@dataclass(frozen=True)
class ReplayReport:
    hazard_total: int
    hazards_clamped: int
    nominal_total: int
    false_positives: int

    @property
    def all_hazards_clamped(self) -> bool:
        return self.hazards_clamped == self.hazard_total

    def to_dict(self) -> dict:
        return {**asdict(self), "all_hazards_clamped": self.all_hazards_clamped}


def replay_kline(rules: RuleSet, s0: Mapping, frames: Sequence[Sequence[float]]) -> list[Verdict]:
    """Deterministically re-run ``frames`` from ``S0`` under ``rules``."""
    return [rules.step(frame, s0) for frame in frames]


def replay(rules: RuleSet, klines: Iterable[KLine]) -> ReplayReport:
    hazard_total = hazards_clamped = nominal_total = false_positives = 0
    for k in klines:
        clamped = any(v.blocked for v in replay_kline(rules, k.s0, k.frames))
        if k.label == HAZARD:
            hazard_total += 1
            hazards_clamped += clamped
        else:
            nominal_total += 1
            false_positives += clamped
    return ReplayReport(hazard_total, hazards_clamped, nominal_total, false_positives)


class KLineStore:
    """Bounded in-memory store: seeded corpus is kept; observed K-lines roll over."""

    def __init__(self, seed: Iterable[KLine] = (), max_observed: int = 1000) -> None:
        self._seed = tuple(seed)
        self._observed: deque[KLine] = deque(maxlen=max_observed)
        self._lock = threading.Lock()

    def add(self, s0: Mapping, frames: Sequence[Sequence[float]], label: str) -> KLine:
        k = KLine(dict(s0), tuple(tuple(f) for f in frames), label)
        with self._lock:
            self._observed.append(k)
        return k

    def all(self) -> list[KLine]:
        with self._lock:
            return [*self._seed, *self._observed]

    def counts(self) -> dict:
        ks = self.all()
        return {
            "total": len(ks),
            "hazard": sum(k.label == HAZARD for k in ks),
            "nominal": sum(k.label == NOMINAL for k in ks),
        }
