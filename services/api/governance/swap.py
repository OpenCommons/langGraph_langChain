"""
Atomic rule swap, no global lock.

A candidate :class:`RuleSet` is compiled off to the side and must pass the
K-line replay check before it is *staged*. ``stage`` is one reference
assignment; ``active()`` — called on each event — observes the staged
reference and adopts it with one more reference assignment. Readers take the
reference once per event, so an in-flight evaluation always sees one complete
immutable rule set: nothing blocks and nothing is dropped. (Atomicity relies on
CPython reference assignment being atomic; it is not a distributed guarantee.)
"""

from __future__ import annotations

from collections.abc import Callable

from governance.adfa import RuleSet
from governance.klines import KLine, ReplayReport, replay


class RuleSetRejected(Exception):
    def __init__(self, report: ReplayReport) -> None:
        super().__init__(
            f"candidate failed replay: {report.hazards_clamped}/{report.hazard_total} hazards "
            f"clamped, {report.false_positives} false positives"
        )
        self.report = report


class RuleManager:
    def __init__(
        self,
        initial: RuleSet,
        klines: Callable[[], list[KLine]],
        max_false_positives: int = 0,
    ) -> None:
        self._active = initial
        self._staged: RuleSet | None = None
        self._klines = klines
        self.max_false_positives = max_false_positives

    def check(self, candidate: RuleSet) -> ReplayReport:
        return replay(candidate, self._klines())

    def eligible(self, report: ReplayReport) -> bool:
        return report.all_hazards_clamped and report.false_positives <= self.max_false_positives

    def stage(self, candidate: RuleSet) -> ReplayReport:
        """Replay-gate then stage. Raises :class:`RuleSetRejected` if ineligible."""
        if candidate.dim != self._active.dim:
            raise ValueError(f"candidate N={candidate.dim} != active N={self._active.dim}")
        report = self.check(candidate)
        if not self.eligible(report):
            raise RuleSetRejected(report)
        self._staged = candidate  # single atomic reference assignment
        return report

    def active(self) -> RuleSet:
        """Call once per event; adopts a staged rule set on the next event."""
        staged = self._staged
        if staged is not None and staged is not self._active:
            self._active = staged
        return self._active

    @property
    def pending_version(self) -> str | None:
        staged = self._staged
        return staged.version if staged is not None and staged is not self._active else None

    def peek(self) -> RuleSet:
        """Read-only view of the active rule set; never adopts a staged one."""
        return self._active
