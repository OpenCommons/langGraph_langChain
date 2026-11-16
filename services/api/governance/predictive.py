"""
Optional predictive hazard estimate (``GOVERNANCE_PREDICTIVE_ENABLED``).

A Markov transition matrix over K-line classes; ``P(K_{t+k} = hazard | K_t)``
is the hazard entry of ``T^k``. When it exceeds the threshold the caller stages
a tightened (widened-bounds) rule set — still subject to the replay gate.
This is an empirical estimate from observed frequencies, not a proof.
"""

from __future__ import annotations

import threading

CLASSES = ("nominal", "elevated", "hazard")
HAZARD_CLASS = "hazard"


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


class MarkovHazardEstimator:
    def __init__(self, classes: tuple[str, ...] = CLASSES, smoothing: float = 1.0) -> None:
        self.classes = classes
        self._ix = {c: i for i, c in enumerate(classes)}
        n = len(classes)
        self._counts = [[0.0] * n for _ in range(n)]
        self._smoothing = smoothing
        self._lock = threading.Lock()

    def observe(self, prev: str, nxt: str) -> None:
        with self._lock:
            self._counts[self._ix[prev]][self._ix[nxt]] += 1.0

    def matrix(self) -> list[list[float]]:
        with self._lock:
            counts = [row[:] for row in self._counts]
        n = len(self.classes)
        out = []
        for row in counts:
            total = sum(row) + self._smoothing * n
            out.append([(c + self._smoothing) / total for c in row])
        return out

    def p_hazard(self, current: str, k: int = 1) -> float:
        if k < 1:
            raise ValueError("k must be >= 1")
        t = self.matrix()
        power = t
        for _ in range(k - 1):
            power = _matmul(power, t)
        return power[self._ix[current]][self._ix[HAZARD_CLASS]]
