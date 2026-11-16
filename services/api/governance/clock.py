"""Lamport logical clock: a per-instance monotonic counter (no wall-clock time)."""

from __future__ import annotations

import threading


class LamportClock:
    """``tick()`` on a local event; ``receive(t)`` sets ``L = max(L, t) + 1``."""

    def __init__(self, start: int = 0) -> None:
        self._value = start
        self._lock = threading.Lock()

    @property
    def value(self) -> int:
        return self._value

    def tick(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def receive(self, message_time: int) -> int:
        with self._lock:
            self._value = max(self._value, int(message_time)) + 1
            return self._value
