"""Cross-thread long-term memory (LangGraph ``BaseStore``).

Hierarchical tuple namespaces, e.g. ``("memories", user_id)``. PostgreSQL-backed
(``PostgresStore``) when ``CHECKPOINT_BACKEND=postgres``; otherwise an in-memory
store that lives as long as the process.
"""

from __future__ import annotations

import threading

from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from config import get_settings

_store: BaseStore | None = None
_lock = threading.Lock()


def get_store() -> BaseStore:
    global _store
    use_postgres = get_settings().checkpoint_backend == "postgres"
    if use_postgres:
        from graphs.checkpoint import postgres_pool

        pool = postgres_pool()
    with _lock:
        if _store is None:
            if use_postgres:
                from langgraph.store.postgres import PostgresStore

                _store = PostgresStore(pool)
            else:
                _store = InMemoryStore()
        return _store


def reset_store() -> None:
    global _store
    with _lock:
        _store = None
