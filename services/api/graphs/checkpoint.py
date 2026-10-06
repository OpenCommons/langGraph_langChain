"""Thread-scoped, durable LangGraph state: SQLite (default) or PostgreSQL.

``CHECKPOINT_BACKEND=postgres`` + ``DATABASE_URL`` selects ``PostgresSaver`` over a
connection pool, so every API replica shares one durable thread state. Tables are
created / migrated by :func:`setup_persistence` (``make db-setup``).
"""

from __future__ import annotations

import pathlib
import sqlite3
import threading

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver

from config import get_settings

_saver: BaseCheckpointSaver | None = None
_pool = None
_lock = threading.Lock()


def postgres_pool():
    """Process-wide psycopg connection pool (shared by the saver and the store)."""
    global _pool
    with _lock:
        if _pool is None:
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            s = get_settings()
            if not s.database_url:
                raise RuntimeError("CHECKPOINT_BACKEND=postgres requires DATABASE_URL")
            _pool = ConnectionPool(
                conninfo=s.database_url,
                max_size=s.postgres_pool_size,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=True,
            )
        return _pool


def get_checkpointer() -> BaseCheckpointSaver:
    global _saver
    s = get_settings()
    if s.checkpoint_backend == "postgres":
        pool = postgres_pool()
        with _lock:
            if _saver is None:
                from langgraph.checkpoint.postgres import PostgresSaver

                _saver = PostgresSaver(pool)
            return _saver
    with _lock:
        if _saver is None:
            path = s.checkpoint_db_path
            if path != ":memory:":
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            _saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        return _saver


def setup_persistence() -> str:
    """Create / migrate the durable tables (idempotent). Returns the backend name."""
    s = get_settings()
    if s.checkpoint_backend != "postgres":
        get_checkpointer()  # SQLite creates its tables on first use
        return "sqlite"
    from graphs.store import get_store

    get_checkpointer().setup()  # type: ignore[attr-defined]
    get_store().setup()  # type: ignore[attr-defined]
    return "postgres"


def reset_checkpointer() -> None:
    global _saver, _pool
    with _lock:
        pool, _pool = _pool, None
        _saver = None
    if pool is not None:
        pool.close()
