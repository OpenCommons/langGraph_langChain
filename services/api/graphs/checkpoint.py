"""SQLite checkpointer: thread-scoped, durable LangGraph state."""

from __future__ import annotations

import pathlib
import sqlite3
import threading

from langgraph.checkpoint.sqlite import SqliteSaver

from config import get_settings

_saver: SqliteSaver | None = None
_lock = threading.Lock()


def get_checkpointer() -> SqliteSaver:
    global _saver
    with _lock:
        if _saver is None:
            path = get_settings().checkpoint_db_path
            if path != ":memory:":
                pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            _saver = SqliteSaver(sqlite3.connect(path, check_same_thread=False))
        return _saver


def reset_checkpointer() -> None:
    global _saver
    _saver = None
