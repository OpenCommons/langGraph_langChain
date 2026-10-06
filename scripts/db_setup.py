#!/usr/bin/env python3
"""Create / migrate the durable LangGraph tables (checkpointer + store). Idempotent.

Usage (from the repo root, with CHECKPOINT_BACKEND=postgres and DATABASE_URL set):

    python3 scripts/db_setup.py          # or: make db-setup
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "services" / "api"))

from graphs.checkpoint import reset_checkpointer, setup_persistence  # noqa: E402


def main() -> int:
    try:
        backend = setup_persistence()
    except Exception as exc:
        print(f"db-setup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        reset_checkpointer()
    print(f"db-setup ok ({backend})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
