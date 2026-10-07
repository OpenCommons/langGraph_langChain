"""
Merkle provenance chain: append-only, ``h_L = SHA256(h_{L-1} || S0 || frame || capability_token_id)``.

``S0`` (the initial active-state configuration) and the frame are canonical JSON,
each length-prefixed so field boundaries cannot be shifted. The chain is
tamper-evident, not tamper-proof: whoever can rewrite the whole log and its head
can forge it. Anchor ``head`` externally for stronger guarantees.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import threading
from typing import Any

GENESIS = "0" * 64


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def link_hash(prev: str, s0: Any, frame: Any, token_id: str) -> str:
    h = hashlib.sha256()
    for part in (prev.encode(), _canonical(s0), _canonical(frame), token_id.encode()):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


def verify_records(records: list[dict]) -> tuple[bool, int | None]:
    """Recompute every link. Returns ``(ok, index_of_first_bad_record)``."""
    prev = GENESIS
    for i, rec in enumerate(records):
        if rec.get("index") != i or rec.get("prev") != prev:
            return False, i
        if rec.get("hash") != link_hash(
            prev, rec.get("s0"), rec.get("frame"), rec.get("token_id", "")
        ):
            return False, i
        prev = rec["hash"]
    return True, None


class ProvenanceChain:
    def __init__(self, path: str | None = None) -> None:
        self._records: list[dict] = []
        self._lock = threading.Lock()
        self._path = pathlib.Path(path) if path else None
        if self._path and self._path.exists():
            for line in self._path.read_text().splitlines():
                if line.strip():
                    self._records.append(json.loads(line))

    def append(self, s0: Any, frame: Any, token_id: str) -> dict:
        with self._lock:
            prev = self._records[-1]["hash"] if self._records else GENESIS
            rec = {
                "index": len(self._records),
                "prev": prev,
                "s0": s0,
                "frame": frame,
                "token_id": token_id,
                "hash": link_hash(prev, s0, frame, token_id),
            }
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a") as fh:
                    fh.write(json.dumps(rec, sort_keys=True) + "\n")
            self._records.append(rec)
            return rec

    @property
    def head(self) -> str:
        with self._lock:
            return self._records[-1]["hash"] if self._records else GENESIS

    def __len__(self) -> int:
        return len(self._records)

    def records(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            recs = list(self._records)
        return recs[-limit:] if limit else recs

    def verify(self) -> tuple[bool, int | None]:
        return verify_records(self.records())
