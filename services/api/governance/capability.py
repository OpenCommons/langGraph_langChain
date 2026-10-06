"""
Capability tokens: restrict which tools and scopes an agent may use.

A token is ``base64url(JSON payload) . hex(HMAC-SHA256)`` signed with a shared
secret. A child token can only narrow its parent: tools and scopes must be
subsets (``"*"`` in the parent allows anything). Future work: replace the shared
secret with SPIFFE SVIDs (workload identity) behind the same ``verify`` /
``delegate`` interface — see docs/GOVERNANCE.md.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

WILDCARD = "*"


class CapabilityError(Exception):
    pass


@dataclass(frozen=True)
class Capability:
    id: str
    subject: str
    tools: frozenset[str]
    scopes: frozenset[str]
    parent: str | None = None
    expires_at: float | None = None

    def allows_tool(self, tool: str) -> bool:
        return WILDCARD in self.tools or tool in self.tools

    def allows_scope(self, scope: str) -> bool:
        return WILDCARD in self.scopes or scope in self.scopes


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _subset(child: frozenset[str], parent: frozenset[str]) -> bool:
    return WILDCARD in parent or (WILDCARD not in child and child <= parent)


class CapabilityAuthority:
    def __init__(self, secret: str | bytes) -> None:
        if not secret:
            raise ValueError("capability secret must not be empty")
        self._secret = secret.encode() if isinstance(secret, str) else secret

    def _sign(self, body: str) -> str:
        return hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()

    def mint(
        self,
        subject: str,
        tools: Iterable[str],
        scopes: Iterable[str],
        *,
        parent: Capability | None = None,
        ttl_s: float | None = None,
        now: float | None = None,
    ) -> str:
        tools_f, scopes_f = frozenset(tools), frozenset(scopes)
        expires = None if ttl_s is None else (now if now is not None else time.time()) + ttl_s
        if parent is not None:
            if not _subset(tools_f, parent.tools) or not _subset(scopes_f, parent.scopes):
                raise CapabilityError("child scope must be a subset of parent scope")
            if parent.expires_at is not None and (expires is None or expires > parent.expires_at):
                expires = parent.expires_at
        payload = {
            "id": uuid.uuid4().hex,
            "sub": subject,
            "tools": sorted(tools_f),
            "scopes": sorted(scopes_f),
            "parent": parent.id if parent else None,
            "exp": expires,
        }
        body = _b64(json.dumps(payload, sort_keys=True).encode())
        return f"{body}.{self._sign(body)}"

    def verify(self, token: str | None, now: float | None = None) -> Capability:
        if not token or "." not in token:
            raise CapabilityError("missing or malformed capability token")
        body, _, sig = token.rpartition(".")
        if not hmac.compare_digest(sig, self._sign(body)):
            raise CapabilityError("bad capability signature")
        try:
            p = json.loads(_unb64(body))
            cap = Capability(
                p["id"],
                p["sub"],
                frozenset(p["tools"]),
                frozenset(p["scopes"]),
                p["parent"],
                p["exp"],
            )
        except (ValueError, KeyError, TypeError) as exc:
            raise CapabilityError("unreadable capability payload") from exc
        if (
            cap.expires_at is not None
            and (now if now is not None else time.time()) >= cap.expires_at
        ):
            raise CapabilityError("capability expired")
        return cap

    def delegate(
        self,
        parent_token: str,
        subject: str,
        tools: Iterable[str],
        scopes: Iterable[str],
        ttl_s: float | None = None,
    ) -> str:
        return self.mint(subject, tools, scopes, parent=self.verify(parent_token), ttl_s=ttl_s)

    def authorize(self, token: str | None, tool: str, scope: str) -> Capability:
        cap = self.verify(token)
        if not cap.allows_tool(tool):
            raise CapabilityError(f"tool {tool!r} not permitted")
        if not cap.allows_scope(scope):
            raise CapabilityError(f"scope {scope!r} not permitted")
        return cap
