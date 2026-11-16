"""
Context encoder: the *whole thread* (not just the latest prompt) plus the proposed
action become a normalized ``[0,1]^N`` vector for the aDFA.

Messages are lower-cased and squashed to ``[a-z0-9]`` and concatenated before the
lexicon is counted, so a phrase split across turns or padded with punctuation
("dr", "op   ta-ble") is still seen — the fragmentation-bypass and
context-blindness failures of per-prompt filters. This is a lexical heuristic: it
is not semantic understanding, and a paraphrase outside the lexicon is not seen.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

_SQUASH = re.compile(r"[^a-z0-9]+")


def squash(text: str) -> str:
    return _SQUASH.sub("", text.lower())


class ContextEncoder:
    def __init__(self, policy: Mapping) -> None:
        self.features: tuple[str, ...] = tuple(policy["features"])
        self._saturation = float(policy.get("saturation", 2))
        self._lexicon = {
            name: tuple(squash(t) for t in terms) for name, terms in policy["lexicon"].items()
        }
        self._tool_risk = dict(policy.get("tool_risk", {}))
        self._default_risk = float(policy.get("default_tool_risk", 0.5))

    def tool_risk(self, tool: str) -> float:
        return float(self._tool_risk.get(tool, self._default_risk))

    def encode(self, context: Sequence[str], tool: str, args: Mapping | str | None) -> list[float]:
        text = squash("".join(context) + _args_text(args))
        out = []
        for feature in self.features:
            if feature == "tool_risk":
                value = self.tool_risk(tool)
            else:
                hits = sum(text.count(term) for term in self._lexicon.get(feature, ()) if term)
                value = min(1.0, hits / self._saturation)
            out.append(round(value, 4))
        return out


def _args_text(args: Mapping | str | None) -> str:
    if args is None:
        return ""
    return args if isinstance(args, str) else json.dumps(args, default=str, sort_keys=True)
