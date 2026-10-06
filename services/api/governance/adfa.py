"""
aDFA guard: a feed-forward, single-transition deterministic finite automaton.

Input is a normalized vector in ``[0,1]^N``. Each rule is a Critical Event
Sequence expressed as interval (axis-aligned polytope) bounds over the vector
components: a rule matches iff *every* bounded component lies inside its
``[lo, hi]`` interval. Evaluation is one pass over the rules and their bounds
— no recursion, no loopback, no state carried between inputs. A match is a
hazard: the output vector is clamped to zeros and the action is blocked.

Cost is O(R·B) for R rules with B bounds each; for a fixed compiled rule set
that is linear in the number of bounded components touched, independent of the
history length. It is a Python runtime guarantee, not a hard real-time one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    """One hazard CES: ``bounds`` maps a feature index to an inclusive ``(lo, hi)``."""

    name: str
    bounds: tuple[tuple[int, float, float], ...]
    modes: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if not self.bounds:
            raise ValueError(f"rule {self.name!r} has no bounds")
        for index, lo, hi in self.bounds:
            if index < 0 or not (0.0 <= lo <= hi <= 1.0):
                raise ValueError(f"rule {self.name!r}: bad bound ({index}, {lo}, {hi})")

    def active_in(self, s0: Mapping | None) -> bool:
        """A rule scoped to ``modes`` only applies when ``S0['mode']`` is one of them."""
        if self.modes is None:
            return True
        return (s0 or {}).get("mode") in self.modes

    def matches(self, vector: Sequence[float]) -> bool:
        return all(
            index < len(vector) and lo <= vector[index] <= hi for index, lo, hi in self.bounds
        )

    def widened(self, margin: float) -> Rule:
        """Same rule with every interval grown by ``margin`` (clamped to [0,1])."""
        grown = tuple(
            (i, max(0.0, lo - margin), min(1.0, hi + margin)) for i, lo, hi in self.bounds
        )
        return Rule(self.name, grown, self.modes)


@dataclass(frozen=True)
class Verdict:
    blocked: bool
    rule: str | None
    output: tuple[float, ...]


@dataclass(frozen=True)
class RuleSet:
    """Immutable, compiled rule set. Never mutated after construction (swap = new object)."""

    version: str
    dim: int
    rules: tuple[Rule, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for rule in self.rules:
            if any(index >= self.dim for index, _, _ in rule.bounds):
                raise ValueError(f"rule {rule.name!r} references a component outside N={self.dim}")

    def step(self, vector: Sequence[float], s0: Mapping | None = None) -> Verdict:
        """The single transition: first matching active rule wins; hazard clamps to zeros."""
        if len(vector) != self.dim:
            raise ValueError(f"expected a vector of {self.dim} components, got {len(vector)}")
        if any(not (0.0 <= v <= 1.0) for v in vector):
            raise ValueError("input vector must be normalized to [0,1]")
        for rule in self.rules:
            if rule.active_in(s0) and rule.matches(vector):
                return Verdict(True, rule.name, (0.0,) * self.dim)
        return Verdict(False, None, tuple(vector))

    def widened(self, margin: float, version: str) -> RuleSet:
        return RuleSet(version, self.dim, tuple(r.widened(margin) for r in self.rules))

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "dim": self.dim,
            "rules": [
                {
                    "name": r.name,
                    "bounds": [list(b) for b in r.bounds],
                    **({"modes": sorted(r.modes)} if r.modes is not None else {}),
                }
                for r in self.rules
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping, features: Sequence[str] | None = None) -> RuleSet:
        """Compile from JSON. Bounds are ``{feature_name: [lo, hi]}`` (needs ``features``)
        or a list of ``[index, lo, hi]``."""
        dim = int(data.get("dim", len(features or ())))
        rules = []
        for r in data.get("rules", []):
            raw = r["bounds"]
            if isinstance(raw, Mapping):
                if features is None:
                    raise ValueError("named bounds need a feature list")
                names = list(features)
                bounds = tuple((names.index(k), float(v[0]), float(v[1])) for k, v in raw.items())
            else:
                bounds = tuple((int(i), float(lo), float(hi)) for i, lo, hi in raw)
            modes = frozenset(r["modes"]) if r.get("modes") is not None else None
            rules.append(Rule(str(r["name"]), bounds, modes))
        return cls(str(data.get("version", "unversioned")), dim, tuple(rules))
