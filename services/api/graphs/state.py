"""Shared state schemas and reducers for LangGraph workflows."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Annotated, Any, Protocol, TypedDict, TypeVar

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

_ItemT = TypeVar("_ItemT", bound=Hashable)


class _SupportsLessThan(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


class _SupportsGreaterThan(Protocol):
    def __gt__(self, other: Any, /) -> bool: ...


_ValueT = TypeVar("_ValueT", bound=_SupportsLessThan | _SupportsGreaterThan)


class BaseGraphState(TypedDict):
    """Common graph state with LangChain's ordered message reducer."""

    messages: Annotated[list[BaseMessage], add_messages]


class StateFactory(BaseGraphState, total=False):
    """TypedDict base for defining domain-specific graph states."""


def reduce_set_union(
    field_name: str,
) -> Callable[[set[_ItemT] | None, set[_ItemT] | None], set[_ItemT]]:
    """Build a reducer that merges a set field with an order-independent union."""
    if not field_name:
        raise ValueError("field_name must not be empty")

    def reducer(left: set[_ItemT] | None, right: set[_ItemT] | None) -> set[_ItemT]:
        return (set() if left is None else left) | (set() if right is None else right)

    return reducer


def reduce_max(field_name: str) -> Callable[[_ValueT | None, _ValueT | None], _ValueT | None]:
    """Build a reducer that keeps the maximum comparable value (None is the identity)."""
    if not field_name:
        raise ValueError("field_name must not be empty")

    def reducer(left: _ValueT | None, right: _ValueT | None) -> _ValueT | None:
        if left is None:
            return right
        if right is None:
            return left
        try:
            return max(left, right)
        except TypeError as exc:
            raise TypeError(f"values for {field_name!r} must be mutually comparable") from exc

    return reducer


def reduce_latest(field_name: str) -> Callable[[_ValueT | None, _ValueT | None], _ValueT | None]:
    """Build a deterministic latest-value reducer using the values' total ordering.

    Values must carry an ordering that represents recency (for example, an ISO
    timestamp or a ``(version, value)`` tuple) when chronological LWW is needed.
    """
    return reduce_max(field_name)
