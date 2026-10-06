"""Shared state schemas and merge-safe reducers for LangGraph workflows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Protocol, TypedDict, TypeVar

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

__all__ = [
    "BaseGraphState",
    "StateFactory",
    "add_messages",
    "reduce_latest",
    "reduce_max",
    "reduce_set_union",
]


class SupportsOrdering(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...

    def __gt__(self, other: Any, /) -> bool: ...


SetItemT = TypeVar("SetItemT")
OrderedT = TypeVar("OrderedT", bound=SupportsOrdering)


def reduce_set_union(
    field_name: str,
) -> Callable[[set[SetItemT] | None, set[SetItemT] | None], set[SetItemT]]:
    """Build a reducer that merges set-valued fields without depending on update order."""

    def reducer(current: set[SetItemT] | None, update: set[SetItemT] | None) -> set[SetItemT]:
        return (current or set()) | (update or set())

    reducer.__name__ = f"reduce_set_union_{field_name}"
    return reducer


def reduce_max(
    field_name: str,
) -> Callable[[int | float | None, int | float | None], int | float | None]:
    """Build a max reducer for numeric fields, treating missing values as identity."""

    def reducer(current: int | float | None, update: int | float | None) -> int | float | None:
        if current is None:
            return update
        if update is None:
            return current
        return max(current, update)

    reducer.__name__ = f"reduce_max_{field_name}"
    return reducer


def reduce_latest(field_name: str) -> Callable[[OrderedT | None, OrderedT | None], OrderedT | None]:
    """Build a deterministic reducer for values whose natural order reflects recency.

    The greatest value wins, so timestamps resolve to the newest timestamp. For
    unorderable values, store an ordered version alongside the value instead.
    """

    def reducer(current: OrderedT | None, update: OrderedT | None) -> OrderedT | None:
        if current is None:
            return update
        if update is None:
            return current
        return max(current, update)

    reducer.__name__ = f"reduce_latest_{field_name}"
    return reducer


class BaseGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Subclass this TypedDict to define domain-specific state fields.
class StateFactory(BaseGraphState, total=False):
    pass
