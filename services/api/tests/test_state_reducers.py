"""Algebraic properties of shared graph-state reducers."""

from datetime import UTC
from typing import Annotated, get_type_hints

from hypothesis import given
from hypothesis import strategies as st

from graphs.state import BaseGraphState, StateFactory, reduce_latest, reduce_max, reduce_set_union


@given(st.sets(st.integers()), st.sets(st.integers()))
def test_set_union_is_commutative(left, right):
    reducer = reduce_set_union("items")
    assert reducer(left, right) == reducer(right, left)


@given(st.sets(st.integers()), st.sets(st.integers()), st.sets(st.integers()))
def test_set_union_is_associative(first, second, third):
    reducer = reduce_set_union("items")
    assert reducer(reducer(first, second), third) == reducer(first, reducer(second, third))


@given(st.sets(st.integers()))
def test_set_union_is_idempotent(items):
    assert reduce_set_union("items")(items, items) == items


@given(st.integers(), st.integers())
def test_max_is_commutative(left, right):
    reducer = reduce_max("steps")
    assert reducer(left, right) == reducer(right, left)


@given(st.integers(), st.integers(), st.integers())
def test_max_is_associative(first, second, third):
    reducer = reduce_max("steps")
    assert reducer(reducer(first, second), third) == reducer(first, reducer(second, third))


@given(st.integers())
def test_max_is_idempotent(value):
    assert reduce_max("steps")(value, value) == value


@given(st.one_of(st.none(), st.text()), st.one_of(st.none(), st.text()))
def test_latest_is_commutative(left, right):
    reducer = reduce_latest("result")
    assert reducer(left, right) == reducer(right, left)


@given(
    st.one_of(st.none(), st.text()),
    st.one_of(st.none(), st.text()),
    st.one_of(st.none(), st.text()),
)
def test_latest_is_associative(first, second, third):
    reducer = reduce_latest("result")
    assert reducer(reducer(first, second), third) == reducer(first, reducer(second, third))


@given(st.one_of(st.none(), st.text()))
def test_latest_is_idempotent(value):
    assert reduce_latest("result")(value, value) == value


@given(
    st.sets(st.integers()),
    st.sets(st.integers()),
)
def test_set_union_handles_none_as_identity(items, other):
    reducer = reduce_set_union("items")
    assert reducer(None, items) == items
    assert reducer(items, None) == items
    assert reducer(None, None) == set()
    assert reducer(items, other) == items | other


@given(st.integers())
def test_max_handles_none_as_identity(value):
    reducer = reduce_max("steps")
    assert reducer(None, value) == value
    assert reducer(value, None) == value
    assert reducer(None, None) is None


@given(st.one_of(st.none(), st.text()))
def test_latest_handles_none_as_identity(value):
    reducer = reduce_latest("result")
    assert reducer(None, value) == value
    assert reducer(value, None) == value
    assert reducer(None, None) is None


@given(
    st.datetimes(timezones=st.just(UTC)),
    st.datetimes(timezones=st.just(UTC)),
)
def test_latest_uses_timestamp_order(left, right):
    reducer = reduce_latest("updated_at")
    assert reducer(left, right) == max(left, right)


def test_state_factory_subclass_preserves_base_and_domain_annotations():
    class DomainState(StateFactory, total=False):
        labels: Annotated[set[str], reduce_set_union("labels")]

    annotations = get_type_hints(DomainState, include_extras=True)
    base_annotations = get_type_hints(BaseGraphState, include_extras=True)
    assert annotations["messages"] == base_annotations["messages"]
    assert annotations["labels"].__metadata__[0].__name__ == "reduce_set_union_labels"
