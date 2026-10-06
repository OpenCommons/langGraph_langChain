"""Algebraic properties of shared graph-state reducers."""

from hypothesis import given, strategies as st

from graphs.state import reduce_latest, reduce_max, reduce_set_union


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
    st.sets(st.integers()),
)
def test_set_union_handles_none_as_identity(first, second, third):
    reducer = reduce_set_union("items")
    assert reducer(None, first) == first
    assert reducer(first, None) == first
    assert reducer(None, None) == set()


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
