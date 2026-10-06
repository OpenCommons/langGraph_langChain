"""Algebraic properties of shared LangGraph state reducers."""

import random
import string
from datetime import UTC, datetime, timedelta
from typing import Annotated, get_type_hints

from graphs.state import (
    BaseGraphState,
    StateFactory,
    reduce_latest,
    reduce_max,
    reduce_set_union,
)


def _assert_reducer_laws(reducer, values, rng):
    for left in values:
        assert reducer(left, left) == left

    for _ in range(100):
        left, middle, right = (rng.choice(values) for _ in range(3))
        assert reducer(left, middle) == reducer(middle, left)
        assert reducer(reducer(left, middle), right) == reducer(left, reducer(middle, right))


def test_set_union_reducer_laws_with_generated_sets():
    rng = random.Random(231)
    values = [set(rng.sample(range(20), rng.randrange(8))) for _ in range(100)]
    _assert_reducer_laws(reduce_set_union("labels"), values, rng)


def test_max_reducer_laws_with_generated_numbers():
    rng = random.Random(232)
    values = [None, *(rng.randint(-100, 100) for _ in range(100))]
    _assert_reducer_laws(reduce_max("steps"), values, rng)


def test_latest_reducer_laws_with_generated_values():
    rng = random.Random(233)
    values = [
        None,
        *("".join(rng.choices(string.ascii_letters, k=rng.randint(1, 12))) for _ in range(100)),
    ]
    _assert_reducer_laws(reduce_latest("status"), values, rng)


def test_latest_reducer_laws_with_generated_timestamps():
    rng = random.Random(234)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    values = [None, *(start + timedelta(seconds=rng.randrange(100_000)) for _ in range(100))]
    _assert_reducer_laws(reduce_latest("updated_at"), values, rng)


def test_state_factory_subclass_preserves_base_and_domain_annotations():
    class DomainState(StateFactory, total=False):
        labels: Annotated[set[str], reduce_set_union("labels")]

    annotations = get_type_hints(DomainState, include_extras=True)
    base_annotations = get_type_hints(BaseGraphState, include_extras=True)

    assert annotations["messages"] == base_annotations["messages"]
    assert annotations["labels"].__metadata__[0].__name__ == "reduce_set_union_labels"
