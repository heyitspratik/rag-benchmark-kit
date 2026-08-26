from itertools import pairwise

import pytest

from rag_bench.components.chunkers.spans import SpanBudget
from rag_bench.core.exceptions import ConfigValidationError


def test_spans_within_budget_are_left_alone() -> None:
    budget = SpanBudget(100)

    assert budget.enforce([(0, 50), (50, 130)]) == [(0, 50), (50, 130)]


def test_an_over_long_span_is_split() -> None:
    budget = SpanBudget(100)

    assert budget.enforce([(0, 250)]) == [(0, 100), (100, 200), (200, 250)]


def test_splitting_covers_the_span_without_gaps_or_overlap() -> None:
    pieces = SpanBudget(30).hard_split(10, 105)

    assert pieces[0][0] == 10
    assert pieces[-1][1] == 105
    assert all(a[1] == b[0] for a, b in pairwise(pieces))


def test_a_span_exactly_at_the_budget_is_not_split() -> None:
    budget = SpanBudget(100)

    assert budget.fits(0, 100)
    assert budget.enforce([(0, 100)]) == [(0, 100)]


def test_an_empty_span_list_stays_empty() -> None:
    assert SpanBudget(100).enforce([]) == []


@pytest.mark.parametrize("max_chars", [0, -1])
def test_a_non_positive_budget_is_rejected(max_chars: int) -> None:
    with pytest.raises(ConfigValidationError, match="max_chars must be positive"):
        SpanBudget(max_chars)
