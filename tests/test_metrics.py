"""Percentiles and the warm-up-aware summary. Pure Python, runs on a clean clone."""

from __future__ import annotations

import math

import pytest

from src.metrics import percentile


def _nearest_rank(values, pct):
    """Textbook nearest-rank, written independently of the implementation."""
    ordered = sorted(values)
    rank = max(1, math.ceil(pct * len(ordered) / 100))
    return ordered[rank - 1]


@pytest.mark.parametrize(
    "n, pct, expected",
    [
        (20, 95, 18),   # rank 19 -- the old round(x + 0.5) returned the max
        (20, 5, 0),
        (20, 50, 9),
        (300, 95, 284),  # the benchmark's default frame count
        (300, 5, 14),
        (100, 7, 6),    # 0.07 * 100 is not exactly 7.0 in floating point
        (10, 100, 9),
        (10, 0, 0),
    ],
)
def test_percentile_is_nearest_rank(n, pct, expected):
    assert percentile(list(range(n)), pct) == expected


@pytest.mark.parametrize("n", [1, 2, 3, 7, 20, 99, 100, 101, 300, 330])
@pytest.mark.parametrize("pct", [5, 25, 50, 75, 95, 99])
def test_percentile_matches_the_reference_definition(n, pct):
    values = list(range(n))
    assert percentile(values, pct) == _nearest_rank(values, pct)


def test_percentile_ignores_input_order():
    assert percentile([5, 1, 4, 2, 3], 50) == 3


def test_percentile_of_nothing_is_zero():
    assert percentile([], 95) == 0.0
