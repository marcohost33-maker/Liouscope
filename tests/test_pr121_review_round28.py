"""Regression probes for the PR #121 supplied-D24 availability boundary."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope._zhou import compute_zhou_predictor


_L = np.diag([0.0, -1.0])


@pytest.mark.parametrize(
    ("gap", "petermann_factor"),
    [
        (float("nan"), float("nan")),
        (float("nan"), 2.0),
        (1.0, float("nan")),
        (float("inf"), 2.0),
        (1.0, float("inf")),
        (float("-inf"), 2.0),
    ],
)
def test_unavailable_supplied_diagnostics_cannot_publish_a_converged_d24(
    gap: float,
    petermann_factor: float,
) -> None:
    result = compute_zhou_predictor(
        _L,
        epsilon=1.0e-3,
        gap=gap,
        petermann_factor=petermann_factor,
    )

    assert result.converged is False
    assert np.isinf(result.mixing_time_lower)
    assert np.isinf(result.mixing_time_upper)

    if np.isnan(gap):
        assert np.isnan(result.gap)
    else:
        assert result.gap == gap
    if np.isnan(petermann_factor):
        assert np.isnan(result.petermann_factor)
    else:
        assert result.petermann_factor == petermann_factor


def test_finite_precomputed_values_still_take_the_fast_path() -> None:
    result = compute_zhou_predictor(
        _L,
        epsilon=1.0e-3,
        gap=1.0,
        petermann_factor=4.0,
    )

    assert result.converged is True
    assert result.gap == 1.0
    assert result.petermann_factor == 4.0
    assert result.mixing_time_lower == pytest.approx(np.log(1000.0))
    assert result.mixing_time_upper == pytest.approx(np.log(2000.0))


def test_existing_nonpositive_gap_abstention_is_unchanged() -> None:
    result = compute_zhou_predictor(
        _L,
        epsilon=1.0e-3,
        gap=-1.0,
        petermann_factor=4.0,
    )

    assert result.converged is False
    assert np.isinf(result.mixing_time_lower)
    assert np.isinf(result.mixing_time_upper)
    assert result.gap == -1.0
    assert result.petermann_factor == 4.0
