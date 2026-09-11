"""Regression probes for the PR #121 supplied-D24 availability boundary.

Arrived on ``pr107-fix`` as ``tests/test_pr121_review_round28.py`` and was
renamed on merge (2026-09-08), for two reasons and neither of them cosmetic.
A file of that name already existed on the branch it merged with, holding the
round-28 eigenpair-assignment review -- an unrelated subject, so keeping both
sets of assertions meant one of them had to move. And the finding these probes
pin is the round-27 one, which the source comment they were written against
says in as many words, so the round-27 label is the accurate one.

Not folded into ``test_pr121_review_round27.py``: these probes were written
independently of it, and two files that were meant to fail independently
should keep doing so.

The overlap was measured rather than assumed, and it is asymmetric. All six
sentinel cases below appear in that file's twelve-case parametrisation, which
additionally covers the mixed ``None``/NaN pairs and carries two positive
controls, and which asserts more per case (non-finite bounds AND verbatim
preservation). What is NOT covered there, and is the reason these stay, is
``test_finite_precomputed_values_still_take_the_fast_path``: it pins the exact
window -- ``log(1000)`` and ``log(2000)`` -- where the other file only asserts
that the bounds are finite and ordered.
"""

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
