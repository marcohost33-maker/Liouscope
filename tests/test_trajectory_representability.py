"""Fail-closed contract for relaxation trajectory representability (#156).

These tests intentionally isolate numerical propagation from the separate
gap-scaled-grid / resolution work in PR #115.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.relaxation import (
    UnrepresentableTrajectoryError,
    _evolve,
)
from liouscope.numerics import propagation
from liouscope.numerics.kronecker import unvec, vec

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_SP = _SM.conj().T
_RHO_PLUS = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)


def _ordinary_generator() -> np.ndarray:
    return build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.sqrt(0.7) * _SM],
    )


def _extreme_thermalising_qubit() -> np.ndarray:
    """Finite Liouvillian whose dense exponential is not representable."""
    rate = 8.5e307
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.sqrt(rate) * _SM, np.sqrt(rate) * _SP],
    )
    assert np.all(np.isfinite(L))
    return L


def test_ordinary_dense_trajectory_is_bitwise_the_existing_formula() -> None:
    """The guard must not change representable dense propagation."""
    L = _ordinary_generator()
    grid = np.array([0.0, 0.2, 1.0, 2.0])
    actual = _evolve(L, _RHO_PLUS, grid)

    expected = np.empty_like(actual)
    initial = vec(_RHO_PLUS)
    for k, t in enumerate(grid):
        if t == 0.0:
            expected[k] = _RHO_PLUS
        else:
            expected[k] = unvec(sla.expm(L * t) @ initial, d=2)
    np.testing.assert_array_equal(actual, expected)


def test_nonfinite_scaled_generator_fails_before_matrix_exponential() -> None:
    """Finite L and t can still overflow in the dimensionless product L*t."""
    L = np.eye(4, dtype=complex) * 1.0e308
    grid = np.array([0.0, 2.0])
    assert np.all(np.isfinite(L))
    assert np.isfinite(grid).all()

    with pytest.raises(UnrepresentableTrajectoryError, match=r"L\*t contains non-finite"):
        _evolve(L, _RHO_PLUS, grid)


def test_extreme_finite_Lt_fails_closed_when_dense_expm_is_unusable() -> None:
    """Finite L*t is necessary, not sufficient, for a usable dense propagator."""
    L = _extreme_thermalising_qubit()
    grid = np.array([0.0, 1.1])
    assert np.all(np.isfinite(L * grid[-1]))

    with pytest.raises(
        UnrepresentableTrajectoryError,
        match=r"scipy\.linalg\.expm",
    ):
        _evolve(L, _RHO_PLUS, grid)


def test_scipy_runtime_warning_is_normalised_to_the_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warnings-as-errors and ordinary callers must see the same contract."""

    def _warn(_scaled: np.ndarray) -> np.ndarray:
        raise RuntimeWarning("synthetic scaling-and-squaring failure")

    monkeypatch.setattr(propagation.sla, "expm", _warn)
    with pytest.raises(UnrepresentableTrajectoryError, match=r"scipy\.linalg\.expm"):
        _evolve(_ordinary_generator(), _RHO_PLUS, np.array([0.0, 1.0]))


def test_nonfinite_action_is_rejected_even_with_finite_propagator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finite propagator can still overflow when applied to the state."""
    huge = np.full((4, 4), 1.0e308, dtype=complex)
    assert np.all(np.isfinite(huge))
    monkeypatch.setattr(propagation.sla, "expm", lambda _scaled: huge)

    with pytest.raises(
        UnrepresentableTrajectoryError,
        match=r"propagated state became non-finite",
    ):
        _evolve(_ordinary_generator(), _RHO_PLUS, np.array([0.0, 1.0]))
