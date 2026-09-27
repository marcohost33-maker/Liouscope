"""Fail-closed contract for relaxation trajectory representability (#156).

These tests intentionally isolate numerical propagation from the separate
gap-scaled-grid / resolution work in PR #115.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics import relaxation
from liouscope.diagnostics.relaxation import (
    UnrepresentableTrajectoryError,
    _evolve,
)
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


def test_ordinary_action_trajectory_agrees_with_dense_reference() -> None:
    """The backend change is bounded against the former dense formula."""
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
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-14)


def test_nonfinite_scaled_generator_fails_before_matrix_exponential() -> None:
    """Finite L and t can still overflow in the dimensionless product L*t."""
    L = np.eye(4, dtype=complex) * 1.0e308
    grid = np.array([0.0, 2.0])
    assert np.all(np.isfinite(L))
    assert np.isfinite(grid).all()

    with pytest.raises(UnrepresentableTrajectoryError, match=r"L\*t contains non-finite"):
        _evolve(L, _RHO_PLUS, grid)


def test_extreme_finite_Lt_stays_fail_closed_under_action_backend() -> None:
    """The action backend must not turn the #156 extreme into a hang or NaN."""
    L = _extreme_thermalising_qubit()
    grid = np.array([0.0, 1.1])
    assert np.all(np.isfinite(L * grid[-1]))

    with pytest.raises(
        UnrepresentableTrajectoryError,
        match=r"\|\|L\*t\|\|_1|expm_multiply",
    ):
        _evolve(L, _RHO_PLUS, grid)


def test_scipy_runtime_warning_is_normalised_to_the_domain_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warnings-as-errors and ordinary callers must see the same contract."""

    def _warn(*args: object, **kwargs: object) -> np.ndarray:
        raise RuntimeWarning("synthetic exponential-action failure")

    monkeypatch.setattr(relaxation.spla, "expm_multiply", _warn)
    with pytest.raises(UnrepresentableTrajectoryError, match=r"expm_multiply"):
        _evolve(_ordinary_generator(), _RHO_PLUS, np.array([0.0, 1.0]))


def test_nonfinite_action_result_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The action backend must not pass NaN/inf states downstream."""

    def _nonfinite(*args: object, **kwargs: object) -> np.ndarray:
        return np.full(4, np.inf, dtype=complex)

    monkeypatch.setattr(relaxation.spla, "expm_multiply", _nonfinite)
    with pytest.raises(
        UnrepresentableTrajectoryError,
        match=r"non-finite",
    ):
        _evolve(
            _ordinary_generator(),
            _RHO_PLUS,
            np.array([0.0, 0.3, 1.0]),
        )
