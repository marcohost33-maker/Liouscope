"""Fail-closed boundary guards for public entry points.

Negative / edge inputs (NaN, inf, empty, wrong-shape) MUST raise a structured,
argument-named ``ValueError`` at the boundary -- before the data reaches
``scipy.linalg.expm`` / ``svd`` where it would surface as an opaque,
location-blind LAPACK message ("array must not contain infs or NaNs" /
"SVD did not converge"). This is the silent-/opaque-failure gate: edge inputs
are part of "done", not a later hardening afterthought.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope.numerics.linalg import require_finite_square_2d

# --- require_finite_square_2d ------------------------------------------------


def test_require_finite_accepts_valid_matrix():
    A = np.eye(3, dtype=complex)
    out = require_finite_square_2d(A, name="A")
    np.testing.assert_array_equal(out, A)


def test_require_finite_rejects_nan():
    A = np.eye(2, dtype=complex)
    A[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"non-finite.*1 NaN"):
        require_finite_square_2d(A, name="A")


def test_require_finite_rejects_inf():
    A = np.eye(2, dtype=complex)
    A[1, 1] = np.inf
    with pytest.raises(ValueError, match=r"non-finite.*1 inf"):
        require_finite_square_2d(A, name="A")


def test_require_finite_rejects_non_square():
    A = np.zeros((2, 3), dtype=complex)
    with pytest.raises(ValueError, match="square 2-D"):
        require_finite_square_2d(A, name="A")


def test_require_finite_rejects_empty():
    A = np.zeros((0, 0), dtype=complex)
    with pytest.raises(ValueError, match="non-empty"):
        require_finite_square_2d(A, name="A")


def test_require_finite_error_names_argument():
    A = np.full((2, 2), np.nan, dtype=complex)
    with pytest.raises(ValueError, match="my_operator"):
        require_finite_square_2d(A, name="my_operator")


# --- diagnose() boundary -----------------------------------------------------


def _good_L(pauli):
    return build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])


def test_diagnose_rejects_nan_liouvillian(pauli):
    L = _good_L(pauli)
    L[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"L_super.*non-finite"):
        diagnose(L, bootstrap_B=10, seed=1)


def test_diagnose_rejects_inf_liouvillian_even_with_steady_state(pauli):
    """The inf path previously reached scipy.expm with an opaque message.

    Supplying ``rho_steady_state`` bypasses the steady-state SVD, so without a
    boundary guard the inf would only be caught deep inside ``expm``.
    """
    L = _good_L(pauli)
    L[1, 1] = np.inf
    rho_ss = np.eye(2, dtype=complex) / 2
    with pytest.raises(ValueError, match=r"L_super.*non-finite"):
        diagnose(L, rho_steady_state=rho_ss, bootstrap_B=10, seed=1)


def test_diagnose_rejects_nan_rho_initial(pauli):
    L = _good_L(pauli)
    rho0 = np.eye(2, dtype=complex) / 2
    rho0[0, 1] = np.nan
    with pytest.raises(ValueError, match=r"rho_initial.*non-finite"):
        diagnose(L, rho_initial=rho0, bootstrap_B=10, seed=1)


def test_diagnose_rejects_mismatched_rho_initial_shape(pauli):
    L = _good_L(pauli)  # d = 2
    rho0 = np.eye(3, dtype=complex) / 3
    with pytest.raises(ValueError, match=r"rho_initial shape"):
        diagnose(L, rho_initial=rho0, bootstrap_B=10, seed=1)


def test_diagnose_rejects_mismatched_rho_steady_shape(pauli):
    L = _good_L(pauli)  # d = 2
    rho_ss = np.eye(3, dtype=complex) / 3
    with pytest.raises(ValueError, match=r"rho_steady_state shape"):
        diagnose(L, rho_steady_state=rho_ss, bootstrap_B=10, seed=1)


def test_diagnose_still_accepts_valid_input(pauli):
    """Guard must not reject legitimate finite input (no false positives)."""
    L = _good_L(pauli)
    report = diagnose(L, bootstrap_B=10, seed=1)
    assert np.isfinite(report.spectral.gap)
