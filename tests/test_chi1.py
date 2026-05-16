"""Smoke tests for sparse/chi1.py."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from liouscope.sparse.chi1 import chi1_lower_bound


def _toy_diagonal_liouvillian(d: int = 4) -> sp.csc_matrix:
    """Diagonal (normal) Liouvillian -> chi_1 lower bound should be ~1.0.

    For a perfectly normal operator the Petermann factor is 1, so the
    sqrt(K) lower bound is exactly 1.0.
    """
    diag = -np.arange(1, d * d + 1, dtype=complex) * 0.1
    diag[0] = 0.0  # steady state
    return sp.diags(diag, format="csc")


def test_chi1_returns_finite_for_normal_l():
    L = _toy_diagonal_liouvillian(d=3)
    chi = chi1_lower_bound(L, k_modes=3, tol=1.0e-7)
    assert np.isfinite(chi)
    assert chi >= 1.0 - 1.0e-6, "Normal Liouvillian gives Petermann factor >= 1."


def test_chi1_does_not_decrease_with_nonnormal_block():
    """A strongly non-normal block should not lower the chi_1 lower bound."""
    d = 3
    n2 = d * d
    L = sp.lil_matrix((n2, n2), dtype=complex)
    for i in range(n2):
        L[i, i] = -0.1 * (i + 1)
    # Add a non-normal off-diagonal coupling
    L[1, 2] = 2.0
    L[2, 3] = 2.0
    L_csc = L.tocsc()
    chi_normal = chi1_lower_bound(_toy_diagonal_liouvillian(d=d), k_modes=3, tol=1.0e-7)
    chi_jordan = chi1_lower_bound(L_csc, k_modes=3, tol=1.0e-7)
    assert chi_jordan >= chi_normal - 1.0e-6, (
        "Non-normal block should not decrease the chi_1 lower bound."
    )
