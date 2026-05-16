"""Smoke tests for sparse/arnoldi.py."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from liouscope.sparse.arnoldi import sparse_spectrum, sparse_steady_state


def _diagonal_super_with_zero(d: int = 3) -> sp.csc_matrix:
    """Diagonal Liouvillian: one zero eigenvalue + dissipative modes.

    Vec form: dimension n^2 = d^2. The (0,0) slot represents the
    diagonal steady-state component; we put zero there.
    """
    n2 = d * d
    diag = -np.arange(n2, dtype=complex) * 0.1
    diag[0] = 0.0
    return sp.diags(diag, format="csc")


def test_sparse_steady_state_diagonal_case():
    d = 3
    L = _diagonal_super_with_zero(d=d)
    rho = sparse_steady_state(L, tol=1.0e-10)
    assert rho.shape == (d, d)
    tr = np.trace(rho)
    assert np.isclose(tr.real, 1.0, atol=1.0e-9), "Steady state has unit trace."
    assert np.allclose(rho, rho.conj().T, atol=1.0e-9), "Hermitian after symmetrisation."


def test_sparse_spectrum_returns_k_eigenvalues():
    L = _diagonal_super_with_zero(d=3)
    vals, vecs = sparse_spectrum(L, k=4, tol=1.0e-10)
    assert vals.shape == (4,)
    assert vecs.shape[1] == 4
    # GKSL property: eigenvalues in left half-plane (with tiny numerical slack)
    assert np.all(np.real(vals) <= 1.0e-9), "Eigenvalues lie in closed left half-plane."
    # Sorted by descending real part
    reals = np.real(vals)
    assert np.all(np.diff(reals) <= 1.0e-12), "Eigenvalues sorted by descending Re(lambda)."


def test_sparse_spectrum_smallest_at_zero():
    """The slowest mode (steady state) is the first entry after sorting."""
    L = _diagonal_super_with_zero(d=3)
    vals, _ = sparse_spectrum(L, k=4, tol=1.0e-10)
    assert abs(vals[0]) < 1.0e-6, "Slowest mode should be close to zero (steady state)."
