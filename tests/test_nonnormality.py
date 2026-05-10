"""Tests for non-normality layer D8-D11."""

from __future__ import annotations

import numpy as np

from liouscope import build_liouvillian
from liouscope.diagnostics.nonnormality import (
    bohr_arithmetic_progression,
    compute_nonnormality_layer,
    henrici_eta_n,
    kreiss_constant,
    petermann_factors,
)


def test_henrici_zero_for_normal_matrix():
    # Diagonal matrix is normal -> Schur upper triangle is zero.
    A = np.diag([1.0, 2.0, 3.0]).astype(complex)
    eta = henrici_eta_n(A)
    assert eta < 1e-9


def test_henrici_positive_for_non_normal_matrix():
    A = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=complex)
    eta = henrici_eta_n(A)
    assert eta > 0


def test_petermann_includes_conjugate_pairs(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.1])
    eigvals, K = petermann_factors(L)
    # Expect complex pair in eigenvalues
    assert np.any(np.abs(np.imag(eigvals)) > 1e-6)
    assert K.size == eigvals.size  # nothing dropped


def test_kreiss_positive(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    K = kreiss_constant(L, n_sigma=12, n_omega=12)
    assert K > 0


def test_bohr_ap_length_at_least_one():
    eigs = np.array([0.0, -0.5, -1.0, -1.5]).astype(complex)
    length, bound = bohr_arithmetic_progression(eigs, d=2)
    assert isinstance(length, int)
    assert length >= 1
    assert bound > 0


def test_compute_nonnormality_layer(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    res = compute_nonnormality_layer(L)
    assert res.henrici_eta > 0
    assert res.petermann_max > 0
    assert res.kreiss > 0
    assert res.bohr_ap_length >= 1
