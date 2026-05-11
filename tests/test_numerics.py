"""Tests for the numerics helpers."""

from __future__ import annotations

import numpy as np

from liouscope.numerics.kronecker import unvec, vec
from liouscope.numerics.linalg import (
    eig_nonhermitian,
    is_density_matrix,
    is_hermitian,
    matrix_2_norm,
)
from liouscope.numerics.resolvent import resolvent_apply_superlu, resolvent_norm


def test_vec_rejects_non_square():
    import pytest
    with pytest.raises(ValueError):
        vec(np.zeros((2, 3)))


def test_unvec_rejects_bad_size():
    import pytest
    with pytest.raises(ValueError):
        unvec(np.zeros(7))


def test_is_hermitian_true(pauli):
    assert is_hermitian(pauli["Z"])


def test_is_hermitian_false():
    A = np.array([[1, 2], [0, 3]], dtype=complex)
    assert not is_hermitian(A)


def test_is_hermitian_rejects_non_square():
    A = np.zeros((2, 3))
    assert not is_hermitian(A)


def test_is_density_matrix_true():
    rho = np.diag([0.4, 0.6]).astype(complex)
    assert is_density_matrix(rho)


def test_is_density_matrix_false_non_unit_trace():
    rho = np.diag([0.4, 0.4]).astype(complex)
    assert not is_density_matrix(rho)


def test_is_density_matrix_false_negative():
    rho = np.diag([-0.1, 1.1]).astype(complex)
    assert not is_density_matrix(rho)


def test_eig_nonhermitian_with_left_vectors():
    A = np.diag([1.0, 2.0, 3.0]).astype(complex)
    decomp = eig_nonhermitian(A, compute_left=True)
    assert decomp.left_vectors is not None
    assert decomp.size == 3


def test_eig_nonhermitian_rejects_non_square():
    import pytest
    with pytest.raises(ValueError):
        eig_nonhermitian(np.zeros((2, 3)))


def test_matrix_2_norm():
    A = np.diag([1.0, 2.0, 3.0])
    assert abs(matrix_2_norm(A) - 3.0) < 1e-12


def test_resolvent_apply_superlu_large(rng):
    n = 64
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    z = 10.0 + 0.0j
    x = resolvent_apply_superlu(A, z, b)
    res = (z * np.eye(n, dtype=complex) - A) @ x - b
    assert np.linalg.norm(res) < 1e-8


def test_resolvent_norm_small():
    A = np.diag([1.0, 2.0]).astype(complex)
    # ||(zI - A)^{-1}|| with z = 0: max 1/|z - lambda| = max 1/lambda = 1
    norm = resolvent_norm(A, 0.0 + 0.0j)
    assert abs(norm - 1.0) < 0.5  # loose tolerance for power-iteration fallback


def test_resolvent_norm_sparse_path_matches_dense(rng):
    """Exercise the n > 128 sparse path and cross-check against the dense
    SVD on a moderate-size random Liouvillian-like matrix."""
    n = 160
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    # Make the matrix safely invertible at z=5 by shifting eigenvalues away.
    A -= 10.0 * np.eye(n, dtype=complex)
    z = 5.0 + 0.0j
    sparse_norm = resolvent_norm(A, z)
    # Dense reference
    dense_inv = np.linalg.solve(z * np.eye(n, dtype=complex) - A, np.eye(n, dtype=complex))
    import scipy.linalg as sla
    dense_norm = float(sla.svdvals(dense_inv)[0])
    rel = abs(sparse_norm - dense_norm) / max(dense_norm, 1.0e-12)
    assert rel < 5.0e-2, f"sparse={sparse_norm}, dense={dense_norm}"
