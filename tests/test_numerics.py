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


def test_resolvent_apply_superlu_sparse_branch(rng):
    # n > 256 exercises the SuperLU sparse solve path (the n=64 test above
    # falls through to the dense scipy.linalg.solve branch).
    n = 300
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    z = 5.0 + 0.0j
    x = resolvent_apply_superlu(A, z, b)
    res = (z * np.eye(n, dtype=complex) - A) @ x - b
    assert np.linalg.norm(res) < 1e-8


def test_resolvent_norm_large_matches_dense(rng):
    # n > 128 exercises the SuperLU power-iteration branch. Regression guard for
    # the conjugate-transpose bug (lu.solve(y.conj()).conj() computes
    # conj(A)^{-1} y, not the required (A^H)^{-1} y) that made the largest
    # singular value of the resolvent wrong by ~50% on non-symmetric A.
    import scipy.linalg as sla

    n = 160
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
    z = 3.0 + 0.5j
    ref = float(sla.svdvals(sla.inv(z * np.eye(n, dtype=complex) - A))[0])
    got = resolvent_norm(A, z)
    assert abs(got - ref) <= 1e-6 * ref


def test_resolvent_norm_large_clustered_singular_values():
    # Robustness guard: two eigenvalues at nearly-equal distance from z make the
    # top two singular values of the resolvent cluster -- the regime where the
    # power method's *eigenvector* convergence stalls. The *value* (the norm)
    # must still be accurate, which is why plain power iteration is sufficient
    # here and Lanczos/svds is not needed (see resolvent_norm docstring).
    import scipy.linalg as sla

    n = 160
    z = 0.0 + 0.0j
    lam = np.empty(n, dtype=complex)
    lam[0] = 1.0
    lam[1] = 1.001  # sigma_2/sigma_1 ~ 0.999 -> tightly clustered top pair
    lam[2:] = np.linspace(5.0, 20.0, n - 2)  # far away -> negligible sigmas
    A = np.diag(lam)
    ref = float(sla.svdvals(sla.inv(z * np.eye(n, dtype=complex) - A))[0])
    got = resolvent_norm(A, z)
    assert abs(got - ref) <= 1e-3 * ref
