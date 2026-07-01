"""Tests for the sparse path (ARPACK shift-invert)."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.sparse import (
    build_sparse_liouvillian,
    chi1_lower_bound,
    sparse_spectrum,
    sparse_steady_state,
)


def test_sparse_matches_dense(pauli):
    L_dense = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    L_sparse = build_sparse_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    np.testing.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-10)


def test_sparse_rejects_non_fortran_order(pauli):
    with pytest.raises(ValueError, match="order='F'"):
        build_sparse_liouvillian(pauli["X"], order="C")  # type: ignore[arg-type]


def test_sparse_rejects_non_square_hamiltonian():
    with pytest.raises(ValueError, match="square"):
        build_sparse_liouvillian(np.zeros((2, 3), dtype=complex))


def test_sparse_rejects_rate_length_mismatch(pauli):
    with pytest.raises(ValueError, match="len\\(rates\\)"):
        build_sparse_liouvillian(pauli["X"], [pauli["Z"]], [0.1, 0.2])


def test_sparse_defaults_no_jumps_to_coherent_only(pauli):
    # jump_ops=None -> purely coherent generator, matching the dense path.
    L_dense = build_liouvillian(0.5 * pauli["X"], [], [])
    L_sparse = build_sparse_liouvillian(0.5 * pauli["X"])
    np.testing.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-10)


def test_sparse_defaults_rates_to_unity(pauli):
    explicit = build_sparse_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [1.0])
    defaulted = build_sparse_liouvillian(0.5 * pauli["X"], [pauli["Z"]])
    np.testing.assert_allclose(defaulted.toarray(), explicit.toarray(), atol=1e-12)


def test_sparse_zero_rate_jump_is_skipped(pauli):
    # A gamma == 0.0 jump must not contribute -> identical to coherent-only.
    coherent = build_sparse_liouvillian(0.5 * pauli["X"])
    with_zero = build_sparse_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.0])
    np.testing.assert_allclose(with_zero.toarray(), coherent.toarray(), atol=1e-12)


def test_sparse_steady_state(pauli):
    L_sparse = build_sparse_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    # Need a system with > 4x4 superoperator for ARPACK; build a 3-qubit example.
    sz_chain = []
    for i in range(3):
        op_list = [np.eye(2, dtype=complex)] * 3
        op_list[i] = pauli["Z"]
        op = op_list[0]
        for o in op_list[1:]:
            op = np.kron(op, o)
        sz_chain.append(op)
    H = sum(0.3 * sz_chain[i] @ sz_chain[(i + 1) % 3] for i in range(3))
    L_sparse = build_sparse_liouvillian(H, sz_chain, [0.1] * 3)
    rho_ss = sparse_steady_state(L_sparse, tol=1e-6)
    tr = float(np.real(np.trace(rho_ss)))
    assert abs(tr - 1.0) < 1e-5


def test_sparse_spectrum_returns_k_modes(pauli):
    sz_chain = []
    for i in range(3):
        op_list = [np.eye(2, dtype=complex)] * 3
        op_list[i] = pauli["Z"]
        op = op_list[0]
        for o in op_list[1:]:
            op = np.kron(op, o)
        sz_chain.append(op)
    H = sum(0.3 * sz_chain[i] @ sz_chain[(i + 1) % 3] for i in range(3))
    L_sparse = build_sparse_liouvillian(H, sz_chain, [0.1] * 3)
    vals, _ = sparse_spectrum(L_sparse, k=4, tol=1e-6)
    assert vals.size == 4


def test_chi1_lower_bound(pauli):
    sz_chain = []
    for i in range(3):
        op_list = [np.eye(2, dtype=complex)] * 3
        op_list[i] = pauli["Z"]
        op = op_list[0]
        for o in op_list[1:]:
            op = np.kron(op, o)
        sz_chain.append(op)
    H = sum(0.3 * sz_chain[i] @ sz_chain[(i + 1) % 3] for i in range(3))
    L_sparse = build_sparse_liouvillian(H, sz_chain, [0.1] * 3)
    chi = chi1_lower_bound(L_sparse, k_modes=3, tol=1e-6)
    assert chi > 0


def test_chi1_lower_bound_matches_dense_petermann_sqrt(pauli):
    """The certificate must equal max_j sqrt(K_j) over the probed modes.

    Regression for the double-square-root bug: the ratio
    ``||r||*||l|| / |<l,r>|`` IS ``sqrt(K_Petermann)`` already; taking another
    square root silently weakened the bound to ``K^(1/4)``.
    """
    import scipy.sparse as sp

    from liouscope.diagnostics.nonnormality import petermann_factors

    # Non-normal two-qubit system (Liouville dim 16 satisfies ARPACK k < N-1):
    # transverse drive + one-sided amplitude damping.
    lower = np.array([[0, 1], [0, 0]], dtype=complex)
    eye2 = np.eye(2, dtype=complex)
    from liouscope import build_liouvillian

    H = 0.5 * np.kron(pauli["X"], eye2) + 0.2 * np.kron(pauli["Z"], pauli["Z"])
    jumps = [np.kron(lower, eye2), np.kron(eye2, lower)]
    L_dense = build_liouvillian(H, jumps, [0.4, 0.15])
    chi = chi1_lower_bound(sp.csc_matrix(L_dense), k_modes=4, tol=1e-10)

    _, K = petermann_factors(L_dense)
    # chi probes the k_modes slowest modes; it must be a genuine sqrt(K_j)
    # value: bounded above by the global max and >= 1 (K_j >= 1 always).
    assert 1.0 - 1e-9 <= chi <= float(np.sqrt(np.max(K))) + 1e-6
    # And it must be attained by one of the modes (not K^(1/4) of one).
    sqrt_Ks = np.sqrt(K)
    assert np.min(np.abs(sqrt_Ks - chi)) < 1e-6 * max(1.0, chi)
