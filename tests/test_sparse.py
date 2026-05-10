"""Tests for the sparse path (ARPACK shift-invert)."""

from __future__ import annotations

import numpy as np

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
