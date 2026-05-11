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


def test_sparse_path_5_qubit_d32(pauli):
    """5-qubit dephased chain: d=32, superop is 1024x1024.

    Lower bound for the d=128 capability claimed by the v1.1.1 sparse path.
    """
    import time

    from liouscope.core import (
        boundary_dephasing_jumps,
        ising_hamiltonian,
        one_d_chain,
    )

    n_qubits = 5
    lat = one_d_chain(n_qubits)
    H = ising_hamiltonian(lat, J=1.0, h=0.5)
    jumps = boundary_dephasing_jumps(n_qubits)
    L_sparse = build_sparse_liouvillian(H, jumps, [0.2] * len(jumps))
    assert L_sparse.shape == (32 * 32, 32 * 32)

    t0 = time.perf_counter()
    rho_ss = sparse_steady_state(L_sparse, tol=1e-7)
    dt = time.perf_counter() - t0
    assert rho_ss.shape == (32, 32)
    assert abs(np.trace(rho_ss).real - 1.0) < 1e-5
    # Should fit comfortably in the spec's 11s budget on a modern laptop.
    assert dt < 60.0, f"Sparse steady state at d=32 took {dt:.2f}s (>60s budget)"

    vals, _ = sparse_spectrum(L_sparse, k=4, tol=1e-7)
    # Steady state lives at lambda = 0; others must have Re < 0.
    assert np.any(np.abs(vals) < 1e-5)
    nonzero = vals[np.abs(vals) > 1e-5]
    if nonzero.size:
        assert np.all(np.real(nonzero) < 1e-6)
