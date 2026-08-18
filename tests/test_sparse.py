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


def _site_ops(op: np.ndarray, n_sites: int = 3) -> list[np.ndarray]:
    ops = []
    for i in range(n_sites):
        op_list = [np.eye(2, dtype=complex)] * n_sites
        op_list[i] = op
        full = op_list[0]
        for o in op_list[1:]:
            full = np.kron(full, o)
        ops.append(full)
    return ops


def test_sparse_steady_state(pauli):
    # Need a system with > 4x4 superoperator for ARPACK; build a 3-qubit chain
    # with amplitude damping on every site -> UNIQUE steady state |000><000|.
    sm = 0.5 * (pauli["X"] + 1j * pauli["Y"])  # |0><1| lowers to ground.
    sz_chain = _site_ops(pauli["Z"])
    sm_chain = _site_ops(sm)
    H = sum(0.3 * sz_chain[i] @ sz_chain[(i + 1) % 3] for i in range(3))
    L_sparse = build_sparse_liouvillian(H, sm_chain, [0.1] * 3)
    rho_ss = sparse_steady_state(L_sparse, tol=1e-6)
    tr = float(np.real(np.trace(rho_ss)))
    assert abs(tr - 1.0) < 1e-5
    expected = np.zeros((8, 8), dtype=complex)
    expected[0, 0] = 1.0
    np.testing.assert_allclose(rho_ss, expected, atol=1e-5)


def test_sparse_steady_state_degenerate_fails_closed(pauli):
    """FAILS-BEFORE (sparse/dense divergence): pure dephasing conserves the
    populations, so the null space is multi-dimensional and the pre-guard
    sparse path silently returned an arbitrary, not-even-PSD representative
    while the dense path raised DegenerateSteadyStateError."""
    from liouscope.core.lindblad import DegenerateSteadyStateError

    sz_chain = _site_ops(pauli["Z"])
    H = sum(0.3 * sz_chain[i] @ sz_chain[(i + 1) % 3] for i in range(3))
    L_sparse = build_sparse_liouvillian(H, sz_chain, [0.1] * 3)
    with pytest.raises(DegenerateSteadyStateError):
        sparse_steady_state(L_sparse, tol=1e-6)
    with pytest.warns(RuntimeWarning, match="not unique"):
        rho = sparse_steady_state(L_sparse, tol=1e-6, allow_degenerate=True)
    assert abs(float(np.real(np.trace(rho))) - 1.0) < 1e-5


def test_sparse_steady_state_diagnosis_invariant_under_rate_rescaling(pauli):
    """FAILS-BEFORE (#97 item 5, sparse parity): absolute guard in rate units.

    The old guard threshold ``max(tol, |sigma_shift|)`` and the fixed shift
    ``sigma_shift=1e-8`` were absolute in the caller's rate units, so a pure
    change of units ``L -> c L`` flipped the uniqueness diagnosis. The
    scale-relative guard must resolve |000><000| identically at every scale,
    and genuine degeneracy must survive rescaling.
    """
    from liouscope.core.lindblad import DegenerateSteadyStateError

    sm = 0.5 * (pauli["X"] + 1j * pauli["Y"])
    sz_chain = _site_ops(pauli["Z"])
    sm_chain = _site_ops(sm)
    H = sum(0.3 * sz_chain[i] @ sz_chain[(i + 1) % 3] for i in range(3))
    L_unique = build_sparse_liouvillian(H, sm_chain, [0.1] * 3)
    L_degen = build_sparse_liouvillian(H, sz_chain, [0.1] * 3)
    expected = np.zeros((8, 8), dtype=complex)
    expected[0, 0] = 1.0
    for c in (1e-10, 1.0, 1e10):
        rho_ss = sparse_steady_state(c * L_unique, tol=1e-6)
        np.testing.assert_allclose(rho_ss, expected, atol=1e-5, err_msg=f"c={c}")
        with pytest.raises(DegenerateSteadyStateError):
            sparse_steady_state(c * L_degen, tol=1e-6)


def test_sparse_builder_rejects_non_hermitian_h():
    """FAILS-BEFORE: the sparse builder skipped the dense builder's physics
    gates (Hermiticity, rate sign/finiteness, jump shapes)."""
    bad_h = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="Hermitian"):
        build_sparse_liouvillian(bad_h)


def test_sparse_builder_rejects_negative_and_nonfinite_rates(pauli):
    with pytest.raises(ValueError, match="non-negative"):
        build_sparse_liouvillian(pauli["Z"], [pauli["Z"]], [-1.0])
    with pytest.raises(ValueError, match="finite"):
        build_sparse_liouvillian(pauli["Z"], [pauli["Z"]], [float("nan")])


def test_sparse_builder_rejects_misshaped_jump_op(pauli):
    with pytest.raises(ValueError, match="jump_op shape"):
        build_sparse_liouvillian(pauli["Z"], [np.eye(3, dtype=complex)], [0.1])


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


def test_sparse_hermiticity_gate_is_invariant_to_real_energy_offsets():
    """Parity with the dense builder (twelfth-round review)."""
    import scipy.sparse as _sp

    bad = _sp.csr_matrix(
        np.array([[0.0, 1.0e-3], [0.0, 0.0]], dtype=complex)
    )
    sm = _sp.csr_matrix(np.array([[0, 1], [0, 0]], dtype=complex))
    eye = _sp.identity(2, dtype=complex, format="csr")

    with pytest.raises(ValueError, match=r"[Hh]ermitian"):
        build_sparse_liouvillian(bad, [sm], [1.0])
    with pytest.raises(ValueError, match=r"[Hh]ermitian"):
        build_sparse_liouvillian(bad + 1.0e9 * eye, [sm], [1.0])
