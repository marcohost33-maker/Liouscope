"""Tests for the Liouvillian builder and steady-state extraction."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import build_liouvillian, steady_state
from liouscope.core.lindblad import DegenerateSteadyStateError


def test_build_liouvillian_no_jumps_purely_unitary(pauli):
    """With no jumps, L generates pure unitary evolution: tr-preserving."""
    H = pauli["Z"]
    L = build_liouvillian(H, [])
    # For unitary dynamics, L vec(rho) should preserve trace: sum of diagonal vec entries
    rho_vec = np.eye(2, dtype=complex).flatten(order="F")
    drho = L @ rho_vec
    trace_change = drho[0] + drho[3]
    assert abs(trace_change) < 1e-10


def test_build_liouvillian_trace_preserving(pauli):
    """For an arbitrary GKSL, the propagator should preserve trace."""
    import scipy.linalg as sla

    H = 0.3 * pauli["X"] + 0.5 * pauli["Z"]
    L = build_liouvillian(H, [pauli["Z"], pauli["X"]], [0.2, 0.1])
    rho_vec = (0.5 * np.eye(2, dtype=complex)).flatten(order="F")
    drho = L @ rho_vec
    trace_dot = drho[0] + drho[3]
    assert abs(trace_dot) < 1e-10
    expL = sla.expm(L * 0.7)
    rho_t = expL @ rho_vec
    assert abs(rho_t[0] + rho_t[3] - 1.0) < 1e-10


def test_build_liouvillian_rejects_non_square():
    with pytest.raises(ValueError):
        build_liouvillian(np.zeros((2, 3), dtype=complex), [])


def test_build_liouvillian_rejects_non_hermitian_H():
    H = np.array([[0, 1], [0, 0]], dtype=complex)
    with pytest.raises(ValueError):
        build_liouvillian(H, [])


def test_build_liouvillian_rejects_mismatched_jumps(pauli):
    H = pauli["Z"]
    with pytest.raises(ValueError):
        build_liouvillian(H, [np.eye(3, dtype=complex)])


def test_build_liouvillian_negative_rate_rejected(pauli):
    H = pauli["Z"]
    with pytest.raises(ValueError):
        build_liouvillian(H, [pauli["Z"]], [-0.5])


def test_build_liouvillian_rejects_non_fortran_order(pauli):
    # The column-stacking (order='F') guard on the dense builder (anchor A).
    with pytest.raises(ValueError, match="order='F'"):
        build_liouvillian(pauli["Z"], [], order="C")  # type: ignore[arg-type]


def test_build_liouvillian_jump_ops_none_equals_empty(pauli):
    # The default jump_ops=None must behave like an explicit empty list.
    H = pauli["Z"]
    np.testing.assert_allclose(build_liouvillian(H), build_liouvillian(H, []))


def test_build_liouvillian_zero_rate_jump_skipped(pauli):
    # A gamma == 0.0 jump contributes nothing -> identical to coherent-only.
    H = pauli["Z"]
    coherent = build_liouvillian(H, [])
    with_zero = build_liouvillian(H, [pauli["Z"]], [0.0])
    np.testing.assert_allclose(with_zero, coherent, atol=1e-12)


def test_build_liouvillian_rate_length_mismatch_raises(pauli):
    # Correct-shape jump but wrong number of rates exercises the rate-length
    # guard, which is distinct from the jump-operator shape guard.
    with pytest.raises(ValueError, match=r"len\(rates\)"):
        build_liouvillian(pauli["Z"], [pauli["Z"]], [0.1, 0.2])


def test_steady_state_rejects_non_square_superoperator():
    # n2 must be a perfect square (d^2); 3 is not -> ValueError.
    with pytest.raises(ValueError, match="square-d"):
        steady_state(np.eye(3, dtype=complex))


def test_steady_state_full_rank_uses_smallest_singular_vector():
    # A full-rank superoperator has no exact null space, so steady_state falls
    # back to the smallest-singular-value direction (vh[:, -1]).
    # diag(1,2,3,4): smallest singular value is at index 0 -> rho = |0><0|.
    # Hardening: the fallback is no longer silent -- the result is NOT a
    # verified steady state (residual ||L rho|| = 1 here), so a RuntimeWarning
    # carrying the residual is mandatory.
    L = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    with pytest.warns(RuntimeWarning, match="NOT a verified steady state"):
        rho = steady_state(L)
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    np.testing.assert_allclose(rho, expected, atol=1e-9)


def test_steady_state_traceless_nullspace_raises_runtime():
    # Unphysical superoperator whose unique null vector unvecs to a traceless
    # matrix (vec of Pauli Z). The SVD candidate and the eig fallback are both
    # trace-0, so the normaliser must fail loudly rather than divide by ~0.
    v = np.array([1, 0, 0, -1], dtype=complex)  # vec(Z), Tr = 0
    proj = np.outer(v, v.conj()) / np.vdot(v, v)
    L = np.eye(4, dtype=complex) - proj  # null space = span{v}, dim 1
    with pytest.raises(RuntimeError, match="Cannot normalise"):
        steady_state(L)


def test_steady_state_dephased_qubit(pauli):
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    np.testing.assert_allclose(rho_ss, 0.5 * np.eye(2, dtype=complex), atol=1e-9)


def test_steady_state_amplitude_damped(pauli):
    sm = 0.5 * (pauli["X"] + 1j * pauli["Y"])  # |0><1| lowers to ground.
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [sm], [0.4])
    rho_ss = steady_state(L)
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    np.testing.assert_allclose(rho_ss, expected, atol=1e-9)


def test_steady_state_degenerate_nullspace_raises(pauli):
    """FAILS-BEFORE (S1): a degenerate NESS was returned silently.

    Pure dephasing with ``H = 0`` conserves populations, so *every* diagonal
    state is stationary: the Liouvillian has a 2-dimensional null space and
    ``rho_ss`` is not unique. The pre-hardening code silently returned an
    arbitrary representative (|0><0|) with no warning. The guarded version
    must fail closed.
    """
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [pauli["Z"]], [0.5])
    # Confirm the null space really is degenerate (two zero singular values).
    s = np.linalg.svd(L, compute_uv=False)
    tol = max(1e-9, L.shape[0] * np.finfo(L.dtype).eps * s[0])
    assert int(np.sum(s <= tol)) == 2, "fixture must have a 2D null space"
    with pytest.raises(DegenerateSteadyStateError) as excinfo:
        steady_state(L)
    assert excinfo.value.null_dim == 2


def test_steady_state_degenerate_allow_warns_and_returns(pauli):
    """allow_degenerate=True returns one representative with a RuntimeWarning."""
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [pauli["Z"]], [0.5])
    with pytest.warns(RuntimeWarning, match="not unique"):
        rho = steady_state(L, allow_degenerate=True)
    # Whatever representative is chosen, it is a valid (Hermitian, unit-trace)
    # density matrix in the steady-state manifold.
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-9)


def test_steady_state_unique_ness_unaffected(pauli):
    """Regression: a unique NESS still resolves silently (no false positive)."""
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    np.testing.assert_allclose(rho_ss, 0.5 * np.eye(2, dtype=complex), atol=1e-9)


@qutip_required
def test_build_liouvillian_matches_qutip_on_three_jumps(pauli):
    import qutip

    H = 0.2 * pauli["X"] + 0.3 * pauli["Z"]
    jumps = [pauli["Z"], pauli["X"], 0.5 * (pauli["X"] - 1j * pauli["Y"])]
    rates = [0.1, 0.15, 0.2]
    L = build_liouvillian(H, jumps, rates)
    c_ops = [np.sqrt(g) * qutip.Qobj(j) for g, j in zip(rates, jumps)]
    L_qt = qutip.liouvillian(qutip.Qobj(H), c_ops).full()
    np.testing.assert_allclose(L, L_qt, atol=1e-10)
