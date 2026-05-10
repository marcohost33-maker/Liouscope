"""Tests for the Liouvillian builder and steady-state extraction."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import build_liouvillian, steady_state


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
