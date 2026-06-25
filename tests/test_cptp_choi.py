"""Independent-oracle tests for the CPTP Choi gate (LIOU-A-011).

Oracles are closed-form, not the gate's own machinery:

* The **transpose map** ``T(rho) = rho^T`` is positive but NOT completely
  positive; its Choi matrix is the SWAP operator with eigenvalues +-1, so
  ``min_eig = -1`` exactly. This pins that ``choi_matrix`` actually detects
  non-CP maps (the whole point of the gate).
* A **dephasing channel** ``exp(dt*L)`` is rank-2 in Kraus form, so its Choi
  matrix has a genuine zero eigenvalue: ``min_eig = 0`` (PSD on the boundary) --
  the value claimed in the entry.
* The **Euler step** ``I + dt*L`` is NOT a CP proof: for a suitable ``dt`` it has
  a negative Choi eigenvalue while ``exp(dt*L)`` stays CP -- exactly the
  correctness seam the entry calls out.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.numerics.cptp import choi_matrix, cptp_choi_gate
from liouscope.numerics.kronecker import unvec, vec

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_RAISE = 0.5 * (_X - 1j * _Y)  # |1><0|
_LOWER = _RAISE.conj().T  # |0><1|, amplitude-damping decay channel


def _transpose_superoperator(d: int) -> np.ndarray:
    """Column-stacking superoperator T with T @ vec(rho) = vec(rho^T)."""
    n2 = d * d
    T = np.zeros((n2, n2), dtype=complex)
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d), dtype=complex)
            E[i, j] = 1.0
            T[:, np.where(vec(E) == 1.0)[0][0]] = vec(E.T)
    return T


def test_transpose_map_is_not_cp_choi_min_eig_minus_one():
    # Closed-form oracle: Choi(transpose) = SWAP, eigenvalues {+1 (x3), -1}.
    T = _transpose_superoperator(2)
    # Sanity: T really transposes.
    rho = np.array([[1, 2j], [3, 4]], dtype=complex)
    np.testing.assert_allclose(unvec(T @ vec(rho), d=2), rho.T, atol=1e-12)

    J = choi_matrix(T)
    min_eig = float(np.linalg.eigvalsh(0.5 * (J + J.conj().T))[0])
    assert min_eig == pytest.approx(-1.0, abs=1e-9)  # NOT completely positive


def test_dephasing_channel_choi_min_eig_is_zero():
    # H = 0, jump = sqrt(gamma) Z. exp(dt*L) is a (CP) dephasing channel whose
    # Choi matrix has a genuine zero eigenvalue (rank-2 Kraus) => min_eig = 0.
    gamma = 0.7
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_Z], [gamma])
    res = cptp_choi_gate(L, dt=0.1)
    assert res.is_cptp
    assert res.min_eig == pytest.approx(0.0, abs=1e-9)  # entry beleg: 0.0 PSD
    assert res.tp_residual == pytest.approx(0.0, abs=1e-9)


def test_amplitude_damping_channel_is_cptp_over_time():
    gamma = 0.4
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [gamma])
    for dt in (0.0, 0.05, 0.5, 2.0, 10.0):
        res = cptp_choi_gate(L, dt=dt)
        assert res.is_cp, f"amplitude damping must stay CP at dt={dt}"
        assert res.is_tp
        assert res.min_eig >= -1e-9


def test_generic_dense_gksl_channel_is_cptp():
    # Driven thermal qubit (H != 0, two jumps): exp(dt*L) is CPTP by GKSL
    # construction; the Choi gate must agree (min_eig >= -tol, finite > -inf).
    H = 0.5 * 1.3 * _Z
    L = build_liouvillian(H, [_LOWER, _RAISE], [0.9, 0.2])
    res = cptp_choi_gate(L, dt=0.1)
    assert res.is_cptp
    assert res.min_eig >= -1e-9
    # Not on the CP boundary => a strictly positive minimum eigenvalue.
    assert res.min_eig > 1e-6


def test_euler_step_is_not_a_cp_proof():
    # The entry's NR: rho + dt*L(rho) >= 0 sampling is NOT complete positivity.
    # Demonstrate a dt where the Euler propagator (I + dt*L) has a negative Choi
    # eigenvalue while exp(dt*L) stays CP.
    gamma = 1.0
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [gamma])
    n2 = L.shape[0]
    found_non_cp_euler = False
    for dt in (0.5, 1.0, 1.5, 2.0, 3.0):
        euler = np.eye(n2, dtype=complex) + dt * L
        J_euler = choi_matrix(euler)
        euler_min = float(np.linalg.eigvalsh(0.5 * (J_euler + J_euler.conj().T))[0])
        expm_res = cptp_choi_gate(L, dt=dt)
        # exp(dt*L) is always CP.
        assert expm_res.is_cp
        if euler_min < -1e-6:
            found_non_cp_euler = True
    assert found_non_cp_euler, (
        "expected at least one dt where the Euler step violates complete "
        "positivity (entry NR-002: Euler positivity is not a CP proof)"
    )


# --- negative / edge-input gates (silent-failure gate) -----------------------


def test_cptp_gate_rejects_negative_dt():
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.4])
    with pytest.raises(ValueError, match="non-negative"):
        cptp_choi_gate(L, dt=-0.1)


def test_cptp_gate_rejects_nonfinite_dt():
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.4])
    with pytest.raises(ValueError, match="finite"):
        cptp_choi_gate(L, dt=float("nan"))


def test_cptp_gate_rejects_non_square_dim():
    with pytest.raises(ValueError, match="perfect square"):
        cptp_choi_gate(np.zeros((6, 6), dtype=complex), dt=0.1)


def test_choi_matrix_rejects_non_square_input():
    with pytest.raises(ValueError, match="square"):
        choi_matrix(np.zeros((2, 3), dtype=complex))
