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
import scipy.linalg as sla

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


# --- B1/B2 cross-family hardening regressions --------------------------------


def test_b1_tp_is_checked_at_propagator_not_dt_blind_generator():
    # B1 regression: a trace-scaling generator whose <<I| M_L residual sits just
    # BELOW the old absolute 1e-9 tolerance (so the dt-/scale-blind GENERATOR
    # check reported is_tp=True), while the actual propagator exp(dt*L) scales
    # the trace ~148x. The propagator-level TP check must reject it.
    L_valid = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.4])
    n2 = L_valid.shape[0]
    eps = 5.0e-10
    L_bad = L_valid + eps * np.eye(n2, dtype=complex)
    dt = float(np.log(148.0) / eps)  # exp(eps*dt) = 148

    # The OLD generator residual ||<<I| M_L|| would have passed (< 1e-9):
    gen_resid = float(np.linalg.norm(vec(np.eye(2, dtype=complex)).conj() @ L_bad))
    assert gen_resid < 1.0e-9

    res = cptp_choi_gate(L_bad, dt=dt)
    assert not res.is_tp  # propagator trace scales ~148x => TP violated
    assert res.tp_residual == pytest.approx(147.0 * np.sqrt(2), rel=1e-3)  # ~207.9
    assert not res.is_cptp


def test_b2_non_hermiticity_preserving_map_is_not_cp_even_if_psd_passes():
    # B2 regression: a depolarizing channel (full-rank PSD Choi) plus a small
    # non-Hermiticity-preserving perturbation. The HERMITISED Choi stays PSD
    # (min_eig > 0 -> the old PSD-only check would have certified "CP"), yet the
    # Choi matrix is non-Hermitian, so the map is not Hermiticity-preserving and
    # therefore not CP. Hermitising before the PSD test masked exactly this.
    I2 = np.eye(2, dtype=complex)
    p = 0.6
    S_dep = p * np.eye(4, dtype=complex) + (1.0 - p) * np.outer(
        vec(I2 / 2.0), vec(I2).conj()
    )
    k01 = np.zeros((2, 2), dtype=complex)
    k01[0, 1] = 1.0  # |0><1|
    k00 = np.zeros((2, 2), dtype=complex)
    k00[0, 0] = 1.0  # |0><0|
    M = np.outer(vec(k01), vec(k00).conj())  # non-HP, nilpotent (M^2 = 0)
    phi = S_dep + 0.15 * M
    L = sla.logm(phi)  # exp(1.0 * L) == phi

    res = cptp_choi_gate(L, dt=1.0)
    assert res.min_eig > 0.0  # Hermitised PSD check ALONE would pass
    assert res.choi_herm_residual > 1.0e-3  # but the Choi is non-Hermitian
    assert not res.is_hp
    assert not res.is_cp  # masking closed: non-HP => not CP
    assert res.is_tp  # trace IS preserved (so the old gate said CPTP)
    assert not res.is_cptp


def test_tolerances_are_relative_scale_invariant():
    # A valid GKSL channel stays CPTP after a large uniform rescale of the
    # generator (relative tolerances must not drift with operator scale).
    L = build_liouvillian(0.5 * 1.3 * _Z, [_LOWER, _RAISE], [0.9, 0.2])
    res = cptp_choi_gate(1e6 * L, dt=1e-7)  # scaled generator, same physics
    assert res.is_cptp
    assert res.is_hp


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


def test_cptp_gate_rejects_nonfinite_generator_entries():
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.4])
    L_bad = L.copy()
    L_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match=r"L_super.*finite"):
        cptp_choi_gate(L_bad, dt=0.1)


def test_cptp_gate_rejects_invalid_tolerances():
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.4])
    with pytest.raises(ValueError, match=r"tol_choi.*non-negative"):
        cptp_choi_gate(L, dt=0.1, tol_choi=-1.0)
    with pytest.raises(ValueError, match=r"tol_tp.*finite"):
        cptp_choi_gate(L, dt=0.1, tol_tp=float("nan"))


def test_choi_matrix_rejects_nonfinite_entries():
    channel = np.eye(4, dtype=complex)
    channel[0, 0] = np.inf
    with pytest.raises(ValueError, match=r"channel_super.*finite"):
        choi_matrix(channel)


def _partial_trace_output_leg(J: np.ndarray, d: int) -> np.ndarray:
    """Trace over the output (channel) leg of ``J = sum_ij Phi(E_ij) (x) E_ij``.

    ``choi_matrix`` builds ``J`` with the channel-output factor *first* in the
    Kronecker product, so reshaping to ``(d, d, d, d)`` gives index order
    ``[out_row, in_row, out_col, in_col]``. Summing the output diagonal
    (``aiaj -> ij``) leaves the input-leg reduced operator, which equals the
    identity exactly when the channel is trace preserving.
    """
    return np.einsum("aiaj->ij", J.reshape(d, d, d, d))


def test_identity_channel_choi_partial_trace_is_identity_across_dimensions():
    # The identity channel's Choi matrix is NOT np.eye(d**2); that is the
    # identity *superoperator*. Its Choi matrix is the unnormalised maximally
    # entangled rank-one operator |Omega><Omega|. Build it through choi_matrix
    # and pin both the tensor-leg convention (partial trace over the output leg
    # is the identity) and CP-boundary PSD-ness across several dimensions.
    for d in (2, 4, 8, 16):
        identity_super = np.eye(d * d, dtype=complex)
        J = choi_matrix(identity_super)
        tr_out = _partial_trace_output_leg(J, d)
        atol = max(100.0 * d * np.finfo(float).eps, 1.0e-12)
        np.testing.assert_allclose(tr_out, np.eye(d), atol=atol)
        eigs = np.linalg.eigvalsh(0.5 * (J + J.conj().T))
        assert float(eigs[0]) >= -atol
