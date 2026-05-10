"""Regression gate for the fourteen LiouScope correctness anchors A-N.

Every anchor here corresponds to a documented historical bug or audit
finding. Any failure must block release.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import qutip_required

from liouscope import (
    DIAGNOSTIC_SCHEMA_VERSION,
    TAXONOMY_VERSION,
    build_liouvillian,
    diagnose,
    steady_state,
)
from liouscope.diagnostics.spectral import _gram_gns
from liouscope.fitting.aicc import aicc
from liouscope.fitting.neff import estimate_neff_geyer
from liouscope.numerics.adjoint import alicki_adjoint, hs_adjoint
from liouscope.numerics.kronecker import unvec, vec
from liouscope.numerics.linalg import eig_nonhermitian, support_check


# Anchor A: Column-stacking (order='F') everywhere.
def test_anchor_A_column_stacking_round_trip(rng):
    rho = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    rho = (rho + rho.conj().T) / 2
    rho_vec = vec(rho)
    rho_back = unvec(rho_vec)
    np.testing.assert_allclose(rho_back, rho, atol=1e-12)


def test_anchor_A_kronecker_identity(rng):
    A = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    B = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    C = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
    # vec(A B C) = (C.T (x) A) vec(B)
    lhs = vec(A @ B @ C)
    rhs = np.kron(C.T, A) @ vec(B)
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_anchor_A_build_liouvillian_rejects_row_stacking():
    with pytest.raises(ValueError, match="column-stacking"):
        build_liouvillian(np.eye(2, dtype=complex), [], None, order="C")  # type: ignore[arg-type]


# Anchor B: GNS Gram is rho_ss.T (x) I, NOT KMS form.
def test_anchor_B_gns_gram_form():
    rho = np.diag([0.6, 0.3, 0.1]).astype(complex)
    G = _gram_gns(rho)
    # KMS gram would have rho^{1/2} on both sides.
    G_expected = np.kron(rho.T, np.eye(3, dtype=complex))
    np.testing.assert_allclose(G, G_expected, atol=1e-12)


# Anchor C: Alicki adjoint direction.
def test_anchor_C_alicki_direction_nontrivial_rho():
    # Use a non-degenerate thermal-style state so that rho != rho^{-1}.
    rho_ss = np.diag([0.7, 0.3]).astype(complex)
    # Construct an arbitrary L of compatible shape.
    rng = np.random.default_rng(0)
    L = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    L_adj = alicki_adjoint(L, rho_ss)
    rho_reg = rho_ss + 1e-12 * np.eye(2, dtype=complex)
    rho_inv = np.linalg.inv(rho_reg)
    eye_d = np.eye(2, dtype=complex)
    expected_forward = np.kron(rho_inv, eye_d) @ L @ np.kron(rho_reg, eye_d)
    expected_reversed = np.kron(rho_reg, eye_d) @ L @ np.kron(rho_inv, eye_d)
    np.testing.assert_allclose(L_adj, expected_forward, atol=1e-10)
    assert not np.allclose(L_adj, expected_reversed, atol=1e-6), (
        "Forward and reversed Alicki adjoint must differ for non-trivial rho_ss; "
        "this is the T17 hardening regression."
    )


def test_anchor_C_alicki_at_equilibrium_pure_dephasing(pauli):
    # Sanity: on a system with rho_ss = I/d (maximally mixed), both directions
    # collapse to L itself. This is the explicit "near-equilibrium" check.
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    L_adj = alicki_adjoint(L, rho_ss)
    # rho_ss should be I/2 here.
    np.testing.assert_allclose(rho_ss, 0.5 * np.eye(2, dtype=complex), atol=1e-9)
    # For rho_ss = I/2, alicki_adjoint reduces to identity-conjugation -> L_adj == L.
    np.testing.assert_allclose(L_adj, L, atol=1e-9)


# Anchor D: zgeev for non-Hermitian Liouvillian.
def test_anchor_D_eigensolver_handles_non_hermitian():
    # Construct a clearly non-Hermitian matrix
    A = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=complex)
    decomp = eig_nonhermitian(A)
    eigs = sorted(np.real(decomp.eigenvalues))
    assert eigs == pytest.approx([1.0, 3.0])


# Anchor E: SuperLU-based resolvent (implicit through numerics.resolvent
# tests below). Smoke-test that the dense path works on a 16x16 matrix.
def test_anchor_E_resolvent_solve(rng):
    from liouscope.numerics.resolvent import resolvent_apply_superlu

    A = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    b = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    z = 5.0 + 0.0j
    x = resolvent_apply_superlu(A, z, b)
    res = (z * np.eye(16, dtype=complex) - A) @ x - b
    assert np.linalg.norm(res) < 1e-9


# Anchor F: AICc-only model comparison. Smoke: aicc handles non-nested models.
def test_anchor_F_aicc_handles_non_nested():
    # k=2 (M0), k=3 (M1), k=4 (M2): AICc should be well-defined for each.
    val_m0 = aicc(-50.0, 2, 80.0)
    val_m1 = aicc(-49.5, 3, 80.0)
    val_m2 = aicc(-48.0, 4, 80.0)
    assert all(np.isfinite([val_m0, val_m1, val_m2]))


# Anchor G: parametric bootstrap (smoke test).
def test_anchor_G_parametric_bootstrap_runs():
    from liouscope.fitting.bootstrap import parametric_bootstrap
    from liouscope.fitting.models import M0

    rng = np.random.default_rng(0)
    t = np.linspace(0, 5, 60)
    y = 1.5 * np.exp(-0.6 * t) + 0.02 * rng.standard_normal(t.size)
    samples, theta_hat = parametric_bootstrap(M0, t, y, np.array([1.0, 0.5]), B=20)
    assert samples.shape == (20, 2)
    assert theta_hat.shape == (2,)


# Anchor H: N_eff via Geyer 1992 IPS.
def test_anchor_H_neff_below_n_for_correlated(rng):
    # Generate AR(1) with rho=0.9
    n = 200
    eps = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.9 * x[i - 1] + eps[i]
    n_eff = estimate_neff_geyer(x)
    assert n_eff < n / 2, f"N_eff={n_eff} not significantly below n={n}"


# Anchor I: conjugate-pair LEP candidates are NOT excluded.
def test_anchor_I_conjugate_pairs_included(pauli):
    # Construct an L with a clear complex pair (oscillatory).
    H = pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.1])
    eigs = eig_nonhermitian(L).eigenvalues
    has_complex = bool(np.any(np.abs(np.imag(eigs)) > 1e-6))
    assert has_complex, "Expected at least one complex eigenpair"
    from liouscope.diagnostics.lep import lep_proximity

    prox, count = lep_proximity(eigs)
    # Proximity reflects ALL pairs including complex conjugates.
    assert np.isfinite(prox)
    assert count >= 1


# Anchor J: supp(rho_0) subset supp(rho_ss) check with eps regularisation.
def test_anchor_J_support_check_flags_outside_support():
    # Rank-deficient steady state
    rho_ss = np.diag([1.0, 0.0]).astype(complex)
    rho_initial = np.diag([0.5, 0.5]).astype(complex)
    ok, reg = support_check(rho_initial, rho_ss, eps=1e-12)
    # rho_initial has half its weight in the null space -> not contained.
    assert ok is False
    # Regularised matrix is positive definite
    evals = np.linalg.eigvalsh(reg)
    assert evals.min() > 0


# Anchor K: HS adjoint != pi-weighted adjoint.
def test_anchor_K_hs_and_alicki_distinct(pauli):
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    L_hs = hs_adjoint(L)
    L_alicki = alicki_adjoint(L, rho_ss)
    assert not np.allclose(L_hs, L_alicki, atol=1e-6)


# Anchor L: TAXONOMY_VERSION stamped on every ClassificationResult.
def test_anchor_L_taxonomy_version_on_result(pauli):
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rng = np.random.default_rng(0)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert report.classification.taxonomy_version == TAXONOMY_VERSION == "A1-A12-v3.1"
    assert report.classification.schema_version == DIAGNOSTIC_SCHEMA_VERSION


# Anchor M: D11 = Bohr-AP-check, D11b = resolvent peak (different modules).
def test_anchor_M_d11_is_bohr_ap():
    from liouscope.diagnostics.nonnormality import bohr_arithmetic_progression
    from liouscope.diagnostics.resolvent import resolvent_peak

    # Just verify they're distinct functions returning distinct types.
    eigs = np.array([0.0, -0.5, -0.5 + 1j, -0.5 - 1j, -1.0 + 2j, -1.0 - 2j])
    length, bound = bohr_arithmetic_progression(eigs, d=2)
    assert isinstance(length, int)
    assert length >= 1
    # Resolvent peak is on a Liouvillian (shape d^2 x d^2)
    L = np.diag([0.0, -0.5, -0.5 + 1j, -0.5 - 1j]).astype(complex)
    peak = resolvent_peak(L)
    assert peak > 0


# Anchor N: D24 = Zhou predictor (not Lee-Bound).
def test_anchor_N_zhou_is_separate_module(pauli):
    from liouscope import _zhou

    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    res = _zhou.compute_zhou_predictor(L, epsilon=1e-3)
    assert res.converged
    assert res.mixing_time_upper >= res.mixing_time_lower > 0


# QuTiP cross-check supplements anchor A.
@qutip_required
def test_qutip_cross_check_dephased_qubit(pauli):
    import qutip

    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    H_qt = qutip.Qobj(H)
    c_ops = [np.sqrt(0.3) * qutip.Qobj(pauli["Z"])]
    L_qt = qutip.liouvillian(H_qt, c_ops).full()
    np.testing.assert_allclose(L, L_qt, atol=1e-10)


@qutip_required
def test_qutip_cross_check_qutrit():
    import qutip

    H = np.diag([0.0, 1.0, 2.5]).astype(complex)
    op = np.zeros((3, 3), dtype=complex)
    op[1, 0] = 1.0
    L = build_liouvillian(H, [op], [0.3])
    H_qt = qutip.Qobj(H)
    c_ops = [np.sqrt(0.3) * qutip.Qobj(op)]
    L_qt = qutip.liouvillian(H_qt, c_ops).full()
    np.testing.assert_allclose(L, L_qt, atol=1e-10)
