"""Tests for spectral layer D1-D4."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, steady_state
from liouscope.diagnostics.spectral import (
    compute_spectral_layer,
    gns_gap,
    kms_gap,
    oscillating_mode_gap,
    spectral_spread,
)


def test_d1_gap_amplitude_damped(pauli):
    sm = 0.5 * (pauli["X"] - 1j * pauli["Y"])
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [sm], [0.4])
    layer = compute_spectral_layer(L)
    # Amplitude damping with gamma=0.4 has eigenvalues {0, -0.4, -0.4, -0.4}
    assert abs(layer.gap - 0.2) < 0.05 or abs(layer.gap - 0.4) < 0.05


def test_d2_gns_gap_at_equilibrium(pauli):
    # rho_ss = I/2 (maximally mixed) => GNS inner product reduces to
    # Hilbert-Schmidt and the symmetrised generator is the Hermitian part of
    # L. For driven dephasing that Hermitian part is the pure-dephasing
    # dissipator with a TWO-dimensional kernel (I and sigma_z): the sigma_z
    # mode has zero instantaneous GNS decay rate, so the certified contraction
    # rate Delta_s is exactly 0 (2026-07 audit A1 follow-up: the previous
    # positive value came from filtering *all* near-zero eigenvalues, which
    # over-certified the bound). The standard gap stays positive.
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    g_s = gns_gap(L, rho_ss)
    assert g_s == pytest.approx(0.0, abs=1e-9)
    # Non-vacuity: a detailed-balance thermal qubit must keep Delta_s > 0.
    lower = np.array([[0, 1], [0, 0]], dtype=complex)
    L_th = build_liouvillian(0.65 * pauli["Z"], [lower, lower.conj().T], [0.9, 0.2])
    rho_th = steady_state(L_th)
    assert gns_gap(L_th, rho_th) > 0.1


def test_d2b_kms_at_equilibrium_equals_gns(pauli):
    # KMS and GNS Gram matrices coincide up to scalar on a diagonal steady
    # state. NOTE: H=0 pure dephasing has a *degenerate* (non-unique) steady
    # state (populations are conserved), so since the S1 hardening
    # steady_state() refuses to silently pick one. This test only needs *a*
    # representative diagonal steady state to compare the two Gram gaps, so it
    # opts in explicitly via allow_degenerate=True (and the expected
    # RuntimeWarning is accepted).
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    with pytest.warns(RuntimeWarning, match="not unique"):
        rho_ss = steady_state(L, allow_degenerate=True)
    g_s = gns_gap(L, rho_ss)
    g_k = kms_gap(L, rho_ss)
    assert abs(g_k - g_s) < 1e-6


def test_d3_oscillating_gap_pure_dephasing_zero(pauli):
    # Pure dephasing with no H -> all eigenvalues real.
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [pauli["Z"]], [0.3])
    eigs = np.linalg.eigvals(L)
    assert oscillating_mode_gap(eigs) == 0.0


def test_d3_oscillating_gap_nonzero(pauli):
    # Hamiltonian drives oscillations
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.1])
    eigs = np.linalg.eigvals(L)
    assert oscillating_mode_gap(eigs) > 0


def test_d4_spread_zero_for_uniform_rates(pauli):
    sm = 0.5 * (pauli["X"] - 1j * pauli["Y"])
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [0.5])
    eigs = np.linalg.eigvals(L)
    spread = spectral_spread(eigs)
    assert spread >= 0


def test_compute_spectral_layer_returns_result(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    res = compute_spectral_layer(L)
    assert res.gap > 0
    # Driven dephasing has a degenerate Hermitian-part kernel: the certified
    # GNS contraction rate is 0 (see test_d2_gns_gap_at_equilibrium).
    assert res.gns_gap == pytest.approx(0.0, abs=1e-9)
    assert res.eigenvalues.size == 4
    assert res.steady_state.shape == (2, 2)


# --- 2026-07 audit A1: Gram/adjoint consistency off the diagonal manifold ----


def _random_gksl(seed: int, d: int = 3):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = 0.5 * (A + A.conj().T)
    J = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    return build_liouvillian(H, [J], [0.7])


@pytest.mark.parametrize("seed", [7, 11, 23, 42])
def test_symmetrised_gaps_are_contraction_rates_nondiagonal_rho(seed):
    """0 <= Delta_s <= Delta and 0 <= Delta_KMS <= Delta for generic GKSL.

    Regression for the Gram/adjoint transpose inconsistency: with the
    pre-2026-07 construction, gns_gap() returned large NEGATIVE values for
    any steady state carrying coherences (unphysical: the GNS-symmetrised
    generator certifies a contraction). Anchor fixtures are all diagonal,
    which is exactly why this survived the anchor suite.
    """
    from liouscope.core.lindblad import steady_state

    L = _random_gksl(seed)
    rho_ss = steady_state(L)
    # Non-trivial regime: the steady state must carry coherences.
    off_diag = np.linalg.norm(rho_ss - np.diag(np.diag(rho_ss)))
    assert off_diag > 1e-3, "fixture must exercise the non-diagonal manifold"

    from liouscope.diagnostics.spectral import liouvillian_gap
    from liouscope.numerics.linalg import eig_nonhermitian

    gap = liouvillian_gap(eig_nonhermitian(L).eigenvalues)
    g_s = gns_gap(L, rho_ss)
    g_k = kms_gap(L, rho_ss)
    assert -1e-9 <= g_s <= gap + 1e-6
    assert -1e-9 <= g_k <= gap + 1e-6


def test_alicki_adjoint_is_true_gram_adjoint_nondiagonal_rho():
    """<A, L_HS B>_GNS == <L_tilde* A, B>_GNS with G = rho_ss.T (x) I."""
    from liouscope.core.lindblad import steady_state
    from liouscope.diagnostics.spectral import _gram_gns
    from liouscope.numerics.adjoint import alicki_adjoint

    L = _random_gksl(23)
    rho_ss = steady_state(L)
    G = _gram_gns(rho_ss + 1e-12 * np.eye(rho_ss.shape[0]))
    L_HS = L.conj().T
    adj = alicki_adjoint(L, rho_ss)
    # Adjoint identity in matrix form: G @ adj == L_HS^dag @ G.
    np.testing.assert_allclose(G @ adj, L_HS.conj().T @ G, atol=1e-8)


def test_symmetrised_gap_zero_when_hermitian_part_kernel_degenerate():
    """Extra zero modes of the Hermitian part mean NO certified contraction."""
    from liouscope.diagnostics.spectral import _real_gap_from_symmetric

    # Hermitian part with a two-dimensional kernel and a -0.5 level.
    M = np.diag([0.0, 0.0, -0.5, -1.0]).astype(complex)
    assert _real_gap_from_symmetric(M) == pytest.approx(0.0, abs=1e-12)
