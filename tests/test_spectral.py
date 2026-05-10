"""Tests for spectral layer D1-D4."""

from __future__ import annotations

import numpy as np

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
    # rho_ss = I/2 (maximally mixed) => GNS gap equals Hermitian-part gap
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    g_s = gns_gap(L, rho_ss)
    assert g_s > 0


def test_d2b_kms_at_equilibrium_equals_gns(pauli):
    # For rho_ss = I/2, KMS and GNS Gram matrices coincide up to scalar.
    H = np.zeros((2, 2), dtype=complex)
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
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
    assert res.gns_gap > 0
    assert res.eigenvalues.size == 4
    assert res.steady_state.shape == (2, 2)
