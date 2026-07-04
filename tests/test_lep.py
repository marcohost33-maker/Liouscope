"""Tests for LEP layer D16-D18."""

from __future__ import annotations

import numpy as np

from liouscope import build_liouvillian, steady_state
from liouscope.diagnostics.lep import (
    compute_lep_layer,
    gap_rate_consistency,
    initial_state_sensitivity,
    lep_proximity,
)
from liouscope.numerics.linalg import eig_nonhermitian


def test_lep_proximity_includes_complex_pairs(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.1])
    eigs = eig_nonhermitian(L).eigenvalues
    prox, count = lep_proximity(eigs)
    assert np.isfinite(prox)
    assert count >= 1


def test_gap_rate_consistency():
    # D17 takes a LINEAR-metric rate (LIOU-#69); param renamed beta_D -> rate.
    assert gap_rate_consistency(rate=0.5, gap=0.5) == 0
    assert np.isinf(gap_rate_consistency(rate=0.5, gap=0.0))
    assert np.isinf(gap_rate_consistency(rate=float("nan"), gap=0.5))


def test_initial_state_sensitivity_smoke(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    sens = initial_state_sensitivity(L, rho_ss, n_samples=4, seed=0)
    assert sens >= 0


def test_compute_lep_layer_returns_result(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    eigs = eig_nonhermitian(L).eigenvalues
    res = compute_lep_layer(L, eigs, beta_D_linear=0.6, gap=0.3, n_haar=4)
    assert np.isfinite(res.gap_rate_consistency)
    assert res.gap_rate_consistency == abs(0.6 - 0.3) / 0.3
    assert res.beta_D_linear == 0.6
    assert res.lep_candidate_count >= 0
