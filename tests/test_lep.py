"""Tests for LEP layer D16-D18."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_lep_proximity_exact_degeneracy_is_strongest_signal():
    # issue #70 A9: an exactly degenerate pair (the strongest EP signal, two
    # eigenvalues coalescing) must yield proximity 0.0 -- NOT be skipped, and NOT
    # return inf ("maximally far from an EP"). Pre-#70 the `sep > atol` filter
    # discarded it entirely.
    eigs = np.array([0.0, -1.0, -1.0, -3.0], dtype=complex)  # -1 is doubly degenerate
    prox, count = lep_proximity(eigs)
    assert prox == 0.0
    assert count >= 1  # the coalesced pair is counted


def test_lep_proximity_fully_degenerate_spectrum_is_zero_not_inf():
    # issue #70 A9: a fully degenerate spectrum is maximal coalescence -> 0.0,
    # not inf. Pre-#70 this returned inf (the exact inverse of the physics).
    eigs = np.array([-2.0, -2.0, -2.0], dtype=complex)
    prox, count = lep_proximity(eigs)
    assert prox == 0.0
    assert count == 3  # all three i<j pairs are within the (atol) window


def test_lep_proximity_near_degenerate_below_atol_clamped_to_zero():
    # A separation below atol is numerically indistinguishable from coalescence.
    eigs = np.array([0.0, -1.0, -1.0 - 1e-13], dtype=complex)  # 1e-13 < EPS_GAP=1e-10
    prox, _ = lep_proximity(eigs)
    assert prox == 0.0


def test_lep_proximity_nondegenerate_is_unchanged():
    # Regression: well-separated eigenvalues keep the ordinary min-separation
    # behaviour (the fix only changes the degenerate / sub-atol regime).
    eigs = np.array([0.0, -1.0, -3.0], dtype=complex)
    prox, count = lep_proximity(eigs)
    assert prox == pytest.approx(1.0)  # min sep is |0 - (-1)| = 1
    assert count >= 1


def test_lep_proximity_rejects_nonfinite_eigenvalues():
    # issue #82: NaN/inf eigenvalues used to be silently ignored by comparisons,
    # typically yielding (1.0, 1) and misclassifying it as a valid D16 signal. That is
    # a silent-failure mode, not a valid D16 signal, so it must fail closed.
    for eigs in (
        np.array([0.0, np.nan, -1.0], dtype=complex),
        np.array([0.0, np.inf, -1.0], dtype=complex),
        np.array([0.0, complex(0.0, np.inf), -1.0], dtype=complex),
    ):
        with pytest.raises(ValueError, match="finite eigenvalues"):
            lep_proximity(eigs)


def test_compute_lep_layer_propagates_nonfinite_eigenvalue_error(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    eigs = np.array([0.0, np.nan, -1.0], dtype=complex)
    with pytest.raises(ValueError, match="finite eigenvalues"):
        compute_lep_layer(L, eigs, beta_D_linear=0.6, gap=0.3, n_haar=4)


def test_lep_proximity_candidate_loop_consistent_with_min_sep():
    # issue #70 A9: both loops apply the SAME data/window. A degenerate pair plus
    # a distant eigenvalue: the closest pair is the degenerate one (window=atol),
    # so only the coalesced pair is counted, not the distant one.
    eigs = np.array([-1.0, -1.0, -50.0], dtype=complex)
    prox, count = lep_proximity(eigs)
    assert prox == 0.0
    assert count == 1  # only the (-1, -1) pair within the atol window


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
