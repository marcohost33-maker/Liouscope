"""Tests for LEP layer D16-D18."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope import build_liouvillian, steady_state
from liouscope.diagnostics.lep import (
    compute_lep_layer,
    gap_rate_consistency,
    initial_state_sensitivity,
    lep_proximity,
)
from liouscope.numerics.linalg import eig_nonhermitian
import liouscope.numerics.propagation as propagation_module


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


def test_initial_state_sensitivity_small_system_is_bitwise_legacy(pauli):
    """n < 64 must keep the pre-#178 dense formula exactly."""
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho_ss = steady_state(L)
    seed = 19
    n_samples = 7
    t_eval = 1.25

    gen = np.random.default_rng(seed)
    propagator = sla.expm(L * t_eval)
    legacy = np.empty(n_samples)
    d = rho_ss.shape[0]
    for k in range(n_samples):
        psi = gen.normal(size=d) + 1j * gen.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())
        rho_t = (propagator @ rho0.reshape(-1, order="F")).reshape((d, d), order="F")
        legacy[k] = float(np.linalg.norm(rho_t - rho_ss, ord="fro"))

    current = initial_state_sensitivity(
        L, rho_ss, n_samples=n_samples, t_eval=t_eval, seed=seed
    )
    assert current == float(np.std(legacy))


def test_initial_state_sensitivity_block_action_agrees_with_dense():
    """D18's block action agrees with its dense reference for n=64."""
    n = 64
    d = 8
    L = -0.2 * np.eye(n, dtype=complex)
    rho_ss = np.eye(d, dtype=complex) / d
    kwargs = dict(n_samples=6, t_eval=1.75, seed=23)
    dense = initial_state_sensitivity(
        L, rho_ss, propagation_backend="dense_expm", **kwargs
    )
    action = initial_state_sensitivity(
        L, rho_ss, propagation_backend="expm_action", **kwargs
    )
    assert action == pytest.approx(dense, rel=1e-12, abs=1e-14)


def test_d18_block_action_scales_condition_3_13_by_rhs_count(monkeypatch):
    """A block must use the n0-scaled exact-norm branch, not the 1-RHS bound."""
    n = 64
    d = 8
    # Shifted norm * t is 30: below the one-RHS margin (~57), but far above
    # the 10-RHS margin (~5.7). A missing / n_rhs would be caught here.
    L = np.diag(np.linspace(-3.0, 3.0, n)).astype(complex)
    rho_ss = np.eye(d, dtype=complex) / d
    real = propagation_module.expm_multiply
    seen = []

    def checked(A, B, **kwargs):
        n_rhs = B.shape[1]
        scaled_norm = propagation_module.shifted_one_norm(A)
        limit = propagation_module._SUBSTEP_NORM / n_rhs
        assert scaled_norm <= limit * (1.0 + 1e-12)
        seen.append((scaled_norm, n_rhs))
        return real(A, B, **kwargs)

    monkeypatch.setattr(propagation_module, "expm_multiply", checked)
    initial_state_sensitivity(
        L,
        rho_ss,
        n_samples=10,
        t_eval=10.0,
        seed=7,
        propagation_backend="expm_action",
    )
    assert len(seen) > 1
    assert {n_rhs for _norm, n_rhs in seen} == {10}


def test_d18_block_action_does_not_consume_global_numpy_rng():
    """Regression pin for SciPy's randomised onenormest side effect."""
    n = 64
    d = 8
    L = np.diag(np.linspace(-3.0, 3.0, n)).astype(complex)
    rho_ss = np.eye(d, dtype=complex) / d
    np.random.seed(20260930)
    before = np.random.get_state()
    initial_state_sensitivity(
        L,
        rho_ss,
        n_samples=10,
        t_eval=10.0,
        seed=11,
        propagation_backend="expm_action",
    )
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_compute_lep_layer_records_d18_action_backend():
    """The numerical method behind reported D18 is auditable."""
    n = 64
    d = 8
    L = -0.1 * np.eye(n, dtype=complex)
    rho_ss = np.eye(d, dtype=complex) / d
    eigs = np.linspace(-1.0, 0.0, n, dtype=complex)
    result = compute_lep_layer(
        L,
        eigs,
        beta_D_linear=0.1,
        gap=0.1,
        rho_steady_state=rho_ss,
        n_haar=4,
        seed=3,
    )
    assert result.initial_state_backend == "expm_action"
