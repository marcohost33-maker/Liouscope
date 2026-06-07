"""Tests for the transient layer (D14 sup-propagator-norm, D15 kappa_trans).

Focus: D14 time-grid is physics-scaled to the spectral gap, so the supremum is
not silently under-resolved for small-gap (slow-decay) non-normal systems --
exactly the regime the diagnostic targets. The independent oracle is a very
dense propagator-norm sweep over a window scaled to the true decay timescale.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope import build_liouvillian
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.diagnostics.transient import (
    TransientGridWarning,
    compute_transient_layer,
    numerical_abscissa,
    trans_amplitude_ratio,
)

_SM = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^-
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _dense_sup(L: np.ndarray, gap: float, *, horizon: float = 40.0, n: int = 8000) -> float:
    """Independent oracle: brute-force sup_t ||e^{tL}||_2 on a very dense grid
    scaled to the true relaxation timescale 1/gap."""
    t_grid = np.linspace(0.0, horizon / max(gap, 1e-12), n)
    return max(float(sla.svdvals(sla.expm(L * t))[0]) for t in t_grid)


def _fixed_legacy_sup(L: np.ndarray) -> float:
    """The old fixed coarse grid, reproduced for the regression contrast."""
    return trans_amplitude_ratio(L)  # gap=None, t_grid=None -> legacy fallback


# --- physics-scaling: matches the dense oracle where the fixed grid fails -----


@pytest.mark.parametrize("damping", [0.02, 0.5, 1.0, 50.0])
def test_d14_matches_dense_oracle_across_timescales(damping: float):
    """sup||e^{tL}|| from the physics-scaled grid must match a dense reference
    to high accuracy across three orders of magnitude in the gap."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [damping])
    gap = compute_spectral_layer(L).gap
    got = trans_amplitude_ratio(L, gap=gap)
    ref = _dense_sup(L, gap)
    assert ref > 0
    assert abs(got - ref) / ref < 1e-3, f"got {got}, ref {ref}, gap {gap}"


def test_d14_small_gap_fixed_grid_underestimates_physics_grid_does_not():
    """Regression for audit P2-1: for a small gap the legacy fixed grid
    [0.01,5] underestimates the transient sup; the physics-scaled grid recovers
    the true value (here the amplitude-damping channel peaks at sqrt(2))."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [0.02])
    gap = compute_spectral_layer(L).gap
    ref = _dense_sup(L, gap)

    legacy = _fixed_legacy_sup(L)
    physics = trans_amplitude_ratio(L, gap=gap)

    # legacy is a meaningful underestimate (>10% low) ...
    assert (ref - legacy) / ref > 0.10
    # ... while the physics-scaled grid is essentially exact.
    assert abs(physics - ref) / ref < 1e-3
    assert physics > legacy


def test_d14_explicit_t_grid_overrides_scaling():
    """An explicit t_grid must take precedence over gap-based scaling."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [1.0])
    gap = compute_spectral_layer(L).gap
    custom = np.linspace(0.0, 10.0, 200)
    got = trans_amplitude_ratio(L, gap=gap, t_grid=custom)
    ref = _dense_sup(L, gap)
    assert abs(got - ref) / ref < 1e-3


def test_d14_legacy_fallback_still_available():
    """Without gap or t_grid the legacy coarse grid is used (no crash, finite)."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [1.0])
    val = trans_amplitude_ratio(L)
    assert np.isfinite(val)
    assert val >= 1.0  # ||e^{0 L}|| = 1 is always in the grid range


# --- boundary-rise warning (silent-underestimation guard) ---------------------


def test_d14_warns_when_sup_at_right_edge():
    """If the user forces a too-short window so the norm is still rising at the
    right edge, the auto-scaled path must warn that the result is a lower bound.
    We trigger it by handing gap-scaling a tiny truncated grid via a forced gap
    that overshoots the real decay scale is not possible through the public API,
    so we drive the guard directly through the auto-scaled branch using a gap so
    large the window is far too short for the slow true decay."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [0.02])  # true slow decay, gap ~ 0.01
    # Lie about the gap (claim it is large) -> auto-window truncates early ->
    # the propagator is still rising at the edge -> warning expected.
    with pytest.warns(TransientGridWarning):
        trans_amplitude_ratio(L, gap=5.0)


def test_d14_no_warning_for_well_resolved_window():
    """A correctly scaled window must not raise the underestimation warning."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [1.0])
    gap = compute_spectral_layer(L).gap
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", TransientGridWarning)
        trans_amplitude_ratio(L, gap=gap)  # must not raise


# --- layer integration + degenerate-gap robustness ----------------------------


def test_compute_transient_layer_forwards_gap():
    """compute_transient_layer must physics-scale via the passed gap and return
    a result consistent with the dense oracle."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [0.05])
    gap = compute_spectral_layer(L).gap
    res = compute_transient_layer(L, gap)
    ref = _dense_sup(L, gap)
    assert abs(res.trans_amplitude_ratio - ref) / ref < 1e-3
    assert np.isfinite(res.numerical_abscissa)
    assert res.kappa_trans == pytest.approx(res.numerical_abscissa / gap)


def test_d14_non_positive_gap_does_not_crash():
    """gap <= 0 (degenerate / no isolated steady state) must still terminate
    and yield a finite sup driven by the growth scale."""
    L = build_liouvillian(0.0 * _SZ, [_SM], [1.0])
    val = trans_amplitude_ratio(L, gap=0.0)
    assert np.isfinite(val)
    assert val >= 1.0


def test_numerical_abscissa_grid_free_sanity():
    """omega(L) is the max eigenvalue of the Hermitian part; for a contractive
    Lindbladian it is >= 0 and finite."""
    L = build_liouvillian(2.0 * _SX, [_SM], [0.1])
    omega = numerical_abscissa(L)
    assert np.isfinite(omega)
