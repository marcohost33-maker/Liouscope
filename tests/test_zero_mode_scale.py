"""Issue #108: rate-unit conformance for the ZERO-MODE separation.

The #101 slice-B suite covers the scale-relative successor diagnostics
(D8b/D10b/D13) but not the layer that decides *which modes are the steady
state*. That filter sat behind an absolute floor ``|lambda| > 1e-10`` in five
modules and broke in both directions under a pure change of rate units:

* small ``c`` -- every genuine mode drops below the floor, so D1 collapses to
  ``0.0`` (firing the gapless F5 reach leg) and D19 ``c_1`` collapses to
  ``0.0``, raising a FALSE A11/F4 Mpemba candidate on the highest-priority
  rung of the classifier;
* large ``c`` -- the round-off zero mode (``~eps * ||L||``) rises ABOVE the
  floor and counts as genuine, producing a NEGATIVE spectral gap, which is
  impossible for a GKSL generator by definition.

These tests pin the corrected contract: dimensionless quotients are invariant,
rate-valued outputs scale by ``c``, and the end-to-end verdict does not move
under a pure unit change.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope._consts import ZERO_MODE_EPS_FACTOR
from liouscope.core.lindblad import steady_state
from liouscope.diagnostics.lep import lep_proximity
from liouscope.diagnostics.mpemba import compute_mpemba_layer, overlap_c1
from liouscope.diagnostics.nonnormality import petermann_factors
from liouscope.diagnostics.spectral import (
    compute_spectral_layer,
    liouvillian_gap,
    oscillating_mode_gap,
    spectral_spread,
)
from liouscope.numerics.scale import spectral_zero_tolerance

# Wider than the #101 set on the upper end: direction B only bites once
# ``eps * ||L||`` exceeds the legacy absolute floor, which happens above ~1e6.
C_SET = (1.0e-10, 1.0e-5, 1.0, 1.0e5, 1.0e10, 1.0e12)

RTOL_INV = 1.0e-8


def _amplitude_damping_L() -> np.ndarray:
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [sm])


def _rabi_damped_L() -> np.ndarray:
    """Rabi-driven amplitude damping: the direction-B repro system."""
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return build_liouvillian(0.7 * sx, [sm], rates=[0.4])


def _plus_state() -> np.ndarray:
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    return np.outer(plus, plus.conj())


# ---------------------------------------------------------------------------
# The shared tolerance helper
# ---------------------------------------------------------------------------


def test_zero_tolerance_is_homogeneous_degree_one():
    ev = np.array([0.0, -0.5, -1.0 + 2.0j], dtype=complex)
    base = spectral_zero_tolerance(ev)
    eps = float(np.finfo(float).eps)
    assert base == pytest.approx(ZERO_MODE_EPS_FACTOR * eps * 2.23606797749979, rel=1e-12)
    for c in C_SET:
        assert spectral_zero_tolerance(c * ev) == pytest.approx(c * base, rel=1e-12)


def test_zero_tolerance_zero_operator_and_empty_semantics():
    assert spectral_zero_tolerance(np.zeros(4, dtype=complex)) == 0.0
    assert spectral_zero_tolerance(np.array([], dtype=complex)) == 0.0


def test_zero_tolerance_legacy_absolute_opt_in():
    ev = np.array([0.0, -1.0e6], dtype=complex)
    assert spectral_zero_tolerance(ev, atol=1.0e-10) == 1.0e-10


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_legacy_atol_still_validates_the_spectrum(bad):
    """A compatibility switch may restore the old threshold, not old silence.

    Before this guard, `liouvillian_gap([0, nan], atol=1e-10)` reported 0.0 --
    corrupted solver output laundered into "no non-zero modes".
    """
    with pytest.raises(ValueError, match="finite"):
        spectral_zero_tolerance(np.array([0.0, bad], dtype=complex), atol=1.0e-10)
    with pytest.raises(ValueError, match="finite"):
        liouvillian_gap(np.array([0.0, bad], dtype=complex), atol=1.0e-10)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_zero_tolerance_fails_closed_on_nonfinite(bad):
    """A NaN scale would make every ``|lambda| > tol`` compare False.

    That is the silent-failure mode this guard exists for: it would report
    "no non-zero modes", i.e. gap 0.0, for corrupted eigensolver output.
    """
    with pytest.raises(ValueError, match="finite"):
        spectral_zero_tolerance(np.array([0.0, bad], dtype=complex))


@pytest.mark.parametrize("bad_rtol", [-1.0, np.nan, np.inf])
def test_zero_tolerance_rejects_invalid_rtol(bad_rtol):
    with pytest.raises(ValueError, match="rtol"):
        spectral_zero_tolerance(np.array([1.0]), rtol=bad_rtol)


# ---------------------------------------------------------------------------
# Direction A: small c -- no mode may vanish, no false Mpemba candidate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", C_SET)
def test_gap_scales_exactly_with_rate_unit(c):
    L = _amplitude_damping_L()
    ev0 = np.linalg.eigvals(L)
    ref = liouvillian_gap(ev0)
    scaled = liouvillian_gap(np.linalg.eigvals(c * L))
    assert scaled / c == pytest.approx(ref, rel=RTOL_INV)


@pytest.mark.parametrize("c", C_SET)
def test_spectral_spread_scales_exactly_with_rate_unit(c):
    L = _amplitude_damping_L()
    ref = spectral_spread(np.linalg.eigvals(L))
    scaled = spectral_spread(np.linalg.eigvals(c * L))
    assert scaled / c == pytest.approx(ref, rel=RTOL_INV)


@pytest.mark.parametrize("c", C_SET)
def test_oscillating_gap_scales_exactly_with_rate_unit(c):
    L = _rabi_damped_L()  # has genuine complex pairs
    ref = oscillating_mode_gap(np.linalg.eigvals(L))
    assert ref > 0.0, "control must have complex pairs, else the test is vacuous"
    scaled = oscillating_mode_gap(np.linalg.eigvals(c * L))
    assert scaled / c == pytest.approx(ref, rel=RTOL_INV)


@pytest.mark.parametrize("c", C_SET)
def test_has_complex_pairs_flag_is_unit_invariant(c):
    L = _rabi_damped_L()
    ref = compute_spectral_layer(L).has_complex_pairs
    assert compute_spectral_layer(c * L).has_complex_pairs is ref


@pytest.mark.parametrize("c", C_SET)
def test_mpemba_overlap_is_unit_invariant(c):
    """D19 is a dimensionless expansion coefficient: it must not move at all."""
    L = _amplitude_damping_L()
    rho0 = _plus_state()
    ref = overlap_c1(L, rho0)
    assert ref == pytest.approx(0.5, rel=1e-9), "control anchors the expected value"
    assert overlap_c1(c * L, rho0) == pytest.approx(ref, rel=RTOL_INV)


@pytest.mark.parametrize("c", C_SET)
def test_no_false_mpemba_candidate_under_rate_rescale(c):
    """The #108 headline defect: a unit change must not create an A11 candidate."""
    L = _amplitude_damping_L()
    rho_ss = steady_state(L)
    rho0 = _plus_state()
    result = compute_mpemba_layer(c * L, rho0, rho_steady_state=rho_ss)
    assert result.is_mpemba_candidate is False
    assert result.overlap_c1 == pytest.approx(0.5, rel=RTOL_INV)


@pytest.mark.parametrize("c", C_SET)
def test_lep_proximity_scales_with_rate_unit(c):
    """D16 is a rate-valued separation, so it must scale by exactly ``c``."""
    L = _rabi_damped_L()
    ref, ref_count = lep_proximity(np.linalg.eigvals(L))
    scaled, scaled_count = lep_proximity(np.linalg.eigvals(c * L))
    assert scaled_count == ref_count
    if np.isfinite(ref) and ref > 0.0:
        assert scaled / c == pytest.approx(ref, rel=RTOL_INV)


@pytest.mark.parametrize("c", C_SET)
def test_petermann_mode_count_is_unit_invariant(c):
    """The number of retained non-zero modes must not depend on the unit."""
    L = _rabi_damped_L()
    ref_eigs, ref_k = petermann_factors(L)
    eigs, k = petermann_factors(c * L)
    assert eigs.size == ref_eigs.size
    assert k == pytest.approx(ref_k, rel=RTOL_INV)


# ---------------------------------------------------------------------------
# Direction B: large c -- the round-off zero mode must stay a zero mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [1.0, 1.0e6, 1.0e10, 1.0e12])
def test_gap_is_never_negative_for_a_gksl_generator(c):
    """A negative gap is impossible: Delta = -max{Re lambda != 0} >= 0.

    Before #108 the round-off zero mode (~eps * ||L||) escaped the absolute
    floor above c ~ 1e6 and was counted as genuine, giving gap = -1.4e-6 at
    c = 1e10.
    """
    L = _rabi_damped_L()
    gap = liouvillian_gap(np.linalg.eigvals(c * L))
    assert gap >= 0.0
    assert gap / c == pytest.approx(0.2, rel=1e-6)


def test_unstable_generator_still_reports_a_negative_gap():
    """The fix removes the round-off cause, it must NOT mask the physical one.

    A genuinely unstable mode (positive real part, i.e. non-GKSL input) has to
    stay visible as a negative gap rather than being clamped to zero.
    """
    ev = np.array([0.0, +0.3, -1.0], dtype=complex)
    assert liouvillian_gap(ev) == pytest.approx(-0.3)


# ---------------------------------------------------------------------------
# End-to-end: the verdict itself must not move under a unit change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [1.0e-10, 1.0, 1.0e10])
def test_diagnose_verdict_is_invariant_under_rate_rescale(c):
    """Full pipeline, metamorphic in the rate unit (t -> t/c keeps the physics)."""
    L = _amplitude_damping_L()
    rho0 = _plus_state()
    rep = diagnose(
        c * L,
        rho_initial=rho0,
        t_grid=np.linspace(0.0, 5.0 / c, 64),
        bootstrap_B=20,
        seed=1,
    )
    cls = rep.classification
    assert (cls.a_class, cls.f_family) == ("A12", "none")
    assert cls.verdict == "NOT_EXCLUDED"
    assert rep.mpemba is not None
    assert rep.mpemba.is_mpemba_candidate is False
    assert rep.spectral.gap / c == pytest.approx(0.5, rel=1e-6)


def test_legacy_absolute_floor_reproduces_the_pre_108_defect():
    """The ``atol`` opt-in must genuinely restore the old behaviour.

    This is the discrimination test for the fix: if passing the legacy floor
    changed nothing, the scale-relative path would not actually be in force.
    """
    L = _amplitude_damping_L()
    ev_small = np.linalg.eigvals(1.0e-10 * L)
    assert liouvillian_gap(ev_small) > 0.0                     # fixed default
    assert liouvillian_gap(ev_small, atol=1.0e-10) == 0.0      # legacy defect


# ---------------------------------------------------------------------------
# Dynamic range: genuine slow modes must survive (Codex review P1)
# ---------------------------------------------------------------------------


def _two_channel_L(rate_slow: float) -> np.ndarray:
    """Two independent damping channels with widely separated rates.

    This is the metastable (A5) regime — a wide separation of physical rates is
    the phenomenon, not an artefact — so the zero-mode filter must not treat the
    slow branch as round-off.
    """
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    return build_liouvillian(
        np.zeros((4, 4), dtype=complex),
        [np.kron(sm, eye), np.kron(eye, sm)],
        rates=[1.0, rate_slow],
    )


@pytest.mark.parametrize("rate_slow", [1.0e-2, 1.0e-6, 1.0e-10, 1.0e-12])
def test_genuine_slow_modes_survive_the_zero_mode_filter(rate_slow):
    """A fixed fraction of the spectral radius would discard these.

    With a `1e-10 * max|lambda|` threshold, `rate_slow = 1e-12` reported a gap
    of 5e-1 instead of 5e-13 — wrong by ten orders of magnitude, and precisely
    on the metastable systems the library exists to study. Tying the threshold
    to the eigensolver backward error (`eps * max|lambda|`) instead resolves
    modes far below any fixed dynamic-range ceiling.
    """
    L = _two_channel_L(rate_slow)
    ev = np.linalg.eigvals(L)
    assert liouvillian_gap(ev) == pytest.approx(0.5 * rate_slow, rel=1.0e-6)


def test_slow_mode_survival_is_also_unit_invariant():
    """Wide rate separation and unit rescaling must compose, not conflict."""
    L = _two_channel_L(1.0e-10)
    for c in (1.0e-6, 1.0, 1.0e6):
        gap = liouvillian_gap(np.linalg.eigvals(c * L))
        assert gap / c == pytest.approx(0.5e-10, rel=1.0e-6)


def test_threshold_sits_far_above_measured_round_off():
    """The calibration claim behind ZERO_MODE_EPS_FACTOR, asserted not assumed.

    Across these controls the numerical zero mode never exceeded ~2 * eps *
    max|lambda|; the default factor keeps orders of magnitude of headroom, which
    is what stops direction B (a round-off mode promoted to genuine, yielding a
    negative gap) from returning.
    """
    eps = float(np.finfo(float).eps)
    for L in (_amplitude_damping_L(), _rabi_damped_L(), _two_channel_L(1.0e-6)):
        for c in (1.0, 1.0e6, 1.0e12):
            ev = np.linalg.eigvals(c * L)
            radius = float(np.max(np.abs(ev)))
            zero_mode = float(np.min(np.abs(ev)))
            assert zero_mode <= 10.0 * eps * radius
            assert zero_mode < spectral_zero_tolerance(ev)


def test_tolerance_tracks_the_spectrum_dtype():
    """Backward error scales with the SOLVER's working precision, not float64.

    A single-precision spectrum has round-off ~1e-7 relative; judging it by
    float64 eps puts the threshold ~9 decades below that solver's noise, so the
    numerical steady-state eigenvalue is retained as physical -- reintroducing
    the negative gap this helper exists to prevent.
    """
    ev64 = np.array([0.0, -0.5, -1.0], dtype=np.complex128)
    ev32 = ev64.astype(np.complex64)
    tol64 = spectral_zero_tolerance(ev64)
    tol32 = spectral_zero_tolerance(ev32)
    assert tol32 > tol64
    ratio = float(np.finfo(np.float32).eps / np.finfo(np.float64).eps)
    assert tol32 / tol64 == pytest.approx(ratio, rel=1e-9)

    # Real (non-complex) single precision must behave the same way.
    assert spectral_zero_tolerance(np.array([0.0, -1.0], dtype=np.float32)) == pytest.approx(
        tol32, rel=1e-9
    )


def test_single_precision_zero_mode_is_still_filtered():
    """The end-to-end consequence of the dtype fix: no negative gap."""
    L = _rabi_damped_L()
    for c in (1.0, 1.0e-3, 1.0e3):
        ev = np.linalg.eigvals(c * L).astype(np.complex64)
        gap = liouvillian_gap(ev)
        assert gap >= 0.0
        assert gap / c == pytest.approx(0.2, rel=1.0e-3)
