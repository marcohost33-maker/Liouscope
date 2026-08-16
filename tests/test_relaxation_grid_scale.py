"""Rate-unit invariance of the DEFAULT relaxation time grid.

The relaxation layer fits every rate it reports on a time grid. A decay rate
carries dimension ``1/time``, so an *absolute* default window silently encodes
a claim about the caller's unit of time. The legacy default
``linspace(0.0, 10.0, 80)`` did exactly that, which made D5/D6/D7, the
M0..M3b AICc comparison, ``beta_D``, its BCa interval and the D17 gap-rate
check unit-dependent: under the pure rescale ``L -> cL`` (identical physics,
different unit of time) the measured behaviour on an amplitude-damped qubit was

    c        beta_D / c     beta_D_linear / c    A-class
    1e+02    1.025          0.254                A10
    1e+00    1.029          0.482                A5
    1e-02    1.085          0.403                A1
    1e-04    1.167          0.965                A12
    1e-06    1.251          109.99               A12

i.e. a 22 % drift in ``beta_D``, a factor ~430 blow-up in the D17 linear rate,
and four different mechanism classes for one system. Rates in real systems are
MHz or GHz, so this was the common regime rather than a corner case.

:func:`liouscope.diagnostics.relaxation.default_relaxation_grid` replaces the
absolute window with ``[0, HORIZON / Delta]`` -- a fixed number of e-foldings
of the slowest mode (D1), which is the only choice carried along by the
rescaling. These tests pin that contract.

Two independent unit dependences remain OUTSIDE this grid and are deliberately
NOT asserted away here: the least-squares solver's own convergence controls
(issue #111) and the rate-dimensioned ``henrici_eta`` / ``resolvent_peak``
consumed by the classifier (issue #101). The A-class assertion below is
therefore restricted to the range where those two do not dominate.
"""

from __future__ import annotations

import warnings
from functools import cache

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope.diagnostics.relaxation import (
    MIN_SAMPLES_PER_FAST_EFOLD,
    RELAXATION_HORIZON,
    RELAXATION_N_POINTS,
    UnderResolvedTransientWarning,
    compute_relaxation_layer,
    default_relaxation_grid,
)
from liouscope.diagnostics.spectral import compute_spectral_layer

# Twelve decades of rate units. Measured worst-case drift across this set is
# ~1.2e-3 relative; the defect this guards against was 2.2e-1, so RTOL_RATE
# sits ~8x above the observed noise and ~20x below the regression.
C_SET = (1.0e-6, 1.0e-4, 1.0e-2, 1.0, 1.0e2, 1.0e4, 1.0e6)
RTOL_RATE = 1.0e-2

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_RHO_PLUS = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)


def _amp_damped(c: float) -> np.ndarray:
    """Amplitude-damped qubit with every rate scaled by ``c`` (gap = c/2)."""
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [np.sqrt(c) * _SM])


def _rabi_dephasing(c: float) -> np.ndarray:
    """Driven, dephasing qubit with every rate scaled by ``c`` (gap = 0.3 c)."""
    return build_liouvillian(0.5 * c * _X, [np.sqrt(0.3 * c) * _Z])


SYSTEMS = {"amp_damped": _amp_damped, "rabi_dephasing": _rabi_dephasing}


@cache
def _report(system: str, c: float):
    """One ``diagnose()`` per (system, rate unit), shared across tests.

    The scaling tests below interrogate different fields of the SAME runs, and
    a full pipeline call is not cheap; without this the module would re-run the
    identical sweep once per assertion and dominate CI wall-clock on a
    five-version matrix. Keyed on the scalars rather than the array because the
    Liouvillian is a pure function of them.
    """
    # The GLS layer legitimately warns about small-n AR(1) bias on an 80-point
    # grid; that is orthogonal to the scaling contract under test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose(SYSTEMS[system](c), rho_initial=_RHO_PLUS, bootstrap_B=10, seed=1)


def _diagnose(L: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return diagnose(L, rho_initial=_RHO_PLUS, bootstrap_B=10, seed=1)


# ---------------------------------------------------------------------------
# default_relaxation_grid: construction contract
# ---------------------------------------------------------------------------


def test_default_grid_is_bit_identical_to_the_legacy_window_at_unit_gap():
    """Backward-compatibility pin: the change is a no-op at ``Delta = 1``.

    Every anchor and validation system whose gap is 1 must see exactly the
    grid it saw before, so a drift there is a real regression rather than a
    tolerance question.
    """
    np.testing.assert_array_equal(
        default_relaxation_grid(1.0), np.linspace(0.0, 10.0, 80)
    )


def test_default_grid_spans_a_fixed_number_of_e_foldings():
    for gap in (1.0e-6, 0.3, 0.5, 1.0, 7.0, 1.0e6):
        grid = default_relaxation_grid(gap)
        assert grid.size == RELAXATION_N_POINTS
        assert grid[0] == 0.0
        # The window measured in the system's own relaxation time is constant.
        assert grid[-1] * gap == pytest.approx(RELAXATION_HORIZON, rel=1.0e-12)


def test_default_grid_is_uniform():
    """The AR(1) whitening in the GLS fit presumes a constant sample interval.

    ``_whiten`` forms ``r_k - rho r_{k-1}`` with a single ``rho``; on a
    log-spaced or two-scale grid the lag-1 correlation would vary along the
    grid and quietly invalidate the whitening, the AR(1) bootstrap and N_eff.
    """
    steps = np.diff(default_relaxation_grid(0.37))
    np.testing.assert_allclose(steps, steps[0], rtol=1.0e-12)


@pytest.mark.parametrize("bad_gap", [0.0, -1.0, -1.0e-30, float("nan"), float("inf")])
def test_default_grid_falls_back_when_no_decay_scale_exists(bad_gap):
    """``Delta <= 0`` means no resolved decay scale: no timescale to scale by.

    Any window is then a convention, so the historical one is kept rather than
    inventing a second arbitrary constant.
    """
    np.testing.assert_array_equal(
        default_relaxation_grid(bad_gap), np.linspace(0.0, 10.0, 80)
    )


# ---------------------------------------------------------------------------
# End-to-end: the fitted rates track a pure change of rate units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_fitted_rates_scale_with_the_rate_unit(system):
    """``beta_D`` and ``beta_D_linear`` are homogeneous of degree 1 in ``c``.

    This is the property the absolute default window broke. It is asserted on
    the RATIO ``beta / c``, which is the dimensionless number the physics
    fixes; an implementation that reverts to an absolute grid fails here by
    more than an order of magnitude, not marginally.
    """
    reports = {c: _report(system, c) for c in C_SET}
    ref = reports[1.0]

    for c, rep in reports.items():
        np.testing.assert_allclose(
            rep.spectral.gap / c, ref.spectral.gap, rtol=1.0e-9,
            err_msg=f"D1 gap not rate-homogeneous at c={c:g}",
        )
        np.testing.assert_allclose(
            rep.relaxation.beta_D / c, ref.relaxation.beta_D, rtol=RTOL_RATE,
            err_msg=f"beta_D not rate-homogeneous at c={c:g}",
        )
        np.testing.assert_allclose(
            rep.relaxation.beta_D_linear / c, ref.relaxation.beta_D_linear,
            rtol=RTOL_RATE,
            err_msg=f"beta_D_linear not rate-homogeneous at c={c:g}",
        )


@pytest.mark.parametrize("system", sorted(SYSTEMS))
def test_aicc_model_selection_is_rate_unit_invariant(system):
    """The winning member of the M-hierarchy is a statement about the CURVE.

    Two runs that differ only by the unit of time see the same dimensionless
    curve, so they must select the same model. Under the absolute window the
    winner flipped between M0 and M2 with ``c``, which changes both the
    reported rate's meaning and the A10/F5 Jordan evidence.
    """
    winners = {c: _report(system, c).relaxation.aicc_model for c in C_SET}
    assert len(set(winners.values())) == 1, f"AICc winner varies with rate unit: {winners}"


def test_mechanism_class_is_stable_across_rate_units():
    """The reported A-class no longer moves with the caller's unit of time.

    Six decades, ``c in [1e-6, 1]``, where it previously produced four
    different classes (A12/A1/A5/A10).

    The upper bound is NOT arbitrary and NOT a tolerance: ``henrici_eta`` (D8)
    is rate-dimensioned and equals ``c`` exactly on this system, so it crosses
    its absolute classifier threshold between ``c = 1`` and ``c = 3`` and flips
    A5 -> A10 regardless of the time grid. That is issue #101 -- a second,
    independent unit dependence in the non-normality layer, not in this window
    -- and this PR neither fixes it nor papers over it. The companion test
    :func:`test_henrici_eta_is_the_remaining_unit_dependence_above_unit_rates`
    pins that boundary explicitly so it cannot be mistaken for a grid problem.
    """
    classes = {
        c: _report("amp_damped", c).classification.a_class
        for c in (1.0e-6, 1.0e-4, 1.0e-2, 1.0)
    }
    assert len(set(classes.values())) == 1, f"A-class varies with rate unit: {classes}"


def test_henrici_eta_is_the_remaining_unit_dependence_above_unit_rates():
    """Locate the residual class flip in D8, not in the relaxation window.

    Documents the open part of issue #101 as a measured fact: the fitted rates
    stay invariant across the flip (so the grid is doing its job), while the
    rate-dimensioned ``henrici_eta`` tracks ``c`` one-for-one and drags the
    A-class with it. Should #101 be closed by making the classifier consume the
    dimensionless D8b instead, this test is the one that must be updated.
    """
    low, high = _report("amp_damped", 1.0), _report("amp_damped", 1.0e2)

    # D8 is rate-dimensioned: it is literally the rescaling factor apart.
    assert low.classification.evidence["henrici_eta"] == pytest.approx(1.0, rel=1e-6)
    assert high.classification.evidence["henrici_eta"] == pytest.approx(1.0e2, rel=1e-6)

    # ... while the quantities this PR governs did NOT move.
    assert high.relaxation.beta_D / 1.0e2 == pytest.approx(
        low.relaxation.beta_D, rel=RTOL_RATE
    )
    assert high.relaxation.aicc_model == low.relaxation.aicc_model
    assert high.classification.a_class != low.classification.a_class


def test_linear_rate_does_not_blow_up_at_small_rate_units():
    """Direct regression pin on the worst observed symptom.

    With the absolute window, ``beta_D_linear / c`` reached ~110 at ``c=1e-6``
    against a true value of ~0.49 -- the D17 gap-rate consistency check was
    reading a rate wrong by a factor of ~430 while every other field looked
    healthy. Pinned separately from the invariance test above so the failure
    message names this specific mode.
    """
    rep = _diagnose(_amp_damped(1.0e-6))
    assert rep.relaxation.beta_D_linear / 1.0e-6 < 2.0


# ---------------------------------------------------------------------------
# Provenance: which window was used, and why, is recorded on the result
# ---------------------------------------------------------------------------


def test_grid_provenance_records_the_gap_scaled_default():
    rep = _diagnose(_amp_damped(1.0e-3))
    assert rep.relaxation.t_grid_source == "gap_scaled"
    # gap = c/2 = 5e-4  ->  span = 10 / 5e-4 = 2e4
    assert rep.relaxation.t_grid_span == pytest.approx(2.0e4, rel=1.0e-9)


def test_stored_grid_identifies_the_sampling_not_just_the_span():
    """A span does not identify a grid; the exported curves need their abscissa.

    Reviewer finding on PR #115 (Codex P2). ``[0, 1, 10]`` and ``[0, 9, 10]``
    share a span of 10 while sampling materially different trajectories, and
    the report already serialises three 80-point curves whose x-axis was
    missing — so a consumer could not re-fit, re-plot or audit the rates.
    """
    a = np.array([0.0, 1.0, 10.0])
    b = np.array([0.0, 9.0, 10.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra = compute_relaxation_layer(
            _amp_damped(1.0), rho_initial=_RHO_PLUS, t_grid=a, bootstrap_B=5, seed=1
        )
        rb = compute_relaxation_layer(
            _amp_damped(1.0), rho_initial=_RHO_PLUS, t_grid=b, bootstrap_B=5, seed=1
        )
    # The span alone cannot tell these apart ...
    assert ra.t_grid_span == rb.t_grid_span == pytest.approx(10.0)
    # ... the stored grid can, and it is the grid actually used.
    np.testing.assert_array_equal(ra.t_grid, a)
    np.testing.assert_array_equal(rb.t_grid, b)


def test_stored_grid_is_a_snapshot_not_an_alias():
    """Mutating the caller's array afterwards must not rewrite the record."""
    grid = np.linspace(0.0, 5.0, 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            _amp_damped(1.0), rho_initial=_RHO_PLUS, t_grid=grid, bootstrap_B=5, seed=1
        )
    original = grid.copy()
    grid[0] = -999.0
    np.testing.assert_array_equal(rep.t_grid, original)


def test_stored_grid_matches_the_curves_it_indexes():
    """The stored grid must be the abscissa of the exported curves, same length."""
    rel = _report("amp_damped", 1.0).relaxation
    n = int(rel.t_grid.size)
    assert n == rel.relative_entropy_curve.size
    assert n == rel.fidelity_curve.size
    assert n == rel.trace_distance_curve.size


def test_grid_provenance_records_an_explicit_caller_grid():
    grid = np.linspace(0.0, 3.0, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = diagnose(
            _amp_damped(1.0), rho_initial=_RHO_PLUS, t_grid=grid,
            bootstrap_B=10, seed=1,
        )
    assert rep.relaxation.t_grid_source == "caller"
    assert rep.relaxation.t_grid_span == pytest.approx(3.0)


def test_explicit_grid_overrides_the_gap():
    """An explicit ``t_grid`` is authoritative; ``gap`` only scales the default."""
    L = _amp_damped(1.0)
    grid = np.linspace(0.0, 4.0, 32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_gap = compute_relaxation_layer(
            L, t_grid=grid, gap=1.0e9, bootstrap_B=5, seed=1
        )
        no_gap = compute_relaxation_layer(L, t_grid=grid, bootstrap_B=5, seed=1)
    assert with_gap.t_grid_source == "caller"
    assert with_gap.beta_D == pytest.approx(no_gap.beta_D, rel=1.0e-12)


def test_direct_caller_without_gap_recomputes_the_same_window():
    """Omitting ``gap`` must not silently fall back to an absolute window.

    :func:`liouscope.diagnose` forwards the D1 gap it already has; a direct
    caller of :func:`compute_relaxation_layer` gets it from the same spectral
    layer, so the two entry points cannot drift apart.
    """
    L = _amp_damped(1.0e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        implicit = compute_relaxation_layer(L, bootstrap_B=5, seed=1)
        explicit = compute_relaxation_layer(L, gap=5.0e-4, bootstrap_B=5, seed=1)
    assert implicit.t_grid_source == "gap_scaled"
    assert implicit.t_grid_span == pytest.approx(explicit.t_grid_span, rel=1.0e-12)
    assert implicit.beta_D == pytest.approx(explicit.beta_D, rel=1.0e-12)


def test_implicit_window_matches_the_certified_d1_gap():
    """The recomputed gap is D1 proper, not "smallest non-zero eigenvalue".

    D1 carries the certified zero-mode tolerance (#112) and the ambiguity rule
    (#113); a local re-derivation would drift from it as those evolve. Pinning
    the window against :func:`compute_spectral_layer` keeps the two bound.
    """
    L = _amp_damped(1.0e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gap = compute_spectral_layer(L).gap
        implicit = compute_relaxation_layer(L, bootstrap_B=5, seed=1)
    assert implicit.t_grid_span == pytest.approx(
        RELAXATION_HORIZON / gap, rel=1.0e-12
    )


# ---------------------------------------------------------------------------
# Multiscale disclosure: what a uniform window cannot do, it must say
# ---------------------------------------------------------------------------


def _two_scale(slow: float, fast: float) -> tuple[np.ndarray, np.ndarray]:
    """Two INDEPENDENT amplitude-damped qubits with separated rates."""
    i2 = np.eye(2, dtype=complex)
    L = build_liouvillian(
        np.zeros((4, 4), dtype=complex),
        [np.sqrt(slow) * np.kron(_SM, i2), np.sqrt(fast) * np.kron(i2, _SM)],
    )
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = np.kron(plus, plus)
    return L, np.outer(psi, psi.conj())


def test_widely_separated_rates_disclose_the_unsampled_fast_mode():
    """The grid cannot resolve six decades of separation, so it must say so.

    Reviewer finding on PR #115 (Codex P1). Confirmed: with rates 1e-6 and 1 the
    fast mode decays to exactly 0.0 within one grid step. The finding was NOT a
    regression — on this same system the previous absolute window put
    ``beta_D_linear`` 3.5e4 relative away from the true gap against 0.58 for the
    gap-scaled one — but it is a real limitation, and silently fitting a
    component that was never sampled is precisely what this project's
    fail-loud convention exists to prevent.
    """
    L, rho0 = _two_scale(1.0e-6, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    assert any(
        issubclass(w.category, UnderResolvedTransientWarning) for w in caught
    ), "no disclosure emitted for a 1e-6 / 1 timescale separation"
    assert rep.samples_per_fast_efolding < MIN_SAMPLES_PER_FAST_EFOLD


@pytest.mark.parametrize("c", [1.0e-4, 1.0, 1.0e4])
def test_single_timescale_systems_do_not_trip_the_disclosure(c):
    """The guard must not cry wolf on ordinary systems, at any rate unit.

    A disclosure that fires on well-resolved input would be filtered away
    wholesale and then miss the case it exists for.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = compute_relaxation_layer(
            _amp_damped(c), rho_initial=_RHO_PLUS, bootstrap_B=5, seed=1
        )
    assert not any(
        issubclass(w.category, UnderResolvedTransientWarning) for w in caught
    ), f"spurious disclosure on a single-timescale system at c={c:g}"
    assert rep.samples_per_fast_efolding > MIN_SAMPLES_PER_FAST_EFOLD


def test_fast_mode_resolution_is_itself_rate_unit_invariant():
    """The resolution measure is a ratio of rates, so it must not move with ``c``.

    If it did, the disclosure would fire on one choice of time unit and not
    another for the same physics — reintroducing, in the guard, exactly the
    defect this module exists to prevent.
    """
    values = [
        compute_relaxation_layer(
            _amp_damped(c), rho_initial=_RHO_PLUS, bootstrap_B=5, seed=1
        ).samples_per_fast_efolding
        for c in (1.0e-4, 1.0, 1.0e4)
    ]
    np.testing.assert_allclose(values, values[0], rtol=1.0e-9)


def test_unresolved_gap_degrades_to_the_documented_legacy_window():
    """A NaN D1 (#113: slow spectrum not resolvable) has no timescale to use.

    The layer must then say so via ``t_grid_source`` rather than scale by a
    meaningless number or silently present the fallback as gap-scaled.
    """
    L = _amp_damped(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(L, gap=float("nan"), bootstrap_B=5, seed=1)
    assert rep.t_grid_source == "legacy_fixed"
    assert rep.t_grid_span == pytest.approx(10.0)
