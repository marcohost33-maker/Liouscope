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
    fastest_decay_rate,
    samples_per_fast_efolding,
)
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.fitting.car1 import neff_car1
from liouscope.fitting.models import initial_guess_m2

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


def test_default_grid_is_uniform_without_a_fast_rate():
    """No second timescale supplied means no second segment: one uniform grid.

    This used to be justified by the AR(1) whitening "presuming a constant
    sample interval" -- a premise since measured to be false and replaced by
    the CAR(1) model (see ``tests/test_car1.py``). The contract survives the
    correction for a different reason: without a ``fast_rate`` the layer has no
    evidence that a second scale exists, and inventing one would be a guess.
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


def _three_scale(slow: float, mid: float, fast: float):
    """Three INDEPENDENT amplitude-damped qubits with separated rates."""
    i2 = np.eye(2, dtype=complex)

    def kron3(a, b, c):
        return np.kron(np.kron(a, b), c)

    L = build_liouvillian(
        np.zeros((8, 8), dtype=complex),
        [
            np.sqrt(slow) * kron3(_SM, i2, i2),
            np.sqrt(mid) * kron3(i2, _SM, i2),
            np.sqrt(fast) * kron3(i2, i2, _SM),
        ],
    )
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = np.kron(np.kron(plus, plus), plus)
    return L, np.outer(psi, psi.conj())


def test_widely_separated_rates_are_resolved_not_merely_disclosed():
    """Reviewer finding on PR #115 (Codex P1): now repaired, not disclosed.

    The previous answer to six decades of separation was an
    ``UnderResolvedTransientWarning`` plus the argument that a non-uniform grid
    is impossible because the GLS layer "whitens with a single AR(1)
    coefficient, which presumes a constant sample interval". That premise is
    false -- it describes the DISCRETE parametrisation, not the noise, whose
    continuous-time form ``exp(-theta dt_k)`` is valid on any grid. With the
    two-scale window and the CAR(1) whitening that goes with it, the fast mode
    is sampled and the model comparison can see it.

    Measured on this system: the fitted second rate is 1.10 against a true 1.0.
    On the previous uniform window the same M2 fit reported 2.17e-05 for it --
    a "successful" two-exponential fit of a component that was never sampled,
    off by a factor 4.6e4. That is the defect this test exists to keep out.
    """
    L, rho0 = _two_scale(1.0e-6, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)

    assert not any(
        issubclass(w.category, UnderResolvedTransientWarning) for w in caught
    ), "still disclosing under-resolution on a separation the grid now resolves"
    assert rep.samples_per_fast_efolding > MIN_SAMPLES_PER_FAST_EFOLD
    assert rep.t_grid_source == "gap_scaled_multiscale"
    assert rep.residual_model == "car1"
    # The grid must still cover the SLOW mode: a fast-resolving window that
    # loses the relaxation would be the strictly-worse absolute window again.
    gap = compute_spectral_layer(L, None).gap
    assert rep.t_grid_span == pytest.approx(RELAXATION_HORIZON / gap, rel=1.0e-9)

    assert rep.aicc_model == "M2", rep.aicc_model
    rates = sorted(
        abs(float(v)) for v in (rep.fits["M2"].params[1], rep.fits["M2"].params[3])
    )
    assert rates[1] == pytest.approx(1.0, rel=0.5), rates
    # The fit must actually have USED the continuous-time whitening; a run that
    # built the two-scale grid and then whitened it with one constant rho would
    # satisfy every assertion above while the residual model was wrong.
    assert np.isfinite(rep.fits["M2"].residual_theta_car1)


def test_neff_on_a_non_uniform_grid_is_the_exact_car1_value():
    """N_eff must come from the same residual model the fit whitened with.

    Geyer's IPS estimator indexes autocorrelation by LAG; on a grid whose step
    varies, lag 1 is not a fixed time separation, so the sequence it sums is
    not an autocorrelation function. Measured against the closed-form ESS on a
    uniform grid, where both are valid, the CAR(1) route is within a factor
    0.97-1.19 across rho in [0, 0.99] while Geyer runs to 3.9x optimistic at
    rho = 0.99 -- so this is not merely a consistency preference.
    """
    L, rho0 = _two_scale(1.0e-6, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    for name, fit in rep.fits.items():
        assert np.isfinite(fit.residual_theta_car1), name
        assert fit.n_eff == pytest.approx(
            neff_car1(rep.t_grid, fit.residual_theta_car1), rel=1.0e-12
        ), name


def test_the_recovered_fast_rate_is_not_the_seed_it_started_from():
    """A parameter that never moved returns its start value and looks perfect.

    This is the failure mode that makes an unresolved mode dangerous rather
    than merely absent: where a component is unsampled the optimiser has no
    gradient on its rate and hands the seed straight back, which reads as a
    confident fit. So the assertion is on the DISPLACEMENT, not on the value:
    ``initial_guess_m2`` seeds the second rate at three times the M0 slope --
    order 3e-06 here -- and the fit must travel five decades away from it.
    """
    L, rho0 = _two_scale(1.0e-6, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    seed = initial_guess_m2(rep.t_grid, rep.relative_entropy_curve)
    fitted = rep.fits["M2"].params
    moved = abs(fitted[3] - seed[3]) / abs(seed[3])
    assert moved > 1.0e3, (seed, fitted, moved)


def test_the_two_scale_window_sharpens_the_d17_linear_rate():
    """The repair has to show up in a REPORTED quantity, not only in the grid.

    ``beta_D_linear`` is the rate D17 compares against the spectral gap, so it
    is the end of the chain this change runs through. Measured relative error
    against the certified D1 gap on this system: 3.5e+04 for the legacy
    absolute window, 0.58 for the uniform gap-scaled one, 0.021 for the
    two-scale one. The ordering is the claim; the tolerances leave room for
    solver noise without admitting either of the other two windows.
    """
    L, rho0 = _two_scale(1.0e-6, 1.0)
    gap = compute_spectral_layer(L, None).gap

    def linear_rate(grid):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = compute_relaxation_layer(
                L, rho_initial=rho0, t_grid=grid, bootstrap_B=5, seed=1
            )
        return abs(rep.beta_D_linear - gap) / gap

    err_absolute = linear_rate(np.linspace(0.0, 10.0, RELAXATION_N_POINTS))
    err_uniform = linear_rate(np.linspace(0.0, 10.0 / gap, RELAXATION_N_POINTS))
    err_two_scale = linear_rate(
        default_relaxation_grid(gap, fast_rate=fastest_decay_rate(L))
    )
    assert err_absolute > 1.0e3, err_absolute
    assert err_uniform > 0.2, err_uniform
    assert err_two_scale < 0.1, err_two_scale
    assert err_two_scale < err_uniform


def test_an_intermediate_timescale_is_still_disclosed():
    """The repair is partial, and the warning must keep saying where.

    A two-scale grid resolves the fastest and the slowest mode by construction;
    a mode BETWEEN them can still fall entirely between the coarse late
    samples. On three independent damped qubits at 1e-6, 1e-3 and 1 the middle
    mode is sampled 0.0019 times per e-folding. Silence here would be the
    original defect moved one scale inward.
    """
    L, rho0 = _three_scale(1.0e-6, 1.0e-3, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    hits = [w for w in caught if issubclass(w.category, UnderResolvedTransientWarning)]
    assert hits, "no disclosure for a middle timescale the two-scale grid misses"
    assert rep.samples_per_fast_efolding < MIN_SAMPLES_PER_FAST_EFOLD
    # It must name the mode it means, not the fastest one.
    assert "0.001" in str(hits[0].message), str(hits[0].message)


def test_a_fine_head_cannot_hide_a_mode_lost_in_the_coarse_tail():
    """The resolution measure must not be fooled by the FIRST step.

    The historical form read ``t[1] - t[0]``, so a grid that starts fine and
    turns coarse reported the fine step and passed. That is not hypothetical
    once the default grid is two-scale: here a rate-1 mode is sampled twice at
    1e-3 spacing and then not at all for ten e-foldings, and the old form would
    have scored it at 1000 samples per e-folding.
    """
    L = _amp_damped(1.0)  # fastest decay rate 1
    grid = np.concatenate([[0.0, 1.0e-3], np.linspace(0.002, 1.0e4, 78)])
    # What the historical ``1 / (r_max * (t[1] - t[0]))`` form would have read:
    assert 1.0 / float(grid[1] - grid[0]) == pytest.approx(1000.0, rel=1.0e-9)
    # What the grid actually does to a rate-1 mode: two samples at 1e-3 spacing
    # and then a 128-unit hole, i.e. 128 e-foldings unobserved.
    assert samples_per_fast_efolding(L, grid) < MIN_SAMPLES_PER_FAST_EFOLD


def test_late_starting_grid_counts_its_unsampled_lead_in():
    """A fine step does not help if the mode is gone before the first sample.

    Reviewer finding on PR #115 (Codex, second round) against the guard added
    in the first. ``diagnose`` permits any non-negative start, and on
    ``linspace(100, 101, 101)`` a rate-1 mode is sampled 100x per e-folding
    while its amplitude at the first sample is ``e^-100``. Measured before the
    fix: no warning, "100 samples per fast e-folding" — and the entire
    relative-entropy curve identically zero, with the fit still returning
    ``beta_D = 1.0``. The blind interval from ``t=0`` is now counted as what it
    is: unsampled, and here the largest gap in the grid.
    """
    L = _amp_damped(1.0)  # rate 1
    grid = np.linspace(100.0, 101.0, 101)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = compute_relaxation_layer(
            L, rho_initial=_RHO_PLUS, t_grid=grid, bootstrap_B=5, seed=1
        )
    assert any(
        issubclass(w.category, UnderResolvedTransientWarning) for w in caught
    ), "a grid starting 100 e-foldings late reported no under-resolution"
    # 1 / (r_max * t[0]) = 1/100, NOT 1 / (r_max * dt) = 100.
    assert rep.samples_per_fast_efolding == pytest.approx(0.01, rel=1.0e-6)


def test_lead_in_never_penalises_a_grid_that_starts_at_zero():
    """The lead-in term must be inert on the default path.

    ``max(dt, t[0])`` is exactly ``dt`` whenever ``t[0] == 0``, so the guard's
    behaviour on every default-grid run is unchanged by the lead-in fix.
    """
    L = _amp_damped(1.0)
    from_zero = np.linspace(0.0, 20.0, 80)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=_RHO_PLUS, t_grid=from_zero, bootstrap_B=5, seed=1
        )
    dt = float(from_zero[1] - from_zero[0])
    assert rep.samples_per_fast_efolding == pytest.approx(1.0 / dt, rel=1.0e-9)


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


# ---------------------------------------------------------------------------
# Positive controls: nothing that already worked may move
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("c", [1.0e-4, 1.0, 1.0e4])
def test_single_timescale_default_window_is_untouched(c):
    """The regime that was already fine must be bit-identical, not merely close.

    The two-scale branch may only fire where the uniform grid FAILS to resolve
    the fast mode. If it also fired on ordinary systems it would move every
    anchor, and "the fix did not break anything" would be a matter of
    tolerances rather than of identity.
    """
    L = _amp_damped(c)
    gap = compute_spectral_layer(L, None).gap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=_RHO_PLUS, bootstrap_B=5, seed=1
        )
    np.testing.assert_array_equal(
        rep.t_grid, np.linspace(0.0, RELAXATION_HORIZON / gap, RELAXATION_N_POINTS)
    )
    assert rep.t_grid_source == "gap_scaled"
    assert rep.residual_model == "ar1"
    for fit in rep.fits.values():
        assert np.isnan(fit.residual_theta_car1)


def test_a_fast_rate_the_uniform_grid_already_resolves_changes_nothing():
    """Supplying ``fast_rate`` is not by itself a request for a new grid.

    The uniform window reaches a spread of ``(n_points - 1) / horizon = 7.9``
    between the slowest and the fastest mode; below that it already samples the
    fast mode at least once per e-folding and must be returned untouched.
    """
    for ratio in (1.0, 4.0, 7.8):
        np.testing.assert_array_equal(
            default_relaxation_grid(1.0, fast_rate=ratio),
            default_relaxation_grid(1.0),
        )


def test_the_switch_happens_exactly_where_the_uniform_grid_stops_resolving():
    """The trigger is the resolution criterion itself, not a separate constant.

    Pinning both sides of the boundary is what makes this a contract rather
    than an observation: a change that widened the trigger would keep the
    "unchanged" assertions green while quietly moving every borderline system
    onto the new path.
    """
    reach = (RELAXATION_N_POINTS - 1) / RELAXATION_HORIZON  # 7.9
    np.testing.assert_array_equal(
        default_relaxation_grid(1.0, fast_rate=reach * 0.99),
        default_relaxation_grid(1.0),
    )
    switched = default_relaxation_grid(1.0, fast_rate=reach * 1.01)
    assert not np.array_equal(switched, default_relaxation_grid(1.0))
    assert not np.allclose(np.diff(switched), np.diff(switched)[0])


def test_two_scale_grid_appears_only_past_the_uniform_grid_s_reach():
    """And when it does appear, it resolves the fast mode by construction."""
    grid = default_relaxation_grid(1.0e-6, fast_rate=1.0)
    assert not np.allclose(np.diff(grid), np.diff(grid)[0])
    assert grid[0] == 0.0
    assert grid.size == RELAXATION_N_POINTS
    assert np.all(np.diff(grid) > 0.0)
    assert grid[-1] == pytest.approx(RELAXATION_HORIZON / 1.0e-6, rel=1.0e-12)
    # n_fast points across HORIZON/fast e-foldings -> n_fast / HORIZON samples
    # per fast e-folding, INDEPENDENT of the rates.
    expected = (RELAXATION_N_POINTS // 2) / RELAXATION_HORIZON
    assert 1.0 / (1.0 * float(np.diff(grid)[0])) == pytest.approx(expected, rel=1.0e-9)


@pytest.mark.parametrize("bad_fast", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_an_unusable_fast_rate_falls_back_to_the_uniform_window(bad_fast):
    """Fail-closed: no fast scale is not a licence to invent one."""
    np.testing.assert_array_equal(
        default_relaxation_grid(1.0e-6, fast_rate=bad_fast),
        default_relaxation_grid(1.0e-6),
    )


def test_two_scale_resolution_is_rate_unit_invariant():
    """The whole point of the module: a pure change of time unit changes nothing."""
    values = []
    for c in (1.0e-3, 1.0, 1.0e3):
        L, rho0 = _two_scale(1.0e-6 * c, 1.0 * c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rep = compute_relaxation_layer(
                L, rho_initial=rho0, bootstrap_B=5, seed=1
            )
        assert rep.t_grid_source == "gap_scaled_multiscale"
        values.append(rep.samples_per_fast_efolding)
    np.testing.assert_allclose(values, values[0], rtol=1.0e-6)


def test_fastest_decay_rate_is_fail_closed_without_a_decaying_mode():
    """No positive rate -> NaN, so the grid builder keeps its uniform default."""
    assert np.isnan(fastest_decay_rate(np.zeros((4, 4))))
    assert fastest_decay_rate(_amp_damped(1.0)) == pytest.approx(1.0, rel=1.0e-9)
