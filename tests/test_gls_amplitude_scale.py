"""Regression tests for observable-amplitude invariance in the GLS optimiser.

Issue #124: the mathematical fit is unchanged by ``y -> c*y`` for ``c > 0``,
but feeding the raw residuals to SciPy lets the gradient termination criterion
accept the initial seed when ``c`` is tiny.  These tests pin both directions:
the tiny-amplitude counterexample must fit, and the ordinary-scale positive
control must remain unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.fitting.gls import fit_gls_ar1

_TRUE_RATE = 1.3
_SEED_RATE = 0.2
_T = np.linspace(0.0, 5.0, 64)


def _fit_exponential(scale: float):
    def model(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        return scale * np.exp(-params[0] * t)

    y = scale * np.exp(-_TRUE_RATE * _T)
    return fit_gls_ar1(
        model,
        _T,
        y,
        np.array([_SEED_RATE]),
        bounds=(np.array([0.0]), np.array([5.0])),
        n_iters=1,
    )


def test_tiny_amplitude_cannot_turn_the_seed_into_a_measurement() -> None:
    """The exact #124 counterexample must move away from its initial seed."""
    tiny = _fit_exponential(1.0e-40)

    assert tiny.success
    assert not tiny.degenerate
    assert tiny.params[0] == pytest.approx(_TRUE_RATE, rel=1.0e-7, abs=1.0e-10)
    assert abs(tiny.params[0] - _SEED_RATE) > 0.5


def test_amplitude_rescaling_preserves_the_fitted_rate() -> None:
    """Positive control: ordinary and tiny amplitudes represent the same fit."""
    ordinary = _fit_exponential(1.0)
    tiny = _fit_exponential(1.0e-40)

    assert ordinary.success and tiny.success
    assert ordinary.params[0] == pytest.approx(_TRUE_RATE, rel=1.0e-7, abs=1.0e-10)
    assert tiny.params[0] == pytest.approx(ordinary.params[0], rel=1.0e-9, abs=1.0e-12)


@pytest.mark.parametrize("scale", [1.0e-160, 1.0e-310])
def test_unrepresentable_rescaling_falls_back_instead_of_dropping_the_fit(
    scale: float,
) -> None:
    """PR #134: the #124 rescaling must not withhold a fit the raw problem makes.

    With a FREE amplitude parameter the rescaled Jacobian and cost carry the
    same ``1/scale`` factor as the residual. Measured on the merge of #124 into
    main: ``_fit_with_model("M0")`` overflowed inside SciPy at these scales and
    returned ``success=False`` / ``aicc=inf``, while main fitted rate 1.3. The
    Jacobian-level overflow is the case the residual-level check cannot see, so
    it is pinned here separately from the #147 subnormal fixture.

    The first scale was 1e-150 until the solver scaled its variables by the
    Jacobian column norms (``x_scale="jac"``): 1e-150 is representable since
    then and no longer reaches the fallback, while 1e-160 still does -- and
    only through the floating-point-event arm (measured: with ``events``
    removed from the check, 1e-155 .. 1e-305 report ``success=False``).
    """
    import warnings

    from liouscope.diagnostics import relaxation as relaxation_mod

    y = scale * np.exp(-_TRUE_RATE * _T)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_result, _ = relaxation_mod._fit_with_model("M0", _T, y)
    assert any(
        "not representable as float64 at the" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]
    assert fit_result.success
    assert np.isfinite(fit_result.aicc)
    assert fit_result.params[1] == pytest.approx(_TRUE_RATE, rel=1.0e-7)


# --------------------------------------------------------------------------
# PR #134 review (Equalita NO-GO for 7ad4829): one discriminating test per
# part of the representability detector.
# --------------------------------------------------------------------------

_FALLBACK = "not representable as float64 at the"
_NOISE = np.random.default_rng(20260911).standard_normal(_T.size)


def _free_amplitude(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    return params[0] * np.exp(-params[1] * t)


def _recorded(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [str(w.message) for w in caught]


@pytest.mark.parametrize("scale", [1.0e200, 1.0e300])
def test_large_amplitude_fails_closed_without_leaking_the_private_exception(
    scale: float,
) -> None:
    """The raw path is main's problem verbatim: ``success=False``, no raise.

    Measured on 7ad4829: the residual detector also ran on the RAW residuals,
    whose whitening overflows here although they are finite, and its private
    ``ArithmeticError`` escaped both public entry points (20/20 cases).
    """
    from liouscope.diagnostics import relaxation as relaxation_mod

    y = scale * np.exp(-_TRUE_RATE * _T)
    fit, _ = _recorded(fit_gls_ar1, _free_amplitude, _T, y, np.array([scale, 0.2]))
    assert not fit.success
    (fit_result, _), _ = _recorded(
        relaxation_mod._fit_with_model, "M0", _T, y * (1.0 + 1.0e-3 * _NOISE)
    )
    assert not fit_result.success
    assert np.isinf(fit_result.aicc)


def test_benign_model_float_event_does_not_trigger_the_fallback() -> None:
    """A 0/0 inside the MODEL belongs to the raw problem, not to the rescaling.

    ``sin(t)/t`` at ``t = 0`` raises 'invalid' and is replaced by ``np.where``;
    the model output is finite. When that event was counted, every fit of this
    model fell back and returned the seed rate 0.2 as converged at 1e-40.
    """

    def run(scale: float) -> tuple[float, list[str]]:
        def model(t: np.ndarray, params: np.ndarray) -> np.ndarray:
            return scale * np.exp(-params[0] * t) * np.where(t > 0, np.sin(t) / t, 1.0)

        with np.errstate(invalid="ignore"):  # building the data is not under test
            y = model(_T, np.array([_TRUE_RATE])) * (1.0 + 1.0e-3 * _NOISE)
        fit, messages = _recorded(
            fit_gls_ar1, model, _T, y, np.array([_SEED_RATE]),
            bounds=(np.array([0.0]), np.array([5.0])), n_iters=1,
        )
        assert fit.success
        return float(fit.params[0]), messages

    ordinary, _ = run(2.0)
    tiny, messages = run(1.0e-40)
    assert not any(_FALLBACK in m for m in messages), messages
    assert tiny == pytest.approx(ordinary, rel=1.0e-9)
    assert abs(tiny - _SEED_RATE) > 0.5


def test_nonfinite_rescaled_result_without_float_events_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-finite arm: a NaN Jacobian set by assignment raises no FP event."""
    from scipy.optimize import least_squares as real


    calls = {"n": 0}

    def first_call_poisoned(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            res.jac = np.full_like(res.jac, np.nan)
        return res

    monkeypatch.setattr("liouscope.fitting.gls.least_squares", first_call_poisoned)
    y = 1.0e-40 * np.exp(-_TRUE_RATE * _T) * (1.0 + 1.0e-3 * _NOISE)
    fit, messages = _recorded(
        fit_gls_ar1, lambda t, p: 1.0e-40 * np.exp(-p[0] * t), _T, y,
        np.array([_SEED_RATE]), bounds=(np.array([0.0]), np.array([5.0])), n_iters=1,
    )
    assert any(_FALLBACK in m for m in messages), messages
    assert calls["n"] == 2


def test_exception_without_float_events_is_not_rerouted() -> None:
    """A model error in the rescaled solve is a failed fit, not a scale problem."""

    def failing(t: np.ndarray, params: np.ndarray) -> np.ndarray:
        if params[0] != _SEED_RATE:
            raise RuntimeError("model refuses this rate")
        return 1.0e-40 * np.exp(-params[0] * t)

    y = 1.0e-40 * np.exp(-_TRUE_RATE * _T) * (1.0 + 1.0e-3 * _NOISE)
    fit, messages = _recorded(fit_gls_ar1, failing, _T, y, np.array([_SEED_RATE]), n_iters=1)
    assert not fit.success
    assert not any(_FALLBACK in m for m in messages), messages


def test_finite_nonconverged_rescaled_solve_is_not_retried_raw() -> None:
    """Retrying a finite non-converged solve raw would re-admit the #124 seed."""
    y = 1.0e-40 * np.exp(-_TRUE_RATE * _T) * (1.0 + 1.0e-3 * _NOISE)
    fit, messages = _recorded(
        fit_gls_ar1, lambda t, p: 1.0e-40 * np.exp(-p[0] * t), _T, y,
        np.array([_SEED_RATE]), bounds=(np.array([0.0]), np.array([5.0])),
        n_iters=1, max_nfev=1,
    )
    assert not fit.success
    assert not any(_FALLBACK in m for m in messages), messages


# --------------------------------------------------------------------------
# PR #134 round 3 (Codex P1): the AR(1) rho between Cochrane-Orcutt iterations
# was estimated from UNSCALED residuals, whose dot products underflow to 0
# below ~1e-162 and overflow to NaN above ~1e154.
# --------------------------------------------------------------------------


def _ar1_curve(scale: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 5.0, 80)
    rng = np.random.default_rng(20260911)
    e = np.zeros(t.size)
    nu = rng.standard_normal(t.size)
    for i in range(t.size):
        e[i] = (0.6 * e[i - 1] if i else 0.0) + nu[i]
    return t, scale * (np.exp(-_TRUE_RATE * t) + 1.0e-3 * e)


@pytest.mark.parametrize("scale", [1.0e-170, 1.0e160])
def test_ar1_rho_between_iterations_is_amplitude_invariant(scale: float) -> None:
    """Amplitude-equivalent correlated curves give the same rho, rate, success.

    Measured on 9a623e8: rho 0.012987 (= the corrected floor 1/(n-3), i.e. an
    underflowed rho_hat of 0) at 1e-170, 0.44637 at 1e0, and NaN with
    ``success=False`` at 1e160.
    """

    def fit(s: float):  # type: ignore[no-untyped-def]
        t, y = _ar1_curve(s)
        out, _ = _recorded(
            fit_gls_ar1, lambda tt, p: s * p[0] * np.exp(-p[1] * tt), t, y,
            np.array([1.0, 0.8]), n_iters=3,
        )
        return out

    ref = fit(1.0)
    other = fit(scale)
    assert ref.success and np.isfinite(ref.rho_ar1) and ref.rho_ar1 > 0.3
    assert other.success == ref.success
    assert other.rho_ar1 == pytest.approx(ref.rho_ar1, rel=1.0e-8)
    assert other.params[1] == pytest.approx(ref.params[1], rel=1.0e-8)


def test_ar1_correlation_is_bit_identical_under_power_of_two_rescaling() -> None:
    """The normalisation must not move rho where the old arithmetic was in range."""
    from liouscope.fitting.neff import ar1_correlation

    _, y = _ar1_curve(1.0)
    base = ar1_correlation(y)
    for k in (-900, -540, 500, 1000):
        assert ar1_correlation(np.ldexp(y, k)) == base, k


# --------------------------------------------------------------------------
# PR #134 round 3 (Equalita follow-up on 9a623e8)
# --------------------------------------------------------------------------


def _sinc_model(t: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Same values at every parameter; the 0/0 at t = 0 is only RAISED at the
    # seed, i.e. inside the optimiser's first evaluation. The post-fit model
    # evaluation (at ~1.3) is event-free, so it cannot mask what the solve did.
    if params[0] == _SEED_RATE:
        sinc = np.where(t > 0, np.sin(t) / t, 1.0)
    else:
        sinc = np.where(t > 0, np.sin(t) / np.where(t > 0, t, 1.0), 1.0)
    return 1.0e-40 * np.exp(-params[0] * t) * sinc


def test_model_runs_under_the_callers_floating_point_policy() -> None:
    """The detector's errstate must not replace the caller's for the MODEL.

    A caller that asks for ``invalid="raise"`` gets the model's 0/0 during the
    solve as a ``FloatingPointError``, exactly as on main. Evaluating the
    model under the detector's own errstate (ignore or call) swallows it.
    """
    y = _sinc_model(_T, np.array([_TRUE_RATE]))  # event-free away from the seed
    with np.errstate(invalid="raise"):
        with pytest.raises(FloatingPointError):
            fit_gls_ar1(
                _sinc_model, _T, y, np.array([_SEED_RATE]),
                bounds=(np.array([0.0]), np.array([5.0])), n_iters=1,
            )


def test_fallback_warning_has_a_dedicated_filterable_category() -> None:
    """``-W error`` users can silence exactly the fallback notice by class."""
    import warnings

    from liouscope.fitting.gls import AmplitudeRescalingFallbackWarning

    assert issubclass(AmplitudeRescalingFallbackWarning, RuntimeWarning)
    scale = 1.0e200
    y = scale * np.exp(-_TRUE_RATE * _T)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_gls_ar1(_free_amplitude, _T, y, np.array([scale, 0.2]))
    fallback = [w for w in caught if _FALLBACK in str(w.message)]
    assert fallback, [str(w.message) for w in caught]
    assert all(w.category is AmplitudeRescalingFallbackWarning for w in fallback)
    assert not fit.success


# --------------------------------------------------------------------------
# PR #134 round 4 (Equalita on 28c7db2): the library model M0 with a FREE
# amplitude, and the ldexp normalisation under a strict underflow policy.
# --------------------------------------------------------------------------


def _m0_fit(scale: float):  # type: ignore[no-untyped-def]
    from liouscope.diagnostics import relaxation as relaxation_mod

    y = scale * (np.exp(-_TRUE_RATE * _T) + 1.0e-3 * _NOISE)
    (fit_result, _), messages = _recorded(relaxation_mod._fit_with_model, "M0", _T, y)
    return fit_result, messages


@pytest.mark.parametrize("scale", [1.0e-40, 1.0e40])
def test_free_amplitude_m0_rate_is_amplitude_invariant(scale: float) -> None:
    """``_fit_with_model("M0")`` fits the same rate at 1e-40, 1 and 1e+40.

    Measured on 28c7db2: at 1e-40 and 1e+40 the fit returned its seed rate
    1.3245053905 with ``success=True`` and no warning, while scale 1 fits
    1.2993514754 -- the Jacobian column of the free amplitude outweighed the
    rate column by the curve's scale.
    """
    from liouscope.fitting.models import initial_guess_m0

    ref, _ = _m0_fit(1.0)
    other, messages = _m0_fit(scale)
    seed_rate = float(initial_guess_m0(_T, scale * (np.exp(-_TRUE_RATE * _T) + 1.0e-3 * _NOISE))[1])
    assert ref.success and other.success
    assert not any(_FALLBACK in m for m in messages), messages
    assert other.params[1] == pytest.approx(ref.params[1], rel=1.0e-8)
    assert other.params[0] == pytest.approx(scale * ref.params[0], rel=1.0e-8)
    # The seed is 0.025 away; the invariance above must not be the seed twice.
    assert abs(other.params[1] - seed_rate) > 1.0e-2


def _ar1_or_event(r: np.ndarray) -> float | str:
    """``ar1_correlation``, with a floating-point exception as a comparable value.

    A raised ``FloatingPointError`` is the defect under test, so it must
    surface as a failed comparison, not as a crash of the test itself.
    """
    from liouscope.fitting.neff import ar1_correlation

    try:
        return ar1_correlation(np.asarray(r, dtype=float))
    except FloatingPointError as exc:
        return f"FloatingPointError: {exc}"


def _ar1_reference(r: np.ndarray) -> float:
    """``ar1_correlation`` as it stood on main, before the normalisation."""
    x = np.asarray(r, dtype=float) - float(np.mean(r))
    num = float(np.dot(x[:-1], x[1:]))
    den = float(np.dot(x, x))
    return 0.0 if den == 0.0 else float(num / den)


def test_ar1_normalisation_does_not_raise_under_a_strict_underflow_policy() -> None:
    """The ldexp that normalises the residuals is this function's own scaling.

    Measured on 28c7db2: under ``np.errstate(under="raise")``,
    ``ar1_correlation([1e150, 1e-200])`` raised ``FloatingPointError`` in
    ``ldexp``; main returns -0.5 there without any floating-point event.
    """

    for series in ([1.0e150, 1.0e-200], [1.0e150, -1.0e150, 1.0e-200], [1.0, 1.0e-320, -1.0, 0.5]):
        # Main's own arithmetic may underflow harmlessly too, and whether it
        # SIGNALS that is BLAS-dependent (measured: OpenBLAS on ubuntu raises
        # in ``dot`` for the second series, the Windows build does not), so
        # the reference value is taken with underflow ignored. The function
        # under test runs under the strict policy.
        with np.errstate(all="raise", under="ignore"):
            expected = _ar1_reference(np.asarray(series))
        with np.errstate(all="raise"):
            got = _ar1_or_event(np.asarray(series))
        assert got == expected, series
    with np.errstate(under="raise"):
        assert _ar1_or_event(np.array([1.0e150, 1.0e-200])) == -0.5


def test_ar1_normalisation_matches_main_wherever_main_was_event_free() -> None:
    """Bit-identical to main on every series main computes in range.

    Magnitudes span 1e-320 .. 1e300 so that the normalisation underflows
    entries in a large share of cases; the count is asserted so that this
    property cannot pass vacuously on series that never reach the fixed path.

    "In range" is decided by MAGNITUDE, not by floating-point flags: whether
    ``np.dot`` signals under/overflow depends on the NumPy/BLAS build
    (measured in CI: NumPy 2.2.6 and 2.0.2 returned main's overflowed NaN
    without raising; 2.4 raises). The mean and the subtraction are ufuncs,
    which signal on every build; the dot products are in range when every
    non-zero deviation lies in [1e-150, 1e150] (n <= 11 products of at most
    1e300 each, none below 1e-300).
    """

    rng = np.random.default_rng(20260930)
    compared = underflowing = 0
    for _ in range(3000):
        n = int(rng.integers(2, 12))
        x = rng.standard_normal(n) * 10.0 ** rng.integers(-320, 300, size=n).astype(float)
        try:
            with np.errstate(all="raise"):
                deviation = np.abs(x - float(np.mean(x)))
        except FloatingPointError:
            continue
        nonzero = deviation[deviation != 0.0]
        if nonzero.size == 0 or nonzero.max() > 1.0e150 or nonzero.min() < 1.0e-150:
            continue
        expected = _ar1_reference(x)
        peak = float(np.max(np.abs(x)))
        try:
            with np.errstate(under="raise"):
                np.ldexp(x, -int(np.frexp(peak)[1]))
        except FloatingPointError:
            underflowing += 1
        with np.errstate(all="raise"):
            assert _ar1_or_event(x) == expected, x
        compared += 1
    assert compared >= 300 and underflowing >= 100, (compared, underflowing)


def test_ar1_normalisation_keeps_nan_fail_closed() -> None:
    """A NaN residual still yields NaN -- never a finite rho -- under any policy."""
    from liouscope.fitting.neff import ar1_correlation

    with np.errstate(under="raise"):
        assert np.isnan(ar1_correlation(np.array([1.0e-200, np.nan, 1.0e150])))
