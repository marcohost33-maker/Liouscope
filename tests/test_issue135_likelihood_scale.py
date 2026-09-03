"""Issue #135: scale-safe Gaussian likelihood and degenerate RSS semantics."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from liouscope.diagnostics import relaxation as relaxation_mod
from liouscope.fitting.aicc import aicc, choose_model, gaussian_log_likelihood
from liouscope.fitting.bootstrap import parametric_bootstrap
from liouscope.fitting.gls import GLSFitOutput, fit_gls_ar1
from liouscope.fitting.models import M0
from liouscope.io.export import _to_jsonable
from liouscope.numerics.norms import scaled_log_sum_squares


def test_scaled_log_rss_spans_underflow_and_overflow_regimes():
    tiny = np.array([1.0e-320, -2.0e-320])
    huge = np.array([1.0e308, -1.0e308])
    log_tiny = scaled_log_sum_squares(tiny)
    log_huge = scaled_log_sum_squares(huge)
    assert np.isfinite(log_tiny)
    assert np.isfinite(log_huge)
    assert log_tiny == pytest.approx(math.log(5.0) + 2.0 * math.log(1.0e-320), rel=2e-5)
    assert log_huge == pytest.approx(math.log(2.0) + 2.0 * math.log(1.0e308), rel=1e-14)
    assert scaled_log_sum_squares(np.zeros(4)) == float("-inf")


def test_profile_loglikelihood_is_scale_covariant_and_delta_invariant():
    x = np.linspace(0.2, 1.2, 64)
    r0 = 0.7 + x + 0.03 * np.sin(5.0 * x)
    r1 = 0.9 + 1.1 * x - 0.02 * np.cos(3.0 * x)
    reference_delta = None
    reference_winner = None
    for scale in (1.0e-150, 1.0e-40, 1.0, 1.0e40, 1.0e150):
        ll0 = gaussian_log_likelihood(scale * r0)
        ll1 = gaussian_log_likelihood(scale * r1)
        assert np.isfinite(ll0) and np.isfinite(ll1)
        delta = ll1 - ll0
        scores = {
            "M0": aicc(ll0, k=2, n_eff=64.0),
            "M1": aicc(ll1, k=3, n_eff=64.0),
        }
        winner = choose_model(scores)
        if reference_delta is None:
            reference_delta = delta
            reference_winner = winner
        else:
            assert delta == pytest.approx(reference_delta, rel=1e-11, abs=1e-10)
            assert winner == reference_winner


def test_profile_loglikelihood_handles_true_rss_above_float_range():
    ll = gaussian_log_likelihood(np.array([1.0e308, -1.0e308]))
    assert np.isfinite(ll)


def test_zero_rss_unknown_sigma_is_unavailable_but_known_sigma_is_valid():
    residuals = np.zeros(8)
    assert np.isnan(gaussian_log_likelihood(residuals))
    ll = gaussian_log_likelihood(residuals, sigma=2.0)
    expected = -0.5 * residuals.size * math.log(2.0 * math.pi * 4.0)
    assert ll == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_gls_exact_fit_marks_likelihood_degenerate_and_bootstrap_refuses():
    t = np.linspace(0.0, 4.0, 48)
    p0 = np.array([1.25, 0.6])
    y = M0(t, p0)
    with pytest.warns(RuntimeWarning, match="likelihood/AICc/CI evidence is unavailable"):
        fit = fit_gls_ar1(M0, t, y, p0, n_iters=1)
    assert not fit.success
    assert fit.likelihood_degenerate
    assert not fit.degenerate
    assert np.isnan(fit.sigma)
    assert np.isnan(fit.log_likelihood)
    # The repository treats unexpected warnings as errors. Here the warning is
    # part of the intended public contract: the base fit first reports why its
    # likelihood evidence is unusable, then bootstrap refuses the non-estimate.
    with pytest.warns(RuntimeWarning, match="likelihood/AICc/CI evidence is unavailable"):
        with pytest.raises(RuntimeError, match="likelihood scale is degenerate"):
            parametric_bootstrap(M0, t, y, p0, B=4)


def test_ordinary_noisy_gls_positive_control_is_not_likelihood_degenerate(rng):
    t = np.linspace(0.0, 4.0, 80)
    p = np.array([1.25, 0.6])
    y = M0(t, p) + 1.0e-3 * rng.standard_normal(t.size)
    fit = fit_gls_ar1(M0, t, y, p, n_iters=1)
    assert fit.success
    assert not fit.likelihood_degenerate
    assert np.isfinite(fit.sigma) and fit.sigma > 0.0
    assert np.isfinite(fit.log_likelihood)


def test_likelihood_degenerate_state_reaches_fitresult_and_is_nonselectable(monkeypatch):
    t = np.linspace(0.0, 1.0, 16)
    y = np.exp(-t)
    fake = GLSFitOutput(
        params=np.array([1.0, 1.0]),
        residuals=np.zeros_like(y),
        rho_ar1=0.0,
        sigma=float("nan"),
        log_likelihood=float("nan"),
        success=False,
        likelihood_degenerate=True,
    )
    monkeypatch.setattr(relaxation_mod, "fit_gls_ar1", lambda *args, **kwargs: fake)
    fit_result, _ = relaxation_mod._fit_with_model("M0", t, y)
    assert not fit_result.success
    assert fit_result.likelihood_degenerate
    assert np.isinf(fit_result.aicc)


# --------------------------------------------------------------------------
# PR #147 round-1 review: an unrepresentable MLE scale withheld the whole FIT,
# not just the scale-dependent evidence.
# --------------------------------------------------------------------------

#: Smallest positive float64. One such residual among an otherwise-zero series
#: keeps ``log_rss`` finite while ``exp(0.5 * (log_rss - log n))`` underflows:
#: the profile likelihood is computable, its scale is not.
_MIN_SUBNORMAL = 5.0e-324


def _constant_model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.full(t.shape, float(p[0]))


def _underflowing_scale_case() -> tuple[np.ndarray, np.ndarray]:
    n = 64
    t = np.linspace(0.0, 5.0, n)
    y = np.zeros(n)
    y[7] = _MIN_SUBNORMAL
    return t, y


def test_the_underflowing_case_really_has_a_finite_log_likelihood() -> None:
    """Positive control: the fixture must exercise the finding, not a NaN RSS.

    Without this the two assertions below could pass on a case where nothing
    was computable in the first place.
    """
    _, y = _underflowing_scale_case()
    log_rss = scaled_log_sum_squares(y)
    assert math.isfinite(log_rss), log_rss
    log_sigma = 0.5 * (log_rss - math.log(y.size))
    assert math.isfinite(log_sigma), log_sigma
    assert math.exp(log_sigma) == 0.0, (
        "the fixture no longer underflows; the finding cannot be reached"
    )
    assert math.isfinite(gaussian_log_likelihood(y))


def test_unrepresentable_scale_keeps_the_fit_selectable() -> None:
    """The fit stays an AICc candidate; only ``sigma`` is withheld.

    Measured before the repair: ``success=False``, ``log_likelihood=nan`` and
    ``likelihood_degenerate=True`` -- so ``_fit_with_model`` scored the model
    ``inf`` and dropped it from selection. That reintroduces an ABSOLUTE scale
    boundary into model selection, which is the boundary issue #135 removed:
    the same curve in different rate units is or is not a candidate.
    """
    t, y = _underflowing_scale_case()
    # Recorded rather than ``pytest.warns``: a missing warning must fail this
    # test at an ASSERTION, so that removing the guard is attributable. With
    # ``pytest.warns`` the test dies inside pytest's own context manager, which
    # a mutation run cannot tell apart from an incidental crash.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_gls_ar1(_constant_model, t, y, np.array([0.0]))
    assert any(
        issubclass(w.category, RuntimeWarning)
        and "scale is not representable" in str(w.message)
        for w in caught
    ), [str(w.message) for w in caught]
    assert fit.success, "the fit was withheld because its SCALE could not be built"
    assert math.isfinite(fit.log_likelihood), fit.log_likelihood
    assert not fit.likelihood_degenerate
    assert fit.scale_unavailable
    assert math.isnan(fit.sigma), (
        f"sigma is {fit.sigma!r}; 0.0 would make _ar1_resample draw identical "
        "replicates and produce a zero-width interval"
    )


def test_unrepresentable_scale_still_withholds_the_bootstrap() -> None:
    """Fail-closed control: selectable must not mean an interval was invented."""
    t, y = _underflowing_scale_case()
    with pytest.warns(RuntimeWarning):
        with pytest.raises(Exception) as excinfo:
            parametric_bootstrap(
                _constant_model, t, y, np.array([0.0]), B=4,
                rng=np.random.default_rng(0),
            )
    assert excinfo.type is RuntimeError, excinfo.type
    assert "no representable residual scale" in str(excinfo.value)


def test_ordinary_fit_is_untouched_by_the_scale_repair() -> None:
    """Over-correction control: a normal fit keeps a finite positive sigma."""
    rng = np.random.default_rng(20260903)
    t = np.linspace(0.0, 4.0, 64)
    p = np.array([1.0, 0.8])
    y = M0(t, p) + 1.0e-3 * rng.standard_normal(t.size)
    fit = fit_gls_ar1(M0, t, y, p, n_iters=1)
    assert not fit.scale_unavailable
    assert np.isfinite(fit.sigma) and fit.sigma > 0.0


# --------------------------------------------------------------------------
# PR #147 round-2 review
# --------------------------------------------------------------------------


def test_scale_unavailable_reaches_fitresult() -> None:
    """The reason a CI was withheld must survive into the persisted report.

    ``bca_ci_beta = (nan, nan)`` is what EVERY bootstrap or jackknife failure
    produces. Without this flag on ``FitResult`` an unrepresentable residual
    scale -- a fit that is otherwise successful and selectable -- was
    indistinguishable from a non-converged resample once the warning stream
    was gone.
    """
    t, y = _underflowing_scale_case()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_result, _ = relaxation_mod._fit_with_model("M0", t, y)
    assert any(
        "scale is not representable" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]
    assert fit_result.scale_unavailable
    # The fit stays a candidate: the flag records why the INTERVAL is missing,
    # it does not withhold the estimate (round-1 review contract).
    assert fit_result.success
    assert math.isfinite(fit_result.log_likelihood)
    # The field is serialised, so an audit artefact carries the reason.
    assert _to_jsonable(fit_result)["scale_unavailable"] is True


def test_ordinary_fit_reaches_fitresult_without_the_flag() -> None:
    """Over-correction control: a normal fit must not be labelled scaleless."""
    rng = np.random.default_rng(20260903)
    t = np.linspace(0.0, 4.0, 64)
    y = M0(t, np.array([1.0, 0.8])) + 1.0e-3 * rng.standard_normal(t.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_result, _ = relaxation_mod._fit_with_model("M0", t, y)
    assert not fit_result.scale_unavailable


def test_explicit_sigma_likelihood_uses_the_full_representable_rss_range() -> None:
    """``-0.5 * RSS`` is representable up to an RSS of ``2 * float64.max``.

    The guard compared the standardised RSS against ``float64.max`` and
    returned ``-inf`` for the octave above it, although the value the formula
    actually forms -- the HALVED RSS -- is perfectly finite there. Dropping a
    model from selection on that boundary is the absolute-scale dependence
    issue #135 exists to remove.
    """
    residual = np.array([1.4e154])
    log_lik = gaussian_log_likelihood(residual, sigma=1.0)
    assert math.isfinite(log_lik), log_lik
    expected = -0.5 * math.log(2.0 * math.pi) - math.exp(
        2.0 * math.log(1.4e154) - math.log(2.0)
    )
    assert log_lik == pytest.approx(expected, rel=1e-12)

    # POSITIVE CONTROL that the fixture really sits in the disputed octave:
    # the raw standardised RSS is NOT representable, only its half is.
    log_rss = scaled_log_sum_squares(residual)
    log_max = math.log(np.finfo(float).max)
    assert log_max < log_rss <= log_max + math.log(2.0), log_rss


def test_a_genuinely_unrepresentable_half_rss_still_returns_minus_inf() -> None:
    """Fail-closed control: the bound moved by one octave, it did not vanish.

    The outcome is captured rather than asserted inline. Without the guard the
    call raises ``OverflowError`` from ``math.exp``, and a test that dies of an
    exception is indistinguishable from an incidental crash in a mutation run --
    the death must be attributable to an ASSERTION for the proof to count.
    """
    residual = np.array([1.4e308])
    log_rss = scaled_log_sum_squares(residual)
    assert log_rss > math.log(np.finfo(float).max) + math.log(2.0)

    outcome: object
    try:
        outcome = gaussian_log_likelihood(residual, sigma=1.0)
    except Exception as exc:
        outcome = exc
    assert not isinstance(outcome, BaseException), (
        f"the likelihood raised {type(outcome).__name__} "
        f"({outcome}) instead of returning -inf"
    )
    assert outcome == float("-inf")


def test_values_inside_the_old_range_are_bit_identical() -> None:
    """Widening a range must not perturb the values already inside it.

    The halving stays ``0.5 * exp(...)`` below the disputed octave, so every
    likelihood the previous implementation could compute is reproduced exactly
    rather than to within an ulp.
    """
    for scale in (1.0e-30, 1.0, 3.7, 1.0e30, 1.0e120):
        residuals = scale * np.array([0.3, -1.1, 2.0, 0.0])
        log_rss = scaled_log_sum_squares(residuals)
        log_standardised = log_rss  # sigma = 1
        assert log_standardised <= math.log(np.finfo(float).max)
        n = residuals.size
        expected = (
            -0.5 * n * math.log(2.0 * math.pi) - 0.5 * math.exp(log_standardised)
        )
        assert gaussian_log_likelihood(residuals, sigma=1.0) == expected
