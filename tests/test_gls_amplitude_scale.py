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


@pytest.mark.parametrize("scale", [1.0e-150, 1.0e-310])
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
    import liouscope.fitting.gls as gls_mod

    real = gls_mod.least_squares
    calls = {"n": 0}

    def first_call_poisoned(*args, **kwargs):  # type: ignore[no-untyped-def]
        res = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] == 1:
            res.jac = np.full_like(res.jac, np.nan)
        return res

    monkeypatch.setattr(gls_mod, "least_squares", first_call_poisoned)
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
