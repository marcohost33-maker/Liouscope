"""PR #127 against current ``main``: the residual family is one decision.

Three things meet here that were built apart:

* #124/#134 rescale the residuals handed to the optimiser;
* #127 picks the whitening from the grid;
* the review of #127 asked that the jackknife keeps the base fit's family and
  that AICc counts every residual parameter of a non-uniform fit, the AR(1)
  fallback included.

Every test here fails on a build that handles one of them on its own.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import liouscope.fitting.bootstrap as bootstrap
import liouscope.fitting.gls as gls
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.relaxation import _fit_with_model, compute_relaxation_layer
from liouscope.fitting.aicc import aicc
from liouscope.fitting.car1 import car1_resample, is_uniform_grid
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M0

# Grids of at most 40 points are the point here; their small-n notice is not.
pytestmark = pytest.mark.filterwarnings("ignore:GLS AR.1. fit on only")

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_RHO_PLUS = 0.5 * np.ones((2, 2), dtype=complex)
_TWO_SCALE = np.concatenate(
    [np.linspace(0.0, 0.5, 20, endpoint=False), np.linspace(0.5, 8.0, 20)]
)


def _noisy_decay(t: np.ndarray, seed: int = 11) -> np.ndarray:
    # Correlated noise from the exact CAR(1) transition, so ``theta`` is interior
    # and identified: on white noise the profile is flat above 1/min(dt), and two
    # equally good values there differ by 15 % (measured).
    rng = np.random.default_rng(seed)
    return np.exp(-1.3 * t) + car1_resample(rng, t, 2.0, 1.0e-3)


def _record_refits(monkeypatch: pytest.MonkeyPatch, n: int) -> list[float]:
    """Wrap the fit the bootstrap module calls; keep ``theta`` of each refit."""
    seen: list[float] = []
    real = bootstrap.fit_gls_ar1

    def spy(model, t, y, p0, **kw):  # type: ignore[no-untyped-def]
        out = real(model, t, y, p0, **kw)
        if np.asarray(t).size == n - 1:
            seen.append(float(out.theta_car1))
        return out

    monkeypatch.setattr(bootstrap, "fit_gls_ar1", spy)
    return seen


def _layer_on(t_grid: np.ndarray) -> object:
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_SM])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_relaxation_layer(
            L, rho_initial=_RHO_PLUS, t_grid=t_grid, bootstrap_B=10, seed=3
        )


# ---------------------------------------------------------------------------
# The jackknife estimates the base fit's acceleration, with the base fit's family
# ---------------------------------------------------------------------------


def test_uniform_grid_jackknife_refits_stay_discrete_ar1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interior deletions used to switch to CAR(1), endpoint deletions did not."""
    t = np.linspace(0.0, 8.0, 40)
    seen = _record_refits(monkeypatch, t.size)
    layer = _layer_on(t)
    assert layer.interval_method == "BCa-delete1"  # type: ignore[attr-defined]
    assert len(seen) == t.size
    assert not any(np.isfinite(v) for v in seen), seen


def test_non_uniform_grid_jackknife_refits_stay_car1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive control: the fix must not pin every refit to AR(1)."""
    seen = _record_refits(monkeypatch, _TWO_SCALE.size)
    layer = _layer_on(_TWO_SCALE)
    assert layer.residual_model == "car1"  # type: ignore[attr-defined]
    assert layer.interval_method == "BCa-delete1"  # type: ignore[attr-defined]
    assert len(seen) == _TWO_SCALE.size
    assert all(np.isfinite(v) for v in seen), seen


def test_jackknife_without_a_family_reads_it_off_the_full_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t = np.linspace(0.0, 8.0, 12)
    y = _noisy_decay(t)
    seen = _record_refits(monkeypatch, t.size)
    theta_hat = np.array([1.0, 1.3])
    bootstrap._jackknife(M0, t, y, theta_hat, None)
    assert len(seen) == t.size
    assert not any(np.isfinite(v) for v in seen)


def test_residual_family_is_honoured_and_validated() -> None:
    t = np.linspace(0.0, 8.0, 40)
    assert is_uniform_grid(t)
    y = _noisy_decay(t)
    p0 = np.array([1.0, 1.0])
    assert not np.isfinite(fit_gls_ar1(M0, t, y, p0).theta_car1)
    assert np.isfinite(fit_gls_ar1(M0, t, y, p0, residual_family="car1").theta_car1)

    assert not is_uniform_grid(_TWO_SCALE)
    y2 = _noisy_decay(_TWO_SCALE)
    assert np.isfinite(fit_gls_ar1(M0, _TWO_SCALE, y2, p0).theta_car1)
    assert not np.isfinite(
        fit_gls_ar1(M0, _TWO_SCALE, y2, p0, residual_family="ar1").theta_car1
    )

    for bad in ("", "AR1", "ar(1)", None, 1):
        with pytest.raises(ValueError, match="residual_family"):
            fit_gls_ar1(M0, t, y, p0, residual_family=bad)  # type: ignore[arg-type]


def test_uniform_family_on_a_uniform_grid_equals_auto() -> None:
    """Fixing the family the grid would pick anyway must change nothing at all."""
    t = np.linspace(0.0, 8.0, 40)
    y = _noisy_decay(t)
    p0 = np.array([1.0, 1.0])
    auto = fit_gls_ar1(M0, t, y, p0)
    fixed = fit_gls_ar1(M0, t, y, p0, residual_family="ar1")
    np.testing.assert_array_equal(auto.params, fixed.params)
    assert auto.log_likelihood == fixed.log_likelihood


# ---------------------------------------------------------------------------
# AICc counts what was fitted, on both branches of a non-uniform grid
# ---------------------------------------------------------------------------


def test_car1_fit_counts_rate_and_variance() -> None:
    y = _noisy_decay(_TWO_SCALE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, _ = _fit_with_model("M0", _TWO_SCALE, y)
    assert np.isfinite(fit.residual_theta_car1)
    p = int(np.asarray(fit.params).size)
    assert fit.aicc == aicc(fit.log_likelihood, p + 2, fit.n_eff)
    assert fit.aicc != aicc(fit.log_likelihood, p + 1, fit.n_eff)


def test_ar1_fallback_on_a_non_uniform_grid_counts_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``car1_mixed`` comparison must not hand the fallback a smaller penalty."""
    monkeypatch.setattr(gls, "estimate_car1_theta", lambda _t, _r: float("nan"))
    y = _noisy_decay(_TWO_SCALE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, _ = _fit_with_model("M0", _TWO_SCALE, y)
    assert not np.isfinite(fit.residual_theta_car1), "fixture must fall back"
    p = int(np.asarray(fit.params).size)
    assert fit.aicc == aicc(fit.log_likelihood, p + 2, fit.n_eff)
    assert fit.aicc != aicc(fit.log_likelihood, p, fit.n_eff)


def test_uniform_grid_keeps_the_historical_count() -> None:
    """Negative control: changing this is PR #157's scope, not this one's."""
    t = np.linspace(0.0, 8.0, 40)
    y = _noisy_decay(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit, _ = _fit_with_model("M0", t, y)
    p = int(np.asarray(fit.params).size)
    assert fit.aicc == aicc(fit.log_likelihood, p, fit.n_eff)


# ---------------------------------------------------------------------------
# #124/#134 rescaling and the #127 whitening work on one objective
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1.0e-40, 1.0e-150, 1.0e40, 1.0e90])
def test_non_uniform_fit_is_amplitude_invariant_under_car1(scale: float) -> None:
    """The rescaled objective has to whiten with CAR(1), as the raw one does.

    Whitening the rescaled residuals with plain AR(1) instead returned a
    different rate at every scale except the one the raw path handles.
    """
    y = _noisy_decay(_TWO_SCALE)
    p0 = np.array([1.0, 1.0])
    reference = fit_gls_ar1(M0, _TWO_SCALE, y, p0)
    assert reference.success and np.isfinite(reference.theta_car1)
    scaled = fit_gls_ar1(M0, _TWO_SCALE, scale * y, p0 * np.array([scale, 1.0]))
    assert scaled.success
    assert scaled.params[1] == pytest.approx(reference.params[1], rel=1.0e-5)
    assert scaled.params[0] / scale == pytest.approx(reference.params[0], rel=1.0e-5)
    assert scaled.theta_car1 == pytest.approx(reference.theta_car1, rel=1.0e-4)


def test_rescaling_fallback_keeps_the_car1_family() -> None:
    """The raw retry after a non-representable rescale must not lose the family."""
    y = _noisy_decay(_TWO_SCALE)
    p0 = np.array([1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_gls_ar1(M0, _TWO_SCALE, 1.0e-310 * y, p0 * np.array([1.0e-310, 1.0]))
    assert out.success
    assert np.isfinite(out.theta_car1)
