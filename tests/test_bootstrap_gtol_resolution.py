"""Bootstrap replicates stall on SciPy's absolute ``gtol`` at small noise.

Finding 2026-11-15 (``claim_status: pending``, KANON ``RESEARCH-LIOUSCOPE``).

``parametric_bootstrap`` starts every replicate fit at ``theta_hat``.
``fit_gls_ar1`` divides the residuals presented to ``least_squares`` by the
curve's own amplitude (issue #124). That makes the solver's stopping rule
amplitude-invariant, but not noise-level-invariant. TRF's ``gtol`` test
(``norm(g_scaled, inf) < 1e-8``) is absolute in those amplitude-normalised
units, and at ``theta_hat`` the gradient of a replicate is proportional to its
noise. Once the relative noise falls to about 1e-8, the test is met at the
starting point (``status=1`` after one evaluation, zero step). The replicate
returns ``theta_hat`` unchanged and counts as converged.

Measured on ``M0 = A exp(-k t)``, 48 points, iid Gaussian noise. The table
compares ``sd_boot(k) / sd_true(k)`` (the true value from 150 independent
realisations) and the number of distinct values among 200 replicates:

    sigma  1e-2   1e-4   1e-6   1e-7   1e-8   1e-9
    ratio  0.709  0.714  0.714  0.713  0.673  0.000
    uniq   200    200    200    200    149    1

There are two consequences, in opposite directions:

* **Fail-open at about 1e-8.** About a quarter of the replicates sit exactly
  on ``theta_hat``, so mass collects at the centre and the interval shrinks.
  This is the same mechanism round 20 of PR #121 measured for failed
  replicates.
* **Spurious refusal at about 1e-9.** Every replicate is identical, so
  ``bca_ci`` reports "degenerate" (issue #125) for noise that is resolvable
  with float64 (1e-9 is seven decades above ``eps``).

The base fit stops on the same rule. On a noise-free ``exp(-1.3 t)`` it ends
with residuals of about 1e-11, about 1e4 ``eps``, not at rounding level. So the
NaN interval that issue #125 requires for that curve is today produced
**by the stall**, not by a property of the data. A fix must keep that refusal
on its own grounds, which the control at the end of this file pins.

A fix changes the fit's convergence rule, which belongs to the methodology
surface that PR #127/#193 is reworking (owner-gated). It is therefore pinned
here as strict xfail, not patched.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope.fitting.bootstrap import bca_ci, parametric_bootstrap
from liouscope.fitting.models import M0

_T = np.linspace(0.0, 5.0, 48)
_TRUE = np.array([1.0, 1.3])
_P0 = np.array([1.0, 1.2])
_B = 40


def _bootstrap(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    y = M0(_T, _TRUE) + np.random.default_rng(99).normal(0.0, sigma, _T.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parametric_bootstrap(
            M0, _T, y, _P0, B=_B, rng=np.random.default_rng(0)
        )


def _stalled(samples: np.ndarray, theta_hat: np.ndarray) -> int:
    """Count replicates whose fitted rate equals ``theta_hat`` exactly.

    A replicate that took any step lands somewhere else. Exact equality with
    the start point is the signature of a zero-step termination, not of a
    coincidence: a continuous estimator hits one float64 value with
    probability zero.
    """
    return int(np.sum(samples[:, 1] == theta_hat[1]))


def test_control_replicates_move_at_ordinary_noise() -> None:
    """Positive control: at 1e-4 relative noise no replicate stalls.

    Without this, the xfail tests below could pass against a bootstrap that
    never moves at any noise level.
    """
    samples, theta_hat = _bootstrap(1.0e-4)
    assert _stalled(samples, theta_hat) == 0


@pytest.mark.xfail(
    strict=True,
    reason="replicates stop on SciPy's absolute gtol at theta_hat (finding 2026-11-15)",
)
def test_replicates_move_at_1e8_relative_noise() -> None:
    """Fail-open direction: stalled replicates pile mass on the centre."""
    samples, theta_hat = _bootstrap(1.0e-8)
    assert _stalled(samples, theta_hat) == 0


@pytest.mark.xfail(
    strict=True,
    reason="every replicate stalls, so bca_ci reports a degenerate distribution",
)
def test_resolvable_noise_at_1e9_yields_a_finite_interval() -> None:
    """Spurious-refusal direction: 1e-9 noise is measurable in float64.

    The expected interval width is of the order of the estimator's standard
    deviation (about 1.3e-9 for the rate, see the module docstring). The
    assertion only requires a finite interval that contains ``theta_hat``.
    It does not require a width, so the test does not depend on the solver's
    exact endpoint.
    """
    samples, theta_hat = _bootstrap(1.0e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = bca_ci(samples, theta_hat)
    assert np.all(np.isfinite(ci[1]))
    assert ci[1, 0] < theta_hat[1] < ci[1, 1]


def test_control_noise_free_curve_still_refuses_an_interval() -> None:
    """Issue #125 must survive the fix: a noise-free curve gets no interval.

    Today this refusal comes from the stall itself. A fix that makes the
    replicates move must reach the same verdict on grounds that belong to the
    data (a residual scale at the solver's or the arithmetic's own
    resolution). It must not hand back a 1e-11-wide interval as a measurement.
    Either a ``RuntimeError`` from the bootstrap or a NaN interval is accepted,
    because the caller routes both to "fit uncertainty UNKNOWN".
    """
    y = M0(_T, _TRUE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            samples, theta_hat = parametric_bootstrap(
                M0, _T, y, _P0, B=_B, rng=np.random.default_rng(0)
            )
        except RuntimeError:
            return
        ci = bca_ci(samples, theta_hat)
    assert np.all(np.isnan(ci))
