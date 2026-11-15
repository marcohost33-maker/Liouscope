"""Bootstrap replicates stalled on SciPy's absolute ``gtol`` at small noise.

Finding 2026-11-15 (``claim_status: pending``, KANON ``RESEARCH-LIOUSCOPE``).

``parametric_bootstrap`` starts every replicate fit at ``theta_hat``. SciPy's
``trf`` stops when ``norm(J^T f, inf) < gtol``. That test is absolute in
amplitude-normalised residual units (issue #124) times inverse parameter
units. At ``theta_hat`` the gradient of a replicate is proportional to its
noise, so from about 1e-8 relative noise the test was met before the first
step. The replicate returned ``theta_hat`` unchanged and counted as converged.

Measured before the fix on ``M0 = A exp(-k t)``, 48 points, iid noise. The
ratio is ``sd_boot(k) / sd_true(k)`` from one realisation; "uniq" is the
number of distinct rates among 200 replicates:

    sigma  1e-2   1e-4   1e-6   1e-7   1e-8   1e-9
    ratio  0.709  0.714  0.714  0.713  0.673  0.000
    uniq   200    200    200    200    149    1

The fix lives in ``parametric_bootstrap`` only, and every one of its fits
runs with ``gtol`` disabled:

1. It polishes the base estimate. A noise-free curve then leaves no
   representable residual scale, and the bootstrap refuses on that ground
   (issues #125/#135) instead of through the stall.
2. It simulates and refits the replicates around the polished optimum.
3. It refuses when the caller's default-tolerance estimate sits more than
   0.1 resampling SD from that optimum. Measured offsets: 0.04 SD at 1e-8,
   0.5 SD at 1e-9, 59 SD at 1e-11.

Fixing the default tolerance globally was rejected on measurement. With
``gtol`` disabled everywhere, nine tests failed through one mechanism: the
converged residuals of a noise-free curve are exactly zero, so the likelihood
scale has no MLE (#135) and the fit leaves AICc model selection. The
pipeline's likelihood on noise-free curves currently rests on the solver's
truncation residual. That is a separate owner decision (#127/#193).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope.fitting.bootstrap import bca_ci, parametric_bootstrap
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M0

_T = np.linspace(0.0, 5.0, 48)
_TRUE = np.array([1.0, 1.3])
_P0 = np.array([1.0, 1.2])
_B = 40


def _curve(sigma: float) -> np.ndarray:
    noise = np.random.default_rng(99).normal(0.0, sigma, _T.size) if sigma else 0.0
    return M0(_T, _TRUE) + noise


def _bootstrap(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return parametric_bootstrap(
            M0, _T, _curve(sigma), _P0, B=_B, rng=np.random.default_rng(0)
        )


def _stalled(samples: np.ndarray, theta_hat: np.ndarray) -> int:
    """Count replicates whose fitted rate equals ``theta_hat`` exactly.

    A replicate that took any step lands elsewhere. A continuous estimator
    hits one float64 value with probability zero, so exact equality with the
    start point is the signature of a zero-step termination.
    """
    return int(np.sum(samples[:, 1] == theta_hat[1]))


def test_default_gtol_is_unchanged_for_every_other_caller() -> None:
    """The new keyword must not move any fit that does not ask for it."""
    y = _curve(1.0e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        implicit = fit_gls_ar1(M0, _T, y, _P0)
        explicit = fit_gls_ar1(M0, _T, y, _P0, gtol=1.0e-8)
    assert np.array_equal(implicit.params, explicit.params)
    assert implicit.sigma == explicit.sigma


def test_control_replicates_move_and_report_at_ordinary_noise() -> None:
    """Silent direction: at 1e-4 nothing stalls and a finite interval comes back."""
    samples, theta_hat = _bootstrap(1.0e-4)
    assert _stalled(samples, theta_hat) == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci = bca_ci(samples, theta_hat)
    assert np.all(np.isfinite(ci))
    assert ci[1, 0] < theta_hat[1] < ci[1, 1]


def test_replicates_move_at_1e8_relative_noise() -> None:
    """Fail-open direction, closed: before the fix about 25 % sat on the centre."""
    samples, theta_hat = _bootstrap(1.0e-8)
    assert _stalled(samples, theta_hat) == 0


def test_1e9_noise_is_refused_for_the_estimate_not_as_degenerate() -> None:
    """Before the fix, every replicate stalled and ``bca_ci`` called 1e-9 noise
    "degenerate". Now the replicates move, and the refusal names the real
    reason: the caller's estimate is about 0.5 resampling SD from the
    converged optimum. An interval around the optimum would therefore not
    describe the value the caller reports.
    """
    with pytest.raises(RuntimeError, match="not converged at the resolution"):
        _bootstrap(1.0e-9)


def test_noise_free_curve_is_refused_for_lack_of_a_residual_scale() -> None:
    """Issue #125 survives the fix, now on a reason that belongs to the data.

    Before, the refusal came from the stall itself (all replicates
    identical). Now the converged fit leaves no representable residual scale,
    so there is no noise to resample.
    """
    with pytest.raises(RuntimeError, match="no representable residual scale"):
        _bootstrap(0.0)
