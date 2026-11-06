"""Reproduction fixture for the default-window migration (PR #127, LIO-04).

A caller who does not pass ``t_grid`` now gets the gap-scaled window
``[0, RELAXATION_HORIZON / gap]`` instead of the absolute legacy
``linspace(0, 10, 80)``. The two coincide only at ``gap = 1``. Elsewhere the
selected model and ``beta_D`` change. The cross-family review's example was
measured on ``_chain(1)`` of ``tests/test_trajectory_propagation.py``
(re-built here so this file has no cross-test import): one transverse-field
site with amplitude damping, ``gap ~ 0.35``.

* Legacy window (explicit ``t_grid``): M0, ``beta_D = 0.75552``, inside its
  interval -- on every platform measured.
* Gap-scaled default: M2 on every platform measured, but ``beta_D`` is NOT
  reproducible to better than ~0.6 %. It lands on one of two values:

  ========  ===========================  =================  ==================
  beta_D    BC interval                  estimate           measured on
  ========  ===========================  =================  ==================
  0.72143   ``[0.70829, 0.71997]``       above, OUTSIDE     Windows x86-64
  0.71738   ``[0.70597, 0.71915]``       inside             Linux x86-64/arm64
  ========  ===========================  =================  ==================

  Why (measured at 2f3b7e3, numpy 2.5.4 / scipy 1.18.1): the second M2
  component runs off to ``beta2 ~ 1e6``; it decays inside the first grid step,
  so its Jacobian column is exactly zero. ``least_squares``' ``xtol`` test is
  relative to ``|x|``, which ``beta2`` then dominates, so the solver stops as
  soon as a ``beta1`` step falls below ~``1e-2`` -- and where that happens is
  decided by a 5e-8 relative difference in the propagated curve. The same
  Windows solver run on the Linux curve returns the Linux value; 25 one-ulp
  shifts of the field amplitude on ONE platform hit both values. With
  ``xtol = 1e-12`` both curves converge to ``beta1 = 0.71731`` at a higher
  likelihood: both reported values are early stops. That is an open finding
  about the fit, not changed here; this file pins what a caller sees today.

The interval is labelled ``interval_method = "BC"``: a bias-corrected
percentile interval of the parametric bootstrap, not BCa (issue #116). Its
endpoints are quantiles of the bootstrap distribution shifted by the
bias-correction ``z0``, so it is centred on that distribution, not on the
point estimate, and nothing forces it to contain the estimate. Whether the
default estimate sits outside its interval is therefore platform-dependent,
and the tests below assert it per attractor, never unconditionally.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from liouscope._consts import INTERVAL_METHOD_BC
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.relaxation import RELAXATION_HORIZON, compute_relaxation_layer

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _chain1() -> tuple[np.ndarray, np.ndarray]:
    L = build_liouvillian(0.3 * _SX, [math.sqrt(0.7) * _SM])
    psi = np.ones(2, dtype=complex) / math.sqrt(2.0)
    return L, np.outer(psi, psi.conj())


def _layer(**kw):
    L, rho0 = _chain1()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=20, **kw)


@pytest.fixture(scope="module")
def default_window():
    return _layer()


@pytest.fixture(scope="module")
def legacy_window():
    return _layer(t_grid=np.linspace(0.0, 10.0, 80))


# Tolerances come from the measured spread, not from one platform (AGENTS.md,
# issue #180). Rates are compared relative to their own scale. Measured at
# 2f3b7e3 on Windows x86-64 (numpy 2.5.4 / scipy 1.18.1), Linux x86-64
# (Python 3.12 and 3.14 with 2.5.4 / 1.18.1, Python 3.12 with the floors
# 2.0.2 / 1.13.1), each of three configurations also over 25 one-ulp shifts of
# the field amplitude (79 runs), plus the Linux arm64 CI value: within one
# attractor ``beta_D`` spreads by <= 6e-6 relative, the bootstrap endpoints by
# <= 2.5e-5, the legacy ``beta_D`` by <= 7e-10.
# _RTOL_BETA (2e-5) must stay below the distance to the converged optimum
# 0.71731 (9e-5 from the lower attractor): a fit that stops converging early
# changes the reported number, and that must turn this file red so the
# migration note gets rewritten. The two attractors are 5.6e-3 apart.
_RTOL_BETA = 2.0e-5
_RTOL_CI = 1.0e-4
_RTOL_LEGACY = 1.0e-6

# (beta_D, lo, hi, estimate_outside_interval) per measured attractor.
_ATTRACTORS = (
    (0.721430, 0.708295, 0.719968, True),
    (0.717376, 0.705967, 0.719150, False),
)
_LEGACY_BETA = 0.7555151515


def _attractor(beta: float) -> tuple[float, float, float, bool]:
    hits = [a for a in _ATTRACTORS if math.isclose(beta, a[0], rel_tol=_RTOL_BETA)]
    assert len(hits) == 1, (
        f"default-window beta_D = {beta!r} matches no measured attractor "
        f"{[a[0] for a in _ATTRACTORS]} at rel_tol={_RTOL_BETA}; the fit changed, "
        "update this fixture AND the migration note (CHANGELOG.md, "
        "docs/DEVELOPMENT_MIGRATION_0.6.0.dev0.md)"
    )
    return hits[0]


def test_attractor_table_is_resolvable() -> None:
    """The tolerance must separate the two attractors, or the per-attractor
    assertions below would compare against the wrong row."""
    (b1, *_), (b2, *_) = _ATTRACTORS
    assert not math.isclose(b1, b2, rel_tol=100.0 * _RTOL_BETA)


def test_default_window_is_gap_scaled_and_not_the_legacy_window(default_window) -> None:
    rel = default_window
    assert rel.t_grid_source == "gap_scaled"
    assert rel.t_grid.size == 80
    # span = HORIZON / gap with gap ~ 0.35 here, not the legacy 10.
    assert math.isclose(rel.t_grid_span, RELAXATION_HORIZON / 0.35, rel_tol=0.01)


def test_default_window_changes_model_and_beta(default_window, legacy_window) -> None:
    assert default_window.aicc_model == "M2"
    _attractor(default_window.beta_D)
    assert legacy_window.t_grid_source == "caller"
    assert legacy_window.aicc_model == "M0"
    assert math.isclose(legacy_window.beta_D, _LEGACY_BETA, rel_tol=_RTOL_LEGACY)
    # The migration itself, platform-independent: the default rate is lower
    # than the legacy one by >= 4 % of the legacy rate. Measured 4.5 % (upper
    # attractor) and 5.0 % (lower); 4 % keeps 0.5 percentage points of margin,
    # ~1000x the within-attractor spread.
    shift = (legacy_window.beta_D - default_window.beta_D) / legacy_window.beta_D
    assert shift >= 0.04


def test_interval_containment_follows_the_fit_attractor(default_window) -> None:
    rel = default_window
    lo, hi = rel.bca_ci_beta
    # The interval says which estimator produced it (the review's CI_method).
    assert rel.interval_method == INTERVAL_METHOD_BC
    assert math.isfinite(lo) and math.isfinite(hi) and lo < hi
    _beta, lo_ref, hi_ref, outside = _attractor(rel.beta_D)
    assert math.isclose(lo, lo_ref, rel_tol=_RTOL_CI)
    assert math.isclose(hi, hi_ref, rel_tol=_RTOL_CI)
    # Containment, with a margin far above the endpoint tolerance: measured
    # (beta_D - hi) / beta_D >= 2.0e-3 on the upper attractor and
    # min(hi - beta_D, beta_D - lo) / beta_D >= 2.4e-3 on the lower one.
    if outside:
        assert (rel.beta_D - hi) / rel.beta_D > 10.0 * _RTOL_CI
    else:
        assert (hi - rel.beta_D) / rel.beta_D > 10.0 * _RTOL_CI
        assert (rel.beta_D - lo) / rel.beta_D > 10.0 * _RTOL_CI


def test_legacy_window_interval_contains_its_estimate(legacy_window) -> None:
    """Contrast case: same system, explicit legacy grid."""
    lo, hi = legacy_window.bca_ci_beta
    assert legacy_window.interval_method == INTERVAL_METHOD_BC
    assert lo <= legacy_window.beta_D <= hi
