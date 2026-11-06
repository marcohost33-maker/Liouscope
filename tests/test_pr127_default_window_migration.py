"""Reproduction fixture for the default-window migration (PR #127, LIO-04).

A caller who does not pass ``t_grid`` now gets the gap-scaled window
``[0, RELAXATION_HORIZON / gap]`` instead of the absolute legacy
``linspace(0, 10, 80)``. The two coincide only at ``gap = 1``. Elsewhere the
selected model and ``beta_D`` change. The cross-family review's example was
measured on ``_chain(1)`` of ``tests/test_trajectory_propagation.py``
(re-built here so this file has no cross-test import): one transverse-field
site with amplitude damping, ``gap ~ 0.35``.

* Gap-scaled default: M2, ``beta_D = 0.72143``, interval
  ``[0.70829, 0.71997]`` -- the point estimate lies OUTSIDE its interval.
* Legacy window (explicit ``t_grid``): M0, ``beta_D = 0.75552``, inside.

The interval is labelled ``interval_method = "BC"``: a bias-corrected
percentile interval of the parametric bootstrap, not BCa (issue #116). Its
endpoints are quantiles of the bootstrap distribution shifted by the
bias-correction ``z0``, so it is centred on that distribution, not on the
point estimate, and nothing forces it to contain the estimate. That a
reported ``beta_D`` sits outside ``bca_ci_beta`` is therefore not by itself
a defect. It is what a caller sees, and this file pins it so the migration
note in CHANGELOG.md cannot drift from the code.
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


# Rates are compared relative to their own scale (AGENTS.md, issue #180):
# the fits stop at least_squares tolerances, far above float64 rounding, and a
# different BLAS moves the fitted rate in the trailing digits only.
_RTOL = 1.0e-4


def test_default_window_is_gap_scaled_and_not_the_legacy_window(default_window) -> None:
    rel = default_window
    assert rel.t_grid_source == "gap_scaled"
    assert rel.t_grid.size == 80
    # span = HORIZON / gap with gap ~ 0.35 here, not the legacy 10.
    assert math.isclose(rel.t_grid_span, RELAXATION_HORIZON / 0.35, rel_tol=0.01)


def test_default_window_changes_model_and_beta(default_window, legacy_window) -> None:
    assert default_window.aicc_model == "M2"
    assert math.isclose(default_window.beta_D, 0.72143, rel_tol=_RTOL)
    assert legacy_window.t_grid_source == "caller"
    assert legacy_window.aicc_model == "M0"
    assert math.isclose(legacy_window.beta_D, 0.75552, rel_tol=_RTOL)


def test_reported_interval_need_not_contain_the_point_estimate(default_window) -> None:
    rel = default_window
    lo, hi = rel.bca_ci_beta
    # The interval says which estimator produced it (the review's CI_method).
    assert rel.interval_method == INTERVAL_METHOD_BC
    assert math.isclose(lo, 0.70829, rel_tol=_RTOL)
    assert math.isclose(hi, 0.71997, rel_tol=_RTOL)
    # The fixture: estimate above the upper endpoint, by ~2e-3 relative --
    # three orders of magnitude above the tolerance used for the values.
    assert rel.beta_D > hi
    assert (rel.beta_D - hi) / rel.beta_D > 10.0 * _RTOL


def test_legacy_window_interval_contains_its_estimate(legacy_window) -> None:
    """Contrast case: same system, explicit legacy grid."""
    lo, hi = legacy_window.bca_ci_beta
    assert legacy_window.interval_method == INTERVAL_METHOD_BC
    assert lo <= legacy_window.beta_D <= hi
