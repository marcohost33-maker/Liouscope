"""Reproduction fixture for the default-window migration (PR #127, LIO-04).

A caller who does not pass ``t_grid`` now gets the gap-scaled window
``[0, RELAXATION_HORIZON / gap]`` instead of the absolute legacy
``linspace(0, 10, 80)``. The two coincide only at ``gap = 1``. The
cross-family review's example was measured on ``_chain(1)`` of
``tests/test_trajectory_propagation.py`` (re-built here so this file has no
cross-test import): one transverse-field site with amplitude damping,
``gap ~ 0.35``.

* Legacy window (explicit ``t_grid``): M0, ``beta_D = 0.75552``.
* Gap-scaled default: M0, ``beta_D = 0.753447`` -- the same model, a rate
  0.27 % lower because the window is different. Reproducible across
  platforms to ~2e-10 relative.

History. Until the M2 identifiability fix (PR #127 option B, prereg
``docs/prereg/2026-11-08_m2_identifiability.md``) this file pinned M2 at one
of two platform-dependent values (0.72143 Windows / 0.71738 Linux). Both
were artefacts: the second M2 rate ran off to ~1e6 under
``x_scale="jac"``, explained only the point ``t = 0``, and the solver stopped
on ``xtol`` where rounding decided. With the M2 fit in characteristic
variables the second rate converges to 7.67 -- still outside the window the
grid resolves (``beta2 * dt = 2.77 > 1``), so M2 is "not identified" and
leaves the comparison; M0 wins.

The interval is labelled ``interval_method = "BC"``: a bias-corrected
percentile interval of the parametric bootstrap, not BCa (issue #116).
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
# issue #180). Measured at fad0f55, numpy 2.5.4 / scipy 1.18.1, Windows x86-64
# and Linux x86-64 (Docker, python:3.12-slim), each over 5 one-ulp shifts of
# the field amplitude: default ``beta_D`` in [0.75344699988, 0.75344700003],
# interval endpoints within 3e-10 of the values below, legacy ``beta_D``
# 0.75551515130 / 0.75551515159. Every tolerance is >= 1000x that spread.
_BETA_DEFAULT = 0.753447
_CI_DEFAULT = (0.7472639, 0.7582746)
_RTOL_BETA = 1.0e-6
_RTOL_CI = 1.0e-6
_RTOL_LEGACY = 1.0e-6
_LEGACY_BETA = 0.7555151515


def test_default_window_is_gap_scaled_and_not_the_legacy_window(default_window) -> None:
    rel = default_window
    assert rel.t_grid_source == "gap_scaled"
    assert rel.t_grid.size == 80
    # span = HORIZON / gap with gap ~ 0.35 here, not the legacy 10.
    assert math.isclose(rel.t_grid_span, RELAXATION_HORIZON / 0.35, rel_tol=0.01)


def test_default_window_keeps_the_model_and_moves_beta(default_window, legacy_window) -> None:
    beta = float(default_window.beta_D)
    assert math.isfinite(beta), beta  # NaN fails closed, not via isclose
    assert default_window.aicc_model == "M0"
    assert math.isclose(beta, _BETA_DEFAULT, rel_tol=_RTOL_BETA), beta
    assert legacy_window.t_grid_source == "caller"
    assert legacy_window.aicc_model == "M0"
    assert math.isclose(legacy_window.beta_D, _LEGACY_BETA, rel_tol=_RTOL_LEGACY)
    # The migration itself: the default rate is 0.274 % below the legacy one
    # (measured 2.737e-3). The band is +-0.3e-3 around it, ~100x the
    # uncertainty the two rel 1e-6 tolerances above leave on the difference,
    # and excludes both "no change" and the former M2 artefact (4.5-5 %).
    shift = (legacy_window.beta_D - beta) / legacy_window.beta_D
    assert 2.4e-3 < shift < 3.0e-3, shift


def test_default_window_m2_is_tried_and_refused(default_window) -> None:
    fr = default_window.fits["M2"]
    assert fr.success
    assert fr.identifiable is False
    assert "outside the window the grid resolves" in fr.identification_reason
    assert not math.isfinite(fr.aicc)


def test_default_window_interval_contains_its_estimate(default_window) -> None:
    rel = default_window
    lo, hi = rel.bca_ci_beta
    assert rel.interval_method == INTERVAL_METHOD_BC
    assert math.isfinite(lo) and math.isfinite(hi) and lo < hi
    assert math.isclose(lo, _CI_DEFAULT[0], rel_tol=_RTOL_CI)
    assert math.isclose(hi, _CI_DEFAULT[1], rel_tol=_RTOL_CI)
    # Containment with margin: measured (beta - lo)/beta = 8.2e-3 and
    # (hi - beta)/beta = 6.4e-3, far above the endpoint tolerance.
    assert (hi - rel.beta_D) / rel.beta_D > 1.0e-3
    assert (rel.beta_D - lo) / rel.beta_D > 1.0e-3


def test_legacy_window_interval_contains_its_estimate(legacy_window) -> None:
    """Contrast case: same system, explicit legacy grid."""
    lo, hi = legacy_window.bca_ci_beta
    assert legacy_window.interval_method == INTERVAL_METHOD_BC
    assert lo <= legacy_window.beta_D <= hi
