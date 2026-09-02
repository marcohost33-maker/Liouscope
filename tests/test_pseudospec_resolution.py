"""D13 resolution contract: an unresolved sweep must not look like a measurement.

Background (2026-09-02)
-----------------------
``pseudospectral_radius`` initialised its accumulator to ``0.0`` and returned it
when no grid node satisfied ``sigma_min(zI - L) <= eps``. ``0.0`` is a
*plausible* pseudospectral radius ("the eps-pseudospectrum sits at the origin"),
so an unresolved sweep was indistinguishable from a measured one.

It reached a decision. ``classification.py`` gates the F5 phantom-relaxation
branch on ``pseudospectral_radius / gap > 2 * gap_to_gns_ratio``; with a
returned ``0.0`` that comparison is False, so an unmeasured D13 silently
reported "no pseudospectral intrusion".

The sibling ``pseudospectrum_extent`` was fail-visible from the start, so the
two implementations of the same quantity returned ``0.0`` and ``nan`` on
byte-identical input. These tests pin the resolved disagreement.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.numerics.pseudospec import (
    pseudospectral_radius,
    pseudospectrum_extent,
    pseudospectrum_sigma_floor,
)

# A NORMAL matrix. For a normal operator the eps-pseudospectrum is EXACTLY the
# union of closed eps-discs around the spectrum (Trefethen & Embree, *Spectra
# and Pseudospectra*, Thm. 2.2), so the true pseudospectral radius is
# max|lambda| + eps -- an analytic oracle, not another implementation of the
# same code path.
_SPECTRUM = np.array([0.0, -0.2, -1.0 - 3.0j, -1.0 + 3.0j, -4.0], dtype=complex)


def _normal_matrix() -> np.ndarray:
    rng = np.random.default_rng(20260902)
    seed = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    q, _ = np.linalg.qr(seed)
    a = q @ np.diag(_SPECTRUM) @ q.conj().T
    assert np.allclose(a @ a.conj().T, a.conj().T @ a), "construction is not normal"
    return a


_FINE_GRID = {"grid_re": (-6.0, 1.0, 101), "grid_im": (-5.0, 5.0, 101)}


def test_unresolved_sweep_returns_nan_not_zero() -> None:
    """The load-bearing regression. Pre-fix this returned ``0.0``."""
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        radius = pseudospectral_radius(a, 1.0e-3, **_FINE_GRID)
    assert np.isnan(radius)


@pytest.mark.parametrize("n", [25, 51, 101, 201])
def test_refining_the_grid_does_not_rescue_a_too_small_eps(n: int) -> None:
    """Documents WHY ``0.0`` was wrong rather than merely imprecise.

    At ``eps = 1e-3`` the pseudospectrum is a set of discs of radius ~1e-3.
    A rectangular sweep only resolves it when a node lands inside one, which
    refinement does not reliably achieve -- so the pre-fix code returned a
    confident ``0.0`` at every resolution up to 201x201 (40'401 SVDs).
    """
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        radius = pseudospectral_radius(
            a, 1.0e-3, grid_re=(-6.0, 1.0, n), grid_im=(-5.0, 5.0, n)
        )
    assert np.isnan(radius)


def test_the_two_estimators_agree_on_unresolved_input() -> None:
    """``pseudospectral_radius`` and ``pseudospectrum_extent`` used to disagree."""
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        legacy = pseudospectral_radius(a, 1.0e-3, **_FINE_GRID)
    extent_radius, extent_abscissa = pseudospectrum_extent(a, 1.0e-3, **_FINE_GRID)
    assert np.isnan(legacy)
    assert np.isnan(extent_radius)
    assert np.isnan(extent_abscissa)


def test_resolved_sweep_matches_the_analytic_oracle() -> None:
    """Positive control: with an eps the grid CAN resolve, the value is right.

    Without this the fix could be satisfied by a function that returns ``nan``
    unconditionally.
    """
    a = _normal_matrix()
    eps = 0.5  # comfortably above the grid's sigma_min floor
    radius = pseudospectral_radius(a, eps, **_FINE_GRID)
    true_radius = float(np.max(np.abs(_SPECTRUM))) + eps
    assert np.isfinite(radius)
    # A grid sweep is a LOWER bound on the true supremum (finite sampling, no
    # globality certificate), so it must not exceed the oracle, and on this
    # resolution it must come within one grid step of it.
    assert radius <= true_radius + 1e-12
    assert radius >= true_radius - 0.15


def test_sigma_floor_reports_the_smallest_eps_that_would_resolve() -> None:
    """The companion that turns "found nothing" into an actionable number."""
    a = _normal_matrix()
    floor = pseudospectrum_sigma_floor(a, **_FINE_GRID)
    assert np.isfinite(floor) and floor > 1.0e-3
    # Just above the floor the sweep resolves; just below it cannot.
    assert np.isfinite(pseudospectral_radius(a, floor * 1.01, **_FINE_GRID))
    with pytest.warns(RuntimeWarning, match="unresolved"):
        assert np.isnan(pseudospectral_radius(a, floor * 0.99, **_FINE_GRID))


def test_warning_names_the_achievable_eps() -> None:
    """A fail-visible message must say what to do, not merely that it failed."""
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning) as record:
        pseudospectral_radius(a, 1.0e-6, **_FINE_GRID)
    message = str(record[0].message)
    assert "sigma_min" in message and "nan" in message
