"""Approximate epsilon-pseudospectrum (D13).

Uses the grid-based singular-value definition

    sigma_eps(L) = { z in C : sigma_min(z I - L) <= eps }

restricted to a rectangular grid bracketing the spectrum (for a Liouvillian
that grid lies in the closed left half-plane). Returns the
maximal modulus of grid points belonging to the pseudospectrum, which we
report as the pseudospectral radius for diagnostic D13.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .linalg import require_finite_square_2d


def pseudospectral_radius(
    L: np.ndarray,
    eps: float = 1.0e-3,
    *,
    grid_re: tuple[float, float, int] | None = None,
    grid_im: tuple[float, float, int] | None = None,
) -> float:
    """Return the pseudospectral radius ``max{|z| : z in sigma_eps(L)}``.

    Parameters
    ----------
    L
        Square matrix.
    eps
        Pseudospectrum threshold.
    grid_re, grid_im
        Tuples ``(lo, hi, n)`` defining the real/imaginary axis grid.
        Defaults pick a region around the spectrum.
    """
    # Fail closed on non-finite / non-square input (consistent with linalg.py and
    # cptp.py) so a NaN/inf-laden operator surfaces a located, named error instead
    # of an opaque LAPACK failure deep in the svd loop.
    L = require_finite_square_2d(L, name="L")
    eigvals = sla.eigvals(L)
    if grid_re is None:
        re_max = float(np.max(np.real(eigvals)))
        re_min = float(np.min(np.real(eigvals)))
        span = max(1.0e-3, re_max - re_min)
        grid_re = (re_min - 0.5 * span, re_max + 0.5 * span, 25)
    if grid_im is None:
        im_max = float(np.max(np.imag(eigvals)))
        im_min = float(np.min(np.imag(eigvals)))
        span = max(1.0e-3, im_max - im_min)
        grid_im = (im_min - 0.5 * span, im_max + 0.5 * span, 25)

    res = np.linspace(*grid_re)
    ims = np.linspace(*grid_im)
    radius = float("nan")
    sigma_floor = float("inf")
    found = False
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    for re in res:
        for im in ims:
            z = re + 1j * im
            sv_min = float(sla.svdvals(z * eye - L)[-1])
            if sv_min < sigma_floor:
                sigma_floor = sv_min
            if sv_min <= eps:
                mod = float(abs(z))
                radius = mod if not found else max(radius, mod)
                found = True
    if not found:
        # FAIL-VISIBLE (2026-09-02 hardening). This branch used to return the
        # initialiser ``0.0``, which is a *plausible measurement* ("the
        # eps-pseudospectrum sits at the origin"), not a missing one. It is the
        # exact inverse of the failure PR #129 found in the workflow gate: there
        # a guard produced no result, here it produced a wrong one.
        #
        # Why this is not a corner case: for a well-separated spectrum the
        # eps-pseudospectrum at eps = 1e-3 is a union of discs of radius ~eps
        # around the eigenvalues. A rectangular grid only "sees" it when a node
        # lands inside such a disc. The DEFAULT grid does so only by accident --
        # ``np.linspace`` over [re_min - s/2, re_max + s/2] with 25 nodes places
        # nodes exactly on re_min/re_max (indices 6 and 18), so nodes coincide
        # with extremal eigenvalues and sigma_min is 0 there. Supply your own
        # grid and the same operator reports 0.0 at ANY resolution: a normal
        # matrix with spectrum {0, -0.2, -1+-3i, -4} and true radius 4.001
        # returned 0.0 on 25x25 through 201x201 grids.
        #
        # NaN matches the sibling :func:`pseudospectrum_extent`, which was built
        # fail-visible from the start; the two used to disagree (0.0 vs nan) on
        # byte-identical input.
        warnings.warn(
            f"eps-pseudospectrum unresolved: no grid node satisfies "
            f"sigma_min <= eps={eps!r} (smallest sigma_min on the grid was "
            f"{sigma_floor:.3e}). Returning nan (UNKNOWN), not 0.0. Refine the "
            f"grid or raise eps to at least ~{sigma_floor:.3e}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return radius


def pseudospectrum_sigma_floor(
    L: np.ndarray,
    *,
    grid_re: tuple[float, float, int],
    grid_im: tuple[float, float, int],
) -> float:
    """Return ``min{sigma_min(z I - L) : z in grid}``.

    This is the *resolution* companion to the radius/extent estimators: it is
    the smallest ``eps`` for which the given grid resolves the
    eps-pseudospectrum at all. Reporting it turns "the sweep found nothing"
    from an unexplained NaN into a located, actionable number.
    """
    L = require_finite_square_2d(L, name="L")
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    floor = float("inf")
    for re in np.linspace(*grid_re):
        for im in np.linspace(*grid_im):
            sv_min = float(sla.svdvals((re + 1j * im) * eye - L)[-1])
            floor = min(floor, sv_min)
    return floor


def pseudospectrum_extent(
    L: np.ndarray,
    eps: float,
    *,
    grid_re: tuple[float, float, int],
    grid_im: tuple[float, float, int],
) -> tuple[float, float]:
    """Return ``(radius, abscissa)`` of the grid eps-pseudospectrum in ONE sweep.

    * ``radius   = max{|z|   : z in grid, sigma_min(z I - L) <= eps}`` -- the
      D13 maximum-modulus quantity;
    * ``abscissa = max{Re(z) : z in grid, sigma_min(z I - L) <= eps}`` -- the
      gap-directed intrusion diagnostic (issue #101 slice A item 5): how far the
      eps-pseudospectrum reaches toward (or past) the imaginary axis. For
      slow/phantom relaxation the literature ties transient behaviour to this
      abscissa/intrusion rather than to the maximum modulus, which can be
      dominated by fast modes far from the origin.

    Both values are GRID LOWER-BOUND ESTIMATES of the true suprema (finite
    sampling; no globality certificate). If NO grid point belongs to the
    eps-pseudospectrum the sweep is under-resolved for this ``eps`` and
    ``(nan, nan)`` is returned -- an honest "not measured" marker instead of a
    fake ``0.0`` (fail-visible; note the legacy :func:`pseudospectral_radius`
    keeps its historical ``0.0`` behaviour unchanged).

    Unlike the legacy function, the caller MUST supply both grids explicitly:
    this keeps the numerics kernel free of absolute span floors -- the
    scale-relative grid policy lives in
    :func:`liouscope.diagnostics.resolvent.compute_resolvent_layer`.
    """
    L = require_finite_square_2d(L, name="L")
    res = np.linspace(*grid_re)
    ims = np.linspace(*grid_im)
    radius = float("nan")
    abscissa = float("nan")
    found = False
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    for re in res:
        for im in ims:
            z = re + 1j * im
            sv_min = float(sla.svdvals(z * eye - L)[-1])
            if sv_min <= eps:
                if not found:
                    radius = float(abs(z))
                    abscissa = float(re)
                    found = True
                else:
                    radius = max(radius, float(abs(z)))
                    abscissa = max(abscissa, float(re))
    return radius, abscissa
