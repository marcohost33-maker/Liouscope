"""Approximate epsilon-pseudospectrum (D13).

Uses the grid-based singular-value definition

    sigma_eps(L) = { z in C : sigma_min(z I - L) <= eps }

restricted to a rectangular grid bracketing the spectrum (for a Liouvillian
that grid lies in the closed left half-plane). Returns the
maximal modulus of grid points belonging to the pseudospectrum, which we
report as the pseudospectral radius for diagnostic D13.

RESOLUTION CONTRACT (2026-09-09, cross-family finding A). Every estimator here
is a GRID LOWER BOUND on the true supremum: finite sampling, no globality
certificate. When NO grid node satisfies ``sigma_min <= eps`` the sweep did not
measure the pseudospectrum at all, and every function in this module reports
that as ``nan`` -- never as ``0.0``, which is a *plausible measurement* ("the
eps-pseudospectrum sits at the origin"). The three estimators share ONE sweep
(:func:`_sweep`) so they cannot drift apart on this contract; two
implementations of the same quantity disagreeing (``0.0`` vs ``nan`` on
byte-identical input) is exactly how the defect stayed invisible.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .linalg import require_finite_square_2d


class _Sweep:
    """Result of one grid sweep: the two extents, the resolution floor, hit flag."""

    __slots__ = ("abscissa", "found", "radius", "sigma_floor")

    def __init__(self, radius: float, abscissa: float, sigma_floor: float, found: bool):
        self.radius = radius
        self.abscissa = abscissa
        self.sigma_floor = sigma_floor
        self.found = found


def _sweep(
    L: np.ndarray,
    eps: float,
    res: np.ndarray,
    ims: np.ndarray,
) -> _Sweep:
    """Evaluate ``sigma_min(z I - L)`` on the grid once and reduce it four ways.

    Single source of truth for the sweep, so ``pseudospectral_radius``,
    ``pseudospectrum_extent`` and ``pseudospectrum_sigma_floor`` cannot report
    contradictory things about the same grid (they did: see the module
    docstring). ``sigma_floor`` is free here -- it reuses the SVDs the
    membership test needs anyway.
    """
    radius = float("nan")
    abscissa = float("nan")
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
                if not found:
                    radius = mod
                    abscissa = float(re)
                    found = True
                else:
                    radius = max(radius, mod)
                    abscissa = max(abscissa, float(re))
    return _Sweep(radius, abscissa, sigma_floor, found)


def _unresolved_message(eps: float, sigma_floor: float) -> str:
    """Say what to do, not merely that something failed."""
    if not np.isfinite(sigma_floor):
        return (
            f"eps-pseudospectrum unresolved: the grid is EMPTY, so "
            f"sigma_min was never evaluated for eps={eps!r}. Returning nan "
            f"(UNKNOWN), not 0.0."
        )
    return (
        f"eps-pseudospectrum unresolved: no grid node satisfies "
        f"sigma_min <= eps={eps!r} (smallest sigma_min on this grid was "
        f"{sigma_floor:.6e}). Returning nan (UNKNOWN), not 0.0. Refine the "
        f"grid, or raise eps to at least {sigma_floor:.6e} -- that is the "
        f"smallest eps THIS grid can resolve."
    )


def _default_grids(
    eigvals: np.ndarray,
    grid_re: tuple[float, float, int] | None,
    grid_im: tuple[float, float, int] | None,
) -> tuple[tuple[float, float, int], tuple[float, float, int]]:
    """The historical default bracketing grid (unchanged: half-span pad, 25x25)."""
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
    return grid_re, grid_im


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

    Returns
    -------
    float
        The grid lower bound on the radius, or ``nan`` if the sweep resolved
        NOTHING (with a :class:`RuntimeWarning` naming the smallest ``eps``
        this grid could resolve).

    Notes
    -----
    FAIL-VISIBLE since 2026-09-09 (cross-family finding A). This function used
    to initialise its accumulator to ``0.0`` and return it when no grid node
    satisfied ``sigma_min <= eps``. ``0.0`` is not a failure marker: it is a
    plausible pseudospectral radius, so an unresolved sweep was
    indistinguishable from a measured one -- the inverse of a guard that
    produces no result, namely a guard that produces a WRONG one.

    Why this is not a corner case: for a well-separated spectrum the
    eps-pseudospectrum at ``eps = 1e-3`` is a union of discs of radius ~eps
    around the eigenvalues, and a rectangular grid sees it only when a node
    falls INSIDE such a disc. The default grid does so by accident, not by
    construction -- ``np.linspace`` over ``[re_min - s/2, re_max + s/2]`` with
    25 nodes places nodes exactly on ``re_min``/``re_max`` (indices 6 and 18),
    i.e. on the extremal eigenvalues, where ``sigma_min`` is 0. Supply your own
    grid and the same operator reported ``0.0`` at ANY resolution: a normal
    matrix with spectrum ``{0, -0.2, -1+-3i, -4}`` and true radius ``4.001``
    returned ``0.0`` from 25x25 through 201x201 (40'401 SVDs). Refinement could
    never help, so the value measured grid geometry, not the operator.

    It reached a decision: the value feeds ``pseudospectral_radius_diag`` ->
    ``ResolventResult`` -> the F5 phantom-relaxation rung, whose reach leg
    tests ``radius / gap > 2 * gap_to_gns_ratio``. With ``0.0`` that is False,
    so an unmeasured D13 silently suppressed F5 and the run reported "no
    pseudospectral intrusion" on evidence it never had -- fail-open in a
    classifier. With ``nan`` the classifier's own machinery takes over:
    ``_strip_unavailable`` removes the NaN key, ``pseudospectral_radius`` is a
    REQUIRED key of the F5 rung, so the rung is UNEVALUABLE instead of
    NOT_SUPPORTED. The rung still does not fire (``nan > x`` is False exactly
    as ``0.0 > x`` was), so no verdict changes; what changes is that the gap is
    visible in the evidence matrix instead of being indistinguishable from a
    measured negative.

    NOT changed here, deliberately: replacing grid membership by a level-set
    method (BLO criss-cross for the abscissa, radial sweep for the radius).
    That is class-influencing, needs calibration, and belongs in its own PR.
    """
    # Fail closed on non-finite / non-square input (consistent with linalg.py and
    # cptp.py) so a NaN/inf-laden operator surfaces a located, named error instead
    # of an opaque LAPACK failure deep in the svd loop.
    L = require_finite_square_2d(L, name="L")
    grid_re, grid_im = _default_grids(sla.eigvals(L), grid_re, grid_im)
    sweep = _sweep(L, eps, np.linspace(*grid_re), np.linspace(*grid_im))
    if not sweep.found:
        warnings.warn(
            _unresolved_message(eps, sweep.sigma_floor),
            RuntimeWarning,
            stacklevel=2,
        )
    return sweep.radius


def pseudospectrum_sigma_floor(
    L: np.ndarray,
    *,
    grid_re: tuple[float, float, int],
    grid_im: tuple[float, float, int],
) -> float:
    """Return ``min{sigma_min(z I - L) : z in grid}`` -- the grid's resolution floor.

    This is the *resolution* companion to the radius/extent estimators: it is
    the smallest ``eps`` for which the given grid resolves the
    eps-pseudospectrum at all (the argmin node satisfies ``sigma_min <= eps``
    by construction for any ``eps >= floor``, and no node does for a smaller
    one). Reporting it turns "the sweep found nothing" from an unexplained NaN
    into a located, actionable number -- a guard should say what to do, not
    only that something is missing.

    Returns ``inf`` for an empty grid: no node was evaluated, so no eps would
    have helped. Note this is a property of the GRID and the operator jointly,
    not of the operator alone.
    """
    L = require_finite_square_2d(L, name="L")
    # eps = -inf: the membership test can never fire, so this is a pure
    # sigma_min reduction over the same code path the other estimators use.
    return _sweep(
        L, float("-inf"), np.linspace(*grid_re), np.linspace(*grid_im)
    ).sigma_floor


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
    fake ``0.0``. This function was built fail-visible for issue #101; since
    2026-09-09 the legacy :func:`pseudospectral_radius` agrees with it (it used
    to return ``0.0`` on the same input) and both share :func:`_sweep`.

    Unlike the legacy function, the caller MUST supply both grids explicitly:
    this keeps the numerics kernel free of absolute span floors -- the
    scale-relative grid policy lives in
    :func:`liouscope.diagnostics.resolvent.compute_resolvent_layer`.

    Deliberately SILENT on an unresolved sweep, unlike
    :func:`pseudospectral_radius`: the NaN pair is this function's documented
    contract, its callers already branch on it, and the warning belongs where
    the value historically lied.
    """
    L = require_finite_square_2d(L, name="L")
    sweep = _sweep(L, eps, np.linspace(*grid_re), np.linspace(*grid_im))
    return sweep.radius, sweep.abscissa
