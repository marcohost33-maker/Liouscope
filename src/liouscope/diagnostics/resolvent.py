"""Resolvent / pseudospectrum layer: D11b, D12, D13 (+ scale-relative variants).

D11b -- :func:`resolvent_peak`. Peak of ``||((sigma + i omega) I - L)^{-1}||``
over a 1D scan along ``omega``. Anchor M: this is **not** D11.

D12 -- :func:`ridge_fwhm`. Full width at half-maximum of the resolvent peak
profile, reported as a gap-corrective scale.

D13 -- :func:`pseudospectral_radius_diag`. Wraps the grid evaluation from
``numerics.pseudospec``.

Scale-relative additions (issue #101 slice A; ``claim_status: pending``,
advisory-only): the legacy defaults above use ABSOLUTE rate constants
(``sigma = 1e-3``, ``eps = 1e-3``, ``omega_max >= 1``, span floors ``1e-3``),
so a pure unit rescale ``L -> cL`` compares different dimensionless offsets and
perturbation families and changes the values beyond the physical ``~c``
scaling (verified in the #101 re-audit). :func:`compute_resolvent_layer` now
ADDITIONALLY evaluates the same quantities on grids parameterised in
dimensionless coordinates relative to ``rate_scale(L) = ||L||_F`` and reports:

* ``resolvent_peak_scaled = rate_scale * peak(sigma = sigma_rel * rate_scale)``
  -- dimensionless, exactly invariant under ``L -> cL``;
* ``ridge_fwhm_rel = fwhm / rate_scale`` -- dimensionless ridge width;
* ``pseudospectral_radius_rel`` / ``pseudospectral_abscissa(_rel)`` -- D13 with
  ``eps_abs = pseudo_eps_rel * rate_scale`` plus the gap-directed intrusion
  (max Re z) from :func:`liouscope.numerics.pseudospec.pseudospectrum_extent`;
  the ``_rel`` values are dimensionless, the bare abscissa is rate-valued.

All legacy fields are computed byte-identically to before; the new fields are
additive. Zero-operator semantics: ``rate_scale == 0`` leaves every
scale-relative field NaN (there is no scale to normalise by -- documented, not
silent).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._types import ResolventResult
from ..numerics.pseudospec import pseudospectral_radius, pseudospectrum_extent
from ..numerics.resolvent import resolvent_norm
from ..numerics.scale import rate_scale


def resolvent_peak_curve(
    L_super: np.ndarray,
    sigma: float = 1.0e-3,
    *,
    n_omega: int = 201,  # ungerade -> das Default-Gitter enthaelt omega=0 (Peak-Lage)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the resolvent norm profile along ``omega`` at fixed ``sigma``.

    Returns ``(omegas, norms)``.

    Each point delegates to :func:`liouscope.numerics.resolvent.resolvent_norm`,
    which is exact (dense inverse + SVD) for ``n <= 128`` and switches to the
    SuperLU shift-and-invert power iteration for larger Liouvillians (where the
    dense ``O(n^3)`` inverse per frequency would be intractable).
    """
    L_super = np.asarray(L_super)
    eigvals = sla.eigvals(L_super)
    omega_max = max(1.0, float(np.max(np.abs(np.imag(eigvals)))) + 1.0)
    omegas = np.linspace(-omega_max, omega_max, n_omega)
    re_max = float(np.max(np.real(eigvals)))
    shift = sigma + re_max
    norms = np.empty(n_omega)
    for i, omega in enumerate(omegas):
        norms[i] = resolvent_norm(L_super, shift + 1j * omega)
    return omegas, norms


def ridge_fwhm(omegas: np.ndarray, norms: np.ndarray) -> float:
    """Full-width-at-half-maximum of the resolvent ridge."""
    peak = float(norms.max())
    if peak <= 0:
        return 0.0
    half = 0.5 * peak
    above = norms >= half
    if not above.any():
        return 0.0
    idx = np.where(above)[0]
    lo, hi = idx[0], idx[-1]
    return float(omegas[hi] - omegas[lo])


def resolvent_peak(L_super: np.ndarray, sigma: float = 1.0e-3) -> float:
    """D11b peak value of ``||((sigma + i omega) I - L)^{-1}||``."""
    _, norms = resolvent_peak_curve(L_super, sigma)
    return float(norms.max())


def resolvent_peak_curve_scaled(
    L_super: np.ndarray,
    sigma_rel: float = 1.0e-3,
    *,
    n_omega: int = 201,  # ungerade -> das Gitter enthaelt omega=0 (Peak-Lage)
) -> tuple[np.ndarray, np.ndarray]:
    """Scale-relative D11b curve: resolvent norms at ``sigma = sigma_rel * ||L||_F``.

    The ``omega`` grid is built in dimensionless coordinates
    ``omega_rel in [-max(1, im_max/scale + 1), +...]`` and converted back to
    absolute rate units, so the whole scan commutes exactly with a positive
    unit rescale ``L -> cL`` (the returned ``omegas`` scale by ``c``, the norms
    by ``1/c``, and dimensionless products such as ``scale * peak`` are
    invariant). Returns ``(omegas, norms)`` in ABSOLUTE units.

    Raises ``ValueError`` for the zero operator (``rate_scale == 0``): there is
    no scale to place the scan at (documented zero-operator semantics; the
    layer-level wrapper reports NaN instead of calling this).
    """
    L_super = np.asarray(L_super)
    if sigma_rel <= 0.0:
        raise ValueError(f"sigma_rel must be > 0, got {sigma_rel}")
    scale = rate_scale(L_super, name="L_super")
    if scale == 0.0:
        raise ValueError(
            "resolvent_peak_curve_scaled: rate_scale(L) == 0 (zero operator); "
            "scale-relative resolvent diagnostics are undefined"
        )
    eigvals = sla.eigvals(L_super)
    omega_rel_max = max(1.0, float(np.max(np.abs(np.imag(eigvals)))) / scale + 1.0)
    omegas = np.linspace(-omega_rel_max * scale, omega_rel_max * scale, n_omega)
    re_max = float(np.max(np.real(eigvals)))
    shift = sigma_rel * scale + re_max
    norms = np.empty(n_omega)
    for i, omega in enumerate(omegas):
        norms[i] = resolvent_norm(L_super, shift + 1j * omega)
    return omegas, norms


def pseudospectral_radius_diag(
    L_super: np.ndarray,
    eps: float = 1.0e-3,
) -> float:
    """D13 pseudospectral radius wrapper."""
    return pseudospectral_radius(np.asarray(L_super), eps)


def compute_resolvent_layer(
    L_super: np.ndarray,
    *,
    sigma: float = 1.0e-3,
    pseudo_eps: float = 1.0e-3,
    sigma_rel: float = 1.0e-3,
    pseudo_eps_rel: float = 1.0e-3,
) -> ResolventResult:
    """Run D11b, D12, D13 plus their scale-relative variants (issue #101 slice A).

    The legacy ``sigma`` / ``pseudo_eps`` path is evaluated exactly as before
    (byte-identical fields). ``sigma_rel`` / ``pseudo_eps_rel`` parameterise the
    additive scale-relative variants in units of ``rate_scale(L) = ||L||_F``;
    for the zero operator (``rate_scale == 0``) those fields stay NaN.
    """
    omegas, norms = resolvent_peak_curve(L_super, sigma)
    peak = float(norms.max())
    fwhm = ridge_fwhm(omegas, norms)
    radius = pseudospectral_radius_diag(L_super, pseudo_eps)

    scale = rate_scale(L_super, name="L_super")
    peak_scaled = float("nan")
    fwhm_rel = float("nan")
    radius_rel = float("nan")
    abscissa = float("nan")
    abscissa_rel = float("nan")
    if scale > 0.0:
        omegas_s, norms_s = resolvent_peak_curve_scaled(L_super, sigma_rel)
        peak_scaled = float(scale * norms_s.max())
        fwhm_rel = float(ridge_fwhm(omegas_s, norms_s) / scale)
        # Scale-relative D13 grid: identical construction to the legacy default
        # (half-span padding around the spectrum, 25 x 25) but with the span
        # floor and eps expressed relative to the operator scale, so the grid,
        # the perturbation family AND the reported dimensionless values commute
        # exactly with L -> cL.
        eigvals = sla.eigvals(np.asarray(L_super))
        re_max = float(np.max(np.real(eigvals)))
        re_min = float(np.min(np.real(eigvals)))
        im_max = float(np.max(np.imag(eigvals)))
        im_min = float(np.min(np.imag(eigvals)))
        span_re = max(1.0e-3 * scale, re_max - re_min)
        span_im = max(1.0e-3 * scale, im_max - im_min)
        radius_abs, abscissa_abs = pseudospectrum_extent(
            np.asarray(L_super),
            pseudo_eps_rel * scale,
            grid_re=(re_min - 0.5 * span_re, re_max + 0.5 * span_re, 25),
            grid_im=(im_min - 0.5 * span_im, im_max + 0.5 * span_im, 25),
        )
        radius_rel = float(radius_abs / scale)
        abscissa = float(abscissa_abs)
        abscissa_rel = float(abscissa_abs / scale)

    return ResolventResult(
        resolvent_peak=peak,
        ridge_fwhm=fwhm,
        pseudospectral_radius=radius,
        pseudospec_eps=pseudo_eps,
        rate_scale=scale,
        sigma_rel=sigma_rel,
        resolvent_peak_scaled=peak_scaled,
        ridge_fwhm_rel=fwhm_rel,
        pseudospec_eps_rel=pseudo_eps_rel,
        pseudospectral_radius_rel=radius_rel,
        pseudospectral_abscissa=abscissa,
        pseudospectral_abscissa_rel=abscissa_rel,
    )


__all__ = [
    "compute_resolvent_layer",
    "pseudospectral_radius_diag",
    "resolvent_peak",
    "resolvent_peak_curve",
    "resolvent_peak_curve_scaled",
    "ridge_fwhm",
]
