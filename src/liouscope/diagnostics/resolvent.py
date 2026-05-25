"""Resolvent / pseudospectrum layer: D11b, D12, D13.

D11b -- :func:`resolvent_peak`. Peak of ``||((sigma + i omega) I - L)^{-1}||``
over a 1D scan along ``omega``. Anchor M: this is **not** D11.

D12 -- :func:`ridge_fwhm`. Full width at half-maximum of the resolvent peak
profile, reported as a gap-corrective scale.

D13 -- :func:`pseudospectral_radius_diag`. Wraps the grid evaluation from
``numerics.pseudospec``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._types import ResolventResult
from ..numerics.pseudospec import pseudospectral_radius


def resolvent_peak_curve(
    L_super: np.ndarray,
    sigma: float = 1.0e-3,
    *,
    n_omega: int = 201,  # ungerade -> das Default-Gitter enthaelt omega=0 (Peak-Lage)
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the resolvent norm profile along ``omega`` at fixed ``sigma``.

    Returns ``(omegas, norms)``.
    """
    L_super = np.asarray(L_super)
    n = L_super.shape[0]
    eigvals = sla.eigvals(L_super)
    omega_max = max(1.0, float(np.max(np.abs(np.imag(eigvals)))) + 1.0)
    omegas = np.linspace(-omega_max, omega_max, n_omega)
    eye = np.eye(n, dtype=complex)
    re_max = float(np.max(np.real(eigvals)))
    shift = sigma + re_max
    norms = np.empty(n_omega)
    for i, omega in enumerate(omegas):
        z = shift + 1j * omega
        Ainv = sla.solve(z * eye - L_super, eye)
        norms[i] = float(sla.svdvals(Ainv)[0])
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
) -> ResolventResult:
    """Run D11b, D12, D13."""
    omegas, norms = resolvent_peak_curve(L_super, sigma)
    peak = float(norms.max())
    fwhm = ridge_fwhm(omegas, norms)
    radius = pseudospectral_radius_diag(L_super, pseudo_eps)
    return ResolventResult(
        resolvent_peak=peak,
        ridge_fwhm=fwhm,
        pseudospectral_radius=radius,
        pseudospec_eps=pseudo_eps,
    )


__all__ = [
    "compute_resolvent_layer",
    "pseudospectral_radius_diag",
    "resolvent_peak",
    "resolvent_peak_curve",
    "ridge_fwhm",
]
