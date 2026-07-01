"""Approximate epsilon-pseudospectrum (D13).

Uses the grid-based singular-value definition

    sigma_eps(L) = { z in C : sigma_min(z I - L) <= eps }

restricted to a rectangular grid bracketing the spectrum (for a Liouvillian
that grid lies in the closed left half-plane). Returns the
maximal modulus of grid points belonging to the pseudospectrum, which we
report as the pseudospectral radius for diagnostic D13.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla


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
    L = np.asarray(L)
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
    radius = 0.0
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    for re in res:
        for im in ims:
            z = re + 1j * im
            sv_min = float(sla.svdvals(z * eye - L)[-1])
            if sv_min <= eps and abs(z) > radius:
                radius = float(abs(z))
    return radius
