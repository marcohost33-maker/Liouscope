"""Transient layer: D14 trans-amplitude ratio, D15 kappa_trans."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._types import TransientResult


def numerical_abscissa(L_super: np.ndarray) -> float:
    """Numerical abscissa ``omega(L) = max eigenvalue of (L + L^H)/2``."""
    A = 0.5 * (np.asarray(L_super) + np.asarray(L_super).conj().T)
    return float(np.max(np.linalg.eigvalsh(A)))


def trans_amplitude_ratio(
    L_super: np.ndarray,
    *,
    t_grid: np.ndarray | None = None,
) -> float:
    """D14: ``sup_t ||e^{tL}||_2`` over a coarse time grid.

    Returns the supremum operator norm of the propagator (which equals the
    relative trans-amplitude ratio for a unit-norm initial state).
    """
    if t_grid is None:
        t_grid = np.linspace(0.01, 5.0, 30)
    L_super = np.asarray(L_super)
    sup = 0.0
    for t in t_grid:
        norm = float(sla.svdvals(sla.expm(L_super * t))[0])
        if norm > sup:
            sup = norm
    return sup


def kappa_trans(omega_L: float, gap: float) -> float:
    """D15: ``kappa_trans = omega(L) / Delta`` (Patch E5).

    Returns the unbounded ratio. ``omega(L) > 0`` is the generic case in
    non-normal Liouvillians.
    """
    if gap <= 0:
        return float("inf")
    return float(omega_L / gap)


def compute_transient_layer(
    L_super: np.ndarray,
    gap: float,
    *,
    t_grid: np.ndarray | None = None,
) -> TransientResult:
    """Run D14, D15 with the spectral gap from D1."""
    omega_L = numerical_abscissa(L_super)
    ratio = trans_amplitude_ratio(L_super, t_grid=t_grid)
    kappa = kappa_trans(omega_L, gap)
    return TransientResult(
        trans_amplitude_ratio=ratio,
        kappa_trans=kappa,
        numerical_abscissa=omega_L,
    )
