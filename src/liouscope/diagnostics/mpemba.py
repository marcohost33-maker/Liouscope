"""Mpemba layer: D19 slowest-mode overlap, D20 expansion-coefficient scaling.

D19 -- :func:`overlap_c1`. The Mpemba-overlap test (Carollo 2021). Decompose
the initial state in left-eigenvector basis of L and test whether the
coefficient on the slowest mode (lambda_1 != 0) vanishes. If ``|c_1| << eps``,
the system relaxes anomalously fast (Mpemba candidate).

D20 -- :func:`expansion_alpha`. Scaling exponent of overlap coefficients
``|c_n|`` against the index n. Polynomial scaling ``Phi_n ~ exp(alpha L)``
signals overlap-amplification (F1).

.. caveat::

    Strong quantum Mpemba effect is **highly sensitive to preparation
    errors** (Mackinnon & Paternostro, New J. Phys. 28, 2026; NR-159). Even a
    small deviation of the prepared state from the ``c_1 = 0`` direction
    causes the asymptotic decay to revert to the gap-dominated rate. The
    classifier therefore requires both an extremely small ``overlap_c1``
    (below ``1e-5`` for high-confidence A11) **and** a clean initial-state
    sensitivity score from :func:`liouscope.diagnostics.lep.initial_state_sensitivity`
    before promoting a verdict to ``CONFIRMED``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_GAP
from .._types import MpembaResult
from ..numerics.kronecker import vec


def overlap_c1(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    *,
    atol: float = EPS_GAP,
) -> float:
    """Magnitude of the slowest-mode overlap ``|c_1|``."""
    L_super = np.asarray(L_super)
    eigvals, vl, vr = sla.eig(L_super, left=True, right=True)
    # Filter the zero mode
    nonzero_mask = np.abs(eigvals) > atol
    eigvals_nz = eigvals[nonzero_mask]
    if eigvals_nz.size == 0:
        return 0.0
    # Slowest non-zero mode = largest real part (least negative)
    order = np.argsort(-np.real(eigvals_nz))
    slowest_idx = np.where(nonzero_mask)[0][order[0]]
    l_slow = vl[:, slowest_idx]
    rho_vec0 = vec(np.asarray(rho_initial))
    norm = max(np.linalg.norm(l_slow), 1.0e-12)
    return float(abs(np.vdot(l_slow, rho_vec0)) / norm)


def expansion_alpha(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    *,
    n_modes: int = 6,
    atol: float = EPS_GAP,
) -> float:
    """Scaling exponent of overlap coefficients vs. mode-index.

    Fits ``log|c_n| = alpha * n + c0`` for the ``n_modes`` slowest modes.
    Returns ``alpha`` (slope). A flat distribution gives ``alpha`` near 0.
    """
    L_super = np.asarray(L_super)
    eigvals, vl, _ = sla.eig(L_super, left=True, right=True)
    mask = np.abs(eigvals) > atol
    eigvals_nz = eigvals[mask]
    if eigvals_nz.size < 2:
        return 0.0
    order = np.argsort(-np.real(eigvals_nz))
    nz_indices = np.where(mask)[0][order]
    rho_vec0 = vec(np.asarray(rho_initial))
    cs: list[float] = []
    for idx in nz_indices[: min(n_modes, len(nz_indices))]:
        l_n = vl[:, idx]
        norm = max(np.linalg.norm(l_n), 1.0e-12)
        cs.append(float(abs(np.vdot(l_n, rho_vec0)) / norm))
    if len(cs) < 2:
        return 0.0
    log_cs = np.log(np.clip(np.asarray(cs), 1.0e-30, None))
    n_arr = np.arange(1, len(cs) + 1, dtype=float)
    slope, _ = np.polyfit(n_arr, log_cs, 1)
    return float(slope)


def compute_mpemba_layer(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    *,
    overlap_threshold: float = 1.0e-4,
) -> MpembaResult:
    """Run D19, D20 and flag Mpemba candidacy."""
    c1 = overlap_c1(L_super, rho_initial)
    alpha = expansion_alpha(L_super, rho_initial)
    return MpembaResult(
        overlap_c1=c1,
        is_mpemba_candidate=c1 < overlap_threshold,
        expansion_alpha=alpha,
    )
