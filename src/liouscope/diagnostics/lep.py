"""LEP layer: D16 Liouvillian-exceptional-point proximity, D17 gap-rate
consistency, D18 initial-state sensitivity.

Anchor I: Conjugate-pair eigenvalues are INCLUDED in the LEP scan. The
original v0.1 implementation excluded them (FIX-3 / external audit), which
hid genuine non-diagonalisability cases (Tay 2023, Claeys 2022).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_GAP
from .._types import LepResult
from ..core.lindblad import steady_state
from ..numerics.kronecker import unvec, vec


def lep_proximity(eigenvalues: np.ndarray, *, atol: float = EPS_GAP) -> tuple[float, int]:
    """Minimum eigenvalue-pair separation, including complex-conjugate pairs.

    Returns ``(min_sep, candidate_count)`` where ``candidate_count`` is the
    number of pairs within ``10 * min_sep`` of each other.
    """
    eigenvalues = np.asarray(eigenvalues)
    n = eigenvalues.size
    if n < 2:
        return float("inf"), 0
    min_sep = float("inf")
    pairs_close = 0
    for i in range(n):
        for j in range(i + 1, n):
            sep = float(abs(eigenvalues[i] - eigenvalues[j]))
            if sep < min_sep and sep > atol:
                min_sep = sep
    if not np.isfinite(min_sep):
        return float("inf"), 0
    for i in range(n):
        for j in range(i + 1, n):
            sep = float(abs(eigenvalues[i] - eigenvalues[j]))
            if sep <= 10.0 * min_sep:
                pairs_close += 1
    return min_sep, pairs_close


def gap_rate_consistency(beta_D: float, gap: float) -> float:
    """D17: ``|beta_D - Delta| / Delta``. Returns inf if gap is zero."""
    if gap <= 0:
        return float("inf")
    if not np.isfinite(beta_D):
        return float("inf")
    return float(abs(beta_D - gap) / gap)


def initial_state_sensitivity(
    L_super: np.ndarray,
    rho_steady_state: np.ndarray,
    *,
    n_samples: int = 10,
    t_eval: float = 1.0,
    seed: int = 7,
) -> float:
    """D18: std of relaxation distance over a Haar-random initial-state ensemble.

    Samples ``n_samples`` Haar-random pure states (or thermal in the
    Hermitian case), evolves to ``t_eval`` and measures
    ``||rho(t) - rho_ss||_F``. Returns the standard deviation across samples.
    """
    L_super = np.asarray(L_super)
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    rng = np.random.default_rng(seed)
    expm_t = sla.expm(L_super * t_eval)
    distances = np.empty(n_samples)
    for k in range(n_samples):
        psi = rng.normal(size=d) + 1j * rng.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())
        rho_t = unvec(expm_t @ vec(rho0), d=d)
        distances[k] = float(np.linalg.norm(rho_t - rho_steady_state, ord="fro"))
    return float(np.std(distances))


def compute_lep_layer(
    L_super: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    beta_D: float,
    gap: float,
    rho_steady_state: np.ndarray | None = None,
    seed: int = 7,
    n_haar: int = 10,
) -> LepResult:
    """Run D16, D17, D18 together."""
    L_super = np.asarray(L_super)
    if rho_steady_state is None:
        rho_steady_state = steady_state(L_super)
    proximity, candidates = lep_proximity(eigenvalues)
    consistency = gap_rate_consistency(beta_D, gap)
    sensitivity = initial_state_sensitivity(
        L_super, rho_steady_state, n_samples=n_haar, seed=seed
    )
    return LepResult(
        lep_proximity=proximity if np.isfinite(proximity) else float("inf"),
        gap_rate_consistency=consistency,
        initial_state_sensitivity=sensitivity,
        lep_candidate_count=candidates,
    )
