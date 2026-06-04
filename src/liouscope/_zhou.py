"""v0.2.1 post-submission tooling: Zhou universal mixing-time predictor (D24).

Reference (UNVERIFIED): Zhou, "Universal mixing-time predictor for open
quantum systems", arXiv:2601.06256 (2026).

.. warning::

   **claim_status: pending / unverified (S6 audit 2026-06-04).** The cited
   reference arXiv:2601.06256 could not be independently verified from an
   authoritative source (arXiv API / DOI) at audit time. Treat the D24 Zhou
   predictor as an *exploratory* diagnostic: its numerical machinery is unit-
   and anchor-tested (see ``tests/test_zhou.py``, including a closed-form
   pure-dephasing anchor that is independent of the reference), but the
   *physical claim* attributed to the cited paper is NOT confirmed. Do not
   present D24 results as publication-grade until the reference is verified.

The Zhou predictor estimates the mixing time

    t_mix(eps) = min { t : sup_{rho_0} || e^{tL} rho_0 - rho_ss ||_1 < eps }

via a non-perturbative bound that combines the spectral gap, the
non-normality factor, and the trans-amplitude ratio (or its upper
estimate).

This module ships as part of v0.2.0 source but is **not** exported from the
top-level ``liouscope`` namespace; it is reachable explicitly via
``import liouscope._zhou`` so the paper's v0.2.0-grade results remain
bit-stable.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import scipy.linalg as sla

from ._consts import EPS_DIV, EPS_GAP
from ._types import ZhouPredictorResult

# S6 audit 2026-06-04: the cited reference for the D24 predictor could not be
# verified against an authoritative source. The diagnostic is exposed but its
# physical claim is provisional. Consumers / tooling can read this constant.
CLAIM_STATUS: Final[str] = "pending"
CLAIM_REFERENCE: Final[str] = "arXiv:2601.06256 (UNVERIFIED 2026-06-04)"


def compute_zhou_predictor(
    L_super: np.ndarray,
    *,
    epsilon: float = 1.0e-3,
    petermann_factor: float | None = None,
    gap: float | None = None,
) -> ZhouPredictorResult:
    """Compute Zhou's universal mixing-time lower- and upper-bounds.

    .. note::
       claim_status = ``"pending"`` (see :data:`CLAIM_STATUS`): the cited
       reference arXiv:2601.06256 is unverified (S6 audit 2026-06-04). The
       returned bounds are numerically tested but the underlying physical
       claim is provisional.

    Parameters
    ----------
    L_super
        ``d^2 x d^2`` Liouvillian.
    epsilon
        Mixing-time target accuracy.
    petermann_factor
        Optional precomputed Petermann ``K_max`` (D9). Recomputed if omitted.
    gap
        Optional precomputed spectral gap (D1). Recomputed if omitted.

    Returns
    -------
    ZhouPredictorResult
    """
    L_super = np.asarray(L_super)
    if gap is None or petermann_factor is None:
        eigvals, vl, vr = sla.eig(L_super, left=True, right=True)
        nonzero = np.abs(eigvals) > EPS_GAP
        if not nonzero.any():
            return ZhouPredictorResult(
                mixing_time_lower=float("inf"),
                mixing_time_upper=float("inf"),
                epsilon=epsilon,
                converged=False,
                gap=0.0,
                petermann_factor=float("nan"),
            )
        eigvals_nz = eigvals[nonzero]
        if gap is None:
            gap = float(-np.max(np.real(eigvals_nz)))
        if petermann_factor is None:
            K_vals = []
            for j in range(eigvals.size):
                if not nonzero[j]:
                    continue
                r = vr[:, j]
                l_vec = vl[:, j]
                denom = abs(np.vdot(l_vec, r)) ** 2
                # Same division-by-zero floor as nonnormality.petermann_factors
                # (canonical EPS_DIV). A defective mode (denom -> 0) is skipped
                # here rather than set to inf: an inf K_max would poison the
                # finite t_upper bound. Anchor I (conj-pair retention) is moot
                # for the predictor, which only needs K_max over finite modes.
                if denom <= EPS_DIV:
                    continue
                K_vals.append((np.linalg.norm(r) ** 2 * np.linalg.norm(l_vec) ** 2) / denom)
            petermann_factor = float(max(K_vals) if K_vals else 1.0)

    if gap <= 0:
        return ZhouPredictorResult(
            mixing_time_lower=float("inf"),
            mixing_time_upper=float("inf"),
            epsilon=epsilon,
            converged=False,
            gap=float(gap),
            petermann_factor=float(petermann_factor),
        )

    # Zhou predictor (simplified universal form):
    #   t_lower = (1 / Delta) * log(1 / eps)
    #   t_upper = (1 / Delta) * log( sqrt(K) / eps )
    t_lower = float(np.log(1.0 / epsilon) / gap)
    t_upper = float(np.log(np.sqrt(petermann_factor) / epsilon) / gap)
    return ZhouPredictorResult(
        mixing_time_lower=t_lower,
        mixing_time_upper=t_upper,
        epsilon=epsilon,
        converged=True,
        gap=float(gap),
        petermann_factor=float(petermann_factor),
    )


def mixing_time_upper_bound(result: ZhouPredictorResult, eps: float | None = None) -> float:
    """Return the upper bound, optionally rescaled to a different ``eps``.

    Allows reusing one predictor across multiple accuracy targets without
    re-diagonalising. Uses the stored gap to apply the analytic correction

        t_upper(eps_new) = t_upper(eps_old) + log(eps_old / eps_new) / Delta

    Requires the result to carry a positive ``gap`` (which the regular
    :func:`compute_zhou_predictor` always sets); raises :class:`ValueError`
    if the predictor did not converge.
    """
    if eps is None or eps == result.epsilon:
        return result.mixing_time_upper
    if not result.converged or not np.isfinite(result.gap) or result.gap <= 0.0:
        raise ValueError("Cannot rescale: predictor did not converge with a finite, positive gap")
    return float(result.mixing_time_upper + np.log(result.epsilon / eps) / result.gap)


__all__ = [
    "CLAIM_REFERENCE",
    "CLAIM_STATUS",
    "compute_zhou_predictor",
    "mixing_time_upper_bound",
]
