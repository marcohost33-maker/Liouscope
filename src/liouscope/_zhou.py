"""v0.2.1 post-submission tooling: Zhou universal mixing-time predictor (D24).

Reference (VERIFIED 2026-06-04, v3): Yi-Neng Zhou, "Universal Predictors for
Mixing Time more than Liouvillian Gap", arXiv:2601.06256 (submitted 2026-01-09;
v3 2026-05-20; quant-ph; Dept. of Theoretical Physics, University of Geneva).

.. note::

   **claim_status: reference-verified-bound-coarser (S6 re-audit 2026-06-04).**
   The cited reference *exists and was independently verified* against the
   arXiv PDF (v3). Our implemented bound is in the *same family* as Zhou's
   central result Eq.(16) (a gap-scaled logarithm of a non-normality
   prefactor over ``eps``) and is congruent in the normal-mode limit
   (verified by the closed-form pure-dephasing anchor in
   ``tests/test_zhou.py``), **but it is not a verbatim implementation of
   Eq.(16)**. The differences are exact and documented below; treat the
   numerical machinery as audited but read the upper bound as a *related,
   generally coarser* surrogate of Zhou's per-mode bound, not as Eq.(16).

The Zhou predictor estimates the mixing time

    t_mix(eps) = min { t : sup_{rho_0} D( e^{tL}[rho_0], rho_ss ) < eps }

where ``D`` is the trace distance. Zhou's central upper bound (paper Eq. 16) is

    t_mix(eps) <= max_{j: lambda_j > 0} (1 / lambda_j)
                  * log( N_mode * C_j / eps ),

where ``lambda_j`` is the decay rate of the j-th decaying Liouvillian
eigenmode, ``N_mode`` the number of nonzero eigenmodes (<= N^2 - 1), and
``C_j`` the per-mode *trace-norm factor* (paper Eqs. 10-12)

    C_j = || P_j ||_{1->1} = || rho_j ||_1 * || sigma_j ||_op,         (Eq. 12)

with ``rho_j`` the right eigen-operator and ``sigma_j`` the dual left
eigen-operator of the mode, normalised by ``Tr(sigma_i rho_j) = delta_ij``.

How our bound differs from Eq.(16) (the reason for the "-coarser" status):

* **Norm.** We use the Petermann factor ``K_j = ||r_j||_2^2 ||l_j||_2^2 /
  |<l_j, r_j>|^2`` (Schatten-2 / Euclidean on the vectorised eigenvectors),
  so ``sqrt(K_j) = ||rho_j||_2 * ||sigma_j||_2`` under the duality
  normalisation. Zhou's ``C_j = ||rho_j||_1 * ||sigma_j||_op`` mixes the
  Schatten-1 and Schatten-inf norms. These coincide only for rank-one /
  normal modes; in general ``sqrt(K_j) != C_j`` (neither dominates the
  other), so our prefactor is a *different* non-normality measure, not Zhou's.
* **Global vs. per-mode.** Zhou maximises ``(1/lambda_j) * log(N_mode C_j)``
  over *every* decaying mode (each with its own decay rate ``lambda_j``). We
  collapse this to a single global ``gap = lambda_min`` and a single global
  ``K_max = max_j K_j``. This is the special case "all modes referred to the
  slowest rate, worst-case K", which is generally *coarser* (a fast-decaying
  mode with large ``C_j`` is damped by ``1/lambda_j`` in Eq.16 but inflated
  by ``1/gap >= 1/lambda_j`` here).
* **N_mode prefactor.** We omit Zhou's ``N_mode`` multiplier inside the log
  (an ``O(log N^2)`` additive shift to the bound).

In the normal / single-mode limit all three differences vanish: ``C_j = 1``,
``K_j = 1``, ``N_mode`` contributes only a constant, and both Zhou's Eq.(16)
and our bound collapse onto ``log(1/eps)/gap`` -- exactly the closed-form
pure-dephasing anchor (``tests/test_zhou.py::
test_anchor_zhou_pure_dephasing_closed_form``).

This module ships as part of v0.2.0 source but is **not** exported from the
top-level ``liouscope`` namespace; it is reachable explicitly via
``import liouscope._zhou`` so the paper's v0.2.0-grade results remain
bit-stable.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from ._consts import EPS_DIV
from ._types import ZhouPredictorResult
from .numerics.linalg import certified_eig, certified_eigvals
from .numerics.scale import spectral_zero_tolerance

# S6 re-audit 2026-06-04: the cited reference is independently verified to
# exist (arXiv PDF v3). Our implemented bound is the same family as Zhou's
# Eq.(16) and congruent in the normal-mode limit, but it is a *related,
# generally coarser* surrogate (different Schatten norm, global gap/K_max
# instead of per-mode, no N_mode factor) -- see module docstring. The status
# below records exactly that; consumers / tooling can read these constants.
CLAIM_STATUS: Final[str] = "reference-verified-bound-coarser"
CLAIM_REFERENCE: Final[str] = (
    "Yi-Neng Zhou, 'Universal Predictors for Mixing Time more than Liouvillian "
    "Gap', arXiv:2601.06256 (verified 2026-06-04, v3 2026-05-20)"
)


def compute_zhou_predictor(
    L_super: np.ndarray,
    *,
    epsilon: float = 1.0e-3,
    petermann_factor: float | None = None,
    gap: float | None = None,
) -> ZhouPredictorResult:
    """Compute Zhou's universal mixing-time lower- and upper-bounds.

    .. note::
       claim_status = ``"reference-verified-bound-coarser"`` (see
       :data:`CLAIM_STATUS`): the cited reference arXiv:2601.06256 is
       independently verified (S6 re-audit 2026-06-04, v3). The returned
       bounds are numerically anchor-tested and in the same family as Zhou's
       Eq.(16), but use the Petermann (Schatten-2) factor and a global
       ``gap``/``K_max`` rather than Zhou's per-mode trace-norm factor
       ``C_j``; the upper bound is therefore a related, generally coarser
       surrogate. See the module docstring for the exact differences.

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
    L_super = np.asarray(L_super, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
    if gap is None or petermann_factor is None:
        # Round-13 review: D24's recomputation must not retain a solver
        # failure that the spectral and Mpemba paths repair. On the stiff
        # #112 fixture the raw ``zgeev`` spectrum made D24 report
        # ``gap = 7.28e-6`` where the certified solve recovers the physical
        # ``1.074e-5`` -- a ~30% shift of the whole mixing-time window.
        #
        # The certificate matches what is CONSUMED (round-16 review): the
        # Petermann recomputation reads eigenvectors and needs the stricter
        # ``certified_eig``, but when the caller supplied ``petermann_factor``
        # and only the gap is missing, no eigenvectors are consumed and the
        # (wider-ladder) eigenvalue certificate decides -- measured: a stiff
        # network whose eigenvalue certificate resolves a usable gap of
        # ``3.32e-6`` while only the eigenvector gate fails returned an
        # unnecessary unconverged record here.
        need_vectors = petermann_factor is None
        if need_vectors:
            decomp, certificate = certified_eig(L_super)
            eigvals = decomp.eigenvalues
            vl, vr = decomp.left_vectors, decomp.right_vectors
            if vl is None:  # pragma: no cover - certified_eig sets left vectors
                raise RuntimeError("certified_eig did not return left eigenvectors")
        else:
            eigvals, certificate = certified_eigvals(L_super)
            vl = vr = None
        if certificate.applicable and not certificate.resolved:
            # The eigendecomposition is demonstrably unreliable (failed
            # certification, or ambiguous in-band modes, issues #112/#113).
            # A predictor built on it would be a claim the arithmetic cannot
            # support: return an unconverged record, honouring any
            # caller-supplied values exactly like the no-nonzero-modes branch.
            return ZhouPredictorResult(
                mixing_time_lower=float("inf"),
                mixing_time_upper=float("inf"),
                epsilon=epsilon,
                converged=False,
                gap=float(gap) if gap is not None else float("nan"),
                petermann_factor=(
                    float(petermann_factor)
                    if petermann_factor is not None
                    else float("nan")
                ),
            )
        # Zero-mode separation on the certificate's own operator-derived
        # scale (round-13): the radius-based proxy is smaller than the
        # eigensolve backward error on strongly non-normal generators, so a
        # certified-resolved stationary mode would survive it as a spurious
        # slow mode. ONLY when the certificate is applicable (round-14):
        # without established trace preservation no zero mode is guaranteed
        # and the operator-norm bound can exceed the whole spectrum of a
        # strongly non-normal input, discarding every eigenvalue; the
        # radius-based filter is the honest fallback there. Scale-relative in
        # either case (issue #108): D24 consumes the gap, so an absolute
        # floor made the predicted mixing-time window depend on the choice of
        # rate unit.
        nonzero = np.abs(eigvals) > (
            certificate.bound
            if certificate.applicable
            else spectral_zero_tolerance(eigvals, name="eigenvalues of L_super")
        )
        if not nonzero.any():
            # Honour caller-supplied values in the unconverged record:
            # hard-coding gap=0.0 here silently overwrote an explicitly
            # passed ``gap``/``petermann_factor``, so the returned
            # (manifest-grade) result contradicted its own call arguments.
            return ZhouPredictorResult(
                mixing_time_lower=float("inf"),
                mixing_time_upper=float("inf"),
                epsilon=epsilon,
                converged=False,
                gap=float(gap) if gap is not None else 0.0,
                petermann_factor=(
                    float(petermann_factor)
                    if petermann_factor is not None
                    else float("nan")
                ),
            )
        eigvals_nz = eigvals[nonzero]
        if gap is None:
            gap = float(-np.max(np.real(eigvals_nz)))
        if petermann_factor is None:
            assert vl is not None and vr is not None  # need_vectors branch above
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
