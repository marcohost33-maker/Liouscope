"""CPTP verification gate via the Choi matrix (LIOU-A-011).

A finite-dimensional GKSL generator ``L`` produces a *completely positive,
trace-preserving* (CPTP) propagator ``Phi_dt = exp(dt * L)`` for every
``dt >= 0`` by construction. This module gives an **independent** numerical
witness of that property for small dense systems, closing a correctness seam
the entry calls out explicitly:

    Euler-step positivity ``rho + dt*L(rho) >= 0`` on sampled ``rho`` is NOT a
    proof of complete positivity.

Complete positivity is decided by the Choi-Jamiolkowski matrix

    J(Phi) = sum_{i,j} Phi(|i><j|) (x) |i><j|

via the Choi theorem (Choi 1975): ``Phi`` is CP iff ``J(Phi)`` is positive
semi-definite. The trace-preservation condition is the generator identity
``<<I| M_L = 0`` (the all-identity covector annihilates ``L``).

Scope (entry NR-002): this is a **dense-only** CP *proof*. For sparse/large
systems the GKSL construction path is the CP guarantee and randomised
positivity sampling is only a sanity proxy -- never a CP theorem.

The Choi construction here is convention-agnostic: it applies the channel to
the basis matrices ``|i><j|`` through the package's own column-stacking
``vec``/``unvec`` primitives, so it stays consistent with
:mod:`liouscope.core.lindblad` regardless of the stacking choice. The minimum
eigenvalue of ``J`` is invariant under the tensor-factor order, so the PSD test
does not depend on whether one writes ``Phi(E_ij) (x) E_ij`` or the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .kronecker import unvec, vec

# CP boundary tolerance: channels on the CP boundary (e.g. full dephasing,
# amplitude damping) have a genuine zero Choi eigenvalue, so a small negative
# numerical excursion must be tolerated. Trace preservation is exact up to
# round-off of the matrix exponential / generator build.
TOL_CHOI: float = 1.0e-9
TOL_TP: float = 1.0e-9


@dataclass(frozen=True, slots=True)
class ChoiGateResult:
    """Outcome of the CPTP Choi gate.

    Attributes
    ----------
    min_eig
        Smallest eigenvalue of the (Hermitised) Choi matrix. ``>= -tol_choi``
        certifies complete positivity.
    tp_residual
        ``|| <<I| M_L ||`` -- the trace-preservation residual of the generator
        (``0`` for an exact GKSL generator).
    is_cp
        ``min_eig >= -tol_choi``.
    is_tp
        ``tp_residual <= tol_tp``.
    is_cptp
        ``is_cp and is_tp``.
    """

    min_eig: float
    tp_residual: float
    is_cp: bool
    is_tp: bool
    is_cptp: bool


def choi_matrix(channel_super: np.ndarray) -> np.ndarray:
    """Choi matrix of a superoperator acting on column-stacked density matrices.

    ``channel_super`` is a ``(d^2, d^2)`` matrix mapping ``vec(rho)`` to
    ``vec(Phi(rho))``. Returns the ``(d^2, d^2)`` Choi matrix
    ``J = sum_{ij} Phi(|i><j|) (x) |i><j|``.
    """
    channel_super = np.asarray(channel_super)
    n2 = channel_super.shape[0]
    if channel_super.ndim != 2 or channel_super.shape[0] != channel_super.shape[1]:
        raise ValueError(
            f"channel_super must be a square (d^2, d^2) matrix, got "
            f"{channel_super.shape}"
        )
    d = int(round(np.sqrt(n2)))
    if d * d != n2:
        raise ValueError(
            f"channel_super dimension {n2} is not a perfect square (d^2)"
        )
    J = np.zeros((n2, n2), dtype=complex)
    for i in range(d):
        for j in range(d):
            E_ij = np.zeros((d, d), dtype=complex)
            E_ij[i, j] = 1.0
            phi_E = unvec(channel_super @ vec(E_ij), d=d)
            J += np.kron(phi_E, E_ij)
    return J


def cptp_choi_gate(
    L_super: np.ndarray,
    dt: float,
    *,
    tol_choi: float = TOL_CHOI,
    tol_tp: float = TOL_TP,
) -> ChoiGateResult:
    """Verify CPTP of ``Phi_dt = exp(dt * L_super)`` via the Choi PSD criterion.

    Parameters
    ----------
    L_super
        Dense GKSL generator ``M_L`` of shape ``(d^2, d^2)``.
    dt
        Non-negative propagation time. ``dt == 0`` yields the identity channel.
    tol_choi
        Negative-eigenvalue tolerance for the PSD (complete-positivity) test.
    tol_tp
        Tolerance for the trace-preservation residual ``|| <<I| M_L ||``.

    Returns
    -------
    ChoiGateResult

    Raises
    ------
    ValueError
        If ``dt`` is negative or ``L_super`` is not a square ``(d^2, d^2)``
        matrix. Negative ``dt`` would propagate *backwards*, which is generally
        not CP and would silently mislabel a valid forward generator -- so it is
        rejected up front rather than producing a meaningless verdict.
    """
    L_super = np.asarray(L_super, dtype=complex)
    if L_super.ndim != 2 or L_super.shape[0] != L_super.shape[1]:
        raise ValueError(f"L_super must be square, got shape {L_super.shape}")
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    if d * d != n2:
        raise ValueError(f"L_super dimension {n2} is not a perfect square (d^2)")
    if not np.isfinite(dt):
        raise ValueError(f"dt must be finite, got {dt}")
    if dt < 0.0:
        raise ValueError(
            f"dt must be non-negative (forward propagation); got {dt}. "
            "Backward propagation exp(dt*L) with dt<0 is generally not CP."
        )

    phi = sla.expm(dt * L_super) if dt > 0.0 else np.eye(n2, dtype=complex)
    J = choi_matrix(phi)
    J_herm = 0.5 * (J + J.conj().T)
    min_eig = float(np.linalg.eigvalsh(J_herm)[0])

    # Trace preservation: the all-identity covector <<I| annihilates the
    # generator. <<I| = vec(I)^T in column stacking; the channel is TP iff
    # vec(I)^T M_L == 0.
    iden_vec = vec(np.eye(d, dtype=complex))
    tp_residual = float(np.linalg.norm(iden_vec.conj() @ L_super))

    is_cp = min_eig >= -tol_choi
    is_tp = tp_residual <= tol_tp
    return ChoiGateResult(
        min_eig=min_eig,
        tp_residual=tp_residual,
        is_cp=is_cp,
        is_tp=is_tp,
        is_cptp=is_cp and is_tp,
    )


__all__ = ["TOL_CHOI", "TOL_TP", "ChoiGateResult", "choi_matrix", "cptp_choi_gate"]
