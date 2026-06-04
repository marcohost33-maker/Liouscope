"""Heisenberg-picture adjoint and pi-weighted Alicki adjoint of the Liouvillian.

Two distinct objects exist; conflating them is the H6/H7 hallucination from
the v2.0 audit:

* :func:`hs_adjoint` -- the Hilbert-Schmidt adjoint ``L*`` in the
  Heisenberg picture::

      L*(A) = +i [H, A] + sum_k ( L_k^dag A L_k - 1/2 {L_k^dag L_k, A} )

  As a superoperator in column-stacking convention,
  ``M_{L*} = M_L^H`` --- the conjugate transpose of the superoperator
  matrix of ``L``.

* :func:`alicki_adjoint` -- the pi-weighted adjoint used to build the
  symmetrised Liouvillian for the GNS gap (Mori-Shirai 2023 Eq. 8)::

      L_tilde*(A) = (rho_ss)^{-1} L(rho_ss A)        (operator form)

  In column-stacking convention this becomes (anchor C, the critical T17
  hardening bug)::

      M_{L_tilde*} = (rho_ss^{-1} (x) I) M_L (rho_ss (x) I)

  **Direction matters.** Swapping the inverse to the right yields
  ``Delta_s / Delta > 1`` on boundary-driven systems, which is
  physically impossible. Without this fix, Paper 1 would have shipped a
  central false claim.
"""

from __future__ import annotations

import numpy as np


def hs_adjoint(L_super: np.ndarray) -> np.ndarray:
    """Hilbert-Schmidt adjoint of a column-stacked superoperator.

    Parameters
    ----------
    L_super
        ``d^2 x d^2`` matrix representing ``L`` in column-stacking convention.

    Returns
    -------
    np.ndarray
        ``d^2 x d^2`` matrix representing ``L*`` in column-stacking convention.
    """
    L_super = np.asarray(L_super)
    if L_super.ndim != 2 or L_super.shape[0] != L_super.shape[1]:
        raise ValueError(f"hs_adjoint expects a square matrix, got {L_super.shape}")
    adj: np.ndarray = L_super.conj().T
    return adj


def alicki_adjoint(
    L_super: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps_reg: float = 1.0e-12,
) -> np.ndarray:
    """pi-weighted adjoint of the Liouvillian (Mori-Shirai 2023 Eq. 8).

    Anchor C. The correct direction in column-stacking convention is::

        L_tilde* = (rho_ss^{-1} (x) I) L (rho_ss (x) I)

    Reversing this yields unphysical ``Delta_s > Delta`` on
    Prosen-Znidaric boundary-driven cases (T17 hardening regression).
    """
    L_super = np.asarray(L_super)
    rho = np.asarray(rho_steady)
    d = rho.shape[0]
    if L_super.shape != (d * d, d * d):
        raise ValueError(
            f"L_super has shape {L_super.shape} but rho is {d} x {d}; "
            f"expected L_super to be ({d * d}, {d * d})"
        )

    # Regularise to avoid singularity in rho_ss^{-1}.
    rho_reg = rho + eps_reg * np.eye(d, dtype=rho.dtype)
    rho_inv = np.linalg.inv(rho_reg)

    eye_d = np.eye(d, dtype=L_super.dtype)
    left = np.kron(rho_inv, eye_d)
    right = np.kron(rho_reg, eye_d)
    pi_adj: np.ndarray = left @ L_super @ right
    return pi_adj


def symmetrised_liouvillian(
    L_super: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps_reg: float = 1.0e-12,
) -> np.ndarray:
    """Return the Mori-Shirai symmetrised generator in the Heisenberg picture.

    Construction::

        L_HS         = L^H               (Heisenberg-picture Liouvillian)
        L_HS_pi_adj  = G^{-1} L G         (= alicki_adjoint, for real diag rho)
        L_HS_sym     = ( L_HS + L_HS_pi_adj ) / 2

    Conjugating ``L_HS_sym`` by ``G^{1/2}`` yields a Hermitian matrix whose
    top eigenvalue is ~0 (corresponding to ``vec(I)`` -- trace preservation)
    and whose second-largest eigenvalue gives the GNS gap ``Delta_s`` (D2).
    """
    L_super = np.asarray(L_super)
    L_HS = L_super.conj().T
    L_HS_pi_adj = alicki_adjoint(L_super, rho_steady, eps_reg=eps_reg)
    sym: np.ndarray = 0.5 * (L_HS + L_HS_pi_adj)
    return sym
