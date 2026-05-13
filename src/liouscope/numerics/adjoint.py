"""Heisenberg-picture adjoint and pi-weighted Alicki adjoint of the Liouvillian.

Two distinct objects exist; conflating them is the H6/H7 hallucination from
the v2.0 audit:

* :func:`hs_adjoint` -- the Hilbert-Schmidt adjoint ``L*`` in the
  Heisenberg picture::

      L*(A) = +i [H, A] + sum_k ( L_k^dag A L_k - 1/2 {L_k^dag L_k, A} )

  As a superoperator in column-stacking convention,
  ``M_{L*} = M_L^H`` --- the conjugate transpose of the superoperator
  matrix of ``L``.

* :func:`alicki_adjoint` -- the pi-weighted Heisenberg-picture adjoint of
  the Schrödinger Liouvillian used to build the symmetrised generator for
  the GNS / KMS gaps. The general construction (column-stacking) is::

      L_pi*  = G^{-1} L G

  where the Gram matrix selects the metric:

  ============  ===========================================
  metric        Gram matrix G (column-stacking, real rho)
  ============  ===========================================
  ``"gns"``     ``G_GNS = rho_ss (x) I``
  ``"kms"``     ``G_KMS = rho_ss^{1/2} (x) rho_ss^{1/2}``
  ============  ===========================================

  **Direction matters.** Swapping the inverse to the right yields
  ``Delta_s / Delta > 1`` on boundary-driven systems, which is
  physically impossible (anchor C). Without this fix, Paper 1 would have
  shipped a central false claim.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.linalg as sla

GramKind = Literal["gns", "kms"]


def hs_adjoint(L_super: np.ndarray) -> np.ndarray:
    """Hilbert-Schmidt adjoint of a column-stacked superoperator."""
    L_super = np.asarray(L_super)
    if L_super.ndim != 2 or L_super.shape[0] != L_super.shape[1]:
        raise ValueError(f"hs_adjoint expects a square matrix, got {L_super.shape}")
    return L_super.conj().T


def gram_matrix(rho_steady: np.ndarray, metric: GramKind = "gns") -> np.ndarray:
    """Return the column-stacking Gram matrix for the given inner product.

    For the GNS metric ``<A,B>_GNS = Tr(A^dag B rho_ss)``::

        G_GNS = rho_ss (x) I       (column stacking)

    For the KMS metric ``<A,B>_KMS = Tr(rho_ss^{1/2} A^dag rho_ss^{1/2} B)``::

        G_KMS = rho_ss^{1/2} (x) rho_ss^{1/2}.conj()

    (The right factor is the conjugate of the square root because the inner
    product is Hermitian and the second argument carries the conjugation in
    the column-stacking convention used throughout the package.)
    """
    rho = np.asarray(rho_steady)
    d = rho.shape[0]
    eye_d = np.eye(d, dtype=complex)
    if metric == "gns":
        return np.kron(rho, eye_d)
    if metric == "kms":
        sqrt_rho = sla.sqrtm(rho).astype(complex)
        return np.kron(sqrt_rho, sqrt_rho.conj())
    raise ValueError(f"unknown metric {metric!r}; expected 'gns' or 'kms'")


def alicki_adjoint(
    L_super: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps_reg: float = 1.0e-12,
    metric: GramKind = "gns",
) -> np.ndarray:
    """pi-weighted Heisenberg-picture adjoint with the chosen metric.

    Returns ``G^{-1} L G`` where ``G`` is selected by :func:`gram_matrix`.

    For the GNS metric this collapses to the column-stacking form
    ``(rho_ss^{-1} (x) I) L (rho_ss (x) I)`` (anchor C).

    For the KMS metric the construction is
    ``(rho_ss^{-1/2} (x) rho_ss^{-1/2}.conj()) L (rho_ss^{1/2} (x) rho_ss^{1/2}.conj())``.
    """
    L_super = np.asarray(L_super)
    rho = np.asarray(rho_steady)
    d = rho.shape[0]
    if L_super.shape != (d * d, d * d):
        raise ValueError(
            f"L_super has shape {L_super.shape} but rho is {d} x {d}; "
            f"expected L_super to be ({d * d}, {d * d})"
        )

    rho_reg = rho + eps_reg * np.eye(d, dtype=rho.dtype)
    G = gram_matrix(rho_reg, metric)
    G_inv = sla.inv(G)
    return G_inv @ L_super @ G


def symmetrised_liouvillian(
    L_super: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps_reg: float = 1.0e-12,
    metric: GramKind = "gns",
) -> np.ndarray:
    """Return the Mori-Shirai-style symmetrised generator.

    Construction (Heisenberg picture)::

        L_HS         = L^H
        L_HS_pi_adj  = G^{-1} L G
        L_HS_sym     = ( L_HS + L_HS_pi_adj ) / 2

    Conjugating ``L_HS_sym`` by ``G^{1/2}`` yields a Hermitian matrix whose
    top eigenvalue is ~0 (corresponding to ``vec(I)`` -- trace preservation)
    and whose second-largest eigenvalue gives the symmetrised gap.

    The ``metric`` keyword selects the Gram matrix: ``"gns"`` reproduces the
    Mori-Shirai 2023 GNS gap (D2); ``"kms"`` reproduces the Fagnola-Umanita
    KMS gap (D2b).
    """
    L_super = np.asarray(L_super)
    L_HS = L_super.conj().T
    L_HS_pi_adj = alicki_adjoint(L_super, rho_steady, eps_reg=eps_reg, metric=metric)
    return 0.5 * (L_HS + L_HS_pi_adj)
