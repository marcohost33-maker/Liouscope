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
  symmetrised Liouvillian for the GNS gap (Mori-Shirai 2023 Eq. 8). It is
  the adjoint of the Heisenberg generator w.r.t. the GNS inner product
  ``<A, B> = Tr(rho_ss A^dag B)``, whose Gram matrix in column-stacking
  convention is ``G = rho_ss.T (x) I`` (anchor B). The adjoint is therefore::

      M_{L_tilde*} = G^{-1} M_L G
                   = ((rho_ss^{-1}).T (x) I) M_L (rho_ss.T (x) I)

  which in operator form reads ``L_tilde*(A) = L(A rho_ss) rho_ss^{-1}``.

  **Direction matters** (anchor C, the critical T17 hardening bug):
  swapping the inverse to the right yields ``Delta_s / Delta > 1`` on
  boundary-driven systems, which is physically impossible. Without this
  fix, Paper 1 would have shipped a central false claim.

  **Transposition matters** (2026-07 audit A1): the Kron factors must be
  the *transposed* ``rho_ss`` — i.e. exactly the Gram ``G`` from anchor B —
  or the "adjoint" is only an adjoint for real (in practice diagonal)
  steady states, and the "symmetrised" generator silently stops being
  G-Hermitian: ``gns_gap`` then returns unphysical ``Delta_s < 0`` for any
  system whose steady state carries coherences. For real-diagonal
  ``rho_ss`` (every anchor fixture) transposition is a no-op, which is why
  the bug survived the anchor suite.
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

    Anchors B + C. This is the GNS-Gram adjoint ``G^{-1} L G`` with
    ``G = rho_ss.T (x) I``, i.e. in column-stacking convention::

        L_tilde* = ((rho_ss^{-1}).T (x) I) L (rho_ss.T (x) I)

    Reversing the direction (inverse on the right) yields unphysical
    ``Delta_s > Delta`` on Prosen-Znidaric boundary-driven cases (T17
    hardening regression). Dropping the transpositions yields unphysical
    ``Delta_s < 0`` whenever ``rho_ss`` carries coherences (2026-07 audit
    A1); both are no-ops for the real-diagonal anchor fixtures.
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
    left = np.kron(rho_inv.T, eye_d)
    right = np.kron(rho_reg.T, eye_d)
    pi_adj: np.ndarray = left @ L_super @ right
    return pi_adj


def gram_adjoint(L_super: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Adjoint of the Heisenberg-picture generator w.r.t. an arbitrary Gram ``G``.

    For the weighted inner product ``<x, y> = x^dag G y`` the adjoint of the
    Heisenberg generator ``L_HS = L^H`` is ``G^{-1} L_HS^dag G = G^{-1} L G``.
    With the GNS Gram ``G = rho_ss.T (x) I`` this reproduces
    :func:`alicki_adjoint`; with the KMS Gram
    ``G = rho^{1/2}.conj() (x) rho^{1/2}`` it yields the Fagnola s=1/2
    adjoint used for D2b.
    """
    L_super = np.asarray(L_super)
    G = np.asarray(G)
    adj: np.ndarray = np.linalg.solve(G, L_super @ G)
    return adj


def symmetrised_liouvillian(
    L_super: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps_reg: float = 1.0e-12,
) -> np.ndarray:
    """Return the Mori-Shirai symmetrised generator in the Heisenberg picture.

    Construction (``G = rho_ss.T (x) I``, the GNS Gram of anchor B)::

        L_HS         = L^H               (Heisenberg-picture Liouvillian)
        L_HS_pi_adj  = G^{-1} L G        (= alicki_adjoint, all rho_ss)
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
