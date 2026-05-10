"""Column-stacking vectorisation primitives.

LiouScope enforces column-stacking via ``order='F'`` everywhere. This is
the canonical convention used by QuTiP, dynamiqs and Forest-Benchmarking and is
required by the Roth-lemma identity

    vec(A B C) = (C.T (x) A) vec(B)

Mixing this with row-stacking silently garbles the physics while leaving
dimensions consistent (anchor A in the v2.0 audit).
"""

from __future__ import annotations

import numpy as np


def vec(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorisation.

    Parameters
    ----------
    rho
        Square array of shape ``(d, d)``.

    Returns
    -------
    np.ndarray
        Vector of length ``d * d``.
    """
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"vec() expects a square matrix, got shape {rho.shape}")
    return rho.flatten(order="F")


def unvec(rho_vec: np.ndarray, d: int | None = None) -> np.ndarray:
    """Inverse of :func:`vec`. Returns the ``(d, d)`` matrix.

    Parameters
    ----------
    rho_vec
        Vector of length ``d * d``.
    d
        Side length. Inferred via ``sqrt(len(rho_vec))`` if omitted.

    Returns
    -------
    np.ndarray
        Reconstructed matrix.
    """
    rho_vec = np.asarray(rho_vec)
    n = rho_vec.size
    if d is None:
        d = int(round(np.sqrt(n)))
    if d * d != n:
        raise ValueError(f"Cannot reshape length {n} as a square matrix")
    return rho_vec.reshape((d, d), order="F")
