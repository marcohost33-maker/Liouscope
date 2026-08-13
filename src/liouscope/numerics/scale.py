"""Shared operator rate scale for scale-relative diagnostics (issue #101 slice A).

Several diagnostics (D8 Henrici, D10 Kreiss grid, D11b/D12 resolvent peak/FWHM,
D13 pseudospectral radius) historically mixed *rate-dimensioned* quantities with
*absolute* grid constants and thresholds (``sigma = 1e-3``, ``eps = 1e-3``,
``omega_max >= 1`` ...). A Liouvillian carries rate dimension, so a pure change
of units ``L -> c L`` (``c > 0``) moved those absolute constants relative to the
spectrum and changed the diagnostic values beyond the physical ``~c`` scaling.

This module defines the ONE positive operator rate scale that every
scale-relative diagnostic shares:

    ``rate_scale(L) = ||L||_F``  (Frobenius norm)

Properties:

* homogeneous of degree one -- ``rate_scale(c L) = |c| rate_scale(L)`` exactly,
  so dimensionless products/quotients built with it are exactly invariant under
  a positive unit rescale;
* basis-robust -- invariant under unitary similarity ``L -> W L W^H``;
* explicit zero-operator semantics -- ``rate_scale(0) == 0.0``; callers must
  branch on that value and document what "no scale" means for their diagnostic
  (see e.g. :func:`liouscope.diagnostics.nonnormality.henrici_relative`);
* fail-closed -- non-finite or non-square input raises a located ``ValueError``
  instead of laundering NaN/inf into a downstream grid.

The Frobenius norm was preferred (issue #97 2026-08-05 re-audit) over the
spectral gap (diverges/degenerates in the gapless limit and couples two
independently used signals) and over ``spectral_spread`` (can vanish for
spectrally collapsed but non-normal operators). NOTE (issue #101 follow-up):
``||L||_F`` grows with Hilbert dimension for extensive lattice generators, so
cross-SIZE comparisons of Frobenius-relative diagnostics carry a dimension
bias; the preregistered calibration study (#101 slice C) must include
system-size families before any classifier threshold consumes these values.
"""

from __future__ import annotations

import numpy as np

from .linalg import require_finite_square_2d


def rate_scale(L: np.ndarray, *, name: str = "L") -> float:
    """Return the shared operator rate scale ``||L||_F``.

    Parameters
    ----------
    L
        Square matrix (typically a ``d^2 x d^2`` Liouvillian superoperator).
    name
        Argument name used in the fail-closed error message.

    Returns
    -------
    float
        ``||L||_F >= 0``. Exactly ``0.0`` for the zero operator (documented
        zero-operator semantics: there is no rate scale to normalise by).

    Raises
    ------
    ValueError
        If ``L`` is not a finite square 2-D matrix (fail-closed).
    """
    L = require_finite_square_2d(L, name=name)
    return float(np.linalg.norm(L, ord="fro"))


__all__ = ["rate_scale"]
