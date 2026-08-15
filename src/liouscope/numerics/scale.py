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

from .._consts import ZERO_MODE_EPS_FACTOR
from .linalg import require_finite_square_2d


def spectral_zero_tolerance(
    eigenvalues: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    atol: float | None = None,
    name: str = "eigenvalues",
) -> float:
    """Backward-error tolerance separating numerical zero modes from real ones.

    Issue #108. The zero-mode filter ``|lambda| > tol`` decides which modes are
    the steady state and which carry physics. With an ABSOLUTE ``tol`` it is not
    invariant under a change of rate units ``L -> cL``, and it fails in BOTH
    directions:

    * small ``c`` -- every genuine mode drops below a fixed floor, so the gap
      collapses to ``0.0`` and the Mpemba slow-mode overlap collapses to
      ``0.0``, firing a FALSE A11/F4 candidate (the highest-priority rung);
    * large ``c`` -- the numerical zero mode (``~eps * ||L||``) grows ABOVE a
      fixed floor and is counted as a genuine mode, which yields a NEGATIVE
      spectral gap -- impossible for a GKSL generator by definition.

    The functions that need this tolerance receive only the spectrum, not the
    operator, so the scale is taken from the spectrum itself: the **spectral
    radius** ``max|lambda|``, which like :func:`rate_scale` is homogeneous of
    degree one and invariant under unitary similarity.

    The threshold is a multiple of the eigensolver BACKWARD ERROR,
    ``tol = rtol * eps * max|lambda|``, rather than a fixed fraction of the
    radius. A fixed fraction would impose a dynamic-range ceiling and discard
    the genuine slow modes of a METASTABLE generator: with ``1e-10 *
    max|lambda|``, two damping channels at rates ``1.0`` and ``1e-12`` (true
    gap ``5e-13``) report a gap of ``5e-1`` -- wrong by ten orders of
    magnitude, and precisely on the A5 systems the library exists to study.
    A computed eigenvalue is uncertain to order ``eps * ||L||``, so that is the
    scale on which "indistinguishable from zero" is properly decided; see
    :data:`liouscope._consts.ZERO_MODE_EPS_FACTOR` for the measured
    calibration behind the default factor.

    Parameters
    ----------
    eigenvalues
        Spectrum to derive the scale from.
    rtol
        Multiplier on the backward error ``eps * max|lambda|``.
    atol
        Legacy ABSOLUTE floor, returned verbatim in place of the computed
        tolerance -- the reproducibility opt-in for pre-#108 behaviour,
        mirroring how #99 preserved the old absolute steady-state tolerance.
        The spectrum is still validated (see ``Raises``): a compatibility
        switch may reproduce the old THRESHOLD, but it must not reintroduce
        the old silent acceptance of corrupted solver output.
    name
        Argument name used in the fail-closed error message.

    Returns
    -------
    float
        The tolerance. Exactly ``0.0`` for an empty spectrum and for the zero
        operator (all eigenvalues zero): there is no scale to normalise by, and
        with ``tol = 0`` the strict ``|lambda| > tol`` test correctly classifies
        every exactly-zero mode as a zero mode.

    Raises
    ------
    ValueError
        If ``rtol``/``atol`` is negative or non-finite, or if ``eigenvalues``
        contains NaN/inf. Fail-closed: a non-finite spectrum would make
        ``max|lambda|`` non-finite, and every ``|lambda| > NaN`` comparison
        would evaluate ``False`` -- silently reporting "no non-zero modes"
        (gap ``0.0``) for corrupted eigensolver output.
    """
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if atol is not None and (not np.isfinite(atol) or atol < 0.0):
        raise ValueError(f"atol must be finite and non-negative, got {atol}")
    ev = np.asarray(eigenvalues)
    # Validate BEFORE honouring the legacy override: the fail-closed contract
    # on corrupted solver output is not part of what the compatibility switch
    # is allowed to turn off.
    if ev.size and not np.all(np.isfinite(ev)):
        bad = np.flatnonzero(~np.isfinite(ev.ravel())).tolist()
        raise ValueError(
            f"{name} must be finite to derive a zero-mode tolerance; "
            f"non-finite entries at flat indices {bad}"
        )
    if atol is not None:
        return float(atol)
    if ev.size == 0:
        return 0.0
    # eps is the DOUBLE-precision machine epsilon, deliberately independent of
    # the array's storage dtype. The backward error is set by the precision the
    # eigensolver COMPUTED in, and NumPy/SciPy solve in double regardless of the
    # input dtype -- measured: a complex64 generator yields a numerical zero mode
    # at ~1.5e-19, i.e. double-level, even though the returned array is
    # complex64. Deriving eps from the storage dtype instead reads that as
    # single-precision round-off and inflates the threshold to ~1.2e-4 relative,
    # which discards clearly resolved slow modes: a complex64 two-channel
    # generator with rates 1.0 and 1e-4 then reports a gap of 5e-1 instead of
    # 5e-5, corrupting exactly the metastable case this tolerance was widened to
    # protect.
    #
    # A genuinely single-precision spectrum (an external or GPU solver) is the
    # one case where a coarser threshold is right, and it is indistinguishable
    # from a downcast double result by inspection. Such callers pass ``rtol``
    # explicitly -- an argument the caller has, and this function does not.
    eps = float(np.finfo(float).eps)
    return float(rtol * eps * float(np.max(np.abs(ev))))


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


__all__ = ["rate_scale", "spectral_zero_tolerance"]
