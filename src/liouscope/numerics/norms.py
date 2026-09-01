"""Overflow/underflow-safe Euclidean norm primitives.

The implementation follows the same numerical contract as LAPACK's xLASSQ
family: scale values before squaring, accumulate in the scaled domain, then
restore the scale.  LiouScope uses a power-of-two scale so the rescaling itself
does not introduce a reciprocal overflow for subnormal complex values and does
not round the mantissas merely to choose numerical units.
"""

from __future__ import annotations

import numpy as np


def scaled_euclidean_norm(values: np.ndarray) -> float:
    """Return ``sqrt(sum(abs(values)**2))`` without spurious under/overflow.

    This is the vector 2-norm and, when ``values`` is a matrix, its Frobenius
    norm.  For finite input it returns a finite number whenever the true norm is
    representable as ``float64``.  It returns ``inf`` only when an infinite
    input component or the mathematical norm itself is not representable, and
    propagates ``NaN`` rather than fabricating evidence from corrupted input.

    Complex values are treated as their real and imaginary components, so the
    calculation is mathematically ``sqrt(sum(re**2 + im**2))``.  We avoid
    computing ``abs(z)`` before selecting the scale because a finite complex
    value near the top of the floating-point range can overflow in the modulus.

    The power-of-two scaling is an xLASSQ-style specialization: if ``m`` is the
    largest finite component magnitude and ``m = f * 2**e`` with
    ``0.5 <= f < 1``, shifting every component by ``2**(-e)`` places the
    largest component below one.  Squaring cannot overflow, at least one square
    is order one, and restoring with ``ldexp`` overflows iff the true norm does.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0

    real = np.asarray(np.real(arr), dtype=float)
    imag = np.asarray(np.imag(arr), dtype=float)
    if np.any(np.isnan(real)) or np.any(np.isnan(imag)):
        return float("nan")
    if np.any(np.isinf(real)) or np.any(np.isinf(imag)):
        return float("inf")

    component_max = float(
        max(
            float(np.max(np.abs(real))),
            float(np.max(np.abs(imag))),
        )
    )
    if component_max == 0.0:
        return 0.0

    exponent = int(np.frexp(component_max)[1])
    scaled_real = np.ldexp(real, -exponent)
    scaled_imag = np.ldexp(imag, -exponent)
    sumsq = float(
        np.sum(scaled_real * scaled_real, dtype=float)
        + np.sum(scaled_imag * scaled_imag, dtype=float)
    )
    scaled_norm = float(np.sqrt(sumsq))
    with np.errstate(over="ignore", under="ignore"):
        return float(np.ldexp(scaled_norm, exponent))
