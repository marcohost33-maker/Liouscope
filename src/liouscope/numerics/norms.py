"""Overflow/underflow-safe norm and cancellation primitives.

The Euclidean norm implementation follows the same numerical contract as
LAPACK's xLASSQ family: scale values before squaring, accumulate in the scaled
domain, then restore the scale. LiouScope uses a power-of-two scale so the
rescaling itself does not introduce a reciprocal overflow for subnormal complex
values and does not round mantissas merely to choose numerical units.
"""

from __future__ import annotations

import math

import numpy as np


def _finite_component_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return real/imag parts shifted so their largest component is O(1).

    ``None`` denotes an exact all-zero array. NaN/inf are deliberately left to
    public callers because their desired sentinel semantics differ by operation.
    """
    arr = np.asarray(values)
    real = np.asarray(np.real(arr), dtype=float)
    imag = np.asarray(np.imag(arr), dtype=float)
    component_max = float(
        max(
            float(np.max(np.abs(real))) if real.size else 0.0,
            float(np.max(np.abs(imag))) if imag.size else 0.0,
        )
    )
    if component_max == 0.0:
        return None
    exponent = int(np.frexp(component_max)[1])
    return np.ldexp(real, -exponent), np.ldexp(imag, -exponent), exponent


def scaled_euclidean_norm(values: np.ndarray) -> float:
    """Return ``sqrt(sum(abs(values)**2))`` without spurious under/overflow.

    This is the vector 2-norm and, when ``values`` is a matrix, its Frobenius
    norm. For finite input it returns a finite number whenever the true norm is
    representable as ``float64``. It returns ``inf`` only when an infinite
    input component or the mathematical norm itself is not representable, and
    propagates ``NaN`` rather than fabricating evidence from corrupted input.

    Complex values are treated as their real and imaginary components, so the
    calculation is mathematically ``sqrt(sum(re**2 + im**2))``. We avoid
    computing ``abs(z)`` before selecting the scale because a finite complex
    value near the top of the floating-point range can overflow in the modulus.
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

    scaled = _finite_component_scale(arr)
    if scaled is None:
        return 0.0
    scaled_real, scaled_imag, exponent = scaled
    sumsq = float(
        np.sum(scaled_real * scaled_real, dtype=float)
        + np.sum(scaled_imag * scaled_imag, dtype=float)
    )
    scaled_norm = float(np.sqrt(sumsq))
    with np.errstate(over="ignore", under="ignore"):
        return float(np.ldexp(scaled_norm, exponent))


def scaled_log_sum_squares(values: np.ndarray) -> float:
    """Return ``log(sum(abs(values)**2))`` without spurious under/overflow.

    The return value is ``-inf`` for an exact all-zero input, ``nan`` when any
    component is NaN, and ``inf`` when any component is infinite. For finite,
    non-zero float64 input the logarithm remains finite even when the true sum
    of squares (or its square root) is outside the representable float64 range.

    This is the likelihood-facing companion to :func:`scaled_euclidean_norm`:
    both use the same exact power-of-two scaling contract, but this function
    never reconstructs RSS in ordinary floating-point units.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return float("-inf")

    real = np.asarray(np.real(arr), dtype=float)
    imag = np.asarray(np.imag(arr), dtype=float)
    if np.any(np.isnan(real)) or np.any(np.isnan(imag)):
        return float("nan")
    if np.any(np.isinf(real)) or np.any(np.isinf(imag)):
        return float("inf")

    scaled = _finite_component_scale(arr)
    if scaled is None:
        return float("-inf")
    scaled_real, scaled_imag, exponent = scaled
    sumsq = float(
        np.sum(scaled_real * scaled_real, dtype=float)
        + np.sum(scaled_imag * scaled_imag, dtype=float)
    )
    return float(math.log(sumsq) + 2.0 * exponent * math.log(2.0))


def scaled_column_sums(values: np.ndarray) -> np.ndarray:
    """Return ``values.sum(axis=0)`` without spurious intermediate overflow.

    Issue #139, P2 review. An ordinary matrix product accumulates in ordinary
    units, so a column whose terms cancel exactly can still overflow on the way
    to its own zero: eight entries of ``+2.5e307`` followed by eight of
    ``-2.5e307`` sum to exactly zero, but ``vec(I)^H @ L`` reaches ``2e308``
    after the eighth addition and returns ``inf+nanj``. The caller then sees
    corrupt evidence about a generator that is perfectly representable, and a
    gate reading that evidence refuses a legal operator.

    Scaling is a precondition of the accumulation, not a refinement of it:
    ``math.fsum`` on the unscaled column above does not return a sentinel, it
    raises ``OverflowError``. Same contract as the rest of this module: values
    are shifted by a POWER OF TWO before accumulation and shifted back afterwards, so no mantissa bit is
    lost to the choice of units. After scaling, every component is below one in
    magnitude, hence a sum of ``k`` terms cannot exceed ``k`` and overflow is
    impossible by construction rather than by tolerance.

    The scale is chosen PER COLUMN AND PER COMPONENT, not once for the whole
    array. A single global scale would be simpler and would still stop the
    overflow, but a matrix holding a column near 1e300 beside one near 1e-300
    would then have the small column shifted into the subnormal range and
    flushed to zero -- turning a real defect into an apparent exact zero. That
    is the same class of failure in the opposite direction, and it is the
    silent one.

    Real and imaginary parts need separate scales for the same reason, one
    level down. :func:`_finite_component_scale` deliberately shares one scale
    between them, which is right for a NORM -- there no term can cancel, so a
    1e-300 imaginary part beside a 1e300 real part contributes nothing anyway.
    A SUM is not monotone: the large part can vanish by cancellation and leave
    the small one as the entire answer. Measured on the column
    ``[1e300+1e-300j, -1e300, 0]``, whose sum is ``1e-300j``, a shared scale
    returned exactly zero.

    ``math.fsum`` accumulates the real and imaginary parts, matching
    :func:`scaled_cancellation_ratio`, which measures the same trace equations.
    This is not a matter of the last few bits: a vectorised variant of this
    function, benchmarked at 2.4x to 11.9x faster, returned 0 for the column
    ``[1, 1e16, -1e16]`` whose exact sum is 1. Ordinary summation does not
    merely round such a defect, it deletes it -- and a deleted trace defect
    reads as an exactly trace-preserving generator, which is the failure
    direction that must never be traded for speed. Measured cost of keeping it:
    0.07 ms at d=4, where the accompanying eigensolve takes 0.26 ms, falling to
    0.30 percent of the eigensolve at d=16.

    Columns holding a NaN or an infinity are summed in ordinary arithmetic, so
    the sentinel the caller expects still arrives: input that genuinely is not
    representable must not be made finite here.
    """
    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(f"values must be a 2-D array, got {arr.ndim}-D")
    cols = int(arr.shape[1])
    out: np.ndarray = np.zeros(cols, dtype=complex)
    if arr.shape[0] == 0 or cols == 0:
        return out

    real = np.asarray(np.real(arr), dtype=float)
    imag = np.asarray(np.imag(arr), dtype=float)
    nonfinite = np.any(~np.isfinite(real), axis=0) | np.any(~np.isfinite(imag), axis=0)

    accumulated = []
    for part in (real, imag):
        component_max = np.max(np.abs(np.where(np.isfinite(part), part, 0.0)), axis=0)
        exponent = np.zeros(cols, dtype=np.int32)
        scalable = (component_max > 0.0) & ~nonfinite
        totals = np.zeros(cols, dtype=float)
        if np.any(scalable):
            exponent[scalable] = np.frexp(component_max[scalable])[1].astype(np.int32)
            # Transposed once so the per-column accumulation reads Python lists
            # instead of paying a NumPy indexing round trip per column, and the
            # rescaling is applied to the assembled vector in one call rather
            # than one scalar call per column. Same arithmetic, less overhead.
            scaled = np.ldexp(part, -exponent[None, :]).T.tolist()
            index = np.flatnonzero(scalable)
            sums = np.fromiter(
                (math.fsum(scaled[j]) for j in index), dtype=float, count=index.size
            )
            with np.errstate(over="ignore", under="ignore"):
                totals[index] = np.ldexp(sums, exponent[index])
        accumulated.append(totals)
    out = accumulated[0] + 1j * accumulated[1]
    if np.any(nonfinite):
        out[nonfinite] = np.sum(arr[:, nonfinite], axis=0)
    return out


def scaled_cancellation_ratio(values: np.ndarray) -> float:
    """Return ``abs(sum(values)) / sum(abs(values))`` scale-safely.

    This is a componentwise relative backward-error primitive for a scalar
    linear equation assembled by cancellation. It is in ``[0, 1]`` for finite
    input, is exactly scale invariant under representable non-zero rescaling,
    and returns zero for an all-zero equation.

    Power-of-two scaling prevents both overflow and subnormal underflow before
    summation. ``math.fsum`` is used separately on real and imaginary parts so
    the numerator measures the represented equation rather than ordinary
    left-to-right summation noise. The denominator is accumulated from stable
    ``hypot`` magnitudes in the same scaled domain, so unrelated large entries
    outside the equation cannot dilute its defect.
    """
    arr = np.asarray(values)
    if arr.size == 0:
        return 0.0
    real = np.asarray(np.real(arr), dtype=float)
    imag = np.asarray(np.imag(arr), dtype=float)
    if np.any(~np.isfinite(real)) or np.any(~np.isfinite(imag)):
        return float("nan")

    scaled = _finite_component_scale(arr)
    if scaled is None:
        return 0.0
    scaled_real, scaled_imag, _ = scaled
    sum_real = math.fsum(float(x) for x in scaled_real.ravel())
    sum_imag = math.fsum(float(x) for x in scaled_imag.ravel())
    numerator = math.hypot(sum_real, sum_imag)
    denominator = math.fsum(
        math.hypot(float(re), float(im))
        for re, im in zip(scaled_real.ravel(), scaled_imag.ravel(), strict=True)
    )
    if denominator == 0.0:
        return 0.0
    return float(min(1.0, numerator / denominator))
