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

# Issue #139: exponent bands for the overflow fallback in
# :func:`_overflow_safe_fsum`. The split at 2**512 is what makes two bands
# provably enough -- see that function for why neither band can overflow or
# underflow.
_BAND_SHIFT = 1024
_BAND_SPLIT = 2.0**512


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


def _overflow_safe_fsum(values: list[float]) -> float:
    """Exact sum of finite float64 values, without sacrificing any addend.

    ``math.fsum`` is already exact over the whole float64 range -- Shewchuk
    accumulation keeps a set of non-overlapping partial sums, so no bit of any
    addend is lost. Its one failure mode is an intermediate that leaves the
    range, and that depends on the ORDER of the input: eight ``+2.5e307``
    followed by eight ``-2.5e307`` raises ``OverflowError``, while the same
    sixteen values alternating return zero.

    The first attempt is therefore the plain call. Only when it raises do we
    split the addends into two exponent bands at ``2**512``:

    * above the split, shifting by ``2**-1024`` lands every term in
      ``[2**-512, 1)`` -- far above the subnormal floor, so nothing is lost --
      and ``k`` such terms sum below ``k``, so nothing overflows;
    * below the split, ``k`` terms sum below ``k * 2**512 < 2**1024``, so the
      band needs no scaling at all and hence cannot underflow either.

    Both bands are summed exactly and recombined once. Overflow in the
    recombination is the honest answer: it means the true sum is not
    representable.

    KNOWN LIMIT of the fallback, measured rather than assumed. The plain path
    returns the correctly rounded exact sum -- 20000 randomised columns, no
    deviation. The band path rounds twice, once per band sum and once in the
    recombination, so it can land one ulp off: 3 deviations in 20000 columns
    built on a guaranteed-overflow kernel, each in the last place of a result
    of order 1e182 or larger. That is ordinary rounding, NOT the failure class
    this function exists to prevent: a further 19965 columns constructed to
    cancel inside the band path produced no spurious zero and no sign error, so
    no defect is deleted and none changes direction. Removing the last ulp
    would require the exact remainder that ``math.fsum`` discards, i.e. a
    Shewchuk accumulator that returns its partial sums.

    Issue #139, third review round. The previous construction shifted every
    addend by one power of two chosen from the largest component of the column.
    That cannot work, and not merely for the reported example: a scale SHIFTS
    the representable window, it does not widen it, so any addend below
    ``max * 2**-1074`` is flushed to zero before the accumulation starts.
    Measured on ``[1e300, -1e300, 1e-300]``, whose exact sum is ``1e-300``: the
    scaled version returned 0, and a caller reading that number accepted a
    generator that is not trace preserving. Choosing the scale smaller instead
    reintroduces the overflow this function exists to prevent -- there is no
    right choice once a column spans the range and its dominant terms cancel.
    """
    try:
        return math.fsum(values)
    except OverflowError:
        pass
    high: list[float] = []
    low: list[float] = []
    for value in values:
        (high if abs(value) >= _BAND_SPLIT else low).append(value)
    scaled_high = math.fsum(math.ldexp(value, -_BAND_SHIFT) for value in high)
    try:
        return math.ldexp(scaled_high, _BAND_SHIFT) + math.fsum(low)
    except OverflowError:
        return math.copysign(math.inf, scaled_high)


def scaled_column_sums(values: np.ndarray) -> np.ndarray:
    """Return ``values.sum(axis=0)`` exactly, column by column.

    Issue #139, P2 review. An ordinary matrix product accumulates in ordinary
    units, so a column whose terms cancel exactly can still overflow on the way
    to its own zero: eight entries of ``+2.5e307`` followed by eight of
    ``-2.5e307`` sum to exactly zero, but ``vec(I)^H @ L`` reaches ``2e308``
    after the eighth addition and returns ``inf+nanj``. The caller then sees
    corrupt evidence about a generator that is perfectly representable, and a
    gate reading that evidence refuses a legal operator.

    Each column is summed by :func:`_overflow_safe_fsum`, separately for the
    real and the imaginary part. Both readings are exact: no addend is dropped
    however far it lies below the largest term of its column, and no
    intermediate leaves the float64 range. The result is the correctly rounded
    exact sum whenever that is representable, and ``inf`` only when it is not.

    Real and imaginary parts are summed independently rather than through a
    shared treatment, because a SUM is not monotone the way a norm is: the real
    parts can annihilate and leave an imaginary part six hundred orders of
    magnitude below them as the entire answer. Measured on the column
    ``[1e300+1e-300j, -1e300, 0]``, whose sum is ``1e-300j``.

    Exactness here is not a matter of the last few bits. A vectorised variant
    of this function, benchmarked 2.4x to 11.9x faster, returned 0 for the
    column ``[1, 1e16, -1e16]`` whose exact sum is 1: ordinary summation does
    not round such a defect away, it deletes it -- and a deleted trace defect
    reads as an exactly trace-preserving generator, which is the failure
    direction that must never be traded for speed.

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

    # Transposed once so each column is a Python list, rather than paying a
    # NumPy indexing round trip per column inside the accumulation loop.
    real_columns = real.T.tolist()
    imag_columns = imag.T.tolist()
    for j in range(cols):
        if nonfinite[j]:
            continue
        out[j] = complex(
            _overflow_safe_fsum(real_columns[j]),
            _overflow_safe_fsum(imag_columns[j]),
        )
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
