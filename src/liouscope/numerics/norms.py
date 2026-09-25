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

# Issue #151: every finite float64 is an integer multiple of 2**-1074, the
# smallest subnormal, so the overflow fallback in :func:`_overflow_safe_fsum`
# accumulates in those units and rounds once at the end.
_FIXED_POINT_BITS = 1074
# Significand width of float64, hidden bit included.
_SIGNIFICAND_BITS = 53


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
    """Correctly rounded exact sum of finite float64 values.

    ``math.fsum`` is already exact over the whole float64 range -- Shewchuk
    accumulation keeps a set of non-overlapping partial sums, so no bit of any
    addend is lost. Its one failure mode is an intermediate that leaves the
    range, and that depends on the ORDER of the input: eight ``+2.5e307``
    followed by eight ``-2.5e307`` raises ``OverflowError``, while the same
    sixteen values alternating return zero.

    The first attempt is therefore the plain call, and every input it can sum
    is answered by it, bit for bit as before. Only when it raises does
    :func:`_fixed_point_sum` take over. That accumulator has no intermediate
    range to leave and no partial sum to round before the end, so it returns
    the correctly rounded exact sum -- including for a residual that survives
    only across exponents far apart, the case the earlier two-band fallback
    could not carry and abstained on (issue #151). Overflow of the FINAL
    rounding is the honest answer and returns a signed infinity: the true sum
    is not representable.

    Stated precisely rather than generously: CPython documents ``fsum`` as
    accurate to under 1 ulp and "typically" correctly rounded, the residual
    being extended-precision double rounding on x87 builds. On IEEE-754
    double hardware with round-half-even -- every platform this package is
    tested on -- the two paths agree bit for bit (measured, see the
    differential test), so the fallback is invisible to inputs the fast path
    can sum.

    A non-finite addend never reaches the fallback through
    :func:`scaled_column_sums`, which sums such columns in ordinary
    arithmetic. If one does arrive here, the specials alone decide the answer
    (``inf``, ``-inf`` or ``nan``), exactly as ``fsum`` treats them, instead
    of ``as_integer_ratio`` raising from inside the accumulator.
    """
    try:
        return math.fsum(values)
    except OverflowError:
        pass
    specials = [value for value in values if not math.isfinite(value)]
    if specials:
        try:
            return math.fsum(specials)
        except ValueError:  # ``inf`` and ``-inf`` together: fsum refuses
            return math.nan
    return _fixed_point_sum(values)


def _fixed_point_sum(values: list[float]) -> float:
    """Exact sum of finite float64 values in a fixed-point integer accumulator.

    Issue #151. Every finite float64 is an integer multiple of ``2**-1074``,
    the smallest subnormal, so ``value * 2**1074`` is an exact Python integer
    for every addend -- the "long accumulator" of Kulisch, or the
    superaccumulator of Neal (arXiv:1505.05571), built here from arbitrary
    precision integers rather than fixed-width chunks. Integer addition is
    exact and associative, so the order of the input cannot matter, no partial
    sum is ever rounded, and the only rounding in the whole computation is the
    single one in :func:`_round_fixed_point`.

    This is the construction the two-band fallback it replaces pointed at: a
    band that rounds its own sum before recombination has already discarded a
    residual living only across the band boundary. Measured on
    ``[7e307]*3 + [-7e307]*3 + [2**513] + [-2**511]*4 + [1e-300]``: the bands
    returned ``nan`` (they could tell the residual was lost, not recover it);
    this returns ``1e-300``, the exact sum.

    It is kept as the fallback rather than the only path because ``fsum`` is
    much cheaper on the inputs it can sum, and the two agree on every one of
    them on IEEE-754 double hardware: both return the correctly rounded
    exact sum. The standard library's ``statistics._sum`` accumulates floats
    the same way -- ``as_integer_ratio`` numerators bucketed by power-of-two
    denominator -- so this is the stdlib's own precedent for exact float
    summation, not a private construction.
    """
    total = 0
    for value in values:
        numerator, denominator = value.as_integer_ratio()
        # ``denominator`` is ``2**k`` with ``0 <= k <= 1074`` for any finite
        # float64, so the shift is non-negative and the product exact.
        total += numerator << (_FIXED_POINT_BITS - denominator.bit_length() + 1)
    return _round_fixed_point(total)


def _round_fixed_point(total: int) -> float:
    """Round ``total * 2**-1074`` to float64, ties to even, exactly once.

    Written in integer arithmetic instead of ``total / 2**1074``: CPython's
    integer true division is correctly rounded in general, but its small-int
    fast path relies on the platform FPU and is documented to double-round on
    x87 hardware (python/cpython#142449). Doing the one rounding here keeps the
    result a property of this function, not of the build.
    """
    magnitude = abs(total)
    excess = magnitude.bit_length() - _SIGNIFICAND_BITS
    if excess > 0:
        # Keep the leading 53 bits and round the rest to nearest, ties to
        # even. A carry out of the top bit (``2**53``) is still exact below.
        kept = magnitude >> excess
        rest = magnitude - (kept << excess)
        half = 1 << (excess - 1)
        if rest > half or (rest == half and kept & 1):
            kept += 1
    else:
        # At most 53 significant bits: representable as-is. Values below
        # ``2**52`` are the subnormals, ``k * 2**-1074`` exactly.
        kept, excess = magnitude, 0
    try:
        # ``kept <= 2**53`` converts to float exactly, and scaling by a power
        # of two is exact whenever the result is in range -- a normal result
        # whenever ``excess > 0``, a subnormal or normal one otherwise.
        result = math.ldexp(float(kept), excess - _FIXED_POINT_BITS)
    except OverflowError:
        # The sign comes from an integer comparison: ``copysign(inf, total)``
        # would first convert the very integer that did not fit, and raise.
        return -math.inf if total < 0 else math.inf
    return -result if total < 0 else result


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
