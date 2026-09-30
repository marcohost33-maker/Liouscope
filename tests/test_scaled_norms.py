"""Numerical-contract tests for the scaled sum-of-squares primitive (#130)."""

from __future__ import annotations

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from liouscope.numerics.norms import (
    _fixed_point_sum,
    scaled_column_sums,
    scaled_euclidean_norm,
)

_FLOAT_MAX = math.nextafter(math.inf, 0.0)


def _correctly_rounded(exact: Fraction) -> float:
    """float64 nearest to ``exact``, ties to even, ``+-inf`` beyond the range.

    ``float(Fraction)`` is CPython's correctly rounded integer division, a code
    path the summation under test does not use; only its OverflowError needs
    translating into the IEEE-754 infinity the summation reports instead.
    """
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def test_tiny_finite_values_do_not_underflow_to_zero() -> None:
    values = np.array([1.0e-200, -2.0e-200, 3.0e-200])
    got = scaled_euclidean_norm(values)

    assert got > 0.0
    # Compare after dividing by the known input scale: pytest.approx's default
    # absolute tolerance would otherwise dwarf a 1e-200 expected value and let
    # a wildly inaccurate non-zero result pass.
    assert got / 1.0e-200 == pytest.approx(np.sqrt(14.0), rel=2.0e-15)


def test_subnormal_values_keep_a_nonzero_norm() -> None:
    values = np.array([1.0e-320, -2.0e-320, 3.0e-320])
    got = scaled_euclidean_norm(values)

    assert got > 0.0
    # Subnormal quantisation limits the attainable relative accuracy here, so
    # compare the dimensionless ratio rather than using an absolute tolerance.
    assert got / 1.0e-320 == pytest.approx(np.sqrt(14.0), rel=2.0e-4)


def test_large_finite_values_stay_finite_when_true_norm_is_representable() -> None:
    values = np.array([1.0e308, -1.0e308, 1.0e308])
    got = scaled_euclidean_norm(values)

    assert np.isfinite(got)
    assert got == pytest.approx(np.sqrt(3.0) * 1.0e308, rel=2.0e-15)


def test_nonrepresentable_true_norm_fails_as_infinity_not_a_false_finite_value() -> None:
    values = np.array([1.0e308, -1.0e308, 1.0e308, -1.0e308])
    assert scaled_euclidean_norm(values) == float("inf")


def test_complex_components_follow_the_same_scaled_contract() -> None:
    got = scaled_euclidean_norm(np.array([1.0e308 + 1.0e308j]))

    assert np.isfinite(got)
    assert got == pytest.approx(np.sqrt(2.0) * 1.0e308, rel=2.0e-15)


@pytest.mark.parametrize("scale", [1.0e-250, 1.0e-100, 1.0, 1.0e100, 1.0e250])
def test_norm_is_scale_equivariant_over_representable_ranges(scale: float) -> None:
    base = np.array([1.0, -2.0, 3.0, -4.0]) * 1.0e-10
    reference = scaled_euclidean_norm(base)
    got = scaled_euclidean_norm(scale * base)
    expected = abs(scale) * reference

    assert got > 0.0
    # Compare the ratio to one so tiny expected values cannot pass through
    # pytest.approx's default absolute tolerance.
    assert got / expected == pytest.approx(1.0, rel=3.0e-15)


def test_zero_nan_and_inf_semantics_are_explicit() -> None:
    assert scaled_euclidean_norm(np.zeros(4)) == 0.0
    assert np.isnan(scaled_euclidean_norm(np.array([1.0, np.nan])))
    assert scaled_euclidean_norm(np.array([1.0, np.inf])) == float("inf")


def _trace_row_block(d: int, upper: float) -> np.ndarray:
    """One column whose trace equation cancels exactly at the float64 ceiling."""
    block = np.zeros((d, 1), dtype=complex)
    block[: d // 2, 0] = upper
    block[d // 2 :, 0] = -upper
    return block


def test_column_sums_survive_an_intermediate_overflow() -> None:
    """#139 P2: a column may cancel to zero via partial sums outside the range.

    Checked against exact rational arithmetic rather than against another float
    computation: eight copies of 2.5e307 less eight copies of the same value is
    exactly zero, while a left-to-right running sum passes 2e308 after the
    eighth term.

    The reference here is unscaled ``math.fsum`` on the very same numbers,
    because it does not return a sentinel for this input -- it RAISES
    ``OverflowError``. The power-of-two scaling is therefore not an accuracy
    refinement in front of the accumulation; without it the accumulation has no
    result at all.
    """
    block = _trace_row_block(16, 2.5e307)

    with pytest.raises(OverflowError):
        math.fsum(block[:, 0].real.tolist())

    assert scaled_column_sums(block)[0] == 0.0


def test_column_sums_do_not_delete_a_defect_by_cancellation() -> None:
    """Ordinary summation does not round this defect away, it removes it."""
    block = np.array([[1.0], [1.0e16], [-1.0e16]], dtype=complex)

    assert scaled_column_sums(block)[0] == 1.0


def test_column_sums_scale_each_column_separately() -> None:
    """A shared scale would flush the small column into the subnormal range.

    This is the failure the overflow repair could have introduced in the
    opposite direction, and it would have been the silent one: a real trace
    defect reported as an exact zero.
    """
    values = np.array([[1.0e300, 1.0e-300], [-1.0e300, 0.0]], dtype=complex)

    got = scaled_column_sums(values)

    assert got[0] == 0.0
    assert got[1] == 1.0e-300


def test_column_sums_match_the_ordinary_product_on_benign_input() -> None:
    """The repair changes the accumulation, not the mathematics."""
    rng = np.random.default_rng(20260910)
    for d in (2, 3, 4):
        n = d * d
        operator = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        vec_i = np.eye(d, dtype=complex).reshape(-1, order="F")

        expected = vec_i.conj() @ operator
        got = scaled_column_sums(operator[np.arange(0, n, d + 1), :])

        assert np.allclose(got, expected, rtol=0.0, atol=1.0e-13)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_column_sums_keep_a_non_finite_column_non_finite(bad: float) -> None:
    """Input that genuinely is not representable must not be made finite."""
    values = np.array([[bad, 1.0], [1.0, 1.0]], dtype=complex)

    got = scaled_column_sums(values)

    assert not np.isfinite(got[0])
    assert got[1] == 2.0


def test_column_sums_refuse_a_non_matrix() -> None:
    with pytest.raises(Exception, match="2-D") as exc:
        scaled_column_sums(np.array([1.0, 2.0]))
    assert isinstance(exc.value, ValueError), (
        f"expected ValueError, got {type(exc.value).__name__}: {exc.value}"
    )


def test_column_sums_scale_the_real_and_imaginary_parts_separately() -> None:
    """The per-column scale is not enough; the components need their own.

    Found by turning the overflow finding against its own repair. A norm may
    share one scale between real and imaginary parts, because nothing cancels
    in a norm. A sum is not monotone: here the real parts annihilate and the
    entire answer is an imaginary part 600 orders of magnitude below them, so a
    shared scale flushes the answer to zero and reports an exact cancellation
    that did not happen.
    """
    values = np.array([[1.0e300 + 1.0e-300j], [-1.0e300 + 0.0j], [0.0j]], dtype=complex)

    assert scaled_column_sums(values)[0] == 1.0e-300j


def test_column_sums_keep_an_addend_below_the_dominant_scale() -> None:
    """#139, third round: a scale shifts the window, it does not widen it.

    The construction this replaced picked one power of two from the largest
    term of the column, which flushed every addend below ``max * 2**-1074`` to
    zero before the accumulation began. Here the two dominant terms annihilate
    and the surviving addend IS the answer.
    """
    values = np.array([[1.0e300], [-1.0e300], [1.0e-300]], dtype=complex)

    assert scaled_column_sums(values)[0] == 1.0e-300


def test_column_sums_keep_a_remainder_in_a_column_that_overflows() -> None:
    """The two failure modes in one column: overflow AND a subnormal remainder.

    Neither the previous construction nor a bare ``math.fsum`` can do this one.
    ``fsum`` raises ``OverflowError`` on these partial sums, and a common scale
    large enough to prevent that annihilates the 1e-320 remainder.
    """
    column = [2.5e307] * 8 + [-2.5e307] * 8 + [1.0e-320]
    values = np.array([[v] for v in column], dtype=complex)

    with pytest.raises(OverflowError):
        math.fsum(column)

    exact = sum(Fraction(v) for v in column)
    assert scaled_column_sums(values)[0] == float(exact)
    assert scaled_column_sums(values)[0] > 0.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_column_sums_report_infinity_only_when_the_true_sum_is_not_representable(
    sign: float,
) -> None:
    """Both signs: the negative one is where the infinity's sign must survive.

    The fixed-point fallback (#151) first wrote ``copysign(inf, total)``, which
    converts the integer ``total`` -- the very number that did not fit -- to a
    float and raises instead of answering.
    """
    values = np.array([[sign * 1.5e308], [sign * 1.5e308]], dtype=complex)

    got = scaled_column_sums(values)[0]

    assert np.isinf(got.real)
    assert math.copysign(1.0, got.real) == sign


def test_column_sums_match_exact_rational_arithmetic_across_the_exponent_range() -> None:
    """Property check against an oracle that cannot share the failure.

    ``fractions.Fraction`` is exact for float64 input, so it is not subject to
    the overflow and underflow this function has to survive -- unlike an
    ordinary float sum, which failed as an oracle for exactly these inputs.
    """
    rng = random.Random(20260910)
    for _ in range(500):
        column = [
            math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-1060, 1020))
            for _ in range(rng.randint(1, 12))
        ]
        # Force the cancellation that makes small addends decisive.
        column.append(-column[0])

        exact = sum(Fraction(v) for v in column)
        got = scaled_column_sums(np.array([[v] for v in column], dtype=complex))[0]

        assert got.real == float(exact), f"mismatch for {column}"


def test_overflow_fallback_is_exact_on_every_column_it_enters() -> None:
    """#151: the fallback answers every column, and every answer is exact.

    The two-band fallback this replaced was allowed to abstain (``nan``) on a
    residual it could not carry across its band boundary; its test therefore
    only required SOME answer. The fixed-point accumulator has no boundary, so
    the contract is now the strong one: no abstention, no wrong number, no
    flipped sign, over the whole exponent range and including sums that are
    genuinely not representable.

    The path is forced, and the forcing is measured rather than assumed: the
    columns are sorted largest first, so the positive terms meet before any
    negative one can cancel them and ``fsum`` overflows on every column. A
    sample that reaches the fallback once in thousands measures nothing, which
    happened once in the history of this function and was only caught by
    counting.
    """
    rng = random.Random(20260922)
    trials = 400
    entered = 0

    for _ in range(trials):
        kernel = [rng.choice([9.0e307, 1.5e308, _FLOAT_MAX]) for _ in range(rng.randint(2, 5))]
        column = (
            kernel
            + [-v for v in kernel]
            + [
                math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-1074, 1023))
                for _ in range(rng.randint(0, 6))
            ]
        )
        column.sort(reverse=True)
        try:
            math.fsum(column)
            continue
        except OverflowError:
            entered += 1

        expected = _correctly_rounded(sum(Fraction(v) for v in column))
        got = scaled_column_sums(np.array([[v] for v in column], dtype=complex))[0].real

        assert got == expected, f"returned {got!r} for an exact sum of {expected!r}"

    assert entered == trials, f"only {entered}/{trials} columns reached the fallback"


def test_overflow_fallback_answers_a_cross_band_residual_exactly() -> None:
    """#139 fifth round, closed by #151: the residual across the old boundary.

    Split at ``2**512``, the bands were ``+2**513`` and ``-2**513 + 1e-300``;
    the low band rounded to exactly ``-2**513`` and the two annihilated. The
    interim fix could only detect that loss and abstain. In fixed point the
    1e-300 is just a low-order bit of one integer, and the answer is exact.
    """
    column = [7e307] * 3 + [-7e307] * 3 + [2.0**513] + [-(2.0**511)] * 4 + [1e-300]

    assert float(sum(Fraction(v) for v in column)) == 1.0e-300
    with pytest.raises(OverflowError):
        math.fsum(column)

    got = scaled_column_sums(np.array([[v] for v in column], dtype=complex))[0]

    assert got == 1.0e-300


def test_fixed_point_sum_agrees_with_fsum_wherever_fsum_can_answer() -> None:
    """Differential check: two exact algorithms, one rounding each, one answer.

    ``math.fsum`` (Shewchuk partials) and the fixed-point accumulator share no
    code, and both claim the correctly rounded exact sum. Wherever ``fsum``
    does not overflow they must therefore agree bit for bit -- which is also
    what makes the fallback invisible to every input the fast path handles.
    Exact ties are manufactured on purpose, because ties are where a rounding
    rule that is merely "nearest" and one that is "nearest, ties to even"
    part ways.

    CPython documents ``fsum`` as accurate to under 1 ulp and "typically"
    correctly rounded, the residual being double rounding on x87 builds. The
    inputs here are fixed by seed, so a mismatch is never flakiness: it is
    a platform whose ``fsum`` is not correctly rounded, surfacing as a fact.
    On IEEE-754 double hardware (every CI runner) the count is zero.
    """
    rng = random.Random(151)
    compared = 0
    ties = 0

    for _ in range(3000):
        column = [
            math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(-1074, 1000))
            for _ in range(rng.randint(1, 10))
        ]
        if rng.random() < 0.3:
            anchor = rng.choice(column)
            column += [math.ulp(anchor) / 2.0, -anchor]
            ties += 1
        try:
            expected = math.fsum(column)
        except OverflowError:  # pragma: no cover - exponents stop at 1000
            continue
        compared += 1
        got = _fixed_point_sum(column)

        assert got == expected, f"{got!r} != fsum {expected!r} for {column}"
        assert math.copysign(1.0, got) == math.copysign(1.0, expected) or got == 0.0

    assert compared == 3000
    assert ties > 0


@pytest.mark.parametrize(
    ("tail", "expected"),
    [
        # 1 + 2**-53 lies exactly between 1 and 1 + 2**-52: the even one is 1.
        ([1.0, 2.0**-53], 1.0),
        # 1 + 3 * 2**-53 lies between 1 + 2**-52 and 1 + 2**-51: even is the latter.
        ([1.0 + 2.0**-52, 2.0**-53], 1.0 + 2.0**-51),
        # Just below a tie rounds down, just above rounds up.
        ([1.0, 2.0**-53, -(2.0**-200)], 1.0),
        ([1.0, 2.0**-53, 2.0**-200], 1.0 + 2.0**-52),
        # At the top of the range the tie rounds to 2**1024, i.e. overflows.
        ([_FLOAT_MAX, math.ulp(_FLOAT_MAX) / 2.0], math.inf),
        ([_FLOAT_MAX, math.ulp(_FLOAT_MAX) / 4.0], _FLOAT_MAX),
        # A subnormal answer is exact: no rounding happens at all.
        ([5.0e-324], 5.0e-324),
    ],
)
def test_overflow_fallback_rounds_once_ties_to_even(tail: list[float], expected: float) -> None:
    """The one rounding the fallback performs, pinned where it can go wrong.

    Each column carries a ``+-MAX`` pair ahead of the tail so that ``fsum``
    overflows and the fixed-point path is the one answering.
    """
    column = [_FLOAT_MAX, _FLOAT_MAX, -_FLOAT_MAX, -_FLOAT_MAX, *tail]
    with pytest.raises(OverflowError):
        math.fsum(column)

    got = scaled_column_sums(np.array([[v] for v in column], dtype=complex))[0].real

    assert got == expected
    assert got == _correctly_rounded(sum(Fraction(v) for v in column))


def test_overflow_fallback_answer_does_not_depend_on_input_order() -> None:
    """Exactness implies order independence; measured rather than inferred.

    The fixed-point accumulator adds integers, and integer addition is
    associative and commutative, so every permutation of a column must yield
    the identical float -- the property Neal (arXiv:1505.05571) names as the
    reason exact summation makes serial and parallel results agree. Every
    permutation here still overflows ``fsum``, so all of them are answered by
    the fallback, not by the fast path.
    """
    rng = random.Random(2026)
    column = [_FLOAT_MAX, _FLOAT_MAX, -_FLOAT_MAX, -_FLOAT_MAX, 3.0e-310, -1.0e100, 1.0e100, 7.0]
    reference = scaled_column_sums(np.array([[v] for v in column], dtype=complex))[0].real
    assert reference == _correctly_rounded(sum(Fraction(v) for v in column))

    for _ in range(50):
        permuted = column[:]
        rng.shuffle(permuted)
        # Put the two maxima first so the fast path overflows for certain.
        permuted.sort(key=lambda v: v != _FLOAT_MAX)
        with pytest.raises(OverflowError):
            math.fsum(permuted)
        got = scaled_column_sums(np.array([[v] for v in permuted], dtype=complex))[0].real
        assert got == reference


@pytest.mark.parametrize(
    ("special", "expected"),
    [
        ([math.inf], math.inf),
        ([-math.inf], -math.inf),
        ([math.nan], math.nan),
        ([math.inf, -math.inf], math.nan),
    ],
)
def test_overflow_fallback_lets_a_special_value_decide(
    special: list[float], expected: float
) -> None:
    """Defence in depth: a non-finite addend past the overflow must not raise.

    ``scaled_column_sums`` keeps such columns away from the fallback, but the
    helper's contract should not depend on its one caller: ``as_integer_ratio``
    raises on ``inf`` and ``nan``, so without the guard an overflowing column
    that also carries a special would escape as an exception instead of the
    sentinel every other path returns.
    """
    from liouscope.numerics.norms import _overflow_safe_fsum

    column = [_FLOAT_MAX, _FLOAT_MAX, *special]
    with pytest.raises(OverflowError):
        math.fsum(column)

    got = _overflow_safe_fsum(column)

    if math.isnan(expected):
        assert math.isnan(got)
    else:
        assert got == expected
