"""Numerical-contract tests for the scaled sum-of-squares primitive (#130)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from liouscope.numerics.norms import scaled_column_sums, scaled_euclidean_norm


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
