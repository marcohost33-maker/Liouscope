"""Numerical-contract tests for the scaled sum-of-squares primitive (#130)."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.numerics.norms import scaled_euclidean_norm


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
