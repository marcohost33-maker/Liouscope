"""Issue #130: componentwise trace-preservation backward-error regressions."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.numerics.linalg import trace_preservation_componentwise_error
from liouscope.numerics.norms import scaled_cancellation_ratio


def test_scaled_cancellation_ratio_exact_cancellation_at_extreme_scales() -> None:
    for scale in (1.0e-300, 1.0, 1.0e300):
        values = np.array([scale, -scale], dtype=complex)
        assert scaled_cancellation_ratio(values) == 0.0


def test_scaled_cancellation_ratio_detects_uncancelled_equation() -> None:
    for scale in (1.0e-300, 1.0, 1.0e300):
        values = np.array([scale, 0.0], dtype=complex)
        assert scaled_cancellation_ratio(values) == pytest.approx(1.0)


def test_scaled_cancellation_ratio_is_scale_invariant_for_partial_cancellation() -> None:
    reference = np.array([2.0 + 3.0j, -2.0 - 2.0j, 0.25 - 0.5j])
    expected = scaled_cancellation_ratio(reference)
    assert 0.0 < expected < 1.0
    for scale in (1.0e-250, 1.0e-100, 1.0, 1.0e100, 1.0e250):
        got = scaled_cancellation_ratio(reference * scale)
        assert got == pytest.approx(expected, rel=2.0e-15, abs=0.0)


def test_tp_componentwise_error_ignores_unrelated_huge_entries() -> None:
    # d=2 => trace-output rows are vectorised matrix rows 0 and 3.
    op = np.zeros((4, 4), dtype=complex)
    op[1, 0] = 1.0e300  # huge but irrelevant to the trace equation
    op[0, 2] = 1.0
    assert trace_preservation_componentwise_error(op) == pytest.approx(1.0)


def test_tp_componentwise_error_accepts_exact_trace_cancellation_at_huge_scale() -> None:
    op = np.zeros((4, 4), dtype=complex)
    op[0, 1] = 1.0e300
    op[3, 1] = -1.0e300
    assert trace_preservation_componentwise_error(op) == 0.0


def test_tp_componentwise_error_is_rate_unit_invariant() -> None:
    op = np.zeros((4, 4), dtype=complex)
    op[0, 0] = 3.0
    op[3, 0] = -2.0
    expected = 1.0 / 5.0
    for scale in (1.0e-300, 1.0e-200, 1.0, 1.0e200, 1.0e300):
        got = trace_preservation_componentwise_error(op * scale)
        assert got == pytest.approx(expected, rel=2.0e-15, abs=0.0)


def test_tp_componentwise_error_fails_closed_on_bad_shape_and_nonfinite() -> None:
    assert np.isnan(trace_preservation_componentwise_error(np.ones((3, 3))))
    bad = np.zeros((4, 4), dtype=complex)
    bad[0, 0] = np.nan
    assert np.isnan(trace_preservation_componentwise_error(bad))
