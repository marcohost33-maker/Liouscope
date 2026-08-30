"""Round-23 review of PR #121: a reference scale lost to underflow (finding 12).

The shape the earlier rounds keep producing -- a gate that certifies something
it did not measure -- one variable further along. trace_preservation_defect
squares its entries before summing, so a generator in small enough rate units
returns (0.0, 0.0); the applicability test 0 > tp_rtol * tiny is then
False for EVERY operator and a non-trace-preserving one becomes certificate-
applicable. The same operator fifty decades larger is correctly refused, so a
pure change of rate unit decided whether the object is a legal generator.

Every guard here is paired with a positive control: a guard that refuses
everything discriminates nothing and must not be able to pass this file. The
overflow twin has its own test for the opposite reason -- it pins a refusal
that must NOT be repaired, because the round-21 guard depends on it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.numerics.linalg import (
    certified_eig,
    certified_eigvals,
    trace_preservation_defect,
)

_H = np.diag([0.0, 1.0]).astype(complex)
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _healthy_generator() -> np.ndarray:
    """A genuine GKSL generator: the over-correction control throughout."""
    return build_liouvillian(_H, [_LOWER], [0.5])


def _non_tp_operator() -> np.ndarray:
    """The reviewer's operator: a trace defect of 3.0 at unit scale."""
    return np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)


# ---------------------------------------------------------------------------
# Finding 12: use underflow-safe norms for the TP gate.
# ---------------------------------------------------------------------------


def test_the_reference_scale_survives_the_reviewers_rate_unit() -> None:
    """``diag([0, -1e-200, -2e-200, -3e-200])`` must not lose its own scale.

    Measured before the fix: ``(defect, fro) == (0.0, 0.0)``. Two zeros are not
    a small trace defect -- they are the absence of a measurement, and the gate
    downstream cannot tell the difference.
    """
    defect, fro = trace_preservation_defect(_non_tp_operator() * 1e-200)
    assert fro > 0.0, "the Frobenius reference scale underflowed to zero"
    assert defect > 0.0, "the trace defect underflowed to zero"
    # The RATIO is the dimensionless quantity the gate actually uses, and it is
    # what must be unit-invariant: 3.0 / sqrt(14) at every scale.
    assert defect / fro == pytest.approx(3.0 / np.sqrt(14.0), rel=1e-12)


def test_a_non_tp_operator_is_refused_in_every_rate_unit() -> None:
    """The verdict must be a property of the operator, not of its units.

    This is the finding itself: at ``1e-150`` the operator was correctly
    refused and at ``1e-200`` it was certified, from the same physics.
    """
    for scale in (1.0, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300):
        op = _non_tp_operator() * scale
        for name, fn in (
            ("certified_eigvals", certified_eigvals),
            ("certified_eig", certified_eig),
        ):
            cert = fn(op)[-1]
            assert not cert.applicable, (
                f"{name} at scale {scale:g}: a trace defect of 3.0 relative to "
                f"a norm of sqrt(14) is not trace preservation, yet the "
                f"certificate was declared applicable "
                f"(certified={cert.certified})"
            )


def test_a_healthy_generator_in_the_same_rate_units_still_certifies() -> None:
    """POSITIVE CONTROL. The rescue must not become a blanket refusal.

    Without this, a guard that simply refused every small-scale operator would
    pass the two tests above.
    """
    for scale in (1.0, 1e-150, 1e-200):
        gen = _healthy_generator() * scale
        cert = certified_eigvals(gen)[-1]
        assert cert.applicable, f"a GKSL generator at scale {scale:g} was refused"
        assert cert.certified, f"a GKSL generator at scale {scale:g} was not certified"


def test_the_overflow_direction_is_deliberately_not_rescued() -> None:
    """The round-21 refusal must SURVIVE the round-23 repair.

    The two directions look symmetric and are not. ``||L||_F = inf`` for the
    round-20 counterexample (cancelling +-1e308 entries, spectrum {1,2,3,4}) is
    the input to the round-21 guard, which refuses a non-finite reference
    scale. Its true Frobenius norm is about 1.4e308 and therefore perfectly
    representable, so a scaled computation would hand back a finite number,
    readmit the operator and reopen a hole that took five interpreter versions
    of CI to close. This test goes red the moment someone makes the rescue
    symmetric.
    """
    # The round-20 operator, entry for entry (see
    # tests/test_pr121_review_round20.py::_overflowing_norm_operator).
    op = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    op[0, 1] = 1.0e308
    op[3, 1] = -1.0e308
    assert np.all(np.isfinite(op)), "the input must be FINITE for this to be the case"
    with np.errstate(over="ignore"):
        defect, fro = trace_preservation_defect(op)
    assert defect == pytest.approx(np.sqrt(17.0)), "a real, order-one trace defect"
    assert not np.isfinite(fro), (
        "the overflowing reference scale was made finite; the round-21 guard "
        "that refuses a non-finite scale now has nothing to refuse"
    )
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cert = certified_eigvals(op)[-1]
    assert not cert.applicable and not cert.certified


def test_the_exactly_zero_operator_keeps_its_zero_scale() -> None:
    """POSITIVE CONTROL for the other edge of the rescue.

    ``fro == 0`` is a lost measurement for a nonzero operator and the correct
    answer for the zero one. Rescaling by a maximum of zero is undefined, so
    the branch must not be entered.
    """
    defect, fro = trace_preservation_defect(np.zeros((4, 4), dtype=complex))
    assert (defect, fro) == (0.0, 0.0)
