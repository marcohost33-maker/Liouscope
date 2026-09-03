"""PR #127 round-20 review findings (external).

This file is added to per finding; the section headers say which is which.

Finding 1 -- **the gauge shift overflowed and the Hermiticity gate then failed
OPEN.** ``build_liouvillian`` removed the identity component with
``np.trace(H).real / d``, forming the total before dividing. Every entry of
``H`` can be finite while their sum is not, and an infinite shift makes both
the gauge-fixed operator and the round-off allowance derived from it
non-finite -- after which ``defect > EPS * scale + allowance`` is False for
every defect there is. The sparse builder carried the identical calculation
and the identical hole.

Reading the neighbourhood turned up a SECOND route to the same fail-open that
no comment reported: the shift lies between the smallest and the largest
diagonal entry, so ``H_ii - gauge_shift`` can reach twice the largest entry and
overflow even when the shift itself is finite. ``scale`` is then ``inf`` and
the relative test is vacuous again.

The four fixtures are a set:

* B and D are the REGRESSIONS, one per route, with a defect (1e300) far above
  the round-off allowance the removed component legitimately earns;
* C and E are the NO-OVER-REJECT controls, the same two diagonals without any
  defect, which must still be accepted -- E in particular is what rules out
  "just refuse a non-finite scale", the first repair tried here, which would
  have refused an exactly Hermitian operator;
* the round-18 fixture (``1e9 * I`` plus a traceless defect of 1e-6) is
  re-measured to show that concession is unmoved.

Both builders are asserted on every fixture, because "in parity with the dense
builder" is the property that keeps them from disagreeing about the same
Hamiltonian.

KNOWN RESIDUAL, deliberately not fixed here: the reviewer's literal example
``[[1e308, 1], [0, 1e308]]`` is still accepted after this repair. Its defect of
1 is below ``d * eps * |gauge_shift| = 4.4e292``, the round-18 allowance for
the removed identity component -- a SECOND, independent mechanism with its own
recorded justification, not the overflow this finding is about. Whether that
allowance should be capped at the gauge-fixed scale is a threshold decision
with anchor consequences and is raised separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.core.lindblad import build_liouvillian
from liouscope.sparse.build import build_sparse_liouvillian


def _trace_overflows(defect: float) -> np.ndarray:
    """Diagonal sum overflows; ``defect`` sits off the diagonal."""
    H = np.array([[1.0e308, defect], [0.0, 1.0e308]], dtype=complex)
    with np.errstate(all="ignore"):
        assert np.all(np.isfinite(H))
        assert not np.isfinite(np.trace(H).real), "fixture: the trace overflows"
    return H


def _gauge_scale_overflows(defect: float) -> np.ndarray:
    """Trace is finite, but the gauge-fixed DIAGONAL is not."""
    H = np.diag([1.7e308, -1.7e308, 1.7e308]).astype(complex)
    H[0, 1] = defect
    with np.errstate(all="ignore"):
        assert np.all(np.isfinite(H))
        shift = float(np.trace(H).real) / 3.0
        assert np.isfinite(shift), "fixture: the shift itself stays finite"
        assert not np.isfinite(
            float(np.max(np.abs(H - shift * np.eye(3, dtype=complex))))
        ), "fixture: the gauge-fixed scale must overflow"
    return H


def _refusal(H: np.ndarray) -> str | None:
    """``None`` if both builders accept ``H``, else a named refusal.

    Deliberately catches broadly and reports the TYPE. A control that simply
    calls the builder dies of whatever exception the code raises, and a red
    test that died of an EXCEPTION is not evidence that the control measured
    anything -- it is indistinguishable from a control that crashed for an
    unrelated reason. Turning the outcome into a value keeps the death cause
    an assertion, which is what a reverse-mutation run can attribute.
    """
    for build in (build_liouvillian, build_sparse_liouvillian):
        try:
            with np.errstate(all="ignore"):
                build(H, [])
        except Exception as exc:  # the TYPE is the finding, so catch broadly
            return f"{build.__name__} raised {type(exc).__name__}: {exc}"
    return None


@pytest.mark.parametrize(
    ("name", "H"),
    [
        ("trace overflow", _trace_overflows(1.0e300)),
        ("gauge-fixed scale overflow", _gauge_scale_overflows(1.0e300)),
    ],
)
def test_a_defect_above_the_allowance_is_rejected_by_both_builders(
    name: str, H: np.ndarray
) -> None:
    """The regression. Before the repair both builders accepted both of these."""
    for build in (build_liouvillian, build_sparse_liouvillian):
        with pytest.raises(ValueError, match="Hermitian"), np.errstate(all="ignore"):
            build(H, [])


@pytest.mark.parametrize(
    ("name", "H"),
    [
        ("trace overflow", _trace_overflows(0.0)),
        ("gauge-fixed scale overflow", _gauge_scale_overflows(0.0)),
    ],
)
def test_an_exactly_hermitian_operator_is_still_accepted(
    name: str, H: np.ndarray
) -> None:
    """No-over-reject control, and the reason the repair is not a refusal.

    Both diagonals are exactly Hermitian. Rejecting on a non-finite derived
    quantity -- the obvious repair, and the one tried first here -- turns these
    into errors. The gate is restated at half scale instead, which cannot
    overflow and is the same predicate rather than a looser one.
    """
    assert _refusal(H) is None, f"exactly Hermitian ({name}) was refused"


def test_the_round_18_allowance_is_unmoved() -> None:
    """``1e9 * I`` plus a traceless defect of 1e-6 stays rejected.

    That fixture is the one the round-18 concession was measured against; if
    this repair had moved it, the concession would have been silently widened.
    """
    H = np.array([[1.0, 1.0e-6], [0.0, 1.0]], dtype=complex) + 1.0e9 * np.eye(
        2, dtype=complex
    )
    for build in (build_liouvillian, build_sparse_liouvillian):
        with pytest.raises(ValueError, match="Hermitian"), np.errstate(all="ignore"):
            build(H, [])


def test_the_gauge_shift_is_the_arithmetic_mean_where_it_is_representable() -> None:
    """The scaled form must not perturb the value it replaces.

    Asserted on the helper directly, over magnitudes spanning the range where
    the direct sum is fine, because the whole non-degradation claim rests on
    the scaled branch being unreachable there.
    """
    from liouscope.numerics.linalg import overflow_safe_mean_real

    rng = np.random.default_rng(20260904)
    for _ in range(500):
        d = int(rng.integers(1, 9))
        H = (
            rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        ) * float(10.0 ** int(rng.integers(-8, 9)))
        assert overflow_safe_mean_real(np.diagonal(H)) == float(
            np.trace(H).real
        ) / d
    assert overflow_safe_mean_real(np.zeros(0, dtype=complex)) == 0.0


# ---------------------------------------------------------------------------
# Finding 2 -- src/liouscope/diagnostics/relaxation.py
#
# ``1 / (r * blind_r)`` raised ZeroDivisionError whenever the PRODUCT of a
# finite positive decay rate and a finite positive grid interval underflowed to
# zero, which happens below ~5e-324 and is purely a choice of rate UNITS -- the
# ratio it computes is dimensionless. The limit of the ratio is the honest
# value: unbounded sampling resolution, which this function already encodes as
# ``inf``. NaN would have been wrong, because NaN here means "no usable
# interval", the opposite statement.
#
# The controls fix the two ends of the range: at unit scale the same generator
# on a coarse and on a fine grid must return exactly what it returned before,
# and a grid whose intervals are all zero-length must still be NaN, because
# that is the case the ``inf`` sentinel must not swallow.
# ---------------------------------------------------------------------------

from liouscope.diagnostics.relaxation import (  # noqa: E402
    decay_rates,
    samples_per_fast_efolding,
)

_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _damping(scale: float) -> np.ndarray:
    return build_liouvillian(
        np.zeros((2, 2), dtype=complex), [_SIGMA_MINUS * scale]
    )


def test_an_underflowed_resolution_ratio_is_unbounded_not_a_crash() -> None:
    """The reviewer's case: rate 1e-200 on the grid ``[0, 1e-200, 2e-200]``."""
    L = _damping(1.0e-100)
    rates = decay_rates(L)
    assert rates.size and float(rates[-1]) > 0.0, "fixture: something must decay"
    assert float(rates[-1]) * 1.0e-200 == 0.0, "fixture: the product must underflow"

    # Caught broadly and reported by TYPE. The defect IS an exception, so a
    # test that simply calls the function dies of it -- and a red test that
    # died of an exception cannot be attributed to a mutation, which is what
    # the reverse-mutation run needs. Turning the crash into a value keeps the
    # death cause an assertion in both directions.
    outcome: object
    try:
        outcome = samples_per_fast_efolding(L, np.array([0.0, 1.0e-200, 2.0e-200]))
    except Exception as exc:  # the TYPE is the finding, so catch broadly
        outcome = exc
    assert not isinstance(outcome, BaseException), (
        f"the resolution ratio raised {type(outcome).__name__}: {outcome}"
    )
    assert outcome == float("inf")
    assert not np.isnan(float(outcome)), (
        "NaN means 'no usable interval', which is the opposite claim"
    )


@pytest.mark.parametrize(
    ("grid", "expected"),
    [
        (np.array([0.0, 1.0, 2.0]), 1.0),
        (np.linspace(0.0, 1.0, 101), 100.0),
    ],
)
def test_unit_scale_resolution_is_untouched(
    grid: np.ndarray, expected: float
) -> None:
    """No-over-reach control at the other end of the range."""
    assert samples_per_fast_efolding(_damping(1.0), grid) == pytest.approx(
        expected, rel=1.0e-9
    )


def test_a_grid_with_no_usable_interval_is_still_nan() -> None:
    """The sentinel this repair must NOT swallow.

    ``inf`` now has two producers -- nothing decays, and every mode is resolved
    without bound -- so the case that genuinely has no interval to measure must
    keep reporting NaN rather than being folded into either.

    Stated precisely, because the reachable path matters: this grid is refused
    by the EARLIER non-positive-interval guard, not by the fallback at the end
    of the loop. That fallback is unreachable from any public entry point --
    once the guards pass, ``t[0] == 0`` forces a positive step into the window
    and ``t[0] > 0`` supplies a positive lead-in, so ``blind`` is never zero --
    and it was unreachable before this change too. It is kept as a defensive
    statement and is deliberately NOT claimed to be covered here.
    """
    value = samples_per_fast_efolding(_damping(1.0), np.array([0.0, 0.0]))
    assert np.isnan(value)
