"""PR #121 round-24 review findings.

Three findings, each a quantity that collapsed to a value the layer above could
not distinguish from a measurement:

* **B1** ``operator_zero_tolerance`` returned an INFINITE cutoff for an operator
  whose entries are all finite, because ``||L||_2`` overflowed on the way to a
  perfectly representable answer. A consumer filtering ``|lambda| > tol`` then
  discards every mode.
* **B2** ``trace_preservation_defect`` repaired underflow only when the WHOLE
  operator lost its scale. The numerator can lose its scale alone, and then a
  non-trace-preserving operator is declared certificate-applicable.
* **B3** ``fit_gls_ar1`` applied its degeneracy test before any shape
  validation, so a mismatched ``(t, y)`` pair came back as a plausible
  ``degenerate=True`` result instead of being refused.

Each test below is paired with a mutation in the round-24 discrimination spec:
taking the fix out must turn the test red at an ASSERTION or a missing raise,
never at an incidental crash.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.numerics.linalg import (
    certified_eig,
    certified_eigvals,
    operator_zero_tolerance,
    trace_preservation_defect,
)

# ---------------------------------------------------------------------------
# B1: keep the operator-derived tolerance finite.
# ---------------------------------------------------------------------------

#: The reviewer's operator: every entry finite, ``||L||_2 = 2e308`` is not.
_OVERFLOWING_NORM = np.full((2, 2), 1e308, dtype=complex)

#: ``rtol * eps * ||L||_2`` for that operator, computed by exact power-of-two
#: scaling: ``1000 * 2.220446049250313e-16 * 2e308``.
_REPRESENTABLE_TOLERANCE = 4.440892098500626e295


def test_the_tolerance_survives_an_operator_norm_that_does_not() -> None:
    """A finite operator must yield the finite tolerance it actually has.

    Measured before the fix: ``inf``. That is not a large tolerance, it is a
    filter no mode can pass -- ``|lambda| > inf`` is False for every mode, so a
    consumer holding the operator reads a spectrum full of huge non-zero modes
    as entirely stationary. The input was valid; only the intermediate was not.
    """
    try:
        tol = operator_zero_tolerance(_OVERFLOWING_NORM)
    except ValueError as exc:  # pragma: no cover - failure path of the finding
        raise AssertionError(
            "the tolerance was refused instead of being computed by scaling; "
            f"it is representable at ~4.4e295: {exc}"
        ) from None
    assert np.isfinite(tol), (
        "the operator-derived cutoff is not finite, so a direct consumer "
        "would discard every mode"
    )
    assert tol == pytest.approx(_REPRESENTABLE_TOLERANCE, rel=1e-12)


def test_a_complex_entry_whose_modulus_overflows_is_also_scaled() -> None:
    """``|1.3e308 + 1.3e308j|`` overflows although both components are finite.

    Same shape as the round-22 spectrum-side finding, on the operator side:
    the component-wise scaling must be on ``max(|Re|, |Im|)``, never on the
    modulus, or the repair overflows in the same place the defect did.
    """
    with warnings.catch_warnings(record=True) as caught:
        # Deliberately NOT the session's error filter: an overflow inside the
        # repair must be JUDGED by this test, not raised past it. A test that
        # dies of the very warning it was written to forbid proves that the
        # run crashed, not that the guard is load-bearing.
        warnings.simplefilter("always")
        try:
            tol = operator_zero_tolerance(np.full((2, 2), 1.3e308 + 1.3e308j))
        except Exception as exc:  # the exception TYPE is part of the finding
            raise AssertionError(
                "scaling on the modulus overflowed and the tolerance was "
                f"refused: {type(exc).__name__}: {exc}"
            ) from None
    overflowed = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not overflowed, (
        "the repair overflowed in the same place the defect did: "
        f"{[str(w.message) for w in overflowed]}"
    )
    assert np.isfinite(tol)
    assert tol > 0.0


def test_a_tolerance_that_is_itself_unrepresentable_is_refused() -> None:
    """Scaling repairs the NORM; it cannot repair an out-of-range ANSWER.

    With ``rtol = 1e300`` the returned quantity genuinely exceeds the double
    range. Fail-closed is then the only honest direction -- the same rule
    ``spectral_zero_tolerance`` applies to its own derived value -- because a
    non-finite cutoff silently reports "no non-zero modes".
    """
    with pytest.raises(ValueError, match="not finite"):
        operator_zero_tolerance(_OVERFLOWING_NORM, rtol=1e300)


def test_the_ordinary_operator_path_is_untouched() -> None:
    """Over-correction control: the finite-norm branch must not change.

    ``diag([0, -1, -2, -3])`` has ``||L||_2 = 3`` and its tolerance is
    ``1000 * eps * 3``. A repair that alters this value would move the cutoff
    for every operator in the library to fix the one that overflows.
    """
    L = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)
    expected = 1000.0 * float(np.finfo(float).eps) * 3.0
    assert operator_zero_tolerance(L) == pytest.approx(expected, rel=1e-14)
    # The zero operator keeps its documented zero-scale semantics.
    assert operator_zero_tolerance(np.zeros((4, 4), dtype=complex)) == 0.0


# ---------------------------------------------------------------------------
# B2: rescale when the TP defect alone underflows.
# ---------------------------------------------------------------------------

_H = np.diag([0.0, 1.0]).astype(complex)
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _healthy_generator() -> np.ndarray:
    """A genuine GKSL generator: ``vec(I)^H L`` is identically zero."""
    return build_liouvillian(_H, [_LOWER], [0.5])


def _numerator_only_underflow() -> np.ndarray:
    """O(1) generator carrying a representable ``1e-200`` TP violation.

    ``vec(I)^H L`` is ``[1e-200, 0, 0, 0]``. Its entries are perfectly normal
    numbers; only their SQUARES fall below ``5e-324``, so ``np.linalg.norm``
    returns ``0.0`` while ``||L||_F`` stays at 1.62 and the whole-operator
    rescue of round 23 never fires.
    """
    L = _healthy_generator().copy()
    L[0, 0] += 1.0e-200
    return L


def test_the_numerator_can_lose_its_scale_while_the_operator_keeps_its_own() -> None:
    """The finding: only the NUMERATOR underflows, so ``fro == 0`` never fires.

    Measured before the fix: ``(0.0, 1.620185174601965)``. Two numbers of which
    one is a measurement and the other is its absence -- and the gate below
    cannot tell which is which.
    """
    L = _numerator_only_underflow()
    row = np.eye(2, dtype=complex).reshape(-1, order="F").conj() @ L
    # Pin the precondition, so a numpy change cannot hollow the test out.
    assert np.all(np.isfinite(L)), "the input must be FINITE for this to be the bug"
    assert float(np.max(np.abs(row))) == pytest.approx(1.0e-200), (
        "the trace-preservation row must carry a representable violation"
    )
    defect, fro = trace_preservation_defect(L)
    assert fro == pytest.approx(1.620185174601965, rel=1e-12), (
        "the operator's own scale is ORDINARY -- that is what makes the "
        "round-23 whole-operator rescue inapplicable here"
    )
    assert defect > 0.0, (
        "the trace defect underflowed to zero; 0.0 is not a small violation, "
        "it is the absence of a measurement"
    )
    assert defect == pytest.approx(1.0e-200, rel=1e-12)


@pytest.mark.parametrize(
    ("name", "fn"),
    [("certified_eigvals", certified_eigvals), ("certified_eig", certified_eig)],
)
def test_neither_certificate_api_admits_the_operator_the_numerator_hid(
    name: str, fn: object
) -> None:
    """The downstream consequence, on both ladders.

    With ``tp_rtol = 0`` the applicability gate reads ``defect > 0.0``. An
    underflowed numerator makes that ``0.0 > 0.0`` -- False -- so a
    demonstrably non-trace-preserving operator came back
    ``applicable=True, trace_defect=0.0``. Measured on both APIs before the fix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert = fn(_numerator_only_underflow(), tp_rtol=0.0)  # type: ignore[operator]
    assert cert.trace_defect > 0.0, (
        f"{name} reported a zero trace defect for a non-TP operator"
    )
    assert cert.applicable is False, (
        f"{name} declared a non-trace-preserving operator certificate-applicable"
    )


def test_a_genuine_generator_still_reports_an_exactly_zero_defect() -> None:
    """Over-correction control: the repair must not manufacture a violation.

    A real GKSL generator has ``vec(I)^H L == 0`` exactly. ``0.0`` is then the
    CORRECT answer, not a lost one, and the new branch must leave it alone --
    otherwise every legal generator acquires a spurious trace defect and the
    gate starts refusing the physics it exists to admit.
    """
    defect, fro = trace_preservation_defect(_healthy_generator())
    assert defect == 0.0
    assert fro > 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert = certified_eigvals(_healthy_generator(), tp_rtol=0.0)
    assert cert.applicable is True


def test_the_one_directional_overflow_policy_is_untouched() -> None:
    """The round-21 decision must survive the round-24 repair.

    ``fro = inf`` is the documented INPUT to the non-finite-reference-scale
    refusal that closed the round-20 counterexample. The numerator repair is
    reachable only from an exact zero, so it can neither observe nor undo an
    overflow -- this pins that, so a later widening of the trigger cannot
    quietly readmit the operator.
    """
    # The round-20 counterexample, entry for entry: cancelling +-1e308 in the
    # ``vec(I)^H L`` combination, so the defect is sqrt(17) while ||L||_F
    # overflows.
    L = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    L[0, 1] = 1.0e308
    L[3, 1] = -1.0e308
    assert np.all(np.isfinite(L))
    with np.errstate(over="ignore"):
        defect, fro = trace_preservation_defect(L)
    assert np.isfinite(defect), "the defect itself is finite -- that is the trap"
    assert not np.isfinite(fro), (
        "if the Frobenius norm stops overflowing, the round-21 refusal this "
        "test guards no longer has an input and the test is hollow"
    )


# ---------------------------------------------------------------------------
# B3: validate fit-array shapes before the flat-curve return.
# ---------------------------------------------------------------------------


def _decay(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    return p[0] * np.exp(-p[1] * t)


def test_a_mismatched_pair_is_refused_not_reported_as_a_flat_curve() -> None:
    """The finding: the degeneracy test reads ``y`` alone and never sees ``t``.

    A 64-point grid against a one-point constant curve came back
    ``degenerate=True`` -- "no resolvable variation" asserted about a curve
    that was never supplied on that grid. The verdict does not mention the
    grid, so nothing downstream can recover the mismatch from it.

    Warnings are silenced INSIDE the block on purpose: without the guard this
    call emits the degeneracy ``RuntimeWarning``, and under the session's
    error filter the test would die of that warning instead of of the missing
    refusal. A probe must die of the thing it is probing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="same length"):
            fit_gls_ar1(
                _decay,
                np.linspace(0.0, 5.0, 64),
                np.array([1.0]),
                np.array([1.0, 1.0]),
            )


def test_an_empty_curve_against_a_populated_grid_is_refused() -> None:
    """``y.size == 0`` reaches the same early return with ``spread = 0.0``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="same length"):
            fit_gls_ar1(
                _decay,
                np.linspace(0.0, 5.0, 64),
                np.zeros(0),
                np.array([1.0, 1.0]),
            )


def test_two_dimensional_observations_are_refused() -> None:
    """Equal SIZE is not equal SHAPE, so the length test alone is not enough.

    An 8x8 pair passes ``t.size == y.size`` and, being constant, short-circuits
    to ``degenerate=True`` with a 2-D NaN residual array -- a structured answer
    about observations that are not a time series at all.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="one-dimensional"):
            fit_gls_ar1(
                _decay,
                np.ones((8, 8)),
                np.ones((8, 8)),
                np.array([1.0, 1.0]),
            )


def test_a_well_formed_flat_curve_is_still_degenerate_not_refused() -> None:
    """Over-correction control: issue #123 must survive the shape gate.

    A genuinely flat curve ON ITS OWN GRID is a legitimate measurement with
    nothing to fit. It must keep returning ``degenerate=True`` -- turning it
    into a ``ValueError`` would trade one silent failure for a loud one.
    """
    t = np.linspace(0.0, 5.0, 64)
    with pytest.warns(RuntimeWarning, match="nothing to fit"):
        out = fit_gls_ar1(_decay, t, np.zeros_like(t), np.array([1.0, 1.0]))
    assert out.degenerate is True
    assert out.success is False
