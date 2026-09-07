"""PR #121 round-25 review findings: two abstentions that were not taken.

Both findings share one shape. A guard exists, the guard is correct, and the
control flow reaches the arithmetic WITHOUT passing through it, so a value that
was never measured is published as if it had been.

* ``_zhou.compute_zhou_predictor`` skips its recomputation branch -- and with
  it the spectral certificate guard -- as soon as BOTH ``gap`` and
  ``petermann_factor`` are supplied. The natural caller reuses an unresolved
  report (``gap=report.spectral.gap``, ``petermann_factor=report.nonnorm
  .petermann_max``), and both fields are then NaN, this library's
  unavailable-value sentinel. NaN is not ``None``, so the branch is skipped;
  ``nan <= 0`` is False, so the ``gap <= 0`` abstention is skipped too. The
  record returned ``converged=True`` with NaN mixing-time bounds.

* ``classification.classify_mechanism`` falls through to ``A12`` when a rung is
  UNEVALUABLE, because an unevaluable rung cannot fire. The evidence matrix
  says so honestly -- A12 ``UNEVALUABLE``, claim floor ``UNDEFINED`` -- while
  the top-level verdict published ``NOT_EXCLUDED``. The two fields of one
  report contradicted each other.

Each test carries its positive control in the same parametrisation, so a guard
that abstains unconditionally (which would pass the defect assertions alone)
fails here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_classification import (  # noqa: E402
    _lep,
    _nonnorm,
    _relaxation,
    _resolvent,
    _spectral,
    _transient,
)

from liouscope._zhou import compute_zhou_predictor  # noqa: E402
from liouscope.diagnostics.classification import (  # noqa: E402
    HYPOTHESIS_SUPPORTED,
    HYPOTHESIS_UNEVALUABLE,
    VERDICT_UNDEFINED,
    classify_mechanism,
)

# A two-level generator with one stationary and one decaying mode. The
# predictor's own recomputation resolves it, which is what makes the supplied
# NaN -- not the operator -- the sole reason for every abstention below.
_L = np.array([[0.0, 0.0], [0.0, -1.0]], dtype=complex)

_NAN = float("nan")
_INF = float("inf")


@pytest.mark.parametrize(
    ("gap", "petermann_factor", "want_converged"),
    [
        # The reviewer's case: an unresolved report handed straight back in.
        (_NAN, _NAN, False),
        # Either sentinel alone is enough; with one of them None the
        # recomputation branch runs and the NaN still has to be refused.
        (_NAN, 2.0, False),
        (1.0, _NAN, False),
        (_NAN, None, False),
        (None, _NAN, False),
        # Not a measurement either, and an infinite K would poison t_upper.
        (_INF, 2.0, False),
        (1.0, _INF, False),
        (-_INF, 2.0, False),
        # The pre-existing abstention must survive the new one.
        (-1.0, 4.0, False),
        (0.0, 4.0, False),
        # POSITIVE CONTROLS. Without these a guard that always abstains would
        # satisfy every assertion above.
        (1.0, 4.0, True),
        (None, None, True),
    ],
)
def test_supplied_values_are_validated_before_certification_is_bypassed(
    gap: float | None,
    petermann_factor: float | None,
    want_converged: bool,
) -> None:
    result = compute_zhou_predictor(
        _L, epsilon=1.0e-3, gap=gap, petermann_factor=petermann_factor
    )
    assert result.converged is want_converged
    if want_converged:
        assert np.isfinite(result.mixing_time_lower)
        assert np.isfinite(result.mixing_time_upper)
        assert result.mixing_time_upper >= result.mixing_time_lower
    else:
        # An unconverged record must not carry a usable-looking window: the
        # defect was precisely a NaN bound travelling under ``converged=True``.
        assert not np.isfinite(result.mixing_time_lower)
        assert not np.isfinite(result.mixing_time_upper)
    # The record honours what the caller passed, so a manifest reader can see
    # WHICH supplied value made the predictor abstain.
    if gap is not None:
        np.testing.assert_array_equal(result.gap, float(gap))
    if petermann_factor is not None:
        np.testing.assert_array_equal(
            result.petermann_factor, float(petermann_factor)
        )


def _classify(**nonnorm_kw):
    return classify_mechanism(
        spectral=_spectral(),
        nonnorm=_nonnorm(**nonnorm_kw),
        relaxation=_relaxation(),
        resolvent=_resolvent(),
        transient=_transient(),
        lep=_lep(),
    )


def _entry(result, a_class: str) -> dict:
    for entry in result.hypothesis_matrix:
        if entry["a_class"] == a_class:
            return entry
    raise AssertionError(f"no matrix entry for {a_class}")


def test_an_unevaluable_a12_fallback_reaches_the_top_level_verdict() -> None:
    """kreiss fires F1's first leg; D9 is withheld, so F1 cannot be decided."""
    result = _classify(kreiss=11.0, petermann_max=_NAN)

    a12 = _entry(result, "A12")
    assert a12["status"] == HYPOTHESIS_UNEVALUABLE, "precondition: F1 undecided"
    assert a12["missing"] == ("petermann_max",)
    assert result.a_class == "A12", "precondition: the fallback wins"

    # The finding: the verdict must not out-claim the winner's own audit floor.
    assert result.verdict == VERDICT_UNDEFINED
    assert result.verdict == a12["claim_floor"]


def test_a_decided_a12_fallback_keeps_its_verdict() -> None:
    """POSITIVE CONTROL: with D9 present nothing is unevaluable.

    Same class, same code path, evidence complete -- the floor must not fire.
    Without this the floor could abstain unconditionally and still pass above.
    """
    result = _classify(kreiss=11.0, petermann_max=1.0)

    a12 = _entry(result, "A12")
    assert a12["status"] == HYPOTHESIS_SUPPORTED
    assert result.a_class == "A12"
    assert result.verdict != VERDICT_UNDEFINED
    assert result.verdict == a12["claim_floor"]


def test_a_firing_rung_is_unaffected_by_the_floor() -> None:
    """POSITIVE CONTROL: a SUPPORTED winner is never unevaluable."""
    result = _classify(kreiss=11.0, petermann_max=11.0)

    assert result.a_class == "A3", "precondition: F1 fires"
    winner = _entry(result, "A3")
    assert winner["status"] == HYPOTHESIS_SUPPORTED
    assert result.verdict != VERDICT_UNDEFINED
    assert result.verdict == winner["claim_floor"]
