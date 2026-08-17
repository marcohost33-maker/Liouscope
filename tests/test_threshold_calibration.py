"""Threshold calibration evidence: pin the SAMPLE, not just the constant.

Two P0 thresholds in this package are justified by a measured distribution:

* ``ZERO_MODE_AMBIGUITY_FACTOR = 30`` (``_consts.py``) -- the #113 split;
* ``henrici_eta > 1.0`` (``classification.py``, F5/A10 leg) -- the #101 gate.

Until this file existed, both justifications were prose. The #113 comment
quoted "83 healthy generators ... 2.38 max, 0.39 median" from a sample nobody
could re-derive, which is the same defect the repository criticised elsewhere
("calibrated from six fixtures"). Enumerating the population did not reproduce
those numbers -- see ``benchmarks/calibrate_zero_mode_ambiguity.py`` and the
committed artefact ``benchmarks/calibration/zero_mode_calibration.json``.

What is pinned here, and what deliberately is not
-------------------------------------------------
PINNED: the structure of the population (how many generators, which families,
which of them are defective by an independent oracle) and the QUALITATIVE
findings that a future edit must not silently lose.

NOT PINNED: the round-off digits. The measured quantity IS backward error, so
a different BLAS legitimately produces different values. Asserting them would
turn a platform difference into a red build and train the next reader to widen
the tolerance until the test means nothing.

Some assertions here pin a LIMITATION rather than a guarantee, in the style of
``test_zero_mode_scale.py::test_mechanism_class_is_NOT_claimed_invariant_under_rate_rescale``:
when the underlying issue is fixed they are expected to fail and be replaced by
a stronger statement, not relaxed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.nonnormality import henrici_eta_n, henrici_relative

_ROOT = Path(__file__).resolve().parent.parent
_ARTEFACT = _ROOT / "benchmarks" / "calibration" / "zero_mode_calibration.json"

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_ID2 = np.eye(2, dtype=complex)

#: The F5/A10 Henrici leg (``classification.py``). Imported as a literal rather
#: than from the rung table, which is built inside a factory function.
HENRICI_F5_GATE = 1.0
#: The F2 skin leg (``classification.py``), for the negative result below.
KAPPA_TRANS_F2_GATE = 2.0


@cache
def _calibration_module() -> Any:
    """Load the benchmark script by path (``benchmarks`` is not a package)."""
    path = _ROOT / "benchmarks" / "calibrate_zero_mode_ambiguity.py"
    spec = importlib.util.spec_from_file_location("calibrate_zero_mode_ambiguity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@cache
def _committed() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


def _amplitude_damping(gamma: float) -> np.ndarray:
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [_SM], [gamma])


def _dephasing(gamma: float) -> np.ndarray:
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [_SZ], [gamma])


# ---------------------------------------------------------------------------
# The artefact itself
# ---------------------------------------------------------------------------


def test_artefact_exists_and_declares_its_provenance() -> None:
    """A calibration artefact without its environment is prose in JSON clothing."""
    payload = _committed()
    assert payload["artefact"] == "zero_mode_ambiguity_calibration"
    for key in ("python", "numpy", "scipy", "platform", "eps"):
        assert key in payload["environment"], f"artefact does not record {key}"
    # Membership must be declared by construction, or the sample can be curated
    # into agreeing with the constant it is supposed to justify.
    assert "construction" in payload["population_rule"]["healthy"]


@pytest.mark.slow
def test_artefact_structure_is_reproducible() -> None:
    """Re-run the sweep and compare what a different BLAS cannot legitimately move.

    Counts, family names and the defect verdicts are structural: they follow
    from how the generators are BUILT and from an analytic gap that no
    eigensolver touches. If any of them drifts, the committed artefact no
    longer describes the code that claims to produce it.
    """
    payload = _committed()
    fresh = _calibration_module().run(
        payload["seed"], payload["random_gksl_draws_per_dim"]
    )

    assert fresh["summary"]["healthy"]["n"] == payload["summary"]["healthy"]["n"]
    assert fresh["summary"]["stiff"]["n"] == payload["summary"]["stiff"]["n"]
    assert fresh["families"] == payload["families"]

    old_sep, new_sep = payload["separation"], fresh["separation"]
    assert new_sep["n_stiff_by_construction"] == old_sep["n_stiff_by_construction"]
    assert new_sep["n_stiff_actually_defective"] == old_sep["n_stiff_actually_defective"]
    assert [c["label"] for c in new_sep["defective_cases"]] == [
        c["label"] for c in old_sep["defective_cases"]
    ]


def test_the_healthy_sample_is_large_enough_to_expose_an_order_statistic() -> None:
    """The old "2.38 max" was a small-sample maximum, and the artefact shows why.

    A maximum grows with n. The tail sweep in the artefact measures the same
    quantity at several sample sizes precisely so that nobody quotes one of
    them as a population bound again.
    """
    payload = _committed()
    tail = payload["tail_stability"]
    assert len(tail) >= 4, "a single sample cannot show that the max is unstable"
    small = [t for t in tail if t["draws"] == min(t2["draws"] for t2 in tail)]
    large = [t for t in tail if t["draws"] == max(t2["draws"] for t2 in tail)]
    assert small and large and small[0]["draws"] != large[0]["draws"]
    assert max(t["max"] for t in large) > max(t["max"] for t in small), (
        "the sweep must actually demonstrate that the maximum grows with n"
    )


def test_the_two_populations_are_not_cleanly_separated() -> None:
    """LIMITATION PIN. The healthy and defective ratios nearly touch.

    Measured: healthy max 4.23, lowest genuine defect 4.87 -- a factor 1.15.
    The comment this replaces claimed ~2x. No value of
    ``ZERO_MODE_AMBIGUITY_FACTOR`` separates these populations, so the constant
    must be read as a deliberately conservative one-sided choice, never as a
    classifier.

    If a future change genuinely separates them, this test SHOULD fail: replace
    it with the stronger claim rather than widening the bound.
    """
    sep = _committed()["separation"]
    factor = sep["separation_factor_healthy_max_to_lowest_defect"]
    assert 1.0 < factor < 3.0, (
        f"separation is {factor:.2f}x -- if this is now comfortable, the "
        "constant's justification changed and the comments must be re-derived"
    )


def test_the_split_provably_misses_a_real_defect() -> None:
    """The documented blind spot is measured, not asserted away.

    At a rate spread of ~1.4e15 the reported gap is wrong by ~4.7e14 while the
    in-band ratio has sunk to 4.87, below the 30x split. The check stays silent.
    Pinning this keeps the caveat in the comments honest.
    """
    sep = _committed()["separation"]
    missed = sep["defects_missed_by_current_split"]
    assert missed, (
        "the artefact no longer exhibits the documented blind spot; if the "
        "detector improved, say so in _consts.py instead of deleting this test"
    )
    by_label = {c["label"]: c for c in sep["defective_cases"]}
    for label in missed:
        assert by_label[label]["gap_rel_error"] > 1.0e6, (
            "a 'missed defect' must be a gross gap error, not a rounding quibble"
        )
        assert by_label[label]["rate_spread"] > 1.0e14


def test_stiff_gap_oracle_is_closed_form_and_needs_no_eigensolver() -> None:
    """DISCRIMINATION: the defect verdict rests on hand arithmetic, not on numpy.

    With ``H = 0`` and matrix-unit jumps ``|t><f|`` the GKSL generator sends
    ``|i><j|`` (``i != j``) to ``-(Gamma_i + Gamma_j)/2 |i><j|``: the coherence
    block is exactly diagonal, so its decay rates are known in closed form. For
    the #112 network the outflow rates are ``Gamma = [1.53e-5, 1.42e-5,
    3.67e-5+fast, 7.28e-6]`` and the slowest coherence is

        (1.42e-5 + 7.28e-6) / 2 = 1.074e-5

    independent of the fast rate. That number is the oracle the stiff
    population is judged against; if this drifts, every defect verdict in the
    artefact is unfounded.
    """
    mod = _calibration_module()
    slow = list(mod.STIFF_SLOW_RATES)
    hand_computed = (1.42e-5 + 7.28e-6) / 2.0
    for fast in mod.STIFF_FAST_RATES:
        rates = [slow[0], slow[1], slow[2], fast, slow[3]]
        assert mod._analytic_gap(rates) == pytest.approx(hand_computed, rel=1e-12), (
            f"closed-form gap moved at fast={fast:g}"
        )


# ---------------------------------------------------------------------------
# Second P0 threshold: the F5 Henrici gate. Same pattern, both populations.
# ---------------------------------------------------------------------------


def test_henrici_f5_gate_is_exactly_a_comparison_of_rate_against_one() -> None:
    """The #101 gate has a closed form on amplitude damping: ``eta_N(L) == gamma``.

    This is stronger than a distribution. For ``H = 0`` and a single
    ``sqrt(gamma) sigma_-`` channel the Schur off-diagonal mass of the
    Liouvillian is exactly ``gamma``, to the last bit, over six decades. So
    ``henrici_eta > 1.0`` is not a statement about non-normality at all -- it
    is the predicate "is the decay rate faster than one inverse time unit".

    The dimensionless twin proves the point: ``henrici_relative`` is CONSTANT
    at sqrt(0.4) across the same range, i.e. every one of these generators is
    equally non-normal while the gate flips.
    """
    reference_relative = henrici_relative(_amplitude_damping(1.0))
    assert reference_relative == pytest.approx(np.sqrt(0.4), rel=1e-12)
    for gamma in (1.0e-3, 0.1, 0.37, 1.0, 2.0, 7.5, 1.0e3):
        L = _amplitude_damping(gamma)
        assert henrici_eta_n(L) / gamma == pytest.approx(1.0, rel=1e-12), (
            f"eta_N is no longer exactly gamma at gamma={gamma:g}; the gate's "
            "meaning is not what the docstring says"
        )
        assert henrici_relative(L) == pytest.approx(reference_relative, rel=1e-12)


def test_henrici_gate_margins_on_both_populations_are_a_unit_not_a_separation() -> None:
    """Mirror of the #113 margin test, for the F5 Henrici leg.

    BELOW by construction: a normal generator. Dephasing with ``H = 0`` and a
    Hermitian jump commutes with its adjoint EXACTLY, so ``eta_N`` is exactly
    zero at every rate -- an infinite, scale-invariant margin.

    ABOVE by construction: the same amplitude-damping family at a faster rate.

    The margin between them, however, is not physical: the two generators below
    differ only in ``gamma``, have identical dimensionless non-normality, and
    sit on opposite sides of the gate. That is the #101 limitation stated as an
    assertion instead of a caveat. When #101 slice C lands this test should
    fail on its last line and be replaced by an invariance claim.
    """
    for gamma in (0.05, 0.5, 3.0, 1.0e6):
        L_normal = _dephasing(gamma)
        commutator = L_normal @ L_normal.conj().T - L_normal.conj().T @ L_normal
        assert float(np.linalg.norm(commutator)) == 0.0, "fixture must be normal"
        assert henrici_eta_n(L_normal) == 0.0
        assert henrici_relative(L_normal) == 0.0

    slow, fast = _amplitude_damping(0.5), _amplitude_damping(2.0)
    assert henrici_relative(slow) == pytest.approx(henrici_relative(fast), rel=1e-12)
    assert henrici_eta_n(slow) < HENRICI_F5_GATE < henrici_eta_n(fast), (
        "the F5 gate no longer flips on a pure rate change -- if that is "
        "intentional (#101 slice C), replace this test with an invariance claim"
    )


def test_henrici_gate_verdict_count_moves_with_the_rate_unit() -> None:
    """Population evidence for the same fact, from the committed artefact.

    The identical 32 generators are measured at three rate scales. The number
    that passes ``henrici_eta > 1.0`` rises with the scale and then saturates at
    the count of generators that are not exactly normal. A gate that measured
    physics would return the same count three times.
    """
    by_scale = _committed()["other_p0_metrics_by_rate_scale"]
    counts = {k: v["henrici_eta_above_1"] for k, v in by_scale.items()}
    assert len(counts) >= 3, "need several rate scales to see the dependence"
    assert counts["c=1"] < counts["c=1e+06"], (
        f"rescaling no longer moves the F5 verdict count: {counts}"
    )
    assert counts["c=1e+06"] == counts["c=1e+12"], (
        "saturation expected: only exactly-normal generators stay below"
    )


def test_kappa_trans_gate_has_no_positive_fixture_in_this_population() -> None:
    """NEGATIVE RESULT, recorded rather than papered over.

    ``kappa_trans > 2.0`` (F2/A4 skin leg) is the other P0 threshold that was a
    candidate for calibration here. It cannot be calibrated from this
    population: across all 96 healthy generators at all three rate scales the
    measured value never exceeds ~0.83, so the sample contains only the
    below-threshold side. The stiff generators do exceed it, but they exceed it
    because they are stiff, not because they exhibit a skin effect -- adopting
    them as the positive population would label by outcome, the exact
    circularity the calibration artefact exists to avoid.

    Calibrating this gate needs a fixture family with transient amplification
    by construction (e.g. Hatano-Nelson boundary drive). Until that exists, the
    2.0 remains uncalibrated, and this test says so out loud.
    """
    by_scale = _committed()["other_p0_metrics_by_rate_scale"]
    for scale, block in by_scale.items():
        assert block["kappa_trans_above_2"] == 0, (
            f"a healthy generator now exceeds the F2 gate at {scale}: the gate "
            "has become calibratable from this population -- write the real "
            "two-sided test and delete this one"
        )
        assert block["kappa_trans"]["max"] < KAPPA_TRANS_F2_GATE
