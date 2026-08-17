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

from liouscope._consts import ZERO_MODE_AMBIGUITY_FACTOR
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.nonnormality import henrici_eta_n, henrici_relative

_ROOT = Path(__file__).resolve().parent.parent
_ARTEFACT = _ROOT / "benchmarks" / "calibration" / "zero_mode_calibration.json"
_SWEEP = _ROOT / "benchmarks" / "calibration" / "zero_mode_seed_sweep.json"

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
def _sweep_module() -> Any:
    """Load the seed-sweep script by path (``benchmarks`` is not a package)."""
    path = _ROOT / "benchmarks" / "calibrate_zero_mode_seed_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "calibrate_zero_mode_seed_sweep", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@cache
def _committed() -> dict[str, Any]:
    return json.loads(_ARTEFACT.read_text(encoding="utf-8"))


@cache
def _committed_sweep() -> dict[str, Any]:
    return json.loads(_SWEEP.read_text(encoding="utf-8"))


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

    Measured at this artefact's single seed: healthy max 4.23, lowest genuine
    defect 4.87 -- a factor 1.15. Over ten seeds the healthy max reaches 4.71
    and the factor falls to 1.03 (``test_seed_sweep_*`` below), so 1.15 is
    itself a draw, not a bound. The comment this replaces claimed ~2x. No value
    of ``ZERO_MODE_AMBIGUITY_FACTOR`` separates these populations, so the
    constant must be read as a deliberately conservative one-sided choice,
    never as a classifier.

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
# Seed dependence of the #113 calibration.
#
# The single-seed artefact above pins ONE draw. Its own tail block varies the
# seed by +0/+1/+2, which is a hint rather than a measurement, and the numbers
# that reached _consts.py ("max 3.36", "4.23 at n=339", "a factor 1.15") were
# read off that one draw. These tests pin what a ten-seed sweep actually shows:
# the maximum is seed noise, the p95 is not, and no healthy generator in the
# sweep comes anywhere near the shipped split.
# ---------------------------------------------------------------------------


def test_seed_sweep_artefact_exists_and_declares_its_seeds() -> None:
    """A sweep whose seeds are not written down cannot be distinguished from a re-roll."""
    payload = _committed_sweep()
    assert payload["artefact"] == "zero_mode_ambiguity_seed_sweep"
    for key in ("python", "numpy", "scipy", "platform", "eps"):
        assert key in payload["environment"], f"sweep artefact does not record {key}"
    assert len(payload["seeds"]) >= 5, (
        "fewer than five seeds cannot separate seed noise from a population fact"
    )
    assert len(set(payload["seeds"])) == len(payload["seeds"]), "seeds must be distinct"
    assert len(payload["draws_levels"]) >= 2, (
        "one sample size cannot separate the seed axis from the order-statistic axis"
    )
    # The shipped constant must be the one that was swept against, or the
    # exceedance count below is about some other threshold.
    assert payload["shipped_split"] == pytest.approx(ZERO_MODE_AMBIGUITY_FACTOR)


@pytest.mark.parametrize("path", [_ARTEFACT, _SWEEP], ids=["single_seed", "seed_sweep"])
def test_calibration_artefacts_are_standards_compliant_json(path: Path) -> None:
    """SILENT FAILURE GUARD. ``json.dumps`` writes ``Infinity``; RFC 8259 has no such literal.

    Python round-trips it without complaint, so an artefact can be committed,
    read back by every test here, and still be rejected by jq, by a JS consumer
    or by any strict re-parse. Caught exactly that way while building the seed
    sweep: the across-seed spread of the MEDIAN divides by zero at the small
    sample size, where every median is exactly 0.0, and the fallback wrote
    ``Infinity`` into the file.

    ``parse_constant`` is the only hook that sees those three bare words, so
    raising from it is the test. Reading the file with plain ``json.loads``
    would pass on a broken artefact.
    """
    def reject(literal: str) -> float:
        raise AssertionError(
            f"{path.name} contains the non-standard JSON literal {literal!r}: "
            "encode undefined values as null (see _jsonable in both benchmark "
            "scripts), do not widen this test"
        )

    json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def test_no_healthy_generator_in_the_sweep_reaches_the_shipped_split() -> None:
    """The question the ``_consts.py`` comment left open, answered with a count.

    A one-sided conservative threshold is only conservative if the healthy side
    stays below it. That is a claim about a tail, so it needs a sample, and the
    sample must be reported with the denominator it was measured against --
    both denominators, because every generator is measured at three rate scales
    and the fixed fixtures repeat in every cell.

    If this ever goes red, the constant is too low and D1 has started emitting
    false ``NaN`` on healthy physics: a P0 finding, not a test to relax.
    """
    pooled = _committed_sweep()["pooled"]
    assert pooled["n_healthy_measurements"] >= 10_000, (
        "the exceedance claim needs a sample large enough to be worth making"
    )
    assert pooled["n_distinct_base_generators"] >= 3_000
    assert pooled["n_at_or_above_shipped_split"] == 0, (
        f"{pooled['n_at_or_above_shipped_split']} healthy generators reached the "
        f"shipped split of {_committed_sweep()['shipped_split']}: the constant is "
        "too low -- do not widen this test"
    )
    # The stronger statement: they do not even reach the WEAKEST genuine defect,
    # so on this population the two are disjoint (by a hair -- see below).
    assert pooled["n_at_or_above_lowest_defect"] == 0
    assert pooled["margin_shipped_split_over_pooled_healthy_max"] > 2.0


def test_a_live_healthy_sample_stays_below_the_shipped_split() -> None:
    """DISCRIMINATION: re-measure instead of re-reading the artefact.

    Every other test in this block reads committed JSON, which pins the
    evidence but not the code. This one rebuilds the healthy population and
    compares against the constant as imported, so lowering
    ``ZERO_MODE_AMBIGUITY_FACTOR`` into the healthy distribution turns it red.

    Seed 1234 is not arbitrary: it produced the largest healthy ratio anywhere
    in the sweep, so it is the worst case this population is known to contain.
    """
    cal = _calibration_module()
    worst = 0.0
    for seed in (1234, _committed()["seed"]):
        rows = [cal.measure(g) for g in cal._healthy_generators(seed, 3)]
        assert rows, "empty sample"
        for row in rows:
            assert row.certified, f"{row.label} did not certify -- not a healthy sample"
            if np.isfinite(row.ratio):
                worst = max(worst, float(row.ratio))
    assert worst > 0.0, "a sample of all-zero ratios would pass vacuously"
    assert worst < ZERO_MODE_AMBIGUITY_FACTOR, (
        f"a freshly measured healthy generator reaches {worst:.3g}x, at or above "
        f"the shipped split {ZERO_MODE_AMBIGUITY_FACTOR:g}"
    )


def test_the_healthy_maximum_is_seed_noise_and_not_a_population_bound() -> None:
    """LIMITATION PIN. Quoting one maximum is quoting one draw.

    Two facts together make the point. First, the maximum moves substantially
    between seeds at a FIXED sample size. Second, in every cell of the sweep the
    largest healthy ratio comes from ``random_gksl`` -- the only family the seed
    controls -- so the movement is the RNG, not fixture drift.

    Note for whoever touches
    ``test_the_healthy_sample_is_large_enough_to_expose_an_order_statistic``
    above: it compares maxima across sample sizes at three neighbouring seeds.
    The sweep shows that comparison is seed-luck -- the mean maximum is 3.05 at
    n=96, 3.68 at n=339 and 3.69 at n=969, i.e. already flat, while the
    seed-to-seed spread at fixed n reaches 2.45x. That test still passes on the
    committed seed and is left alone; do not read it as evidence of growth.
    """
    payload = _committed_sweep()
    smallest = str(min(payload["draws_levels"]))
    block = payload["by_draws"][smallest]
    spread = block["across_seeds_max"]["spread_factor_max_over_min"]
    assert spread > 1.5, (
        f"the healthy maximum now moves only {spread:.2f}x across seeds -- if it "
        "has genuinely become stable, say what made it stable instead of "
        "quoting a maximum again"
    )
    families = {c["argmax_family"] for b in payload["by_draws"].values() for c in b["cells"]}
    assert families == {"random_gksl"}, (
        f"the largest healthy ratio no longer comes only from the seeded family: "
        f"{sorted(families)} -- the 'this is seed noise' diagnosis needs redoing"
    )
    # The p95 is the statistic that survives: it must move less than the max.
    largest = str(max(payload["draws_levels"]))
    big = payload["by_draws"][largest]
    assert (
        big["across_seeds_p95"]["spread_factor_max_over_min"]
        < big["across_seeds_max"]["spread_factor_max_over_min"]
    )


def test_the_seed_sweep_does_not_flatter_the_separation() -> None:
    """Sweeping seeds may only make the nearest pair TIGHTER, never wider.

    The single-seed artefact's healthy maximum is one draw; pooling more draws
    of the same population can only find a larger maximum, hence a smaller gap
    to the lowest genuine defect. If this inequality ever inverted, the two
    artefacts would be measuring different populations.

    Measured: 1.15 at the single seed, 1.03 over the sweep.
    """
    sweep = _committed_sweep()
    assert _committed()["seed"] in sweep["seeds"], (
        "the sweep must contain the single-seed artefact's seed, or the two "
        "numbers below are not comparable"
    )
    single = _committed()["separation"]
    pooled = sweep["pooled"]
    assert pooled["healthy_max_over_whole_sweep"] >= single["healthy_max_ratio"]
    assert pooled["nearest_pair_factor_over_whole_sweep"] <= (
        single["separation_factor_healthy_max_to_lowest_defect"]
    )
    assert 1.0 < pooled["nearest_pair_factor_over_whole_sweep"] < 1.5, (
        "if the populations are now comfortably apart, the constant's "
        "justification changed and the comments must be re-derived"
    )


def test_both_calibration_artefacts_agree_on_the_seed_free_stiff_reference() -> None:
    """The stiff population takes no seed, so both artefacts must see it identically.

    This is what makes the cross-artefact comparison above legitimate: the
    defect side is the same six generators, not a second sample.
    """
    single = _committed()["separation"]
    stiff = _committed_sweep()["stiff_reference"]
    assert stiff["lowest_defective_ratio"] == pytest.approx(
        single["lowest_defective_ratio"], rel=1e-12
    )
    assert stiff["missed_by_shipped_split"] == sorted(
        single["defects_missed_by_current_split"]
    )
    assert len(stiff["defective_labels"]) == single["n_stiff_actually_defective"]


@pytest.mark.slow
def test_seed_sweep_is_reproducible_in_its_structure() -> None:
    """Re-run one cell of the sweep and compare what round-off cannot move.

    Same rule as the single-seed artefact: counts and provenance are pinned,
    the digits are not, because the measured quantity IS backward error.
    """
    payload = _committed_sweep()
    mod = _sweep_module()
    cal = _calibration_module()
    seed = payload["seeds"][0]
    draws = min(payload["draws_levels"])
    fresh = mod._one_cell(
        cal, seed, draws, payload["pooled"]["lowest_defective_stiff_ratio"]
    )
    committed = next(
        c for c in payload["by_draws"][str(draws)]["cells"] if c["seed"] == seed
    )
    assert fresh["all"]["n"] == committed["all"]["n"]
    assert fresh["n_base_random_generators"] == committed["n_base_random_generators"]
    assert fresh["n_base_fixed_generators"] == committed["n_base_fixed_generators"]
    assert fresh["argmax_family"] == committed["argmax_family"]
    assert fresh["n_at_or_above_shipped_split"] == (
        committed["n_at_or_above_shipped_split"]
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
