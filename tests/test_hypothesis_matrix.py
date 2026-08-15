"""Issue #102 — hypothesis-wise evidence matrix + support_score semantics.

The classifier keeps returning ONE dominant class; the matrix reports, for
EVERY hypothesis of the A1-A12 taxonomy, what supported it, what contradicted
it, what evidence was missing, and what this run could honestly claim about it
(the claim floor). These tests pin four contracts:

1. **Coverage** — the matrix spans the full taxonomy: every decision rung,
   the A12 fallback, and the schema-reserved A6/A7/A9 (marked RESERVED,
   permanently UNDEFINED, excluded from claims/coverage denominators).
2. **No second copy of the conditions** — SUPPORTED in the matrix is exactly
   ``fires`` in the decision ladder, for every rung, because both are derived
   from the same ``_ladder_spec``.
3. **Claim floors follow the explicit rules** — RESERVED/UNEVALUABLE →
   UNDEFINED, NOT_SUPPORTED → NOT_EXCLUDED, SUPPORTED → the verdict the
   hypothesis would receive were it the winner (so the winner's floor equals
   the reported verdict).
4. **Report-only** — computing the matrix changes no decision, and
   ``support_score`` equals the legacy ``confidence`` value (rename option 1,
   non-probabilistic ordinal semantics).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from liouscope._consts import (
    A_CLASSES,
    REACHABLE_A_CLASSES,
    RESERVED_A_CLASSES,
    VERDICT_CANDIDATE,
    VERDICT_CONFIRMED,
    VERDICT_NOT_EXCLUDED,
    VERDICT_UNDEFINED,
)
from liouscope.diagnostics.classification import (
    HYPOTHESIS_NOT_SUPPORTED,
    HYPOTHESIS_RESERVED,
    HYPOTHESIS_SUPPORTED,
    HYPOTHESIS_UNEVALUABLE,
    _hypothesis_ladder,
    _pick_a_class,
    hypothesis_evidence_matrix,
)
from liouscope.io.export import _to_jsonable


class _Rel:
    """Minimal RelaxationResult stand-in for the fields the matrix reads."""

    def __init__(self, aicc_model: str = "M0", beta_D: float = 1.0) -> None:
        self.aicc_model = aicc_model
        self.beta_D = beta_D


def _ev(**over: float) -> dict[str, float]:
    base = {
        "mpemba_is_candidate": 0.0, "gap": 1.0, "pseudospectral_radius": 0.1,
        "gap_to_gns_ratio": 1.0, "henrici_eta": 0.1, "kreiss": 1.0,
        "petermann_max": 1.0, "trans_amplitude_ratio": 1.0, "kappa_trans": 1.0,
        "gns_certified": 0.0, "gap_rate_consistency": 1.0,
        "d17_linear_single_exp": 0.0, "has_complex_pairs": 0.0,
    }
    base.update(over)
    return base


_CASES = [
    _ev(),
    _ev(mpemba_is_candidate=1.0),
    _ev(henrici_eta=2.0, pseudospectral_radius=10.0),
    _ev(kreiss=10.0, petermann_max=10.0),
    _ev(trans_amplitude_ratio=10.0, kappa_trans=5.0),
    _ev(gap_to_gns_ratio=2.0, gns_certified=1.0),
    _ev(gap_rate_consistency=0.01, d17_linear_single_exp=1.0),
    _ev(gap_rate_consistency=0.15),
    _ev(gap=0.0, henrici_eta=2.0),  # gapless F5 limit
    _ev(mpemba_is_candidate=1.0, henrici_eta=2.0, pseudospectral_radius=10.0,
        trans_amplitude_ratio=10.0, kappa_trans=5.0, gap_rate_consistency=0.01,
        d17_linear_single_exp=1.0),
]


def _entry(matrix, rule_id):
    hits = [e for e in matrix if e["rule_id"] == rule_id]
    assert len(hits) == 1, f"expected exactly one {rule_id} entry"
    return hits[0]


# ---------------------------------------------------------------------------
# 1. Coverage over the full taxonomy
# ---------------------------------------------------------------------------


def test_matrix_covers_every_taxonomy_class():
    matrix = hypothesis_evidence_matrix(_ev(), relaxation=_Rel())
    assert {e["a_class"] for e in matrix} == set(A_CLASSES)


def test_reserved_classes_are_marked_and_excluded_from_claims():
    matrix = hypothesis_evidence_matrix(_ev(), relaxation=_Rel())
    reserved = [e for e in matrix if e["status"] == HYPOTHESIS_RESERVED]
    assert {e["a_class"] for e in reserved} == set(RESERVED_A_CLASSES)
    for e in reserved:
        assert e["claim_floor"] == VERDICT_UNDEFINED
        assert e["support_score"] is None
        assert e["f_family"] is None
        assert e["note"] == RESERVED_A_CLASSES[e["a_class"]]


def test_reachable_denominator_is_taxonomy_minus_reserved():
    assert set(REACHABLE_A_CLASSES) == set(A_CLASSES) - set(RESERVED_A_CLASSES)
    assert len(REACHABLE_A_CLASSES) == 9
    matrix = hypothesis_evidence_matrix(_ev(), relaxation=_Rel())
    evaluable = {e["a_class"] for e in matrix if e["status"] != HYPOTHESIS_RESERVED}
    assert evaluable == set(REACHABLE_A_CLASSES)


def test_import_time_coverage_guard_accepts_the_real_spec():
    from liouscope.diagnostics.classification import _reachable_coverage_check

    assert _reachable_coverage_check() == frozenset(REACHABLE_A_CLASSES)


def test_import_time_coverage_guard_fails_closed_on_drift():
    from liouscope.diagnostics.classification import _reachable_coverage_check

    with pytest.raises(RuntimeError, match="reachability gate"):
        _reachable_coverage_check(frozenset({"A1", "A6"}))


def test_matrix_order_is_decision_order_then_fallback_then_reserved():
    matrix = hypothesis_evidence_matrix(_ev(), relaxation=_Rel())
    ladder_ids = [r[0] for r in _hypothesis_ladder(_ev(), relaxation=_Rel())]
    ids = [e["rule_id"] for e in matrix]
    assert ids == [*ladder_ids, "A12_FALLBACK", "RESERVED_A6", "RESERVED_A7", "RESERVED_A9"]


# ---------------------------------------------------------------------------
# 2. SUPPORTED == the decision ladder's ``fires``, always
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ev", _CASES)
@pytest.mark.parametrize("model", ["M0", "M2", "M3a", "M3b"])
def test_supported_status_equals_ladder_fires(ev, model):
    rel = _Rel(aicc_model=model)
    ladder = {r[0]: r[3] for r in _hypothesis_ladder(ev, relaxation=rel)}
    matrix = hypothesis_evidence_matrix(ev, relaxation=rel)
    for rule_id, fires in ladder.items():
        assert (_entry(matrix, rule_id)["status"] == HYPOTHESIS_SUPPORTED) == fires


def test_a12_fallback_status_is_negation_of_any_rung_firing():
    quiet = hypothesis_evidence_matrix(_ev(), relaxation=_Rel())
    assert _entry(quiet, "A12_FALLBACK")["status"] == HYPOTHESIS_SUPPORTED

    loud = hypothesis_evidence_matrix(
        _ev(kreiss=10.0, petermann_max=10.0), relaxation=_Rel()
    )
    assert _entry(loud, "A12_FALLBACK")["status"] == HYPOTHESIS_NOT_SUPPORTED


# ---------------------------------------------------------------------------
# Supporting / counterevidence / missing content
# ---------------------------------------------------------------------------


def test_failed_condition_is_listed_as_counterevidence_with_values():
    # kreiss passes, petermann fails: F1 must show one of each, with values.
    matrix = hypothesis_evidence_matrix(
        _ev(kreiss=10.0, petermann_max=2.0), relaxation=_Rel()
    )
    f1 = _entry(matrix, "F1_OVERLAP_AMPLIFICATION")
    assert f1["status"] == HYPOTHESIS_NOT_SUPPORTED
    assert len(f1["supporting"]) == 1
    assert f1["supporting"][0]["values"] == {"kreiss": 10.0}
    assert len(f1["counterevidence"]) == 1
    assert f1["counterevidence"][0]["values"] == {"petermann_max": 2.0}


def test_missing_required_evidence_makes_the_rung_unevaluable():
    ev = _ev()
    del ev["mpemba_is_candidate"]  # Mpemba layer not run
    matrix = hypothesis_evidence_matrix(ev, relaxation=_Rel())
    f4 = _entry(matrix, "F4_MPEMBA")
    assert f4["status"] == HYPOTHESIS_UNEVALUABLE
    assert f4["missing"] == ("mpemba_is_candidate",)
    assert f4["supporting"] == () and f4["counterevidence"] == ()
    assert f4["claim_floor"] == VERDICT_UNDEFINED
    # The decision ladder must agree that an unevaluable rung cannot fire.
    ladder = {r[0]: r[3] for r in _hypothesis_ladder(ev, relaxation=_Rel())}
    assert ladder["F4_MPEMBA"] is False


def test_partially_missing_rung_keeps_its_evaluable_evidence():
    """A sibling's missing key must not discard a measured condition.

    `kreiss = 11` is genuine, measured support for F1; only `petermann_max` is
    absent. Reporting an empty evidence list here would throw away exactly the
    audit trail the matrix exists for, while the status must still degrade —
    partial evidence never promotes a hypothesis.
    """
    ev = _ev(kreiss=11.0)
    del ev["petermann_max"]
    f1 = _entry(
        hypothesis_evidence_matrix(ev, relaxation=_Rel()), "F1_OVERLAP_AMPLIFICATION"
    )
    assert f1["status"] == HYPOTHESIS_UNEVALUABLE
    assert f1["missing"] == ("petermann_max",)
    assert len(f1["supporting"]) == 1
    assert f1["supporting"][0]["values"] == {"kreiss": 11.0}
    assert f1["claim_floor"] == VERDICT_UNDEFINED
    assert f1["support_score"] is None


def test_unsupported_hypotheses_carry_no_support_score():
    """A winner-only grade beside explicit counterevidence would mislead.

    `_confidence` keys on a SUBSET of each firing rule (A3 on `kreiss` alone),
    so `kreiss = 11` with `petermann_max = 1` fails F1 while `_confidence`
    would still hand A3 the 0.85 confirmation grade.
    """
    ev = _ev(kreiss=11.0, petermann_max=1.0)
    matrix = hypothesis_evidence_matrix(ev, relaxation=_Rel())
    f1 = _entry(matrix, "F1_OVERLAP_AMPLIFICATION")
    assert f1["status"] == HYPOTHESIS_NOT_SUPPORTED
    assert f1["support_score"] is None
    for entry in matrix:
        if entry["status"] != HYPOTHESIS_SUPPORTED:
            assert entry["support_score"] is None, entry["rule_id"]
        else:
            assert isinstance(entry["support_score"], float)


def test_relaxation_fields_are_recorded_for_model_driven_rungs():
    matrix = hypothesis_evidence_matrix(_ev(), relaxation=_Rel(aicc_model="M3a"))
    a10 = _entry(matrix, "A10_JORDAN_M3A")
    assert a10["status"] == HYPOTHESIS_SUPPORTED
    assert a10["supporting"][0]["values"] == {"aicc_model": "M3a"}


# ---------------------------------------------------------------------------
# 3. Claim floors follow the explicit rules
# ---------------------------------------------------------------------------


def test_not_supported_floor_is_not_excluded_never_excluded():
    matrix = hypothesis_evidence_matrix(_ev(), relaxation=_Rel())
    for e in matrix:
        if e["status"] == HYPOTHESIS_NOT_SUPPORTED:
            assert e["claim_floor"] == VERDICT_NOT_EXCLUDED


def test_supported_a1_floor_needs_the_positive_certificate_for_confirmed():
    ev = _ev(gap_rate_consistency=0.01, d17_linear_single_exp=1.0)
    floor = _entry(
        hypothesis_evidence_matrix(ev, relaxation=_Rel()), "A1_LINEAR_SINGLE_EXP"
    )["claim_floor"]
    assert floor == VERDICT_CANDIDATE  # no sym_gap_corroborated -> no CONFIRMED

    ev["sym_gap_corroborated"] = 1.0
    floor = _entry(
        hypothesis_evidence_matrix(ev, relaxation=_Rel()), "A1_LINEAR_SINGLE_EXP"
    )["claim_floor"]
    assert floor == VERDICT_CONFIRMED


def test_supported_floor_equals_winner_verdict_for_the_winning_rung():
    """The matrix must not promise more (or less) than the classifier claims."""
    from liouscope.diagnostics.classification import (
        _apply_single_state_maxmix_floor,
        _confidence,
        _pick_verdict_tier,
    )

    for ev in _CASES:
        for model in ("M0", "M2", "M3a"):
            rel = _Rel(aicc_model=model)
            a_class, _fam = _pick_a_class(ev, relaxation=rel)
            verdict, tier = _pick_verdict_tier(a_class, rel, _confidence(ev, a_class))
            verdict, _ = _apply_single_state_maxmix_floor(
                a_class, verdict, tier,
                maximally_mixed=False, ensemble_confirmation=False,
            )
            matrix = hypothesis_evidence_matrix(ev, relaxation=rel)
            winner_entries = [
                e for e in matrix
                if e["a_class"] == a_class and e["status"] == HYPOTHESIS_SUPPORTED
            ]
            assert winner_entries, f"winner {a_class} missing from matrix"
            assert winner_entries[0]["claim_floor"] == verdict


def test_a11_single_state_maxmix_floor_reaches_the_matrix():
    ev = _ev(mpemba_is_candidate=1.0, mpemba_overlap_c1=1.0e-9)
    ev["maximally_mixed_steady_state"] = 1.0
    ev["ensemble_confirmation"] = 0.0
    matrix = hypothesis_evidence_matrix(ev, relaxation=_Rel())
    assert _entry(matrix, "F4_MPEMBA")["claim_floor"] == VERDICT_UNDEFINED

    ev["ensemble_confirmation"] = 1.0
    matrix = hypothesis_evidence_matrix(ev, relaxation=_Rel())
    assert _entry(matrix, "F4_MPEMBA")["claim_floor"] == VERDICT_CANDIDATE


def test_nonfinite_beta_d_floors_every_supported_hypothesis_to_undefined():
    ev = _ev(kreiss=10.0, petermann_max=10.0)
    matrix = hypothesis_evidence_matrix(ev, relaxation=_Rel(beta_D=float("nan")))
    for e in matrix:
        if e["status"] == HYPOTHESIS_SUPPORTED:
            assert e["claim_floor"] == VERDICT_UNDEFINED


# ---------------------------------------------------------------------------
# 4. Report-only + serialisation + support_score
# ---------------------------------------------------------------------------


def test_matrix_computation_does_not_change_the_decision():
    ev = _ev(mpemba_is_candidate=1.0, henrici_eta=2.0, pseudospectral_radius=10.0)
    rel = _Rel()
    before = _pick_a_class(ev, relaxation=rel)
    hypothesis_evidence_matrix(ev, relaxation=rel)  # must be side-effect free
    assert _pick_a_class(ev, relaxation=rel) == before


def test_matrix_is_json_serialisable_rfc8259():
    """Evidence values can legitimately be inf; the export tagging must hold."""
    ev = _ev(gap=0.0, henrici_eta=2.0)
    ev["gap_to_gns_ratio"] = float("inf")
    matrix = hypothesis_evidence_matrix(ev, relaxation=_Rel())
    text = json.dumps(_to_jsonable(matrix), allow_nan=False)  # must not raise
    restored = json.loads(text)
    assert [e["rule_id"] for e in restored] == [e["rule_id"] for e in matrix]


def test_diagnose_populates_matrix_and_support_score():
    from liouscope import diagnose
    from liouscope.core.lindblad import build_liouvillian

    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm])
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    rep = diagnose(L, rho_initial=rho0, t_grid=np.linspace(0.0, 5.0, 64))
    cls = rep.classification

    # support_score is the honestly-named twin of the legacy confidence field.
    assert cls.support_score == cls.confidence
    assert math.isfinite(cls.support_score)

    matrix = cls.hypothesis_matrix
    assert {e["a_class"] for e in matrix} == set(A_CLASSES)
    winner = [
        e for e in matrix
        if e["a_class"] == cls.a_class and e["status"] == HYPOTHESIS_SUPPORTED
    ]
    assert winner, "the reported class must be SUPPORTED in its own matrix"
    assert winner[0]["claim_floor"] == cls.verdict
    assert winner[0]["support_score"] == cls.support_score
    # F4 was evaluable on this run (Mpemba layer runs by default).
    assert _entry(matrix, "F4_MPEMBA")["status"] != HYPOTHESIS_UNEVALUABLE


def test_support_score_defaults_nan_for_legacy_construction():
    """Older serialised results / direct constructions stay valid (additive)."""
    from liouscope._types import ClassificationResult

    legacy = ClassificationResult(
        a_class="A1", f_family="none", verdict="CANDIDATE", tier="CONFIRMATION",
        confidence=0.7, evidence={},
    )
    assert math.isnan(legacy.support_score)
    assert legacy.hypothesis_matrix == ()
