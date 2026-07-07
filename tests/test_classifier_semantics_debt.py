"""Contract tests for the residual classifier-semantics debt (issue #70 B3/B4).

Issue #70 A5/A6/A8/A9 were fixed in PR #81 (behaviour-changing, golden-pinned).
This module pins the two REMAINING, behaviour-PRESERVING sub-findings:

* **B3 -- reserved A-classes.** The taxonomy ``A1-A12-v3.1`` advertises twelve
  classes but ``_pick_a_class`` emits only nine; A6/A7/A9 have no branch yet.
  These are now recorded as an explicit reserved contract in
  ``_consts.RESERVED_A_CLASSES`` (mirroring ``RESERVED_DIAGNOSTIC_SLOTS`` for
  D21-D23). The tests below pin that the reserved set is exactly the set of
  never-returned classes -- so the "12 classes" name stays honest and a future
  wiring of A6/A7/A9 is forced to update the contract in lock-step.

* **B4 -- advisory (unused) evidence.** ``lep_proximity`` (D16),
  ``bohr_ap_length`` (D11) and ``mpemba_expansion_alpha`` (D20) are surfaced in
  the ``evidence`` dict but deliberately do NOT drive class/verdict/confidence.
  ``ADVISORY_EVIDENCE_KEYS`` names that contract; the metamorphic test proves the
  non-influence by perturbing each key across extreme values and asserting the
  decision is invariant. This is behaviour-preserving: it changes no real-input
  result, it only makes the pre-existing gap an auditable, tested contract.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np

from liouscope._consts import (
    A_CLASSES,
    EPS_MAXMIX,
    RESERVED_A_CLASSES,
    TIER_CONFIRMATION,
    TIER_EXPLORATION,
    VERDICT_CANDIDATE,
    VERDICT_UNDEFINED,
)
from liouscope._types import (
    LepResult,
    MpembaResult,
    NonNormalityResult,
    RelaxationResult,
    ResolventResult,
    SpectralResult,
    TransientResult,
)
from liouscope.diagnostics.classification import (
    ADVISORY_EVIDENCE_KEYS,
    ENSEMBLE_OVERRIDE_EVIDENCE_KEY,
    SINGLE_STATE_MAXMIX_FLOOR_CLASS,
    _apply_single_state_maxmix_floor,
    _confidence,
    _gather_evidence,
    _is_maximally_mixed,
    _pick_a_class,
    _pick_verdict_tier,
)

_CLF_SRC = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "liouscope"
    / "diagnostics"
    / "classification.py"
)


def _returned_a_classes_via_ast() -> set[str]:
    """Statically collect every A-class literal ``_pick_a_class`` can return.

    Reads the real module source so the reachability contract cannot silently
    drift from the code: adding/removing a ``return "Ax", ...`` branch changes
    this set and forces ``RESERVED_A_CLASSES`` to be updated in lock-step.
    """
    tree = ast.parse(_CLF_SRC.read_text(encoding="utf-8"))
    returned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_pick_a_class":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(
                    inner.value, ast.Tuple
                ):
                    first = inner.value.elts[0]
                    if isinstance(first, ast.Constant) and isinstance(
                        first.value, str
                    ):
                        returned.add(first.value)
    return returned


# ---------------------------------------------------------------------------
# B3 -- reserved A-classes A6/A7/A9
# ---------------------------------------------------------------------------


def test_reserved_a_classes_cover_a6_a7_a9_only():
    assert set(RESERVED_A_CLASSES) == {"A6", "A7", "A9"}


def test_reserved_a_classes_are_valid_taxonomy_ids():
    assert set(RESERVED_A_CLASSES).issubset(set(A_CLASSES))


def test_reserved_a_classes_marked_reserved_and_not_wired():
    for note in RESERVED_A_CLASSES.values():
        low = note.lower()
        assert "reserved" in low
        assert "no classifier branch yet" in low


def test_reachable_classes_are_exactly_taxonomy_minus_reserved():
    returned = _returned_a_classes_via_ast()
    # Only valid taxonomy ids are ever returned.
    assert returned.issubset(set(A_CLASSES))
    # The nine reachable classes are exactly the taxonomy minus the reserved set.
    assert returned == set(A_CLASSES) - set(RESERVED_A_CLASSES)
    assert len(returned) == 9


def test_reserved_classes_are_never_returned():
    returned = _returned_a_classes_via_ast()
    assert returned.isdisjoint(set(RESERVED_A_CLASSES))


# ---------------------------------------------------------------------------
# B4 -- advisory (non-influencing) evidence
# ---------------------------------------------------------------------------

_ARR = np.zeros(1, dtype=complex)


def _spectral(**kw) -> SpectralResult:
    base = {
        "gap": 0.5,
        "gns_gap": 0.5,
        "kms_gap": 0.5,
        "oscillating_gap": 0.1,
        "spectral_spread": 1.0,
        "eigenvalues": _ARR,
        "steady_state": np.zeros((1, 1), dtype=complex),
        "has_complex_pairs": False,
    }
    base.update(kw)
    return SpectralResult(**base)


def _nonnorm(**kw) -> NonNormalityResult:
    base = {
        "henrici_eta": 0.5,
        "petermann_max": 1.0,
        "petermann_factors": _ARR,
        "kreiss": 1.0,
        "bohr_ap_length": 1,
        "bohr_ap_pauli_bound": 0.0,
    }
    base.update(kw)
    return NonNormalityResult(**base)


def _relaxation(**kw) -> RelaxationResult:
    base = {
        "von_neumann_entropy": 0.0,
        "relative_entropy_curve": _ARR.real,
        "fidelity_curve": _ARR.real,
        "entanglement_asymmetry": None,
        "fits": {},
        "aicc_model": "M1",
        "beta_D": 0.5,
        "bca_ci_beta": (0.4, 0.6),
    }
    base.update(kw)
    return RelaxationResult(**base)


def _resolvent(**kw) -> ResolventResult:
    base = {
        "resolvent_peak": 1.0,
        "ridge_fwhm": 1.0,
        "pseudospectral_radius": 0.5,
        "pseudospec_eps": 1.0e-3,
    }
    base.update(kw)
    return ResolventResult(**base)


def _transient(**kw) -> TransientResult:
    base = {
        "trans_amplitude_ratio": 1.0,
        "kappa_trans": 1.0,
        "numerical_abscissa": 0.0,
    }
    base.update(kw)
    return TransientResult(**base)


def _lep(**kw) -> LepResult:
    base = {
        "lep_proximity": 1.0,
        "gap_rate_consistency": 1.0,
        "initial_state_sensitivity": 0.0,
        "lep_candidate_count": 0,
    }
    base.update(kw)
    return LepResult(**base)


def _mpemba(**kw) -> MpembaResult:
    base = {
        "overlap_c1": 0.5,
        "expansion_alpha": 1.0,
        "is_mpemba_candidate": False,
    }
    base.update(kw)
    return MpembaResult(**base)


def _full_evidence() -> dict[str, float]:
    return _gather_evidence(
        _spectral(),
        _nonnorm(),
        _relaxation(),
        _resolvent(),
        _transient(),
        _lep(),
        _mpemba(),
    )


def test_advisory_keys_are_present_in_gathered_evidence():
    ev = _full_evidence()
    for key in ADVISORY_EVIDENCE_KEYS:
        assert key in ev, f"advisory key {key!r} missing from evidence"


def test_advisory_keys_do_not_drive_the_decision():
    """Metamorphic non-influence: perturbing an advisory key leaves the full
    (a_class, f_family, verdict, tier, confidence) decision invariant."""
    relaxation = _relaxation()
    ev0 = _full_evidence()
    a0, f0 = _pick_a_class(ev0, relaxation=relaxation)
    c0 = _confidence(ev0, a0)
    v0, t0 = _pick_verdict_tier(a0, relaxation, c0)

    perturbations = [0.0, 1.0, -1.0e9, 1.0e9, math.inf, -math.inf, math.nan]
    for key in ADVISORY_EVIDENCE_KEYS:
        for val in perturbations:
            ev = dict(ev0)
            ev[key] = val
            a, f = _pick_a_class(ev, relaxation=relaxation)
            c = _confidence(ev, a)
            v, t = _pick_verdict_tier(a, relaxation, c)
            assert (a, f, v, t, c) == (a0, f0, v0, t0, c0), (
                f"advisory key {key!r}={val!r} changed the decision "
                f"{(a0, f0, v0, t0, c0)} -> {(a, f, v, t, c)}"
            )


# ---------------------------------------------------------------------------
# issue #78 / decision E0706-13 -- single-state maximally-mixed A11 floor contract
#
# Pins the CONDITIONAL insufficient-evidence floor as an explicit, tested contract
# (mirroring the RESERVED_A_CLASSES / ADVISORY_EVIDENCE_KEYS style above): the
# downgrade to UNDEFINED fires on exactly (a_class == A11) AND (rho_ss = I/d) AND
# (no ensemble confirmation), and nothing else.
# ---------------------------------------------------------------------------


def test_maxmix_floor_class_is_a11():
    # The floor targets the Mpemba class only. If a future change repoints it, the
    # contract must be updated in lock-step (this test forces that).
    assert SINGLE_STATE_MAXMIX_FLOOR_CLASS == "A11"
    assert SINGLE_STATE_MAXMIX_FLOOR_CLASS in A_CLASSES


def test_ensemble_override_key_name_is_stable():
    assert ENSEMBLE_OVERRIDE_EVIDENCE_KEY == "ensemble_confirmation"


def test_is_maximally_mixed_detects_identity_over_d():
    for d in (2, 3, 4):
        assert _is_maximally_mixed(np.eye(d, dtype=complex) / d)
    # A structured (diagonal, non-uniform) steady state is NOT maximally mixed.
    assert not _is_maximally_mixed(np.diag([0.7, 0.3]).astype(complex))
    # A pure state is not maximally mixed.
    assert not _is_maximally_mixed(np.diag([1.0, 0.0]).astype(complex))


def test_is_maximally_mixed_tolerance_is_tight():
    # A deviation an order of magnitude ABOVE the tolerance is NOT flagged (the
    # floor must not trigger on a merely near-mixed but structured state); solver
    # noise well BELOW the tolerance still counts as maximally mixed.
    d = 2
    base = np.eye(d, dtype=complex) / d
    structured = base.copy()
    structured[0, 0] += 10.0 * EPS_MAXMIX
    structured[1, 1] -= 10.0 * EPS_MAXMIX
    assert not _is_maximally_mixed(structured)
    noisy = base.copy()
    noisy[0, 1] += 1.0e-2 * EPS_MAXMIX
    assert _is_maximally_mixed(noisy)


def test_is_maximally_mixed_ignores_sub_2d_placeholders():
    # d < 2 (incl. the 1x1 synthetic placeholders used across the classifier unit
    # tests) is never flagged: maximal mixing is only meaningful for d >= 2.
    assert not _is_maximally_mixed(np.zeros((1, 1), dtype=complex))
    assert not _is_maximally_mixed(np.ones((1, 1), dtype=complex))


def test_maxmix_floor_truth_table():
    # Exhaustive contract: (verdict, tier) is downgraded to (UNDEFINED, EXPLORATION)
    # iff a_class == A11 AND maximally_mixed AND not ensemble_confirmation.
    base_v, base_t = VERDICT_CANDIDATE, TIER_CONFIRMATION
    for a_class in ("A11", "A10", "A1", "A12"):
        for mm in (False, True):
            for ens in (False, True):
                v, t = _apply_single_state_maxmix_floor(
                    a_class,
                    base_v,
                    base_t,
                    maximally_mixed=mm,
                    ensemble_confirmation=ens,
                )
                downgraded = (a_class == "A11") and mm and not ens
                if downgraded:
                    assert (v, t) == (VERDICT_UNDEFINED, TIER_EXPLORATION)
                else:
                    assert (v, t) == (base_v, base_t)


def test_advisory_non_influence_holds_across_several_base_classes():
    """Non-influence must not depend on which class is picked: repeat the probe
    on an A1 (gap-controlled) and an A10/F5 (phantom) baseline as well."""
    cases: list[dict] = [
        # A1 / "none" baseline (strong gap-rate consistency).
        {
            "relaxation": _relaxation(aicc_model="M0"),
            "lep": _lep(gap_rate_consistency=0.01),
        },
        # A10 / F5 baseline (phantom relaxation).
        {
            "resolvent": _resolvent(pseudospectral_radius=5.0),
            "nonnorm": _nonnorm(henrici_eta=2.0),
        },
    ]
    for overrides in cases:
        relaxation = overrides.get("relaxation", _relaxation())
        ev0 = _gather_evidence(
            overrides.get("spectral", _spectral()),
            overrides.get("nonnorm", _nonnorm()),
            relaxation,
            overrides.get("resolvent", _resolvent()),
            overrides.get("transient", _transient()),
            overrides.get("lep", _lep()),
            _mpemba(),
        )
        a0, f0 = _pick_a_class(ev0, relaxation=relaxation)
        c0 = _confidence(ev0, a0)
        for key in ADVISORY_EVIDENCE_KEYS:
            ev = dict(ev0)
            ev[key] = 1.0e12
            a, f = _pick_a_class(ev, relaxation=relaxation)
            assert (a, f) == (a0, f0)
            assert _confidence(ev, a) == c0
