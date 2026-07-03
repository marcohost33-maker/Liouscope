"""Tests for the A1-A12 mechanism classifier."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import (
    A_CLASSES,
    DIAGNOSTIC_SCHEMA_VERSION,
    F_FAMILIES,
    TAXONOMY_VERSION,
    build_liouvillian,
    classify_mechanism,
    diagnose,
)
from liouscope._consts import (
    TIER_CONFIRMATION,
    TIER_EXPLORATION,
    TIER_PUBLICATION,
    VERDICT_CANDIDATE,
    VERDICT_CONFIRMED,
    VERDICT_EXCLUDED,
    VERDICT_NOT_EXCLUDED,
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
from liouscope.diagnostics.classification import _confidence, _pick_verdict_tier


def test_a_class_set_size():
    assert len(A_CLASSES) == 12
    assert "A11" in A_CLASSES


def test_f_family_set_size():
    assert len(F_FAMILIES) == 6  # F1-F5 + "none"


def test_family_citations_consistent_between_docstring_and_consts():
    # The classifier module docstring enumerates the F1-F5 mechanism families
    # with their primary references. Those must stay in lock-step with the
    # authoritative _consts.F_FAMILY_DESCRIPTIONS (literature-validated PRL/arXiv
    # IDs). This guard prevents the two from drifting apart again.
    import liouscope.diagnostics.classification as clf
    from liouscope._consts import F_FAMILY_DESCRIPTIONS

    primary_ref_token = {
        "F1": "230604",      # PRL 125, 230604 (2020) Mori-Shirai overlap
        "F2": "070402",      # PRL 127, 070402 (2021) Liouvillian skin effect
        "F3": "230404",      # PRL 130, 230404 (2023) symmetrised gap
        "F4": "060401",      # PRL 127, 060401 (2021) quantum Mpemba
        "F5": "2306.07876",  # arXiv:2306.07876 (2023) phantom relaxation
    }
    doc_lines = (clf.__doc__ or "").splitlines()
    for fam, token in primary_ref_token.items():
        assert token in F_FAMILY_DESCRIPTIONS[fam], f"_consts drift for {fam}"
        doc_line = next(
            (ln for ln in doc_lines if ln.lstrip().startswith(f"* {fam} ")), ""
        )
        assert token in doc_line, f"docstring drift for {fam}"


def test_classification_stamps_versions(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert report.classification.taxonomy_version == TAXONOMY_VERSION
    assert report.classification.schema_version == DIAGNOSTIC_SCHEMA_VERSION


def test_classification_a_class_is_valid(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert report.classification.a_class in A_CLASSES
    assert report.classification.f_family in F_FAMILIES


def test_classification_confidence_bounded(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert 0.0 <= report.classification.confidence <= 1.0


def test_classify_mechanism_callable_directly(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    cls = classify_mechanism(
        spectral=report.spectral,
        nonnorm=report.nonnorm,
        relaxation=report.relaxation,
        resolvent=report.resolvent,
        transient=report.transient,
        lep=report.lep,
        mpemba=report.mpemba,
    )
    assert cls.a_class in A_CLASSES


# ---------------------------------------------------------------------------
# Synthetic-input regression tests pinning the A1-A12 decision tree.
#
# The decision logic in ``_pick_a_class`` / ``_pick_verdict_tier`` /
# ``_confidence`` is heuristic but deterministic; these tests freeze the
# current behaviour so an unintended change to a threshold or branch ordering
# is caught. They feed hand-built result dataclasses (the classifier only reads
# scalar fields, never the arrays) directly into ``classify_mechanism``.
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
        "is_mpemba_candidate": False,
        "expansion_alpha": 1.0,
    }
    base.update(kw)
    return MpembaResult(**base)


def _classify(
    *,
    spectral=None,
    nonnorm=None,
    relaxation=None,
    resolvent=None,
    transient=None,
    lep=None,
    mpemba=None,
):
    return classify_mechanism(
        spectral=spectral or _spectral(),
        nonnorm=nonnorm or _nonnorm(),
        relaxation=relaxation or _relaxation(),
        resolvent=resolvent or _resolvent(),
        transient=transient or _transient(),
        lep=lep or _lep(),
        mpemba=mpemba,
    )


def test_branch_default_fallback_a12():
    # All-neutral evidence -> A12 fallback, verdict NOT_EXCLUDED, low confidence.
    cls = _classify()
    assert (cls.a_class, cls.f_family) == ("A12", "none")
    assert cls.verdict == VERDICT_NOT_EXCLUDED
    assert cls.tier == TIER_EXPLORATION
    assert cls.confidence == pytest.approx(0.20)


def test_branch_a11_mpemba():
    cls = _classify(mpemba=_mpemba(overlap_c1=1.0e-6))
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.confidence == pytest.approx(0.90)
    assert cls.verdict == VERDICT_CONFIRMED
    assert cls.tier == TIER_PUBLICATION


def test_branch_a11_mpemba_weak_confidence():
    # overlap below the A11 trigger (1e-4) but above the high-confidence cut
    # (1e-5) -> default 0.5 confidence -> NOT_EXCLUDED.
    cls = _classify(mpemba=_mpemba(overlap_c1=5.0e-5))
    assert cls.a_class == "A11"
    assert cls.confidence == pytest.approx(0.5)
    assert cls.verdict == VERDICT_NOT_EXCLUDED


def test_branch_a10_phantom_relaxation():
    cls = _classify(
        resolvent=_resolvent(pseudospectral_radius=3.0),
        nonnorm=_nonnorm(henrici_eta=2.0),
    )
    assert (cls.a_class, cls.f_family) == ("A10", "F5")
    assert cls.confidence == pytest.approx(0.70)
    assert cls.verdict == VERDICT_CANDIDATE
    assert cls.tier == TIER_CONFIRMATION


def test_branch_a3_overlap_amplified():
    # A3 = overlap/eigenvector-amplified (Mori-Shirai 2020) -> family F1.
    cls = _classify(nonnorm=_nonnorm(kreiss=11.0, petermann_max=6.0))
    assert (cls.a_class, cls.f_family) == ("A3", "F1")
    assert cls.confidence == pytest.approx(0.85)


def test_branch_a4_skin_effect():
    cls = _classify(transient=_transient(trans_amplitude_ratio=11.0, kappa_trans=3.0))
    assert (cls.a_class, cls.f_family) == ("A4", "F2")
    assert cls.confidence == pytest.approx(0.85)


def test_branch_a2_symmetrised_gap():
    cls = _classify(spectral=_spectral(gns_gap=0.3))  # gap/gns = 1.667 > 1.2
    assert (cls.a_class, cls.f_family) == ("A2", "F3")
    assert cls.confidence == pytest.approx(0.85)  # ratio also > 1.5


def test_branch_a2_zero_gns_gap_infinite_ratio():
    # gns_gap == 0 -> ratio is +inf (the guarded division branch).
    cls = _classify(spectral=_spectral(gns_gap=0.0))
    assert cls.a_class == "A2"
    assert np.isinf(cls.evidence["gap_to_gns_ratio"])
    assert np.isinf(cls.evidence["kms_to_gns_ratio"])


def test_branch_a8_oscillatory_transient():
    cls = _classify(
        spectral=_spectral(has_complex_pairs=True),
        relaxation=_relaxation(aicc_model="M3b"),
    )
    assert (cls.a_class, cls.f_family) == ("A8", "none")


def test_branch_a10_jordan_block_m3a():
    cls = _classify(relaxation=_relaxation(aicc_model="M3a"))
    assert (cls.a_class, cls.f_family) == ("A10", "F5")


def test_branch_a5_biexponential_m2():
    cls = _classify(relaxation=_relaxation(aicc_model="M2"))
    assert (cls.a_class, cls.f_family) == ("A5", "none")


def test_branch_a1_strong_gap_consistency():
    cls = _classify(
        relaxation=_relaxation(aicc_model="M0"),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert (cls.a_class, cls.f_family) == ("A1", "none")
    assert cls.confidence == pytest.approx(0.95)
    assert cls.verdict == VERDICT_CONFIRMED


def test_branch_a1_moderate_gap_consistency():
    # gap_rate_consistency < 0.20 but not the M0 + < 0.05 strong combo.
    cls = _classify(lep=_lep(gap_rate_consistency=0.1))
    assert (cls.a_class, cls.f_family) == ("A1", "none")
    assert cls.confidence == pytest.approx(0.5)
    assert cls.verdict == VERDICT_NOT_EXCLUDED


def test_branch_evidence_without_mpemba():
    cls = _classify(mpemba=None)
    assert "mpemba_overlap_c1" not in cls.evidence
    assert "mpemba_expansion_alpha" not in cls.evidence


def test_undefined_when_beta_not_finite():
    cls = _classify(relaxation=_relaxation(aicc_model="M2", beta_D=float("nan")))
    assert cls.verdict == VERDICT_UNDEFINED
    assert cls.tier == TIER_EXPLORATION


# Direct unit tests for the verdict/tier mapping, including branches that are
# unreachable through the natural confidence values (e.g. EXCLUDED requires a
# non-A12 class with confidence < 0.30).
@pytest.mark.parametrize(
    "confidence,expected_verdict,expected_tier",
    [
        (0.90, VERDICT_CONFIRMED, TIER_PUBLICATION),
        (0.85, VERDICT_CONFIRMED, TIER_PUBLICATION),
        (0.70, VERDICT_CANDIDATE, TIER_CONFIRMATION),
        (0.60, VERDICT_CANDIDATE, TIER_CONFIRMATION),
        (0.45, VERDICT_NOT_EXCLUDED, TIER_EXPLORATION),
        (0.20, VERDICT_EXCLUDED, TIER_EXPLORATION),
    ],
)
def test_pick_verdict_tier_thresholds(confidence, expected_verdict, expected_tier):
    verdict, tier = _pick_verdict_tier("A3", _relaxation(), confidence)
    assert verdict == expected_verdict
    assert tier == expected_tier


def test_pick_verdict_tier_a12_short_circuits():
    verdict, tier = _pick_verdict_tier("A12", _relaxation(), 0.95)
    assert verdict == VERDICT_NOT_EXCLUDED
    assert tier == TIER_EXPLORATION


def test_confidence_default_midpoint():
    # An A-class with no matching confidence rule keeps the 0.5 prior.
    assert _confidence({}, "A8") == pytest.approx(0.5)
