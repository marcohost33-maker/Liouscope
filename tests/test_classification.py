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
        "expansion_alpha": 1.0,
    }
    base.update(kw)
    # Synthetic evidence carries no steady state, so mirror the real
    # (non-trivial) candidate rule: a below-threshold overlap is a candidate
    # unless the caller explicitly marks it trivial or overrides the flag.
    base.setdefault(
        "is_mpemba_candidate",
        bool(base["overlap_c1"] < 1.0e-4 and not kw.get("trivial_overlap", False)),
    )
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
    ensemble_confirmation=False,
):
    return classify_mechanism(
        spectral=spectral or _spectral(),
        nonnorm=nonnorm or _nonnorm(),
        relaxation=relaxation or _relaxation(),
        resolvent=resolvent or _resolvent(),
        transient=transient or _transient(),
        lep=lep or _lep(),
        mpemba=mpemba,
        ensemble_confirmation=ensemble_confirmation,
    )


def _maxmix_spectral(d: int = 2, **kw) -> SpectralResult:
    """A spectral result whose steady state is the maximally mixed state I/d."""
    kw.setdefault("steady_state", np.eye(d, dtype=complex) / d)
    return _spectral(**kw)


def test_branch_default_fallback_a12():
    # All-neutral evidence -> A12 fallback, verdict NOT_EXCLUDED, low confidence.
    cls = _classify()
    assert (cls.a_class, cls.f_family) == ("A12", "none")
    assert cls.verdict == VERDICT_NOT_EXCLUDED
    assert cls.tier == TIER_EXPLORATION
    assert cls.confidence == pytest.approx(0.20)


def test_branch_a11_mpemba():
    # issue #68: a single-state Mpemba overlap is CANDIDATE-grade, never
    # PUBLICATION_GRADE-confirmed (confirmation needs a reference family).
    cls = _classify(mpemba=_mpemba(overlap_c1=1.0e-6))
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.confidence == pytest.approx(0.70)
    assert cls.verdict == VERDICT_CANDIDATE
    assert cls.tier == TIER_CONFIRMATION


def test_branch_a11_mpemba_weak_confidence():
    # overlap below the A11 trigger (1e-4) but above the high-confidence cut
    # (1e-5) -> default 0.5 confidence -> NOT_EXCLUDED.
    cls = _classify(mpemba=_mpemba(overlap_c1=5.0e-5))
    assert cls.a_class == "A11"
    assert cls.confidence == pytest.approx(0.5)
    assert cls.verdict == VERDICT_NOT_EXCLUDED


def test_branch_a11_trivial_overlap_not_flagged():
    # issue #68: a symmetry-protected zero overlap (trivial_overlap=True) must
    # NOT reach A11 even though overlap_c1 is below the raw trigger.
    cls = _classify(mpemba=_mpemba(overlap_c1=1.0e-6, trivial_overlap=True))
    assert cls.a_class != "A11"
    assert cls.evidence["mpemba_trivial_overlap"] == 1.0
    assert cls.evidence["mpemba_is_candidate"] == 0.0


# --- issue #78 / decision E0706-13: single-state maximally-mixed A11 floor -----


def test_a11_single_state_maximally_mixed_is_undetermined():
    # DECISION E0706-13: a SINGLE-STATE run that picks A11 on a MAXIMALLY MIXED
    # steady state (rho_ss = I/d, which collapses to a single eigenprojector so
    # the issue-#68 triviality guard cannot fire) is INSUFFICIENT EVIDENCE, not a
    # CANDIDATE.
    # The verdict drops to UNDEFINED / EXPLORATION; the A11/F4 best-fit label and
    # the confidence heuristic are preserved (only the CLAIM changes).
    cls = _classify(
        spectral=_maxmix_spectral(2),
        mpemba=_mpemba(overlap_c1=1.0e-6),
    )
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.verdict == VERDICT_UNDEFINED
    assert cls.tier == TIER_EXPLORATION
    assert cls.confidence == pytest.approx(0.70)  # hypothesis strength unchanged
    assert cls.evidence["maximally_mixed_steady_state"] == 1.0
    assert cls.evidence["ensemble_confirmation"] == 0.0


def test_a11_maximally_mixed_ensemble_override_suppresses_floor():
    # The exclusion is CONDITIONAL: a reference-ensemble Mpemba confirmation
    # (ensemble_confirmation=True) suppresses the downgrade, so the SAME
    # maximally-mixed A11 evidence is once again allowed to be a CANDIDATE. This
    # pins the ensemble-override hook/contract (a future thermal-family path plugs
    # in here without re-touching the floor logic).
    cls = _classify(
        spectral=_maxmix_spectral(2),
        mpemba=_mpemba(overlap_c1=1.0e-6),
        ensemble_confirmation=True,
    )
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.verdict == VERDICT_CANDIDATE
    assert cls.tier == TIER_CONFIRMATION
    assert cls.confidence == pytest.approx(0.70)
    assert cls.evidence["ensemble_confirmation"] == 1.0


def test_a11_non_maximally_mixed_single_state_stays_candidate():
    # Regression: the floor is TIGHT -- an A11 pick on a structured (NOT maximally
    # mixed) steady state is unaffected and stays a CANDIDATE.
    structured = _spectral(steady_state=np.diag([0.7, 0.3]).astype(complex))
    cls = _classify(spectral=structured, mpemba=_mpemba(overlap_c1=1.0e-6))
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.verdict == VERDICT_CANDIDATE
    assert cls.evidence["maximally_mixed_steady_state"] == 0.0


def test_maximally_mixed_floor_only_affects_a11():
    # A maximally-mixed steady state must NOT touch a non-A11 verdict. Feed the
    # A10/F5 phantom branch a maximally-mixed rho_ss: it stays A10 CANDIDATE.
    cls = _classify(
        spectral=_maxmix_spectral(2),
        resolvent=_resolvent(pseudospectral_radius=5.0),
        nonnorm=_nonnorm(henrici_eta=2.0),
    )
    assert (cls.a_class, cls.f_family) == ("A10", "F5")
    assert cls.verdict == VERDICT_CANDIDATE
    assert cls.evidence["maximally_mixed_steady_state"] == 1.0


def test_branch_a10_phantom_relaxation():
    cls = _classify(
        resolvent=_resolvent(pseudospectral_radius=5.0),
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


def test_branch_a2_zero_gns_gap_is_uncertified_not_a2():
    # issue #88: gns_gap == 0 is the conservative floor sentinel ("GNS
    # uncertified"), NOT a measured symmetrised-gap reduction. The infinite
    # ratio stays in the evidence for audit, but it must no longer buy A2/F3.
    cls = _classify(spectral=_spectral(gns_gap=0.0))
    assert cls.a_class != "A2"
    assert np.isinf(cls.evidence["gap_to_gns_ratio"])
    assert np.isinf(cls.evidence["kms_to_gns_ratio"])
    assert cls.evidence["gns_certified"] == 0.0


def test_branch_a2_floored_gns_sentinel_not_a2():
    # issue #88 repro shape: gns_gap floored to O(machine-eps)*Delta
    # (~1e-12 relative) -> exploded-but-finite ratio, uncertified -> no A2/F3,
    # and in particular no CONFIRMED/PUBLICATION_GRADE off the sentinel.
    cls = _classify(spectral=_spectral(gns_gap=5.0e-12))  # gap = 0.5
    assert cls.a_class != "A2"
    assert cls.evidence["gap_to_gns_ratio"] > 1.0e10
    assert cls.evidence["gns_certified"] == 0.0
    assert not (
        cls.verdict == VERDICT_CONFIRMED and cls.tier == TIER_PUBLICATION
    )


def test_branch_a2_certified_tiny_gns_gap_still_fires():
    # Positive control for the #88 gate: a genuinely tiny but RESOLVED
    # Delta_GNS (well above GNS_CERTIFIED_RTOL * gap) is a measured dramatic
    # reduction and must still classify A2/F3 at high confidence.
    cls = _classify(spectral=_spectral(gns_gap=1.0e-4))  # gap/gns = 5e3, certified
    assert (cls.a_class, cls.f_family) == ("A2", "F3")
    assert cls.evidence["gns_certified"] == 1.0
    assert cls.confidence == pytest.approx(0.85)


def test_branch_a8_oscillatory_transient():
    cls = _classify(
        spectral=_spectral(has_complex_pairs=True),
        relaxation=_relaxation(aicc_model="M3b"),
    )
    assert (cls.a_class, cls.f_family) == ("A8", "none")


@pytest.mark.parametrize("c", [0.001, 0.01, 0.1, 1.0, 10.0, 1000.0])
def test_a8_f5_decision_rule_invariant_under_exactly_scaled_evidence(c):
    # issue #70 / #82 A8: this pins the DECISION RULE only, and deliberately
    # feeds evidence whose RATE-dimensioned quantities (gap, gns_gap, kms_gap,
    # pseudospectral_radius) are ALL hand-scaled by exactly ``c``. Given such
    # idealised, exactly-scaled evidence the dimensionless reach ``radius/gap``
    # is trivially constant, so ``_pick_a_class`` must NOT flip the A10/F5
    # verdict. That is the intended dimensional coherence of the rule itself.
    #
    # NOTE (issue #82, anti-overclaim): this does NOT show that the full D13
    # pipeline is scale-invariant. The real ``pseudospectral_radius`` is computed
    # at a *fixed* ``eps`` and therefore only scales ~linearly under L -> cL, so
    # the end-to-end reach drifts (it is scale-invariant to LEADING ORDER only).
    # That real, bounded residual drift is measured separately in
    # ``test_a8_f5_pseudospectral_reach_is_only_leading_order_scale_invariant``.
    spectral = _spectral(gap=0.5 * c, gns_gap=0.5 * c, kms_gap=0.5 * c)
    resolvent = _resolvent(pseudospectral_radius=5.0 * c)
    nonnorm = _nonnorm(henrici_eta=2.0)
    cls = _classify(spectral=spectral, resolvent=resolvent, nonnorm=nonnorm)
    assert (cls.a_class, cls.f_family) == ("A10", "F5")


def test_a8_f5_pseudospectral_reach_is_only_leading_order_scale_invariant():
    # issue #82 part 1: END-TO-END metamorphic probe that RECOMPUTES the D13
    # pseudospectral radius from a genuinely rescaled operator ``c * M`` (rather
    # than hand-scaling the stored evidence). It documents -- and bounds -- the
    # real residual drift of the dimensionless reach ``radius / gap``.
    #
    # Physics: for eps-pseudospectra ``sigma_eps(cL) = c * sigma_{eps/c}(L)``.
    # D13 (``resolvent.pseudospectral_radius``) uses a FIXED ``eps = 1e-3``, so
    # the radius scales only ~linearly and ``radius/gap`` is NOT exactly
    # invariant. As ``c`` grows the fixed eps becomes negligible and the reach
    # converges to the true spectral-radius/gap; as ``c`` shrinks the fixed eps
    # inflates the pseudospectrum and the reach grows. This test pins both the
    # existence of that drift (the overclaim would be that there is none) and its
    # boundedness (the leading-order claim that is actually true).
    from liouscope.numerics.pseudospec import pseudospectral_radius

    # Strongly non-normal operator, real eigenvalues {-1, -2, -3} (large upper
    # off-diagonal coupling => strong non-normality). Slowest decay rate = 1.0
    # is the spectral gap; it scales EXACTLY under M -> c*M. The true spectral
    # radius is 3, so the exact (eps->0) reach would be 3/1 = 3.
    g = 10.0
    base = np.array(
        [[-1.0, g, 0.0], [0.0, -2.0, g], [0.0, 0.0, -3.0]], dtype=complex
    )
    gap0 = 1.0
    eps = 1.0e-3  # the fixed D13 default (resolvent.pseudospectral_radius_diag)

    reaches = {
        c: pseudospectral_radius(c * base, eps) / (c * gap0)
        for c in (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e1, 1.0e3)
    }
    reach_small = reaches[1.0e-3]
    reach_large = reaches[1.0e3]

    # (1) The reach is NOT scale-invariant: fixed eps inflates it at small scale.
    # This is the concrete falsification of the old "scale-invariant" claim.
    assert reach_small > reach_large * 1.05

    # (2) But the drift is BOUNDED over six decades (leading-order invariance):
    # the ratio of extremes stays well under 2x. (Measured ~1.34 here.)
    assert reach_small / reach_large < 1.8

    # (3) Mechanism check: at large scale the fixed eps is negligible, so the
    # reach converges to the true spectral-radius/gap = 3.0.
    assert reach_large == pytest.approx(3.0, rel=0.05)


def test_a8_old_bare_radius_rule_would_have_flipped_under_rescale():
    # Load-bearing regression: the pre-#70 rule compared the BARE radius (a rate)
    # against 2 * gap_to_gns (dimensionless). Under a strong down-scale the bare
    # radius drops below the threshold and the OLD rule would have dropped A10/F5,
    # whereas the same physical system stays A10/F5 with the dimension-coherent
    # rule. This pins that the fix is real, not cosmetic.
    c = 0.01
    spectral = _spectral(gap=0.5 * c, gns_gap=0.5 * c, kms_gap=0.5 * c)
    resolvent = _resolvent(pseudospectral_radius=5.0 * c)
    nonnorm = _nonnorm(henrici_eta=2.0)
    cls = _classify(spectral=spectral, resolvent=resolvent, nonnorm=nonnorm)
    ev = cls.evidence
    # New (dimension-coherent) rule fires:
    assert ev["pseudospectral_radius"] / ev["gap"] > 2.0 * ev["gap_to_gns_ratio"]
    assert (cls.a_class, cls.f_family) == ("A10", "F5")
    # Old (bare-radius) rule would NOT have fired on the same rescaled system:
    assert not (ev["pseudospectral_radius"] > 2.0 * ev["gap_to_gns_ratio"])


def test_a8_f5_fires_on_gapless_strongly_nonnormal_operator():
    # issue #70 A8 edge: a vanishing spectral gap with strong non-normality is
    # the phantom/critical limit -> infinite pseudospectral reach -> F5 fires.
    cls = _classify(
        spectral=_spectral(gap=0.0),
        resolvent=_resolvent(pseudospectral_radius=5.0),
        nonnorm=_nonnorm(henrici_eta=2.0),
    )
    assert (cls.a_class, cls.f_family) == ("A10", "F5")


def test_branch_a10_jordan_block_m3a():
    cls = _classify(relaxation=_relaxation(aicc_model="M3a"))
    assert (cls.a_class, cls.f_family) == ("A10", "F5")


def test_branch_a5_biexponential_m2():
    cls = _classify(relaxation=_relaxation(aicc_model="M2"))
    assert (cls.a_class, cls.f_family) == ("A5", "none")


def test_branch_a1_strong_gap_consistency():
    # issue #70 A6: A1 (gap-controlled, primitive QMS) is the NO-gap-failure case
    # and must map to family "none", not F1 (Mori-Shirai overlap gap-FAILURE).
    cls = _classify(
        relaxation=_relaxation(aicc_model="M0"),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert (cls.a_class, cls.f_family) == ("A1", "none")
    assert cls.confidence == pytest.approx(0.95)
    assert cls.verdict == VERDICT_CONFIRMED


def test_branch_a1_moderate_gap_consistency():
    # gap_rate_consistency < 0.20 but not the M0 + < 0.05 strong combo.
    # issue #70 A6: A1 -> family "none" (no gap-failure mechanism flagged).
    cls = _classify(lep=_lep(gap_rate_consistency=0.1))
    assert (cls.a_class, cls.f_family) == ("A1", "none")
    assert cls.confidence == pytest.approx(0.5)
    assert cls.verdict == VERDICT_NOT_EXCLUDED


def test_a1_early_branch_reached_when_no_gap_failure_family():
    # LIOU-#69: the dimension-coherent A1 early branch fires when the observable
    # (linear) relaxation is a single exponential at the gap rate AND no
    # gap-failure family (F1-F5) is present.
    cls = _classify(
        relaxation=_relaxation(aicc_model="M2", linear_fit_model="M0"),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert (cls.a_class, cls.f_family) == ("A1", "none")  # issue #70 A6
    assert cls.verdict == VERDICT_CONFIRMED  # confidence 0.95 (D17 < 0.05)


def test_a1_uncorroborated_sym_gap_caps_at_candidate():
    # issue #80: the hypothetical adversary -- weakly non-normal (kreiss/
    # petermann/henrici/trans all below the F1-F5 thresholds, the defaults),
    # single-exp at the gap rate, BUT no positive symmetrised-gap certificate:
    # GNS floored/uncertified (#88) and KMS measurably reduced (gap/kms > 1.2).
    # Before #80 this earned A1 CONFIRMED / PUBLICATION_GRADE (0.95) purely off
    # threshold exhaustiveness; now it caps at CANDIDATE / CONFIRMATION (0.70).
    cls = _classify(
        spectral=_spectral(gns_gap=5.0e-12, kms_gap=0.3),  # gap 0.5
        relaxation=_relaxation(aicc_model="M0"),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert (cls.a_class, cls.f_family) == ("A1", "none")
    assert cls.evidence["sym_gap_corroborated"] == 0.0
    assert cls.confidence == pytest.approx(0.70)
    assert cls.verdict == VERDICT_CANDIDATE
    assert cls.tier == TIER_CONFIRMATION


def test_a1_kms_certificate_alone_corroborates():
    # issue #80: with the GNS gap floored/uncertified (coherent steady state,
    # #88 regime) a KMS gap EQUAL to Delta is still a measured, operator-
    # intrinsic certificate of gap-controlled contraction -> 0.95 stands.
    cls = _classify(
        spectral=_spectral(gns_gap=5.0e-12, kms_gap=0.5),  # gap 0.5
        relaxation=_relaxation(aicc_model="M0"),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert (cls.a_class, cls.f_family) == ("A1", "none")
    assert cls.evidence["sym_gap_corroborated"] == 1.0
    assert cls.confidence == pytest.approx(0.95)
    assert cls.verdict == VERDICT_CONFIRMED
    assert cls.tier == TIER_PUBLICATION


def test_a1_floored_kms_is_not_a_certificate():
    # issue #80 fail-closed: BOTH symmetrised gaps floored to ~0 -> ratios
    # explode -> no certificate; the A1 label survives (measured single-exp at
    # the gap) but only at CANDIDATE grade.
    cls = _classify(
        spectral=_spectral(gns_gap=0.0, kms_gap=0.0),
        relaxation=_relaxation(aicc_model="M0"),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert cls.a_class == "A1"
    assert cls.evidence["sym_gap_corroborated"] == 0.0
    assert cls.confidence == pytest.approx(0.70)
    assert cls.verdict == VERDICT_CANDIDATE


def test_a1_early_branch_does_not_shadow_f5_phantom():
    # LIOU-#69 (Equalita #79 adversarial): a STRONGLY non-normal phantom operator
    # (F5: pseudospectral_radius > 2*gap_to_gns AND henrici > 1 -- both
    # operator-INTRINSIC) combined with an rho_0 that excites only the slow gap
    # mode yields a clean single-exp trace distance at the gap rate (the
    # initial-state-DEPENDENT A1 signal). The A1 early branch must NOT award
    # A1/F1 CONFIRMED here; the reorder places F5 first, so the true phantom
    # mechanism wins. Same evidence, only the reorder decides A10 vs A1.
    cls = _classify(
        relaxation=_relaxation(aicc_model="M0", linear_fit_model="M0"),
        resolvent=_resolvent(pseudospectral_radius=5.0),
        nonnorm=_nonnorm(henrici_eta=2.0),
        lep=_lep(gap_rate_consistency=0.01),  # A1-early WOULD fire if it came first
    )
    assert (cls.a_class, cls.f_family) == ("A10", "F5")
    assert cls.a_class != "A1"


def test_a1_early_branch_does_not_shadow_f2_skin():
    # Same guard for the F2 skin family (operator-intrinsic trans_amplitude).
    cls = _classify(
        relaxation=_relaxation(aicc_model="M0", linear_fit_model="M0"),
        transient=_transient(trans_amplitude_ratio=11.0, kappa_trans=3.0),
        lep=_lep(gap_rate_consistency=0.01),
    )
    assert (cls.a_class, cls.f_family) == ("A4", "F2")


def test_branch_evidence_without_mpemba():
    cls = _classify(mpemba=None)
    assert "mpemba_overlap_c1" not in cls.evidence
    assert "mpemba_expansion_alpha" not in cls.evidence


def test_undefined_when_beta_not_finite():
    cls = _classify(relaxation=_relaxation(aicc_model="M2", beta_D=float("nan")))
    assert cls.verdict == VERDICT_UNDEFINED
    assert cls.tier == TIER_EXPLORATION


# Direct unit tests for the verdict/tier mapping. issue #70 A5: there is no
# longer an EXCLUDED branch -- low confidence in the best-fit class is epistemic
# uncertainty ("unresolved" = NOT_EXCLUDED), not active counter-evidence. The
# sub-0.30 row below now PINS that corrected semantics (previously it wrongly
# expected EXCLUDED, a verdict diagnose() could never emit anyway).
@pytest.mark.parametrize(
    "confidence,expected_verdict,expected_tier",
    [
        (0.90, VERDICT_CONFIRMED, TIER_PUBLICATION),
        (0.85, VERDICT_CONFIRMED, TIER_PUBLICATION),
        (0.70, VERDICT_CANDIDATE, TIER_CONFIRMATION),
        (0.60, VERDICT_CANDIDATE, TIER_CONFIRMATION),
        (0.45, VERDICT_NOT_EXCLUDED, TIER_EXPLORATION),
        (0.20, VERDICT_NOT_EXCLUDED, TIER_EXPLORATION),
        (0.0, VERDICT_NOT_EXCLUDED, TIER_EXPLORATION),
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
