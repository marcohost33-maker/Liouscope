"""Mechanism classifier A1-A12 with F1-F5 family mapping.

The classifier emits a mechanism class ``A1..A12`` and a gap-failure *family*
``F1..F5`` (or ``"none"``). Both label sets and their literature anchors are
defined authoritatively in ``liouscope._consts`` (``A_CLASS_DESCRIPTIONS`` /
``F_FAMILY_DESCRIPTIONS``); this module must stay consistent with them. The
families denote the physical gap-failure *mechanism*, each tied to a primary
reference:

* F1 Mori-Shirai overlap amplification    (PRL 125, 230604, 2020)
* F2 Liouvillian skin effect              (PRL 127, 070402, 2021)
* F3 symmetrised Liouvillian gap          (PRL 130, 230404, 2023)
* F4 quantum Mpemba effect                (PRL 127, 060401, 2021)
* F5 phantom relaxation                   (arXiv:2306.07876, 2023)

Verdict in {CONFIRMED, CANDIDATE, NOT_EXCLUDED, UNDEFINED} (issue #70 A5: the
unreachable EXCLUDED verdict was removed). Tier in {PUBLICATION_GRADE,
CONFIRMATION, EXPLORATION}.

Anchor L: ``taxonomy_version`` is stamped on every ClassificationResult.

Taxonomy coverage (issue #70 B3): three of the twelve A-classes -- A6, A7, A9 --
have no decision branch yet and are recorded as a reserved, not-yet-reachable
contract in ``_consts.RESERVED_A_CLASSES`` (see there for the per-class
rationale). ``_pick_a_class`` therefore emits nine distinct classes; that gap is
explicit, not silent.

Advisory evidence (issue #70 B4): several diagnostics are surfaced in the
``evidence`` dict for audit/serialisation but deliberately do NOT influence the
class/verdict/confidence -- see ``ADVISORY_EVIDENCE_KEYS`` below. Wiring any of
them is a class-influencing design decision with false-positive risk and belongs
in a dedicated PR with anchor coverage + FP tests, not a blind hook-up here.
"""

from __future__ import annotations

import numpy as np

from .._consts import (
    DIAGNOSTIC_SCHEMA_VERSION,
    TAXONOMY_VERSION,
    TIER_CONFIRMATION,
    TIER_EXPLORATION,
    TIER_PUBLICATION,
    VERDICT_CANDIDATE,
    VERDICT_CONFIRMED,
    VERDICT_NOT_EXCLUDED,
    VERDICT_UNDEFINED,
)
from .._types import (
    ClassificationResult,
    LepResult,
    MpembaResult,
    NonNormalityResult,
    RelaxationResult,
    ResolventResult,
    SpectralResult,
    TransientResult,
)

# issue #70 B4: evidence keys that are gathered into the ``evidence`` dict (for
# audit / serialisation) but are, by design, NOT read by any of the decision
# functions ``_pick_a_class`` / ``_confidence`` / ``_pick_verdict_tier``. They are
# advisory context, not classification drivers. Making the non-influence an
# explicit, pinned contract (rather than an accidental gap) is the
# behaviour-preserving resolution of the "gathered evidence unused" debt: it
# changes no real-input result. A metamorphic test
# (``tests/test_classifier_semantics_debt.py``) proves that perturbing any of
# these keys leaves (a_class, f_family, verdict, confidence) invariant. Wiring
# one into the decision -- e.g. D18 ``initial_state_sensitivity`` as a *confidence
# dampener* for the state-dependent A1 label -- is a class-influencing change with
# false-positive risk and must ship in a dedicated PR with anchor + FP coverage,
# not be blind-hooked here. Note: the LEP layer also computes
# ``lep_candidate_count``, ``initial_state_sensitivity`` (D18) and ``ridge_fwhm``
# (D12) in their Result dataclasses; those are not even surfaced in ``evidence``
# and so are a fortiori non-influencing (documented for completeness).
ADVISORY_EVIDENCE_KEYS: frozenset[str] = frozenset(
    {
        "lep_proximity",           # D16 eigenvalue-coalescence proximity
        "bohr_ap_length",          # D11 Bohr almost-periodicity depth
        "mpemba_expansion_alpha",  # D20 Phi_n scaling exponent (present iff mpemba)
    }
)


def _gather_evidence(
    spectral: SpectralResult,
    nonnorm: NonNormalityResult,
    relaxation: RelaxationResult,
    resolvent: ResolventResult,
    transient: TransientResult,
    lep: LepResult,
    mpemba: MpembaResult | None,
) -> dict[str, float]:
    ev: dict[str, float] = {}
    ev["gap_to_gns_ratio"] = (
        float(spectral.gap / spectral.gns_gap) if spectral.gns_gap > 0 else float("inf")
    )
    ev["kms_to_gns_ratio"] = (
        float(spectral.kms_gap / spectral.gns_gap)
        if spectral.gns_gap > 0
        else float("inf")
    )
    ev["has_complex_pairs"] = float(spectral.has_complex_pairs)
    ev["kreiss"] = float(nonnorm.kreiss)
    ev["petermann_max"] = float(nonnorm.petermann_max)
    ev["henrici_eta"] = float(nonnorm.henrici_eta)
    ev["bohr_ap_length"] = float(nonnorm.bohr_ap_length)
    ev["kappa_trans"] = float(transient.kappa_trans)
    ev["trans_amplitude_ratio"] = float(transient.trans_amplitude_ratio)
    ev["lep_proximity"] = float(lep.lep_proximity)
    ev["gap_rate_consistency"] = float(lep.gap_rate_consistency)
    # LIOU-#69: expose the D17 inputs explicitly so the metric multiplier is
    # auditable rather than hidden. ``beta_D`` is the relative-entropy rate;
    # ``beta_D_linear`` is the dimension-coherent trace-distance rate fed to D17;
    # ``gap`` is the spectral gap Delta. The implied relative-entropy metric
    # multiplier m = beta_D / beta_D_linear is ~2 for a faithful (full-rank)
    # steady state and ~1 for a rank-deficient one (verified across V1-V5).
    ev["beta_D"] = float(relaxation.beta_D)
    ev["beta_D_linear"] = float(relaxation.beta_D_linear)
    ev["gap"] = float(spectral.gap)
    _blin = float(relaxation.beta_D_linear)
    ev["d17_metric_multiplier"] = (
        float(relaxation.beta_D / _blin)
        if np.isfinite(_blin) and _blin > 0.0 and np.isfinite(relaxation.beta_D)
        else float("nan")
    )
    # 1.0 iff the observable (linear-metric) relaxation is a single exponential
    # (M0/M1) -- i.e. one dominant mode. Combined with a small
    # gap_rate_consistency this is the textbook signature of gap-controlled
    # relaxation, independent of the relative-entropy fit SHAPE.
    ev["d17_linear_single_exp"] = (
        1.0 if relaxation.linear_fit_model in ("M0", "M1") else 0.0
    )
    ev["resolvent_peak"] = float(resolvent.resolvent_peak)
    ev["pseudospectral_radius"] = float(resolvent.pseudospectral_radius)
    if mpemba is not None:
        ev["mpemba_overlap_c1"] = float(mpemba.overlap_c1)
        ev["mpemba_expansion_alpha"] = float(mpemba.expansion_alpha)
        ev["mpemba_is_candidate"] = float(mpemba.is_mpemba_candidate)
        ev["mpemba_trivial_overlap"] = float(mpemba.trivial_overlap)
    return ev


def _pick_a_class(
    ev: dict[str, float],
    *,
    relaxation: RelaxationResult,
) -> tuple[str, str]:
    """Return ``(a_class, f_family)`` based on evidence priorities."""
    # F4 Mpemba check first (high salience for current literature risk). The
    # candidate flag already folds in the non-triviality guard (issue #68), so a
    # symmetry-protected zero overlap (diagonal rho_0 vs a coherence slow mode)
    # no longer reaches A11 -- only a genuine, fine-tuned skip does.
    if ev.get("mpemba_is_candidate", 0.0) > 0.5:
        return "A11", "F4"
    # F5 phantom relaxation (issue #70 A8). The rule must be dimension-coherent
    # AND scale-invariant. ``pseudospectral_radius`` (D13) is the max modulus
    # max{|z| : z in sigma_eps(L)} -- a RATE-dimensioned quantity that scales
    # ~linearly under a uniform Liouvillian rescale L -> cL (all eigenvalues,
    # and the bracketing grid, scale by c). ``gap_to_gns_ratio`` is a pure
    # dimensionless number (Delta / Delta_s), invariant under L -> cL. Comparing
    # a rate directly against a dimensionless ratio (the pre-#70 rule) was
    # incoherent: rescaling L -> cL flipped the A10/F5 verdict even though the
    # physics is unchanged. Normalising the radius by the spectral gap Delta
    # yields the dimensionless pseudospectral reach (radius / Delta) -- how far
    # the eps-pseudospectrum extends relative to the asymptotic decay rate --
    # which is the physically meaningful phantom-relaxation signature (Znidaric
    # 2023) and is scale-invariant to leading order (both radius and gap scale
    # as c). A vanishing gap (no spectral gap) is treated as inf reach: a
    # gapless, strongly non-normal operator is the phantom/critical limit.
    _gap = ev.get("gap", 0.0)
    _psr_reach = (
        ev["pseudospectral_radius"] / _gap if _gap > 0.0 else float("inf")
    )
    if _psr_reach > 2.0 * ev.get("gap_to_gns_ratio", 1.0) and ev["henrici_eta"] > 1.0:
        return "A10", "F5"
    # F1 overlap/eigenvector amplification (Mori-Shirai 2020): non-normal
    # amplification flagged by high Kreiss constant + Petermann factor
    if ev["kreiss"] > 5.0 and ev["petermann_max"] > 5.0:
        return "A3", "F1"
    # F2 skin effect: large trans-amplitude ratio + kappa_trans
    if ev["trans_amplitude_ratio"] > 5.0 and ev["kappa_trans"] > 2.0:
        return "A4", "F2"
    # F3 symmetrised gap correction
    if ev["gap_to_gns_ratio"] > 1.2:
        return "A2", "F3"
    # A1 gap-controlled (LIOU-#69): the OBSERVABLE (linear trace-distance)
    # relaxation is a single exponential whose rate matches the spectral gap
    # (dimension-coherent D17 < 0.05). This takes priority over the M2/M3a/M3b
    # SHAPE branches below -- the relative-entropy curve carries a metric
    # multiplier and can prefer a bi-exponential even for single-mode dynamics.
    # It must NOT precede the F1-F5 gap-failure families (Equalita #79 review):
    # gap_rate_consistency + linear_fit_model come from the initial-state-
    # DEPENDENT trace-distance curve, whereas pseudospectral_radius / henrici /
    # trans_amplitude / kreiss / petermann are operator-INTRINSIC. A strongly
    # non-normal phantom/skin operator with an rho_0 that excites only the slow
    # gap mode yields a clean single-exp at the gap rate; awarding A1/"none"
    # CONFIRMED there would shadow the true A10/F5 (or A3/A4) mechanism. So the
    # gap-failure families are decided first; A1 is reached only when none fire.
    if (
        ev.get("gap_rate_consistency", float("inf")) < 0.05
        and ev.get("d17_linear_single_exp", 0.0) > 0.5
    ):
        return "A1", "none"
    # Oscillatory transient
    if ev["has_complex_pairs"] > 0 and relaxation.aicc_model == "M3b":
        return "A8", "none"
    # Jordan-block / LEP (M3a winner)
    if relaxation.aicc_model == "M3a":
        return "A10", "F5"
    # Biexponential => metastable plateau or operator spreading
    if relaxation.aicc_model == "M2":
        return "A5", "none"
    # Residual gap consistency: the linear-single-exp A1 branch above already
    # claimed the strong (D17 < 0.05) single-mode case, so a plain
    # ``gap_rate_consistency`` threshold is the only distinction left here. The
    # former ``< 0.05 and aicc_model == "M0"`` pre-check was dead: it returned the
    # identical ("A1", "none") that the ``< 0.20`` line below returns, and the A1
    # confidence keys on ``gap_rate_consistency`` alone (not the aicc model), so
    # it changed neither the label nor the score.
    if ev["gap_rate_consistency"] < 0.20:
        return "A1", "none"
    return "A12", "none"


def _pick_verdict_tier(
    a_class: str,
    relaxation: RelaxationResult,
    confidence: float,
) -> tuple[str, str]:
    """Return ``(verdict, tier)``."""
    if not np.isfinite(relaxation.beta_D):
        return VERDICT_UNDEFINED, TIER_EXPLORATION
    if a_class == "A12":
        return VERDICT_NOT_EXCLUDED, TIER_EXPLORATION
    if confidence >= 0.85:
        return VERDICT_CONFIRMED, TIER_PUBLICATION
    if confidence >= 0.60:
        return VERDICT_CANDIDATE, TIER_CONFIRMATION
    # issue #70 A5: no genuine EXCLUDED (active-rejection) branch here. A
    # single-pass, maximum-evidence classifier reports the BEST-fit A-class with
    # its support -- it never reports a class it is simultaneously ruling out, so
    # a per-class "EXCLUDED" verdict is not expressible in this architecture
    # (active exclusion needs per-hypothesis scoring, deferred; see PR body). The
    # old ``confidence < 0.30 -> EXCLUDED`` branch was also SEMANTICALLY wrong:
    # low confidence in the best-fit class is epistemic uncertainty ("unresolved"
    # = NOT_EXCLUDED), not positive counter-evidence ("ruled out"). It was
    # additionally unreachable (the only sub-0.30 confidence, A12 = 0.20,
    # short-circuits to NOT_EXCLUDED above). Low confidence now correctly falls
    # through to NOT_EXCLUDED.
    return VERDICT_NOT_EXCLUDED, TIER_EXPLORATION


def _confidence(ev: dict[str, float], a_class: str) -> float:
    """Heuristic confidence score in ``[0, 1]``.

    Combines multiple weak signals; not a posterior probability.
    """
    score = 0.5
    if a_class == "A1" and ev.get("gap_rate_consistency", 1.0) < 0.05:
        score = 0.95
    elif (a_class == "A2" and ev.get("gap_to_gns_ratio", 1.0) > 1.5) or (a_class == "A3" and ev.get("kreiss", 0.0) > 10.0) or (a_class == "A4" and ev.get("trans_amplitude_ratio", 0.0) > 10.0):
        score = 0.85
    elif a_class == "A10":
        score = 0.70
    elif a_class == "A11" and ev.get("mpemba_overlap_c1", 1.0) < 1.0e-5:
        # A single initial state skipping the slowest mode is a *candidate*, not
        # a confirmed anomalous Mpemba effect: confirmation needs a reference
        # family (e.g. thermal states at different temperatures), which the
        # single-state pipeline does not provide. Capped below the
        # PUBLICATION_GRADE threshold so a lone overlap cannot self-certify
        # (issue #68); genuine cases still surface as A11 CANDIDATE.
        score = 0.70
    elif a_class == "A12":
        score = 0.20
    return float(min(max(score, 0.0), 1.0))


def classify_mechanism(
    spectral: SpectralResult,
    nonnorm: NonNormalityResult,
    relaxation: RelaxationResult,
    resolvent: ResolventResult,
    transient: TransientResult,
    lep: LepResult,
    mpemba: MpembaResult | None = None,
) -> ClassificationResult:
    """Classify the dominant relaxation mechanism into A1..A12 with F1..F5 tag."""
    ev = _gather_evidence(
        spectral, nonnorm, relaxation, resolvent, transient, lep, mpemba
    )
    a_class, f_family = _pick_a_class(ev, relaxation=relaxation)
    conf = _confidence(ev, a_class)
    verdict, tier = _pick_verdict_tier(a_class, relaxation, conf)
    return ClassificationResult(
        a_class=a_class,
        f_family=f_family,
        verdict=verdict,  # type: ignore[arg-type]
        tier=tier,        # type: ignore[arg-type]
        confidence=conf,
        evidence=ev,
        taxonomy_version=TAXONOMY_VERSION,
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
    )
