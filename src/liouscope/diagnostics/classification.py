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

Verdict in {CONFIRMED, EXCLUDED, CANDIDATE, NOT_EXCLUDED, UNDEFINED}.
Tier in {PUBLICATION_GRADE, CONFIRMATION, EXPLORATION}.

Anchor L: ``taxonomy_version`` is stamped on every ClassificationResult.
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
    VERDICT_EXCLUDED,
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


def _gather_evidence(
    spectral: SpectralResult,
    nonnorm: NonNormalityResult,
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
    # F5 phantom relaxation
    if ev["pseudospectral_radius"] > 2.0 * ev.get("gap_to_gns_ratio", 1.0) and ev["henrici_eta"] > 1.0:
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
    # Oscillatory transient
    if ev["has_complex_pairs"] > 0 and relaxation.aicc_model == "M3b":
        return "A8", "none"
    # Jordan-block / LEP (M3a winner)
    if relaxation.aicc_model == "M3a":
        return "A10", "F5"
    # Biexponential => metastable plateau or operator spreading
    if relaxation.aicc_model == "M2":
        return "A5", "none"
    # Strong gap consistency (M0 winner, beta_D close to Delta)
    if ev["gap_rate_consistency"] < 0.05 and relaxation.aicc_model == "M0":
        return "A1", "F1"
    if ev["gap_rate_consistency"] < 0.20:
        return "A1", "F1"
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
    if confidence < 0.30:
        return VERDICT_EXCLUDED, TIER_EXPLORATION
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
    ev = _gather_evidence(spectral, nonnorm, resolvent, transient, lep, mpemba)
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
