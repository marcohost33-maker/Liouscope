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
CONFIRMATION, EXPLORATION}. UNDEFINED doubles as the INSUFFICIENT-EVIDENCE floor:
it is emitted when ``beta_D`` is non-finite AND (issue #78 / decision E0706-13)
when a SINGLE-STATE run picks A11 on a MAXIMALLY MIXED steady state with no
reference-ensemble confirmation -- see ``SINGLE_STATE_MAXMIX_FLOOR_CLASS`` /
``_apply_single_state_maxmix_floor`` for the conditional ensemble-override
contract.

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

Hypothesis-wise evidence matrix (issue #102): the priority chain is defined
declaratively in ``_ladder_spec`` (rungs of atomic ``_Condition`` predicates).
``_hypothesis_ladder`` (decision + shadow report) and
``hypothesis_evidence_matrix`` (per-hypothesis supporting/counter/missing
evidence with explicit claim floors over the FULL A1-A12 taxonomy, reserved
classes included) are both derived from that one spec, so the audit surface
cannot drift from the decision. The matrix is report-only: no decision function
consumes it. ``support_score`` is the honestly-named ordinal twin of the legacy
``confidence`` field (issue #102 option 1); both carry the same deterministic,
non-probabilistic value until a calibrated replacement passes the preregistered
validation design.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from .._consts import (
    DIAGNOSTIC_SCHEMA_VERSION,
    EPS_MAXMIX,
    GNS_CERTIFIED_RTOL,
    REACHABLE_A_CLASSES,
    RESERVED_A_CLASSES,
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
        # Issue #101 slice A: scale-relative candidate diagnostics for the F5
        # evidence path. Deliberately ADVISORY: #101 explicitly forbids any
        # classifier/verdict change before the preregistered calibration study
        # (slice C) and independent physics review. Promoting any of these to a
        # decision driver is a class-influencing change and must ship in a
        # dedicated PR with FP/TP calibration + anchor coverage.
        "henrici_relative",             # D8b eta_N / ||L||_F (dimensionless)
        "kreiss_scaled",                # D10b scale-relative grid lower bound
        "pseudospectral_radius_rel",    # D13 radius / rate_scale (eps_rel grid)
        "pseudospectral_abscissa_rel",  # gap-directed intrusion / rate_scale
    }
)
# NOTE (#80): ``gap_to_kms_ratio`` was advisory when introduced in #89 (#88
# option 2 deferred). It is now class-INFLUENCING: it feeds the positive
# ``sym_gap_corroborated`` certificate that gates A1 PUBLICATION_GRADE (see
# ``_gather_evidence`` / ``_confidence``). Using it as an F3 *veto* remains
# deferred to its own false-positive study.

# issue #78 / decision E0706-13: the single-state maximally-mixed A11 floor.
#
# PR #77 stopped A11 self-certifying PUBLICATION_GRADE/CONFIRMED. Residual: when
# ``rho_ss`` is MAXIMALLY MIXED (``rho_ss = I/d``) it collapses to a single
# degenerate eigenprojector, so the issue-#68 triviality guard
# (``is_trivial_overlap``, which needs a NON-TRIVIAL eigenprojector decomposition
# of ``rho_ss``) CANNOT fire -- there is no sector structure in the FIXPOINT for
# the guard to test against. (This is a statement about the eigenprojectors of
# ``rho_ss``, NOT a claim that the Liouvillian lacks a protecting / decoherence-
# free sector: protecting sectors are properties of the Liouvillian's structure,
# not of how far ``rho_ss`` sits from ``I/d`` -- a unital / pure-dephasing
# Liouvillian can carry invariant structure even at ``rho_ss = I/d``.) The README
# quickstart ``|+>`` case (dephasing, ``rho_ss = I/2``) therefore reached A11/F4
# CANDIDATE (confidence 0.70), which OVER-claims: a single initial state whose
# steady state is maximally mixed supplies insufficient comparative/dynamical
# evidence for a Mpemba claim. Mpemba depends on the Liouvillian spectrum, the
# mode overlaps and the initial ensemble -- not on the fixpoint alone.
#
# DECIDED behaviour (physics FINAL, cross-family-corrected -- do NOT re-litigate):
#   * Single-state maximally-mixed A11 -> VERDICT_UNDEFINED / EXPLORATION
#     (INSUFFICIENT EVIDENCE). NOT CANDIDATE (over-claim); NOT hard-EXCLUDED (a
#     unital / doubly-stochastic Markov process with multiple relaxation rates CAN
#     exhibit Mpemba -- hard exclusion was falsified cross-family).
#   * The exclusion is CONDITIONAL and applies ONLY to the single-state path. If
#     reference-ensemble Mpemba confirmation across a thermal family is present
#     (evidence key ``ensemble_confirmation`` truthy), the downgrade is SUPPRESSED
#     and A11 may still be confirmed/candidate. The current pipeline has no
#     ensemble-evidence input, so ``ensemble_confirmation`` is an explicit,
#     tested hook/contract: a future ensemble path sets it truthy and plugs in
#     WITHOUT re-touching this branch.
#
# ``VERDICT_UNDEFINED`` (enum literal "UNDEFINED") is REUSED as the existing
# insufficient-evidence floor (it is the same verdict emitted when ``beta_D`` is
# non-finite); no new verdict is invented. The (class, family, confidence) are
# preserved -- A11/F4 remains the best-fit mechanism HYPOTHESIS; only the VERDICT
# (what we are willing to claim) drops to UNDEFINED. Pinned by
# ``tests/test_classifier_semantics_debt.py`` (contract) and
# ``tests/test_classification.py`` (behaviour).
SINGLE_STATE_MAXMIX_FLOOR_CLASS: str = "A11"
ENSEMBLE_OVERRIDE_EVIDENCE_KEY: str = "ensemble_confirmation"

# Issues #112/#113 (twelfth-round review): evidence key recording whether the
# spectral layer's structural zero-mode certificate resolved. Written by
# ``classify_mechanism`` from ``SpectralResult.zero_mode_certificate`` when one
# is present; absent for synthetic results that carry no certificate, so every
# pre-existing test and serialised result is untouched. Class-INFLUENCING by
# design, in the fail-closed direction only: an unresolved certificate means
# D1/D3/D4 -- and every evidence ratio built on them -- came from a spectrum
# the solver demonstrably could not resolve, so no mechanism claim may stand
# on it. The verdict floors to UNDEFINED (insufficient evidence), exactly like
# the non-finite-beta_D floor; class/family remain as the best-fit HYPOTHESIS.
SPECTRAL_RESOLVED_EVIDENCE_KEY: str = "spectral_resolved"


def _apply_spectral_certificate_floor(
    verdict: str,
    tier: str,
    *,
    spectral_resolved: bool,
) -> tuple[str, str]:
    """Insufficient-evidence floor for an unresolved spectral certificate."""
    if not spectral_resolved:
        return VERDICT_UNDEFINED, TIER_EXPLORATION
    return verdict, tier


def _is_maximally_mixed(rho_steady_state: np.ndarray, *, atol: float = EPS_MAXMIX) -> bool:
    """Is ``rho_ss`` the maximally mixed state ``I/d`` (to ``atol``)?

    Returns ``True`` only for a genuine ``d x d`` density matrix with ``d >= 2``
    whose every entry matches ``I/d`` within ``atol``. ``d < 2`` (including the
    synthetic ``1x1`` placeholders used in the classifier unit tests) is never
    flagged: maximal mixing -- and hence the single-state insufficient-evidence
    floor it gates -- is only meaningful for ``d >= 2``.
    """
    rho = np.asarray(rho_steady_state, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        # Fail closed: an uninterpretable steady-state shape (e.g. a still-
        # vectorised (d^2,) array) must not silently DISABLE the A11
        # insufficient-evidence floor by reading as "not maximally mixed".
        raise ValueError(
            "steady_state must be a square 2-D density matrix to evaluate the "
            f"maximally-mixed floor, got shape {rho.shape}"
        )
    if rho.shape[0] < 2:
        return False
    d = rho.shape[0]
    # Pure ABSOLUTE tolerance (rtol=0): the default numpy rtol=1e-5 would swamp a
    # 1e-9 atol and wrongly flag near-mixed states, defeating the tight-floor
    # intent (only the exact I/d limit must trigger).
    return bool(np.allclose(rho, np.eye(d, dtype=complex) / d, rtol=0.0, atol=atol))


def _apply_single_state_maxmix_floor(
    a_class: str,
    verdict: str,
    tier: str,
    *,
    maximally_mixed: bool,
    ensemble_confirmation: bool,
) -> tuple[str, str]:
    """Conditional insufficient-evidence floor for single-state maximally-mixed A11.

    See ``SINGLE_STATE_MAXMIX_FLOOR_CLASS`` above for the full rationale. Downgrades
    ``(verdict, tier)`` to ``(VERDICT_UNDEFINED, TIER_EXPLORATION)`` iff the picked
    class is A11, the steady state is maximally mixed, AND no reference-ensemble
    Mpemba confirmation is present. Otherwise the verdict/tier pass through
    unchanged -- in particular the ensemble override cleanly suppresses the floor.
    """
    if (
        a_class == SINGLE_STATE_MAXMIX_FLOOR_CLASS
        and maximally_mixed
        and not ensemble_confirmation
    ):
        return VERDICT_UNDEFINED, TIER_EXPLORATION
    return verdict, tier


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
    # Issue #88: ``gns_gap`` deliberately floors to ~0 (O(machine-eps) relative
    # to the operator scale) whenever the GNS symmetrisation cannot certify a
    # contraction -- e.g. any non-detailed-balance system whose steady state
    # carries coherences. That sentinel makes ``gap_to_gns_ratio`` explode
    # (~1e10..inf) WITHOUT any measured symmetrised-gap reduction. The flag
    # below distinguishes a MEASURED Delta_GNS (certified: resolved above the
    # numerical noise floor relative to Delta) from the uncertified sentinel;
    # decision branches keying on ``gap_to_gns_ratio`` must require it.
    # Finiteness is part of the certificate: a non-finite gns_gap would pass
    # the >= floor test (inf >= anything) and, downstream, drive
    # gap_to_gns_ratio to 0.0 -- silently *granting* the corroboration
    # certificate below. Non-finite input must fail closed, not upgrade.
    ev["gns_certified"] = (
        1.0
        if np.isfinite(spectral.gap)
        and np.isfinite(spectral.gns_gap)
        and spectral.gap > 0.0
        and spectral.gns_gap >= GNS_CERTIFIED_RTOL * spectral.gap
        else 0.0
    )
    # KMS counterpart of the gap ratio (#88/#80): a genuine Mori-Shirai
    # symmetrised-gap reduction usually shows up in the KMS gap as well, while
    # the repro class of #88 has Delta_KMS == Delta exactly. Since #80 this is
    # class-influencing via ``sym_gap_corroborated`` below (it is NOT an F3
    # veto -- that remains deferred, see the ADVISORY_EVIDENCE_KEYS note).
    ev["gap_to_kms_ratio"] = (
        float(spectral.gap / spectral.kms_gap) if spectral.kms_gap > 0 else float("inf")
    )
    # Issue #80: POSITIVE certificate that the spectral gap really controls
    # contraction, used to gate A1 CONFIRMED/PUBLICATION_GRADE. Rationale: the
    # A1 early branch fires when no F1-F5 family fired, i.e. on the *absence*
    # of failure triggers -- but threshold exhaustiveness cannot be proven
    # (a hypothetical weakly-non-normal gap failure below ALL thresholds with
    # a single-exp-at-gap trajectory would slip through). A *measured*
    # symmetrised gap equal to Delta (no F3-grade reduction, <= 1.2) is
    # operator-intrinsic positive evidence: it certifies exponential
    # contraction at the gap rate in the GNS (certified only, #88) or KMS
    # geometry. A floored/uncertified GNS and a reduced or floored KMS both
    # fail this, fail-closed.
    # Both legs additionally require a FINITE symmetrised gap: with e.g.
    # ``kms_gap = inf`` the ratio collapses to 0.0 <= 1.2 and the certificate
    # would be granted by garbage -- the exact inversion of fail-closed. A
    # non-finite gap is not a measurement and certifies nothing (#80).
    ev["sym_gap_corroborated"] = (
        1.0
        if np.isfinite(spectral.gap)
        and spectral.gap > 0.0
        and (
            (ev["gns_certified"] > 0.5 and ev["gap_to_gns_ratio"] <= 1.2)
            or (np.isfinite(spectral.kms_gap) and ev["gap_to_kms_ratio"] <= 1.2)
        )
        else 0.0
    )
    # ``None`` is the unavailable marker for this bool (withheld together
    # with D1/D3/D4 when the certificate is applicable but uncertified). NaN
    # is the evidence dict's unavailable sentinel, and ``_strip_unavailable``
    # removes it so the A8 rung does not hold on an unanswered question.
    ev["has_complex_pairs"] = (
        float("nan") if spectral.has_complex_pairs is None
        else float(spectral.has_complex_pairs)
    )
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
    # Issue #101 slice A: scale-relative variants, surfaced for audit /
    # serialisation only (see ADVISORY_EVIDENCE_KEYS -- no decision function
    # reads them; the metamorphic non-influence test pins that contract).
    ev["henrici_relative"] = float(nonnorm.henrici_relative)
    ev["kreiss_scaled"] = float(nonnorm.kreiss_scaled)
    ev["pseudospectral_radius_rel"] = float(resolvent.pseudospectral_radius_rel)
    ev["pseudospectral_abscissa_rel"] = float(resolvent.pseudospectral_abscissa_rel)
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
    """Return ``(a_class, f_family)`` — the winner of :func:`_hypothesis_ladder`."""
    for rung in _hypothesis_ladder(ev, relaxation=relaxation):
        if rung[3]:
            return rung[1], rung[2]
    return "A12", "none"


@dataclass(frozen=True)
class _Condition:
    """One atomic requirement of a decision rung (issue #102).

    ``keys`` names the REQUIRED evidence-dict entries: the predicate indexes
    them directly, so their absence makes the rung UNEVALUABLE in the matrix
    (e.g. the Mpemba layer was not run) -- unless an evaluated sibling
    condition is already false, which refutes the conjunction conclusively
    (round-13 review). ``optional_keys`` names entries the
    predicate reads through ``ev.get(key, default)`` — it has documented
    semantics without them, so they are reported in ``values`` when present but
    never trigger UNEVALUABLE.

    Keeping the two apart is what preserves the matrix/ladder equivalence: if a
    defaulted key were declared required, the matrix would report UNEVALUABLE
    for evidence on which the ladder happily fires (the F5 reach leg reads
    ``ev.get("gap_to_gns_ratio", 1.0)`` and has documented semantics without
    it), and the two would disagree exactly on the partially collected
    evidence the matrix exists to describe.

    The converse error is the round-22 finding: a key whose DEFAULT is
    positive evidence must not be optional. ``gap`` was defaulted to ``0.0``
    in the F5 reach leg, i.e. to the gapless limit, so a missing D1 read as
    support for the phantom-relaxation hypothesis rather than as the absence
    of the measurement. A default is only admissible when the predicate has
    documented semantics WITHOUT the value -- not when it invents one.

    ``relaxation_fields`` names any :class:`RelaxationResult` attributes read,
    for the same audit purpose. ``fn`` MUST reproduce the decision predicate
    exactly — it is the single source both the ladder and the matrix evaluate.
    """

    description: str
    keys: tuple[str, ...]
    fn: Callable[[dict[str, float], RelaxationResult], bool]
    relaxation_fields: tuple[str, ...] = ()
    optional_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Rung:
    """One priority rung: fires iff ALL of its conditions hold."""

    rule_id: str
    a_class: str
    f_family: str
    conditions: tuple[_Condition, ...] = field(default_factory=tuple)


def _f5_reach(ev: dict[str, float], relaxation: RelaxationResult) -> bool:
    # F5 reach leg (issue #70 A8). ``pseudospectral_radius`` (D13) is the max
    # modulus max{|z| : z in sigma_eps(L)} -- a RATE-dimensioned quantity that
    # scales ~linearly under a uniform Liouvillian rescale L -> cL (all
    # eigenvalues, and the bracketing grid, scale by c). ``gap_to_gns_ratio``
    # is a pure dimensionless number (Delta / Delta_s), invariant under
    # L -> cL. Comparing a rate directly against a dimensionless ratio (the
    # pre-#70 rule) was incoherent: rescaling L -> cL flipped the A10/F5
    # verdict even though the physics is unchanged. Normalising the radius by
    # the spectral gap Delta yields the dimensionless pseudospectral reach
    # (radius / Delta) -- how far the eps-pseudospectrum extends relative to
    # the asymptotic decay rate -- which is the physically meaningful
    # phantom-relaxation signature (Znidaric 2023) and is scale-invariant to
    # leading order (both radius and gap scale as c). A vanishing gap (no
    # spectral gap) is treated as inf reach: a gapless, strongly non-normal
    # operator is the phantom/critical limit.
    # ROUND-22 REVIEW (PR #121). ``ev.get("gap", 0.0)`` read a MISSING
    # measurement as a measured zero. When the certificate is unresolved the
    # spectral layer withholds D1 as NaN, ``_strip_unavailable`` removes the
    # key, and the default then supplied the gapless limit -- positive
    # evidence for the very leg that could not be computed. With
    # ``henrici_eta > 1`` the rung fired, the classifier returned A10/F5 and
    # the matrix reported the F5 hypothesis SUPPORTED. The certificate floor
    # downstream caps the VERDICT at UNDEFINED, but it does not withdraw the
    # class, the family or the matrix status, so a fabricated mechanism label
    # survived the floor that was supposed to contain it.
    #
    # ``gap`` is therefore a REQUIRED key of this condition (see the rung
    # below): absent, the rung is unevaluable and cannot fire. A gap that was
    # MEASURED as 0.0 still means gapless and still counts as reach evidence --
    # the distinction is between "no gap" and "no measurement", which is
    # exactly the distinction the default erased.
    _gap = ev["gap"]
    if _gap > 0.0:
        return bool(
            ev["pseudospectral_radius"] / _gap > 2.0 * ev.get("gap_to_gns_ratio", 1.0)
        )
    # Gapless limit: infinite reach dominates ANY ratio. The former
    # ``inf > 2*ratio`` comparison silently failed when the GNS gap was
    # ALSO floored (ratio = inf, and ``inf > inf`` is False) -- i.e. in the
    # realistic gapless case, since ``gns_certified`` requires gap > 0 and
    # the sentinel floors there. The documented contract (gapless +
    # strongly non-normal = phantom/critical limit) now holds
    # unconditionally on the reach side; henrici still gates in the second
    # condition. NOTE (#101 addendum): treating a vanishing gap as positive
    # reach evidence is a known blind spot -- the gapless-abstention rule is
    # gated on the slice-C calibration study and NOT changed here.
    return True


def _ladder_spec() -> tuple[_Rung, ...]:
    """The priority ladder as declarative rungs (issue #102).

    Single source of truth for the decision: :func:`_hypothesis_ladder`
    (winner + shadow report) and :func:`hypothesis_evidence_matrix`
    (per-hypothesis supporting/counter/missing evidence) are BOTH derived from
    this spec, so there is no second copy of the conditions to drift.

    Order is the decision order; do not reorder without a physics rationale.
    The per-rung physics rationale (F4 salience first, operator-intrinsic
    F-families before the state-dependent A1 branches, ...) is documented on
    the conditions and in the docstrings below; behaviour is pinned by
    ``tests/test_branch_shadowing.py`` and the anchor suite.
    """
    # F4 Mpemba check first (high salience for current literature risk). The
    # candidate flag already folds in the non-triviality guard (issue #68), so a
    # symmetry-protected zero overlap (diagonal rho_0 vs a coherence slow mode)
    # no longer reaches A11 -- only a genuine, fine-tuned skip does. NOTE: A11 is
    # still the best-fit mechanism label here; the SINGLE-STATE maximally-mixed
    # case (rho_ss = I/d, where the issue-#68 guard structurally cannot fire) is
    # handled downstream at the VERDICT level by _apply_single_state_maxmix_floor
    # (issue #78) -- it drops the verdict to UNDEFINED (insufficient evidence)
    # rather than suppressing the A11 label.
    f4 = _Rung(
        rule_id="F4_MPEMBA",
        a_class="A11",
        f_family="F4",
        conditions=(
            _Condition(
                description="mpemba_is_candidate > 0.5 (D19 slow-mode skip, "
                "non-triviality guard folded in)",
                keys=(),
                optional_keys=("mpemba_is_candidate",),
                fn=lambda ev, rel: ev.get("mpemba_is_candidate", 0.0) > 0.5,
            ),
        ),
    )
    # F5 phantom relaxation: dimensionless reach leg + (legacy, rate-
    # dimensioned) Henrici gate. The gate's missing unit invariance is the
    # documented #101 limitation; it is reported, not silently fixed here.
    f5 = _Rung(
        rule_id="F5_PSEUDOSPECTRAL",
        a_class="A10",
        f_family="F5",
        conditions=(
            _Condition(
                description="pseudospectral reach: radius/gap > 2 * gap_to_gns_ratio "
                "(gapless => infinite reach; see #101 gapless blind spot)",
                keys=("pseudospectral_radius", "gap"),
                optional_keys=("gap_to_gns_ratio",),
                fn=_f5_reach,
            ),
            _Condition(
                description="henrici_eta > 1.0 (D8; rate-dimensioned legacy gate, "
                "not unit-invariant -- #101 known limitation)",
                keys=("henrici_eta",),
                fn=lambda ev, rel: bool(ev["henrici_eta"] > 1.0),
            ),
        ),
    )
    # F1 overlap/eigenvector amplification (Mori-Shirai 2020): non-normal
    # amplification flagged by high Kreiss constant + Petermann factor
    f1 = _Rung(
        rule_id="F1_OVERLAP_AMPLIFICATION",
        a_class="A3",
        f_family="F1",
        conditions=(
            _Condition(
                description="kreiss > 5.0 (D10 grid lower bound)",
                keys=("kreiss",),
                fn=lambda ev, rel: bool(ev["kreiss"] > 5.0),
            ),
            _Condition(
                description="petermann_max > 5.0 (D9)",
                keys=("petermann_max",),
                fn=lambda ev, rel: bool(ev["petermann_max"] > 5.0),
            ),
        ),
    )
    # F2 skin effect: large trans-amplitude ratio + kappa_trans
    f2 = _Rung(
        rule_id="F2_SKIN",
        a_class="A4",
        f_family="F2",
        conditions=(
            _Condition(
                description="trans_amplitude_ratio > 5.0 (D14, unstructured HS norm)",
                keys=("trans_amplitude_ratio",),
                fn=lambda ev, rel: bool(ev["trans_amplitude_ratio"] > 5.0),
            ),
            _Condition(
                description="kappa_trans > 2.0 (D15)",
                keys=("kappa_trans",),
                fn=lambda ev, rel: bool(ev["kappa_trans"] > 2.0),
            ),
        ),
    )
    # F3 symmetrised gap correction (issue #88). A2/F3 semantics are a
    # *measured* Mori-Shirai symmetrised-gap reduction (Delta_GNS genuinely
    # below Delta), so the branch must key on a CERTIFIED Delta_GNS. When
    # ``gns_gap`` is the conservative floor sentinel (~0: the GNS
    # symmetrisation certifies no contraction at all -- the documented 2026-07
    # audit-A1 behaviour for non-detailed-balance steady states with
    # coherences), the exploded ratio is an artefact of the sentinel, not
    # positive mechanism evidence: firing F3 off it labelled a textbook
    # Rabi-driven amplitude-damped qubit (Delta_KMS == Delta, no real
    # reduction) as A2/F3 CONFIRMED / PUBLICATION_GRADE. Uncertified cases
    # fall through to the state-dependent branches below (A1/A5/A8/...),
    # which is the honest floor: "GNS uncertified" is absence of evidence.
    f3 = _Rung(
        rule_id="F3_SYMMETRISED_GAP",
        a_class="A2",
        f_family="F3",
        conditions=(
            _Condition(
                description="gap_to_gns_ratio > 1.2 (D1/D2 measured reduction)",
                keys=("gap_to_gns_ratio",),
                fn=lambda ev, rel: bool(ev["gap_to_gns_ratio"] > 1.2),
            ),
            _Condition(
                description="gns_certified > 0.5 (Delta_GNS measured, not the "
                "floor sentinel -- #88)",
                keys=(),
                optional_keys=("gns_certified",),
                fn=lambda ev, rel: ev.get("gns_certified", 0.0) > 0.5,
            ),
        ),
    )
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
    a1_strong = _Rung(
        rule_id="A1_LINEAR_SINGLE_EXP",
        a_class="A1",
        f_family="none",
        conditions=(
            _Condition(
                description="gap_rate_consistency < 0.05 (D17, dimension-coherent)",
                keys=(),
                optional_keys=("gap_rate_consistency",),
                fn=lambda ev, rel: ev.get("gap_rate_consistency", float("inf")) < 0.05,
            ),
            _Condition(
                description="linear trace-distance relaxation is single-exponential "
                "(M0/M1 winner)",
                keys=(),
                optional_keys=("d17_linear_single_exp",),
                fn=lambda ev, rel: ev.get("d17_linear_single_exp", 0.0) > 0.5,
            ),
        ),
    )
    # Oscillatory transient
    a8 = _Rung(
        rule_id="A8_OSCILLATORY",
        a_class="A8",
        f_family="none",
        conditions=(
            _Condition(
                description="has_complex_pairs (D3 oscillating modes present)",
                keys=("has_complex_pairs",),
                fn=lambda ev, rel: bool(ev["has_complex_pairs"] > 0),
            ),
            _Condition(
                description="AICc winner is M3b (damped-oscillation model)",
                keys=(),
                fn=lambda ev, rel: rel.aicc_model == "M3b",
                relaxation_fields=("aicc_model",),
            ),
        ),
    )
    # Jordan-block / LEP (M3a winner)
    a10_m3a = _Rung(
        rule_id="A10_JORDAN_M3A",
        a_class="A10",
        f_family="F5",
        conditions=(
            _Condition(
                description="AICc winner is M3a (polynomial-times-exponential, "
                "Jordan/LEP signature)",
                keys=(),
                fn=lambda ev, rel: rel.aicc_model == "M3a",
                relaxation_fields=("aicc_model",),
            ),
        ),
    )
    # Biexponential => metastable plateau or operator spreading
    a5 = _Rung(
        rule_id="A5_BIEXPONENTIAL",
        a_class="A5",
        f_family="none",
        conditions=(
            _Condition(
                description="AICc winner is M2 (biexponential / metastable plateau)",
                keys=(),
                fn=lambda ev, rel: rel.aicc_model == "M2",
                relaxation_fields=("aicc_model",),
            ),
        ),
    )
    # Residual gap consistency: the linear-single-exp A1 branch above already
    # claimed the strong (D17 < 0.05) single-mode case, so a plain
    # ``gap_rate_consistency`` threshold is the only distinction left here. The
    # former ``< 0.05 and aicc_model == "M0"`` pre-check was dead: it returned the
    # identical ("A1", "none") that the ``< 0.20`` line below returns, and the A1
    # confidence keys on ``gap_rate_consistency`` alone (not the aicc model), so
    # it changed neither the label nor the score.
    a1_residual = _Rung(
        rule_id="A1_GAP_CONSISTENT",
        a_class="A1",
        f_family="none",
        conditions=(
            _Condition(
                description="gap_rate_consistency < 0.20 (D17 residual consistency)",
                keys=("gap_rate_consistency",),
                fn=lambda ev, rel: bool(ev["gap_rate_consistency"] < 0.20),
            ),
        ),
    )
    return (f4, f5, f1, f2, f3, a1_strong, a8, a10_m3a, a5, a1_residual)


def _reachable_coverage_check(
    spec_classes: frozenset[str] | None = None,
) -> frozenset[str]:
    """Import-time coverage gate over ``REACHABLE_A_CLASSES`` (issue #102).

    The rungs of :func:`_ladder_spec` plus the A12 fallback must emit exactly
    the reachable taxonomy (``A_CLASSES`` minus ``RESERVED_A_CLASSES``).
    Adding a rung for a reserved class — or removing the last rung of a
    reachable one — without updating the reserved contract in lock-step is a
    reachability drift; failing here at import keeps that drift out of
    consumers and out of coverage-denominator statements. The AST-level test
    in ``tests/test_classifier_semantics_debt.py`` pins the same contract
    statically; this guard covers installed environments where the test
    suite never runs.

    ``spec_classes`` exists for tests only; production callers use the
    default (the real spec).
    """
    classes = (
        frozenset(rung.a_class for rung in _ladder_spec()) | {"A12"}
        if spec_classes is None
        else spec_classes
    )
    if classes != frozenset(REACHABLE_A_CLASSES):
        raise RuntimeError(
            "classifier ladder drifted from the reachable taxonomy: spec emits "
            f"{sorted(classes)} but REACHABLE_A_CLASSES is "
            f"{sorted(REACHABLE_A_CLASSES)}; update RESERVED_A_CLASSES / the "
            "ladder in lock-step (issue #102 reachability gate)"
        )
    return classes


_reachable_coverage_check()


def _hypothesis_ladder(
    ev: dict[str, float],
    *,
    relaxation: RelaxationResult,
) -> list[tuple[str, str, str, bool]]:
    """The full priority ladder as ``(rule_id, a_class, f_family, fires)``.

    Issue #102: the classifier resolves by PRIORITY, so a system that shows
    several mechanisms at once reports only the first. Evaluating every rung
    here — and deriving both the winner (:func:`_pick_a_class`) and the
    shadow report (:func:`triggered_hypotheses`) from this one list — makes the
    suppressed hypotheses visible without changing any verdict, and without a
    second copy of the conditions that could drift from the decision itself.

    The rungs and their conditions come from :func:`_ladder_spec`; a rung
    fires iff ALL of its conditions hold (short-circuit evaluation, matching
    the historical inline ``and`` chains exactly).

    NaN-valued evidence is stripped before evaluation
    (:func:`_strip_unavailable`): NaN is the library's unavailable-value
    sentinel, so a NaN-encoded missing measurement must behave exactly like an
    absent key — for defaulted (optional) keys the documented default applies,
    for indexed (required) keys the rung cannot fire. Without the strip, the
    same missing measurement changed the outcome depending on its ENCODING:
    ``gap_to_gns_ratio`` absent let the F5 reach leg apply its default ``1.0``,
    while ``gap_to_gns_ratio = NaN`` made every comparison False.

    A condition whose required keys are unavailable is treated as NOT holding
    rather than invoked (eleventh-round review): letting the predicate index a
    stripped key aborted REPORT GENERATION with a ``KeyError`` whenever one
    condition passed and its sibling's evidence was NaN. Not firing is the
    fail-closed direction — no mechanism is claimed — and the evidence matrix
    carries the audit trail (the rung reports UNEVALUABLE with the missing
    keys, and the A12 fallback inherits the uncertainty).
    """
    ev = _strip_unavailable(ev)
    return [
        (
            rung.rule_id,
            rung.a_class,
            rung.f_family,
            all(
                all(k in ev for k in cond.keys) and cond.fn(ev, relaxation)
                for cond in rung.conditions
            ),
        )
        for rung in _ladder_spec()
    ]


def triggered_hypotheses(
    ev: dict[str, float],
    *,
    relaxation: RelaxationResult,
) -> tuple[dict[str, object], ...]:
    """Every hypothesis that fires, in priority order — issue #102.

    The classifier returns ONE dominant class. When a system simultaneously
    shows, say, Mpemba overlap and pseudospectral phantom evidence, the
    priority chain reports only the first and the second becomes invisible.
    This function reports all of them, each marked ``shadowed`` when priority
    suppressed a genuinely DIFFERENT mechanism.

    ``shadowed`` keys on the (class, family) pair, not on ladder position.
    Several rungs can reach the SAME conclusion -- ``gap_rate_consistency <
    0.05`` (with D17) and the residual ``< 0.20`` rung both yield ``A1/none``,
    and the strong one implies the broad one -- so position alone would mark
    A1 as shadowed by A1 and report a mechanism conflict where none exists.
    Every firing rung is still listed: two rules supporting one conclusion is
    corroboration worth seeing, it is simply not shadowing.

    The tuple is RULE-level, so one suppressed mechanism can occupy several
    entries -- ``A1`` through both its rungs, ``A10/F5`` through the
    pseudospectral and the M3a rung. Counting entries therefore overcounts
    conflicts; count distinct ``(a_class, f_family)`` pairs among the shadowed
    ones instead. The tuple is deliberately NOT deduplicated: dropping a rung
    would hide a rule that genuinely fired, and which rules fired is the audit
    handle this report exists for.

    It is a REPORT, not a decision: the verdict, class, family and confidence
    are untouched. Consuming it as evidence would be a classifier change and
    belongs behind the calibration study of issue #102.
    """
    fired = [r for r in _hypothesis_ladder(ev, relaxation=relaxation) if r[3]]
    winner = (fired[0][1], fired[0][2]) if fired else None
    return tuple(
        {
            "rule_id": rid,
            "a_class": a_class,
            "f_family": f_family,
            "shadowed": (a_class, f_family) != winner,
        }
        for rid, a_class, f_family, _fires in fired
    )


# Issue #102: hypothesis-status vocabulary for the evidence matrix. SUPPORTED
# is by construction equivalent to the rung firing in _hypothesis_ladder (the
# matrix evaluates the SAME _ladder_spec conditions); the remaining statuses
# grade the reasons a hypothesis did not fire, fail-closed.
HYPOTHESIS_SUPPORTED: Final[str] = "SUPPORTED"
HYPOTHESIS_NOT_SUPPORTED: Final[str] = "NOT_SUPPORTED"
HYPOTHESIS_UNEVALUABLE: Final[str] = "UNEVALUABLE"
HYPOTHESIS_RESERVED: Final[str] = "RESERVED"
A12_FALLBACK_RULE_ID: Final[str] = "A12_FALLBACK"


def _strip_unavailable(ev: dict[str, float]) -> dict[str, float]:
    """Return ``ev`` without NaN-valued entries (the unavailable sentinel).

    Evaluating predicates against the stripped view makes the NaN encoding of
    a missing measurement behave exactly like its absence, in the decision
    ladder and the evidence matrix alike — otherwise the claim floor of a rung
    depended on whether the same unavailable value arrived as a NaN entry or
    as no entry at all.
    """
    if not any(math.isnan(v) for v in ev.values()):
        return ev
    return {k: v for k, v in ev.items() if not math.isnan(v)}


def _is_unavailable(ev: dict[str, float], key: str) -> bool:
    """Is ``key`` absent, or present but NaN (the unavailable-value sentinel)?

    Diagnostics use NaN for "not computed" throughout the result dataclasses
    (``henrici_relative``, ``beta_D_linear``, the D14b/D14c fields ...), so a
    presence-only check would treat an unknown measurement as collected
    evidence: every comparison against NaN is ``False``, which silently reads
    as "the threshold was not met" and lets the A12 fallback conclude that no
    mechanism applies. Infinities are NOT swept in here -- they are legitimate
    measured values with documented per-predicate semantics (a floored gap
    drives ``gap_to_gns_ratio`` to ``inf`` by design).
    """
    if key not in ev:
        return True
    return math.isnan(ev[key])


def _hypothesis_claim_floor(
    a_class: str,
    status: str,
    ev: dict[str, float],
    relaxation: RelaxationResult,
) -> str:
    """Explicit claim-floor rules for one hypothesis (issue #102).

    The floor answers "what would LiouScope be willing to claim about THIS
    hypothesis, given this run" — derived from the SAME verdict pipeline the
    winner goes through, so the matrix cannot promise more than the classifier
    would:

    * ``RESERVED`` / ``UNEVALUABLE`` -> ``UNDEFINED``: no decision rule, or
      required evidence missing — insufficient evidence, fail-closed.
    * ``NOT_SUPPORTED`` -> ``NOT_EXCLUDED``: a threshold that did not fire is
      absence of support, not positive counter-evidence (issue #70 A5: the
      diagnostics establish positive evidence; they cannot prove absence).
    * ``SUPPORTED`` -> the verdict this hypothesis would receive were it the
      winner: :func:`_confidence` -> :func:`_pick_verdict_tier` -> the A11
      single-state maximally-mixed floor. For the actual winner this equals
      the reported verdict exactly (pinned in tests).
    """
    if status in (HYPOTHESIS_RESERVED, HYPOTHESIS_UNEVALUABLE):
        return VERDICT_UNDEFINED
    if status == HYPOTHESIS_NOT_SUPPORTED:
        return VERDICT_NOT_EXCLUDED
    verdict, tier = _pick_verdict_tier(a_class, relaxation, _confidence(ev, a_class))
    verdict, tier = _apply_single_state_maxmix_floor(
        a_class,
        verdict,
        tier,
        maximally_mixed=ev.get("maximally_mixed_steady_state", 0.0) > 0.5,
        ensemble_confirmation=ev.get(ENSEMBLE_OVERRIDE_EVIDENCE_KEY, 0.0) > 0.5,
    )
    verdict, _tier = _apply_spectral_certificate_floor(
        verdict,
        tier,
        spectral_resolved=ev.get(SPECTRAL_RESOLVED_EVIDENCE_KEY, 1.0) > 0.5,
    )
    return verdict


def hypothesis_evidence_matrix(
    ev: dict[str, float],
    *,
    relaxation: RelaxationResult,
) -> tuple[dict[str, object], ...]:
    """Hypothesis-wise evidence matrix over the full A1-A12 taxonomy (#102).

    For every decision rung of :func:`_ladder_spec`, plus the A12 fallback and
    the schema-reserved classes, one entry records:

    * ``supporting`` — the atomic conditions that hold, with the evidence
      values they read (the "supporting measurements"). Conditions whose own
      inputs are present are evaluated even when a SIBLING condition is
      unevaluable, so a partially collected run keeps its usable evidence;
    * ``counterevidence`` — the conditions that fail, with their values
      (threshold non-exceedance; NOT proof of absence, see claim-floor rules);
    * ``missing`` — REQUIRED evidence keys absent from ``ev``. These make the
      rung ``UNEVALUABLE`` only when no evaluated sibling condition is already
      false (round-13 review): the rung is a conjunction, so one false
      condition refutes it conclusively — ``NOT_SUPPORTED`` — no matter what
      the missing measurement would have said, and the absent keys stay
      listed here for the audit trail; ``missing_optional`` separately records absent keys
      the predicate reads with a documented default (e.g. the Mpemba layer was
      not run), which are worth auditing but do not stop the rung being
      decided — the ladder decides it from the same default;
    * ``status`` — SUPPORTED / NOT_SUPPORTED / UNEVALUABLE / RESERVED, where
      SUPPORTED is by construction the rung's ``fires`` in the decision
      ladder (same conditions, same evaluation);
    * ``claim_floor`` — what this run could claim about the hypothesis, from
      the explicit rules in :func:`_hypothesis_claim_floor`;
    * ``support_score`` — the ordinal heuristic score :func:`_confidence`
      assigns this class (NOT a probability), and ``None`` unless the
      hypothesis is SUPPORTED: the score answers "what grade would this class
      get as the winner" and keys on a subset of the firing rule, so attaching
      it to a failed rung would print a confirmation-grade number beside that
      rung's own counterevidence.

    Trust boundary: the matrix REPORTS the supplied ``ev`` dict, it does not
    re-validate it. In the production path ``ev`` is built by
    :func:`_gather_evidence` and the ensemble override is written by
    :func:`classify_mechanism` from an argument that :func:`liouscope.diagnose`
    has already validated against a typed
    :class:`liouscope.ensemble.EnsembleEvidence` (a bare boolean is rejected
    fail-closed there, per AGENTS.md A11). A hand-built ``ev`` passed straight
    to this function therefore carries exactly the trust its caller gives it —
    the same contract as :func:`classify_mechanism` and
    :func:`_apply_single_state_maxmix_floor`, which likewise take the override
    as a plain argument.

    The matrix is a REPORT, not a decision: the dominant class remains the
    convenience projection produced by the unchanged priority chain, and no
    decision function consumes the matrix. Reserved classes (A6/A7/A9) are
    marked ``RESERVED`` with a permanent ``UNDEFINED`` floor and are thereby
    excluded from claims and from coverage denominators (the reachable-class
    denominator is ``REACHABLE_A_CLASSES``).

    Per-hypothesis numerical-uncertainty and perturbation-robustness columns
    (issue #102 wishlist) need machinery that does not exist yet and are
    deliberately NOT faked here; they remain open in the issue.
    """
    ev = _strip_unavailable(ev)
    entries: list[dict[str, object]] = []
    any_fired = False
    any_unevaluable = False
    blocking_missing: list[str] = []
    for rung in _ladder_spec():
        # ``missing`` is the issue-#102 column "missing REQUIRED evidence": it
        # drives UNEVALUABLE. A defaulted key is tracked separately -- the
        # auditor still wants to know the Mpemba layer never ran, but its
        # absence does not stop the rung being decided, because the ladder
        # decides it too (from the documented default). The same key can be
        # required for one rung and optional for another, which a single mixed
        # list could not express.
        missing_list: list[str] = []
        missing_optional_list: list[str] = []
        for cond in rung.conditions:
            for k in cond.keys:
                if _is_unavailable(ev, k) and k not in missing_list:
                    missing_list.append(k)
            for k in cond.optional_keys:
                if _is_unavailable(ev, k) and k not in missing_optional_list:
                    missing_optional_list.append(k)
        missing = tuple(missing_list)
        missing_optional = tuple(missing_optional_list)
        missing_required = bool(missing)
        supporting: list[dict[str, object]] = []
        counterevidence: list[dict[str, object]] = []
        # Evaluate every condition whose OWN required inputs are present, even
        # when a sibling condition is unevaluable: a run that measured a high
        # `kreiss` but never computed `petermann_max` should still show the
        # Kreiss support next to the missing Petermann value. Discarding it
        # would throw away exactly the audit trail this matrix exists for.
        for cond in rung.conditions:
            if any(_is_unavailable(ev, k) for k in cond.keys):
                continue
            values: dict[str, object] = {k: float(ev[k]) for k in cond.keys}
            # Optional keys are reported when present; when absent the
            # predicate supplies its documented default, so the condition is
            # still evaluated and still agrees with the ladder.
            for k in cond.optional_keys:
                if k in ev:
                    values[k] = float(ev[k])
            for f_name in cond.relaxation_fields:
                values[f_name] = getattr(relaxation, f_name)
            record: dict[str, object] = {
                "condition": cond.description,
                "values": values,
            }
            if cond.fn(ev, relaxation):
                supporting.append(record)
            else:
                counterevidence.append(record)
        # Status precedence (round-13 review). The rung is a conjunction, so
        # ONE evaluated-false condition refutes it conclusively -- no value of
        # the missing evidence could make it fire (``kreiss = 1`` refutes F1
        # whether or not ``petermann_max`` was ever measured). UNEVALUABLE is
        # reserved for the genuinely open case: no evaluated condition is
        # false AND a missing REQUIRED key could still flip the rung to
        # supported. A missing optional key never blocks either way: the rung
        # keeps a real verdict because the ladder likewise evaluates it from
        # the documented default -- declaring it unevaluable would make the
        # matrix contradict the decision it describes.
        if counterevidence:
            status = HYPOTHESIS_NOT_SUPPORTED
        elif missing_required:
            status = HYPOTHESIS_UNEVALUABLE
            any_unevaluable = True
            for k in missing:
                if k not in blocking_missing:
                    blocking_missing.append(k)
        else:
            # No counterevidence and nothing missing: every condition was
            # evaluated and held.
            status = HYPOTHESIS_SUPPORTED
            any_fired = True
        entries.append(
            {
                "rule_id": rung.rule_id,
                "a_class": rung.a_class,
                "f_family": rung.f_family,
                "status": status,
                "supporting": tuple(supporting),
                "counterevidence": tuple(counterevidence),
                "missing": missing,
                "missing_optional": missing_optional,
                "claim_floor": _hypothesis_claim_floor(
                    rung.a_class, status, ev, relaxation
                ),
                # Only a SUPPORTED hypothesis carries a score. ``_confidence``
                # answers "what grade would this class get as the WINNER" and
                # keys on a subset of the firing rule, so a failed rung would
                # otherwise show a confirmation-grade number right next to its
                # own counterevidence -- e.g. kreiss=11 with petermann_max=1
                # fails F1 yet would report A3 at 0.85. A score without support
                # is not a weaker claim, it is a misleading one.
                "support_score": (
                    _confidence(ev, rung.a_class)
                    if status == HYPOTHESIS_SUPPORTED
                    else None
                ),
            }
        )
    # A12 fallback: "mixed / unresolved" is what the classifier returns when no
    # rung fires. A fired rung refutes it outright. But ``not any_fired`` alone
    # does NOT establish it: an UNEVALUABLE rung might have fired had its
    # evidence been collected, so claiming "no mechanism applies" would assert
    # more than the run measured. With any rung unevaluable the fallback is
    # itself unevaluable -- fail-closed, and consistent with the ladder, whose
    # ``A12`` default is likewise only reached after evaluating every rung.
    # A partially collected rung with counterevidence does NOT block (round-13
    # review): it is conclusively refuted, so when every rung is either
    # refuted or fired-out, the ladder's deterministic A12 is matched by a
    # SUPPORTED fallback here -- the matrix/decision equivalence holds again.
    if any_fired:
        a12_status = HYPOTHESIS_NOT_SUPPORTED
        a12_missing: tuple[str, ...] = ()
    elif any_unevaluable:
        a12_status = HYPOTHESIS_UNEVALUABLE
        # The fallback's unevaluability is inherited: these are the required
        # keys whose absence left a rung undecided, and therefore left "no
        # mechanism applies" unestablished. Carrying them here keeps the
        # serialised entry self-auditing -- a consumer can see WHICH missing
        # measurement blocked the fallback claim without walking the rungs.
        a12_missing = tuple(blocking_missing)
    else:
        a12_status = HYPOTHESIS_SUPPORTED
        a12_missing = ()
    entries.append(
        {
            "rule_id": A12_FALLBACK_RULE_ID,
            "a_class": "A12",
            "f_family": "none",
            "status": a12_status,
            "supporting": (),
            "counterevidence": (),
            "missing": a12_missing,
            "missing_optional": (),
            "claim_floor": _hypothesis_claim_floor("A12", a12_status, ev, relaxation),
            "support_score": (
                _confidence(ev, "A12") if a12_status == HYPOTHESIS_SUPPORTED else None
            ),
            "note": "fallback: fires iff no A1-A11 decision rule fires",
        }
    )
    # Schema-reserved classes (issue #102 reachability/ontology gate): A6/A7/A9
    # have no code-backed decision rule. They are excluded from claims (floor
    # permanently UNDEFINED) and from coverage denominators; the per-class
    # rationale travels with the entry so a serialised report is self-auditing.
    for a_class in sorted(RESERVED_A_CLASSES, key=lambda a: int(a[1:])):
        entries.append(
            {
                "rule_id": f"RESERVED_{a_class}",
                "a_class": a_class,
                "f_family": None,
                "status": HYPOTHESIS_RESERVED,
                "supporting": (),
                "counterevidence": (),
                "missing": (),
                "missing_optional": (),
                "claim_floor": VERDICT_UNDEFINED,
                "support_score": None,
                "note": RESERVED_A_CLASSES[a_class],
            }
        )
    return tuple(entries)


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
        # Issue #80: A1 CONFIRMED/PUBLICATION_GRADE (0.95) requires the
        # POSITIVE symmetrised-gap certificate, not merely the absence of all
        # F1-F5 triggers. Without corroboration the honest grade is
        # CANDIDATE/CONFIRMATION (0.70): the observable is single-exp at the
        # gap rate (measured), but gap control is not operator-intrinsically
        # certified, and F1-F5 threshold exhaustiveness must not carry a
        # publication-grade claim alone.
        score = 0.95 if ev.get("sym_gap_corroborated", 0.0) > 0.5 else 0.70
    elif (
        # A2 high confidence needs the ratio AND a certified (measured) GNS
        # gap -- defence in depth: _pick_a_class already refuses A2 off the
        # floor sentinel (issue #88), so an uncertified A2 cannot reach 0.85
        # even if the branch guards drift apart.
        a_class == "A2"
        and ev.get("gap_to_gns_ratio", 1.0) > 1.5
        and ev.get("gns_certified", 0.0) > 0.5
    ) or (a_class == "A3" and ev.get("kreiss", 0.0) > 10.0) or (a_class == "A4" and ev.get("trans_amplitude_ratio", 0.0) > 10.0):
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
    *,
    ensemble_confirmation: bool = False,
) -> ClassificationResult:
    """Classify the dominant relaxation mechanism into A1..A12 with F1..F5 tag.

    ``ensemble_confirmation`` (issue #78 / decision E0706-13) is the ensemble
    override hook: when ``True`` it records that reference-ensemble Mpemba
    confirmation across a thermal family is available, which SUPPRESSES the
    single-state maximally-mixed A11 insufficient-evidence floor
    (``_apply_single_state_maxmix_floor``). It defaults to ``False`` (the current
    single-state pipeline supplies no ensemble evidence) and is surfaced in the
    ``evidence`` dict under ``ensemble_confirmation`` for audit/serialisation.
    """
    ev = _gather_evidence(
        spectral, nonnorm, relaxation, resolvent, transient, lep, mpemba
    )
    maximally_mixed = _is_maximally_mixed(spectral.steady_state)
    ev["maximally_mixed_steady_state"] = float(maximally_mixed)
    ev[ENSEMBLE_OVERRIDE_EVIDENCE_KEY] = float(bool(ensemble_confirmation))
    certificate = getattr(spectral, "zero_mode_certificate", None)
    spectral_resolved = True
    if isinstance(certificate, dict) and certificate.get("applicable"):
        spectral_resolved = bool(certificate.get("resolved", certificate.get("certified")))
        ev[SPECTRAL_RESOLVED_EVIDENCE_KEY] = float(spectral_resolved)
    a_class, f_family = _pick_a_class(ev, relaxation=relaxation)
    conf = _confidence(ev, a_class)
    verdict, tier = _pick_verdict_tier(a_class, relaxation, conf)
    verdict, tier = _apply_single_state_maxmix_floor(
        a_class,
        verdict,
        tier,
        maximally_mixed=maximally_mixed,
        ensemble_confirmation=bool(ensemble_confirmation),
    )
    verdict, tier = _apply_spectral_certificate_floor(
        verdict, tier, spectral_resolved=spectral_resolved
    )
    return ClassificationResult(
        a_class=a_class,
        f_family=f_family,
        verdict=verdict,  # type: ignore[arg-type]
        tier=tier,        # type: ignore[arg-type]
        confidence=conf,
        # Issue #102 option 1: the honestly-named field. Same value as
        # ``confidence`` (which stays as the legacy alias) -- an ordinal,
        # deterministic, rule-based support score, NOT a probability.
        support_score=conf,
        evidence=ev,
        triggered_hypotheses=triggered_hypotheses(ev, relaxation=relaxation),
        hypothesis_matrix=hypothesis_evidence_matrix(ev, relaxation=relaxation),
        taxonomy_version=TAXONOMY_VERSION,
        schema_version=DIAGNOSTIC_SCHEMA_VERSION,
    )
