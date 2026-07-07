"""Canonical version strings and tolerances stamped on every result.

These constants encode the canon from the v2.0 consolidated report
(2026-05-10). They must never drift between modules.
"""

from __future__ import annotations

from typing import Final

TAXONOMY_VERSION: Final[str] = "A1-A12-v3.1"
DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "D1-D24-Übersicht-v3-2026-04-24"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.3.0"

# Reserved diagnostic slots. The schema name spans D1-D24, but only D1-D20
# (peer-review set) and D24 (opt-in Zhou mixing-time predictor) are implemented
# in this repository. D21-D23 are defined in the Drive-side canon schema and
# reserved here as an explicit code-level contract (issue #71 C2) so the gap is
# discoverable next to the schema version instead of only in prose. Reserving a
# slot is NOT a claim that it is implemented -- consumers must treat these ids as
# absent from any run's diagnostics block until they graduate to real code with
# their own anchor coverage (``claim_status:pending`` until anchors confirm).
RESERVED_DIAGNOSTIC_SLOTS: Final[dict[str, str]] = {
    "D21": "reserved (canon schema D1-D24-Übersicht-v3; not implemented here)",
    "D22": "reserved (canon schema D1-D24-Übersicht-v3; not implemented here)",
    "D23": "reserved (canon schema D1-D24-Übersicht-v3; not implemented here)",
}

CORE_SCOPE: Final[str] = "time-homogeneous finite-dimensional GKSL/QMS"
RELEASE_STATE: Final[str] = "engineering release-ready"
PAPER_STATE: Final[str] = "arXiv v5 submitted; peer-review pending"

EPS_SUPP: Final[float] = 1.0e-12
EPS_GAP: Final[float] = 1.0e-10
EPS_HERMITICITY: Final[float] = 1.0e-9
EPS_TRACE: Final[float] = 1.0e-10

# Maximally-mixed steady-state tolerance (issue #78 / decision E0706-13). A
# PURELY NUMERICAL guard that identifies the EXACT ``rho_ss = I/d`` limit: a
# steady state counts as maximally mixed only when every entry matches I/d to
# within this absolute tolerance. Set at the hermiticity/degeneracy scale (1e-9)
# -- the same order used for the steady-state eigenprojector grouping in
# mpemba._steady_state_eigenprojectors -- so genuine solver noise on an exact I/d
# null space (~1e-16 in practice) is absorbed. This is NOT a claim that a state
# whose deviation from I/d exceeds the tolerance thereby carries a protecting
# symmetry sector: protecting / decoherence-free sectors are properties of the
# Liouvillian's structure, not of the fixpoint's distance from I/d (a finite-T
# amplitude-damping qubit, rho_ss = diag(1/2+e, 1/2-e) with e > 1e-9, has no such
# sector). The bound is deliberately TIGHT so the A11 insufficient-evidence floor
# fires ONLY at the exact maximally-mixed limit; a near-mixed but structured
# steady state simply follows the normal classification path.
EPS_MAXMIX: Final[float] = 1.0e-9

# Certification floor for the symmetrised (GNS) gap, RELATIVE to the spectral
# gap Delta (issue #88). ``diagnostics.spectral.gns_gap`` deliberately floors
# Delta_GNS to ~0 whenever the GNS symmetrisation cannot certify exponential
# contraction (the documented 2026-07 audit-A1 behaviour for non-detailed-
# balance steady states carrying coherences). That floored value is a
# *sentinel* -- numerically it comes out at O(machine-eps) relative to the
# operator scale (observed ~1e-11..1e-12 x Delta) -- and certifies the ABSENCE
# of a GNS contraction bound, not a measured Mori-Shirai symmetrised-gap
# reduction. The classifier therefore treats Delta_GNS as a MEASURED gap only
# when ``gns_gap >= GNS_CERTIFIED_RTOL * gap``. The threshold sits far above
# the eigensolver noise floor (O(d * 2.2e-16)) and far below any physically
# resolvable reduction, so it separates "floored to numerical zero" from
# "genuinely tiny but resolved" with a wide margin on both sides.
GNS_CERTIFIED_RTOL: Final[float] = 1.0e-8

# U1 (solver uncertainty) nominal floor. ``compute_uncertainty_layer`` reports
# this constant for U1 unless the caller supplies a real ``solver_residual``
# from an ODE-tolerance sweep. It is a *nominal placeholder* -- a conservative
# lower bound on the solver contribution, NOT a measured residual. Kept here as
# a single named source so the semantics are explicit in code (issue #71 B5)
# rather than a bare magic number inside the uncertainty layer.
U1_NOMINAL_FLOOR: Final[float] = 1.0e-10

# Division-by-zero floor (NOT a physical tolerance). Guards denominators that
# can legitimately underflow to ~0 for perfectly defective modes, e.g. the
# Petermann inner product |<l, r>|^2. Set near the smallest positive normal
# double (~2.2e-308) so that only a genuinely vanishing denominator yields inf;
# a merely small-but-finite denominator still produces a large finite value.
# Deliberately distinct from EPS_GAP/EPS_SUPP, which encode physics-scale
# thresholds — collapsing this onto EPS_GAP would wrongly flag near-defective
# (but finite) modes as inf. See nonnormality.petermann_factors and _zhou.
EPS_DIV: Final[float] = 1.0e-300

VERDICT_CONFIRMED: Final[str] = "CONFIRMED"
# issue #70 A5: VERDICT_EXCLUDED removed -- it was unreachable through
# ``diagnose()`` and semantically inexpressible in a single-pass best-class
# classifier (active exclusion needs per-hypothesis scoring; deferred).
VERDICT_CANDIDATE: Final[str] = "CANDIDATE"
VERDICT_NOT_EXCLUDED: Final[str] = "NOT_EXCLUDED"
VERDICT_UNDEFINED: Final[str] = "UNDEFINED"

TIER_PUBLICATION: Final[str] = "PUBLICATION_GRADE"
TIER_CONFIRMATION: Final[str] = "CONFIRMATION"
TIER_EXPLORATION: Final[str] = "EXPLORATION"

QUALITY_STABLE: Final[str] = "stable"
QUALITY_MODERATE: Final[str] = "moderate"
QUALITY_EXPLORATORY: Final[str] = "exploratory"

A_CLASSES: Final[tuple[str, ...]] = (
    "A1", "A2", "A3", "A4", "A5", "A6",
    "A7", "A8", "A9", "A10", "A11", "A12",
)

# Reserved A-classes (issue #70 B3). The taxonomy ``A1-A12-v3.1`` defines twelve
# mechanism classes, but the single-pass heuristic ``_pick_a_class`` currently
# emits only nine of them: A6 (accelerated-decay / operator-spreading), A7
# (weak-dissipation singular, Mori 2024) and A9 (prethermalization / ETH) have no
# decision branch yet and would each require a diagnostic that does not exist in
# this repository (A6: an accelerated-decay detector distinguishing it from the
# A5 metastable plateau; A7: a dissipation-strength / singular-perturbation
# probe; A9: ETH / level-statistics signals). Rather than let ``diagnose()``
# silently never emit three advertised labels, they are recorded here as an
# explicit, discoverable code-level contract -- exactly like
# ``RESERVED_DIAGNOSTIC_SLOTS`` (D21-D23). Reserving a class is NOT a claim that
# it is emitted: consumers must treat these ids as taxonomy-defined but
# not-yet-reachable until a dedicated PR adds a branch WITH anchor coverage and
# false-positive tests (``claim_status:pending`` until anchors confirm). This is
# behaviour-preserving: no real-input classification result changes -- it only
# names a pre-existing gap. Pinned by ``tests/test_classifier_semantics_debt.py``.
RESERVED_A_CLASSES: Final[dict[str, str]] = {
    "A6": "reserved (taxonomy A1-A12-v3.1; no classifier branch yet -- needs an "
    "accelerated-decay / operator-spreading detector distinct from A5)",
    "A7": "reserved (taxonomy A1-A12-v3.1; no classifier branch yet -- needs a "
    "weak-dissipation singular-perturbation probe, Mori 2024)",
    "A9": "reserved (taxonomy A1-A12-v3.1; no classifier branch yet -- needs "
    "ETH / level-statistics signals for the prethermalization regime)",
}

F_FAMILIES: Final[tuple[str, ...]] = ("F1", "F2", "F3", "F4", "F5", "none")

A_CLASS_DESCRIPTIONS: Final[dict[str, str]] = {
    "A1": "Asymptotic-gap-controlled (primitive QMS)",
    "A2": "Sym-gap-corrected transient (Mori-Shirai 2023)",
    "A3": "Overlap/eigenvector-amplified (Mori-Shirai 2020)",
    "A4": "Skin-affected (Haga 2021)",
    "A5": "Metastable plateau (Macieszczak 2016)",
    "A6": "Accelerated-decay / operator-spreading",
    "A7": "Weak-dissipation singular (Mori 2024)",
    "A8": "Oscillatory transient (complex pairs)",
    "A9": "Prethermalization-affected (ETH regime)",
    "A10": "Phantom relaxation (Znidaric 2023)",
    "A11": "Non-normal Mpemba (Entropy 27, 581, 2025)",
    "A12": "Mixed / unresolved",
}

F_FAMILY_DESCRIPTIONS: Final[dict[str, str]] = {
    "F1": "Mori-Shirai overlap (PRL 125, 230604, 2020)",
    "F2": "Liouvillian skin effect (PRL 127, 070402, 2021)",
    "F3": "Symmetrised gap (PRL 130, 230404, 2023)",
    "F4": "Quantum Mpemba effect (PRL 127, 060401, 2021)",
    "F5": "Phantom relaxation (arXiv:2306.07876, 2023)",
    "none": "No gap-failure mechanism flagged",
}
