"""Canonical version strings and tolerances stamped on every result.

These constants encode the canon from the v2.0 consolidated report
(2026-05-10). They must never drift between modules.
"""

from __future__ import annotations

from typing import Final

TAXONOMY_VERSION: Final[str] = "A1-A12-v3.1"
DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "D1-D24-Übersicht-v3-2026-04-24"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.2.0"

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
