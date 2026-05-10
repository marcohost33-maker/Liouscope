"""Canonical version strings and tolerances stamped on every result.

These constants encode the canon from the v2.0 consolidated report
(2026-05-10). They must never drift between modules.
"""

from __future__ import annotations

from typing import Final

TAXONOMY_VERSION: Final[str] = "A1-A12-v3.1"
DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "D1-D24-Übersicht-v3-2026-04-24"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.2.0"

CORE_SCOPE: Final[str] = "time-homogeneous finite-dimensional GKSL/QMS"
RELEASE_STATE: Final[str] = "engineering release-ready"
PAPER_STATE: Final[str] = "arXiv v5 submitted; peer-review pending"

EPS_SUPP: Final[float] = 1.0e-12
EPS_GAP: Final[float] = 1.0e-10
EPS_HERMITICITY: Final[float] = 1.0e-9
EPS_TRACE: Final[float] = 1.0e-10

VERDICT_CONFIRMED: Final[str] = "CONFIRMED"
VERDICT_EXCLUDED: Final[str] = "EXCLUDED"
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
