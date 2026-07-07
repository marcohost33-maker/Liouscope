"""Canonical version strings, tolerances and taxonomy constants.

These values are stamped into reports and must remain synchronized with the
packaged schemas and public documentation.
"""

from __future__ import annotations

from typing import Final

TAXONOMY_VERSION: Final[str] = "A1-A12-v3.1"
DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "D1-D24-Übersicht-v3-2026-04-24"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.4.0"

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
# Tight numerical detector for the exact maximally mixed state I/d. It is not a
# statement about protecting sectors, which are properties of the Liouvillian.
EPS_MAXMIX: Final[float] = 1.0e-9
# Separates a measured GNS gap from the numerical floor sentinel.
GNS_CERTIFIED_RTOL: Final[float] = 1.0e-8
# Nominal placeholder unless a measured solver residual is supplied.
U1_NOMINAL_FLOOR: Final[float] = 1.0e-10
# Denominator guard near the smallest positive normal double.
EPS_DIV: Final[float] = 1.0e-300

VERDICT_CONFIRMED: Final[str] = "CONFIRMED"
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
