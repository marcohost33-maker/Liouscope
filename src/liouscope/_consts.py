"""Canonical version strings, tolerances and taxonomy constants.

These values are stamped into reports and must remain synchronized with the
packaged schemas and public documentation.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

TAXONOMY_VERSION: Final[str] = "A1-A12-v3.1"
DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "D1-D24-Übersicht-v3-2026-04-24"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.5.0"

RESERVED_DIAGNOSTIC_SLOTS: Final[Mapping[str, str]] = MappingProxyType({
    "D21": "reserved (canon schema D1-D24-Übersicht-v3; not implemented here)",
    "D22": "reserved (canon schema D1-D24-Übersicht-v3; not implemented here)",
    "D23": "reserved (canon schema D1-D24-Übersicht-v3; not implemented here)",
})

CORE_SCOPE: Final[str] = "time-homogeneous finite-dimensional GKSL/QMS"
RELEASE_STATE: Final[str] = "engineering release-ready"
PAPER_STATE: Final[str] = "arXiv v5 submitted; peer-review pending"

EPS_SUPP: Final[float] = 1.0e-12
# LEGACY ABSOLUTE zero-mode floor. Retained as an explicit opt-in only (issue
# #108): a Liouvillian carries rate dimension, so an absolute floor is not
# invariant under a change of rate units L -> cL. Scale-relative separation is
# the default -- see ZERO_MODE_EPS_FACTOR and
# numerics.scale.spectral_zero_tolerance.
EPS_GAP: Final[float] = 1.0e-10
# Issue #108: zero-mode tolerance as a multiple of the EIGENSOLVER BACKWARD
# ERROR, ``eps * max|lambda|``, not a fixed fraction of the spectral radius.
#
# The distinction matters for metastable systems (A5), whose whole point is a
# wide separation of physical rates. A tolerance of ``1e-10 * max|lambda|``
# would impose a fixed dynamic-range ceiling of 1e10 and discard a genuine slow
# mode below it -- e.g. two damping channels at rates 1.0 and 1e-12 have a true
# gap of 5e-13 and would be reported as 5e-1, wrong by ten orders of magnitude.
#
# A computed eigenvalue carries an uncertainty of order ``eps * ||L||``, so that
# -- not a fixed ratio -- is the scale on which "indistinguishable from zero"
# is decided. Measured across amplitude damping, Rabi-driven damping,
# dephasing, a strongly non-normal near-defective generator and a 64-dim
# 3-qubit chain, each at c in {1, 1e6, 1e12}, the numerical zero mode never
# exceeded ``1.94 * eps * max|lambda|``. The factor below therefore keeps ~500x
# headroom above the observed round-off while resolving genuine modes down to
# ~2e-13 relative -- a wider dynamic range than the pre-#108 absolute floor
# afforded at unit scale.
#
# A mode below this threshold is not merely filtered by choice: it is not
# reliably separable from round-off by a dense eigensolver at all. Callers who
# need a different trade-off can pass ``rtol`` explicitly.
#
# NOTE: this does NOT reproduce the pre-#108 absolute floor at unit scale. At
# max|lambda| = 1 the threshold is ~2.2e-13, not 1e-10, so a mode between those
# two values is now classified as GENUINE where it was previously discarded --
# a change that applies without any rescaling. That direction is the intended
# improvement (it is what rescues metastable slow modes); it is recorded here
# because "only rescaled generators change" would be false.
ZERO_MODE_EPS_FACTOR: Final[float] = 1.0e3
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

RESERVED_A_CLASSES: Final[Mapping[str, str]] = MappingProxyType({
    "A6": "reserved (taxonomy A1-A12-v3.1; no classifier branch yet -- needs an "
    "accelerated-decay / operator-spreading detector distinct from A5)",
    "A7": "reserved (taxonomy A1-A12-v3.1; no classifier branch yet -- needs a "
    "weak-dissipation singular-perturbation probe, Mori 2024)",
    "A9": "reserved (taxonomy A1-A12-v3.1; no classifier branch yet -- needs "
    "ETH / level-statistics signals for the prethermalization regime)",
})

# Issue #102 reachability/ontology gate: the coverage DENOMINATOR for any
# "n of N classes" statement. Reserved classes have no code-backed decision
# rule, are excluded from claims (their hypothesis-matrix claim floor is
# permanently UNDEFINED) and must not inflate coverage denominators.
REACHABLE_A_CLASSES: Final[tuple[str, ...]] = tuple(
    a for a in A_CLASSES if a not in RESERVED_A_CLASSES
)

F_FAMILIES: Final[tuple[str, ...]] = ("F1", "F2", "F3", "F4", "F5", "none")

A_CLASS_DESCRIPTIONS: Final[Mapping[str, str]] = MappingProxyType({
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
})

F_FAMILY_DESCRIPTIONS: Final[Mapping[str, str]] = MappingProxyType({
    "F1": "Mori-Shirai overlap (PRL 125, 230604, 2020)",
    "F2": "Liouvillian skin effect (PRL 127, 070402, 2021)",
    "F3": "Symmetrised gap (PRL 130, 230404, 2023)",
    "F4": "Quantum Mpemba effect (PRL 127, 060401, 2021)",
    "F5": "Phantom relaxation (arXiv:2306.07876, 2023)",
    "none": "No gap-failure mechanism flagged",
})
