"""LiouScope -- Multi-diagnostic relaxation analysis for open quantum systems.

Public API:

* :func:`build_liouvillian` -- construct a column-stacking GKSL superoperator.
* :func:`steady_state` -- find ``rho_ss`` with ``L rho_ss = 0`` and unit trace.
* :func:`diagnose` -- run the full six-layer diagnostic pipeline.
* :func:`classify_mechanism` -- standalone mechanism classifier.

Constants:

* :data:`TAXONOMY_VERSION` = ``"A1-A12-v3.1"``
* :data:`DIAGNOSTIC_SCHEMA_VERSION` = ``"D1-D24-Übersicht-v3-2026-04-24"``
"""

from ._classify import classify_mechanism
from ._consts import (
    A_CLASS_DESCRIPTIONS,
    A_CLASSES,
    DIAGNOSTIC_SCHEMA_VERSION,
    F_FAMILIES,
    F_FAMILY_DESCRIPTIONS,
    TAXONOMY_VERSION,
)
from ._diagnostics import diagnose
from ._liouvillian import build_liouvillian, steady_state
from ._types import (
    ClassificationResult,
    DiagnosticReport,
    FitResult,
    GovernanceMetadata,
    LepResult,
    MpembaResult,
    NonNormalityResult,
    RelaxationResult,
    ResolventResult,
    SpectralResult,
    TransientResult,
    UncertaintyResult,
)
from ._version import __version__
from .ensemble import (
    ENSEMBLE_EVIDENCE_SCHEMA_VERSION,
    ENSEMBLE_MPEMBA_CONFIRMED,
    EnsembleEvidence,
)
from .io.seed import derive_seed, seed_everything

__all__ = [
    "A_CLASSES",
    "A_CLASS_DESCRIPTIONS",
    "DIAGNOSTIC_SCHEMA_VERSION",
    "ENSEMBLE_EVIDENCE_SCHEMA_VERSION",
    "ENSEMBLE_MPEMBA_CONFIRMED",
    "F_FAMILIES",
    "F_FAMILY_DESCRIPTIONS",
    "TAXONOMY_VERSION",
    "ClassificationResult",
    "DiagnosticReport",
    "EnsembleEvidence",
    "FitResult",
    "GovernanceMetadata",
    "LepResult",
    "MpembaResult",
    "NonNormalityResult",
    "RelaxationResult",
    "ResolventResult",
    "SpectralResult",
    "TransientResult",
    "UncertaintyResult",
    "__version__",
    "build_liouvillian",
    "classify_mechanism",
    "derive_seed",
    "diagnose",
    "seed_everything",
    "steady_state",
]
