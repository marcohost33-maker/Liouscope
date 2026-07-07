"""Public mechanism-classifier wrapper with validated ensemble evidence."""

from __future__ import annotations

from ._types import (
    ClassificationResult,
    LepResult,
    MpembaResult,
    NonNormalityResult,
    RelaxationResult,
    ResolventResult,
    SpectralResult,
    TransientResult,
)
from .diagnostics.classification import classify_mechanism as _classify_mechanism
from .ensemble import EnsembleEvidence, reject_legacy_ensemble_confirmation


def classify_mechanism(
    spectral: SpectralResult,
    nonnorm: NonNormalityResult,
    relaxation: RelaxationResult,
    resolvent: ResolventResult,
    transient: TransientResult,
    lep: LepResult,
    mpemba: MpembaResult | None = None,
    *,
    ensemble_evidence: EnsembleEvidence | None = None,
    ensemble_confirmation: bool | None = None,
) -> ClassificationResult:
    """Classify a mechanism without accepting a trust-by-boolean override.

    ``ensemble_confirmation`` remains as a compatibility trap: ``False``/``None``
    are harmless, while ``True`` raises.  Only a validated ``EnsembleEvidence``
    object with a passing, specific gate reason can suppress the single-state
    maximally-mixed A11 insufficient-evidence floor.
    """

    reject_legacy_ensemble_confirmation(ensemble_confirmation)
    confirmed = bool(
        ensemble_evidence is not None
        and ensemble_evidence.permits_claim_floor_override
    )
    return _classify_mechanism(
        spectral,
        nonnorm,
        relaxation,
        resolvent,
        transient,
        lep,
        mpemba,
        ensemble_confirmation=confirmed,
    )


__all__ = ["classify_mechanism"]
