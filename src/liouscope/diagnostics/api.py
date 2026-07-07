"""Validated public API for the mechanism classifier."""

from __future__ import annotations

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
from ..ensemble import EnsembleEvidence, reject_legacy_ensemble_confirmation
from .classification import classify_mechanism as _classify_mechanism


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
    """Classify without accepting a caller-controlled Boolean as evidence."""

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
