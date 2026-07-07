"""Typed, immutable evidence contract for ensemble-level Mpemba claims.

A single caller-controlled boolean is not evidence.  This module defines the
minimum structured provenance required before the single-state maximally-mixed
A11 insufficient-evidence floor may be lifted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Literal

EnsembleGateStatus = Literal["PASS", "FAIL", "REVIEW"]
ENSEMBLE_EVIDENCE_SCHEMA_VERSION = "1.0"
ENSEMBLE_MPEMBA_CONFIRMED = "ENSEMBLE_MPEMBA_CONFIRMED"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(
            f"{name} must be a 64-character lowercase hexadecimal SHA-256 digest"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class EnsembleEvidence:
    """Validated provenance for a reference-family Mpemba comparison.

    The object is deliberately descriptive rather than self-certifying: it binds
    the claim to immutable digests, run identities, the comparison method and two
    attestations.  It does not prove that an attester is independent; repository
    review and release policy remain separate controls.
    """

    manifest_sha256: str
    initial_state_family: str
    ordering_parameter: str
    run_ids: tuple[str, ...]
    input_hashes: tuple[str, ...]
    relaxation_metric: str
    comparison_test: str
    uncertainty_method: str
    software_version: str
    gate_status: EnsembleGateStatus
    reason_code: str
    producer_attestation_sha256: str
    reviewer_attestation_sha256: str
    schema_version: str = ENSEMBLE_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Canonicalise sequence inputs even when a caller supplied lists.
        object.__setattr__(self, "run_ids", tuple(self.run_ids))
        object.__setattr__(self, "input_hashes", tuple(self.input_hashes))

        if self.schema_version != ENSEMBLE_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                "schema_version must be "
                f"{ENSEMBLE_EVIDENCE_SCHEMA_VERSION!r}, got {self.schema_version!r}"
            )
        _require_sha256("manifest_sha256", self.manifest_sha256)
        _require_sha256(
            "producer_attestation_sha256", self.producer_attestation_sha256
        )
        _require_sha256(
            "reviewer_attestation_sha256", self.reviewer_attestation_sha256
        )
        if self.producer_attestation_sha256 == self.reviewer_attestation_sha256:
            raise ValueError("producer and reviewer attestations must be distinct")

        for name, value in (
            ("initial_state_family", self.initial_state_family),
            ("ordering_parameter", self.ordering_parameter),
            ("relaxation_metric", self.relaxation_metric),
            ("comparison_test", self.comparison_test),
            ("uncertainty_method", self.uncertainty_method),
            ("software_version", self.software_version),
            ("reason_code", self.reason_code),
        ):
            _require_nonempty(name, value)

        if self.gate_status not in ("PASS", "FAIL", "REVIEW"):
            raise ValueError(
                "gate_status must be one of {'PASS', 'FAIL', 'REVIEW'}, "
                f"got {self.gate_status!r}"
            )
        if len(self.run_ids) < 2:
            raise ValueError("run_ids must contain at least two reference-family runs")
        if len(self.run_ids) != len(self.input_hashes):
            raise ValueError("run_ids and input_hashes must have equal length")
        if len(set(self.run_ids)) != len(self.run_ids):
            raise ValueError("run_ids must be unique")
        for index, digest in enumerate(self.run_ids):
            _require_sha256(f"run_ids[{index}]", digest)
        for index, digest in enumerate(self.input_hashes):
            _require_sha256(f"input_hashes[{index}]", digest)

    @property
    def permits_claim_floor_override(self) -> bool:
        """Whether this evidence may lift the A11 insufficient-evidence floor."""

        return (
            self.gate_status == "PASS"
            and self.reason_code == ENSEMBLE_MPEMBA_CONFIRMED
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible evidence payload."""

        return {
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "initial_state_family": self.initial_state_family,
            "ordering_parameter": self.ordering_parameter,
            "run_ids": list(self.run_ids),
            "input_hashes": list(self.input_hashes),
            "relaxation_metric": self.relaxation_metric,
            "comparison_test": self.comparison_test,
            "uncertainty_method": self.uncertainty_method,
            "software_version": self.software_version,
            "gate_status": self.gate_status,
            "reason_code": self.reason_code,
            "producer_attestation_sha256": self.producer_attestation_sha256,
            "reviewer_attestation_sha256": self.reviewer_attestation_sha256,
        }

    @property
    def sha256(self) -> str:
        """SHA-256 over canonical JSON for hashing and report provenance."""

        encoded = json.dumps(
            self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def reject_legacy_ensemble_confirmation(value: bool | None) -> None:
    """Reject the former trust-by-boolean override while tolerating false/no-op calls."""

    if value is None or value is False:
        return
    if value is True:
        raise ValueError(
            "ensemble_confirmation=True is no longer accepted: pass a validated "
            "EnsembleEvidence object via ensemble_evidence instead"
        )
    raise TypeError("ensemble_confirmation must be bool or None")


__all__ = [
    "ENSEMBLE_EVIDENCE_SCHEMA_VERSION",
    "ENSEMBLE_MPEMBA_CONFIRMED",
    "EnsembleEvidence",
    "EnsembleGateStatus",
]
