"""Unit tests for the typed ensemble-evidence provenance contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from liouscope import ENSEMBLE_MPEMBA_CONFIRMED, EnsembleEvidence


def _evidence(**overrides: object) -> EnsembleEvidence:
    values: dict[str, object] = {
        "manifest_sha256": "1" * 64,
        "initial_state_family": "thermal Gibbs family",
        "ordering_parameter": "inverse_temperature_beta",
        "run_ids": ("2" * 64, "3" * 64),
        "input_hashes": ("4" * 64, "5" * 64),
        "relaxation_metric": "trace_distance",
        "comparison_test": "family_ordered_crossing_test",
        "uncertainty_method": "BCa bootstrap 95% CI with multiplicity control",
        "software_version": "0.6.0.dev0",
        "gate_status": "PASS",
        "reason_code": ENSEMBLE_MPEMBA_CONFIRMED,
        "producer_attestation_sha256": "6" * 64,
        "reviewer_attestation_sha256": "7" * 64,
    }
    values.update(overrides)
    return EnsembleEvidence(**values)  # type: ignore[arg-type]


def test_evidence_is_frozen_and_digest_is_stable():
    evidence = _evidence()
    with pytest.raises(FrozenInstanceError):
        evidence.gate_status = "FAIL"  # type: ignore[misc]
    assert evidence.sha256 == _evidence().sha256
    assert len(evidence.sha256) == 64


def test_override_requires_pass_and_specific_reason_code():
    assert _evidence().permits_claim_floor_override
    assert not _evidence(gate_status="REVIEW").permits_claim_floor_override
    assert not _evidence(reason_code="UNRELATED_PASS").permits_claim_floor_override


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"manifest_sha256": "not-a-digest"}, "manifest_sha256"),
        ({"run_ids": ("2" * 64,)}, "at least two"),
        ({"input_hashes": ("4" * 64,)}, "equal length"),
        ({"run_ids": ("2" * 64, "2" * 64)}, "unique"),
        (
            {
                "producer_attestation_sha256": "6" * 64,
                "reviewer_attestation_sha256": "6" * 64,
            },
            "distinct",
        ),
    ],
)
def test_malformed_evidence_fails_closed(overrides: dict[str, object], message: str):
    with pytest.raises(ValueError, match=message):
        _evidence(**overrides)
