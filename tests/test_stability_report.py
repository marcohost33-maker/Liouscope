"""Tests for the StabilityReport v2.1 contract (LIOU-RPT-001).

Oracles:

* For a genuine GKSL generator and its steady state the three invariant
  residuals (independently recomputed here) are all ~0: trace preservation
  ``<<I|L = 0``, hermiticity ``rho_ss = rho_ss^dag`` and stationarity
  ``L vec(rho_ss) = 0``.
* The produced payload round-trips through the bundled JSON schema
  (``validate_stability_report``) and a dump/load cycle.
* The verdict fields are fail-closed: an out-of-enum claim_level / direction /
  cp_evidence_level, or a tampered required field, raises.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose, steady_state
from liouscope.io.stability_report import (
    EvidenceBundle,
    build_stability_report,
    dump_stability_report,
    invariant_residuals,
    validate_stability_report,
)

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_LOWER = (0.5 * (_X - 1j * _Y)).conj().T


def _gksl() -> tuple[np.ndarray, np.ndarray]:
    # Amplitude damping (H=0): a fast, real-spectrum GKSL with a unique steady
    # state. (A driven system with a Bohr frequency makes the unrelated
    # resolvent/LEP layers slow; the StabilityReport contract is system-agnostic
    # so the cheap canonical channel suffices here.)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.6])
    return L, steady_state(L)


def _report_and_payload():
    L, rho_ss = _gksl()
    report = diagnose(L, rho_steady_state=rho_ss, bootstrap_B=50, include_mpemba=False)
    payload = build_stability_report(
        report,
        L_super=L,
        rho_ss=rho_ss,
        claim_level="REVIEW",
        direction="slowdown",
        cp_evidence_level="choi_psd_proof",
        cp_choi_min_eig=1.0e-6,
        evidence_bundle=EvidenceBundle(
            benchmarks=("B003",), oracles=("qutip",), external_refs=("PRL130.230404",)
        ),
        pending_diagnostics=("D_tr_curve", "cptp_choi_gate"),
        system_label="amplitude-damping qubit",
    )
    return report, payload, L, rho_ss


def test_invariant_residuals_are_zero_for_gksl():
    L, rho_ss = _gksl()
    inv = invariant_residuals(L, rho_ss)
    assert inv["trace_preservation_residual"] < 1e-9
    assert inv["hermiticity_residual"] < 1e-9
    assert inv["stationarity_residual"] < 1e-8


def test_payload_validates_against_schema():
    _, payload, _, _ = _report_and_payload()
    validate_stability_report(payload)  # must not raise
    assert payload["schema_version"] == "liouscope.stability_report.v2.1"
    for key in (
        "schema_version", "run_id", "system", "generator", "invariants",
        "stationary_structure", "diagnostics", "classification",
        "evidence_bundle", "provenance",
    ):
        assert key in payload
    cls = payload["classification"]
    assert {"mechanism", "direction", "claim_level"} <= set(cls)
    assert cls["claim_level"] == "REVIEW"
    assert cls["cp_evidence_level"] == "choi_psd_proof"


def test_pending_diagnostics_are_stamped():
    _, payload, _, _ = _report_and_payload()
    assert payload["diagnostics"]["D_tr_curve"] == {"claim_status": "pending"}
    assert payload["diagnostics"]["cptp_choi_gate"] == {"claim_status": "pending"}
    # Anchor-confirmed diagnostics carry their numeric value, not 'pending'.
    assert isinstance(payload["diagnostics"]["D1_gap"], float)


def test_run_id_shared_with_report():
    report, payload, _, _ = _report_and_payload()
    assert payload["run_id"] == report.governance.run_id
    assert payload["provenance"]["input_hash"] == report.governance.input_hash


def test_dump_round_trip(tmp_path):
    _, payload, _, _ = _report_and_payload()
    out = tmp_path / "nested" / "stability.json"
    dump_stability_report(payload, out)
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    validate_stability_report(reloaded)
    assert reloaded["run_id"] == payload["run_id"]


# --- negative / edge-input gates ---------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"claim_level": "GREEN"}, "claim_level"),
        ({"direction": "sideways"}, "direction"),
        ({"cp_evidence_level": "vibes"}, "cp_evidence_level"),
        ({"null_dimension": 0}, "null_dimension"),
    ],
)
def test_build_rejects_bad_verdict_fields(kwargs, match):
    L, rho_ss = _gksl()
    report = diagnose(L, rho_steady_state=rho_ss, bootstrap_B=50, include_mpemba=False)
    base = {
        "L_super": L,
        "rho_ss": rho_ss,
        "claim_level": "SAFE",
        "direction": "slowdown",
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_stability_report(report, **base)


def test_validate_rejects_missing_field():
    _, payload, _, _ = _report_and_payload()
    broken = dict(payload)
    del broken["invariants"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_stability_report(broken)


def test_validate_rejects_tampered_claim_level():
    _, payload, _, _ = _report_and_payload()
    payload["classification"]["claim_level"] = "MAYBE"
    with pytest.raises(ValueError, match="claim_level"):
        validate_stability_report(payload)


def test_validate_rejects_bad_run_id():
    _, payload, _, _ = _report_and_payload()
    payload["run_id"] = "not-a-sha"
    with pytest.raises(ValueError, match="run_id"):
        validate_stability_report(payload)
