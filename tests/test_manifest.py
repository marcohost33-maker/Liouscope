"""Tests for IO/manifest layer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from liouscope import (
    DIAGNOSTIC_SCHEMA_VERSION,
    TAXONOMY_VERSION,
    build_liouvillian,
    diagnose,
    seed_everything,
)
from liouscope.io import (
    compute_input_hash,
    dump_manifest,
    dump_report,
    load_report,
    make_run_id,
    manifest_payload,
    validate_manifest,
)


def test_seed_everything_returns_generator():
    rng = seed_everything(0)
    a = rng.standard_normal(4)
    rng2 = seed_everything(0)
    b = rng2.standard_normal(4)
    np.testing.assert_array_equal(a, b)


def test_input_hash_deterministic():
    A = np.array([1.0, 2.0, 3.0])
    h1 = compute_input_hash(A, 42)
    h2 = compute_input_hash(A, 42)
    assert h1 == h2
    h3 = compute_input_hash(A, 43)
    assert h1 != h3


def test_run_id_format():
    rid = make_run_id("a" * 64, 42, "0.2.0")
    assert len(rid) == 64
    int(rid, 16)  # valid hex


def test_diagnose_run_id_deterministic(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    r1 = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    r2 = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    # Run IDs are deterministic given (input, seed, version) and exclude timestamp.
    assert r1.governance.run_id == r2.governance.run_id


def test_dump_report_roundtrip(tmp_path: Path, pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    fname = tmp_path / "report.json"
    dump_report(report, fname)
    loaded = load_report(fname)
    assert loaded["spectral"]["gap"] == report.spectral.gap
    assert loaded["governance"]["run_id"] == report.governance.run_id
    # File is valid JSON
    json.loads(fname.read_text())


def _small_report():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(0.5 * sx, [sz], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    return diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)


def test_manifest_payload_carries_schema_versions():
    payload = manifest_payload(_small_report())
    assert payload["taxonomy_version"] == TAXONOMY_VERSION
    assert payload["diagnostic_schema_version"] == DIAGNOSTIC_SCHEMA_VERSION
    assert payload["schema_version"] == "1.2.0"
    assert re.fullmatch(r"[0-9a-f]{64}", payload["run_id"])


def test_manifest_payload_passes_validation():
    validate_manifest(manifest_payload(_small_report()))


def test_validate_manifest_rejects_missing_field():
    payload = manifest_payload(_small_report())
    del payload["taxonomy_version"]
    with pytest.raises(ValueError, match="missing"):
        validate_manifest(payload)


def test_validate_manifest_rejects_bad_run_id():
    payload = manifest_payload(_small_report())
    payload["run_id"] = "not-hex"
    with pytest.raises(ValueError, match="run_id"):
        validate_manifest(payload)


def test_dump_manifest_writes_schema_compliant_json(tmp_path: Path):
    report = _small_report()
    fname = tmp_path / "manifest.json"
    dump_manifest(report, fname)
    loaded = json.loads(fname.read_text())
    validate_manifest(loaded)
    assert loaded["framework_version"] == report.governance.framework_version


def test_governance_timestamp_is_iso_with_z_suffix():
    report = _small_report()
    ts = report.governance.timestamp
    assert ts.endswith("Z")
    # Parses cleanly as ISO 8601 once the Z is replaced with a UTC offset.
    import datetime

    datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
