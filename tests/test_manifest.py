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


def test_input_hash_framing_is_injective():
    """Length-framed encoding (schema 1.5.0) removes concatenation collisions.

    Before framing, objects were absorbed via bare ``repr`` concatenation, so
    ``(12, 3)`` and ``(1, 23)`` both hashed the stream "123". The length-framed,
    type-tagged encoding must keep distinct input tuples distinct regardless of
    how the boundary between adjacent fields falls.
    """
    assert compute_input_hash(12, 3) != compute_input_hash(1, 23)
    assert compute_input_hash("a", "bc") != compute_input_hash("ab", "c")
    # Re-grouping array content across the shape boundary must not alias either.
    assert compute_input_hash(np.array([1.0, 2.0]), np.array([3.0])) != (
        compute_input_hash(np.array([1.0]), np.array([2.0, 3.0]))
    )
    # A raw ndarray byte block must not alias a repr that shares its bytes.
    assert compute_input_hash(np.array([1], dtype=np.int8)) != compute_input_hash(
        b"\x01"
    )


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


def test_run_id_distinguishes_analysis_config(pauli):
    """Two runs that differ only in an output-affecting analysis knob must get
    distinct run_id / input_hash. Otherwise reports that differ (e.g. a wider
    BCa CI from a larger bootstrap count) collide on the same provenance key,
    breaking the manifest's reproducibility contract."""
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    r_small = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    r_large = diagnose(L, rho_initial=rho0, bootstrap_B=80, seed=42)
    assert r_small.governance.input_hash != r_large.governance.input_hash
    assert r_small.governance.run_id != r_large.governance.run_id
    # include_mpemba is likewise an output-affecting input.
    r_no_mpemba = diagnose(
        L, rho_initial=rho0, bootstrap_B=20, seed=42, include_mpemba=False
    )
    assert r_no_mpemba.governance.run_id != r_small.governance.run_id


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
    assert payload["schema_version"] == "1.5.0"
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

    parsed = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo == datetime.timezone.utc


def test_governance_timestamp_has_fixed_width():
    """timespec='microseconds' must produce a constant-length string."""
    # YYYY-MM-DDTHH:MM:SS.ffffff + 'Z' = 27 characters.
    ts1 = _small_report().governance.timestamp
    ts2 = _small_report().governance.timestamp
    assert len(ts1) == len(ts2) == 27


def test_schema_loads_from_package():
    """``MANIFEST_SCHEMA.json`` must ship inside the package, not the repo root.

    A wheel install does not include files outside ``src/liouscope/``;
    pyproject.toml declares the schema as package-data, so the file
    must live next to ``__init__.py``.
    """
    from importlib import resources

    files = resources.files("liouscope")
    schema_text = files.joinpath("MANIFEST_SCHEMA.json").read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    assert schema["properties"]["schema_version"]["const"] == "1.5.0"
    assert schema["properties"]["taxonomy_version"]["const"] == TAXONOMY_VERSION
    assert schema["properties"]["diagnostic_schema_version"]["const"] == DIAGNOSTIC_SCHEMA_VERSION


def test_validator_is_cached():
    """The compiled jsonschema validator must be reused across calls."""
    pytest.importorskip("jsonschema")
    from liouscope.io.manifest import _compiled_validator

    a = _compiled_validator()
    b = _compiled_validator()
    assert a is b  # lru_cache memoises the validator


# --- Version single-source (audit 2026-06-06, P0) -------------------------


def test_version_single_source():
    """Wheel metadata, runtime ``__version__`` and the framework_version a run
    records must all be the same string.

    Regression for the 0.2.0/0.3.0 drift: ``pyproject.toml`` carried a second
    hard-coded literal that diverged from ``_version.py``. We now resolve the
    version dynamically from ``_version.py``; this test fails closed if the two
    planes ever diverge again.
    """
    from importlib.metadata import version as dist_version

    import liouscope

    installed = dist_version("liouscope")
    assert installed == liouscope.__version__, (
        f"installed dist metadata {installed!r} != runtime "
        f"liouscope.__version__ {liouscope.__version__!r}"
    )
    # And the manifest writer stamps exactly that string as framework_version.
    assert _small_report().governance.framework_version == liouscope.__version__


# --- Manifest byte-determinism (audit 2026-06-06, P1) ---------------------


def _manifest_bytes(report, tmp_path: Path, name: str) -> bytes:
    p = tmp_path / name
    dump_manifest(report, p)
    return p.read_bytes()


def test_manifests_byte_identical_modulo_timestamp(tmp_path: Path):
    """Two runs (same seed/inputs) are byte-identical once ``timestamp`` is
    removed — the *true* reproducibility property the README now claims.

    Also asserts ``run_id`` and ``input_hash`` are run-invariant (never derived
    from the clock).
    """
    p1 = manifest_payload(_small_report())
    p2 = manifest_payload(_small_report())
    assert p1["run_id"] == p2["run_id"]
    assert p1["input_hash"] == p2["input_hash"]
    del p1["timestamp"]
    del p2["timestamp"]
    # Byte-compare the canonical (sorted-key) JSON projection.
    b1 = json.dumps(p1, indent=2, sort_keys=True).encode("utf-8")
    b2 = json.dumps(p2, indent=2, sort_keys=True).encode("utf-8")
    assert b1 == b2


def test_manifests_byte_identical_with_source_date_epoch(tmp_path: Path, monkeypatch):
    """With ``SOURCE_DATE_EPOCH`` fixed, manifests are *fully* byte-identical
    (timestamp included) — proves the strong reproducibility option.
    """
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")
    b1 = _manifest_bytes(_small_report(), tmp_path, "m1.json")
    b2 = _manifest_bytes(_small_report(), tmp_path, "m2.json")
    assert b1 == b2, "manifests differ despite fixed SOURCE_DATE_EPOCH"
    # The fixed epoch is actually reflected in the timestamp field.
    payload = json.loads(b1)
    assert payload["timestamp"] == "2023-11-14T22:13:20.000000Z"


def test_source_date_epoch_rejects_garbage(monkeypatch):
    """A non-integer SOURCE_DATE_EPOCH fails closed rather than silently
    falling back to the wall clock (which would re-introduce non-determinism).
    """
    from liouscope.io.manifest import _utc_now_iso

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "not-a-number")
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        _utc_now_iso()
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "-5")
    with pytest.raises(ValueError, match="non-negative"):
        _utc_now_iso()


def test_source_date_epoch_unset_uses_wall_clock(monkeypatch):
    """When unset/empty, the writer still produces a valid Z-suffixed ISO
    timestamp (no regression of the default path).
    """
    from liouscope.io.manifest import _utc_now_iso

    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    ts = _utc_now_iso()
    assert ts.endswith("Z")
    assert len(ts) == 27
