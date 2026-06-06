"""Robustness of the JSON export/load layer (``liouscope.io.export``).

Covers the fail-open bug class: a report file read raw after an implicit
existence assumption. ``load_report`` must fail closed with structured,
path-bearing errors for a missing file, malformed JSON, or non-object JSON --
not a bare ``FileNotFoundError`` / ``json.JSONDecodeError`` with no context.
Also covers parent-directory creation on ``dump_report`` / ``dump_manifest``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope.io import dump_manifest, dump_report, load_report, validate_manifest


def _report():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(0.5 * sx, [sz], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    return diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)


# --- load_report fail-closed paths ------------------------------------------


def test_load_report_missing_file_raises_structured(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="report file not found"):
        load_report(missing)


def test_load_report_directory_is_not_a_file(tmp_path: Path):
    # A directory path "exists" but is not a readable report.
    with pytest.raises(FileNotFoundError, match="report file not found"):
        load_report(tmp_path)


def test_load_report_malformed_json_raises_valueerror(tmp_path: Path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_report(bad)


def test_load_report_non_object_json_raises_valueerror(tmp_path: Path):
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object at top level"):
        load_report(arr)


def test_load_report_error_mentions_path(tmp_path: Path):
    bad = tmp_path / "weird_name.json"
    bad.write_text("nonsense", encoding="utf-8")
    with pytest.raises(ValueError, match=r"weird_name\.json"):
        load_report(bad)


# --- dump_report / dump_manifest robustness ---------------------------------


def test_dump_report_creates_missing_parent_dirs(tmp_path: Path):
    nested = tmp_path / "a" / "b" / "report.json"
    report = _report()
    dump_report(report, nested)
    loaded = load_report(nested)
    assert loaded["governance"]["run_id"] == report.governance.run_id


def test_dump_manifest_creates_missing_parent_dirs(tmp_path: Path):
    nested = tmp_path / "out" / "run42" / "manifest.json"
    report = _report()
    dump_manifest(report, nested)
    loaded = json.loads(nested.read_text(encoding="utf-8"))
    validate_manifest(loaded)


def test_dump_then_load_roundtrip_string_path(tmp_path: Path):
    """``str`` paths (not just ``Path``) must work end to end."""
    report = _report()
    fname = str(tmp_path / "r.json")
    dump_report(report, fname)
    loaded = load_report(fname)
    assert loaded["spectral"]["gap"] == report.spectral.gap
