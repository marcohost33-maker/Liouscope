"""Tests for IO/manifest layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from liouscope import build_liouvillian, diagnose, seed_everything
from liouscope.io import compute_input_hash, dump_report, load_report, make_run_id


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
