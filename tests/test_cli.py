"""Tests for the ``python -m liouscope`` command-line entry point."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from liouscope import build_liouvillian


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "liouscope", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_version():
    result = _run(["version"])
    assert result.returncode == 0
    assert "liouscope" in result.stdout
    assert "A1-A12-v3.1" in result.stdout


def test_cli_info():
    result = _run(["info"])
    assert result.returncode == 0
    assert "Mechanism classes" in result.stdout
    assert "A11" in result.stdout
    assert "F4" in result.stdout


def test_cli_diagnose_dephased_qubit(pauli, tmp_path: Path):
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    L_path = tmp_path / "L.npy"
    np.save(L_path, L)

    out_path = tmp_path / "report.json"
    result = _run(["diagnose", str(L_path), "--no-mpemba", "--bootstrap-B", "10",
                    "--seed", "42", "--output", str(out_path)])
    assert result.returncode == 0, result.stderr
    assert "Run ID" in result.stdout
    assert "D1 gap" in result.stdout

    obj = json.loads(out_path.read_text())
    assert "spectral" in obj
    assert obj["classification"]["taxonomy_version"] == "A1-A12-v3.1"


def test_cli_diagnose_missing_file_returns_2(tmp_path: Path):
    result = _run(["diagnose", str(tmp_path / "does_not_exist.npy")])
    assert result.returncode == 2
    assert "not found" in result.stderr


def test_cli_diagnose_rejects_non_square(tmp_path: Path):
    bad = np.zeros((4, 5), dtype=complex)
    path = tmp_path / "bad.npy"
    np.save(path, bad)
    result = _run(["diagnose", str(path)])
    assert result.returncode == 2
    assert "square" in result.stderr


def test_cli_invalid_subcommand_returns_nonzero():
    result = _run(["nonexistent-subcommand"])
    assert result.returncode != 0
