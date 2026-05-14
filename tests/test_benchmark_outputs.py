"""Regression tests against the golden benchmark JSON fixtures.

Each ``benchmarks/golden/BM-NNN.json`` snapshot was produced by
``python benchmarks/run.py BM-NNN``. The tests re-run the benchmark with
the same seed and compare the resulting JSON byte-for-byte (after
loading) against the golden file. Any drift means either:

  - a deliberate numerical change (then bump the golden file in the same
    PR as the code change), or
  - an unintended numerical regression (then investigate before merging).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = REPO_ROOT / "benchmarks" / "golden"
RUNNER = REPO_ROOT / "benchmarks" / "run.py"

# Acceptable absolute tolerance per numeric field. The runs are
# deterministic on identical hardware but cross-platform floating-point
# differences can shift the lowest-order bits. ``1e-9`` is tight enough
# to catch any meaningful drift.
TOLERANCE = 1.0e-9

BENCHMARKS = ["BM-001", "BM-003", "BM-003b"]


def _run_benchmark(bm_id: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(RUNNER), bm_id],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=True,
    )
    # The runner prints "Running ...", then the JSON, then "SHA-256: ...".
    # Find the JSON block by looking for the opening brace.
    lines = result.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("{"))
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "}")
    return json.loads("\n".join(lines[start : end + 1]))


def _compare(observed: dict, golden: dict, path: str = "") -> None:
    """Recursive deep-compare with per-numeric tolerance."""
    assert isinstance(observed, type(golden)) or (
        isinstance(observed, (int, float)) and isinstance(golden, (int, float))
    ), f"type mismatch at {path}: {type(observed)} vs {type(golden)}"

    if isinstance(golden, dict):
        assert set(observed) == set(golden), (
            f"key set mismatch at {path}: "
            f"{set(observed) ^ set(golden)}"
        )
        for k in golden:
            _compare(observed[k], golden[k], f"{path}.{k}" if path else k)
    elif isinstance(golden, list):
        assert len(observed) == len(golden), (
            f"length mismatch at {path}: {len(observed)} vs {len(golden)}"
        )
        for i, (o, g) in enumerate(zip(observed, golden, strict=True)):
            _compare(o, g, f"{path}[{i}]")
    elif isinstance(golden, float):
        assert abs(observed - golden) <= TOLERANCE, (
            f"float drift at {path}: {observed} vs golden {golden} "
            f"(|delta| > {TOLERANCE})"
        )
    elif isinstance(golden, (int, str, bool)) or golden is None:
        assert observed == golden, (
            f"value mismatch at {path}: {observed!r} vs {golden!r}"
        )


@pytest.mark.slow
@pytest.mark.parametrize("bm_id", BENCHMARKS)
def test_benchmark_matches_golden(bm_id):
    """Re-run BM-NNN and compare against the committed golden fixture."""
    golden_path = GOLDEN_DIR / f"{bm_id}.json"
    if not golden_path.exists():
        pytest.skip(f"golden fixture missing: {golden_path}")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    observed = _run_benchmark(bm_id)
    _compare(observed, golden)
