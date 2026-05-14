"""Regression tests against the golden benchmark JSON fixtures.

Each ``benchmarks/golden/BM-NNN.json`` snapshot was produced by
``python benchmarks/run.py BM-NNN``. The tests re-run the benchmark with
the same seed (writing through ``--output`` to a temp file, then loading
the JSON directly so we are not parsing stdout) and compare the result
against the golden file. Any drift means either:

  - a deliberate numerical change (then bump the golden file in the same
    PR as the code change), or
  - an unintended numerical regression (then investigate before merging).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
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

BENCHMARKS = ["BM-001", "BM-002", "BM-003", "BM-003b"]


def _run_benchmark(bm_id: str) -> dict:
    """Re-run ``benchmarks/run.py`` and read the canonical JSON payload.

    Writes through ``--output`` to a temp file so we never have to parse
    the runner's stdout (which mixes a "Running..." header with the JSON
    body and a trailing ``SHA-256:`` marker).
    """
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        out_path = Path(fh.name)
    try:
        proc = subprocess.run(
            [sys.executable, str(RUNNER), bm_id, "--output", str(out_path)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        if proc.returncode != 0:
            pytest.fail(
                f"runner failed for {bm_id} (rc={proc.returncode}):\n"
                f"--- stdout ---\n{proc.stdout[-2000:]}\n"
                f"--- stderr ---\n{proc.stderr[-2000:]}\n"
            )
        return json.loads(out_path.read_text(encoding="utf-8"))
    finally:
        out_path.unlink(missing_ok=True)


def _compare(observed, golden, path: str = "") -> None:
    """Recursive deep-compare with per-numeric tolerance.

    Note: bool is **not** treated as a numeric type even though
    ``isinstance(True, int)`` is True in Python. This prevents
    ``True``/``1`` and ``False``/``0`` from being silently accepted as
    equivalent in the comparison.
    """
    # Bool first, before int, because bool is a subclass of int.
    if isinstance(golden, bool) or isinstance(observed, bool):
        assert isinstance(observed, bool) and isinstance(golden, bool), (
            f"bool/int conflation at {path}: "
            f"{type(observed).__name__} vs {type(golden).__name__}"
        )
        assert observed == golden, f"bool mismatch at {path}: {observed} vs {golden}"
        return
    if isinstance(golden, dict):
        assert isinstance(observed, dict), (
            f"type mismatch at {path}: {type(observed).__name__} vs dict"
        )
        assert set(observed) == set(golden), (
            f"key set mismatch at {path}: {set(observed) ^ set(golden)}"
        )
        for k in golden:
            _compare(observed[k], golden[k], f"{path}.{k}" if path else k)
        return
    if isinstance(golden, list):
        assert isinstance(observed, list), (
            f"type mismatch at {path}: {type(observed).__name__} vs list"
        )
        assert len(observed) == len(golden), (
            f"length mismatch at {path}: {len(observed)} vs {len(golden)}"
        )
        for i, (o, g) in enumerate(zip(observed, golden, strict=True)):
            _compare(o, g, f"{path}[{i}]")
        return
    if isinstance(golden, float):
        assert isinstance(observed, (int, float)) and not isinstance(observed, bool), (
            f"type mismatch at {path}: {type(observed).__name__} vs float"
        )
        assert abs(float(observed) - golden) <= TOLERANCE, (
            f"float drift at {path}: {observed} vs golden {golden} "
            f"(|delta| > {TOLERANCE})"
        )
        return
    if isinstance(golden, int):
        assert isinstance(observed, int) and not isinstance(observed, bool), (
            f"type mismatch at {path}: {type(observed).__name__} vs int"
        )
        assert observed == golden, (
            f"int mismatch at {path}: {observed} vs {golden}"
        )
        return
    if isinstance(golden, str) or golden is None:
        assert observed == golden, (
            f"value mismatch at {path}: {observed!r} vs {golden!r}"
        )
        return
    pytest.fail(f"unsupported comparison type at {path}: {type(golden).__name__}")


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
