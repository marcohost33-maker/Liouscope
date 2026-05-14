"""Performance regression: ``reproduce_paper.py`` <= 30 s on a modern laptop.

The spec (v2.0 consolidated report, Teil 16.5) records the canonical
``reproduce_paper.py`` wall time as ~11 s for the V1-V5 sweep with
``d <= 128``. We hold the regression line at 30 s on whatever CI runner
we happen to be on, which is generous enough to absorb CI noise while
still flagging a clear slowdown.

The test is marked ``slow`` so it can be deselected for fast local runs::

    pytest -m "not slow"
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "benchmarks" / "reproduce_paper.py"

# Walltime ceiling, in seconds. Bump only with a CHANGELOG entry explaining
# the cause; otherwise the regression must be investigated.
WALL_BUDGET_SECONDS = 30.0


@pytest.mark.slow
def test_reproduce_paper_under_30_seconds():
    """``reproduce_paper.py`` must finish under 30 seconds.

    Asserts both:
      * exit code 0
      * walltime <= WALL_BUDGET_SECONDS
    """
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=WALL_BUDGET_SECONDS * 2,  # absolute kill switch
    )
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, (
        f"reproduce_paper.py exited with code {result.returncode}\n"
        f"stderr tail:\n{result.stderr[-1000:]}"
    )
    assert elapsed <= WALL_BUDGET_SECONDS, (
        f"reproduce_paper.py took {elapsed:.2f}s, budget is "
        f"{WALL_BUDGET_SECONDS:.0f}s. Investigate the slowdown or, if "
        f"intentional, bump WALL_BUDGET_SECONDS with a CHANGELOG entry."
    )
