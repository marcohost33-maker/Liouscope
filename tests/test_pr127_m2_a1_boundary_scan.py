"""The A1 boundary scan exercises the SHIPPED selection policy (PR #193 review).

``benchmarks/pr127_m2_a1_boundary_scan.py`` supports owner decision E1116-11
with selection and withholding counts. Its first version passed the simulated
noise ``sigma`` to ``_select_model`` as ``data_error``. That argument bounds the
numerical FORMATION error of the curve; ``_dominant_rate`` takes it from
``_curve_data_error`` (~1e-16 for these curves), so the scan widened every
likelihood band by 10-13 orders of magnitude and counted a different policy.

Oracle: the production path itself. Both the scan and ``_dominant_rate`` are run
on the same curve with ``_select_model`` wrapped by a recorder; the data error
and the winner must be identical.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from liouscope.diagnostics import relaxation as rx

_SCAN = Path(__file__).resolve().parent.parent / "benchmarks" / "pr127_m2_a1_boundary_scan.py"


@cache
def _scan() -> Any:
    spec = importlib.util.spec_from_file_location("pr127_m2_a1_boundary_scan", _SCAN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["pr127_m2_a1_boundary_scan"] = module
    spec.loader.exec_module(module)
    return module


def _recording(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    seen: list[float] = []
    original = rx._select_model

    def spy(fits, t, y, data_error):  # type: ignore[no-untyped-def]
        seen.append(float(data_error))
        return original(fits, t, y, data_error)

    monkeypatch.setattr(rx, "_select_model", spy)
    return seen


# Noisy cases on both sides of the bound (9 and 12 steps of dt_0), where
# sigma = 1e-3 differs from the production level by ~13 orders of magnitude.
@pytest.mark.parametrize(("steps", "seed"), [(9.0, 0), (12.0, 3)])
def test_scan_selects_with_the_production_data_error(
    monkeypatch: pytest.MonkeyPatch, steps: float, seed: int
) -> None:
    scan = _scan()
    sigma = 1.0e-3
    t = np.linspace(0.0, 10.0, 80)
    b2 = steps / float(t[1] - t[0])
    seen = _recording(monkeypatch)

    row = scan._case(t, b2, sigma, seed)
    assert len(seen) == 1
    scan_error = seen.pop()

    rng = np.random.default_rng(seed)
    y = np.exp(-0.5 * t) + 0.5 * np.exp(-b2 * t) + sigma * rng.standard_normal(t.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _rate, winner = rx._dominant_rate(t, y)
    assert len(seen) == 1
    production_error = seen.pop()

    assert scan_error == production_error
    assert production_error == rx._curve_data_error(y, d=1, formation_scale=0.0)
    # discriminating: the policy the first scan ran is ~1e13 times wider
    assert scan_error < 1.0e-10 * sigma
    assert row["winner"] == winner
