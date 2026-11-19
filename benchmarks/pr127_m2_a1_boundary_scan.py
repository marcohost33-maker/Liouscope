"""Boundary scan for amendment A1 of the M2 identifiability gate (PR #127).

Released by owner decision 2026-11-16 (E1116-11): an M2 rate counts as
identified only if ``beta * dt_0 <= 10`` (``relaxation._M2_MAX_RATE_STEPS``).
This scan asks the two questions that decision rests on, near the bound:

1. Is a CORRECT M2 fit ever refused inside the bound?
2. Does a refusal ever move the selection to a model with a wrong rate
   (the P1-1 failure), instead of ``INDETERMINATE``?

plus a metamorphic check that the gate verdict is invariant under a change of
time unit (``t -> c t``, rates ``-> rates / c``), i.e. that the bound is
dimensionally right.

Curve ``exp(-0.5 t) + 0.5 exp(-b t) + sigma * N(0, 1)`` on
``linspace(0, 10, 80)`` (``dt_0 = 0.1266``), ``b * dt_0`` in
{4, 6, 8, 9, 9.5, 9.9, 10.1, 10.5, 12, 15}, ``sigma`` in {0, 1e-6, 1e-4, 1e-3},
seeds 0..5 (one realisation at ``sigma = 0``). "Correct" = slow rate within
2 %, fast rate within 5 % of the truth. Deterministic; prints one JSON
document.

    python benchmarks/pr127_m2_a1_boundary_scan.py > result.json
"""

from __future__ import annotations

import json
import math
import platform
import sys
import warnings

import numpy as np
import scipy

from liouscope.diagnostics import relaxation as rx

_STEPS = (4.0, 6.0, 8.0, 9.0, 9.5, 9.9, 10.1, 10.5, 12.0, 15.0)
_SIGMAS = (0.0, 1.0e-6, 1.0e-4, 1.0e-3)
_UNIT = 1.0e6


def _case(t: np.ndarray, b2: float, sigma: float, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    y = np.exp(-0.5 * t) + 0.5 * np.exp(-b2 * t) + sigma * rng.standard_normal(t.size)
    fits = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("M0", "M1", "M2", "M3a", "M3b"):
            try:
                fits[name], _ = rx._fit_with_model(name, t, y)
            except (ValueError, RuntimeError):
                continue
        # The SHIPPED rounding policy (PR #193 review): ``data_error`` bounds
        # the numerical formation error of the curve, not its stochastic
        # noise. ``_dominant_rate`` takes it from ``_curve_data_error``; the
        # first version of this scan passed ``sigma`` and so widened every
        # likelihood band by 10-13 orders of magnitude.
        data_error = rx._curve_data_error(y, d=1, formation_scale=0.0)
        winner, _tied = rx._select_model(fits, t, y, data_error)
    m2 = fits.get("M2")
    p = m2.params if m2 is not None else np.full(4, np.nan)
    rates = sorted([float(p[1]), float(p[3])])
    correct = bool(
        m2 is not None
        and m2.success
        and math.isclose(rates[0], 0.5, rel_tol=0.02)
        and math.isclose(rates[1], b2, rel_tol=0.05)
    )
    unit_invariant = None
    if m2 is not None and np.all(np.isfinite(p)):
        q = p.copy()
        q[1] /= _UNIT
        q[3] /= _UNIT
        unit_invariant = (
            rx._m2_identification(t * _UNIT, q)[0] == rx._m2_identification(t, p)[0]
        )
    return {
        "steps": float(b2 * float(t[1] - t[0])),
        "sigma": sigma,
        "seed": seed,
        "m2_success": bool(m2 is not None and m2.success),
        "m2_identified": bool(m2 is not None and m2.identifiable),
        "rates": rates,
        "correct": correct,
        "winner": winner,
        "unit_invariant": unit_invariant,
    }


def main() -> None:
    t = np.linspace(0.0, 10.0, 80)
    dt0 = float(t[1] - t[0])
    rows = []
    for steps in _STEPS:
        for sigma in _SIGMAS:
            for seed in range(6 if sigma else 1):
                rows.append(_case(t, steps / dt0, sigma, seed))
    m2_winners = [r for r in rows if r["winner"] == "M2"]
    summary = {
        "cases": len(rows),
        "correct_refused_inside_bound": sum(
            1 for r in rows if r["correct"] and not r["m2_identified"] and r["steps"] <= 10.0
        ),
        "correct_refused_outside_bound": sum(
            1 for r in rows if r["correct"] and not r["m2_identified"] and r["steps"] > 10.0
        ),
        "fallback_to_other_model": sum(
            1 for r in rows if r["winner"] in ("M0", "M1", "M3a", "M3b")
        ),
        "indeterminate": sum(1 for r in rows if r["winner"] == rx.MODEL_SELECTION_INDETERMINATE),
        "m2_winner_slow_rate_range": [
            min(r["rates"][0] for r in m2_winners),
            max(r["rates"][0] for r in m2_winners),
        ],
        "unit_noninvariant": sum(1 for r in rows if r["unit_invariant"] is False),
    }
    json.dump(
        {
            "env": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "platform": platform.platform(),
            },
            "bound_steps": rx._M2_MAX_RATE_STEPS,
            "dt0": dt0,
            "summary": summary,
            "rows": rows,
        },
        sys.stdout,
        indent=1,
        default=str,
    )


if __name__ == "__main__":
    main()
