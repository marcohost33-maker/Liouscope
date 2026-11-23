"""CAR(1) AICc sample size and rate identifiability (PR #127, review LIO-02).

PRE-REGISTERED. Parameters, endpoints and the decision rule below were fixed
and committed BEFORE the first run; the result file records this script's
SHA-256 so a later edit is visible.

Question
--------
On a NON-uniform grid ``relaxation._fit_with_model`` counts ``k = p + 2``
(mean-function parameters + CAR(1) rate + stationary variance) and uses the
OBSERVED count ``n_obs`` as the AICc sample size, while a uniform grid keeps
``k = p`` and the effective sample size ``N_eff``. The cross-family review
asked for a documented comparison/identifiability study, "insbesondere bei
theta = 0.05", and for model selection to support INDETERMINATE when the
CAR(1) rate is not identifiable.

Design (fixed)
--------------
* seeds ``0..39`` (``numpy.random.default_rng(seed)``), one noise draw each;
* truths: ``M0`` ``y = exp(-t)``; ``M2`` ``y = 0.6 exp(-t) + 0.4 exp(-20 t)``
  (slow rate = gap = 1, fast rate 20);
* noise: stationary CAR(1), standard deviation ``sigma = 1e-3``, correlation
  ``exp(-theta dt)``, ``theta`` in ``{0.05, 0.5, 2.0}`` (units of the gap);
* grids: ``uniform`` = ``default_relaxation_grid(1.0)`` (80 points on
  ``[0, 10]``) and ``two_scale`` = ``default_relaxation_grid(1.0,
  fast_rate=20.0)`` (non-uniform, the PR's default for a fast mode);
* every candidate M0..M3b fitted ONCE with the shipped
  ``relaxation._fit_with_model``; from its log-likelihood the AICc is formed
  under both sample sizes, ``N_eff`` (as the fit reports it) and ``n_obs``,
  with the shipped ``k``. Selection = smallest finite AICc among succeeded
  fits. The shipped selection (``relaxation._select_model``, rounding tie
  policy, data error = representation level of the curve) is recorded too.

Endpoints (fixed)
-----------------
* E1 correct selection: the truth model is selected, per (grid, truth,
  theta) cell and criterion, out of 40.
* E2 rate identifiability: the CAR(1) rate fitted with the TRUE model is
  finite and within a factor 2 of ``theta``, out of 40 (two_scale only; the
  uniform grid fits AR(1)).
* E3 statistical near-tie: best and second-best AICc (shipped criterion)
  within 2 units, out of 40 -- informational.
* E4 INDETERMINATE count of the shipped selection -- informational.

Decision rule (fixed)
---------------------
* The ``n_obs`` choice on non-uniform grids is SUPPORTED if on ``two_scale``
  E1(n_obs) >= E1(N_eff) in at least 5 of the 6 (truth, theta) cells and is
  never lower by more than 5. Otherwise it is NOT SUPPORTED and the method
  is reported as open, unchanged.
* A cell with E2 < 20/40 is reported as "CAR(1) rate not identifiable on this
  window". The study does not by itself introduce a statistical INDETERMINATE
  rule; it reports E3 so that decision can be made on numbers.

Run::

    python benchmarks/pr127_car1_aicc_identifiability.py \
        --out benchmarks/results/pr127_car1_aicc_identifiability.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy

from liouscope._version import __version__
from liouscope.diagnostics import relaxation as rx
from liouscope.diagnostics.relaxation import default_relaxation_grid
from liouscope.fitting.aicc import aicc
from liouscope.fitting.car1 import is_uniform_grid

SEEDS = tuple(range(40))
THETAS = (0.05, 0.5, 2.0)
SIGMA = 1.0e-3
FAST_RATE = 20.0
MODELS = ("M0", "M1", "M2", "M3a", "M3b")


def truth_curve(name: str, t: np.ndarray) -> np.ndarray:
    if name == "M0":
        return np.exp(-t)
    if name == "M2":
        return 0.6 * np.exp(-t) + 0.4 * np.exp(-FAST_RATE * t)
    raise ValueError(name)


def car1_noise(t: np.ndarray, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Exact stationary CAR(1) draw on an arbitrary increasing grid."""
    z = rng.standard_normal(t.size)
    e = np.empty(t.size)
    e[0] = sigma * z[0]
    a = np.exp(-theta * np.diff(t))
    for k in range(1, t.size):
        e[k] = a[k - 1] * e[k - 1] + sigma * math.sqrt(1.0 - a[k - 1] ** 2) * z[k]
    return e


def select(scores: dict[str, float]) -> str:
    finite = {k: v for k, v in scores.items() if math.isfinite(v)}
    return min(finite, key=finite.__getitem__) if finite else "none"


def run() -> dict:
    grids = {
        "uniform": default_relaxation_grid(1.0),
        "two_scale": default_relaxation_grid(1.0, fast_rate=FAST_RATE),
    }
    assert is_uniform_grid(grids["uniform"]) and not is_uniform_grid(grids["two_scale"])
    u = 0.5 * float(np.finfo(float).eps)
    cells = []
    for grid_name, t in grids.items():
        uniform = is_uniform_grid(t)
        for truth in ("M0", "M2"):
            for theta in THETAS:
                sel_neff, sel_nobs, sel_shipped = [], [], []
                rate_ok = 0
                near_tie = 0
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    y = truth_curve(truth, t) + car1_noise(t, theta, SIGMA, rng)
                    fits = {}
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for m in MODELS:
                            try:
                                fits[m], _ = rx._fit_with_model(m, t, y)
                            except (ValueError, RuntimeError):
                                continue
                        shipped, _tied = rx._select_model(
                            fits, t, y, u * float(np.max(np.abs(y)))
                        )
                    s_neff, s_nobs, s_ship = {}, {}, {}
                    for m, fr in fits.items():
                        if not fr.success:
                            continue
                        k = int(fr.params.size) + (0 if uniform else 2)
                        s_neff[m] = aicc(fr.log_likelihood, k, fr.n_eff)
                        s_nobs[m] = aicc(fr.log_likelihood, k, float(t.size))
                        s_ship[m] = fr.aicc
                    sel_neff.append(select(s_neff))
                    sel_nobs.append(select(s_nobs))
                    sel_shipped.append(shipped)
                    ranked = sorted(v for v in s_ship.values() if math.isfinite(v))
                    if len(ranked) >= 2 and ranked[1] - ranked[0] < 2.0:
                        near_tie += 1
                    th = float(fits[truth].residual_theta_car1) if truth in fits else float("nan")
                    if math.isfinite(th) and th > 0.0 and abs(math.log(th / theta)) < math.log(2.0):
                        rate_ok += 1

                def counts(sel: list[str]) -> dict[str, int]:
                    return {m: sel.count(m) for m in sorted(set(sel))}

                cells.append({
                    "grid": grid_name,
                    "truth": truth,
                    "theta": theta,
                    "E1_correct_neff": sel_neff.count(truth),
                    "E1_correct_nobs": sel_nobs.count(truth),
                    "E1_correct_shipped": sel_shipped.count(truth),
                    "E2_rate_within_factor2": rate_ok if not uniform else None,
                    "E3_near_tie_lt2": near_tie,
                    "E4_indeterminate_shipped": sel_shipped.count("INDETERMINATE"),
                    "selected_neff": counts(sel_neff),
                    "selected_nobs": counts(sel_nobs),
                    "selected_shipped": counts(sel_shipped),
                })
                c = cells[-1]
                print(f"{grid_name:9s} {truth} theta={theta:<5} E1 neff={c['E1_correct_neff']:2d} "
                      f"nobs={c['E1_correct_nobs']:2d} shipped={c['E1_correct_shipped']:2d} "
                      f"E2={c['E2_rate_within_factor2']} E3={near_tie} E4={c['E4_indeterminate_shipped']}",
                      flush=True)
    two = [c for c in cells if c["grid"] == "two_scale"]
    not_worse = sum(c["E1_correct_nobs"] >= c["E1_correct_neff"] for c in two)
    worst_drop = max(c["E1_correct_neff"] - c["E1_correct_nobs"] for c in two)
    verdict = "SUPPORTED" if (not_worse >= 5 and worst_drop <= 5) else "NOT_SUPPORTED"
    non_identifiable = [
        {"truth": c["truth"], "theta": c["theta"]}
        for c in two
        if c["E2_rate_within_factor2"] is not None and c["E2_rate_within_factor2"] < 20
    ]
    return {
        "cells": cells,
        "decision": {
            "two_scale_cells_nobs_not_worse": not_worse,
            "two_scale_worst_drop_nobs_vs_neff": worst_drop,
            "nobs_on_nonuniform": verdict,
            "car1_rate_not_identifiable_cells": non_identifiable,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = run()
    payload = {
        "study": "pr127_car1_aicc_identifiability",
        "script_sha256": script_sha,
        "liouscope_version": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "parameters": {
            "seeds": [SEEDS[0], SEEDS[-1]],
            "thetas": list(THETAS),
            "sigma": SIGMA,
            "fast_rate": FAST_RATE,
            "truths": ["M0", "M2"],
            "grids": ["uniform", "two_scale"],
        },
        **result,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("decision:", json.dumps(result["decision"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
