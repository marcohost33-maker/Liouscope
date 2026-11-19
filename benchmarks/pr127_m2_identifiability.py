"""Evaluate the pre-registered M2 identifiability criteria S1-S7.

Design, cases, seeds and criteria are fixed in
``docs/prereg/2026-11-08_m2_identifiability.md`` (committed before this
script). Prints one JSON document; the result note quotes it.

    python benchmarks/pr127_m2_identifiability.py > result.json
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics import relaxation as rel
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M2, initial_guess_m2

SEEDS = range(11, 21)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _chain1():
    L = build_liouvillian(0.3 * _SX, [math.sqrt(0.7) * _SM])
    psi = np.ones(2, dtype=complex) / math.sqrt(2.0)
    return rel.compute_relaxation_layer(L, rho_initial=np.outer(psi, psi.conj()), bootstrap_B=0)


def _curve(seed: int, beta2: float, a2: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, 80)
    y = np.exp(-0.5 * t) + a2 * np.exp(-beta2 * t) + 1.0e-3 * rng.standard_normal(t.size)
    return t, y


def _m2_record(t: np.ndarray, y: np.ndarray) -> dict[str, object]:
    fr, _ = rel._fit_with_model("M2", t, y)
    rate, model = rel._dominant_rate(t, y)
    return {
        "params": [float(v) for v in fr.params],
        "success": bool(fr.success),
        "identifiable": bool(fr.identifiable),
        "reason": fr.identification_reason,
        "logL": float(fr.log_likelihood),
        "aicc": float(fr.aicc),
        "selected_model": model,
        "selected_rate": float(rate),
    }


def _multistart(t: np.ndarray, y: np.ndarray, n: int = 50) -> list[dict[str, object]]:
    rng = np.random.default_rng(20261108)
    span, dt0 = float(t[-1] - t[0]), float(t[1] - t[0])
    lo, hi = math.log(5.0e-3 / span), math.log(10.0 / dt0)
    amp = 0.5 * float(np.max(np.abs(y)))
    out = []
    original = rel.initial_guess_m2
    try:
        for _ in range(n):
            b1, b2 = np.exp(rng.uniform(lo, hi, size=2))
            start = np.array([amp, b1, amp, b2])
            rel.initial_guess_m2 = lambda _t, _y, s=start: s.copy()
            fr, _ = rel._fit_with_model("M2", t, y)
            r = y - M2(t, fr.params)
            out.append({
                "start_rates": [float(b1), float(b2)],
                "params": [float(v) for v in fr.params],
                "success": bool(fr.success),
                "identifiable": bool(fr.identifiable),
                "cost": 0.5 * float(r @ r),
                "aicc": float(fr.aicc),
            })
    finally:
        rel.initial_guess_m2 = original
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    res: dict[str, object] = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    c0 = _chain1()
    t0 = np.asarray(c0.t_grid, dtype=float)
    y0 = np.asarray(c0.relative_entropy_curve, dtype=float)
    res["S1"] = {"aicc_model": c0.aicc_model, "beta_D": float(c0.beta_D),
                 "pass": c0.aicc_model == "M0"
                 and math.isclose(float(c0.beta_D), 0.753447, rel_tol=1e-6)}
    m2 = c0.fits["M2"]
    jac_gate = rel._m2_identification(
        t0, np.array([1.14268627, 0.721428703, 0.0697351193, 1.05517695e6]))
    # Amendment A1: S2 = the gate refuses the "jac" endpoint of _chain1.
    res["S2"] = {"params": [float(v) for v in m2.params], "identifiable": bool(m2.identifiable),
                 "reason": m2.identification_reason, "aicc": float(m2.aicc),
                 "n_eff": float(m2.n_eff), "gate_on_jac_endpoint": list(jac_gate),
                 "pass": (not jac_gate[0]) and "rate" in jac_gate[1]}
    # Amendment A1: C2 (beta2 = 50) descriptive, C2p (beta2 = 500) carries S4.
    cases = {"C1": (3.0, 0.5), "C2": (50.0, 0.5), "C2p": (500.0, 0.5), "C3": (3.0, 0.0)}
    for name, (b2, a2) in cases.items():
        res[name] = {str(s): _m2_record(*_curve(s, b2, a2)) for s in SEEDS}
    c1 = res["C1"].values()  # type: ignore[union-attr]
    c2 = res["C2p"].values()  # type: ignore[union-attr]
    c3 = res["C3"].values()  # type: ignore[union-attr]
    n_c1 = sum(1 for r in c1 if r["identifiable"] and r["selected_model"] == "M2")
    res["S3"] = {"identified_and_selected": n_c1, "of": len(SEEDS), "pass": n_c1 >= 9}
    n_c2_id = sum(1 for r in c2 if r["identifiable"])
    n_c2_sel = sum(1 for r in c2 if r["selected_model"] == "M2")
    res["S4"] = {"identified": n_c2_id, "selected": n_c2_sel,
                 "pass": n_c2_id == 0 and n_c2_sel == 0}
    counts: dict[str, int] = {}
    for r in c3:
        counts[str(r["selected_model"])] = counts.get(str(r["selected_model"]), 0) + 1
    res["S5"] = {"selected_counts": counts, "descriptive": True}
    t1, y1 = _curve(11, 3.0, 0.5)
    ref = fit_gls_ar1(M2, t1, y1, initial_guess_m2(t1, y1))
    worst = 0.0
    for c in (1.0e-40, 1.0e40):
        o = fit_gls_ar1(M2, t1, c * y1, initial_guess_m2(t1, c * y1))
        worst = max(worst, *(abs(o.params[i] / ref.params[i] - 1.0) for i in (1, 3)))
    res["S6"] = {"worst_rel_rate_diff": worst, "pass": worst <= 1e-9}
    ms0 = _multistart(t0, y0)
    m0_aicc = float(c0.fits["M0"].aicc)
    beat = sum(1 for r in ms0 if r["identifiable"] and r["aicc"] < m0_aicc)
    ms1 = _multistart(t1, y1)
    best = min(r["cost"] for r in ms1 if r["success"])
    good = sum(1 for r in ms1 if r["success"] and r["identifiable"]
               and math.isclose(r["cost"], best, rel_tol=1e-6))
    res["S7"] = {"C0_identified_beating_M0": beat,
                 "C0_identified_any": sum(1 for r in ms0 if r["identifiable"]),
                 "C1_good": good, "C1_best_cost": best,
                 "pass": beat == 0 and good >= 45}
    # numpy scalars (np.bool_, np.float64) -> Python scalars.
    print(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
