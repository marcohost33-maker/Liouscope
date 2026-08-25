"""Repro fuer Issue #116 AM PRODUKTIVPFAD (nicht am rohen Baustein).

Ruft compute_relaxation_layer() mit DEFAULT-Argumenten auf und instrumentiert
die Naht relaxation.bca_ci, um zu messen, ob der Acceleration-Term jemals
geliefert wird. Zusaetzlich wird das echte a berechnet, das bca_ci intern
verwendet haette.
"""
import numpy as np
import liouscope.diagnostics.relaxation as R
from liouscope import build_liouvillian

calls = []
_orig = R.bca_ci

def spy(samples, theta_hat, *, alpha=0.05, jackknife_estimates=None):
    calls.append({
        "jackknife_supplied": jackknife_estimates is not None,
        "B": int(np.asarray(samples).shape[0]),
    })
    return _orig(samples, theta_hat, alpha=alpha, jackknife_estimates=jackknife_estimates)

R.bca_ci = spy

grid_seen = {}
_orig_evolve = R._evolve
def spy_evolve(L, rho, t):
    grid_seen["n_points"] = int(np.asarray(t).size)
    grid_seen["span"] = float(np.asarray(t)[-1])
    return _orig_evolve(L, rho, t)
R._evolve = spy_evolve

lower = np.array([[0, 1], [0, 0]], dtype=complex)
L = build_liouvillian(np.zeros((2, 2), dtype=complex), [lower], [1.0])
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho0 = np.outer(plus, plus.conj())

res = R.compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=10, seed=1)

print("default grid        : span %.1f, n_points %d" % (grid_seen["span"], grid_seen["n_points"]))
print("bca_ci calls        :", len(calls))
print("jackknife supplied  :", [c["jackknife_supplied"] for c in calls])
print("=> acceleration a   :", "0.0  (BC, NICHT BCa)" if not any(c["jackknife_supplied"] for c in calls) else "aus Jackknife (BCa)")
print("reported bca_ci_beta:", res.bca_ci_beta)
print("has attr interval_method:", hasattr(res, "interval_method"))
