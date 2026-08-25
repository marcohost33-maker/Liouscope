"""Repro fuer PR #115 AM PRODUKTIVPFAD: Ratenskalen-Abhaengigkeit der Default-Zeitgitters.

Reine Umskalierung L -> c*L ist dieselbe Physik in einer anderen Zeiteinheit.
beta_D/c und beta_D_linear/c MUESSEN daher invariant sein.
"""
import warnings
import numpy as np
from liouscope import build_liouvillian
from liouscope.diagnostics.relaxation import compute_relaxation_layer

lower = np.array([[0, 1], [0, 0]], dtype=complex)
L0 = build_liouvillian(np.zeros((2, 2), dtype=complex), [lower], [1.0])
plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
rho0 = np.outer(plus, plus.conj())

print(f"{'c':>10} {'beta_D/c':>14} {'beta_lin/c':>16} {'winner':>8} {'t_grid_source':>14}")
vals = []
for c in (1e2, 1e0, 1e-2, 1e-4, 1e-6):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = compute_relaxation_layer(c * L0, rho_initial=rho0, bootstrap_B=10, seed=1)
    src = getattr(r, "t_grid_source", "<Feld fehlt>")
    print(f"{c:10.0e} {r.beta_D/c:14.6f} {getattr(r,'beta_D_linear',float('nan'))/c:16.4f} {r.aicc_model:>8} {src:>14}")
    vals.append(r.beta_D / c)

lo, hi = min(vals), max(vals)
print(f"\nbeta_D/c Spanne: {lo:.6f} .. {hi:.6f}   relative Drift = {(hi-lo)/lo:.3%}")
print("INVARIANT" if (hi - lo) / lo < 1e-2 else "NICHT INVARIANT (Defekt aus #115 reproduziert)")
