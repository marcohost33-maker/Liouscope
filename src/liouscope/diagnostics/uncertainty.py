"""Uncertainty layer U: U0 (fit), U1 (solver), U2 (system size)."""

from __future__ import annotations

import numpy as np

from .._consts import U1_NOMINAL_FLOOR
from .._types import RelaxationResult, UncertaintyResult


def compute_uncertainty_layer(
    relaxation: RelaxationResult,
    *,
    solver_residual: float | None = None,
    size_residual: float | None = None,
    bootstrap_B: int = 200,
) -> UncertaintyResult:
    """Aggregate U0/U1/U2.

    U0 (fit) is derived from the BCa half-width of beta_D.
    U1 (solver) reports the nominal floor :data:`liouscope._consts.U1_NOMINAL_FLOOR`
    (a conservative placeholder, *not* a measured residual) unless the caller
    supplies ``solver_residual`` from an ODE-tolerance sweep. The sweep is a
    caller responsibility: ``diagnose()`` does not run one, so its U1 is the
    nominal floor by design (issue #71 B5).
    U2 (size) is ``None`` unless the caller supplies ``size_residual`` (e.g.
    from finite-size scaling); ``diagnose()`` leaves it unset.
    """
    lo, hi = relaxation.bca_ci_beta
    if np.isfinite(lo) and np.isfinite(hi):
        u0 = float(0.5 * (hi - lo))
    else:
        u0 = float("nan")
    u1 = float(solver_residual) if solver_residual is not None else U1_NOMINAL_FLOOR
    u2 = None if size_residual is None else float(size_residual)
    return UncertaintyResult(
        fit_uncertainty=u0,
        solver_uncertainty=u1,
        size_uncertainty=u2,
        bootstrap_B=bootstrap_B,
    )
