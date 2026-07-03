"""Top-level :func:`diagnose` orchestrator."""

from __future__ import annotations

import numpy as np

from ._types import DiagnosticReport, MpembaResult
from .core.lindblad import steady_state
from .diagnostics.classification import classify_mechanism
from .diagnostics.lep import compute_lep_layer
from .diagnostics.mpemba import compute_mpemba_layer
from .diagnostics.nonnormality import compute_nonnormality_layer
from .diagnostics.relaxation import compute_relaxation_layer
from .diagnostics.resolvent import compute_resolvent_layer
from .diagnostics.spectral import compute_spectral_layer
from .diagnostics.transient import compute_transient_layer
from .diagnostics.uncertainty import compute_uncertainty_layer
from .io.manifest import build_manifest, compute_input_hash
from .numerics.linalg import require_finite_square_2d


_VALID_SOLVER_PATHS = {"dense", "sparse_arpack"}


def _validate_solver_path(solver_path: str) -> None:
    """Fail closed until non-dense orchestrator paths are genuinely wired.

    ``solver_path`` is part of the public governance/manifest surface, so a
    caller must never be allowed to request one execution path while receiving a
    different one silently. Today the top-level pipeline is dense-only; the
    low-level ``liouscope.sparse`` helpers exist but are not integrated into
    :func:`diagnose` yet.
    """
    if solver_path not in _VALID_SOLVER_PATHS:
        allowed = ", ".join(sorted(_VALID_SOLVER_PATHS))
        raise ValueError(f"solver_path must be one of {{{allowed}}}, got {solver_path!r}")
    if solver_path == "sparse_arpack":
        raise NotImplementedError(
            "solver_path='sparse_arpack' is reserved: diagnose() currently runs the "
            "dense pipeline only. Use liouscope.sparse low-level helpers directly "
            "or keep solver_path='dense' until the sparse orchestrator path is wired."
        )


def diagnose(
    L_super: np.ndarray,
    *,
    rho_initial: np.ndarray | None = None,
    rho_steady_state: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    include_mpemba: bool = True,
    bootstrap_B: int = 200,
    seed: int = 42,
    solver_path: str = "dense",
) -> DiagnosticReport:
    """Run the full six-layer multi-diagnostic pipeline on a Liouvillian.

    Parameters
    ----------
    L_super
        ``d^2 x d^2`` column-stacking superoperator.
    rho_initial
        Optional initial state. Defaults to maximally-mixed.
    rho_steady_state
        Optional pre-computed steady state.
    t_grid
        Time grid for the relaxation layer.
    include_mpemba
        Compute D19/D20.
    bootstrap_B
        Number of parametric bootstrap resamples.
    seed
        PRNG seed for any stochastic step (jackknife, bootstrap, Haar).
    solver_path
        ``"dense"`` (default). ``"sparse_arpack"`` is a reserved manifest value
        and currently raises ``NotImplementedError`` rather than silently running
        the dense path.

    Returns
    -------
    DiagnosticReport
        A fully-populated frozen report with governance metadata.
    """
    _validate_solver_path(solver_path)
    # Fail-closed boundary guard: reject non-finite / non-square operators here
    # with a structured, argument-named error instead of letting NaN/inf flow
    # into scipy.linalg.expm / svd and surface as an opaque LAPACK message.
    L_super = require_finite_square_2d(L_super, name="L_super")
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    if d * d != n2:
        raise ValueError(f"L_super must have square-d dimension, got {n2}")

    if rho_initial is not None:
        rho_initial = require_finite_square_2d(rho_initial, name="rho_initial")
        if rho_initial.shape != (d, d):
            raise ValueError(
                f"rho_initial shape {rho_initial.shape} != ({d}, {d})"
            )
    if rho_steady_state is not None:
        rho_steady_state = require_finite_square_2d(
            rho_steady_state, name="rho_steady_state"
        )
        if rho_steady_state.shape != (d, d):
            raise ValueError(
                f"rho_steady_state shape {rho_steady_state.shape} != ({d}, {d})"
            )

    if rho_steady_state is None:
        rho_steady_state = steady_state(L_super)
    if rho_initial is None:
        rho_initial = np.eye(d, dtype=complex) / d

    spectral = compute_spectral_layer(L_super, rho_steady_state)
    nonnorm = compute_nonnormality_layer(L_super)
    resolvent = compute_resolvent_layer(L_super)
    relaxation = compute_relaxation_layer(
        L_super,
        rho_initial=rho_initial,
        rho_steady_state=rho_steady_state,
        t_grid=t_grid,
        bootstrap_B=bootstrap_B,
        seed=seed,
    )
    transient = compute_transient_layer(L_super, spectral.gap)
    lep = compute_lep_layer(
        L_super,
        spectral.eigenvalues,
        beta_D=relaxation.beta_D,
        gap=spectral.gap,
        rho_steady_state=rho_steady_state,
        seed=seed,
    )
    mpemba: MpembaResult | None = None
    if include_mpemba:
        mpemba = compute_mpemba_layer(L_super, rho_initial)

    classification = classify_mechanism(
        spectral=spectral,
        nonnorm=nonnorm,
        relaxation=relaxation,
        resolvent=resolvent,
        transient=transient,
        lep=lep,
        mpemba=mpemba,
    )
    uncertainty = compute_uncertainty_layer(
        relaxation,
        solver_residual=None,
        size_residual=None,
        bootstrap_B=bootstrap_B,
    )
    input_hash = compute_input_hash(L_super, rho_initial)
    governance = build_manifest(
        input_hash=input_hash,
        seed=seed,
        solver_path=solver_path,  # type: ignore[arg-type]
        tier=classification.tier,
    )
    return DiagnosticReport(
        spectral=spectral,
        nonnorm=nonnorm,
        relaxation=relaxation,
        resolvent=resolvent,
        transient=transient,
        lep=lep,
        uncertainty=uncertainty,
        classification=classification,
        governance=governance,
        mpemba=mpemba,
    )


__all__ = ["diagnose"]
