"""Relaxation layer R: D5 VNE, D6 rel-entropy, D7 fidelity, D7b ent. asym.,
plus the M0-M3b fit hierarchy orchestrated through AICc with N_eff.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_SUPP
from .._types import FitResult, RelaxationResult
from ..core.lindblad import steady_state
from ..fitting.aicc import aicc, choose_model
from ..fitting.bootstrap import _jackknife, bca_ci, parametric_bootstrap
from ..fitting.gls import fit_gls_ar1
from ..fitting.models import (
    M0,
    M1,
    M2,
    M3a,
    M3b,
    initial_guess_m0,
    initial_guess_m1,
    initial_guess_m2,
    initial_guess_m3a,
)
from ..fitting.neff import estimate_neff_geyer
from ..fitting.prony import prony_seed
from ..numerics.kronecker import unvec, vec
from ..numerics.linalg import support_check


def _evolve(L_super: np.ndarray, rho0: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Propagate ``rho(t) = expm(L t) rho_0`` and return the trajectory."""
    rho_vec0 = vec(rho0)
    d = rho0.shape[0]
    traj = np.empty((t_grid.size, d, d), dtype=complex)
    for k, t in enumerate(t_grid):
        if t == 0.0:
            traj[k] = rho0
        else:
            rho_vec_t = sla.expm(L_super * t) @ rho_vec0
            traj[k] = unvec(rho_vec_t, d=d)
    return traj


def von_neumann_entropy(rho: np.ndarray) -> float:
    """D5: ``S(rho) = -Tr(rho log rho)``."""
    rho = np.asarray(rho)
    rho_sym = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rho_sym)
    evals = evals[evals > EPS_SUPP]
    if evals.size == 0:
        return 0.0
    return float(-np.sum(evals * np.log(evals)))


def relative_entropy(rho: np.ndarray, pi: np.ndarray) -> float:
    """D6: ``D(rho || pi) = Tr(rho (log rho - log pi))``.

    Uses ``support_check`` to regularise (anchor J).
    """
    rho = np.asarray(rho)
    pi = np.asarray(pi)
    _, pi_reg = support_check(rho, pi)
    rho_sym = 0.5 * (rho + rho.conj().T)
    pi_sym = 0.5 * (pi_reg + pi_reg.conj().T)
    evals_rho, evecs_rho = np.linalg.eigh(rho_sym)
    evals_pi, evecs_pi = np.linalg.eigh(pi_sym)
    log_pi = (
        evecs_pi
        @ np.diag(np.log(np.clip(evals_pi, EPS_SUPP, None)))
        @ evecs_pi.conj().T
    )
    log_rho = (
        evecs_rho
        @ np.diag(np.log(np.clip(evals_rho, EPS_SUPP, None)))
        @ evecs_rho.conj().T
    )
    val = float(np.real(np.trace(rho_sym @ (log_rho - log_pi))))
    return max(val, 0.0)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """D7: Uhlmann fidelity ``F = Tr sqrt( sqrt(rho) sigma sqrt(rho) )``.

    Computes ``sqrt(rho)`` via spectral decomposition so rank-deficient
    inputs (pure-state limits) do not trigger ``sqrtm`` warnings.
    """
    rho = np.asarray(rho)
    rho_sym = 0.5 * (rho + rho.conj().T)
    evals, evecs = np.linalg.eigh(rho_sym)
    evals = np.clip(evals.real, 0.0, None)
    sqrt_evals = np.sqrt(evals)
    rho_sqrt = (evecs * sqrt_evals) @ evecs.conj().T
    inner = rho_sqrt @ np.asarray(sigma) @ rho_sqrt
    eigs = np.linalg.eigvalsh(0.5 * (inner + inner.conj().T))
    eigs = np.clip(eigs.real, 0.0, None)
    return float(np.sum(np.sqrt(eigs)))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """D_tr: half the trace-norm distance ``(1/2) || rho - sigma ||_1`` (LIOU-F-018).

    For Hermitian inputs this equals ``(1/2) sum_i |lambda_i|`` over the
    eigenvalues of ``rho - sigma`` (Nielsen & Chuang). It is the observable
    relaxation metric complementing D5 (von-Neumann entropy), D6 (relative
    entropy) and D7 (Uhlmann fidelity): a metric (``0 <= D_tr <= 1`` for density
    operators, ``D_tr = 0`` iff ``rho == sigma``) that contracts monotonically
    under any CPTP map (data-processing inequality) and upper-bounds the
    distinguishability of ``rho`` and ``sigma`` by measurement (Helstrom).

    The difference is Hermitised before the eigenvalue decomposition so a tiny
    non-Hermitian numerical excursion in either argument cannot leak a complex
    part into the (real, by construction) distance.
    """
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    if rho.shape != sigma.shape:
        raise ValueError(
            f"trace_distance requires equal shapes, got {rho.shape} vs {sigma.shape}"
        )
    diff = rho - sigma
    diff_sym = 0.5 * (diff + diff.conj().T)
    evals = np.linalg.eigvalsh(diff_sym)
    return float(0.5 * np.sum(np.abs(evals)))


def entanglement_asymmetry(rho: np.ndarray) -> float:
    """D7b: Rylands et al. 2024 entanglement-asymmetry measure (single block).

    For a 2-site reduced state this returns the SU(2) symmetry-breaking
    Delta S_A defined via the symmetrised state. Falls back to 0 outside
    the supported 2-qubit case.
    """
    rho = np.asarray(rho)
    d = rho.shape[0]
    if d not in (4,):
        return float("nan")
    # Symmetrise by averaging over Pauli twirl on second qubit.
    paulis = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    rho_sym = np.zeros_like(rho)
    for P in paulis:
        op = np.kron(np.eye(2, dtype=complex), P)
        rho_sym += op @ rho @ op.conj().T
    rho_sym /= 4.0
    return float(max(von_neumann_entropy(rho_sym) - von_neumann_entropy(rho), 0.0))


def _fit_with_model(
    model_name: str,
    t: np.ndarray,
    y: np.ndarray,
) -> tuple[FitResult, np.ndarray]:
    fns = {"M0": M0, "M1": M1, "M2": M2, "M3a": M3a, "M3b": M3b}
    seeds = {
        "M0": initial_guess_m0,
        "M1": initial_guess_m1,
        "M2": initial_guess_m2,
        "M3a": initial_guess_m3a,
        "M3b": lambda t_, y_: np.asarray(prony_seed(t_, y_)),
    }
    model = fns[model_name]
    p0 = seeds[model_name](t, y)
    fit = fit_gls_ar1(model, t, y, p0)
    k = p0.size
    n_eff = estimate_neff_geyer(fit.residuals)
    aic = aicc(fit.log_likelihood, k, n_eff)
    fit_result = FitResult(
        model=model_name,
        params=fit.params,
        log_likelihood=fit.log_likelihood,
        aicc=aic,
        n_eff=n_eff,
        residual_ar1_rho=fit.rho_ar1,
        success=fit.success,
    )
    return fit_result, fit.params


def _beta_from_params(model_name: str, params: np.ndarray) -> float:
    if model_name == "M0":
        return float(params[1])
    if model_name == "M1":
        return float(params[1])
    if model_name == "M2":
        return float(min(params[1], params[3]))
    if model_name == "M3a":
        return float(params[2])
    if model_name == "M3b":
        return float(params[1])
    return float("nan")


def _beta_index(model_name: str, params: np.ndarray) -> int:
    """Index in ``params`` of the value that :func:`_beta_from_params` returns.

    The BCa CI is taken on ``cis[beta_index]`` and MUST describe the same
    parameter as the point estimate ``beta_D``. For M2 (stretched
    bi-exponential) ``beta_D = min(beta1, beta2)``, so the index depends on the
    fitted params -- not a fixed constant. Previously M2 was hardcoded to index
    1 (beta1), so when beta2 < beta1 the reported CI described a different
    parameter than the point estimate.
    """
    if model_name == "M3a":
        return 2
    if model_name == "M2":
        return 1 if float(params[1]) <= float(params[3]) else 3
    # M0, M1, M3b (and fallback): beta sits at index 1
    return 1


def compute_relaxation_layer(
    L_super: np.ndarray,
    *,
    rho_initial: np.ndarray | None = None,
    rho_steady_state: np.ndarray | None = None,
    t_grid: np.ndarray | None = None,
    bootstrap_B: int = 200,
    seed: int = 42,
) -> RelaxationResult:
    """Run D5-D7b and the M0..M3b fit hierarchy."""
    L_super = np.asarray(L_super)
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    if rho_steady_state is None:
        rho_steady_state = steady_state(L_super)
    if rho_initial is None:
        rho_initial = np.eye(d, dtype=complex) / d
    if t_grid is None:
        t_grid = np.linspace(0.0, 10.0, 80)

    traj = _evolve(L_super, rho_initial, t_grid)
    final_rho = traj[-1]
    rel_entropy = np.array(
        [relative_entropy(traj[k], rho_steady_state) for k in range(traj.shape[0])]
    )
    fidelity_curve = np.array(
        [fidelity(traj[k], rho_steady_state) for k in range(traj.shape[0])]
    )
    trace_distance_curve = np.array(
        [trace_distance(traj[k], rho_steady_state) for k in range(traj.shape[0])]
    )

    # Fit hierarchy on relative entropy decay (ensures positivity).
    fits: dict[str, FitResult] = {}
    for name in ("M0", "M1", "M2", "M3a", "M3b"):
        try:
            fit_result, _ = _fit_with_model(name, t_grid, rel_entropy)
            fits[name] = fit_result
        except (ValueError, RuntimeError):
            continue

    aiccs = {name: fr.aicc for name, fr in fits.items()}
    winner = choose_model(aiccs) if fits else "M0"
    if winner not in fits:
        winner = next(iter(fits)) if fits else "M0"

    # Bootstrap on the winning model for beta_D
    beta_D = _beta_from_params(winner, fits[winner].params) if winner in fits else float("nan")
    bca_lo, bca_hi = beta_D, beta_D
    if winner in fits and np.isfinite(beta_D):
        winner_fn = {"M0": M0, "M1": M1, "M2": M2, "M3a": M3a, "M3b": M3b}[winner]
        try:
            samples, theta_hat = parametric_bootstrap(
                winner_fn, t_grid, rel_entropy, fits[winner].params,
                B=bootstrap_B, rng=np.random.default_rng(seed),
            )
            jk = None
            if t_grid.size <= 60:
                jk = _jackknife(winner_fn, t_grid, rel_entropy, theta_hat, None)
            cis = bca_ci(samples, theta_hat, jackknife_estimates=jk)
            beta_idx = _beta_index(winner, fits[winner].params)
            bca_lo, bca_hi = float(cis[beta_idx, 0]), float(cis[beta_idx, 1])
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            pass

    try:
        ent_asym = entanglement_asymmetry(final_rho)
    except Exception:
        ent_asym = float("nan")

    return RelaxationResult(
        von_neumann_entropy=von_neumann_entropy(final_rho),
        relative_entropy_curve=rel_entropy,
        fidelity_curve=fidelity_curve,
        trace_distance_curve=trace_distance_curve,
        entanglement_asymmetry=None if np.isnan(ent_asym) else ent_asym,
        fits=fits,
        aicc_model=winner,
        beta_D=float(beta_D),
        bca_ci_beta=(bca_lo, bca_hi),
    )
