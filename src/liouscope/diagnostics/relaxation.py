"""Relaxation layer R: D5 VNE, D6 rel-entropy, D7 fidelity, D7b ent. asym.,
plus the M0-M3b fit hierarchy orchestrated through AICc with N_eff.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

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


class UnrepresentableTrajectoryError(RuntimeError):
    """The requested relaxation propagation is not representable reliably.

    This is a numerical-domain failure, not a statement about the underlying
    GKSL dynamics. Returning a non-finite propagated state would launder an
    arithmetic failure into entropy, fitting and uncertainty calculations, so
    the relaxation layer fails closed.
    """


def _operator_is_finite(operator: Any) -> bool:
    """Exact stored-entry finiteness check without densifying sparse matrices."""
    if sp.issparse(operator):
        data = np.asarray(operator.data)
        return bool(np.all(np.isfinite(data)))
    return bool(np.all(np.isfinite(np.asarray(operator))))


def _scaled_action_operand(
    L_super: Any, t: float
) -> Any:
    """Return t*L after representability checks needed by expm_multiply.

    The action algorithm needs norm information internally. A matrix whose
    represented entries are finite can still have an unrepresentable 1-norm;
    passing that case onward can turn a numerical-domain failure into an
    effectively unbounded scaling loop. Refuse only when the matrix or its
    mathematical 1-norm is not representable -- no physics threshold is used.
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        scaled = L_super * t
    if not _operator_is_finite(scaled):
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: L*t contains non-finite entries at "
            f"t={float(t):.6g}"
        )

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                norm_1 = (
                    float(spla.norm(scaled, ord=1))
                    if sp.issparse(scaled)
                    else float(np.linalg.norm(np.asarray(scaled), ord=1))
                )
    except (RuntimeWarning, OverflowError, ValueError) as exc:
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: ||L*t||_1 is not representable in float64 "
            f"at t={float(t):.6g}"
        ) from exc
    if not np.isfinite(norm_1):
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: ||L*t||_1 is not representable in float64 "
            f"at t={float(t):.6g}"
        )
    return scaled


def _expm_action(operator: Any, state: np.ndarray, *, t: float) -> np.ndarray:
    """Compute one exponential action and normalise numerical failures."""
    scaled = _scaled_action_operand(operator, t)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                out = spla.expm_multiply(scaled, state)
    except (
        RuntimeWarning,
        OverflowError,
        ValueError,
        np.linalg.LinAlgError,
        sla.LinAlgError,
    ) as exc:
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: scipy.sparse.linalg.expm_multiply could not "
            f"represent the exponential action at t={float(t):.6g}"
        ) from exc
    out = np.asarray(out, dtype=complex)
    if not np.all(np.isfinite(out)):
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: scipy.sparse.linalg.expm_multiply returned "
            f"a non-finite state at t={float(t):.6g}"
        )
    return out


def _is_exact_linspace(t_grid: np.ndarray) -> bool:
    """Whether t_grid is exactly reproducible by NumPy linspace."""
    if t_grid.ndim != 1 or t_grid.size < 2:
        return False
    expected = np.linspace(
        float(t_grid[0]), float(t_grid[-1]), int(t_grid.size), endpoint=True
    )
    return bool(np.array_equal(np.asarray(t_grid, dtype=float), expected))


def _evolve_with_backend(
    L_super: Any, rho0: np.ndarray, t_grid: np.ndarray
) -> tuple[np.ndarray, str]:
    """Propagate by exponential action without materialising exp(tL).

    Exactly linspace-generated grids use SciPy interval mode so setup work can
    be reused. Arbitrary grids use independent actions from the same initial
    state; this avoids accumulated stepwise error on non-uniform grids.
    """
    rho_vec0 = vec(rho0)
    d = rho0.shape[0]
    times = np.asarray(t_grid, dtype=float)
    if times.ndim != 1:
        raise ValueError(
            "relaxation trajectory: t_grid must be one-dimensional; "
            f"got shape {times.shape}"
        )
    if not np.all(np.isfinite(times)):
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: t_grid contains non-finite time points"
        )
    traj = np.empty((times.size, d, d), dtype=complex)

    if times.size == 0:
        return traj, "expm_multiply_pointwise"

    if _is_exact_linspace(times) and times.size >= 2:
        max_t = float(np.max(np.abs(times)))
        _scaled_action_operand(L_super, max_t)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    states = spla.expm_multiply(
                        L_super,
                        rho_vec0,
                        start=float(times[0]),
                        stop=float(times[-1]),
                        num=int(times.size),
                        endpoint=True,
                    )
        except (
            RuntimeWarning,
            OverflowError,
            ValueError,
            np.linalg.LinAlgError,
            sla.LinAlgError,
        ) as exc:
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: scipy.sparse.linalg.expm_multiply "
                "interval propagation failed"
            ) from exc
        states = np.asarray(states, dtype=complex)
        if states.shape != (times.size, rho_vec0.size) or not np.all(
            np.isfinite(states)
        ):
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: expm_multiply interval propagation "
                "returned a non-finite or malformed state sequence"
            )
        for k in range(times.size):
            traj[k] = unvec(states[k], d=d)
        if times[0] == 0.0:
            traj[0] = rho0
        return traj, "expm_multiply_interval"

    for k, t in enumerate(times):
        if t == 0.0:
            traj[k] = rho0
            continue
        traj[k] = unvec(_expm_action(L_super, rho_vec0, t=float(t)), d=d)
    return traj, "expm_multiply_pointwise"


def _evolve(L_super: Any, rho0: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Compatibility wrapper returning only the propagated trajectory."""
    return _evolve_with_backend(L_super, rho0, t_grid)[0]


def _trajectory_audit(traj: np.ndarray) -> tuple[float, float, float]:
    """Return trace error, Hermiticity defect, and minimum Hermitian eigenvalue.

    These are measurements, not positivity repairs or gates. In particular a
    small negative eigenvalue is recorded as drift rather than silently clipped.
    """
    max_trace_error = 0.0
    max_hermiticity_defect = 0.0
    min_eigenvalue = float("inf")
    for rho in np.asarray(traj):
        max_trace_error = max(
            max_trace_error, float(abs(np.trace(rho) - 1.0))
        )
        anti = rho - rho.conj().T
        max_hermiticity_defect = max(
            max_hermiticity_defect, float(np.linalg.norm(anti, ord="fro"))
        )
        herm = 0.5 * (rho + rho.conj().T)
        min_eigenvalue = min(
            min_eigenvalue, float(np.min(np.linalg.eigvalsh(herm)).real)
        )
    return max_trace_error, max_hermiticity_defect, min_eigenvalue


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
    r"""D7b: Rylands et al. 2024 entanglement-asymmetry measure (single block).

    The entanglement asymmetry of a subsystem state ``rho`` with respect to a
    U(1) charge ``Q`` (here the total magnetisation of the block) is

    .. math::

        \Delta S_A \;=\; S(\rho_Q) - S(\rho),
        \qquad \rho_Q \;=\; \sum_q \Pi_q\, \rho\, \Pi_q,

    where ``\Pi_q`` projects onto the eigenspace of ``Q`` with eigenvalue ``q``.
    Block-dephasing ``rho -> rho_Q`` removes exactly the coherences *between*
    distinct charge sectors, so ``\Delta S_A >= 0`` and ``\Delta S_A = 0`` iff
    ``rho`` commutes with ``Q`` (i.e. the state is symmetric). This is the
    published Ares-Murciano-Calabrese / Rylands construction, and it differs
    from a full single-qubit Pauli twirl, which would instead maximally mix the
    twirled qubit and report an entropy *deficit* rather than a symmetry breaking
    (a Bell state ``(|01> + |10>)/sqrt2`` lives in the single sector ``q = 1`` and
    is therefore exactly symmetric, ``\Delta S_A = 0``).

    For the supported two-qubit block (``d = 4``) the charge is the number
    operator ``Q = n_1 + n_2`` in the computational basis, with sectors
    ``q in {0, 1, 2}`` (basis-state populations by Hamming weight). Returns
    ``nan`` outside the supported ``d = 4`` case; the report layer maps that to
    ``entanglement_asymmetry=None``.
    """
    rho = np.asarray(rho)
    d = rho.shape[0]
    if d not in (4,):
        return float("nan")
    # Charge = total magnetisation / particle number of the 2-qubit block:
    # basis index i in {0,1,2,3} -> Hamming weight (00->0, 01/10->1, 11->2).
    charge = np.array([bin(i).count("1") for i in range(d)])
    # rho_Q = sum_q Pi_q rho Pi_q keeps only the intra-sector (equal-charge)
    # blocks of rho; inter-sector coherences are projected out. Because the
    # computational basis diagonalises Q, this is the Hadamard mask that zeroes
    # every entry rho[i, j] with charge[i] != charge[j].
    mask = charge[:, None] == charge[None, :]
    rho_q = np.where(mask, rho, 0.0)
    # von_neumann_entropy Hermitises defensively; the mask preserves Hermiticity
    # exactly (it is symmetric), so rho_q stays a valid density operator.
    return float(max(von_neumann_entropy(rho_q) - von_neumann_entropy(rho), 0.0))


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
    # Round-17 review (PR #121). ``fit_gls_ar1`` flips ``success`` when the
    # final model evaluation ended inside the magnitude guards -- convergence
    # for the wrong reason, the plateau where every derivative is exactly
    # zero. Flipping a flag no consumer reads changed nothing: the failed fit
    # still carried a finite log-likelihood, hence a finite AICc, and could
    # WIN ``choose_model`` and supply the reported decay rate.
    #
    # Made non-selectable HERE rather than enforced at each consumer, and
    # deliberately so: ``_dominant_rate``, ``compute_relaxation_layer``, the
    # bootstrap and the holdout path are four independent obligations to
    # remember, which is precisely the shape of the four-fold miss this same
    # review round found in the zero-mode filters. ``choose_model`` already
    # drops non-finite entries, so ``inf`` is the existing vocabulary for
    # "not a candidate"; the fit itself stays in ``fits`` so the report still
    # shows that the model was tried and why it lost.
    aic = aicc(fit.log_likelihood, k, n_eff) if fit.success else float("inf")
    fit_result = FitResult(
        model=model_name,
        params=fit.params,
        log_likelihood=fit.log_likelihood,
        aicc=aic,
        n_eff=n_eff,
        residual_ar1_rho=fit.rho_ar1,
        success=fit.success,
        likelihood_degenerate=fit.likelihood_degenerate,
        # ROUND-2 REVIEW (PR #147). This conversion carried
        # ``likelihood_degenerate`` and dropped ``scale_unavailable``, so the
        # newly supported underflow case -- ``success=True`` with a withheld
        # ``sigma`` -- reached the report indistinguishable from an ordinary
        # fit. Its bootstrap then fails and ``compute_relaxation_layer``
        # records ``bca_ci_beta = (nan, nan)``, exactly what a jackknife
        # failure or a non-converged resample produces. The reason lived only
        # in a RuntimeWarning, which a persisted artefact does not keep.
        scale_unavailable=fit.scale_unavailable,
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


def _dominant_rate(t: np.ndarray, curve: np.ndarray) -> tuple[float, str]:
    """Fit the M0..M3b hierarchy (AICc, no bootstrap) to a decay ``curve`` and
    return ``(dominant_rate, winning_model)``.

    Used to obtain a LINEAR-metric relaxation rate (from the trace-distance
    curve) for the D17 gap-rate consistency check. Unlike relative entropy,
    a linear distance metric decays at the bare mode rate ``Delta`` (no metric
    multiplier), so its dominant rate is dimension-coherent with the spectral
    gap. Returns ``(nan, "none")`` if every model fit fails -- fail-closed, so
    the consistency check reads "unknown" rather than a spurious match.
    """
    curve = np.asarray(curve, dtype=float)
    fits: dict[str, FitResult] = {}
    for name in ("M0", "M1", "M2", "M3a", "M3b"):
        try:
            fit_result, _ = _fit_with_model(name, t, curve)
            fits[name] = fit_result
        except (ValueError, RuntimeError):
            continue
    if not fits:
        return float("nan"), "none"
    # Round-17 review: ``choose_model`` falls back to "M0" when nothing has
    # a finite AICc, and the old ``next(iter(fits))`` then handed back an
    # arbitrary FAILED fit's rate. Every candidate saturated == the rate is
    # unknown, which D17 already knows how to read.
    # Round-17 review, CORRECTED after the full suite. The predicate is
    # ``success``, NOT ``isfinite(aicc)``: a non-finite AICc is not the same
    # event as a failed fit. Measured on validation system V4 (thermal
    # two-level), trace-distance curve: all five models converge, yet every
    # AICc is ``inf`` because the Geyer-corrected ``n_eff`` of that smooth,
    # strongly autocorrelated residual series is too small for the
    # small-sample correction. Keying on finiteness withheld D17 on a system
    # where nothing had failed. ``choose_model`` keeps its documented "all
    # entries inf -> M0" fallback; the only thing enforced here is that the
    # winner must come from the SUCCEEDED set.
    selectable = {n: fr.aicc for n, fr in fits.items() if fr.success}
    if not selectable:
        return float("nan"), "none"
    winner = choose_model(selectable)
    if winner not in selectable:
        # The all-inf fallback named a model outside the succeeded set. Keep
        # it inside, in ladder order -- deterministic, and identical to the
        # previous behaviour whenever M0 itself succeeded.
        winner = next(iter(selectable))
    return _beta_from_params(winner, fits[winner].params), winner


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

    traj, trajectory_backend = _evolve_with_backend(L_super, rho_initial, t_grid)
    (
        trajectory_max_trace_error,
        trajectory_max_hermiticity_defect,
        trajectory_min_eigenvalue,
    ) = _trajectory_audit(traj)
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

    # Round-17 review: only fits that did NOT end on the magnitude plateau
    # are candidates. With none left the winner is the "none" sentinel that
    # ``_dominant_rate`` and ``linear_fit_model`` already use -- beta_D then
    # falls through to NaN below instead of being read off a failed fit, and
    # the A-class rungs that compare ``aicc_model`` simply do not fire.
    # Round-17 review, CORRECTED after the full suite. The predicate is
    # ``success``, NOT ``isfinite(aicc)``: a non-finite AICc is not the same
    # event as a failed fit. Measured on validation system V4 (thermal
    # two-level), trace-distance curve: all five models converge, yet every
    # AICc is ``inf`` because the Geyer-corrected ``n_eff`` of that smooth,
    # strongly autocorrelated residual series is too small for the
    # small-sample correction. Keying on finiteness withheld D17 on a system
    # where nothing had failed. ``choose_model`` keeps its documented "all
    # entries inf -> M0" fallback; the only thing enforced here is that the
    # winner must come from the SUCCEEDED set.
    selectable = {name: fr.aicc for name, fr in fits.items() if fr.success}
    winner = choose_model(selectable) if selectable else "none"
    if selectable and winner not in selectable:
        winner = next(iter(selectable))

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
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            # Fail loud, not certain: a swallowed bootstrap failure used to
            # leave the degenerate CI (beta_D, beta_D), which the uncertainty
            # layer reads as fit_uncertainty == 0.0 — *perfect* confidence as
            # the failure mode of an uncertainty pipeline. NaN propagates as
            # "unknown" instead.
            warnings.warn(
                f"parametric bootstrap for beta_D failed ({exc!r}); "
                "bca_ci_beta set to (nan, nan) — fit uncertainty is UNKNOWN, "
                "not zero.",
                RuntimeWarning,
            )
            bca_lo, bca_hi = float("nan"), float("nan")

    try:
        ent_asym = entanglement_asymmetry(final_rho)
    except Exception:
        ent_asym = float("nan")

    # LINEAR-metric relaxation rate for D17 (LIOU-#69). ``beta_D`` above is fit
    # on the RELATIVE-ENTROPY curve, which decays at a metric multiplier m times
    # the bare mode rate (m=2 for a faithful/full-rank steady state where D is
    # quadratic near pi; m=1 for a rank-deficient pi where the null-space-leakage
    # term is linear). That multiplier makes beta_D dimensionally incomparable to
    # the spectral gap Delta. The trace-distance curve is a linear distance
    # metric, so its dominant rate decays at the bare mode rate and IS
    # dimension-coherent with Delta -- this is what the gap-rate consistency
    # check (D17) must use. (_dominant_rate is fail-closed: it returns nan if
    # every model fit fails, so D17 then reads "unknown" rather than a spurious
    # match.)
    beta_D_linear, linear_fit_model = _dominant_rate(t_grid, trace_distance_curve)

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
        beta_D_linear=float(beta_D_linear),
        linear_fit_model=linear_fit_model,
        trajectory_backend=trajectory_backend,
        trajectory_max_trace_error=trajectory_max_trace_error,
        trajectory_max_hermiticity_defect=trajectory_max_hermiticity_defect,
        trajectory_min_eigenvalue=trajectory_min_eigenvalue,
    )
