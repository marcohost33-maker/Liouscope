"""Relaxation layer R: D5 VNE, D6 rel-entropy, D7 fidelity, D7b ent. asym.,
plus the M0-M3b fit hierarchy orchestrated through AICc with N_eff.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_SUPP
from .._types import FitResult, RelaxationResult
from ..core.lindblad import steady_state
from ..fitting.aicc import aicc, choose_model
from ..fitting.bootstrap import _jackknife, bca_ci, parametric_bootstrap
from ..fitting.car1 import is_uniform_grid, neff_car1
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
from .spectral import compute_spectral_layer

# Default relaxation window, expressed in the system's OWN slowest relaxation
# time ``1 / Delta`` (D1) rather than in absolute time units.
#
# A decay rate carries dimension 1/time, so any *absolute* default window is
# implicitly a claim about the caller's unit of time. The legacy default
# ``linspace(0.0, 10.0, 80)`` made the whole relaxation layer -- D5, D6, D7,
# the M0..M3b AICc comparison, ``beta_D``, its BCa interval and the D17
# gap-rate check -- silently dependent on that choice: under the pure change of
# rate units ``L -> cL`` (identical physics, different unit of time) ``beta_D``
# drifted by >20%, the D17 linear rate ``beta_D_linear`` was wrong by a factor
# ~430 at ``c = 1e-6``, and ``diagnose()`` returned four different A-classes
# across ``c in [1e-6, 1e2]``. Rates in real systems are MHz or GHz, not O(1),
# so the affected regime is the common one, not an exotic corner.
#
# ``[0, HORIZON / Delta]`` fixes the window at a fixed number of e-foldings of
# the slowest mode, which is the only unit-invariant choice: it is carried
# along by the rescaling and therefore samples the same dimensionless curve
# whatever units the caller works in. ``HORIZON = 10`` is ``e^-10 ~ 4.5e-5`` of
# the initial deviation -- deep enough to identify the rate, shallow enough
# that the tail is not almost entirely numerical floor.
#
# Chosen to be exactly backward compatible at ``Delta = 1``: the grid is then
# bit-identical to the legacy ``linspace(0.0, 10.0, 80)``.
RELAXATION_HORIZON: float = 10.0
RELAXATION_N_POINTS: int = 80

# Legacy absolute window, retained ONLY for the degenerate case where no decay
# scale exists (``Delta <= 0``: no isolated steady state, or a gap that floored
# to zero). There is no physical timescale to scale by there, so any window is
# a convention; keeping the historical one avoids inventing a second arbitrary
# constant and keeps those runs comparable with earlier releases.
_LEGACY_T_MAX: float = 10.0


# Minimum samples per e-folding of the WORST-RESOLVED decaying mode before
# that mode counts as stepped over rather than measured. Below 1 the component
# loses more than 63% of its amplitude between consecutive samples, so no fit
# can identify it from this grid.
#
# A curve-shape heuristic was tried first and rejected: the share of the decay
# landing in the first interval measures how much AMPLITUDE the fast component
# carries, not whether it was RESOLVED, and the two are independent (measured
# 0.49 for a fully unsampled 1e-6/1 separation against 0.46 for a mild 0.1/1
# one). The spectrum answers the actual question directly.
MIN_SAMPLES_PER_FAST_EFOLD: float = 1.0

# Number of samples the two-scale grid places inside the fast mode's own
# window, as a fraction of ``n_points``. With ``n_points = 80`` and
# ``horizon = 10`` this puts 40 points across ten e-foldings of the fastest
# mode -- exactly 4 samples per fast e-folding, INDEPENDENTLY of the rates,
# because the segment is measured in that mode's own relaxation time. The
# remaining points carry the window out to ``horizon / gap`` so the slow mode
# is still measured over the same ten e-foldings as before.
_FAST_SEGMENT_FRAC: float = 0.5


def decay_rates(L_super: np.ndarray) -> np.ndarray:
    """Distinct positive decay rates ``-Re lambda`` of the Liouvillian.

    Every mode of ``exp(L t)`` decays as ``exp(-r t)`` with ``r = -Re lambda``;
    the zero mode (steady state) and any numerically non-finite eigenvalue are
    dropped. Returned sorted ascending, so ``[-1]`` is the fastest rate.
    Empty when nothing decays.
    """
    rates = -np.real(np.linalg.eigvals(np.asarray(L_super)))
    positive = rates[np.isfinite(rates) & (rates > 0.0)]
    return np.sort(positive)


def fastest_decay_rate(L_super: np.ndarray) -> float:
    """``max(-Re lambda)``: the rate that sets the finest timescale to resolve.

    Returns NaN when no finite positive rate exists -- fail-closed, so a caller
    sizing a grid from this value falls back to its documented default instead
    of scaling by a fabricated rate.
    """
    rates = decay_rates(L_super)
    return float(rates[-1]) if rates.size else float("nan")


def default_relaxation_grid(
    gap: float,
    *,
    fast_rate: float | None = None,
    horizon: float = RELAXATION_HORIZON,
    n_points: int = RELAXATION_N_POINTS,
) -> np.ndarray:
    """Time grid spanning ``horizon`` e-foldings of the slowest mode.

    Without ``fast_rate`` this is ``linspace(0, horizon / gap, n_points)``:
    a uniform window measured in the system's own relaxation time ``1 / gap``
    (D1). See the module-level note on :data:`RELAXATION_HORIZON` for why the
    default must not be an absolute window.

    With a ``fast_rate`` that the uniform grid cannot resolve, the grid becomes
    **two-scale**: the first ``_FAST_SEGMENT_FRAC`` of the points cover
    ``[0, horizon / fast_rate]`` -- ten e-foldings of the FASTEST mode -- and
    the rest carry the window out to ``horizon / gap`` as before. Both ends of
    the spectrum are then sampled by one grid, at 4 samples per fast e-folding
    for the default constants, whatever the separation.

    This replaces a disclosure with a repair. The previous docstring argued a
    non-uniform grid was impossible because the GLS layer "whitens with a
    single AR(1) coefficient, which presumes a constant sample interval". That
    is a property of the DISCRETE parametrisation, not of the noise: the
    stationary continuous-time process has ``Corr(t, t+d) = exp(-theta d)``, so
    whitening with the per-step ``exp(-theta dt_k)`` is valid on ANY grid.
    :mod:`liouscope.fitting.car1` implements that, and the fitting layer
    switches to it whenever the grid is non-uniform -- measured lag-1
    autocorrelation of the whitened residuals on a two-scale grid: 0.4231 with
    one constant rho against 0.0945 with the per-step coefficient, the latter
    at the level of the near-uniform positive control (0.0728).

    The uniform grid is kept, bit-for-bit, whenever it already resolves the
    fast mode (``1 / (fast_rate * dt) >= MIN_SAMPLES_PER_FAST_EFOLD``), so no
    system that was well sampled before changes at all, and the switch happens
    exactly in the regime that previously only got a warning.

    ``gap <= 0`` or a non-finite ``gap`` means no usable decay scale was
    resolved; the historical absolute window is returned in that case. A
    missing, non-finite or non-positive ``fast_rate`` likewise falls back to
    the uniform grid rather than guessing a fast scale.
    """
    if not np.isfinite(gap) or gap <= 0.0:
        return np.linspace(0.0, _LEGACY_T_MAX, n_points)
    t_max = horizon / float(gap)
    uniform = np.linspace(0.0, t_max, n_points)

    # --- fail-closed guards: any doubt keeps the historical uniform grid ---
    if fast_rate is None or not np.isfinite(fast_rate) or fast_rate <= 0.0:
        return uniform
    n_fast = int(n_points * _FAST_SEGMENT_FRAC)
    if n_fast < 2 or n_points - n_fast < 2:
        return uniform
    dt_uniform = float(uniform[1] - uniform[0])
    if dt_uniform <= 0.0:
        return uniform
    # Already resolved: leave it exactly as it was.
    if 1.0 / (float(fast_rate) * dt_uniform) >= MIN_SAMPLES_PER_FAST_EFOLD:
        return uniform
    t_split = horizon / float(fast_rate)
    if not np.isfinite(t_split) or t_split <= 0.0 or t_split >= t_max:
        return uniform

    early = np.linspace(0.0, t_split, n_fast, endpoint=False)
    late = np.linspace(t_split, t_max, n_points - n_fast)
    grid = np.concatenate([early, late])
    # Structural post-condition rather than trust in the arithmetic above: a
    # grid that is not strictly increasing and finite would corrupt every
    # consumer downstream (the propagator, the whitening, N_eff), so a failure
    # here degrades to the uniform grid instead of propagating.
    if (
        grid.size != n_points
        or not np.all(np.isfinite(grid))
        or np.any(np.diff(grid) <= 0.0)
        or grid[0] != 0.0
    ):
        return uniform
    return grid


class UnderResolvedTransientWarning(UserWarning):
    """A decaying mode is stepped over by the grid rather than measured.

    A grid can only resolve dynamics slower than the largest interval it leaves
    unsampled in the region where that mode still has amplitude. On the DEFAULT
    path this is now largely repaired rather than merely disclosed: when the
    uniform gap-scaled window cannot resolve the fastest mode,
    :func:`default_relaxation_grid` switches to a two-scale grid that samples
    both ends of the spectrum, and the fitting layer switches to the CAR(1)
    residual model that is valid on it (:mod:`liouscope.fitting.car1`).

    Two situations remain, and this warning exists for them:

    * a CALLER-supplied ``t_grid`` that does not resolve its own system -- for
      instance one starting long after the mode has decayed, where no step size
      compensates for the unsampled lead-in;
    * an INTERMEDIATE timescale on the default two-scale grid. The two segments
      resolve the fastest and the slowest mode; a mode between them can still
      fall between the coarse late samples. Measured on three independent
      damped qubits with rates 1e-6, 1e-3 and 1, the middle mode is sampled
      0.0039 times per e-folding and the warning fires -- correctly.

    Widening the window is still not a fix -- an absolute window resolves the
    fast mode and misses the relaxation entirely (measured on two independent
    damped qubits with rates 1e-6 and 1: the absolute window put
    ``beta_D_linear`` 3.5e4 relative away from the true gap, the gap-scaled
    window 0.58). What changed is that a non-uniform grid IS now available,
    because the whitening it needs was implemented rather than assumed
    impossible.

    When this warning does fire, the rates the layer reports describe the
    dynamics the window resolves; a caller who needs the unresolved component
    must supply a ``t_grid`` covering that timescale (and read the resulting
    rates as describing that window).
    """


def _resolution_detail(
    L_super: np.ndarray, t_grid: np.ndarray
) -> tuple[float, float, float]:
    """How finely the grid samples the WORST-RESOLVED decaying mode.

    Returns ``(samples_per_efolding, rate, blind_interval)`` for the mode that
    the grid resolves least well.

    For each decay rate ``r`` of ``L_super`` this forms ``1 / (r * blind_r)``,
    where ``blind_r`` is the largest interval the grid leaves unsampled *while
    that mode still has amplitude*, and returns the MINIMUM over modes.

    Two deliberate choices:

    ``blind_r`` counts the lead-in from ``t = 0``, not merely the step. The
    dynamics starts at ``rho_initial`` at ``t = 0`` while the trajectory is
    only evaluated from ``t[0]``, and :func:`liouscope.diagnose` permits any
    non-negative start, so a late-starting grid leaves a lead-in that no step
    size compensates for: a rate-1 mode on ``linspace(100, 101, 101)`` is
    sampled 100x per e-folding by its step and is nonetheless long gone by the
    first sample (amplitude ``e^-100``, and the whole relative-entropy curve
    measures identically zero while the fit still returns a confident-looking
    rate).

    ``blind_r`` only considers intervals that START before ``1 / r``. A
    coarse interval after the mode has decayed is not blindness -- there is
    nothing left to see -- and counting it would make every two-scale grid look
    under-resolved because of its own late segment. On a uniform grid the
    restriction is inert (every interval has the same length), so this reduces
    exactly to the historical ``1 / (r_max * max(dt, t[0]))``; the pinned
    values for uniform and late-starting grids are unchanged.

    Taking the minimum over ALL modes rather than only the fastest is what
    makes the measure honest on a non-uniform grid. The old form read
    ``t[1] - t[0]``, which on a fine-then-coarse grid reports the FINE step and
    would wave through a middle timescale that falls entirely between the
    coarse samples. That case is created, not hypothetical: it is exactly the
    intermediate scale a two-scale grid does not resolve.

    Returns ``inf`` when nothing decays (no finite positive rate) and NaN when
    the grid has no usable interval.
    """
    t = np.asarray(t_grid, dtype=float)
    if t.size < 2 or not np.all(np.isfinite(t)):
        return float("nan"), float("nan"), float("nan")
    rates = decay_rates(L_super)
    if rates.size == 0:
        return float("inf"), float("nan"), float("nan")

    # Unsampled intervals as (start, length): the lead-in [0, t[0]] first, then
    # every grid step. Starts are non-decreasing for an increasing grid, so a
    # running maximum of the lengths answers "largest interval starting before
    # T" by a single search.
    lead_in = float(t[0]) if t[0] > 0.0 else 0.0
    starts = np.concatenate([[0.0], t[:-1]])
    lengths = np.concatenate([[lead_in], np.diff(t)])
    if not np.all(np.isfinite(lengths)) or np.any(lengths[1:] <= 0.0):
        return float("nan"), float("nan"), float("nan")
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    running_max = np.maximum.accumulate(lengths[order])

    worst = float("inf")
    worst_rate = float("nan")
    worst_blind = float("nan")
    for r in rates:
        horizon_r = 1.0 / float(r)
        idx = int(np.searchsorted(starts, horizon_r, side="left"))
        # ``starts[0] == 0.0 < horizon_r`` for any finite positive rate, so the
        # slice is never empty.
        blind = float(running_max[max(idx, 1) - 1])
        if blind <= 0.0:
            continue
        value = 1.0 / (float(r) * blind)
        if value < worst:
            worst, worst_rate, worst_blind = value, float(r), blind
    if not np.isfinite(worst):
        # Every interval was zero-length: no usable step.
        return float("nan"), float("nan"), float("nan")
    return worst, worst_rate, worst_blind


def samples_per_fast_efolding(L_super: np.ndarray, t_grid: np.ndarray) -> float:
    """Samples per e-folding of the worst-resolved mode.

    Thin wrapper over :func:`_resolution_detail`; see its docstring. The two
    share one implementation deliberately, so the number this returns and the
    number quoted in :class:`UnderResolvedTransientWarning` can never describe
    different intervals.
    """
    return _resolution_detail(L_super, t_grid)[0]


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
    # N_eff must be computed under the SAME residual model the fit whitened
    # with, or the AICc it feeds compares likelihoods from one model against a
    # sample size from another. The Geyer IPS estimator sums autocorrelations
    # indexed by LAG; on a grid whose step varies, lag 1 is not a fixed time
    # separation, so the sequence it sums is not an autocorrelation function.
    # The CAR(1) form is exact there and reduces to the same double sum on a
    # uniform grid. Fail-closed: if it cannot be formed, fall back to Geyer.
    n_eff = float("nan")
    if np.isfinite(fit.theta_car1):
        n_eff = neff_car1(t, fit.theta_car1)
    if not np.isfinite(n_eff):
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
        residual_theta_car1=float(fit.theta_car1),
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
    winner = choose_model({n: fr.aicc for n, fr in fits.items()})
    if winner not in fits:
        winner = next(iter(fits))
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
    gap: float | None = None,
    bootstrap_B: int = 200,
    seed: int = 42,
) -> RelaxationResult:
    """Run D5-D7b and the M0..M3b fit hierarchy.

    Parameters
    ----------
    gap
        Spectral gap ``Delta`` (D1), used only to scale the DEFAULT time grid
        to the system's own relaxation time when ``t_grid`` is not given (see
        :func:`default_relaxation_grid`). :func:`liouscope.diagnose` forwards
        the gap it has already computed; a direct caller who omits it gets the
        same value recomputed from ``L_super`` via D1, so the default window is
        rate-unit invariant either way. Ignored when ``t_grid`` is supplied.
    """
    L_super = np.asarray(L_super)
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    if rho_steady_state is None:
        rho_steady_state = steady_state(L_super)
    if rho_initial is None:
        rho_initial = np.eye(d, dtype=complex) / d

    if t_grid is not None:
        t_grid_source = "caller"
    else:
        # Recomputing D1 here (rather than falling back to an absolute window)
        # keeps a direct call to this function as unit-invariant as the full
        # pipeline. It calls the spectral layer itself rather than re-deriving
        # the gap: D1 is not simply the smallest non-zero eigenvalue but carries
        # the certified zero-mode tolerance and the #113 ambiguity rule (an
        # unresolvable slow spectrum yields NaN, not a fast mode). Duplicating
        # any of that here would let the two entry points drift apart. The
        # already-computed steady state is passed through so only the spectrum
        # is recomputed, and a NaN gap degrades to the documented legacy window.
        if gap is None:
            gap = compute_spectral_layer(L_super, rho_steady_state).gap
        # The FAST scale comes from the spectrum, never from a guess: it is
        # ``max(-Re lambda)``, the same eigenvalues the resolution guard reads.
        # ``fastest_decay_rate`` returns NaN when no finite positive rate
        # exists, and ``default_relaxation_grid`` then keeps the uniform window
        # -- fail-closed, so an unusable spectrum degrades to the historical
        # behaviour plus its warning rather than to an invented timescale.
        fast_rate = fastest_decay_rate(L_super)
        t_grid = default_relaxation_grid(gap, fast_rate=fast_rate)
        if not (np.isfinite(gap) and gap > 0.0):
            t_grid_source = "legacy_fixed"
        elif is_uniform_grid(t_grid):
            t_grid_source = "gap_scaled"
        else:
            t_grid_source = "gap_scaled_multiscale"
    t_grid = np.asarray(t_grid, dtype=float)
    # One decision for the whole layer: the grid, not the data, selects the
    # residual model, so every M0..M3b fit below is whitened the same way.
    residual_model = "ar1" if is_uniform_grid(t_grid) else "car1"

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

    # Under-resolution disclosure. Applies to the window actually used, so a
    # caller-supplied grid is checked on the same terms as the default.
    fast_resolution, worst_rate, blind = _resolution_detail(L_super, t_grid)
    if np.isfinite(fast_resolution) and fast_resolution < MIN_SAMPLES_PER_FAST_EFOLD:
        span_kind = (
            "unsampled lead-in from t=0"
            if blind > float(t_grid[1] - t_grid[0])
            else "sample interval"
        )
        warnings.warn(
            f"Relaxation layer: the mode at rate {worst_rate:.4g} decays by "
            f"{100.0 * (1.0 - np.exp(-1.0 / fast_resolution)):.1f}% across the "
            f"largest gap the grid leaves before it ({span_kind} = "
            f"{blind:.4g}, {fast_resolution:.3g} samples per e-folding), so it "
            "is stepped over rather than measured. The reported rates describe "
            "the dynamics this window DOES resolve; that component is not "
            "identifiable from this grid and the M0..M3b comparison cannot see "
            "it. The default grid resolves the fastest and the slowest mode "
            "(see default_relaxation_grid); supply an explicit t_grid if you "
            "need an intermediate one -- note the rates then describe that "
            "window instead.",
            UnderResolvedTransientWarning,
            stacklevel=2,
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
        t_grid_source=t_grid_source,
        t_grid_span=float(t_grid[-1] - t_grid[0]),
        t_grid=t_grid.copy(),
        samples_per_fast_efolding=fast_resolution,
        residual_model=residual_model,
    )
