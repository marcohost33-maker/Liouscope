"""Trajectory propagation backends for the relaxation layer (issue #162).

The relaxation layer needs ``rho(t_k) = exp(t_k L) rho_0`` on a time grid --
the *action* of the matrix exponential on one vector, not the matrix
exponential itself. Two backends compute it:

``"dense_expm"`` (reference)
    Materialises ``scipy.linalg.expm(L * t_k)`` for every grid point and
    applies it to ``vec(rho_0)``. This is the historical formula, kept
    bit-for-bit. Its cost is ``O(n^3)`` per grid point (``n = d^2``) with only
    a logarithmic dependence on ``||L t||`` (scaling and squaring), so it is
    the right tool for small or stiff generators.

``"expm_action"``
    Al-Mohy & Higham's exponential-action algorithm
    (``scipy.sparse.linalg.expm_multiply``), stepping the state from grid point
    to grid point with ``tau_k = t_k - t_{k-1}``. Only matrix-vector products
    with ``L`` are formed, so it works unchanged for dense and sparse
    generators and never allocates an ``n x n`` propagator. Its cost is linear
    in ``||L - mu I||_1 * t_max`` (``mu = tr(L)/n``), which makes it the right
    tool for larger, non-stiff generators and the wrong one for stiff ones.

Stepping instead of restarting from ``rho_0`` at every grid point costs one
long propagation instead of ``N`` overlapping ones. The price is that rounding
errors of successive steps accumulate. For a GKSL generator every step is a
CPTP map and hence a trace-norm contraction, so a step can never amplify the
error it inherits: the accumulated error is bounded by the sum of the per-step
errors (``<= N * u`` in the backward sense), which the dense cross-check tests
measure rather than assume.

Backend selection (``"auto"``)
------------------------------
The choice is a deterministic function of ``(L, t_grid)`` only -- no timing,
no machine state -- so it is reproducible and recorded by the caller
(``RelaxationResult.trajectory_backend``). The rule:

1. ``n < ACTION_MIN_DIM`` (``d < 8``): dense. Measured (single-threaded BLAS,
   80-point grid, dissipative spin chain): at ``n = 4`` and ``n = 16`` dense
   takes 3-5 ms against 11-13 ms for the action, whose per-call overhead
   dominates; from ``n = 64`` on the action wins (0.08 s vs 0.02 s at
   ``n = 64``, 2.9 s vs 0.09 s at ``n = 256``, 120 s vs 2.6 s at
   ``n = 1024``). Every system below the threshold -- which includes every
   anchor fixture -- therefore keeps the historical dense trajectory
   bit-for-bit.
2. The grid must be finite, non-negative and non-decreasing for stepping;
   otherwise dense (whose per-point formula handles any grid).
3. The action is chosen only if its matrix-vector-product count bound is
   finite, within ``max_action_matvecs`` and no larger than the dense
   estimate expressed in the same unit. This is what keeps a stiff generator
   (``||L|| t`` of order ``1e7`` and beyond) on the dense path instead of
   letting the action loop for hours. Sparse generators follow the same
   comparison while densifying them is practical (``n <= DENSE_MAX_DIM``);
   above that the action is the only candidate, and an over-budget action
   is refused rather than replaced by an ``n x n`` materialisation.

Each grid step is split into sub-steps whose shifted norm stays below the
threshold of Al-Mohy & Higham's condition (3.13) (63.36 for one vector). Above
it SciPy would estimate matrix-power norms with the randomised ``onenormest``,
which draws from the caller's GLOBAL ``np.random`` state; below it the
parameter choice is a deterministic function of the exact 1-norm. The
propagation therefore never touches global random state, and its cost bound is
exact.

The bound is an *upper* bound on what SciPy's parameter selection will spend:
SciPy minimises ``m * s`` with ``s = ceil(alpha_p / theta_m)`` over the same
``theta_m`` table, and ``alpha_p <= ||A||_1`` for every ``p``
(Al-Mohy & Higham 2011, eqs. (3.11)-(3.13)). Using the 1-norm itself can only
overestimate the cost, i.e. err towards the dense reference, never towards a
hang.

Failure semantics
-----------------
Both backends fail closed with :class:`UnrepresentableTrajectoryError` when
the propagation leaves the representable float64 domain (non-finite ``L*t``,
a SciPy runtime warning or a non-finite propagator/state) and the action
backend additionally when its cost bound exceeds the budget. A finite
generator and a finite time do not imply a representable trajectory, and a
NaN that reached entropy, fitting or bootstrap would be laundered into a
diagnostic.

Reference: A. H. Al-Mohy and N. J. Higham, *Computing the Action of the Matrix
Exponential, with an Application to Exponential Integrators*, SIAM J. Sci.
Comput. 33(2), 488-511 (2011), doi:10.1137/100788860.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from .generator_guard import require_finite_generator

BACKEND_DENSE: str = "dense_expm"
BACKEND_ACTION: str = "expm_action"
BACKEND_AUTO: str = "auto"
TRAJECTORY_BACKENDS: tuple[str, ...] = (BACKEND_DENSE, BACKEND_ACTION)

# Smallest superoperator dimension n = d^2 at which "auto" considers the
# action backend (d = 8). Measured crossover, see the module docstring.
ACTION_MIN_DIM: int = 64

# Default ceiling on the action backend's matrix-vector-product bound. At
# n = 1024 a benign 80-point trajectory needs ~1.5e3 products, so the ceiling
# only ever bites on stiff inputs, where it turns an hours-long loop into an
# immediate, explicit refusal.
DEFAULT_ACTION_MATVEC_BUDGET: int = 1_000_000

# Largest superoperator dimension a SPARSE generator is densified for when
# "auto" compares against, or falls back to, the dense backend: one complex
# n x n array is 64 MiB here, and expm needs several. Above it the dense
# backend is not a practical fallback, so "auto" uses the action or refuses.
DENSE_MAX_DIM: int = 2048

# theta_m for double precision: m = 1..30 from Higham, *Functions of Matrices*
# (SIAM 2008), table A.3; m = 35..55 from Al-Mohy & Higham (2011), table 3.1.
# The same table SciPy's ``expm_multiply`` selects (m, s) from.
_THETA: dict[int, float] = {
    1: 2.29e-16, 2: 2.58e-8, 3: 1.39e-5, 4: 3.40e-4, 5: 2.40e-3,
    6: 9.07e-3, 7: 2.38e-2, 8: 5.00e-2, 9: 8.96e-2, 10: 1.44e-1,
    11: 2.14e-1, 12: 3.00e-1, 13: 4.00e-1, 14: 5.14e-1, 15: 6.41e-1,
    16: 7.81e-1, 17: 9.31e-1, 18: 1.09, 19: 1.26, 20: 1.44,
    21: 1.62, 22: 1.82, 23: 2.01, 24: 2.22, 25: 2.43,
    26: 2.64, 27: 2.86, 28: 3.08, 29: 3.31, 30: 3.54,
    35: 4.7, 40: 6.0, 45: 7.2, 50: 8.5, 55: 9.9,
}

# Condition (3.13) of Al-Mohy & Higham (2011) for one right-hand side with
# SciPy's m_max = 55, ell = 2 (p_max = 8): ||A||_1 <= 2*ell*p_max*(p_max+3) *
# theta_55 / m_max = 63.36. At or below it SciPy selects (m, s) from the EXACT
# 1-norm; above it SciPy estimates ||A^p||_1 with ``onenormest``, which is
# randomised and draws from the GLOBAL legacy ``np.random`` state -- a hidden
# side effect on the caller's random stream and a data-dependent (m, s). The
# action backend therefore splits every grid step into sub-steps whose shifted
# norm stays below this value (with a 10% margin for the rounding of the norm
# SciPy recomputes), so every call takes the deterministic exact-norm path.
_CONDITION_3_13_NORM: float = 2 * 2 * 8 * (8 + 3) * 9.9 / 55
_SUBSTEP_NORM: float = 0.9 * _CONDITION_3_13_NORM

# Dense estimate, in matrix-vector-product units (one n x n matmul = n
# products): a degree-13 Pade approximant costs ~6 matmuls plus one LU solve
# (~2 matmul equivalents), followed by s squarings with
# s = ceil(log2(||L t||_1 / theta_13)).
_DENSE_BASE_MATMULS: int = 8
_THETA_13: float = 5.371920351148152


class UnrepresentableTrajectoryError(RuntimeError):
    """The requested relaxation propagation is not representable reliably.

    This is a numerical-domain failure, not a statement about the underlying
    GKSL dynamics. Returning a non-finite propagator or state would launder an
    arithmetic failure into entropy, fitting and uncertainty calculations, so
    the relaxation layer fails closed. The action backend raises it too when
    its cost bound exceeds the budget: a propagation that cannot finish is not
    one that can be reported.
    """


@dataclass(frozen=True, slots=True)
class TrajectoryPropagation:
    """Propagated states plus the audit record of how they were obtained."""

    states: np.ndarray            # shape (len(t_grid), n), complex
    backend: str                  # BACKEND_DENSE or BACKEND_ACTION
    action_matvec_bound: float    # upper bound on the action's products (inf if ineligible)


def _is_sparse(L: object) -> bool:
    return bool(sp.issparse(L))


def _trace(L: np.ndarray | sp.spmatrix) -> complex:
    if _is_sparse(L):
        return complex(L.diagonal().sum())  # type: ignore[union-attr]
    return complex(np.trace(L))


def _one_norm(A: np.ndarray | sp.spmatrix) -> float:
    """Exact induced 1-norm (max column sum); ``inf`` on overflow."""
    with np.errstate(over="ignore", invalid="ignore"):
        if _is_sparse(A):
            col = np.asarray(abs(A).sum(axis=0)).ravel()  # type: ignore[operator]
        else:
            col = np.abs(A).sum(axis=0)
        value = float(col.max()) if col.size else 0.0
    return value if math.isfinite(value) else math.inf


def _max_abs_entry(L: np.ndarray | sp.spmatrix) -> float:
    if _is_sparse(L):
        data = np.asarray(L.data)  # type: ignore[union-attr]
        return float(np.max(np.abs(data))) if data.size else 0.0
    return float(np.max(np.abs(L))) if L.size else 0.0


def _shifted_norm_parts(L: np.ndarray | sp.spmatrix) -> tuple[float, float]:
    """``||L - mu I||_1`` as ``(base, scale)`` with ``norm = scale * base``.

    ``scale`` is the power of two just below ``max |L_ij|``, so ``L / scale``
    is exact (no rounding, barring subnormals) and its shifted column sums can
    no longer overflow. A finite generator whose unscaled norm overflows -- two
    ``1e308`` entries in one column -- still gets a finite ``tau * norm`` for a
    small enough ``tau`` (PR #176 review): the product is formed only after
    scaling, see :func:`_step_norm`.
    """
    n = L.shape[0]
    amax = _max_abs_entry(L)
    if n == 0 or amax == 0.0:
        return 0.0, 1.0
    # frexp: amax = m * 2**e with m in [0.5, 1); 2**(e - 1) <= amax < 2**e.
    # The lower power keeps the scale representable even for amax ~ 1.8e308
    # (2**1024 is not) and leaves |L_ij / scale| < 2.
    scale = math.ldexp(1.0, math.frexp(amax)[1] - 1)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        Ls = L / scale
        mu = _trace(Ls) / n
        if _is_sparse(Ls):
            shifted = Ls - mu * sp.identity(n, dtype=complex, format="csr")
        else:
            shifted = Ls - mu * np.eye(n, dtype=complex)
    return _one_norm(shifted), scale


def _step_norm(tau: float, base: float, scale: float) -> float:
    """``tau * scale * base`` without spurious intermediate overflow/underflow."""
    if tau == 0.0 or base == 0.0:
        return 0.0
    direct = tau * (scale * base) if math.isfinite(scale * base) else math.inf
    if math.isfinite(direct) and direct > 0.0:
        return direct
    log2_norm = math.log2(abs(tau)) + math.log2(scale) + math.log2(base)
    return math.inf if log2_norm >= 1024.0 else 2.0**log2_norm


def shifted_one_norm(L: np.ndarray | sp.spmatrix) -> float:
    """``||L - mu I||_1`` with ``mu = tr(L)/n`` -- the norm the action scales by.

    ``inf`` if the norm itself exceeds the float range; the propagation works
    with the overflow-safe ``(base, scale)`` form internally.
    """
    base, scale = _shifted_norm_parts(L)
    return _step_norm(1.0, base, scale)


def _action_step_cost(scaled_norm: float) -> float:
    """Upper bound on ``m * s`` for one ``expm_multiply`` call on ``tau * A``."""
    if not math.isfinite(scaled_norm):
        return math.inf
    if scaled_norm == 0.0:
        return 0.0
    best = math.inf
    for m, theta in _THETA.items():
        # A finite norm can still overflow the quotient for the tiny theta_m
        # of low degrees; that degree is then simply not a candidate.
        quotient = scaled_norm / theta
        if math.isfinite(quotient):
            # Float arithmetic on purpose: m * ceil(q) as an int can exceed the
            # float range, float multiplication saturates to inf instead.
            best = min(best, m * max(1.0, float(math.ceil(quotient))))
    return best


def _substep_count(scaled_norm: float) -> float:
    """Sub-steps that keep each call at or below ``_SUBSTEP_NORM`` (float: may be huge)."""
    if not math.isfinite(scaled_norm):
        return math.inf
    return max(1.0, float(math.ceil(scaled_norm / _SUBSTEP_NORM)))


def _step_cost(scaled_norm: float) -> float:
    """Products for one grid step of shifted norm ``scaled_norm``, sub-steps included."""
    count = _substep_count(scaled_norm)
    if not math.isfinite(count):
        return math.inf
    return count * _action_step_cost(scaled_norm / count)


def _grid_is_steppable(t_grid: np.ndarray) -> bool:
    """Stepping needs a finite, non-negative, non-decreasing grid."""
    if t_grid.size == 0:
        return True
    return bool(
        np.all(np.isfinite(t_grid))
        and t_grid[0] >= 0.0
        and np.all(np.diff(t_grid) >= 0.0)
    )


def action_matvec_bound(L: np.ndarray | sp.spmatrix, t_grid: np.ndarray) -> float:
    """Upper bound on the matrix-vector products the action backend will form.

    Every call runs on SciPy's exact-norm branch (see ``_SUBSTEP_NORM``), where
    SciPy's ``(m, s)`` is the minimiser of ``m * ceil(norm / theta_m)`` over the
    same table, so the bound equals SciPy's ``sum m * s``; SciPy's early
    termination can only use fewer products. ``inf`` when the grid cannot be
    stepped or the bound is not representable.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    if not _grid_is_steppable(t_grid):
        return math.inf
    base, scale = _shifted_norm_parts(L)
    total = 0.0
    previous = 0.0
    for t in t_grid:
        tau = float(t) - previous
        previous = float(t)
        if tau == 0.0:
            continue
        with np.errstate(over="ignore"):
            total += _step_cost(_step_norm(tau, base, scale))
        if not math.isfinite(total):
            return math.inf
    return total


def dense_matvec_estimate(L: np.ndarray | sp.spmatrix, t_grid: np.ndarray) -> float:
    """Cost estimate of the dense backend, in matrix-vector-product units."""
    t_grid = np.asarray(t_grid, dtype=float)
    n = L.shape[0]
    norm = _one_norm(L)
    total = 0.0
    for t in t_grid:
        if t == 0.0:
            continue
        scaled = abs(float(t)) * norm
        if not math.isfinite(scaled):
            return math.inf
        # Compare before taking the logarithm: for a subnormal ``scaled`` the
        # quotient underflows to 0.0 and log2 would raise.
        squarings = 0 if scaled <= _THETA_13 else math.ceil(math.log2(scaled / _THETA_13))
        total += (_DENSE_BASE_MATMULS + squarings) * n + 1
    return total


def select_trajectory_backend(
    L: np.ndarray | sp.spmatrix,
    t_grid: np.ndarray,
    *,
    max_action_matvecs: float = DEFAULT_ACTION_MATVEC_BUDGET,
) -> tuple[str, float]:
    """Deterministic ``"auto"`` rule; returns ``(backend, action_matvec_bound)``.

    See the module docstring for the rule and the measurements behind it. A
    sparse generator is compared against the dense estimate like a dense one
    as long as densifying it is practical (``n <= DENSE_MAX_DIM``); above that
    the action is the only candidate, and when it is over budget as well the
    rule refuses with :class:`UnrepresentableTrajectoryError` rather than
    materialising an ``n x n`` matrix nobody asked for.
    """
    t_grid = np.asarray(t_grid, dtype=float)
    n = L.shape[0]
    bound = action_matvec_bound(L, t_grid)
    action_affordable = math.isfinite(bound) and bound <= max_action_matvecs
    if _is_sparse(L) and n > DENSE_MAX_DIM:
        if action_affordable:
            return BACKEND_ACTION, bound
        reason = (
            "the time grid is not finite, non-negative and non-decreasing, so "
            "the action backend cannot step it"
            if not _grid_is_steppable(t_grid)
            else f"its exponential-action cost bound ({bound:.3g} matrix-vector "
            f"products) exceeds the budget ({float(max_action_matvecs):.3g})"
        )
        raise UnrepresentableTrajectoryError(
            "relaxation trajectory: sparse generator of dimension "
            f"{n} > DENSE_MAX_DIM={DENSE_MAX_DIM} and {reason}; neither backend "
            "is practical"
        )
    if n < ACTION_MIN_DIM or not action_affordable:
        return BACKEND_DENSE, bound
    if bound <= dense_matvec_estimate(L, t_grid):
        return BACKEND_ACTION, bound
    return BACKEND_DENSE, bound


def _propagate_dense(
    L: np.ndarray | sp.spmatrix, rho_vec0: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:
    """Reference path: ``expm(L t) @ rho_vec0`` per grid point, fail-closed."""
    L_dense = L.toarray() if _is_sparse(L) else L  # type: ignore[union-attr]
    states = np.empty((t_grid.size, rho_vec0.size), dtype=complex)
    for k, t in enumerate(t_grid):
        if t == 0.0:
            states[k] = rho_vec0
            continue

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scaled = np.asarray(L_dense * t)
        if not np.all(np.isfinite(scaled)):
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: L*t contains non-finite entries at "
                f"t={float(t):.6g}; the requested dimensionless propagation "
                "is outside the current float64 dense-expm domain"
            )

        try:
            # SciPy's scaling-and-squaring path can emit RuntimeWarning outside
            # NumPy's errstate. Convert that into the same explicit domain
            # failure so warnings-as-errors and ordinary callers agree.
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning)
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    propagator = sla.expm(scaled)
        except (
            RuntimeWarning,
            OverflowError,
            ValueError,
            np.linalg.LinAlgError,
            sla.LinAlgError,
            # SciPy >= 1.9 raises RuntimeError for a nonzero internal LAPACK
            # status ("scipy.linalg.expm got an internal LAPACK error").
            # MemoryError is deliberately NOT converted: it is a resource
            # failure, not a statement about the propagation's domain.
            RuntimeError,
        ) as exc:
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: scipy.linalg.expm could not represent "
                f"a finite propagator at t={float(t):.6g}; use a different "
                "numerical propagation method or time window"
            ) from exc

        if not np.all(np.isfinite(propagator)):
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: scipy.linalg.expm returned a non-finite "
                f"propagator at t={float(t):.6g} although L*t is finite"
            )

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            rho_vec_t = propagator @ rho_vec0
        if not np.all(np.isfinite(rho_vec_t)):
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: the propagated state became non-finite "
                f"at t={float(t):.6g}"
            )
        states[k] = rho_vec_t
    return states


def _propagate_action(
    L: np.ndarray | sp.spmatrix, rho_vec0: np.ndarray, t_grid: np.ndarray
) -> np.ndarray:
    """Exponential-action path: step ``v <- exp(tau L) v`` along the grid.

    Each grid step ``tau`` is split into ``k`` equal sub-steps with
    ``(tau / k) * ||L - mu I||_1 <= _SUBSTEP_NORM`` so that SciPy never takes
    its randomised norm-estimation branch. ``k * (tau / k)`` may differ from
    ``tau`` by ``O(k)`` units in the last place of ``tau``; for ``k == 1``
    (every benign grid) the step is exactly ``t_k - t_{k-1}``.
    """
    base, scale = _shifted_norm_parts(L)
    states = np.empty((t_grid.size, rho_vec0.size), dtype=complex)
    v = np.asarray(rho_vec0, dtype=complex)
    previous = 0.0
    for k, t in enumerate(t_grid):
        tau = float(t) - previous
        if tau == 0.0:
            # t == 0 keeps rho_0 exactly (as the dense path does); a repeated
            # grid point repeats the state exactly.
            states[k] = v
            continue

        with np.errstate(over="ignore"):
            count = _substep_count(_step_norm(tau, base, scale))
        if not math.isfinite(count):
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: ||L - mu I||_1 * tau is not representable "
                f"at t={float(t):.6g}; outside the float64 exponential-action domain"
            )
        substeps = int(count)
        tau_sub = tau / substeps
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            scaled = L * tau_sub
            # Trace of the already-scaled matrix: tr(L) itself may overflow
            # where tr(L * tau_sub) does not.
            trace_sub = _trace(scaled)
        scaled_values = scaled.data if _is_sparse(scaled) else scaled  # type: ignore[union-attr]
        if not (np.all(np.isfinite(scaled_values)) and np.isfinite(trace_sub)):
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: L*tau contains non-finite entries at "
                f"t={float(t):.6g}; the requested dimensionless propagation "
                "is outside the float64 exponential-action domain"
            )
        for _ in range(substeps):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                        v = expm_multiply(scaled, v, traceA=trace_sub)
            except (
                RuntimeWarning,
                OverflowError,
                ValueError,
                np.linalg.LinAlgError,
                RuntimeError,
            ) as exc:
                raise UnrepresentableTrajectoryError(
                    "relaxation trajectory: scipy.sparse.linalg.expm_multiply could "
                    f"not represent the propagated state at t={float(t):.6g}"
                ) from exc
            if not np.all(np.isfinite(v)):
                raise UnrepresentableTrajectoryError(
                    "relaxation trajectory: the propagated state became non-finite "
                    f"at t={float(t):.6g}"
                )
        states[k] = v
        previous = float(t)
    return states


def propagate_trajectory(
    L: np.ndarray | sp.spmatrix,
    rho_vec0: np.ndarray,
    t_grid: np.ndarray,
    *,
    backend: str = BACKEND_AUTO,
    max_action_matvecs: float = DEFAULT_ACTION_MATVEC_BUDGET,
) -> TrajectoryPropagation:
    """Propagate ``vec(rho)`` along ``t_grid`` under ``d/dt vec(rho) = L vec(rho)``.

    ``backend`` is ``"auto"`` (deterministic rule, see module docstring),
    ``"dense_expm"`` (reference) or ``"expm_action"``. Forcing the action
    backend on a grid it cannot step raises ``ValueError``; on a cost bound
    above ``max_action_matvecs`` it raises
    :class:`UnrepresentableTrajectoryError` instead of running unbounded.
    """
    if backend not in (BACKEND_AUTO, *TRAJECTORY_BACKENDS):
        allowed = ", ".join(repr(b) for b in (BACKEND_AUTO, *TRAJECTORY_BACKENDS))
        raise ValueError(f"backend must be one of {allowed}, got {backend!r}")
    if not (_is_sparse(L) or isinstance(L, np.ndarray)):
        L = np.asarray(L)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be a square matrix, got shape {L.shape}")
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1:
        raise ValueError(f"t_grid must be one-dimensional, got shape {t_grid.shape}")
    if not np.all(np.isfinite(t_grid)):
        raise ValueError("propagate_trajectory: t_grid contains non-finite time points")
    rho_vec0 = np.asarray(rho_vec0)
    if rho_vec0.shape != (L.shape[0],):
        raise ValueError(
            f"rho_vec0 must have shape ({L.shape[0]},), got {rho_vec0.shape}"
        )
    # Validate the inputs themselves: the exact t == 0 shortcut returns
    # rho_vec0 without ever reaching a post-propagation finiteness check, so a
    # zero-only grid would otherwise hand a NaN state straight back. The check
    # runs AFTER conversion to the complex128 working dtype: an extended-
    # precision input (clongdouble) can be finite as given and overflow to inf
    # on the narrowing store into the complex128 trajectory (PR #174 review).
    with np.errstate(over="ignore", invalid="ignore"):
        rho_vec0 = rho_vec0.astype(np.complex128)
    if not np.all(np.isfinite(rho_vec0)):
        raise ValueError(
            "propagate_trajectory: rho_vec0 contains non-finite entries or entries "
            "not representable in complex128"
        )
    if not _is_sparse(L) and np.dtype(L.dtype).itemsize > np.dtype(np.complex128).itemsize:
        # Extended-precision generators would otherwise make every product
        # extended-precision and narrow silently on store; work in double.
        with np.errstate(over="ignore", invalid="ignore"):
            L = L.astype(np.complex128)
    try:
        require_finite_generator(L, builder="propagate_trajectory")
    except ValueError as exc:
        raise ValueError("propagate_trajectory: L contains non-finite entries") from exc

    if backend == BACKEND_AUTO:
        chosen, bound = select_trajectory_backend(
            L, t_grid, max_action_matvecs=max_action_matvecs
        )
    elif backend == BACKEND_ACTION:
        if not _grid_is_steppable(t_grid):
            raise ValueError(
                "backend='expm_action' steps along the grid and needs a finite, "
                "non-negative, non-decreasing t_grid"
            )
        chosen, bound = BACKEND_ACTION, action_matvec_bound(L, t_grid)
        if not math.isfinite(bound) or bound > max_action_matvecs:
            raise UnrepresentableTrajectoryError(
                "relaxation trajectory: the exponential-action cost bound "
                f"({bound:.3g} matrix-vector products) exceeds the budget "
                f"({float(max_action_matvecs):.3g}); ||L - mu I||_1 * t is too "
                "large for the action algorithm -- use backend='dense_expm'"
            )
    else:
        chosen, bound = BACKEND_DENSE, action_matvec_bound(L, t_grid)

    if chosen == BACKEND_ACTION:
        states = _propagate_action(L, rho_vec0, t_grid)
    else:
        states = _propagate_dense(L, rho_vec0, t_grid)
    return TrajectoryPropagation(states=states, backend=chosen, action_matvec_bound=bound)
