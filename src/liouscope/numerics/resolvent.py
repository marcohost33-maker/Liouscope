"""Resolvent computations.

Uses :class:`scipy.sparse.linalg.splu` (LU factorisation, SuperLU backend) to
solve ``(z I - L) x = b``. BiCGSTAB does not converge reliably on
Liouvillian-resolvent systems (anchor E / patch E10).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg.lapack import ztrtrs

from .conditioning import _unit_scaled
from .linalg import require_finite_square_2d

# Dense explicit-inverse + SVD is exact and O(n^3); used up to this size. Above
# it the SuperLU shift-and-invert power iteration (O(n) memory, sparse solves)
# takes over for a single resolvent norm, and grid sweeps over many ``z`` use
# :class:`ShiftedSchurResolvent` (issue #178). resolvent_apply_superlu uses a
# larger cutoff because a single dense solve is much cheaper than a full SVD.
_DENSE_NORM_CUTOFF = 128
# Krylov budget of one ShiftedSchurResolvent evaluation, its convergence
# tolerance on the Ritz residual relative to the Ritz value, and how often the
# small tridiagonal eigenproblem is solved to test it. A point that does not
# converge within the budget is recomputed by a dense SVD of the shifted
# triangle, so the budget bounds the work, never the accuracy. Measured on the
# D10b grids of the damped Ising chain (N = 4) and the driven ladder (d = 16):
# median 36 / 44 steps, maximum 96 / 112, no fallback.
_LANCZOS_MAX_STEPS = 128
_LANCZOS_RTOL = 1.0e-10
_LANCZOS_CHECK_EVERY = 4
# A Ritz residual certifies SOME eigenvalue. If the start vector carries
# (almost) no component of the dominant eigenvector -- e.g. it lies in an
# invariant subspace of subdominant modes -- the residual is small for the
# wrong one (PR #189 review). Before a Ritz value is accepted, an independent
# probe takes this many power steps; a Rayleigh quotient above the Ritz value
# by more than the relative margin means a larger mode was missed, and the
# point is recomputed by the dense SVD. The margin sits far above rounding of
# the Rayleigh quotient, so a missed mode closer than it to the accepted value
# is not detected -- the value is then wrong by at most that margin.
_PROBE_STEPS = 3
_PROBE_RTOL = 1.0e-8
_DENSE_SOLVE_CUTOFF = 256
# Power-iteration budget for the largest singular value of the resolvent. 200
# lets moderately clustered top singular values (sigma_2/sigma_1 >~ 0.98)
# converge; tighter clusters exhaust the budget but, because it is the dominant
# *value* that is sought (not the eigenvector), the estimate is still accurate
# to ~1e-4 relative and a ResolventConvergenceWarning is emitted.
_POWER_ITERS = 200
_POWER_TOL = 1.0e-9


class ResolventConvergenceWarning(UserWarning):
    """Emitted when the resolvent-norm power iteration hits its budget.

    The cross-product power iteration for ``||(zI - L)^{-1}||_2`` did not meet
    its relative tolerance within :data:`_POWER_ITERS` steps -- the top singular
    values of the resolvent are tightly clustered. The returned value is a
    slight underestimate (lower bound); it is typically still accurate to
    ~1e-4 relative.
    """


def resolvent_apply_superlu(L: np.ndarray, z: complex, b: np.ndarray) -> np.ndarray:
    """Solve ``(z I - L) x = b`` via SuperLU.

    Falls back to dense ``scipy.linalg.solve`` for small dense matrices,
    using LU with partial pivoting (``zgetrf``).
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= _DENSE_SOLVE_CUTOFF:
        solved: np.ndarray = sla.solve(A, b, assume_a="gen")
        return solved
    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    sp_solved: np.ndarray = lu.solve(np.asarray(b, dtype=complex))
    return sp_solved


def resolvent_norm(L: np.ndarray, z: complex) -> float:
    """Return ``|| (z I - L)^{-1} ||_2``, the resolvent 2-norm at ``z``.

    This is the central quantity of pseudospectra (Trefethen, *Pseudospectra
    of Linear Operators*, SIAM Rev. 1997): ``||(zI - L)^{-1}||_2`` equals the
    reciprocal of the smallest singular value of ``zI - L``.

    Two regimes:

    * ``n <= 128`` -- form the explicit inverse and take its largest singular
      value (exact, dense ``scipy.linalg``).
    * ``n > 128`` -- the standard EigTool shift-and-invert approach: factorise
      ``A = zI - L`` once (SuperLU) and run a power iteration for the largest
      singular value of ``M = A^{-1}`` on the cross-product ``M^H M``, applying
      ``M`` via ``lu.solve`` and ``M^H = (A^H)^{-1}`` via ``lu.solve(trans="H")``.
      Power iteration is used (rather than Lanczos) deliberately: it is the
      *dominant value* that is needed, and value-convergence -- unlike
      eigenvector-convergence -- is robust to clustering of the top singular
      values (when ``sigma_1 ~ sigma_2`` any vector in the near-degenerate top
      subspace yields ``||M^H M x|| ~ sigma_1^2``). Empirically the estimate
      stays within ~1e-3 relative of the dense reference across separated and
      clustered spectra alike, so the extra ARPACK machinery -- and its own
      non-convergence failure modes -- buys nothing here.

    If the power iteration exhausts its budget without meeting the tolerance, a
    :class:`ResolventConvergenceWarning` is emitted and the returned value is a
    lower bound (still typically accurate to ~1e-4 relative).
    """
    L = np.asarray(L)
    n = L.shape[0]
    eye = np.eye(n, dtype=complex)
    A = z * eye - L
    if n <= _DENSE_NORM_CUTOFF:
        Ainv = sla.solve(A, eye)
        return float(sla.svdvals(Ainv)[0])

    A_sp = sp.csc_matrix(A)
    lu = spla.splu(A_sp)
    rng = np.random.default_rng(0)
    x: np.ndarray = np.asarray(
        rng.standard_normal(n) + 1j * rng.standard_normal(n), dtype=complex
    )
    x /= np.linalg.norm(x)
    sigma = 0.0
    sigma_prev = 0.0
    converged = False
    for _ in range(_POWER_ITERS):
        y = lu.solve(x)
        # Apply M^H = (A^{-1})^H = (A^H)^{-1} via the LU's conjugate-transpose
        # solve. NOTE: lu.solve(y.conj()).conj() computes conj(A)^{-1} y, which
        # equals (A^H)^{-1} y only for symmetric A and is wrong for the
        # non-normal Liouvillians this kernel targets -- use trans="H" instead.
        z_vec = lu.solve(y, trans="H")
        sigma = float(np.linalg.norm(z_vec))
        if sigma == 0.0:
            converged = True
            break
        x = z_vec / sigma
        if abs(sigma - sigma_prev) < _POWER_TOL * max(1.0, sigma):
            converged = True
            break
        sigma_prev = sigma
    if not converged:
        warnings.warn(
            f"resolvent_norm power iteration did not converge in {_POWER_ITERS} "
            f"steps (z={z!r}); the top singular values of the resolvent are "
            "clustered. Returned value is a lower bound.",
            ResolventConvergenceWarning,
            stacklevel=2,
        )
    return float(np.sqrt(sigma))


def _lanczos_top_eigenvalue(
    apply: Callable[[np.ndarray], np.ndarray],
    start: np.ndarray,
    *,
    max_steps: int,
    rtol: float,
    check_every: int,
) -> float | None:
    """Largest eigenvalue of a Hermitian positive semidefinite operator.

    Lanczos with full reorthogonalisation (classical Gram-Schmidt, applied
    twice). Every ``check_every`` steps, returns the largest Ritz value once its
    residual ``beta_k |y_k|`` is at most ``rtol`` times that value -- a Ritz
    value of a Hermitian operator lies within its residual of an eigenvalue,
    and never above the largest one -- or once the Krylov space is invariant
    (``beta = 0``). Returns ``None`` when the budget runs out or any
    intermediate quantity is not finite.

    Floating-point exceptions are suppressed and replaced by explicit
    finiteness checks (PR #189 review): for a strongly non-normal operator the
    products stay finite while a sum of squares such as ``||w||`` overflows,
    and that must send the point to the dense fallback, not raise.
    """
    n = start.size
    k_max = min(max_steps, n)
    basis = np.empty((n, k_max), dtype=complex, order="F")
    tri = np.zeros((k_max, k_max))
    basis[:, 0] = start
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for k in range(k_max):
            w = apply(basis[:, k])
            if not np.all(np.isfinite(w)):
                return None
            span = basis[:, : k + 1]
            coeff = span.conj().T @ w
            w = w - span @ coeff
            again = span.conj().T @ w
            w = w - span @ again
            tri[k, k] = float(np.real(coeff[k] + again[k]))
            beta = float(np.linalg.norm(w))
            if not (np.isfinite(beta) and np.isfinite(tri[k, k])):
                return None
            if (k + 1) % check_every == 0 or k + 1 == k_max or beta == 0.0:
                values, vectors = np.linalg.eigh(tri[: k + 1, : k + 1])
                theta = float(values[-1])
                if not np.isfinite(theta) or theta <= 0.0:
                    return None
                if beta * abs(vectors[-1, -1]) <= rtol * theta:
                    return theta
            if k + 1 < k_max:
                basis[:, k + 1] = w / beta
                tri[k, k + 1] = tri[k + 1, k] = beta
    return None


def _probe_finds_larger_mode(
    apply: Callable[[np.ndarray], np.ndarray],
    probe: np.ndarray,
    theta: float,
    *,
    steps: int,
    rtol: float,
) -> bool:
    """Whether power steps from ``probe`` exceed the accepted Ritz value ``theta``.

    Every Rayleigh quotient of a Hermitian operator is at most its largest
    eigenvalue, so a quotient above ``theta * (1 + rtol)`` proves that Lanczos
    converged to a smaller mode. Non-finite intermediates also count as
    "found", which sends the point to the dense fallback.
    """
    v = probe
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(steps):
            w = apply(v)
            if not np.all(np.isfinite(w)):
                return True
            quotient = float(np.real(np.vdot(v, w)))
            norm = float(np.linalg.norm(w))
            if not (np.isfinite(quotient) and np.isfinite(norm)) or norm == 0.0:
                return True
            if quotient > theta * (1.0 + rtol):
                return True
            v = w / norm
    return False


class ShiftedSchurResolvent:
    """Many ``sigma_min(zI - L)`` evaluations of one operator (issue #178).

    The resolvent and non-normality layers evaluate ``sigma_min(zI - L)``, or
    its reciprocal ``||(zI - L)^{-1}||_2``, on grids of 201 to 1250 points of
    ONE operator. Measured at ``d = 16`` (n = 256, single thread) those sweeps
    were about 90 % of a ``diagnose()`` call: a dense SVD per point for D10 and
    D13, and a SuperLU factorisation of the dense matrix plus a power iteration
    for D10b and D11b -- on average 113 iterations, with one D10b point in five
    stopping at the 200-step cap and returning a lower bound.

    This is the pseudospectrum algorithm of Trefethen (Acta Numerica 8, 1999)
    and EigTool (Wright & Trefethen, SISC 23, 2001). The complex Schur form
    ``L = U T U^H`` is computed once. Singular values are unitarily invariant,
    so ``sigma(zI - L) = sigma(zI - T)``, and the smallest one is the inverse
    square root of the largest eigenvalue of ``(zI - T)^{-1} (zI - T)^{-H}``,
    which inverse Lanczos reaches with two triangular solves, ``O(n^2)``, per
    step. A converged Ritz value is accepted only if an independent probe
    finds no larger Rayleigh quotient (:data:`_PROBE_STEPS` power steps). A
    point that fails either test, does not converge within
    :data:`_LANCZOS_MAX_STEPS`, or meets a non-finite intermediate is
    recomputed by a dense SVD of the shifted triangle: no value is a capped
    lower bound, and no :class:`ResolventConvergenceWarning` is emitted. Like
    any Krylov method with deterministic start vectors, a dominant mode
    orthogonal to BOTH fixed vectors would go unnoticed; physical generators
    do not single out those two random directions.

    The operator is rescaled by an exact power of two before the Schur
    decomposition and the result scaled back, for the same LAPACK reason as in
    :func:`liouscope.numerics.conditioning._unit_scaled` (issue #183). The
    Krylov start vector and the probe come from a private ``default_rng(0)``;
    the global NumPy random state is never touched, and each value is a pure
    function of ``(L, z)``.
    """

    def __init__(self, L: np.ndarray) -> None:
        L = require_finite_square_2d(L, name="L")
        scaled, self._exponent = _unit_scaled(L)
        T, _ = sla.schur(scaled, output="complex")
        self._minus_T = np.asfortranarray(-T)
        self.n = int(T.shape[0])
        self._diagonal = np.diag_indices(self.n)
        rng = np.random.default_rng(0)
        start = rng.standard_normal(self.n) + 1j * rng.standard_normal(self.n)
        self._start = start / np.linalg.norm(start)
        probe = rng.standard_normal(self.n) + 1j * rng.standard_normal(self.n)
        self._probe = probe / np.linalg.norm(probe)
        self.evaluations = 0
        self.dense_fallbacks = 0

    def sigma_min(self, z: complex) -> float:
        """Smallest singular value of ``zI - L``."""
        z = complex(z)
        shift = complex(
            np.ldexp(z.real, -self._exponent), np.ldexp(z.imag, -self._exponent)
        )
        A = self._minus_T.copy(order="F")
        A[self._diagonal] += shift
        self.evaluations += 1
        if not np.all(np.diag(A)):
            return 0.0

        def apply(x: np.ndarray) -> np.ndarray:
            # (zI - T)^{-1} (zI - T)^{-H} x by two triangular solves.
            y, _ = ztrtrs(A, x, lower=0, trans=2)
            w, _ = ztrtrs(A, y, lower=0, trans=0)
            out: np.ndarray = w
            return out

        theta = _lanczos_top_eigenvalue(
            apply,
            self._start,
            max_steps=_LANCZOS_MAX_STEPS,
            rtol=_LANCZOS_RTOL,
            check_every=_LANCZOS_CHECK_EVERY,
        )
        if theta is not None and _probe_finds_larger_mode(
            apply, self._probe, theta, steps=_PROBE_STEPS, rtol=_PROBE_RTOL
        ):
            theta = None
        if theta is None:
            self.dense_fallbacks += 1
            smallest = float(sla.svdvals(A)[-1])
        else:
            smallest = float(1.0 / np.sqrt(theta))
        return float(np.ldexp(smallest, self._exponent))

    def norm(self, z: complex) -> float:
        """``||(zI - L)^{-1}||_2``; raises ``LinAlgError`` where ``zI - L`` is singular."""
        smallest = self.sigma_min(z)
        if smallest == 0.0:
            raise np.linalg.LinAlgError(f"zI - L is singular at z={complex(z)!r}")
        return float(1.0 / smallest)


def resolvent_norm_evaluator(L: np.ndarray) -> Callable[[complex], float]:
    """``z -> ||(zI - L)^{-1}||_2`` for a sweep over many ``z`` of one operator.

    For ``n <= 128`` this is :func:`resolvent_norm` itself, bit-for-bit. Above
    it one :class:`ShiftedSchurResolvent` serves the whole sweep.
    """
    L = np.asarray(L)
    if L.shape[0] <= _DENSE_NORM_CUTOFF:
        return lambda z: resolvent_norm(L, z)
    return ShiftedSchurResolvent(L).norm


def sigma_min_evaluator(L: np.ndarray) -> Callable[[complex], float]:
    """``z -> sigma_min(zI - L)`` for a sweep over many ``z`` of one operator.

    For ``n <= 128`` this is the dense ``svdvals(z * I - L)[-1]`` the D13 sweeps
    have always used, bit-for-bit. Above it one :class:`ShiftedSchurResolvent`
    serves the whole sweep.
    """
    L = np.asarray(L)
    n = L.shape[0]
    if n <= _DENSE_NORM_CUTOFF:
        eye = np.eye(n, dtype=complex)
        return lambda z: float(sla.svdvals(z * eye - L)[-1])
    return ShiftedSchurResolvent(L).sigma_min
