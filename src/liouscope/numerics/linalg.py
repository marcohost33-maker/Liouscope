"""General linear-algebra helpers used across diagnostics.

Includes:

* :func:`eig_nonhermitian` --- wraps ``scipy.linalg.eig`` (LAPACK ``zgeev``),
  the non-Hermitian eigensolver required by Liouvillians. Anchor D.
* :func:`is_hermitian`, :func:`is_density_matrix` --- input validation.
* :func:`support_check` --- enforces ``supp(rho_0) subset supp(rho_ss)``
  with ``eps = 1e-12`` regularisation (anchor J).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_HERMITICITY, EPS_SUPP, EPS_TRACE


@dataclass(frozen=True, slots=True)
class EigenDecomposition:
    """Right eigendecomposition of a (possibly non-Hermitian) matrix."""

    eigenvalues: np.ndarray
    right_vectors: np.ndarray
    left_vectors: np.ndarray | None = None

    @property
    def size(self) -> int:
        return self.eigenvalues.size


def eig_nonhermitian(
    A: np.ndarray,
    *,
    compute_left: bool = False,
) -> EigenDecomposition:
    """Return the eigendecomposition of a generic non-Hermitian matrix.

    Always uses LAPACK ``zgeev`` (via ``scipy.linalg.eig``). Anchor D: never
    fall back to ``eigh`` which assumes Hermiticity that Liouvillians do not
    have.

    The promotion to ``complex128`` below is what makes the "always zgeev"
    contract TRUE: unlike ``numpy.linalg`` (which computes in double
    regardless of input dtype and only casts the result back),
    ``scipy.linalg.eig`` dispatches by dtype and would run single-precision
    ``cgeev`` on a ``complex64`` input — measured at ~30x the eigenvalue error
    of the double solve on the same stored matrix. Every zero-mode/backward-
    error tolerance downstream (issue #108) is calibrated against the double
    solve, so the solver precision must not silently follow the caller's
    storage dtype. Representation error already present in single-precision
    INPUT data is the caller's data quality and is not (cannot be) undone
    here.
    """
    A = np.asarray(A, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"eig_nonhermitian expects a square matrix, got {A.shape}")
    if compute_left:
        eigenvalues, vl, vr = sla.eig(A, left=True, right=True)
        return EigenDecomposition(
            eigenvalues=np.asarray(eigenvalues),
            right_vectors=np.asarray(vr),
            left_vectors=np.asarray(vl),
        )
    eigenvalues, vr = sla.eig(A, left=False, right=True)
    return EigenDecomposition(
        eigenvalues=np.asarray(eigenvalues),
        right_vectors=np.asarray(vr),
    )


def require_finite_square_2d(A: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Validate ``A`` as a finite, square, 2-D array at a public boundary.

    Returns ``np.asarray(A)`` unchanged when valid, else raises a
    :class:`ValueError` that names the offending argument *and* the concrete
    defect. This is a fail-closed guard for public entry points: without it a
    ``NaN``/``inf``-laden Liouvillian flows into ``scipy.linalg.expm`` / ``svd``
    and surfaces as an opaque, location-blind error deep in LAPACK
    (``"array must not contain infs or NaNs"`` / ``"SVD did not converge"``),
    leaving the caller no clue which input was malformed.

    Parameters
    ----------
    A
        Candidate array.
    name
        Human-readable argument name used in error messages.
    """
    arr = np.asarray(A)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        n_nan = int(np.count_nonzero(np.isnan(arr)))
        n_inf = int(np.count_nonzero(np.isinf(arr)))
        raise ValueError(
            f"{name} contains non-finite entries "
            f"({n_nan} NaN, {n_inf} inf); diagnostics require a finite operator"
        )
    return arr


def is_hermitian(A: np.ndarray, atol: float = EPS_HERMITICITY) -> bool:
    """Return True iff ``A`` is Hermitian within ``atol``.

    ``rtol=0`` is explicit: ``np.allclose`` defaults to ``rtol=1e-5``, which
    would silently widen this gate to ``atol + 1e-5*|entry|`` (~1e-5 for O(1)
    density matrices, ~10^4x the advertised ``atol``). A validation predicate
    must mean exactly what its ``atol`` says, so the tolerance is absolute.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    return bool(np.allclose(A, A.conj().T, rtol=0.0, atol=atol))


def is_density_matrix(
    rho: np.ndarray,
    *,
    atol_trace: float = EPS_TRACE,
    atol_psd: float = 1.0e-10,
) -> bool:
    """Return True iff ``rho`` is a valid density matrix.

    Checks Hermiticity, unit trace and positive semi-definiteness.
    """
    rho = np.asarray(rho)
    if not is_hermitian(rho, atol=1.0e-9):
        return False
    if abs(np.trace(rho) - 1.0) > atol_trace:
        return False
    # Hermitian => use eigvalsh for stability.
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return bool(evals.min() >= -atol_psd)


def support_check(
    rho_initial: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps: float = EPS_SUPP,
) -> tuple[bool, np.ndarray]:
    """Verify ``supp(rho_initial) subset supp(rho_steady)``.

    Returns a tuple ``(ok, regularised_steady)`` where ``regularised_steady``
    is ``rho_steady + eps * I / d`` shifted to avoid singularities when
    computing the relative entropy ``D(rho || pi)``.

    Anchor J: this guards against NaNs when ``rho_steady`` is rank-deficient.
    """
    rho_initial = np.asarray(rho_initial)
    rho_steady = np.asarray(rho_steady)
    d = rho_steady.shape[0]

    evals_steady, evecs_steady = np.linalg.eigh((rho_steady + rho_steady.conj().T) / 2)
    null_mask = evals_steady < eps
    if not null_mask.any():
        return True, rho_steady

    null_space = evecs_steady[:, null_mask]
    # Projector onto null space
    P_null = null_space @ null_space.conj().T
    leakage = float(np.real(np.trace(P_null @ rho_initial @ P_null)))
    ok = leakage < eps
    rho_steady_reg = rho_steady + (eps / d) * np.eye(d, dtype=rho_steady.dtype)
    return ok, rho_steady_reg


def matrix_2_norm(A: np.ndarray) -> float:
    """Spectral norm via SVD; works for non-square arrays."""
    return float(sla.svdvals(np.asarray(A))[0])
