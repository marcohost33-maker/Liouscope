"""General linear-algebra helpers used across diagnostics.

The module deliberately separates numerical measurements from scientific
certificate policy. In particular, trace-preservation norms are measured with
scaled arithmetic, while applicability of the structural zero-mode theorem is
also checked equation-by-equation so unrelated large matrix entries cannot hide
a local trace defect.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .._consts import (
    EPS_HERMITICITY,
    EPS_SUPP,
    EPS_TRACE,
    VECTOR_RESIDUAL_REL_MAX,
    ZERO_MODE_AMBIGUITY_FACTOR,
    ZERO_MODE_APOSTERIORI_MARGIN,
    ZERO_MODE_EPS_FACTOR,
)
from .norms import scaled_cancellation_ratio, scaled_euclidean_norm


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

    Input is promoted to complex128 so the solver precision does not silently
    follow a caller's complex64 storage dtype.
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


@dataclass(frozen=True, slots=True)
class ZeroModeCertificate:
    """Outcome of the structural zero-mode check on a computed spectrum."""

    applicable: bool
    certified: bool
    solver: str
    residual: float
    bound: float
    trace_defect: float
    zero_mode_count: int = 1
    ambiguous_count: int = 0
    zero_tolerance: float = float("nan")

    @property
    def applied_tolerance(self) -> float:
        """``zero_tolerance`` with the pre-refinement fallback to ``bound``."""
        return self.bound if not np.isfinite(self.zero_tolerance) else self.zero_tolerance

    def zero_set_tolerance(
        self,
        eigenvalues: np.ndarray,
        *,
        atol: float | None = None,
        name: str = "eigenvalues",
    ) -> float:
        """Return the one cutoff a zero-mode filter may apply to this spectrum."""
        from .scale import spectral_zero_tolerance

        if atol is None and self.applicable:
            atol = self.applied_tolerance
        return spectral_zero_tolerance(eigenvalues, atol=atol, name=name)

    @property
    def resolved(self) -> bool:
        """True iff every mode is decidably zero or decidably non-zero."""
        return self.applicable and self.certified and self.ambiguous_count == 0

    def as_dict(self) -> dict[str, object]:
        """JSON-serialisable view for the run report (RFC 8259: no NaN/inf)."""

        def _f(x: float) -> float | None:
            return float(x) if np.isfinite(x) else None

        return {
            "applicable": self.applicable,
            "certified": self.certified,
            "resolved": self.resolved,
            "solver": self.solver,
            "residual": _f(self.residual),
            "bound": _f(self.bound),
            "trace_defect": _f(self.trace_defect),
            "zero_mode_count": int(self.zero_mode_count),
            "ambiguous_count": int(self.ambiguous_count),
            "zero_tolerance": _f(self.applied_tolerance),
        }


def operator_zero_tolerance(
    L_super: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    name: str = "L_super",
) -> float:
    """Operator-derived zero-mode tolerance ``rtol * eps * ||L||_2``."""
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    L_c = np.asarray(require_finite_square_2d(L_super, name=name), dtype=complex)
    norm2 = float(np.linalg.norm(L_c, 2)) if L_c.size else 0.0
    return float(rtol * float(np.finfo(float).eps) * norm2)


def _hilbert_dimension_from_superoperator(L_super: np.ndarray) -> int | None:
    """Return d for a d^2 x d^2 superoperator, else ``None``."""
    arr = np.asarray(L_super)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    n = int(arr.shape[0])
    d = int(round(np.sqrt(n)))
    return d if d * d == n else None


def trace_preservation_defect(L_super: np.ndarray) -> tuple[float, float]:
    """Return robust ``(||vec(I)^H L||_2, ||L||_F)``.

    Both quantities are measurements and therefore use the same xLASSQ-style
    scaled Euclidean norm. They remain finite whenever the true float64 norm is
    representable; policy is handled separately by
    :func:`trace_preservation_equation_defect` and the certificate gates.
    """
    L_super = np.asarray(L_super)
    d = _hilbert_dimension_from_superoperator(L_super)
    if d is None:
        return float("nan"), float("nan")
    vec_i = np.eye(d, dtype=complex).reshape(-1, order="F")
    defect_vector = vec_i.conj() @ L_super
    return scaled_euclidean_norm(defect_vector), scaled_euclidean_norm(L_super)


def trace_preservation_equation_defect(L_super: np.ndarray) -> float:
    """Return the worst componentwise relative defect of ``vec(I)^H L = 0``.

    A global quotient ``||vec(I)^H L|| / ||L||`` is useful as a measurement,
    but it is too weak as the precondition for an exact zero-mode theorem: an
    unrelated entry of size ``1e308`` can make an order-one trace defect look
    relatively tiny. Each column of ``vec(I)^H L = 0`` is therefore judged
    against only the terms that actually participate in that scalar equation:

    ``eta_j = |sum_i L_(ii),j| / sum_i |L_(ii),j|``.

    The returned value is ``max_j eta_j``. This is a componentwise relative
    backward-error criterion, analogous to LAPACK's BERR philosophy. The ratio
    is evaluated with power-of-two scaling, so it is invariant to representable
    rate-unit rescaling and remains defined in the subnormal range.
    """
    L_super = np.asarray(L_super)
    d = _hilbert_dimension_from_superoperator(L_super)
    if d is None:
        return float("nan")
    if not np.all(np.isfinite(L_super)):
        return float("nan")
    identity_rows = np.arange(d, dtype=int) * (d + 1)
    worst = 0.0
    for j in range(L_super.shape[1]):
        ratio = scaled_cancellation_ratio(L_super[identity_rows, j])
        if not np.isfinite(ratio):
            return float("nan")
        worst = max(worst, ratio)
    return float(worst)


def band_discriminates(
    magnitudes: np.ndarray, zero_count: int, bound: float
) -> bool:
    """Return whether the zero-mode acceptance band rejected anything."""
    if bound <= 0.0:
        return True
    return zero_count < int(magnitudes.size)


def certified_nonzero_modes(
    L_c: np.ndarray,
    eigenvalues: np.ndarray,
    in_band: np.ndarray,
    *,
    right_vectors: np.ndarray | None = None,
    left_vectors: np.ndarray | None = None,
    margin: float = ZERO_MODE_APOSTERIORI_MARGIN,
) -> np.ndarray:
    """Identify in-band modes that a posteriori residuals prove non-stationary."""
    idx = np.flatnonzero(in_band)
    out = np.zeros(eigenvalues.shape, dtype=bool)
    if idx.size < 2:
        return out
    idx = idx[idx != idx[int(np.argmin(np.abs(eigenvalues[idx])))]]

    vr, vl, ref = right_vectors, left_vectors, eigenvalues
    if vr is None or vl is None:
        try:
            ref, vl, vr = sla.eig(L_c, left=True, right=True)
        except (ValueError, sla.LinAlgError):
            return out

    tiny = np.finfo(float).tiny
    for i in idx:
        lam = eigenvalues[i]
        j = int(np.argmin(np.abs(ref - lam)))
        mag = float(abs(ref[j]))
        if abs(ref[j] - lam) > 0.1 * max(abs(lam), mag) or mag == 0.0:
            continue
        x = vr[:, j] / max(float(np.linalg.norm(vr[:, j])), tiny)
        y = vl[:, j] / max(float(np.linalg.norm(vl[:, j])), tiny)
        sep = float(abs(np.vdot(y, x)))
        if sep <= 0.0:
            continue
        res = max(
            float(np.linalg.norm(L_c @ x - ref[j] * x)),
            float(np.linalg.norm(L_c.conj().T @ y - np.conj(ref[j]) * y)),
        )
        if mag > margin * (res / sep):
            out[i] = True
    return out


def refine_zero_band(
    L_c: np.ndarray,
    eigenvalues: np.ndarray,
    magnitudes: np.ndarray,
    bound: float,
    *,
    right_vectors: np.ndarray | None = None,
    left_vectors: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return refined zero membership and its corresponding filter tolerance."""
    nonzero = certified_nonzero_modes(
        L_c,
        eigenvalues,
        magnitudes <= bound,
        right_vectors=right_vectors,
        left_vectors=left_vectors,
    )
    in_band = (magnitudes <= bound) & ~nonzero
    if not nonzero.any():
        return in_band, bound
    hi = float(np.min(magnitudes[nonzero]))
    lo = float(np.max(magnitudes[in_band])) if in_band.any() else 0.0
    if lo >= hi:
        return magnitudes <= bound, bound
    if lo <= 0.0:
        return in_band, hi * 0.5
    return in_band, float(np.sqrt(lo) * np.sqrt(hi))


def _tp_certificate_applicable(
    L_c: np.ndarray,
    tp_defect: float,
    fro: float,
    tp_rtol: float,
) -> bool:
    """Return whether trace preservation is established strongly enough.

    The historical normwise criterion is retained for compatibility and broad
    scale checking. The componentwise equation criterion is the load-bearing
    addition from #130: it prevents unrelated large entries from diluting a
    local violation of the exact trace equations.
    """
    equation_defect = trace_preservation_equation_defect(L_c)
    if not np.isfinite(tp_defect) or not np.isfinite(fro):
        return False
    if not np.isfinite(equation_defect):
        return False
    if equation_defect > tp_rtol:
        return False
    if tp_defect > tp_rtol * fro:
        return False
    return True


def certified_eigvals(
    L_super: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    tp_rtol: float = 1.0e-10,
) -> tuple[np.ndarray, ZeroModeCertificate]:
    """Eigenvalues checked against the exact trace-preserving zero-mode fact."""
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if not np.isfinite(tp_rtol) or tp_rtol < 0.0:
        raise ValueError(f"tp_rtol must be finite and non-negative, got {tp_rtol}")
    L_super = require_finite_square_2d(L_super, name="L_super")
    L_c = np.asarray(L_super, dtype=complex)
    tp_defect, fro = trace_preservation_defect(L_c)
    norm2 = float(np.linalg.norm(L_c, 2)) if L_c.size else 0.0
    bound = rtol * float(np.finfo(float).eps) * norm2

    primary: np.ndarray | None
    primary_error: sla.LinAlgError | None
    try:
        primary = np.asarray(eig_nonhermitian(L_c).eigenvalues)
        primary_error = None
    except sla.LinAlgError as exc:
        primary, primary_error = None, exc

    if not _tp_certificate_applicable(L_c, tp_defect, fro, tp_rtol):
        if primary is None:
            assert primary_error is not None
            raise primary_error
        return primary, ZeroModeCertificate(
            applicable=False,
            certified=False,
            solver="zgeev",
            residual=float(np.min(np.abs(primary))) if primary.size else float("nan"),
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=0,
        )

    def _candidates() -> Iterator[tuple[str, np.ndarray]]:
        if primary is not None:
            yield ("zgeev", primary)
        if not np.any(L_c.imag):
            with contextlib.suppress(ValueError, sla.LinAlgError):
                yield ("dgeev-real", np.linalg.eigvals(L_c.real).astype(complex))
        with contextlib.suppress(ValueError, sla.LinAlgError):
            yield ("zgees-schur", np.diag(sla.schur(L_c, output="complex")[0]))
        with contextlib.suppress(ValueError, sla.LinAlgError):
            balanced = sla.matrix_balance(L_c, permute=True)[0]
            yield ("balanced-zgeev", np.linalg.eigvals(balanced))

    split = ZERO_MODE_AMBIGUITY_FACTOR * float(np.finfo(float).eps) * norm2
    best: tuple[str, np.ndarray, float] | None = None
    ambiguous_best: tuple[str, np.ndarray, float, int, int, float] | None = None
    for name, ev in _candidates():
        magnitudes = np.abs(ev)
        residual = float(np.min(magnitudes)) if ev.size else float("inf")
        if residual <= bound:
            in_band, zero_tolerance = refine_zero_band(L_c, ev, magnitudes, bound)
            zero_count = int(np.count_nonzero(in_band))
            ambiguous = int(np.count_nonzero(in_band & (magnitudes > split)))
            if not band_discriminates(magnitudes, zero_count, bound):
                if best is None or residual < best[2]:
                    best = (name, ev, residual)
                continue
            if ambiguous == 0:
                return ev, ZeroModeCertificate(
                    applicable=True,
                    certified=True,
                    solver=name,
                    residual=residual,
                    bound=bound,
                    trace_defect=tp_defect,
                    zero_mode_count=zero_count,
                    zero_tolerance=zero_tolerance,
                )
            if ambiguous_best is None or ambiguous < ambiguous_best[4]:
                ambiguous_best = (
                    name,
                    ev,
                    residual,
                    zero_count,
                    ambiguous,
                    zero_tolerance,
                )
            continue
        if best is None or residual < best[2]:
            best = (name, ev, residual)

    if ambiguous_best is not None:
        name, ev, residual, zero_count, ambiguous, zero_tolerance = ambiguous_best
        return ev, ZeroModeCertificate(
            applicable=True,
            certified=True,
            solver=name,
            residual=residual,
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=zero_count,
            ambiguous_count=ambiguous,
            zero_tolerance=zero_tolerance,
        )
    if best is None:
        assert primary_error is not None
        raise primary_error
    return best[1], ZeroModeCertificate(
        applicable=True,
        certified=False,
        solver=best[0],
        residual=best[2],
        bound=bound,
        trace_defect=tp_defect,
        zero_mode_count=0,
    )


def certified_eig(
    L_super: np.ndarray,
    *,
    rtol: float = ZERO_MODE_EPS_FACTOR,
    tp_rtol: float = 1.0e-10,
) -> tuple[EigenDecomposition, ZeroModeCertificate]:
    """Like :func:`certified_eigvals`, also returning left/right eigenvectors."""
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError(f"rtol must be finite and non-negative, got {rtol}")
    if not np.isfinite(tp_rtol) or tp_rtol < 0.0:
        raise ValueError(f"tp_rtol must be finite and non-negative, got {tp_rtol}")
    L_super = require_finite_square_2d(L_super, name="L_super")
    L_c = np.asarray(L_super, dtype=complex)
    tp_defect, fro = trace_preservation_defect(L_c)
    norm2 = float(np.linalg.norm(L_c, 2)) if L_c.size else 0.0
    bound = rtol * float(np.finfo(float).eps) * norm2

    primary: EigenDecomposition | None
    primary_error: sla.LinAlgError | None
    try:
        primary = eig_nonhermitian(L_c, compute_left=True)
        primary_error = None
    except sla.LinAlgError as exc:
        primary, primary_error = None, exc

    if not _tp_certificate_applicable(L_c, tp_defect, fro, tp_rtol):
        if primary is None:
            assert primary_error is not None
            raise primary_error
        return primary, ZeroModeCertificate(
            applicable=False,
            certified=False,
            solver="zgeev",
            residual=(
                float(np.min(np.abs(primary.eigenvalues)))
                if primary.eigenvalues.size
                else float("nan")
            ),
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=0,
        )

    def _candidates() -> Iterator[tuple[str, EigenDecomposition]]:
        if primary is not None:
            yield ("zgeev", primary)
        if not np.any(L_c.imag):
            with contextlib.suppress(ValueError, sla.LinAlgError):
                w, vl, vr = sla.eig(L_c.real, left=True, right=True)
                yield (
                    "dgeev-real",
                    EigenDecomposition(
                        eigenvalues=np.asarray(w).astype(complex),
                        right_vectors=np.asarray(vr).astype(complex),
                        left_vectors=np.asarray(vl).astype(complex),
                    ),
                )

    def _vector_residual(decomp: EigenDecomposition) -> float:
        ev = decomp.eigenvalues
        if not ev.size:
            return 0.0
        vr = decomp.right_vectors
        vl = decomp.left_vectors
        assert vl is not None
        tiny = np.finfo(float).tiny
        res_r = np.linalg.norm(L_c @ vr - vr * ev[None, :], axis=0) / np.maximum(
            np.linalg.norm(vr, axis=0), tiny
        )
        res_l = np.linalg.norm(
            L_c.conj().T @ vl - vl * np.conj(ev)[None, :], axis=0
        ) / np.maximum(np.linalg.norm(vl, axis=0), tiny)
        res = np.maximum(res_r, res_l)
        magnitudes = np.abs(ev)
        consumed = magnitudes > bound
        if not consumed.any():
            return 0.0
        allowed = np.maximum(VECTOR_RESIDUAL_REL_MAX * magnitudes[consumed], bound)
        offending = res[consumed] > allowed
        return float(np.max(res[consumed][offending])) if offending.any() else 0.0

    split = ZERO_MODE_AMBIGUITY_FACTOR * float(np.finfo(float).eps) * norm2
    best: tuple[str, EigenDecomposition, float] | None = None
    ambiguous_best: (
        tuple[str, EigenDecomposition, float, int, int, float] | None
    ) = None
    vector_failed: tuple[str, EigenDecomposition, float, float] | None = None

    for name, decomp in _candidates():
        magnitudes = np.abs(decomp.eigenvalues)
        residual = float(np.min(magnitudes)) if magnitudes.size else float("inf")
        if residual <= bound:
            in_band, zero_tolerance = refine_zero_band(
                L_c,
                decomp.eigenvalues,
                magnitudes,
                bound,
                right_vectors=decomp.right_vectors,
                left_vectors=decomp.left_vectors,
            )
            zero_count = int(np.count_nonzero(in_band))
            ambiguous = int(np.count_nonzero(in_band & (magnitudes > split)))
            if not band_discriminates(magnitudes, zero_count, bound):
                if best is None or residual < best[2]:
                    best = (name, decomp, residual)
                continue
            if ambiguous == 0:
                vec_residual = _vector_residual(decomp)
                if vec_residual == 0.0:
                    return decomp, ZeroModeCertificate(
                        applicable=True,
                        certified=True,
                        solver=name,
                        residual=residual,
                        bound=bound,
                        trace_defect=tp_defect,
                        zero_mode_count=zero_count,
                        zero_tolerance=zero_tolerance,
                    )
                if vector_failed is None or vec_residual < vector_failed[3]:
                    vector_failed = (name, decomp, residual, vec_residual)
                continue
            if ambiguous_best is None or ambiguous < ambiguous_best[4]:
                ambiguous_best = (
                    name,
                    decomp,
                    residual,
                    zero_count,
                    ambiguous,
                    zero_tolerance,
                )
            continue
        if best is None or residual < best[2]:
            best = (name, decomp, residual)

    if ambiguous_best is not None:
        name, decomp, residual, zero_count, ambiguous, zero_tolerance = ambiguous_best
        return decomp, ZeroModeCertificate(
            applicable=True,
            certified=True,
            solver=name,
            residual=residual,
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=zero_count,
            ambiguous_count=ambiguous,
            zero_tolerance=zero_tolerance,
        )
    if vector_failed is not None:
        name, decomp, residual, vec_residual = vector_failed
        return decomp, ZeroModeCertificate(
            applicable=True,
            certified=False,
            solver=name,
            residual=max(residual, vec_residual),
            bound=bound,
            trace_defect=tp_defect,
            zero_mode_count=0,
        )
    if best is None:
        assert primary_error is not None
        raise primary_error
    return best[1], ZeroModeCertificate(
        applicable=True,
        certified=False,
        solver=best[0],
        residual=best[2],
        bound=bound,
        trace_defect=tp_defect,
        zero_mode_count=0,
    )


def require_finite_square_2d(A: np.ndarray, *, name: str = "matrix") -> np.ndarray:
    """Validate ``A`` as a finite, square, 2-D array at a public boundary."""
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


def hermiticity_defect(A: np.ndarray) -> tuple[float, float]:
    """Return ``(max|A - A^H|, max|A|)``."""
    A = np.asarray(A)
    defect = float(np.max(np.abs(A - A.conj().T))) if A.size else 0.0
    scale = float(np.max(np.abs(A))) if A.size else 0.0
    return defect, scale


def is_hermitian(
    A: np.ndarray,
    atol: float | None = None,
    *,
    rtol: float = EPS_HERMITICITY,
) -> bool:
    """Return True iff ``A`` is Hermitian, scale-relative by default."""
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    defect, scale = hermiticity_defect(A)
    tol = float(atol) if atol is not None else rtol * scale
    return bool(defect <= tol)


def is_density_matrix(
    rho: np.ndarray,
    *,
    atol_trace: float = EPS_TRACE,
    atol_psd: float = 1.0e-10,
) -> bool:
    """Return True iff ``rho`` is a valid density matrix."""
    rho = np.asarray(rho)
    if not is_hermitian(rho, atol=1.0e-9):
        return False
    if abs(np.trace(rho) - 1.0) > atol_trace:
        return False
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    return bool(evals.min() >= -atol_psd)


def support_check(
    rho_initial: np.ndarray,
    rho_steady: np.ndarray,
    *,
    eps: float = EPS_SUPP,
) -> tuple[bool, np.ndarray]:
    """Verify ``supp(rho_initial) subset supp(rho_steady)``."""
    rho_initial = np.asarray(rho_initial)
    rho_steady = np.asarray(rho_steady)
    d = rho_steady.shape[0]

    evals_steady, evecs_steady = np.linalg.eigh((rho_steady + rho_steady.conj().T) / 2)
    null_mask = evals_steady < eps
    if not null_mask.any():
        return True, rho_steady

    null_space = evecs_steady[:, null_mask]
    P_null = null_space @ null_space.conj().T
    leakage = float(np.real(np.trace(P_null @ rho_initial @ P_null)))
    ok = leakage < eps
    rho_steady_reg = rho_steady + (eps / d) * np.eye(d, dtype=rho_steady.dtype)
    return ok, rho_steady_reg


def matrix_2_norm(A: np.ndarray) -> float:
    """Spectral norm via SVD; works for non-square arrays."""
    return float(sla.svdvals(np.asarray(A))[0])
