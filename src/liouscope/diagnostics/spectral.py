"""Spectral layer S: D1 gap, D2 GNS-gap, D2b KMS-gap, D3 oscillating-mode, D4 spread.

The GNS construction uses the **Mori-Shirai 2023** definition with Gram matrix
``G_GNS = rho_ss.T (x) I`` (FIX-1, anchor B). The KMS Gram matrix is
``G_KMS = rho_ss^{1/2}.conj() (x) rho_ss^{1/2}`` (Fagnola-Umanita ``s = 1/2``).

For the GNS gap we build

    L_sym = (L + L_tilde*) / 2

where ``L_tilde*`` is the Alicki pi-weighted adjoint (anchor C, correct
direction). After Gram-similarity transform ``M = G^{1/2} L_sym G^{-1/2}``
is provably Hermitian; its eigenvalues are real and the second-largest
gives the GNS gap ``Delta_s``.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_GAP
from .._types import SpectralResult
from ..core.lindblad import steady_state
from ..numerics.adjoint import gram_adjoint, symmetrised_liouvillian
from ..numerics.linalg import certified_eigvals
from ..numerics.scale import spectral_zero_tolerance


def _gram_gns(rho: np.ndarray) -> np.ndarray:
    """``G_GNS = rho.T (x) I``. Anchor B."""
    d = rho.shape[0]
    eye = np.eye(d, dtype=rho.dtype)
    return np.kron(rho.T, eye)


def _gram_kms(rho: np.ndarray) -> np.ndarray:
    """``G_KMS = rho^{1/2}.conj() (x) rho^{1/2}``."""
    sqrt_rho = sla.sqrtm(rho)
    if np.iscomplexobj(rho):
        sqrt_rho = sqrt_rho.astype(complex)
    return np.kron(sqrt_rho.conj(), sqrt_rho)


def _real_gap_from_symmetric(M: np.ndarray, *, atol: float = EPS_GAP) -> float:
    """Return the second-largest real eigenvalue magnitude of a (numerically) Hermitian M.

    For a contractive ``L_sym`` the spectrum lives in ``Re(lambda) <= 0`` with
    a zero corresponding to the steady state. The GNS gap is
    ``Delta_s = -lambda_1`` where ``lambda_1`` is the second-largest real part.
    """
    M_sym = 0.5 * (M + M.conj().T)
    evals = np.linalg.eigvalsh(M_sym)
    sorted_evals = np.sort(evals)[::-1]
    if sorted_evals.size < 2:
        return 0.0
    # sorted_evals[0] is the steady-state zero mode (~0 for a consistent
    # Gram/adjoint pair; M is a contraction so everything else is <= 0).
    # Deflate exactly ONE zero mode and read the gap from the complement.
    # Do NOT filter all |eval| <= atol: the Hermitian part can carry
    # *additional* (near-)zero eigenvalues — isometric directions of the
    # numerical range — and then there is no certified exponential
    # contraction at all, i.e. Delta_s = 0, not the next negative level
    # (2026-07 audit A1 follow-up; skipping them inflated Delta_s past
    # Delta on generic non-detailed-balance systems).
    lam1 = float(sorted_evals[1])
    return max(0.0, -lam1)


def gns_gap(L_super: np.ndarray, rho_steady: np.ndarray) -> float:
    """D2: symmetrised GNS gap using Mori-Shirai 2023 construction."""
    rho_steady = np.asarray(rho_steady)
    d = rho_steady.shape[0]
    # Make rho positive-definite by adding a tiny regulariser (anchor J)
    rho_reg = rho_steady + 1.0e-14 * np.eye(d, dtype=rho_steady.dtype)
    L_sym = symmetrised_liouvillian(L_super, rho_reg)

    G = _gram_gns(rho_reg)
    G_half = sla.sqrtm(G)
    G_inv_half = sla.inv(G_half)
    M = G_half @ L_sym @ G_inv_half
    return _real_gap_from_symmetric(M)


def kms_gap(L_super: np.ndarray, rho_steady: np.ndarray) -> float:
    """D2b: KMS-symmetrised gap (Fagnola JFA 2025, ``s = 1/2``).

    The symmetrisation must use the adjoint w.r.t. the *KMS* Gram — mixing
    the GNS (pi-weighted) adjoint with a KMS similarity transform (pre-2026-07
    behaviour) produces a non-Hermitian ``M`` whose Hermitisation silently
    reports unphysical ``Delta_s < 0`` off the real-diagonal-``rho_ss``
    manifold (audit A1).
    """
    L_super = np.asarray(L_super)
    rho_steady = np.asarray(rho_steady)
    d = rho_steady.shape[0]
    rho_reg = rho_steady + 1.0e-14 * np.eye(d, dtype=rho_steady.dtype)
    G = _gram_kms(rho_reg)
    L_HS = L_super.conj().T
    L_sym = 0.5 * (L_HS + gram_adjoint(L_super, G))
    G_half = sla.sqrtm(G)
    G_inv_half = sla.inv(G_half)
    M = G_half @ L_sym @ G_inv_half
    return _real_gap_from_symmetric(M)


def _zero_tol(
    eigenvalues: np.ndarray, *, rtol: float | None, atol: float | None
) -> float:
    """Resolve the zero-mode tolerance, forwarding an explicit ``rtol``.

    ``rtol`` is the multiplier on the double-precision backward error
    ``eps64 * max|lambda|`` (issue #108). The library's own dense solves are
    promoted to complex128, so the default is correct for every spectrum
    LiouScope computes itself; a caller feeding eigenvalues from a genuinely
    single-precision external/GPU solver widens the filter here — e.g.
    ``rtol = 1e3 * (eps32 / eps64)`` — instead of falling back to an absolute
    floor.
    """
    if rtol is None:
        return spectral_zero_tolerance(eigenvalues, atol=atol, name="eigenvalues")
    return spectral_zero_tolerance(eigenvalues, rtol=rtol, atol=atol, name="eigenvalues")


def liouvillian_gap(
    eigenvalues: np.ndarray,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> float:
    """D1: ``Delta = -max{ Re(lambda) : lambda in sigma(L), lambda != 0 }``.

    The zero-mode separation is SCALE-RELATIVE by default (issue #108):
    ``tol = ZERO_MODE_RTOL * max|lambda|``, so ``Delta`` scales exactly as
    ``c Delta`` under a pure change of rate units ``L -> cL``. With the former
    absolute floor, a rescaled spectrum either lost every genuine mode (gap
    collapsing to ``0.0``, which unconditionally fires the gapless F5 reach
    leg) or promoted the round-off zero mode to a genuine one, yielding a
    NEGATIVE gap -- impossible for a GKSL generator. Pass ``atol`` to opt back
    into the legacy absolute floor.
    """
    eigenvalues = np.asarray(eigenvalues)
    # Filter zero eigenvalues (steady state)
    tol = _zero_tol(eigenvalues, rtol=rtol, atol=atol)
    mask = np.abs(eigenvalues) > tol
    if not mask.any():
        return 0.0
    nonzero = eigenvalues[mask]
    max_re = float(np.max(np.real(nonzero)))
    # NOTE: no clamp at zero. After the scale-relative filter a positive
    # Re(lambda) is no longer a round-off artefact of the zero mode; it means
    # the supplied generator has a genuinely unstable mode (non-GKSL input).
    # Clamping would mask that, so the negative gap stays visible as an honest
    # signal -- the #108 fix removes the round-off cause, it does not hide the
    # physical one.
    return float(-max_re)


def oscillating_mode_gap(
    eigenvalues: np.ndarray,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> float:
    """D3: minimal ``|Im(lambda)|`` over complex-conjugate pairs.

    Returns 0 if no complex pairs exist. The "is this imaginary part real or
    round-off" threshold is scale-relative (issue #108); pass ``atol`` for the
    legacy absolute floor.
    """
    eigenvalues = np.asarray(eigenvalues)
    tol = _zero_tol(eigenvalues, rtol=rtol, atol=atol)
    complex_mask = np.abs(np.imag(eigenvalues)) > tol
    if not complex_mask.any():
        return 0.0
    return float(np.min(np.abs(np.imag(eigenvalues[complex_mask]))))


def spectral_spread(
    eigenvalues: np.ndarray,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> float:
    """D4: ``max|Re(lambda)| - min|Re(lambda)|`` over non-zero eigenvalues.

    Scale-relative zero-mode separation (issue #108); pass ``atol`` for the
    legacy absolute floor.
    """
    eigenvalues = np.asarray(eigenvalues)
    tol = _zero_tol(eigenvalues, rtol=rtol, atol=atol)
    mask = np.abs(eigenvalues) > tol
    if not mask.any():
        return 0.0
    re_abs = np.abs(np.real(eigenvalues[mask]))
    return float(re_abs.max() - re_abs.min())


def compute_spectral_layer(
    L_super: np.ndarray,
    rho_steady: np.ndarray | None = None,
) -> SpectralResult:
    """Run all spectral diagnostics D1-D4."""
    L_super = np.asarray(L_super, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
    if rho_steady is None:
        rho_steady = steady_state(L_super)
    # Issue #112: the zero-mode FILTER (#108) can only be as good as the
    # spectrum it filters. On a stiff generator zgeev's deflation can drop the
    # exact zero mode entirely and put a spurious mode in its place, which no
    # tolerance can undo. certified_eigvals checks the computed spectrum
    # against the exact structural fact vec(I)^H L = 0 and repairs via a
    # different LAPACK route when the check fails.
    eigenvalues, certificate = certified_eigvals(L_super)
    # Round-13 review: certification and filtering must use ONE scale. The
    # certificate accepts a stationary residual up to the eigensolver backward
    # error ``eps * ||L||_2``; the radius-based default of
    # ``spectral_zero_tolerance`` is only a spectrum-side proxy for it. On a
    # strongly non-normal generator (measured: ``||L||_2 / max|lambda| ~
    # 3.9e3`` at 4x4) a certified-resolved zero mode survived the smaller
    # radius filter and D1 reported ``~1e-12`` -- occasionally negative --
    # instead of the true gap ``1``. The operator is in hand here, so every
    # D1-D4 filter below receives the certificate's own bound. Genuine slow
    # modes inside the coarser band are not silently swallowed: the
    # ambiguity split (#113) reports them as ``resolved=False`` and D1
    # becomes NaN below.
    zero_tol = certificate.bound
    if certificate.applicable and not certificate.certified:
        warnings.warn(
            "Spectral layer: the eigensolver returned no zero mode for a "
            f"trace-preserving generator (smallest |lambda| = "
            f"{certificate.residual:.3e}, admissible backward error = "
            f"{certificate.bound:.3e}). The generator is too stiff for a dense "
            "eigensolve in this basis, so D1/D3/D4 are NOT reliable for this "
            "system; no repair route succeeded. Treat the spectral layer as "
            "unresolved rather than as a measurement (issue #112).",
            RuntimeWarning,
            stacklevel=2,
        )
    delta = liouvillian_gap(eigenvalues, atol=zero_tol)
    if certificate.applicable and certificate.certified and not certificate.resolved:
        # Issue #113. Some eigenvalue sits inside the #108 zero-mode tolerance
        # while being far above the bare backward error -- neither machine-zero
        # nor resolved. The filter discards it as "zero" and D1 reads off the
        # next surviving mode, which is a FAST one: wrong in the fail-open
        # direction and by many orders of magnitude (measured: 5.0e7 reported
        # for a true gap of 1.07e-5 once the spectral spread reaches ~1e13).
        # Reporting that would be a claim the arithmetic cannot support, so D1
        # becomes NaN. Deliberately NOT 0.0, which means "gapless" and fires the
        # F5 reach leg, and deliberately not the fast mode.
        #
        # A merely DEGENERATE stationary manifold does not land here: its extra
        # zero modes sit at machine zero, so they are unambiguous and the gap
        # taken from the complement is correct.
        warnings.warn(
            f"Spectral layer: {certificate.ambiguous_count} eigenvalue(s) fall "
            f"inside the zero-mode tolerance ({certificate.bound:.3e}) without "
            "being machine-zero, so the slow spectrum is not resolvable at "
            "double precision. D1 is reported as NaN rather than as the next "
            "surviving mode, which would be a fast one (issue #113).",
            RuntimeWarning,
            stacklevel=2,
        )
        delta = float("nan")
    delta_s = gns_gap(L_super, rho_steady)
    kms = kms_gap(L_super, rho_steady)
    osc = oscillating_mode_gap(eigenvalues, atol=zero_tol)
    spread = spectral_spread(eigenvalues, atol=zero_tol)
    # Same separation as D3: with an absolute floor a rescaled spectrum
    # flipped this flag, and ``has_complex_pairs`` gates the A8
    # oscillatory-transient rung. The threshold is the shared operator-derived
    # bound (round-13): an imaginary part below the eigensolve backward error
    # is round-off, not an oscillating pair.
    has_complex = bool(np.any(np.abs(np.imag(eigenvalues)) > zero_tol))
    # Sort by real part descending so [0] is the steady state.
    order = np.argsort(-np.real(eigenvalues))
    return SpectralResult(
        gap=delta,
        gns_gap=delta_s,
        kms_gap=kms,
        oscillating_gap=osc,
        spectral_spread=spread,
        eigenvalues=eigenvalues[order],
        steady_state=np.asarray(rho_steady),
        has_complex_pairs=has_complex,
        zero_mode_certificate=certificate.as_dict(),
    )


