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

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_GAP
from .._types import SpectralResult
from ..core.lindblad import steady_state
from ..numerics.adjoint import gram_matrix, symmetrised_liouvillian
from ..numerics.linalg import eig_nonhermitian


def _real_gap_from_symmetric(M: np.ndarray, *, atol: float = EPS_GAP) -> float:
    """Return the symmetrised gap from the Hermitian-part eigenvalues of ``M``.

    ``M`` is the Gram-conjugated symmetrised generator. Mori-Shirai 2023
    defines the symmetrised gap as the smallest decay rate, i.e. the
    magnitude of the largest **negative** eigenvalue. For non-detailed-
    balance systems ``M`` may have isolated **positive** eigenvalues
    associated with transient GNS-norm amplification; these are not gaps
    and must be filtered out before reporting.
    """
    M_sym = 0.5 * (M + M.conj().T)
    evals = np.linalg.eigvalsh(M_sym)
    sorted_desc = np.sort(evals)[::-1]
    if sorted_desc.size < 2:
        return 0.0
    # Closest-to-zero negative eigenvalue: that's the slowest decay direction.
    negative = sorted_desc[sorted_desc < -atol]
    if negative.size == 0:
        return 0.0
    return float(-negative[0])


def _symmetrised_gap(
    L_super: np.ndarray,
    rho_steady: np.ndarray,
    metric: str,
) -> float:
    """Common machinery for the GNS / KMS symmetrised gaps.

    Builds the metric-specific symmetrised Heisenberg generator, conjugates
    by ``G^{1/2}`` (which makes the result Hermitian) and returns the
    second-largest non-zero eigenvalue with a sign flip.
    """
    rho_steady = np.asarray(rho_steady)
    d = rho_steady.shape[0]
    rho_reg = rho_steady + 1.0e-14 * np.eye(d, dtype=rho_steady.dtype)
    L_sym = symmetrised_liouvillian(L_super, rho_reg, metric=metric)  # type: ignore[arg-type]
    G = gram_matrix(rho_reg, metric)  # type: ignore[arg-type]
    G_half = sla.sqrtm(G)
    G_inv_half = sla.inv(G_half)
    M = G_half @ L_sym @ G_inv_half
    return _real_gap_from_symmetric(M)


def gns_gap(L_super: np.ndarray, rho_steady: np.ndarray) -> float:
    """D2: symmetrised GNS gap using Mori-Shirai 2023 construction."""
    return _symmetrised_gap(L_super, rho_steady, metric="gns")


def kms_gap(L_super: np.ndarray, rho_steady: np.ndarray) -> float:
    """D2b: KMS-symmetrised gap (Fagnola JFA 2025, ``s = 1/2``).

    Uses the proper KMS Gram matrix ``G_KMS = rho_ss^{1/2} (x) rho_ss^{1/2}``
    in **both** the pi-adjoint construction and the similarity conjugation,
    so the result is genuinely the KMS-symmetrised gap rather than the
    GNS-symmetrised generator viewed through the KMS metric.
    """
    return _symmetrised_gap(L_super, rho_steady, metric="kms")


def liouvillian_gap(eigenvalues: np.ndarray, *, atol: float = EPS_GAP) -> float:
    """D1: ``Delta = -max{ Re(lambda) : lambda in sigma(L), lambda != 0 }``."""
    eigenvalues = np.asarray(eigenvalues)
    # Filter zero eigenvalues (steady state)
    mask = np.abs(eigenvalues) > atol
    if not mask.any():
        return 0.0
    nonzero = eigenvalues[mask]
    max_re = float(np.max(np.real(nonzero)))
    return float(-max_re)


def oscillating_mode_gap(eigenvalues: np.ndarray, *, atol: float = EPS_GAP) -> float:
    """D3: minimal ``|Im(lambda)|`` over complex-conjugate pairs.

    Returns 0 if no complex pairs exist.
    """
    eigenvalues = np.asarray(eigenvalues)
    complex_mask = np.abs(np.imag(eigenvalues)) > atol
    if not complex_mask.any():
        return 0.0
    return float(np.min(np.abs(np.imag(eigenvalues[complex_mask]))))


def spectral_spread(eigenvalues: np.ndarray, *, atol: float = EPS_GAP) -> float:
    """D4: ``max|Re(lambda)| - min|Re(lambda)|`` over non-zero eigenvalues."""
    eigenvalues = np.asarray(eigenvalues)
    mask = np.abs(eigenvalues) > atol
    if not mask.any():
        return 0.0
    re_abs = np.abs(np.real(eigenvalues[mask]))
    return float(re_abs.max() - re_abs.min())


def compute_spectral_layer(
    L_super: np.ndarray,
    rho_steady: np.ndarray | None = None,
) -> SpectralResult:
    """Run all spectral diagnostics D1-D4."""
    L_super = np.asarray(L_super)
    if rho_steady is None:
        rho_steady = steady_state(L_super)
    decomp = eig_nonhermitian(L_super)
    eigenvalues = decomp.eigenvalues
    delta = liouvillian_gap(eigenvalues)
    delta_s = gns_gap(L_super, rho_steady)
    kms = kms_gap(L_super, rho_steady)
    osc = oscillating_mode_gap(eigenvalues)
    spread = spectral_spread(eigenvalues)
    has_complex = bool(np.any(np.abs(np.imag(eigenvalues)) > EPS_GAP))
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
    )
