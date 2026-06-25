"""Non-normality layer N: D8 Henrici, D9 Petermann, D10 Kreiss, D11 Bohr-AP.

D8 -- :func:`henrici_eta_n`. Schur-based departure-from-normality measure
``eta_N = sqrt(sum_{i != j} |N_ij|^2)`` where ``N`` is the strictly upper
triangle of the Schur form (Henrici 1962). Anchor: HS-adjoint, not
pi-weighted adjoint (FIX-6).

D9 -- :func:`petermann_factors`. Mode-resolved condition numbers
``K_j = ||r_j||^2 ||l_j||^2 / |<l_j, r_j>|^2``. Anchor I: complex-conjugate
pairs are INCLUDED.

    SAFE caveat (LIOU-NG-003 / NR-004): a large Petermann factor (eigenvector
    condition number) is a *necessary but not sufficient* indicator of
    transient amplification. It must NOT be read as a direct propagator
    amplification ``sup_t ||e^{tL}||``. The actual transient growth is bounded
    by the Kreiss constant (``K(L)/e <= sup_t ||e^{tL}||``, D10) and its initial
    slope by the numerical abscissa ``omega(L)`` (D15); the pseudospectrum (D13)
    gives the geometric picture. Petermann is therefore *cross-evidence* about
    spectral geometry, not a standalone proof of amplification (sharpens
    LIOU-NG-001: non-normality measures are corroborating signals, not single
    proofs).

D10 -- :func:`kreiss_constant`. Mitchell SIMAX 41(4) (2020) discrete-grid
algorithm: ``K = sup_{sigma > 0, omega} sigma ||((sigma + i omega) I - L)^{-1}||``.

D11 -- :func:`bohr_arithmetic_progression`. Basso arXiv:2510.07267 (2025)
length of the longest arithmetic-progression of imaginary parts in the
spectrum. Returns ``(length, pauli_bound)`` where the Pauli bound is
``log_2(d)`` (depth Bound of the AP). Anchor M: this is D11, NOT the
resolvent peak (which is D11b in :mod:`liouscope.diagnostics.resolvent`).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_DIV, EPS_GAP
from .._types import NonNormalityResult


def henrici_eta_n(A: np.ndarray) -> float:
    """Henrici departure-from-normality measure via Schur form."""
    A = np.asarray(A)
    T, _ = sla.schur(A, output="complex")
    # The diagonal carries eigenvalues; the strictly-upper triangle is N.
    n = T.shape[0]
    # Frobenius norm of the strictly upper triangle.
    fro_squared = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            fro_squared += abs(T[i, j]) ** 2
    return float(np.sqrt(fro_squared))


def petermann_factors(
    L_super: np.ndarray,
    *,
    atol: float = EPS_GAP,
) -> tuple[np.ndarray, np.ndarray]:
    """Petermann factors per non-zero eigenmode.

    Returns ``(eigenvalues, K_j)`` with the eigenvalues sorted by real part
    descending, the steady-state mode (``lambda = 0``) dropped, and
    complex-conjugate pairs retained (anchor I).

    Interpretation caveat (LIOU-NG-003): ``K_j`` quantifies the conditioning of
    eigenmode ``j``; it does NOT equal the realised transient amplification of
    the propagator. Use D10 (Kreiss) / D15 (numerical abscissa) / D13
    (pseudospectrum) to bound ``sup_t ||e^{tL}||``. See the module docstring.
    """
    L_super = np.asarray(L_super)
    eigvals, vl, vr = sla.eig(L_super, left=True, right=True)
    # Normalise eigenvectors
    K_list: list[float] = []
    for j in range(eigvals.size):
        r = vr[:, j]
        l = vl[:, j]
        denom = abs(np.vdot(l, r)) ** 2
        if denom <= EPS_DIV:
            K = np.inf
        else:
            K = float((np.linalg.norm(r) ** 2 * np.linalg.norm(l) ** 2) / denom)
        K_list.append(K)
    K_arr = np.asarray(K_list)
    # Drop zero eigenvalue but KEEP complex pairs.
    mask = np.abs(eigvals) > atol
    eigvals_filt = eigvals[mask]
    K_filt = K_arr[mask]
    order = np.argsort(-np.real(eigvals_filt))
    return eigvals_filt[order], K_filt[order]


def kreiss_constant(
    L_super: np.ndarray,
    *,
    n_sigma: int = 24,
    n_omega: int = 25,  # ungerade -> das Default-Gitter enthaelt omega=0 (Peak-Lage)
) -> float:
    """Kreiss constant via Mitchell 2020 grid search.

    Estimates ``K = sup_{sigma > 0, omega} sigma * ||((sigma + i omega) I - L)^{-1}||``.

    The grid is set adaptively from the spectrum.
    """
    L_super = np.asarray(L_super)
    eigvals = sla.eigvals(L_super)
    re_part = np.real(eigvals)
    re_max = float(re_part.max())
    re_min = float(re_part.min())
    sigma_lo = 1.0e-3
    sigma_hi = max(1.0, abs(re_min) + 1.0)
    omega_max = max(1.0, float(np.max(np.abs(np.imag(eigvals)))) + 1.0)

    sigmas = np.geomspace(sigma_lo, sigma_hi, n_sigma)
    # Default n_omega ist ungerade -> Gitter enthaelt omega=0 (wo der sup oft sitzt;
    # ein gerades Gitter springt 0 ueber). Explizite n_omega bleiben respektiert.
    omegas = np.linspace(-omega_max, omega_max, n_omega)
    n = L_super.shape[0]
    eye = np.eye(n, dtype=complex)

    K = 0.0
    for sigma in sigmas:
        # sigma must shift the spectrum to the right of the imaginary axis.
        shift = sigma + re_max
        if shift <= 0:
            continue
        for omega in omegas:
            z = shift + 1j * omega
            try:
                Ainv = sla.solve(z * eye - L_super, eye)
            except np.linalg.LinAlgError:
                continue
            norm = float(sla.svdvals(Ainv)[0])
            cand = float(sigma * norm)
            if cand > K:
                K = cand
    return K


def bohr_arithmetic_progression(
    eigenvalues: np.ndarray,
    d: int,
    *,
    tol: float = 1.0e-3,
) -> tuple[int, float]:
    """Longest arithmetic progression of imaginary parts (Basso 2025).

    Returns ``(length, pauli_bound)`` where ``pauli_bound = log_2(d)``
    on the depth ``D(omega_n)`` for primitive Davies generators.

    Implementation: cluster ``|Im(lambda)|`` values into approximate APs by
    sweeping potential common differences and counting compatible nearest-
    integer projections.
    """
    eigenvalues = np.asarray(eigenvalues)
    imags = np.unique(np.round(np.abs(np.imag(eigenvalues)), 9))
    imags = imags[imags > tol]
    if imags.size < 2:
        return 1, float(np.log2(max(d, 2)))

    best_len = 1
    for i in range(imags.size):
        for j in range(i + 1, imags.size):
            common_diff = imags[j] - imags[i]
            if common_diff <= tol:
                continue
            length = 2
            current = imags[j]
            while True:
                next_val = current + common_diff
                # Look for any imag within tolerance
                if np.any(np.abs(imags - next_val) <= tol):
                    length += 1
                    current = next_val
                else:
                    break
            if length > best_len:
                best_len = length
    pauli_bound = float(np.log2(max(d, 2)))
    return best_len, pauli_bound


def compute_nonnormality_layer(L_super: np.ndarray) -> NonNormalityResult:
    """Run D8, D9, D10, D11 together."""
    L_super = np.asarray(L_super)
    eta_n = henrici_eta_n(L_super)
    eigvals_filt, K_factors = petermann_factors(L_super)
    K_max = float(np.max(K_factors)) if K_factors.size else 1.0
    K = kreiss_constant(L_super)
    n2 = L_super.shape[0]
    d = int(round(np.sqrt(n2)))
    ap_length, pauli_bound = bohr_arithmetic_progression(eigvals_filt, d)
    return NonNormalityResult(
        henrici_eta=eta_n,
        petermann_max=K_max,
        petermann_factors=K_factors,
        kreiss=K,
        bohr_ap_length=ap_length,
        bohr_ap_pauli_bound=pauli_bound,
    )
