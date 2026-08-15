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

D10 -- :func:`kreiss_constant`. Finite-grid LOWER-BOUND estimate of
``K = sup_{sigma > 0, omega} sigma ||((sigma + i omega) I - L)^{-1}||``.
Estimator label (issue #101 re-audit): this evaluates a finite, hand-chosen
``(sigma, omega)`` grid. It is NOT Mitchell's (SIMAX 41(4), 2020) globally
convergent algorithm and carries no globality certificate; the returned value
is a grid lower bound of the true Kreiss constant. Its ``sigma`` grid floor is
ABSOLUTE (``1e-3`` in rate units), so the legacy value is not invariant under a
unit rescale ``L -> cL``; it is preserved unchanged for backward compatibility.

D10b -- :func:`kreiss_grid_lower_bound`. Scale-relative successor (issue #101
slice A): same supremum, but the grid is built in DIMENSIONLESS coordinates
relative to ``rate_scale(L) = ||L||_F``, so the (dimensionless) estimate is
exactly invariant under ``L -> cL`` (``c > 0``). Returns grid/refinement
metadata (:class:`liouscope._types.KreissGridEstimate`) so coarse-grid results
are auditable as estimates, not certified constants. ``claim_status: pending``
until anchors confirm it.

D8b -- :func:`henrici_relative`. Dimensionless Henrici index
``eta_N(L) / ||L||_F`` in ``[0, 1]``; exactly invariant under a positive unit
rescale (issue #101 slice A / issue #97 item 2). Additive: legacy D8
``henrici_eta`` (rate-dimensioned) is preserved unchanged. ``claim_status:
pending``; the classifier does NOT consume it (advisory evidence only).

D11 -- :func:`bohr_arithmetic_progression`. Basso arXiv:2510.07267 (2025)
length of the longest arithmetic-progression of imaginary parts in the
spectrum. Returns ``(length, pauli_bound)`` where the Pauli bound is
``log_2(d)`` (depth Bound of the AP). Anchor M: this is D11, NOT the
resolvent peak (which is D11b in :mod:`liouscope.diagnostics.resolvent`).
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg as sla

from .._consts import EPS_DIV
from .._types import KreissGridEstimate, NonNormalityResult
from ..numerics.linalg import certified_eig, require_finite_square_2d
from ..numerics.resolvent import resolvent_norm
from ..numerics.scale import rate_scale, spectral_zero_tolerance

# Fail-closed ceiling for the dimensionless Henrici index: mathematically
# ``eta_N <= ||L||_F`` (Schur unitary invariance), so ``henrici_relative`` can
# exceed 1 only by rounding of order ``n * machine-eps``. Violations within this
# relative tolerance are clipped to 1.0; anything larger indicates a broken
# Schur decomposition and raises instead of silently clamping (issue #97
# 2026-08-05 re-audit, implementation slice item 2).
HENRICI_REL_CLIP_TOL = 1.0e-9


def henrici_eta_n(A: np.ndarray) -> float:
    """Henrici departure-from-normality measure via Schur form."""
    A = np.asarray(A, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
    T, _ = sla.schur(A, output="complex")
    # The diagonal carries eigenvalues; the strictly-upper triangle is N.
    n = T.shape[0]
    # Frobenius norm of the strictly upper triangle.
    fro_squared = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            fro_squared += abs(T[i, j]) ** 2
    return float(np.sqrt(fro_squared))


def henrici_relative(A: np.ndarray) -> float:
    """D8b: dimensionless Henrici index ``eta_N(A) / ||A||_F`` in ``[0, 1]``.

    Scale-relative counterpart of :func:`henrici_eta_n` (issue #101 slice A /
    issue #97 item 2): the legacy Henrici departure ``eta_N`` carries rate
    dimension and transforms as ``eta_N(cA) = |c| eta_N(A)``, so any absolute
    threshold on it is unit-dependent. Dividing by the shared operator rate
    scale ``rate_scale(A) = ||A||_F`` yields a dimensionless index that is
    exactly invariant under a positive unit rescale ``A -> cA`` and lies in
    ``[0, 1]``: ``0`` for normal matrices, approaching ``1`` when the whole
    Frobenius mass sits in the non-normal (strictly upper Schur) part.

    Zero-operator semantics (documented, issue #101 slice A item 1): the zero
    operator has no rate scale and is trivially normal, so the index is ``0.0``.

    Fail-closed: non-finite input raises via :func:`rate_scale`; a computed
    ratio exceeding ``1 + HENRICI_REL_CLIP_TOL`` (broken Schur decomposition)
    raises instead of being silently clamped. Values within the tolerance are
    clipped to the mathematical range ``[0, 1]``.

    ``claim_status: pending`` -- advisory evidence only; no classifier branch
    consumes this value (the F5 gate calibration is issue #101 slice C).
    """
    A = np.asarray(A)
    fro = rate_scale(A, name="A")
    if fro == 0.0:
        return 0.0
    rel = henrici_eta_n(A) / fro
    if rel > 1.0 + HENRICI_REL_CLIP_TOL:
        raise ValueError(
            "henrici_relative: eta_N / ||A||_F = "
            f"{rel:.17g} exceeds 1 beyond numerical tolerance "
            f"({HENRICI_REL_CLIP_TOL:g}); the Schur decomposition is unreliable "
            "for this input"
        )
    return float(min(max(rel, 0.0), 1.0))


def kreiss_grid_lower_bound(
    L_super: np.ndarray,
    *,
    n_sigma: int = 24,
    n_omega: int = 25,  # ungerade -> das Gitter enthaelt omega=0 (Peak-Lage)
    refine_levels: int = 1,
) -> KreissGridEstimate:
    """D10b: scale-relative grid LOWER BOUND of the Kreiss constant.

    Estimates ``K = sup_{sigma > 0, omega} sigma ||((sigma + i omega) I - L)^{-1}||``
    with ``sigma`` measured from the spectral abscissa (as in the legacy D10).

    Differences from the legacy :func:`kreiss_constant` (issue #101 slice A
    items 3/6 and the 2026-08-05 re-audit addendum):

    * the ``(sigma, omega)`` grid lives in DIMENSIONLESS coordinates relative to
      ``rate_scale(L) = ||L||_F`` and is converted to absolute rate units only
      per evaluation point, so the (dimensionless) estimate is exactly invariant
      under a positive unit rescale ``L -> cL`` (up to floating-point rounding);
    * the result is honestly labelled: a :class:`KreissGridEstimate` carrying
      the grid ranges/resolution, whether the maximiser sits on a grid EDGE
      (then the sup plausibly lies outside the sampled window), and the relative
      change contributed by local grid refinement (convergence metadata);
    * ``refine_levels`` local zoom passes around the coarse maximiser tighten
      the lower bound without a global re-sweep.

    The exact continuous-time Kreiss constant is invariant under ``L -> cL``;
    a certified global computation (Mitchell 2020) is NOT implemented -- the
    returned value is a lower bound, never a certificate.

    Zero-operator semantics: ``K(0) = sup_sigma sigma * 1/sigma = 1`` exactly;
    returned with NaN grid metadata (there is no grid).

    ``claim_status: pending`` -- advisory evidence only.
    """
    L_super = require_finite_square_2d(L_super, name="L_super")
    if n_sigma < 2 or n_omega < 2:
        raise ValueError(
            f"n_sigma and n_omega must each be >= 2, got {n_sigma}, {n_omega}"
        )
    if refine_levels < 0:
        raise ValueError(f"refine_levels must be >= 0, got {refine_levels}")
    scale = rate_scale(L_super)
    if scale == 0.0:
        return KreissGridEstimate(
            value=1.0,
            coarse_value=1.0,
            edge_maximizer=False,
            refinement_delta=0.0,
            sigma_rel_lo=float("nan"),
            sigma_rel_hi=float("nan"),
            omega_rel_max=float("nan"),
            n_sigma=n_sigma,
            n_omega=n_omega,
            rate_scale=0.0,
        )

    eigvals = sla.eigvals(L_super)
    re_max_rel = float(np.max(np.real(eigvals))) / scale
    re_min_rel = float(np.min(np.real(eigvals))) / scale
    im_abs_rel = float(np.max(np.abs(np.imag(eigvals)))) / scale
    sigma_rel_lo = 1.0e-3
    sigma_rel_hi = max(1.0, abs(re_min_rel) + 1.0)
    omega_rel_max = max(1.0, im_abs_rel + 1.0)

    def _k_value(sigma_rel: float, omega_rel: float) -> float:
        # sigma is measured from the spectral abscissa (legacy D10 semantics);
        # the evaluation point must lie right of the spectrum.
        shift_rel = sigma_rel + re_max_rel
        if shift_rel <= 0.0:
            return 0.0
        z_abs = complex(shift_rel * scale, omega_rel * scale)
        try:
            norm = resolvent_norm(L_super, z_abs)
        except np.linalg.LinAlgError:
            return 0.0
        return float(sigma_rel * scale * norm)

    def _sweep(
        sigmas: np.ndarray, omegas: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        values = np.empty((sigmas.size, omegas.size))
        for i, s in enumerate(sigmas):
            for j, w in enumerate(omegas):
                values[i, j] = _k_value(float(s), float(w))
        flat = int(np.argmax(values))
        bi, bj = divmod(flat, omegas.size)
        return values, float(values[bi, bj]), bi, bj

    sigmas = np.geomspace(sigma_rel_lo, sigma_rel_hi, n_sigma)
    omegas = np.linspace(-omega_rel_max, omega_rel_max, n_omega)
    values, coarse, bi, bj = _sweep(sigmas, omegas)
    # Edge flag with a tie tolerance: on a near-flat plateau (e.g. normal
    # operators, where sigma * ||R|| -> 1 from below as sigma grows) rounding
    # decides WHICH of many near-equal grid points is the argmax, so a strict
    # argmax-on-edge test would jitter under exact unit rescales. Flag instead
    # whether ANY grid point within a small relative band of the maximum sits
    # on the grid boundary -- that is the physically meaningful statement "the
    # sup plausibly continues beyond the sampled window".
    tie_band = values >= coarse * (1.0 - 1.0e-9)
    edge = bool(
        tie_band[0, :].any()
        or tie_band[-1, :].any()
        or tie_band[:, 0].any()
        or tie_band[:, -1].any()
    )

    best = coarse
    for _ in range(refine_levels):
        s_lo = float(sigmas[max(bi - 1, 0)])
        s_hi = float(sigmas[min(bi + 1, sigmas.size - 1)])
        w_lo = float(omegas[max(bj - 1, 0)])
        w_hi = float(omegas[min(bj + 1, omegas.size - 1)])
        if s_hi <= s_lo or w_hi <= w_lo:
            break
        sigmas = np.geomspace(s_lo, s_hi, n_sigma)
        omegas = np.linspace(w_lo, w_hi, n_omega)
        _, refined, bi, bj = _sweep(sigmas, omegas)
        if refined > best:
            best = refined

    delta = (best - coarse) / best if best > 0.0 else 0.0
    return KreissGridEstimate(
        value=best,
        coarse_value=coarse,
        edge_maximizer=edge,
        refinement_delta=float(delta),
        sigma_rel_lo=sigma_rel_lo,
        sigma_rel_hi=sigma_rel_hi,
        omega_rel_max=omega_rel_max,
        n_sigma=n_sigma,
        n_omega=n_omega,
        rate_scale=scale,
    )


def petermann_factors(
    L_super: np.ndarray,
    *,
    atol: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Petermann factors per non-zero eigenmode.

    Returns ``(eigenvalues, K_j)`` with the eigenvalues sorted by real part
    descending, the steady-state mode (``lambda = 0``) dropped, and
    complex-conjugate pairs retained (anchor I).

    Interpretation caveat (LIOU-NG-003): ``K_j`` quantifies the conditioning of
    eigenmode ``j``; it does NOT equal the realised transient amplification of
    the propagator. Use D10 (Kreiss) / D15 (numerical abscissa) / D13
    (pseudospectrum) to bound ``sup_t ||e^{tL}||``. See the module docstring.

    The eigendecomposition is CERTIFIED (round-14 review; issue #112). This
    function previously recomputed its own raw ``zgeev`` decomposition, so on
    a stiff trace-preserving generator it consumed exactly the solver failure
    the spectral and Mpemba layers repair: the lost zero mode cannot be
    restored by any cutoff (it is absent from the raw spectrum), and the
    measured result was 16 "non-zero" modes with ``petermann_max ~ 622.7``
    against the certified 15 modes with ``~ 2.0`` -- corrupting D9 and the
    D11 input while the spectral layer looked resolved, i.e. a possible
    false F1 signal that no verdict floor would catch. When the certificate
    is applicable but NOT resolved, both returned arrays are a single NaN --
    the library's unavailable sentinel -- so ``petermann_max`` becomes NaN
    and the F1 rung reads UNEVALUABLE instead of a fabricated value.
    """
    L_super = np.asarray(L_super, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
    decomp, certificate = certified_eig(L_super)
    if certificate.applicable and not certificate.resolved:
        warnings.warn(
            "Petermann factors: the eigendecomposition of this trace-"
            "preserving generator is unresolved (smallest |lambda| = "
            f"{certificate.residual:.3e}, bound {certificate.bound:.3e}); "
            "D9 is withheld as NaN rather than computed from a demonstrably "
            "wrong spectrum (issues #112/#113, round-14 review).",
            RuntimeWarning,
            stacklevel=2,
        )
        nan_modes = np.full(1, np.nan, dtype=complex)
        return nan_modes, np.full(1, np.nan)
    eigvals = decomp.eigenvalues
    vl, vr = decomp.left_vectors, decomp.right_vectors
    if vl is None:  # pragma: no cover - certified_eig always sets left vectors
        raise RuntimeError("certified_eig did not return left eigenvectors")
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
    # Drop zero eigenvalue but KEEP complex pairs. Scale-relative zero-mode
    # separation (issue #108): with an absolute floor, a rescaled generator
    # either dropped every mode (empty K array) or retained the round-off zero
    # mode, whose Petermann factor is numerical noise feeding the F1 gate.
    # The scale is the certificate's own operator-derived bound (round-13
    # review): on a strongly non-normal generator the radius-based proxy is
    # smaller than the eigensolve backward error ``eps * ||L||_2``, so the
    # displaced zero mode survived the filter as a spurious extra eigenmode
    # (measured: 4 modes reported where 3 exist). Radius fallback when the
    # certificate is inapplicable (round-14): without established trace
    # preservation no zero mode is guaranteed and the operator-norm bound can
    # exceed the whole spectrum of a strongly non-normal input.
    default_tol = certificate.bound if certificate.applicable else None
    mask = np.abs(eigvals) > spectral_zero_tolerance(
        eigvals,
        atol=default_tol if atol is None else atol,
        name="eigenvalues of L_super",
    )
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
    """Legacy D10: finite-grid LOWER-BOUND estimate of the Kreiss constant.

    Estimates ``K = sup_{sigma > 0, omega} sigma * ||((sigma + i omega) I - L)^{-1}||``
    by evaluating a finite ``(sigma, omega)`` grid (partially adapted to the
    spectrum). Estimator label (issue #101 re-audit): this is NOT Mitchell's
    (SIMAX 41(4), 2020) globally convergent algorithm and carries no globality
    certificate -- the returned value is a grid lower bound of the true Kreiss
    constant, with no edge/refinement metadata. The ``sigma`` grid floor
    (``1e-3``) and the ``sigma``/``omega`` caps (``>= 1``) are ABSOLUTE rate
    constants, so this legacy value is not invariant under a unit rescale
    ``L -> cL``. It is preserved byte-identically for backward compatibility;
    use :func:`kreiss_grid_lower_bound` (D10b) for the scale-relative,
    metadata-carrying successor.
    """
    L_super = np.asarray(L_super, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
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
    """Run D8, D8b, D9, D10, D10b, D11 together.

    Legacy fields (``henrici_eta``, ``kreiss`` ...) are computed exactly as
    before; the scale-relative D8b/D10b additions (issue #101 slice A) are
    purely additive and stamped ``claim_status: pending`` at the report level.
    """
    L_super = np.asarray(L_super)
    eta_n = henrici_eta_n(L_super)
    eta_rel = henrici_relative(L_super)
    eigvals_filt, K_factors = petermann_factors(L_super)
    K_max = float(np.max(K_factors)) if K_factors.size else 1.0
    K = kreiss_constant(L_super)
    K_scaled = kreiss_grid_lower_bound(L_super)
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
        henrici_relative=eta_rel,
        kreiss_scaled=K_scaled.value,
        kreiss_scaled_edge_maximizer=K_scaled.edge_maximizer,
        kreiss_scaled_refinement_delta=K_scaled.refinement_delta,
    )
