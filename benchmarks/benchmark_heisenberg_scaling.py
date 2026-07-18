"""Heisenberg-XXZ boundary-dephasing scaling benchmark.

Fits ``beta_D ~ N^{-alpha}`` for N = 2..5 (small to keep the dense pipeline
fast). The full paper benchmark uses N = 3..7 with the sparse path; this script
showcases the orchestration end-to-end.

Physics caveat -- what this benchmark actually measures
-------------------------------------------------------
The boundary bath here is **sigma_z (dephasing) driving**. Each jump operator
``L_k = sigma_z`` on a boundary site commutes exactly with the total
magnetisation ``Sz_tot = sum_i sigma_z_i / 2``::

    [L_k, Sz_tot] = 0   and   [H_XXZ, Sz_tot] = 0   (exactly)

so the full GKSL generator preserves the U(1) symmetry. Consequently:

* The Liouvillian kernel is **(N + 1)-fold degenerate** -- one stationary state
  per total-``Sz`` sector (magnetisation values -N/2 .. +N/2). This is an
  *expected* conservation law, not a solver error, and is asserted below via
  ``_kernel_dim(L) == N + 1``.
* The relaxation rate extracted here reflects the **integrable regime gap** of
  the symmetry-resolved Liouvillian (see Znidaric, arXiv:1507.07773), NOT a
  diffusive spin-transport coefficient. Dephasing conserves ``Sz_tot`` and
  therefore cannot drive a magnetisation current across the chain.
* To probe genuine (sub)diffusive **transport** one needs boundary jump
  operators that break U(1) -- e.g. ``sigma^+`` / ``sigma^-`` incoherent
  driving with unequal rates, which pump magnetisation and select a unique
  current-carrying NESS.

We deliberately report only the exact commutator fact and the regime statement;
we do NOT claim a specific scaling exponent from this small-N (N <= 5),
three-point fit, which is not statistically authoritative.

Degenerate-NESS handling
-------------------------
Because the kernel is degenerate, ``diagnose`` cannot pick a unique steady state
by itself (it fails closed with ``DegenerateSteadyStateError`` -- by design).
We resolve this *symmetry-consistently*: for a definite-magnetisation initial
state ``rho_0`` living in the ``Sz_tot = m`` sector, the physical NESS is the
maximally mixed state within that sector,

    rho_ss = P_m / tr(P_m),

where ``P_m`` projects onto the ``Sz_tot = m`` eigenspace. This ``rho_ss`` is an
*exact* stationary state (``L @ vec(rho_ss) == 0``, verified below) rather than
an arbitrary null-space representative, so the relaxation fit is well defined.
"""

from __future__ import annotations

import time

import numpy as np

import liouscope as ls
from liouscope.core import (
    boundary_dephasing_jumps,
    heisenberg_xxz_hamiltonian,
    one_d_chain,
)
from liouscope.core.hamiltonian import single_site_operator
from liouscope.numerics.kronecker import vec


def _total_sz(N: int) -> np.ndarray:
    """Diagonal total-magnetisation operator Sz_tot = sum_i sigma_z_i / 2."""
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    dim = 2 ** N
    op = np.zeros((dim, dim), dtype=complex)
    for site in range(N):
        op += single_site_operator(sz, site, N)
    return 0.5 * op


def _kernel_dim(L: np.ndarray, atol: float = 1.0e-9) -> int:
    """Dimension of the Liouvillian null space (SVD null count)."""
    s = np.linalg.svd(L, compute_uv=False)
    tol = max(atol, L.shape[0] * float(np.finfo(complex).eps) * s[0])
    return int(np.sum(s <= tol))


def _sector_states(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (rho_initial, rho_steady_state) for the domain-wall sector.

    rho_initial = |up..up down..down><...|  (a definite-Sz product state).
    rho_steady_state = P_m / tr(P_m), the maximally mixed state in that sector.
    """
    up = np.array([1.0, 0.0], dtype=complex)
    down = np.array([0.0, 1.0], dtype=complex)
    psi = np.array([1.0 + 0j])
    for site in range(N):
        psi = np.kron(psi, up if site < N // 2 else down)
    rho0 = np.outer(psi, psi.conj())

    sz_tot = _total_sz(N)
    m = float(np.real(psi.conj() @ (sz_tot @ psi)))
    diag = np.real(np.diag(sz_tot))
    mask = np.isclose(diag, m, atol=1.0e-9)
    proj = np.diag(mask.astype(complex))
    rho_ss = proj / np.trace(proj)
    return rho0, rho_ss


def main() -> None:
    Ns = list(range(2, 6))
    betas: list[float] = []
    print("Heisenberg-XXZ boundary-dephasing scaling benchmark")
    print("(measures the U(1)/integrable regime gap, NOT diffusive transport)")
    print("  N   dim   ker(L)   beta_D    wall(s)")
    for N in Ns:
        lat = one_d_chain(N)
        H = heisenberg_xxz_hamiltonian(lat, J=1.0, Delta=1.0)
        jumps = boundary_dephasing_jumps(N)
        rates = [0.25] * len(jumps)
        L = ls.build_liouvillian(H, jumps, rates)

        # Expected U(1) invariant: one steady state per total-Sz sector.
        kdim = _kernel_dim(L)
        assert kdim == N + 1, (
            f"expected (N+1)={N + 1} degenerate steady states from the "
            f"conserved Sz_tot (dephasing bath), got ker(L)={kdim}. A mismatch "
            "means the U(1) symmetry was broken or the tolerance is off."
        )

        rho0, rho_ss = _sector_states(N)
        # Independent physics oracle: the sector-projected state is an EXACT
        # stationary state of L (not an arbitrary null-space representative).
        residual = float(np.linalg.norm(L @ vec(rho_ss)))
        assert residual < 1.0e-9, (
            f"sector NESS not stationary: ||L vec(rho_ss)|| = {residual:.2e}"
        )

        t0 = time.perf_counter()
        report = ls.diagnose(
            L,
            rho_initial=rho0,
            rho_steady_state=rho_ss,
            bootstrap_B=20,
            include_mpemba=False,
            seed=42,
        )
        dt = time.perf_counter() - t0
        beta = float(report.relaxation.beta_D)
        betas.append(beta)
        print(f"  {N}   {L.shape[0]:>3}   {kdim:>4}    {beta: .4f}   {dt:.2f}")

    if all(b > 0 for b in betas):
        log_betas = np.log(np.asarray(betas))
        log_Ns = np.log(np.asarray(Ns, dtype=float))
        slope, _intercept = np.polyfit(log_Ns, log_betas, 1)
        print(f"\nApparent log-log slope alpha = {-slope:.3f}")
        print("NOTE: descriptive only -- a 4-point N<=5 fit of the integrable "
              "regime gap is NOT an authoritative transport exponent.")


if __name__ == "__main__":
    main()
