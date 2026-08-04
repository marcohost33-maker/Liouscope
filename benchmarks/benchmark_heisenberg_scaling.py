"""Heisenberg-XXZ boundary-dephasing relaxation benchmark.

Runs a small dense demonstration for ``N = 2..5``.  The apparent log-log slope
is descriptive only: four small-system points do not establish a thermodynamic
scaling exponent.

Physics scope
-------------
The boundary collapse operators are ``sigma_z`` dephasers.  They and the XXZ
Hamiltonian commute with total magnetisation
``Sz_tot = sum_i sigma_z_i / 2``, so the operator dynamics decomposes into
invariant charge blocks.  This symmetry alone does *not* prove that the full
Liouvillian kernel has exactly ``N + 1`` dimensions or that every charge block
has a unique attractor.  For this specific fixed fixture and the tested sizes we
therefore verify both statements numerically:

* the full nullity is ``N + 1``;
* the operator block selected by the domain-wall initial state has nullity one
  and no additional peripheral (purely imaginary) modes;
* ``P_m / tr(P_m)`` has a scale-relative stationarity residual below tolerance.

The script is a symmetry-resolved relaxation demo, not a spin-conductivity or
boundary-driven transport benchmark.  Conserving total magnetisation does not
forbid Hamiltonian spin transport; rather, pure boundary dephasing supplies no
magnetisation bias or pumping that would define a current-carrying NESS.
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

_NULL_RTOL = 1.0e-9
_SECTOR_ATOL = 1.0e-12


def _total_sz(N: int) -> np.ndarray:
    """Return ``Sz_tot = sum_i sigma_z_i / 2`` in the computational basis."""
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    dim = 2**N
    op = np.zeros((dim, dim), dtype=complex)
    for site in range(N):
        op += single_site_operator(sz, site, N)
    return 0.5 * op


def _nullity(
    matrix: np.ndarray,
    *,
    rtol: float = _NULL_RTOL,
    atol: float = 0.0,
) -> int:
    """Return numerical nullity with a rate-scale-relative SVD threshold.

    The default has no absolute floor, so ``nullity(c * A) == nullity(A)`` for
    positive finite ``c`` up to floating-point limits.  This mirrors the
    production ``steady_state`` tolerance contract introduced in PR #99.
    """
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {arr.shape}")
    if rtol < 0.0 or atol < 0.0 or not np.isfinite(rtol) or not np.isfinite(atol):
        raise ValueError("rtol and atol must be finite and non-negative")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix must contain only finite values")

    singular_values = np.linalg.svd(arr, compute_uv=False)
    if singular_values.size == 0:
        return 0
    eps = float(np.finfo(arr.real.dtype).eps)
    scale = float(singular_values[0])
    threshold = max(atol, max(rtol, max(arr.shape) * eps) * scale)
    return int(np.count_nonzero(singular_values <= threshold))


def _domain_wall_state(N: int) -> np.ndarray:
    """Return ``|up..up down..down><...|`` as a density matrix."""
    up = np.array([1.0, 0.0], dtype=complex)
    down = np.array([0.0, 1.0], dtype=complex)
    psi = np.array([1.0 + 0.0j])
    for site in range(N):
        psi = np.kron(psi, up if site < N // 2 else down)
    return np.outer(psi, psi.conj())


def _sector_mask(N: int, rho_initial: np.ndarray) -> np.ndarray:
    """Return the Hilbert-basis mask for the initial state's ``Sz_tot`` sector."""
    sz_tot = _total_sz(N)
    charge = float(np.real(np.trace(rho_initial @ sz_tot)))
    charges = np.real(np.diag(sz_tot))
    mask = np.isclose(charges, charge, rtol=0.0, atol=_SECTOR_ATOL)
    if not np.any(mask):
        raise RuntimeError(f"no basis states found in magnetisation sector m={charge}")
    return mask


def _sector_operator_indices(mask: np.ndarray) -> np.ndarray:
    """Indices of ``P_m rho P_m`` in Liouville ``vec(order='F')`` convention."""
    basis = np.flatnonzero(mask)
    dim = int(mask.size)
    return np.asarray(
        [row + dim * col for col in basis for row in basis],
        dtype=int,
    )


def _sector_liouvillian(L: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restrict ``L`` to operators supported inside one charge sector."""
    indices = _sector_operator_indices(mask)
    return np.asarray(L)[np.ix_(indices, indices)]


def _sector_states(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the domain-wall state and maximally mixed state in its sector."""
    rho0 = _domain_wall_state(N)
    mask = _sector_mask(N, rho0)
    projector = np.diag(mask.astype(complex))
    rho_ss = projector / np.trace(projector)
    return rho0, rho_ss


def _relative_stationarity_residual(L: np.ndarray, rho: np.ndarray) -> float:
    """Return ``||L vec(rho)|| / (||L||_2 ||vec(rho)||_2)``."""
    rho_vec = vec(rho)
    denominator = float(np.linalg.norm(L, ord=2) * np.linalg.norm(rho_vec))
    numerator = float(np.linalg.norm(L @ rho_vec))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _assert_sector_relaxes(L_sector: np.ndarray) -> None:
    """Fail if the selected block has extra stationary or peripheral modes."""
    if _nullity(L_sector) != 1:
        raise AssertionError("selected charge block does not have a unique NESS")

    eigenvalues = np.linalg.eigvals(L_sector)
    scale = float(np.linalg.norm(L_sector, ord=2))
    eps = float(np.finfo(L_sector.real.dtype).eps)
    zero_tol = max(_NULL_RTOL, max(L_sector.shape) * eps) * scale
    nonzero = eigenvalues[np.abs(eigenvalues) > zero_tol]
    if nonzero.size and float(np.max(np.real(nonzero))) >= -zero_tol:
        raise AssertionError(
            "selected charge block has a non-decaying peripheral mode; "
            "stationarity alone does not establish relaxation"
        )


def main() -> None:
    sizes = list(range(2, 6))
    betas: list[float] = []
    print("Heisenberg-XXZ boundary-dephasing relaxation benchmark")
    print("(symmetry-resolved demo; not a transport-exponent measurement)")
    print("  N   dim   ker(L)   ker(block)   beta_D    wall(s)")

    for N in sizes:
        lattice = one_d_chain(N)
        H = heisenberg_xxz_hamiltonian(lattice, J=1.0, Delta=1.0)
        jumps = boundary_dephasing_jumps(N)
        rates = [0.25] * len(jumps)
        L = ls.build_liouvillian(H, jumps, rates)

        # Fixture-specific regression, not a theorem inferred from U(1) alone.
        full_nullity = _nullity(L)
        assert full_nullity == N + 1, (
            f"for this tested fixture expected ker(L)={N + 1}, got {full_nullity}"
        )

        rho0, rho_ss = _sector_states(N)
        mask = _sector_mask(N, rho0)
        L_sector = _sector_liouvillian(L, mask)
        _assert_sector_relaxes(L_sector)
        sector_nullity = _nullity(L_sector)

        residual = _relative_stationarity_residual(L, rho_ss)
        assert residual < 1.0e-10, (
            f"sector NESS relative residual is too large: {residual:.2e}"
        )

        started = time.perf_counter()
        report = ls.diagnose(
            L,
            rho_initial=rho0,
            rho_steady_state=rho_ss,
            bootstrap_B=20,
            include_mpemba=False,
            seed=42,
        )
        elapsed = time.perf_counter() - started
        beta = float(report.relaxation.beta_D)
        betas.append(beta)
        print(
            f"  {N}   {L.shape[0]:>3}   {full_nullity:>4}"
            f"       {sector_nullity:>4}     {beta: .4f}   {elapsed:.2f}"
        )

    if all(beta > 0.0 for beta in betas):
        log_betas = np.log(np.asarray(betas))
        log_sizes = np.log(np.asarray(sizes, dtype=float))
        slope, _intercept = np.polyfit(log_sizes, log_betas, 1)
        print(f"\nApparent four-point log-log slope alpha = {-slope:.3f}")
        print(
            "NOTE: descriptive only -- N=2..5 is not sufficient to infer a "
            "thermodynamic scaling or transport exponent."
        )


if __name__ == "__main__":
    main()
