"""Heisenberg-XXZ boundary-dephasing relaxation benchmark.

Runs a small dense demonstration for ``N = 2..5``. The apparent log-log slope
is descriptive only: four small-system points do not establish a thermodynamic
scaling exponent.

Physics scope
-------------
The boundary collapse operators are ``sigma_z`` dephasers. They and the XXZ
Hamiltonian commute with total magnetisation
``Sz_tot = sum_i sigma_z_i / 2``, so the operator dynamics decomposes into
invariant charge blocks. This symmetry alone does *not* prove that the full
Liouvillian kernel has exactly ``N + 1`` dimensions or that every charge block
has a unique attractor. For this specific fixed fixture and the tested sizes we
therefore verify the required numerical conditions:

* the full nullity is ``N + 1``;
* the domain-wall initial state is supported in exactly one charge sector;
* that operator block is invariant, has a simple semisimple zero mode, and has
  no additional peripheral modes;
* ``P_m / tr(P_m)`` has a scale-relative stationarity residual below tolerance.

The script is a symmetry-resolved relaxation demo, not a spin-conductivity or
boundary-driven transport benchmark. Conserving total magnetisation does not
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


def _svd_nullity_data(
    matrix: np.ndarray,
    *,
    rtol: float = _NULL_RTOL,
    atol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return validated matrix, singular values and scale-relative threshold."""
    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got shape {arr.shape}")
    if (
        rtol < 0.0
        or atol < 0.0
        or not np.isfinite(rtol)
        or not np.isfinite(atol)
    ):
        raise ValueError("rtol and atol must be finite and non-negative")
    if not np.all(np.isfinite(arr)):
        raise ValueError("matrix must contain only finite values")

    singular_values = np.linalg.svd(arr, compute_uv=False)
    if singular_values.size == 0:
        return arr, singular_values, float(atol)
    eps = float(np.finfo(arr.real.dtype).eps)
    scale = float(singular_values[0])
    threshold = max(atol, max(rtol, max(arr.shape) * eps) * scale)
    return arr, singular_values, threshold


def _nullity(
    matrix: np.ndarray,
    *,
    rtol: float = _NULL_RTOL,
    atol: float = 0.0,
) -> int:
    """Return numerical nullity with a rate-scale-relative SVD threshold."""
    _arr, singular_values, threshold = _svd_nullity_data(
        matrix,
        rtol=rtol,
        atol=atol,
    )
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
    """Return the unique total-magnetisation sector supporting ``rho_initial``.

    An expectation value alone is insufficient: a mixture of distinct sectors
    can have an expectation equal to a third sector. This routine therefore
    checks the full support condition ``rho = P_m rho P_m`` and fails closed if
    the state spans more than one charge sector.
    """
    rho = np.asarray(rho_initial, dtype=complex)
    dim = 2**N
    if rho.shape != (dim, dim):
        raise ValueError(f"rho_initial must have shape {(dim, dim)}, got {rho.shape}")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho_initial must contain only finite values")
    if not np.allclose(rho, rho.conj().T, rtol=0.0, atol=_SECTOR_ATOL):
        raise ValueError("rho_initial must be Hermitian")
    if not np.isclose(np.trace(rho), 1.0, rtol=0.0, atol=_SECTOR_ATOL):
        raise ValueError("rho_initial must have unit trace")

    charges = np.real(np.diag(_total_sz(N)))
    matching_masks: list[np.ndarray] = []
    for charge in np.unique(charges):
        mask = np.isclose(charges, charge, rtol=0.0, atol=_SECTOR_ATOL)
        projector = np.diag(mask.astype(complex))
        projected = projector @ rho @ projector
        residual = float(np.linalg.norm(rho - projected, ord="fro"))
        if residual <= _SECTOR_ATOL * max(1.0, float(np.linalg.norm(rho, ord="fro"))):
            matching_masks.append(mask)

    if len(matching_masks) != 1:
        raise ValueError(
            "rho_initial must be supported in exactly one total-magnetisation "
            f"sector; found {len(matching_masks)} matching sectors"
        )
    return matching_masks[0]


def _sector_operator_indices(mask: np.ndarray) -> np.ndarray:
    """Indices of ``P_m rho P_m`` in Liouville ``vec(order='F')`` convention."""
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 1 or not np.any(mask_arr):
        raise ValueError("mask must be a non-empty one-dimensional boolean array")
    basis = np.flatnonzero(mask_arr)
    dim = int(mask_arr.size)
    return np.asarray(
        [row + dim * col for col in basis for row in basis],
        dtype=int,
    )


def _sector_liouvillian(L: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restrict ``L`` to operators supported inside one charge sector."""
    arr = np.asarray(L, dtype=complex)
    dim = int(np.asarray(mask).size)
    expected = dim * dim
    if arr.shape != (expected, expected):
        raise ValueError(
            f"L must have shape {(expected, expected)} for mask size {dim}, "
            f"got {arr.shape}"
        )
    indices = _sector_operator_indices(mask)
    return arr[np.ix_(indices, indices)]


def _relative_block_leakage(L: np.ndarray, mask: np.ndarray) -> float:
    """Return relative leakage from ``P_m rho P_m`` to its complement."""
    arr = np.asarray(L, dtype=complex)
    inside = _sector_operator_indices(mask)
    outside = np.setdiff1d(np.arange(arr.shape[0]), inside)
    leakage = arr[np.ix_(outside, inside)]
    denominator = max(float(np.linalg.norm(arr, ord="fro")), np.finfo(float).tiny)
    return float(np.linalg.norm(leakage, ord="fro")) / denominator


def _sector_states(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the domain-wall state and maximally mixed state in its sector."""
    rho0 = _domain_wall_state(N)
    mask = _sector_mask(N, rho0)
    projector = np.diag(mask.astype(complex))
    rho_ss = projector / np.trace(projector)
    return rho0, rho_ss


def _relative_stationarity_residual(L: np.ndarray, rho: np.ndarray) -> float:
    """Return ``||L vec(rho)||_2 / (||L||_F ||vec(rho)||_2)``."""
    rho_vec = vec(rho)
    denominator = float(np.linalg.norm(L, ord="fro") * np.linalg.norm(rho_vec))
    numerator = float(np.linalg.norm(L @ rho_vec))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _assert_sector_relaxes(L_sector: np.ndarray) -> None:
    """Fail unless the selected block has one semisimple zero and decays."""
    arr, _singular_values, _threshold = _svd_nullity_data(L_sector)
    nullity = _nullity(arr)
    if nullity != 1:
        raise AssertionError(
            f"selected charge block must have one stationary mode, got {nullity}"
        )

    # A one-dimensional kernel does not exclude a Jordan chain at zero. For a
    # semisimple zero eigenvalue ker(L) == ker(L^2); otherwise polynomial terms
    # can survive and stationarity alone does not establish relaxation.
    squared_nullity = _nullity(arr @ arr)
    if squared_nullity != nullity:
        raise AssertionError(
            "selected charge block has a defective zero mode: "
            f"nullity(L)={nullity}, nullity(L^2)={squared_nullity}"
        )

    eigenvalues = np.linalg.eigvals(arr)
    scale = float(np.linalg.norm(arr, ord="fro"))
    eps = float(np.finfo(arr.real.dtype).eps)
    zero_tol = max(_NULL_RTOL, max(arr.shape) * eps) * scale
    zero_count = int(np.count_nonzero(np.abs(eigenvalues) <= zero_tol))
    if zero_count != 1:
        raise AssertionError(
            f"selected charge block must have one algebraic zero mode, got {zero_count}"
        )

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
        expected_nullity = N + 1
        if full_nullity != expected_nullity:
            raise RuntimeError(
                f"for this fixture expected ker(L)={expected_nullity}, "
                f"got {full_nullity}"
            )

        rho0, rho_ss = _sector_states(N)
        mask = _sector_mask(N, rho0)
        leakage = _relative_block_leakage(L, mask)
        if leakage >= 1.0e-12:
            raise RuntimeError(
                f"selected charge block is not invariant: relative leakage={leakage:.2e}"
            )

        L_sector = _sector_liouvillian(L, mask)
        _assert_sector_relaxes(L_sector)
        sector_nullity = _nullity(L_sector)

        residual = _relative_stationarity_residual(L, rho_ss)
        if residual >= 1.0e-10:
            raise RuntimeError(
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
        if not np.isfinite(beta):
            raise RuntimeError(f"diagnose returned non-finite beta_D for N={N}: {beta}")
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
