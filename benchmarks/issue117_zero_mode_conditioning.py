#!/usr/bin/env python3
"""Issue #117 step 3: measure zero-mode conditioning across five families.

Runs the campaign whose numbers are quoted in
``src/liouscope/numerics/conditioning.py`` and in the PR body, so a reader can
re-derive them rather than trust them. Nothing here is a gate; the script only
prints measurements.

    python benchmarks/issue117_zero_mode_conditioning.py

Families
--------
A  the non-normal 4x4 fixture from the 2026-09-11 external review of PR #127:
   does a conditioning-scaled forward estimate explain a displacement the
   certificate band under-predicts?
B  a diagonal similarity that leaves the spectrum invariant: whose operator is
   being conditioned, the caller's or a balanced surrogate's?
C  near-degenerate clusters: per-mode ``s`` against the subspace ``sigma_min``.
D  the canonical stiff #112 network: the NEGATIVE control -- its premise (raw
   zgeev really loses the zero mode) is established first, then conditioning is
   shown to miss it, so no later change claims otherwise.
E  physical GKSL generators across four decades of rate and drive: is there a
   false ``CONDITIONING_LIMITED`` to pay for?
F  the timing campaign behind the overhead claim: the audit runs by default, so
   what it costs has to be measurable here rather than asserted in a docstring.
"""

from __future__ import annotations

import math
import time
from functools import partial

import numpy as np
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.numerics.conditioning import (
    cluster_conditioning,
    eigenvalue_conditioning,
    zero_mode_conditioning,
)
from liouscope.numerics.linalg import (
    certified_eigvals,
    eig_nonhermitian,
    trace_preservation_defect,
)
from liouscope.numerics.traceless import trace_vector

SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

# The canonical #112 repro (tests/test_spectral_certificate.py): raw zgeev
# loses the zero mode here and the certificate repairs via dgeev-real.
STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
STIFF_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]


def _classical_network(pairs, rates, d=4):
    jumps = []
    for to, frm in pairs:
        jump = np.zeros((d, d), dtype=complex)
        jump[to, frm] = 1.0
        jumps.append(jump)
    return build_liouvillian(np.zeros((d, d), dtype=complex), jumps, rates)


def _rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def _pr127_fixture() -> np.ndarray:
    block = np.zeros((4, 4), dtype=complex)
    block[0, 1] = 1.0e-14
    block[1, 0] = 1.0
    block[2, 2] = -1.0
    block[3, 3] = -2.0
    seed = np.eye(4, dtype=complex)
    seed[:, 0] = trace_vector(2)
    basis, _ = np.linalg.qr(seed)
    phase = np.vdot(basis[:, 0], trace_vector(2))
    basis = basis * (phase / abs(phase))
    return basis @ block @ basis.conj().T


def family_a() -> None:
    _rule("A  PR #127 non-normal fixture: band vs conditioning-scaled estimate")
    L = _pr127_fixture()
    raw_defect, _scale = trace_preservation_defect(L)
    _values, certificate = certified_eigvals(L)
    evidence = zero_mode_conditioning(
        L, zero_tolerance=certificate.bound, eigenvalues=_values
    )
    observed = evidence.observed_displacement
    # Round-2 review: print the quantity the estimate actually divides. The raw
    # ``trace_preservation_defect`` is ``||vec(I)^H L||``; the perturbation norm
    # is that over sqrt(d), so printing the raw value under a ``||q^H L||``
    # label made the displayed division fail to reproduce its own result.
    print(f"  raw defect ||vec(I)^H L||             {raw_defect:.4e}")
    print(f"  perturbation norm ||q^H L||           {evidence.trace_defect:.4e}")
    print(f"  certificate band rtol*eps*||L||_2     {certificate.bound:.4e}")
    print(f"  OBSERVED displacement min|lambda|     {observed:.4e}")
    print(f"  reciprocal condition s(lambda_0)      {evidence.reciprocal_condition:.4e}")
    print(f"  ||q^H L|| / s (cond. estimate)        "
          f"{evidence.structural_forward_estimate:.4e}")
    print(f"  band under-predicts by                "
          f"{observed / certificate.bound:.3e}x")
    print(f"  conditioning estimate is off by       "
          f"{observed / evidence.structural_forward_estimate:.3f}x")
    print(f"  verdict                               {evidence.verdict}")


def family_b() -> None:
    _rule("B  diagonal similarity: spectrum invariant, conditioning is not")
    rng = np.random.default_rng(7)
    A = rng.standard_normal((9, 9)) + 1j * rng.standard_normal((9, 9))
    scaling = np.diag(
        np.float64([1e-6, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e6, 1e8, 1e9])
    )
    scaled = scaling @ A @ np.linalg.inv(scaling)
    plain = zero_mode_conditioning(A)
    similar = zero_mode_conditioning(scaled)
    drift = float(
        np.max(
            np.abs(
                np.sort_complex(np.linalg.eigvals(A))
                - np.sort_complex(np.linalg.eigvals(scaled))
            )
        )
    )
    # Round-3 review: this is a round-off quantity and differs between BLAS
    # builds, so the documentation states a bound and the run prints the value.
    print(f"  spectrum drift (round-off, < 1e-14)   {drift:.3e}")
    print(f"  min s over spectrum, A                "
          f"{plain.spectrum_min_reciprocal_condition:.3e}")
    print(f"  min s over spectrum, D A D^-1         "
          f"{similar.spectrum_min_reciprocal_condition:.3e}")
    print("  -> s follows the operator AS WRITTEN. LAPACK xGEEVX would have")
    print("     reported the BALANCED problem's numbers instead; ?geev's")
    print("     back-transformed vectors give the caller's.")


def family_c() -> None:
    _rule("C  near-degenerate cluster: per-mode s collapses, the subspace does not")
    print(f"  {'delta':>10} {'per-mode s':>14} {'sigma_min(Y^H X)':>18}")
    for delta in (1e-2, 1e-6, 1e-10, 1e-14):
        A = np.array(
            [[0.0, 1.0, 0.0], [0.0, delta, 0.0], [0.0, 0.0, -1.0]], dtype=complex
        )
        values, left, right = sla.eig(A, left=True, right=True)
        idx = np.argsort(np.abs(values))[:2]
        per = float(eigenvalue_conditioning(right, left)[idx].min())
        print(f"  {delta:10.0e} {per:14.3e} {cluster_conditioning(right, left, idx):18.3f}")
    print("  -> a degenerate stationary manifold is PHYSICAL; a per-mode gate")
    print("     would withhold on it. The reported figure is the subspace one.")


def family_d() -> None:
    _rule("D  NEGATIVE CONTROL: the canonical stiff #112 network")
    L = _classical_network(STIFF_PAIRS, STIFF_RATES)
    raw = eig_nonhermitian(np.asarray(L, dtype=complex)).eigenvalues
    accepted, certificate = certified_eigvals(L)
    print("  PREMISE -- the raw solve must actually fail, or there is nothing to miss:")
    print(f"    raw zgeev min|lambda|      {float(np.abs(raw).min()):.4e}")
    print(f"    certificate band           {certificate.bound:.4e}")
    print(f"    zero mode lost by zgeev?   {float(np.abs(raw).min()) > certificate.bound}")
    print(f"    accepted route             {certificate.solver}, "
          f"min|lambda| {float(np.abs(accepted).min()):.4e}")
    evidence = zero_mode_conditioning(L)
    print("  CONTROL -- what conditioning says about that WRONG spectrum:")
    print(f"    conditioned eigenvalue     {evidence.observed_displacement:.4e}")
    print(f"    s(lambda_0)                {evidence.reciprocal_condition:.4f}")
    print(f"    worst s over the spectrum  {evidence.spectrum_min_reciprocal_condition:.4f}"
          f"  (condition number "
          f"{1.0 / evidence.spectrum_min_reciprocal_condition:.1f})")
    print("    -> condition numbers 1-25: the wrong spectrum is WELL conditioned,")
    print("       the same range recorded in linalg.py on 2026-08-25. The")
    print("       Ahues-Tisseur deflation destroys the slow spectrum before any")
    print("       conditioning estimate can see it.")
    guarded = zero_mode_conditioning(
        L,
        zero_tolerance=certificate.zero_set_tolerance(accepted),
        eigenvalues=accepted,
    )
    print(f"  GUARD -- with the accepted spectrum handed in: available="
          f"{guarded.available}, reason={guarded.reason!r}")
    print("       the audit refuses rather than describe a spectrum the report")
    print("       does not contain (PR #166 review).")


def family_e() -> None:
    _rule("E  physical GKSL generators: is there a false alarm to pay for?")
    print(f"  {'gamma':>9} {'omega':>9} {'s(lambda_0)':>13} {'||L||/max|lam|':>16} {'verdict':>20}")
    worst = 1.0
    for gamma in (1e0, 1e3, 1e6, 1e9):
        for omega in (0.0, 1.0, 1e3, 1e6):
            L = build_liouvillian(
                0.5 * omega * SIGMA_X,
                [math.sqrt(gamma) * SIGMA_MINUS, math.sqrt(1e-9) * SIGMA_Z],
            )
            values, certificate = certified_eigvals(L)
            evidence = zero_mode_conditioning(
                L,
                zero_tolerance=certificate.zero_set_tolerance(values),
                eigenvalues=values,
            )
            ratio = float(np.linalg.norm(L, 2) / max(np.abs(values).max(), 1e-300))
            worst = min(worst, evidence.reciprocal_condition)
            print(f"  {gamma:9.0e} {omega:9.0e} {evidence.reciprocal_condition:13.3f} "
                  f"{ratio:16.3f} {evidence.verdict:>20}")
    print(f"  -> worst reciprocal condition over the family: {worst:.3f}")


def _best_of(call, reps: int) -> float:
    """Minimum wall time over ``reps`` runs.

    The MINIMUM, not the mean, as ``timeit`` documents: process noise on a
    shared runner only ever adds time, so the smallest observation is the one
    closest to the cost of the work itself. Measured with the mean at three
    repetitions this campaign produced a -70% "overhead" at d=8, which is how
    the first, unmeasured percentages in the docstring came to look plausible.
    """
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def family_f(reps: int = 9) -> None:
    _rule("F  cost of the audit, measured on THIS machine")
    print("  Round-3 review: fixed percentages (+3%/+9%/+15%) used to sit in the")
    print("  compute_spectral_layer docstring with nothing in the repository")
    print("  producing them -- and they were wrong. They had been measured for")
    print("  the extra eigensolve ALONE, which turns out not to be what the")
    print("  audit spends its time on at small d. The campaign is committed here")
    print("  so the figure a reader sees is their own machine's.")
    print()
    print(f"  {'d':>4} {'superop':>8} {'layer':>10} {'audit':>10} {'of which eig':>13} {'rel':>7}")
    rng = np.random.default_rng(3)
    for d in (2, 4, 8, 12):
        H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        H = 0.5 * (H + H.conj().T)
        jump = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        L = build_liouvillian(H, [0.1 * jump])
        values, certificate = certified_eigvals(L)
        tol = certificate.zero_set_tolerance(values)
        compute_spectral_layer(L, conditioning_audit=False)  # untimed warm-up
        layer = _best_of(
            partial(compute_spectral_layer, L, conditioning_audit=False), reps
        )
        audit = _best_of(
            partial(zero_mode_conditioning, L, zero_tolerance=tol, eigenvalues=values),
            reps,
        )
        eig_only = _best_of(partial(sla.eig, L, left=True, right=True), reps)
        print(f"  {d:4d} {d * d:8d} {layer * 1e3:8.2f} ms {audit * 1e3:8.2f} ms "
              f"{eig_only * 1e3:11.2f} ms {100.0 * audit / layer:6.0f}%")
    print(f"  (best of {reps} runs each -- the MINIMUM, as timeit documents, since")
    print("   runner noise only ever adds time.)")
    print()
    print("  Two readings that hold across runs on this machine:")
    print("   * the extra eigensolve is NOT what the audit costs at small d")
    print("     (0.02 ms of 0.65 ms at d=2). The per-mode work -- unit-")
    print("     normalising each eigenvector, the scale-safe norms, the SVD and")
    print("     the residual loop -- is Python-level and dominates there;")
    print("   * in ABSOLUTE terms that is well under a millisecond at d=2.")
    print()
    print("  The relative column is NOT stable on a shared runner: the d=8 audit")
    print("  was measured at 20 ms and 40 ms in consecutive runs of this very")
    print("  script. Read it as an order of magnitude, re-run it for your own")
    print("  hardware, and do not quote a single figure from one run -- which is")
    print("  the mistake this section exists to stop repeating.")
    print("  conditioning_audit=False switches the audit off with no other effect.")


def main() -> None:
    family_a()
    family_b()
    family_c()
    family_d()
    family_e()
    family_f()
    print()
    print("All figures above are AUDIT evidence. No filter, gap, verdict or tier")
    print("reads any of them (issue #117 step 2; step 4 stays open until a")
    print("certified-yet-misconditioned generator is exhibited).")


if __name__ == "__main__":
    main()
