"""LiouScope quickstart with QuTiP cross-check.

Run::

    python examples/quickstart.py

Requires the ``qutip`` optional dependency; falls back to a no-cross-check
mode if QuTiP is unavailable.
"""

from __future__ import annotations

import numpy as np

import liouscope as ls


def main() -> None:
    print(f"LiouScope v{ls.__version__}")
    print(f"  Taxonomy:   {ls.TAXONOMY_VERSION}")
    print(f"  Schema:     {ls.DIAGNOSTIC_SCHEMA_VERSION}")
    print()

    # Dephased qubit
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    H = 0.5 * sx
    jumps = [sz]
    rates = [0.3]

    L = ls.build_liouvillian(H, jumps, rates)
    print(f"Liouvillian shape: {L.shape}, dtype {L.dtype}")

    try:
        import qutip

        H_qt = qutip.Qobj(H)
        c_ops = [np.sqrt(rates[0]) * qutip.Qobj(jumps[0])]
        L_qt = qutip.liouvillian(H_qt, c_ops).full()
        ok = bool(np.allclose(L, L_qt, atol=1.0e-10))
        diff = float(np.max(np.abs(L - L_qt)))
        print(f"QuTiP cross-check (column-stacking): {'PASS' if ok else 'FAIL'}  "
              f"(max-abs-diff {diff:.2e})")
    except ImportError:
        print("QuTiP not installed; skipping cross-check.")

    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = ls.diagnose(L, rho_initial=rho0, bootstrap_B=100, seed=42)
    print()
    print("Diagnostic report:")
    print(f"  D1 Gap Delta            = {report.spectral.gap:.6f}")
    print(f"  D2 GNS Gap              = {report.spectral.gns_gap:.6f}")
    print(f"  D2b KMS Gap             = {report.spectral.kms_gap:.6f}")
    print(f"  D9 Petermann max        = {report.nonnorm.petermann_max:.3e}")
    print(f"  D10 Kreiss              = {report.nonnorm.kreiss:.3e}")
    print(f"  M-hierarchy winner      = {report.relaxation.aicc_model}")
    print(f"  beta_D                  = {report.relaxation.beta_D:.6f}")
    print(
        f"  BCa CI                  = "
        f"[{report.relaxation.bca_ci_beta[0]:.6f}, {report.relaxation.bca_ci_beta[1]:.6f}]"
    )
    if report.mpemba is not None:
        print(f"  D19 Mpemba c_1          = {report.mpemba.overlap_c1:.3e}")
        print(f"      Mpemba candidate?   = {report.mpemba.is_mpemba_candidate}")
    print(f"  Mechanism class         = {report.classification.a_class}")
    print(f"  Gap-failure family      = {report.classification.f_family}")
    print(f"  Verdict                 = {report.classification.verdict}")
    print(f"  Tier                    = {report.classification.tier}")
    print(f"  Confidence              = {report.classification.confidence:.2f}")
    print(f"  Run ID (16 hex)         = {report.governance.run_id[:16]}")


if __name__ == "__main__":
    main()
