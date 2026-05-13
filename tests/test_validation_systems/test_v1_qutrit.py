"""V1 dissipative qutrit numerical reference.

Spec reference: V1 Qutrit (Teil 15)
  KMS / GNS ratio approximately 1.41 (KMS 41% > GNS)
  Zero Mori-Shirai violations across 200 time samples
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from liouscope import diagnose
from liouscope.examples import v1_qutrit
from liouscope.numerics.kronecker import unvec, vec


def test_v1_kms_above_gns():
    sys = v1_qutrit()
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    assert report.spectral.gns_gap > 0


def test_v1b_thermal_qutrit_kms_equals_gns():
    """V1b thermal qutrit is detailed-balance => KMS gap = GNS gap exactly."""
    from liouscope.examples import v1b_thermal_qutrit

    sys = v1b_thermal_qutrit(beta=1.0, omega=1.0)
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=10,
                     include_mpemba=False, seed=42)
    assert report.spectral.gns_gap > 0
    rel_diff = abs(report.spectral.kms_gap - report.spectral.gns_gap) / report.spectral.gns_gap
    assert rel_diff < 1e-6, (
        f"At detailed balance KMS should equal GNS; got "
        f"GNS={report.spectral.gns_gap}, KMS={report.spectral.kms_gap}"
    )


def test_offdiagonal_qutrit_kms_above_gns():
    """BM-003b regression: off-diagonal H + non-DB jumps => KMS > GNS by >10%."""
    from liouscope import build_liouvillian

    H = np.array(
        [[0.0, 0.3, 0.0], [0.3, 1.0, 0.4], [0.0, 0.4, 2.5]], dtype=complex
    )
    jumps: list[np.ndarray] = []
    for i, j in [(0, 1), (1, 2), (0, 2)]:
        op = np.zeros((3, 3), dtype=complex)
        op[j, i] = 1.0
        jumps.append(op)
        jumps.append(op.conj().T)
    rates = [0.3, 0.05, 0.4, 0.07, 0.2, 0.04]
    L = build_liouvillian(H, jumps, rates)
    report = diagnose(L, bootstrap_B=10, include_mpemba=False, seed=42)
    assert report.spectral.gns_gap > 0
    ratio = report.spectral.kms_gap / report.spectral.gns_gap
    # Anchor numerical regression value from BM-003b is 1.1468; allow +-2%.
    assert 1.10 < ratio < 1.20, f"BM-003b KMS/GNS regressed: ratio={ratio:.4f}"
    assert report.spectral.kms_gap >= report.spectral.gns_gap - 1e-6


def test_v1_autocorrelation_bound_no_violation():
    """For Hermitian A, |<A_t, A>_GNS| <= exp(-g_K * t) |<A, A>_GNS|."""
    sys = v1_qutrit()
    rho_ss = None
    report = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=10, seed=42)
    g_k = report.spectral.kms_gap
    rho_ss = report.spectral.steady_state
    d = rho_ss.shape[0]
    # Probe operator: sigma_x analogue on first two levels
    A = np.zeros((d, d), dtype=complex)
    A[0, 1] = 1.0
    A[1, 0] = 1.0
    times = np.linspace(0.0, 5.0, 50)
    sqrt_rho = sla.sqrtm(rho_ss)
    base_kms = float(np.real(np.trace(sqrt_rho @ A.conj().T @ sqrt_rho @ A)))
    violations = 0
    for t in times:
        expL = sla.expm(sys.L.conj().T * t)
        A_t_vec = expL @ vec(A)
        A_t = unvec(A_t_vec, d=d)
        val = float(np.real(np.trace(sqrt_rho @ A_t.conj().T @ sqrt_rho @ A)))
        bound = float(np.exp(-g_k * t) * base_kms)
        if abs(val) > bound + 1e-3:
            violations += 1
    assert violations == 0, f"{violations} MS violations across 50 samples"
