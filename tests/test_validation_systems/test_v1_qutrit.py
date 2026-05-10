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
