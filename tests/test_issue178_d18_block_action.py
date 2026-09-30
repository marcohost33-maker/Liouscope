"""Issue #178: D18 block exponential-action regression tests."""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from liouscope.diagnostics.lep import compute_lep_layer, initial_state_sensitivity
from liouscope.numerics.kronecker import unvec, vec


def _rng_state_equal(left, right) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_d18_large_system_matches_dense_reference_and_preserves_global_rng():
    d = 8
    n = d * d
    L = -0.2 * np.eye(n, dtype=complex)
    rho_ss = np.eye(d, dtype=complex) / d
    n_samples = 6
    seed = 23
    t_eval = 1.0

    gen = np.random.default_rng(seed)
    dense = sla.expm(L * t_eval)
    expected = np.empty(n_samples)
    for k in range(n_samples):
        psi = gen.normal(size=d) + 1j * gen.normal(size=d)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())
        rho_t = unvec(dense @ vec(rho0), d=d)
        expected[k] = np.linalg.norm(rho_t - rho_ss, ord="fro")

    np.random.seed(991)
    before = np.random.get_state()
    actual = initial_state_sensitivity(
        L, rho_ss, n_samples=n_samples, seed=seed, t_eval=t_eval
    )
    after = np.random.get_state()

    assert np.isclose(actual, np.std(expected), rtol=1e-12, atol=1e-14)
    assert _rng_state_equal(before, after)


def test_compute_lep_layer_records_d18_action_backend_at_n64():
    d = 8
    n = d * d
    L = -0.1 * np.eye(n, dtype=complex)
    rho_ss = np.eye(d, dtype=complex) / d
    eigenvalues = np.linspace(0.0, -1.0, n, dtype=complex)

    result = compute_lep_layer(
        L,
        eigenvalues,
        beta_D_linear=0.1,
        gap=0.1,
        rho_steady_state=rho_ss,
        n_haar=4,
        seed=5,
    )

    assert result.initial_state_backend == "expm_action"
    assert np.isfinite(result.initial_state_sensitivity)
