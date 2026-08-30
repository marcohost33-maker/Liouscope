"""Round-23 evidence-hardening regressions for PR #121.

D16 is a property of the same eigenvalue spectrum as D1/D3/D4. When the
zero-mode certificate is unresolved, the spectral layer marks D1 unavailable
with NaN. The LEP layer must not publish a finite proximity from those same
candidate eigenvalues.
"""

from __future__ import annotations

import numpy as np

from liouscope.diagnostics.lep import compute_lep_layer


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The zero Liouvillian keeps D18 cheap/deterministic. Supplying rho_ss avoids
    # asking the steady-state solver to choose from the degenerate manifold.
    L = np.zeros((4, 4), dtype=complex)
    rho_ss = np.eye(2, dtype=complex) / 2.0
    eigenvalues = np.array([0.0, -1.0, -1.000001, -3.0], dtype=complex)
    return L, rho_ss, eigenvalues


def test_d16_is_withheld_when_the_spectral_gap_is_unavailable() -> None:
    L, rho_ss, eigenvalues = _fixture()
    result = compute_lep_layer(
        L,
        eigenvalues,
        beta_D_linear=1.0,
        gap=float("nan"),
        rho_steady_state=rho_ss,
        seed=7,
        n_haar=4,
    )
    assert np.isnan(result.lep_proximity)
    assert result.lep_candidate_count == 0
    # D17 already treats an unavailable/non-positive gap as non-comparable.
    assert np.isinf(result.gap_rate_consistency)
    # D18 does not use the candidate spectrum and remains independently measured.
    assert np.isfinite(result.initial_state_sensitivity)


def test_a_finite_gap_still_computes_d16() -> None:
    """Positive control: the guard must not disable the ordinary LEP path."""
    L, rho_ss, eigenvalues = _fixture()
    result = compute_lep_layer(
        L,
        eigenvalues,
        beta_D_linear=1.0,
        gap=1.0,
        rho_steady_state=rho_ss,
        seed=7,
        n_haar=4,
    )
    assert np.isfinite(result.lep_proximity)
    assert result.lep_proximity > 0.0
    assert result.lep_candidate_count >= 1
    assert result.gap_rate_consistency == 0.0


def test_measured_gapless_input_is_not_confused_with_unavailable() -> None:
    """A finite measured 0.0 gap is data, while NaN means unavailable."""
    L, rho_ss, eigenvalues = _fixture()
    result = compute_lep_layer(
        L,
        eigenvalues,
        beta_D_linear=1.0,
        gap=0.0,
        rho_steady_state=rho_ss,
        seed=7,
        n_haar=4,
    )
    assert np.isfinite(result.lep_proximity)
    assert np.isinf(result.gap_rate_consistency)
