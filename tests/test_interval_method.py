"""Issue #116: the reported interval estimator must be the computed one.

The public surface advertised BCa unconditionally, but the leave-one-out
jackknife that supplies the BCa acceleration term only runs for time grids of
<= 60 points, while the default grid has 80 — so the default pipeline computes
a bias-corrected (BC) interval. These tests pin the disclosure field
``RelaxationResult.interval_method`` in every reachable state, pin the gate
location itself, and include a discrimination test proving the acceleration
term is genuinely live when the jackknife is supplied (without which a
hard-coded label would satisfy every other assertion here).
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope._types import RelaxationResult
from liouscope.diagnostics import relaxation as relaxation_mod
from liouscope.diagnostics.relaxation import compute_relaxation_layer
from liouscope.fitting.bootstrap import bca_ci
from liouscope.io.export import _to_jsonable


@pytest.fixture()
def damped_qubit() -> tuple[np.ndarray, np.ndarray]:
    """Amplitude-damped qubit and |+><+| initial state (fixed repro system)."""
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [sigma_minus], [1.0])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    return L, rho0


def test_default_grid_reports_bc_not_bca(damped_qubit):
    # The issue's headline: the default 80-point grid skips the jackknife, so
    # the interval is BC. The label must say so instead of the field name
    # implying BCa.
    L, rho0 = damped_qubit
    res = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=10, seed=1)
    assert res.relative_entropy_curve.size == 80  # default grid unchanged
    assert res.interval_method == "BC"
    assert np.isfinite(res.bca_ci_beta[0]) and np.isfinite(res.bca_ci_beta[1])


def test_diagnose_default_pipeline_reports_bc(damped_qubit):
    # End-to-end: diagnose() without t_grid lands in the BC branch too.
    L, rho0 = damped_qubit
    report = diagnose(L, rho_initial=rho0, bootstrap_B=10, seed=1)
    assert report.relaxation.interval_method == "BC"


def test_short_grid_reports_bca(damped_qubit):
    L, rho0 = damped_qubit
    res = compute_relaxation_layer(
        L, rho_initial=rho0, t_grid=np.linspace(0.0, 10.0, 60),
        bootstrap_B=10, seed=1,
    )
    assert res.interval_method == "BCa"
    assert np.isfinite(res.bca_ci_beta[0]) and np.isfinite(res.bca_ci_beta[1])


def test_gate_boundary_is_60_points(damped_qubit):
    # Pins the gate location: 61 points is the first grid that loses the
    # acceleration term. If the gate moves, this test and the documented
    # "<= 60" wording must move together.
    L, rho0 = damped_qubit
    res = compute_relaxation_layer(
        L, rho_initial=rho0, t_grid=np.linspace(0.0, 10.0, 61),
        bootstrap_B=10, seed=1,
    )
    assert res.interval_method == "BC"


def test_bootstrap_failure_reports_none(damped_qubit, monkeypatch):
    # The no-interval outcome must not masquerade as an estimator.
    def _boom(*args, **kwargs):
        raise ValueError("forced bootstrap failure (test)")

    monkeypatch.setattr(relaxation_mod, "parametric_bootstrap", _boom)
    L, rho0 = damped_qubit
    with pytest.warns(RuntimeWarning, match="parametric bootstrap"):
        res = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=10, seed=1)
    assert res.interval_method == "none"
    assert np.isnan(res.bca_ci_beta[0]) and np.isnan(res.bca_ci_beta[1])


def test_acceleration_term_is_live_when_jackknife_supplied():
    # Discrimination: "BCa" must name a genuinely different estimator, not a
    # relabelled BC. On a skewed bootstrap distribution with a skewed
    # jackknife, a != 0 and the endpoints move; a no-op implementation that
    # ignored jackknife_estimates would fail here.
    rng = np.random.default_rng(7)
    samples = (rng.gamma(shape=2.0, scale=0.5, size=400) ** 2).reshape(-1, 1)
    theta_hat = np.array([float(np.median(samples))])
    jk = (np.linspace(0.5, 2.0, 30) ** 3).reshape(-1, 1)
    bc = bca_ci(samples, theta_hat, jackknife_estimates=None)
    bca = bca_ci(samples, theta_hat, jackknife_estimates=jk)
    assert not np.allclose(bc, bca)


def test_degenerate_jackknife_reduces_to_bc():
    # All-equal jackknife estimates => a = 0 by the den == 0 guard, and the
    # two calls must agree exactly. Pins that "BCa" differs from "BC" only
    # through the acceleration term.
    rng = np.random.default_rng(7)
    samples = rng.normal(1.0, 0.2, size=300).reshape(-1, 1)
    theta_hat = np.array([1.0])
    jk = np.full((20, 1), 1.0)
    bc = bca_ci(samples, theta_hat, jackknife_estimates=None)
    bca = bca_ci(samples, theta_hat, jackknife_estimates=jk)
    np.testing.assert_array_equal(bc, bca)


def test_field_is_additive_with_honest_default():
    # Pre-#116 constructor calls / deserialised reports carry no estimator
    # record; the default must say so rather than guess one.
    res = RelaxationResult(
        von_neumann_entropy=0.0,
        relative_entropy_curve=np.zeros(3),
        fidelity_curve=np.ones(3),
        entanglement_asymmetry=None,
        fits={},
        aicc_model="M0",
        beta_D=1.0,
        bca_ci_beta=(0.9, 1.1),
    )
    assert res.interval_method == "unreported"


def test_interval_method_serialises(damped_qubit):
    L, rho0 = damped_qubit
    res = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=10, seed=1)
    obj = _to_jsonable(res)
    assert obj["interval_method"] == "BC"
