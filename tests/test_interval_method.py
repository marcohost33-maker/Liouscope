"""Issue #116: the reported interval estimator must be the computed one.

The public surface advertised "BCa" unconditionally, but the leave-one-out
jackknife that supplies the BCa acceleration term only runs on time grids of
at most ``JACKKNIFE_MAX_GRID_POINTS`` (60) points, while the default grid has
80. With ``a = 0`` the interval is bias-corrected (BC), not BCa. These tests
pin ``RelaxationResult.interval_method`` in every reachable state, pin the
gate, check the BC arithmetic against an independent formula, and prove that
the acceleration term is live when a jackknife is supplied. Without that last
check, a hard-coded label would satisfy every other assertion here.

Builds on the disclosure design of PR #119, extended for what landed since:
- the degenerate-interval semantics of #125, where a NaN interval is
  "unavailable" and never credited to an estimator;
- the 2026-09-19 triage, where the jackknife path is labelled "BCa-delete1"
  because its delete-one acceleration is Efron's iid construction applied to
  serially correlated residuals.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from liouscope import build_liouvillian, diagnose
from liouscope._consts import (
    INTERVAL_METHOD_BC,
    INTERVAL_METHOD_BCA_DELETE1,
    INTERVAL_METHOD_UNAVAILABLE,
    INTERVAL_METHOD_UNREPORTED,
    INTERVAL_METHODS,
    JACKKNIFE_MAX_GRID_POINTS,
)
from liouscope._types import RelaxationResult
from liouscope.diagnostics import relaxation as relaxation_mod
from liouscope.diagnostics.relaxation import compute_relaxation_layer
from liouscope.diagnostics.uncertainty import compute_uncertainty_layer
from liouscope.fitting.bootstrap import bca_ci
from liouscope.io.export import _to_jsonable


@pytest.fixture()
def damped_qubit() -> tuple[np.ndarray, np.ndarray]:
    """Amplitude-damped qubit and |+><+| initial state (fixed repro system)."""
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [sigma_minus], [1.0])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    return L, np.outer(plus, plus.conj())


def _layer(L: np.ndarray, rho0: np.ndarray, **kwargs: object) -> RelaxationResult:
    return compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=20, seed=1, **kwargs)


# ---------------------------------------------------------------------------
# Every reachable state of the field
# ---------------------------------------------------------------------------


def test_default_grid_reports_bc_not_bca(damped_qubit) -> None:
    """The issue's headline: the default 80-point grid skips the jackknife."""
    res = _layer(*damped_qubit)
    assert res.relative_entropy_curve.size == 80  # premise: default grid unchanged
    assert res.relative_entropy_curve.size > JACKKNIFE_MAX_GRID_POINTS
    assert res.interval_method == INTERVAL_METHOD_BC
    assert np.all(np.isfinite(res.bca_ci_beta))


def test_diagnose_default_pipeline_reports_bc(damped_qubit) -> None:
    L, rho0 = damped_qubit
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=1)
    assert report.relaxation.interval_method == INTERVAL_METHOD_BC


def test_grid_at_the_gate_reports_bca_delete1(damped_qubit) -> None:
    res = _layer(*damped_qubit, t_grid=np.linspace(0.0, 10.0, JACKKNIFE_MAX_GRID_POINTS))
    assert res.interval_method == INTERVAL_METHOD_BCA_DELETE1
    assert np.all(np.isfinite(res.bca_ci_beta))


def test_one_point_past_the_gate_reports_bc(damped_qubit) -> None:
    """Pins the boundary: the first grid that loses the acceleration term."""
    res = _layer(*damped_qubit, t_grid=np.linspace(0.0, 10.0, JACKKNIFE_MAX_GRID_POINTS + 1))
    assert res.interval_method == INTERVAL_METHOD_BC


def test_gate_value_is_unchanged_by_the_disclosure() -> None:
    """Issue #116 option 1 changes labels, not numbers: the gate stays at 60."""
    assert JACKKNIFE_MAX_GRID_POINTS == 60


def test_bootstrap_failure_reports_unavailable(damped_qubit, monkeypatch) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("forced bootstrap failure (test)")

    monkeypatch.setattr(relaxation_mod, "parametric_bootstrap", _boom)
    with pytest.warns(RuntimeWarning, match="parametric bootstrap"):
        res = _layer(*damped_qubit)
    assert res.interval_method == INTERVAL_METHOD_UNAVAILABLE
    assert np.all(np.isnan(res.bca_ci_beta))


def test_degenerate_bootstrap_distribution_reports_unavailable(damped_qubit, monkeypatch) -> None:
    """#125 semantics: a NaN interval must not be credited to "BC".

    PR #119 predates #125; it labelled every successful bootstrap "BC" or
    "BCa", including one whose interval ``bca_ci`` withholds as degenerate.
    """
    real_bootstrap = relaxation_mod.parametric_bootstrap

    def _collapsed(*args: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        samples, theta_hat = real_bootstrap(*args, **kwargs)
        return np.tile(theta_hat, (samples.shape[0], 1)), theta_hat

    monkeypatch.setattr(relaxation_mod, "parametric_bootstrap", _collapsed)
    with pytest.warns(RuntimeWarning, match="degenerate"):
        res = _layer(*damped_qubit)
    assert np.all(np.isnan(res.bca_ci_beta))
    assert res.interval_method == INTERVAL_METHOD_UNAVAILABLE


def test_field_is_additive_with_an_honest_default() -> None:
    """Pre-#116 constructor calls carry no estimator record; none is guessed."""
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
    assert res.interval_method == INTERVAL_METHOD_UNREPORTED


def test_vocabulary_is_closed_and_serialises(damped_qubit) -> None:
    res = _layer(*damped_qubit)
    assert res.interval_method in INTERVAL_METHODS
    assert len(set(INTERVAL_METHODS)) == len(INTERVAL_METHODS)
    assert _to_jsonable(res)["interval_method"] == INTERVAL_METHOD_BC


def test_u0_is_finite_exactly_when_an_interval_exists(damped_qubit, monkeypatch) -> None:
    """U0 inherits the label: finite iff ``interval_method`` names an estimator."""
    res = _layer(*damped_qubit)
    assert np.isfinite(compute_uncertainty_layer(res).fit_uncertainty)

    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("forced bootstrap failure (test)")

    monkeypatch.setattr(relaxation_mod, "parametric_bootstrap", _boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        failed = _layer(*damped_qubit)
    assert failed.interval_method == INTERVAL_METHOD_UNAVAILABLE
    assert np.isnan(compute_uncertainty_layer(failed).fit_uncertainty)


# ---------------------------------------------------------------------------
# The arithmetic behind the labels
# ---------------------------------------------------------------------------


def test_bc_interval_matches_the_closed_form_percentiles() -> None:
    """Independent oracle: with a = 0 the endpoints are Phi(2 z0 + z_alpha).

    Written from Efron's formula rather than from ``bca_ci``. If the "BC" label
    named anything other than the bias-corrected percentile interval, this
    would fail.
    """
    rng = np.random.default_rng(116)
    samples = rng.gamma(shape=3.0, scale=0.2, size=(2000, 1))
    theta_hat = np.array([0.55])
    alpha = 0.05

    boot = np.sort(samples[:, 0])
    prop = np.mean(boot < theta_hat[0]) + 0.5 * np.mean(boot == theta_hat[0])
    z0 = ndtri(prop)
    expected = [
        np.quantile(boot, ndtr(2.0 * z0 + ndtri(q)), method="linear")
        for q in (alpha / 2.0, 1.0 - alpha / 2.0)
    ]
    np.testing.assert_allclose(bca_ci(samples, theta_hat, alpha=alpha)[0], expected, rtol=1e-12)
    assert abs(z0) > 0.05  # premise: a real bias correction, not the percentile case


def test_acceleration_term_is_live_when_jackknife_is_supplied() -> None:
    """Discrimination: "BCa-delete1" names a different estimator, not a relabel."""
    rng = np.random.default_rng(7)
    samples = (rng.gamma(shape=2.0, scale=0.5, size=400) ** 2).reshape(-1, 1)
    theta_hat = np.array([float(np.median(samples))])
    jk = (np.linspace(0.5, 2.0, 30) ** 3).reshape(-1, 1)
    bc = bca_ci(samples, theta_hat, jackknife_estimates=None)
    bca = bca_ci(samples, theta_hat, jackknife_estimates=jk)
    assert not np.allclose(bc, bca)


def test_constant_jackknife_reduces_to_bc_exactly() -> None:
    """Zero jackknife spread gives a = 0, so the two estimators coincide."""
    rng = np.random.default_rng(7)
    samples = rng.normal(1.0, 0.2, size=300).reshape(-1, 1)
    theta_hat = np.array([1.0])
    bc = bca_ci(samples, theta_hat, jackknife_estimates=None)
    bca = bca_ci(samples, theta_hat, jackknife_estimates=np.full((20, 1), 1.0))
    np.testing.assert_array_equal(bc, bca)
