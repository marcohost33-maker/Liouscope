"""PR #127, review round 21 (external, 2026-09-11/12): four findings, each pinned.

Every test here failed on the head it was written against (``e556c96``) and
each docstring records the measured pre-fix value, so a green run says the
finding is closed rather than that the test was too weak to see it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import liouscope.diagnostics.relaxation as relaxation_mod
from liouscope import build_liouvillian, diagnose
from liouscope.core.lindblad import steady_state
from liouscope.diagnostics.relaxation import (
    UnderResolvedTransientWarning,
    compute_relaxation_layer,
    decay_rates,
    default_relaxation_grid,
    fastest_decay_rate,
    samples_per_fast_efolding,
)
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.fitting.car1 import estimate_car1_theta

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _two_scale(slow: float, fast: float) -> tuple[np.ndarray, np.ndarray]:
    L = build_liouvillian(
        np.zeros((4, 4), dtype=complex),
        [np.sqrt(slow) * np.kron(_SM, _I2), np.sqrt(fast) * np.kron(_I2, _SM)],
    )
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = np.kron(plus, plus)
    return L, np.outer(psi, psi.conj())


def _three_scale(slow: float, mid: float, fast: float):
    def kron3(a, b, c):
        return np.kron(np.kron(a, b), c)

    L = build_liouvillian(
        np.zeros((8, 8), dtype=complex),
        [
            np.sqrt(slow) * kron3(_SM, _I2, _I2),
            np.sqrt(mid) * kron3(_I2, _SM, _I2),
            np.sqrt(fast) * kron3(_I2, _I2, _SM),
        ],
    )
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    psi = kron3(plus, plus, plus)
    return L, np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Finding 1 (thread on relaxation.py): availability before family
# ---------------------------------------------------------------------------


def test_a_run_with_no_successful_fit_claims_no_whitening() -> None:
    """Measured before the fix: ``"car1_fallback_ar1"`` with 5/5 ``success=False``.

    ``rho_initial == rho_steady_state`` trips the flat-curve guard in
    ``fit_gls_ar1`` BEFORE any residual model is selected, so no AR(1) fallback
    ever happened -- yet the label asserted one.
    """
    L, _ = _two_scale(1.0e-6, 1.0)
    rho_ss = steady_state(L)
    grid = default_relaxation_grid(
        compute_spectral_layer(L, rho_ss).gap, fast_rate=fastest_decay_rate(L)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=rho_ss, rho_steady_state=rho_ss,
            t_grid=grid, bootstrap_B=5, seed=1,
        )
    assert rep.fits and not any(f.success for f in rep.fits.values()), (
        "fixture must leave every fit unsuccessful"
    )
    assert rep.residual_model == "car1_unavailable"


def test_a_uniform_grid_with_no_successful_fit_is_not_labelled_ar1() -> None:
    """Same guard on the uniform path: ``"ar1"`` was a statement about the grid."""
    L, _ = _two_scale(1.0e-2, 1.0)
    rho_ss = steady_state(L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=rho_ss, rho_steady_state=rho_ss,
            t_grid=np.linspace(0.0, 10.0, 40), bootstrap_B=5, seed=1,
        )
    assert rep.fits and not any(f.success for f in rep.fits.values())
    assert rep.residual_model == "ar1_unavailable"


def test_the_fallback_label_survives_for_successful_fits_that_fell_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Over-correction control: SUCCESSFUL fits whose theta failed keep their label."""
    import liouscope.fitting.gls as gls_mod

    monkeypatch.setattr(gls_mod, "estimate_car1_theta", lambda t, r: float("nan"))
    L, rho0 = _two_scale(1.0e-6, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    assert any(f.success for f in rep.fits.values())
    assert rep.residual_model == "car1_fallback_ar1"


# ---------------------------------------------------------------------------
# Finding 2 (thread on car1.py): theta must not depend on residual amplitude
# ---------------------------------------------------------------------------


def _ou_path(n: int = 40, theta: float = 0.05, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, 50.0, n))
    t[0] = 0.0
    r = np.zeros(n)
    r[0] = rng.normal()
    for k in range(1, n):
        a = np.exp(-theta * (t[k] - t[k - 1]))
        r[k] = a * r[k - 1] + np.sqrt(1.0 - a * a) * rng.normal()
    return t, r


@pytest.mark.parametrize("amplitude", [1.0e-170, 1.0e-150, 1.0e-6, 1.0e6, 1.0e150])
def test_car1_theta_is_invariant_under_residual_amplitude(amplitude: float) -> None:
    """Measured before the fix on this series: 0.0885 at 1, NaN at 1e-170.

    The tolerance is the estimator's OWN precision, not a concession: the
    profiled likelihood is flat near its minimum, so a 1-ulp perturbation of
    the residuals (which a multiply-then-divide by ``amplitude`` is) moves the
    bounded argmin by ~1e-8 relative on this 40-point series -- measured 7e-8
    at 1e-150 after the fix, against a 4e-7 drift before it and NaN at 1e-170.
    The finding was the NaN and the fallback it triggered; 1e-6 is two decades
    above the round-off floor and three below the pre-fix drift.
    """
    t, r = _ou_path()
    reference = estimate_car1_theta(t, r)
    assert np.isfinite(reference) and reference > 0.0
    scaled = estimate_car1_theta(t, amplitude * r)
    assert np.isfinite(scaled), amplitude
    assert scaled == pytest.approx(reference, rel=1.0e-6), (amplitude, scaled, reference)


def test_a_genuinely_constant_series_is_still_degenerate() -> None:
    """The degeneracy test moved from a sum of squares to a maximum; it must still fire."""
    t, _ = _ou_path()
    assert np.isnan(estimate_car1_theta(t, np.full(t.size, 0.3)))
    assert np.isnan(estimate_car1_theta(t, np.zeros(t.size)))


# ---------------------------------------------------------------------------
# Finding 3 (thread on relaxation.py:818): reuse the certified spectrum
# ---------------------------------------------------------------------------


def _count_eigvals_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    calls: list[tuple[int, ...]] = []
    original = np.linalg.eigvals

    def spy(a, *args, **kwargs):
        calls.append(tuple(np.asarray(a).shape))
        return original(a, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "eigvals", spy)
    return calls


def test_the_relaxation_layer_does_not_resolve_a_forwarded_spectrum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured before the fix: two ``eigvals`` calls on the 16x16 generator
    inside the relaxation layer even with the gap forwarded."""
    L, rho0 = _two_scale(1.0e-6, 1.0)
    rho_ss = steady_state(L)
    spectral = compute_spectral_layer(L, rho_ss)
    calls = _count_eigvals_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=rho0, rho_steady_state=rho_ss,
            gap=spectral.gap, eigenvalues=spectral.eigenvalues,
            bootstrap_B=5, seed=1,
        )
    assert [c for c in calls if c == (16, 16)] == [], calls
    assert rep.t_grid_source == "gap_scaled_multiscale"


def test_diagnose_forwards_the_certified_spectrum(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pipeline must hand the relaxation layer the eigenvalues D1 came from."""
    seen: list[np.ndarray | None] = []
    original = relaxation_mod.decay_rates

    def spy(L_super, *, eigenvalues=None):
        seen.append(eigenvalues)
        return original(L_super, eigenvalues=eigenvalues)

    monkeypatch.setattr(relaxation_mod, "decay_rates", spy)
    L, rho0 = _two_scale(1.0e-6, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        diagnose(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    assert seen, "decay_rates was never consulted"
    assert all(e is not None for e in seen), "diagnose() let the layer re-solve"


def test_a_direct_caller_without_a_gap_also_gets_one_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting ``gap`` calls the spectral layer once; its spectrum is reused."""
    L, rho0 = _two_scale(1.0e-6, 1.0)
    calls = _count_eigvals_calls(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    # ``certified_eigvals`` may take more than one route on its own; the claim
    # is only that the relaxation layer adds NONE of its own on top.
    calls_after_spectral = _count_eigvals_calls(monkeypatch)
    rho_ss = steady_state(L)
    compute_spectral_layer(L, rho_ss)
    n_spectral = len([c for c in calls_after_spectral if c == (16, 16)])
    assert len([c for c in calls if c == (16, 16)]) == n_spectral, (calls, n_spectral)


def test_forwarded_eigenvalues_are_what_the_rates_are_read_from() -> None:
    """A synthetic spectrum must win over the operator it is passed with."""
    L, _ = _two_scale(1.0e-6, 1.0)
    fake = np.array([0.0, -3.0, -7.0, -0.5])
    assert decay_rates(L, eigenvalues=fake).tolist() == [0.5, 3.0, 7.0]
    assert fastest_decay_rate(L, eigenvalues=fake) == 7.0
    grid = np.linspace(0.0, 1.0, 11)
    assert samples_per_fast_efolding(L, grid, eigenvalues=fake) == pytest.approx(
        1.0 / (7.0 * 0.1)
    )


# ---------------------------------------------------------------------------
# Finding 4 (thread on relaxation.py:989): name the worst-resolved mode
# ---------------------------------------------------------------------------


def test_the_report_names_the_intermediate_mode_the_grid_missed() -> None:
    """Three scales: the minimum belongs to the 1e-3 mode, not the fastest.

    Measured before the fix: ``samples_per_fast_efolding = 0.00195`` with no
    field saying which mode; ``_resolution_detail`` knew (rate 1.001e-3,
    blind interval 5.1e5 starting at 9.99) and the warning quoted it.
    """
    L, rho0 = _three_scale(1.0e-6, 1.0e-3, 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = compute_relaxation_layer(L, rho_initial=rho0, bootstrap_B=5, seed=1)
    hits = [w for w in caught if issubclass(w.category, UnderResolvedTransientWarning)]
    assert hits
    assert rep.worst_resolved_rate == pytest.approx(1.0e-3, rel=1.0e-2)
    assert rep.worst_resolved_rate < fastest_decay_rate(L) / 100.0
    assert np.isfinite(rep.worst_resolved_blind_interval)
    assert rep.worst_resolved_blind_interval > 0.0
    assert rep.worst_resolved_blind_start > 0.0
    # The persisted triple must be the one the warning was built from.
    msg = str(hits[0].message)
    assert f"{rep.worst_resolved_rate:.4g}" in msg
    assert f"{rep.worst_resolved_blind_interval:.4g}" in msg
    assert f"t={rep.worst_resolved_blind_start:.4g}" in msg


def test_a_late_start_is_recorded_as_a_lead_in() -> None:
    """``blind_start == 0.0`` with ``t_grid[0] > 0`` identifies an unsampled lead-in."""
    L, rho0 = _two_scale(1.0e-2, 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=rho0, t_grid=np.linspace(100.0, 101.0, 21),
            bootstrap_B=5, seed=1,
        )
    assert rep.worst_resolved_blind_start == 0.0
    assert rep.worst_resolved_blind_interval == pytest.approx(100.0)
    # Every mode sees the same lead-in, so the least-resolved one is the
    # fastest: on two coupled damping channels that is the SUM mode, 1 + 1e-2.
    assert rep.worst_resolved_rate == pytest.approx(fastest_decay_rate(L))
    assert rep.worst_resolved_rate == pytest.approx(1.01)


def test_the_worst_mode_fields_default_to_nan_when_nothing_decays() -> None:
    """A generator with no decaying mode measures nothing: NaN, not a fabricated mode."""
    L = np.zeros((4, 4), dtype=complex)
    rho = np.eye(2, dtype=complex) / 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = compute_relaxation_layer(
            L, rho_initial=rho, rho_steady_state=rho,
            t_grid=np.linspace(0.0, 1.0, 5), bootstrap_B=5, seed=1,
        )
    assert rep.samples_per_fast_efolding == float("inf")
    assert np.isnan(rep.worst_resolved_rate)
    assert np.isnan(rep.worst_resolved_blind_interval)
    assert np.isnan(rep.worst_resolved_blind_start)
