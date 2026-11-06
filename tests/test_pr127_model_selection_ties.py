"""Model selection at the rounding level (PR #127, cross-family LIO-02/LIO-03).

On a decay curve that is an exact single exponential to float64 precision
every M0..M3b candidate fits the same rounding noise, and the AICc winner
became a property of the platform's rounding: perturbing the phantom-ladder
operator of ``test_d17_gap_coherence`` by ONE ulp per entry flipped the
trace-distance winner from M0 to M3a in 7 of 25 seeds (numerics platform
probe, arm64). ``relaxation._select_model`` widens every AICc by how far the
rounding level of the data can move its log-likelihood; candidates that cannot
be separated from the minimum are tied, a tie containing the nested M0
selects M0, and any other tie reports ``INDETERMINATE``.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest

from liouscope import diagnose
from liouscope._consts import INTERVAL_METHOD_UNAVAILABLE, MODEL_SELECTION_INDETERMINATE
from liouscope.diagnostics import relaxation as rx
from liouscope.fitting.aicc import choose_model

U = 0.5 * float(np.finfo(float).eps)


# --------------------------------------------------------------------------
# The bound itself
# --------------------------------------------------------------------------


def _profile_loglik(resid_norm: float, n: int) -> float:
    return -n * math.log(resid_norm)


@pytest.mark.parametrize("q", [0.0, 0.1, 0.5, 0.9])
def test_band_contains_every_likelihood_shift_of_that_size(q: float) -> None:
    """Triangle-inequality bound: for any ``delta`` with ``||delta|| = q ||r||``
    the profile log-likelihood moves by at most ``[-down, +up]``. Checked
    against the extreme directions (``delta`` parallel / antiparallel to ``r``)
    and random ones."""
    rng = np.random.default_rng(7)
    n = 40
    r = rng.standard_normal(n)
    up, down = rx._loglik_rounding_band(n, float(np.linalg.norm(r)), q * float(np.linalg.norm(r)))
    base = _profile_loglik(float(np.linalg.norm(r)), n)
    directions = [r, -r] + [rng.standard_normal(n) for _ in range(50)]
    for v in directions:
        delta = q * np.linalg.norm(r) * v / np.linalg.norm(v)
        new = np.linalg.norm(r + delta)
        if new == 0.0:
            continue
        shift = _profile_loglik(float(new), n) - base
        assert -down - 1e-9 * n <= shift <= up + 1e-9 * n
    # The extremes are attained: the bound is not loose.
    assert math.isclose(down, n * math.log1p(q), rel_tol=1e-12, abs_tol=0.0)
    assert math.isclose(up, -n * math.log1p(-q), rel_tol=1e-12, abs_tol=0.0)


def test_band_is_unbounded_when_the_residual_is_no_larger_than_the_data_error() -> None:
    up, down = rx._loglik_rounding_band(10, 1.0, 1.0)
    assert up == float("inf") and math.isfinite(down)
    up, down = rx._loglik_rounding_band(10, 0.0, 1.0)
    assert up == float("inf")
    up, down = rx._loglik_rounding_band(10, float("nan"), 1.0)
    assert up == float("inf") and math.isnan(down)


# --------------------------------------------------------------------------
# Selection on a synthetic curve at the rounding level
# --------------------------------------------------------------------------


def _fits(t: np.ndarray, y: np.ndarray) -> dict:
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("M0", "M1", "M2", "M3a", "M3b"):
            try:
                out[name], _ = rx._fit_with_model(name, t, y)
            except (ValueError, RuntimeError):
                continue
    return out


def _floor_curve(seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """``0.04 exp(-t)`` over ten e-foldings plus noise of 20 u absolute --
    the phantom ladder's measured trajectory defect."""
    t = np.linspace(0.0, 10.0, 80)
    level = 20.0 * U
    rng = np.random.default_rng(seed)
    return t, 0.04 * np.exp(-t) + level * rng.standard_normal(t.size), level


def test_rounding_level_curve_selects_m0_and_the_plain_argmin_does_not() -> None:
    plain, policy = [], []
    for seed in range(8):
        t, y, level = _floor_curve(seed)
        fits = _fits(t, y)
        plain.append(choose_model({n: f.aicc for n, f in fits.items() if f.success}))
        winner, tied = rx._select_model(fits, t, y, level)
        policy.append(winner)
        assert winner == "M0" and "M0" in tied
    # Positive control: without the band the scenario is a live tie -- the
    # argmin leaves M0 on some realisation. Otherwise this test is blind.
    assert any(w != "M0" for w in plain), plain


def test_resolved_curve_is_not_widened_into_a_tie() -> None:
    """Noise far above the rounding level: the band is negligible and the
    policy returns exactly the plain AICc winner."""
    t = np.linspace(0.0, 10.0, 80)
    rng = np.random.default_rng(3)
    y = 0.6 * np.exp(-t) + 0.4 * np.exp(-6.0 * t) + 1e-4 * rng.standard_normal(t.size)
    fits = _fits(t, y)
    plain = choose_model({n: f.aicc for n, f in fits.items() if f.success})
    winner, tied = rx._select_model(fits, t, y, 20.0 * U)
    assert winner == plain and tied == (plain,)
    assert plain != "M0"


def test_tie_without_the_nested_model_is_indeterminate() -> None:
    t, y, level = _floor_curve(0)
    fits = _fits(t, y)
    fits["M0"] = dataclasses.replace(fits["M0"], success=False)
    winner, tied = rx._select_model(fits, t, y, level)
    assert winner == MODEL_SELECTION_INDETERMINATE
    assert len(tied) >= 2 and "M0" not in tied


def test_nan_band_keeps_the_candidate_tied() -> None:
    """Orientation: a candidate is excluded only by a TRUE comparison. A NaN
    residual (NaN parameters) cannot exclude M0, so the tie still selects it."""
    t, y, level = _floor_curve(0)
    fits = _fits(t, y)
    fits["M0"] = dataclasses.replace(
        fits["M0"], residual_ar1_rho=float("nan"), residual_theta_car1=float("nan"),
        aicc=-1.0e6,
    )
    # M0 now holds the minimum AICc but its residual model is unknown, so its
    # band -- and the winner's upper end -- is NaN: no comparison against it
    # is true, nothing may be excluded.
    winner, tied = rx._select_model(fits, t, y, level)
    assert winner == "M0"
    candidates = [n for n, f in fits.items() if f.success and np.isfinite(f.aicc)]
    assert tied == tuple(candidates)


# --------------------------------------------------------------------------
# The pipeline: INDETERMINATE propagates as "unknown", not as a number
# --------------------------------------------------------------------------


def test_indeterminate_selection_withholds_rate_and_interval(monkeypatch) -> None:
    from liouscope.examples import v3_amplitude_damped_qubit

    def always_indeterminate(fits, t, y, data_error):
        return MODEL_SELECTION_INDETERMINATE, tuple(fits)

    monkeypatch.setattr(rx,"_select_model", always_indeterminate)
    s = v3_amplitude_damped_qubit()
    rep = diagnose(s.L, rho_initial=s.rho_initial, bootstrap_B=20, seed=42)
    rel = rep.relaxation
    assert rel.aicc_model == MODEL_SELECTION_INDETERMINATE
    assert math.isnan(rel.beta_D)
    assert all(math.isnan(v) for v in rel.bca_ci_beta)
    assert rel.interval_method == INTERVAL_METHOD_UNAVAILABLE
    assert rel.linear_fit_model == MODEL_SELECTION_INDETERMINATE
    assert math.isnan(rel.beta_D_linear)
    assert rep.classification.evidence["d17_linear_single_exp"] == 0.0


# --------------------------------------------------------------------------
# The reported case: 1-ulp operator perturbation of the phantom ladder
# --------------------------------------------------------------------------

# Seeds whose perturbed operator flipped the trace-distance winner to M3a at
# d4aa5a7 (1-ulp probe, 7 of 25 flipped: 1, 3, 5, 8, 9, 23, 24). Four suffice
# to keep the test short; the full 25-seed probe is in the PR record.
_FLIP_SEEDS = (1, 3, 8, 9)


@pytest.mark.filterwarnings("ignore")
def test_one_ulp_operator_perturbation_no_longer_flips_the_linear_model() -> None:
    from test_validation_systems.test_d17_gap_coherence import _phantom_ladder

    L, rho0 = _phantom_ladder()
    eps = float(np.finfo(float).eps)
    plain = []
    for s in _FLIP_SEEDS:
        rng = np.random.default_rng(1000 + s)
        Lp = L + eps * np.abs(L) * (
            rng.standard_normal(L.shape) + 1j * rng.standard_normal(L.shape)
        )
        rel = diagnose(Lp, rho_initial=rho0, bootstrap_B=20, seed=42).relaxation
        assert rel.linear_fit_model == "M0", s
        t, y = rel.t_grid, rel.trace_distance_curve
        fits = _fits(t, y)
        plain.append(choose_model({n: f.aicc for n, f in fits.items() if f.success}))
    # Positive control: the plain argmin on these very curves is not M0 on at
    # least one seed, so the assertion above is carried by the tie policy.
    assert any(w != "M0" for w in plain), plain
