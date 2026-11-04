"""Issue #178 item 2: grid sweeps of ``sigma_min(zI - L)`` from one Schur form.

The resolvent and non-normality layers evaluate ``sigma_min(zI - L)`` (or its
reciprocal, the resolvent norm) on grids of 201 to 1250 points of one
operator. Above ``n = 128`` those sweeps now share one complex Schur
decomposition and run inverse Lanczos on the shifted triangle
(:class:`liouscope.numerics.resolvent.ShiftedSchurResolvent`). These tests pin:

* agreement with the dense SVD to the backward-error level, also next to the
  spectrum where ``sigma_min`` itself is of the order of round-off;
* that nothing below the cutoff changes (no Schur form is ever built there);
* that every consumer above the cutoff matches its own dense formula;
* that a clustered resolvent, where the previous power iteration stopped at its
  cap and warned, is now returned converged and without a warning;
* the exact dense fallback, the power-of-two rescaling, singular shifts and
  isolation from the global NumPy random state.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import liouscope.numerics.resolvent as resolvent
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.nonnormality import kreiss_constant, kreiss_grid_lower_bound
from liouscope.diagnostics.resolvent import (
    compute_resolvent_layer,
    resolvent_peak_curve,
    resolvent_peak_curve_scaled,
)
from liouscope.numerics.pseudospec import pseudospectral_radius, pseudospectrum_extent

# Module attributes are read at call time, so monkeypatching the module (the
# cutoff, the Schur class, the Krylov budget) reaches every consumer.
CUTOFF = resolvent._DENSE_NORM_CUTOFF
ShiftedSchurResolvent = resolvent.ShiftedSchurResolvent
resolvent_norm_evaluator = resolvent.resolvent_norm_evaluator
sigma_min_evaluator = resolvent.sigma_min_evaluator


def _random_gksl(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    jumps = [
        (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / d
        for _ in range(2)
    ]
    return build_liouvillian((H + H.conj().T) / np.sqrt(d), jumps)


def _ladder(d: int) -> np.ndarray:
    """Driven, damped and dephased ``d``-level ladder (non-normal)."""
    a = np.diag(np.sqrt(np.arange(1, d)), 1).astype(complex)
    H = a.conj().T @ a + 0.3 * (a + a.conj().T)
    return build_liouvillian(H, [np.sqrt(0.5) * a, np.sqrt(0.05) * a.conj().T @ a])


def _probe_points(L: np.ndarray) -> list[complex]:
    """Right of, inside and next to the spectrum, as the layers' grids are."""
    ev = sla.eigvals(L)
    re_max = float(np.max(ev.real))
    span = max(1.0, float(np.max(np.abs(ev.imag))) + 1.0)
    points = [
        complex(re_max + s, w)
        for s in (1e-3, 1e-1, 1.0, 10.0)
        for w in np.linspace(-span, span, 5)
    ]
    points += [complex(float(np.mean(ev.real)), w) for w in (-0.5 * span, 0.0, 0.5 * span)]
    points += [complex(e) + 1e-6 for e in ev[:: max(1, ev.size // 4)]]
    return points


def _dense_sigma_min(L: np.ndarray, z: complex) -> float:
    return float(sla.svdvals(z * np.eye(L.shape[0], dtype=complex) - L)[-1])


@pytest.mark.parametrize(
    "L", [_ladder(12), _random_gksl(12, 1)], ids=["ladder12", "gksl12"]
)
def test_schur_sigma_min_matches_the_dense_svd_to_backward_error(L):
    assert L.shape[0] > CUTOFF
    schur = ShiftedSchurResolvent(L)
    # The SVD of zI - L is itself accurate only to ~n eps ||zI - L||_F in
    # absolute terms, so next to an eigenvalue (sigma_min ~ 1e-6 here) the
    # comparison is absolute; away from it, it is 1e-9 relative.
    floor = 10.0 * L.shape[0] * np.finfo(float).eps * np.linalg.norm(L)
    for z in _probe_points(L):
        ref = _dense_sigma_min(L, z)
        got = schur.sigma_min(z)
        assert abs(got - ref) <= 1e-9 * ref + floor, (z, got, ref)
    assert schur.evaluations == len(_probe_points(L))
    assert schur.dense_fallbacks == 0


def test_nothing_below_the_cutoff_builds_a_schur_form(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("ShiftedSchurResolvent must not be built for n <= 128")

    monkeypatch.setattr(resolvent, "ShiftedSchurResolvent", refuse)
    L = _random_gksl(8, 2)
    assert L.shape[0] <= CUTOFF
    compute_resolvent_layer(L)
    kreiss_constant(L)
    kreiss_grid_lower_bound(L, n_sigma=4, n_omega=5)


def test_evaluators_below_the_cutoff_are_the_historical_dense_formulas():
    L = _random_gksl(8, 3)
    eye = np.eye(L.shape[0], dtype=complex)
    norm_at = resolvent_norm_evaluator(L)
    sigma_at = sigma_min_evaluator(L)
    for z in _probe_points(L)[:8]:
        # Bit-for-bit, not approximately: these are the pre-#178 expressions.
        assert norm_at(z) == resolvent.resolvent_norm(L, z)
        assert sigma_at(z) == float(sla.svdvals(z * eye - L)[-1])


def test_every_consumer_above_the_cutoff_matches_its_dense_formula(monkeypatch):
    L = _ladder(12)
    assert L.shape[0] > CUTOFF
    ev = sla.eigvals(L)
    grid_re = (float(ev.real.min()) - 1.0, float(ev.real.max()) + 1.0, 6)
    grid_im = (float(ev.imag.min()) - 1.0, float(ev.imag.max()) + 1.0, 5)

    def measure():
        return {
            "peak": resolvent_peak_curve(L, 1e-2, n_omega=11)[1],
            "peak_scaled": resolvent_peak_curve_scaled(L, 1e-3, n_omega=11)[1],
            "kreiss": kreiss_constant(L, n_sigma=5, n_omega=5),
            "kreiss_scaled": kreiss_grid_lower_bound(L, n_sigma=5, n_omega=5).value,
            "radius": pseudospectral_radius(L, 0.5, grid_re=grid_re, grid_im=grid_im),
            "extent": pseudospectrum_extent(L, 0.5, grid_re=grid_re, grid_im=grid_im),
        }

    fast = measure()
    monkeypatch.setattr(resolvent, "_DENSE_NORM_CUTOFF", 10**9)
    dense = measure()
    np.testing.assert_allclose(fast["peak"], dense["peak"], rtol=1e-9)
    np.testing.assert_allclose(fast["peak_scaled"], dense["peak_scaled"], rtol=1e-9)
    assert fast["kreiss"] == pytest.approx(dense["kreiss"], rel=1e-9)
    assert fast["kreiss_scaled"] == pytest.approx(dense["kreiss_scaled"], rel=1e-9)
    # D13 is a set-membership decision; it is identical unless a grid point
    # sits within the solver tolerance of eps, which this grid does not.
    assert fast["radius"] == dense["radius"]
    assert fast["extent"] == dense["extent"]


def _clustered(n: int, d2: float) -> np.ndarray:
    lam = np.empty(n, dtype=complex)
    lam[0] = 1.0
    lam[1] = d2
    lam[2:] = np.linspace(5.0, 20.0, n - 2)
    return np.diag(lam)


def test_a_clustered_resolvent_is_converged_where_the_power_iteration_was_capped():
    # The pair (1, 1.001) makes the top singular values of the resolvent at
    # z = 0 agree to 1e-3. resolvent_norm's 200-step power iteration stops at
    # its cap there and warns (tests/test_numerics.py pins that). The sweep
    # evaluator returns the exact value with no warning: pytest turns any
    # warning into an error (pyproject filterwarnings = "error").
    A = _clustered(160, 1.001)
    ref = float(sla.svdvals(sla.inv(-A))[0])
    got = resolvent_norm_evaluator(A)(0.0j)
    assert got == pytest.approx(ref, rel=1e-12)


def test_an_exhausted_krylov_budget_falls_back_to_an_exact_svd(monkeypatch):
    monkeypatch.setattr(resolvent, "_LANCZOS_MAX_STEPS", 2)
    L = _random_gksl(12, 4)
    schur = ShiftedSchurResolvent(L)
    floor = 10.0 * L.shape[0] * np.finfo(float).eps * np.linalg.norm(L)
    for z in _probe_points(L)[:6]:
        ref = _dense_sigma_min(L, z)
        assert abs(schur.sigma_min(z) - ref) <= 1e-12 * ref + floor
    assert schur.dense_fallbacks > 0


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_sigma_min_is_homogeneous_at_extreme_scales(scale):
    L = _random_gksl(12, 5)
    z = complex(float(np.max(sla.eigvals(L).real)) + 0.5, 0.3)
    ref = ShiftedSchurResolvent(L).sigma_min(z)
    got = ShiftedSchurResolvent(scale * L).sigma_min(scale * z)
    assert got == pytest.approx(scale * ref, rel=1e-9)


@pytest.mark.parametrize("scale", [1e-200, 1.0, 1e200])
def test_the_schur_form_is_taken_of_the_power_of_two_rescaled_operator(
    monkeypatch, scale
):
    # Same LAPACK hazard as the conditioning audit (issue #183): a build whose
    # xGEEV rescales an operator above ~1e146 internally failed to scale back.
    # This stack's xGEES and xGESDD handle 1e+-200 correctly, so the guard is
    # pinned directly: the factorised operator has its largest component in
    # [0.5, 1), which is exact (a power of two) and keeps LAPACK off that path.
    seen = []
    schur = sla.schur

    def recording(A, *args, **kwargs):
        arr = np.asarray(A)
        seen.append(max(np.max(np.abs(arr.real)), np.max(np.abs(arr.imag))))
        return schur(A, *args, **kwargs)

    monkeypatch.setattr(resolvent.sla, "schur", recording)
    ShiftedSchurResolvent(scale * _random_gksl(12, 5))
    assert len(seen) == 1
    assert 0.5 <= seen[0] < 1.0


def test_a_singular_shift_is_reported_not_divided_by():
    lam = np.concatenate([[0.0], -np.linspace(1.0, 5.0, CUTOFF + 1)]).astype(complex)
    A = np.diag(lam) + np.diag(np.full(lam.size - 1, 0.1), 1)
    schur = ShiftedSchurResolvent(A)
    assert schur.sigma_min(0.0j) == 0.0
    with pytest.raises(np.linalg.LinAlgError):
        schur.norm(0.0j)


def test_the_global_random_state_is_untouched():
    L = _random_gksl(12, 6)
    before = np.random.get_state()
    schur = ShiftedSchurResolvent(L)
    for z in _probe_points(L)[:4]:
        schur.sigma_min(z)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_each_value_is_a_function_of_the_operator_and_the_point_only():
    # No state carries over between evaluations: the same point gives the same
    # bits whether it is evaluated first, last or in a fresh evaluator.
    L = _ladder(12)
    points = _probe_points(L)
    sweep = ShiftedSchurResolvent(L)
    forward = [sweep.sigma_min(z) for z in points]
    backward = [sweep.sigma_min(z) for z in reversed(points)][::-1]
    fresh = [ShiftedSchurResolvent(L).sigma_min(z) for z in points[:3]]
    assert forward == backward
    assert fresh == forward[:3]


def test_an_overflowing_krylov_norm_falls_back_instead_of_raising():
    # PR #189 review: for a Jordan-like operator the triangular solves stay
    # finite while ||w||^2 overflows. That raised under the repository's
    # warnings-as-errors filter; it must reach the dense fallback instead.
    n = CUTOFF + 16
    L = -np.eye(n) + np.diag(np.ones(n - 1), 1)
    ref = _dense_sigma_min(L, -0.8)
    schur = ShiftedSchurResolvent(L)
    assert schur.sigma_min(-0.8) == pytest.approx(ref, rel=1e-9)
    assert schur.dense_fallbacks == 1


def _operator_hiding_its_dominant_mode(n: int) -> np.ndarray:
    """Upper-triangular ``T`` whose ``(-T)^{-1}(-T)^{-H}`` has eigenvalues
    ``10, 2, 1.7, 1.4, 1.1, 1, ...`` with the fixed Krylov start inside the
    invariant subspace of ``2 .. 1.1`` (PR #189 review)."""
    start = ShiftedSchurResolvent(-np.diag(np.arange(1.0, n + 1)))._start
    rng = np.random.default_rng(1)
    X = np.column_stack(
        [start] + [rng.standard_normal(n) + 1j * rng.standard_normal(n) for _ in range(n - 1)]
    )
    Q, _ = np.linalg.qr(X)
    W, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    Q[:, :4] = Q[:, :4] @ W
    lam = np.ones(n)
    lam[:5] = [2.0, 1.7, 1.4, 1.1, 10.0]
    M = (Q * lam) @ Q.conj().T
    flip = np.eye(n)[::-1]
    B = flip @ np.linalg.cholesky(flip @ M @ flip) @ flip  # upper, B B^H = M
    return -np.triu(np.linalg.inv(B))


def test_a_dominant_mode_hidden_from_the_krylov_start_is_caught_by_the_probe():
    # Lanczos alone certifies 2 (sigma = 0.707) by a small residual; the probe
    # sees a larger Rayleigh quotient and the dense SVD returns 1/sqrt(10).
    T = _operator_hiding_its_dominant_mode(CUTOFF + 16)
    schur = ShiftedSchurResolvent(T)
    assert schur.sigma_min(0.0) == pytest.approx(1.0 / np.sqrt(10.0), rel=1e-9)
    assert schur.dense_fallbacks == 1


def test_the_probe_raises_no_false_alarm_on_physical_generators():
    L = _ladder(12)
    schur = ShiftedSchurResolvent(L)
    for z in _probe_points(L):
        schur.sigma_min(z)
    assert schur.dense_fallbacks == 0


def test_a_shift_outside_the_operators_scale_is_evaluated_unscaled(monkeypatch):
    # PR #189 review: for a 1e-320 operator the legacy D13 grid (absolute
    # 1e-3 floor) puts z far outside the operator's power-of-two scale, and
    # ldexp(z, 1062) overflowed. Such a point is the historical dense SVD.
    L = -1e-320 * np.eye(CUTOFF + 16)
    fast = pseudospectral_radius(L)
    schur = ShiftedSchurResolvent(L)
    assert schur.sigma_min(1e-3) == _dense_sigma_min(L, 1e-3)
    assert schur.dense_fallbacks == 1
    monkeypatch.setattr(resolvent, "_DENSE_NORM_CUTOFF", 10**9)
    assert fast == pseudospectral_radius(L)


def test_a_shift_that_underflows_under_the_scale_is_evaluated_unscaled():
    # PR #189 review: against a 1e200 operator whose other eigenvalues are 0,
    # z = 1e-200 is not negligible, but ldexp(z, -665) underflowed to 0 and
    # reported an exactly singular zI - L (sigma 0, LinAlgError in norm()).
    L = np.diag(np.r_[-1e200, np.zeros(CUTOFF + 15)]).astype(complex)
    schur = ShiftedSchurResolvent(L)
    assert schur.sigma_min(1e-200) == pytest.approx(1e-200, rel=1e-12)
    assert schur.norm(1e-200) == pytest.approx(1e200, rel=1e-12)


def test_an_operator_the_rescale_would_round_is_used_unscaled():
    # Scaling diag(-1e10, -1e-300, ...) down by 2**-34 would make -1e-300
    # subnormal and round it; the operator is then factorised as given.
    L = np.diag(np.r_[-1e10, -1e-300, -np.ones(CUTOFF + 14)]).astype(complex)
    schur = ShiftedSchurResolvent(L)
    assert schur._exponent == 0
    assert schur.sigma_min(0.0) == pytest.approx(1e-300, rel=1e-12)
