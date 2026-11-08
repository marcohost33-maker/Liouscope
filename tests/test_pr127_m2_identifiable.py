"""M2 is a model-selection candidate only when it is fitted identifiably.

Follow-up to PR #127 (option B), pre-registered in
``docs/prereg/2026-11-08_m2_identifiability.md``.

Root cause (measured at 6b1cd93, Windows numpy 2.5.4 / scipy 1.18.1): on the
gap-scaled default window of ``_chain1`` the M2 fit starts from beta2 ~ 48.8,
where the Jacobian column of beta2 is almost zero. ``x_scale="jac"`` divides
by that column norm, the solver jumps to beta2 ~ 1e6 -- a component that
decays inside the first grid step and explains the single point t = 0 --
and stops on ``xtol`` (status 3) at a point decided by rounding. AICc then
preferred that M2 over M0 by ~290 units, on every platform, at one of two
platform-dependent ``beta_D`` values.

Two changes, tested here:

* M2 is optimised with the characteristic scale of its variables
  (amplitudes in units of the curve, rates in units of the inverse window)
  instead of the Jacobian column norms (``models.characteristic_x_scale``);
* an M2 fit whose rates lie outside the window the grid can identify
  (``ALPHA_SEED_FLOOR_FRAC / T <= beta <= 10 / dt_0``, amendment A1 of the
  prereg), or whose column-equilibrated Jacobian is singular, is "not
  identified": infinite
  AICc and excluded from EVERY branch of ``_select_model``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from liouscope._types import FitResult
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics import relaxation as rel
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M2, initial_guess_m2

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

# M0 rate of ``_chain1`` on the default window. Measured 0.75344700 with a
# spread of +-5e-10 (Windows) / +-1e-10 (Linux) over 1-ulp shifts of the field
# amplitude (decision dossier 2026-11-07, A.1 item 5) and 0.753447 on the
# legacy-style M0 fit of this branch. rel 1e-6 is >= 1000x that spread and
# 4 orders of magnitude below the distance to either former M2 value
# (0.71738 / 0.72143, 4-5 % away), so it separates "M0 fitted" from "M2
# artefact" on every platform while failing on any real change of the fit.
_BETA_M0 = 0.753447
_RTOL_BETA_M0 = 1.0e-6


def _chain1_layer(**kw):
    L = build_liouvillian(0.3 * _SX, [math.sqrt(0.7) * _SM])
    psi = np.ones(2, dtype=complex) / math.sqrt(2.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rel.compute_relaxation_layer(
            L, rho_initial=np.outer(psi, psi.conj()), bootstrap_B=0, **kw
        )


@pytest.fixture(scope="module")
def chain1():
    return _chain1_layer()


def _two_rate(seed: int, beta2: float, a2: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 10.0, 80)
    y = np.exp(-0.5 * t) + a2 * np.exp(-beta2 * t) + 1.0e-3 * rng.standard_normal(t.size)
    return t, y


# --------------------------------------------------------------------------
# The reported case
# --------------------------------------------------------------------------


def test_chain1_default_window_selects_m0_at_its_rate(chain1) -> None:
    assert chain1.t_grid_source == "gap_scaled"
    beta = float(chain1.beta_D)
    # NaN must fail closed: isclose(nan, x) is False, and the explicit check
    # names the failure instead of a bare comparison.
    assert math.isfinite(beta), f"beta_D is not finite: {beta!r}"
    assert chain1.aicc_model == "M0"
    assert math.isclose(beta, _BETA_M0, rel_tol=_RTOL_BETA_M0), beta


def test_chain1_m2_converges_inside_the_window_and_loses_on_its_own_score(chain1) -> None:
    """With the scaling fix the second rate converges to 7.67
    (``beta2 * dt_0 = 2.77``, inside the identifiable window of amendment A1).
    M2 then leaves the comparison through its own AICc: on this uniform grid
    the sample size is ``N_eff`` = 4.7 < k + 1 = 5, so the small-sample
    correction is infinite (decision dossier 2026-11-07, A.1 item 4)."""
    fr = chain1.fits["M2"]
    assert fr.success
    dt0 = float(chain1.t_grid[1] - chain1.t_grid[0])
    fast = max(float(fr.params[1]), float(fr.params[3]))
    assert 1.0 < fast * dt0 < 10.0, fr.params
    assert fr.identifiable, fr.identification_reason
    assert fr.n_eff <= 4 + 1
    assert not np.isfinite(fr.aicc)


def test_gate_refuses_the_former_chain1_endpoint(chain1) -> None:
    """The ``x_scale="jac"`` endpoint measured at 6b1cd93 (Windows): the
    one-point component the artefact was made of."""
    t = np.asarray(chain1.t_grid, dtype=float)
    jac_endpoint = np.array([1.14268627, 0.721428703, 0.0697351193, 1.05517695e6])
    ok, reason = rel._m2_identification(t, jac_endpoint)
    assert ok is False
    assert "rate" in reason, reason


def test_m2_fit_on_chain1_reaches_an_interior_optimum(chain1) -> None:
    """The scaling fix on its own, without the gate: the fast rate stays on
    the grid's scale (measured 7.67, beta2*dt = 2.77) instead of running to
    ~1e6, and the likelihood is higher there (484.79 vs 472.29)."""
    t = np.asarray(chain1.t_grid, dtype=float)
    y = np.asarray(chain1.relative_entropy_curve, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_gls_ar1(M2, t, y, initial_guess_m2(t, y))
    dt = float(t[1] - t[0])
    fast = max(float(out.params[1]), float(out.params[3]))
    assert out.success
    assert fast * dt < 10.0, out.params
    assert out.log_likelihood > 480.0, out.log_likelihood


# --------------------------------------------------------------------------
# Positive and negative controls on simulated curves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_resolved_two_rate_curve_still_selects_m2(seed: int) -> None:
    """Positive control: the gate must not remove a genuine, resolved
    second rate (beta2 dt = 0.38)."""
    t, y = _two_rate(seed, 3.0)
    fr, _ = rel._fit_with_model("M2", t, y)
    assert fr.identifiable, fr.identification_reason
    assert np.isfinite(fr.aicc)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, model = rel._dominant_rate(t, y)
    assert model == "M2"
    assert math.isclose(rate, 0.5, rel_tol=1.0e-2)


@pytest.mark.parametrize("seed", [11, 12, 13])
def test_stepped_over_second_rate_never_wins_with_a_wrong_slow_rate(seed: int) -> None:
    """beta2 dt = 63 (case C2' of amendment A1). What the gate guarantees:
    M2 is selected only as an identified fit, and then its slow rate is the
    true 0.5 -- the fast rate it reports is not a measurement either way."""
    t, y = _two_rate(seed, 500.0)
    fr, _ = rel._fit_with_model("M2", t, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rate, model = rel._dominant_rate(t, y)
    if model == "M2":
        assert fr.identifiable
        assert math.isclose(rate, 0.5, rel_tol=1.0e-2), rate
    else:
        assert fr.identifiable is False
        assert "rate" in fr.identification_reason


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Open limit (prereg result, run 2, S4'): with noise 1e-3 a component "
        "at beta2 = 500 is fitted at the plateau edge, beta2 ~ 48-53 "
        "(beta2*dt_0 ~ 6 < 10); window and conditioning cannot see a flat "
        "rate profile there. Measured 6/10 seeds identified. Needs a "
        "profile-likelihood test (decision dossier 2026-11-07, option B item 2)."
    ),
)
@pytest.mark.parametrize("seed", [11, 12])
def test_stepped_over_second_rate_is_not_identified(seed: int) -> None:
    t, y = _two_rate(seed, 500.0)
    fr, _ = rel._fit_with_model("M2", t, y)
    assert fr.identifiable is False


# --------------------------------------------------------------------------
# The gate itself: fail-closed edges
# --------------------------------------------------------------------------


_T = np.linspace(0.0, 10.0, 80)


@pytest.mark.parametrize(
    ("params", "fragment"),
    [
        ([1.0, 0.5, 0.5, float("nan")], "finite"),
        ([1.0, float("inf"), 0.5, 3.0], "finite"),
        ([1.0, 0.5, 0.5, 1.0e6], "rate"),       # stepped over
        ([1.0, 0.5, 0.5, -0.3], "rate"),        # growing component
        ([1.0, 0.5, 0.5, 1.0e-5], "rate"),      # indistinguishable from C
        ([1.0, 0.5, 0.0, 3.0], "Jacobian"),     # A2 = 0: beta2 column zero
        ([1.0, 0.5, 0.5, 0.5], "Jacobian"),     # equal rates: rank 3
    ],
)
def test_identification_refuses(params: list[float], fragment: str) -> None:
    ok, reason = rel._m2_identification(_T, np.array(params))
    assert ok is False
    assert fragment in reason, reason


def test_identification_accepts_a_resolved_pair() -> None:
    ok, reason = rel._m2_identification(_T, np.array([1.0, 0.5, 0.5, 3.0]))
    assert ok is True, reason
    assert reason == ""


def test_identification_is_invariant_under_a_change_of_time_unit() -> None:
    p = np.array([1.0, 0.5, 0.5, 3.0])
    for c in (1.0e-9, 1.0e9):
        q = p.copy()
        q[[1, 3]] /= c
        assert rel._m2_identification(_T * c, q) == (True, "")
    bad = np.array([1.0, 0.5, 0.5, 1.0e6])
    for c in (1.0e-9, 1.0e9):
        q = bad.copy()
        q[[1, 3]] /= c
        assert rel._m2_identification(_T * c, q)[0] is False


def test_identification_refuses_a_degenerate_grid() -> None:
    ok, reason = rel._m2_identification(np.zeros(5), np.array([1.0, 0.5, 0.5, 3.0]))
    assert ok is False
    assert "grid" in reason


def _fr(name: str, aicc: float, *, success: bool = True, identifiable: bool = True,
        n_params: int = 2) -> FitResult:
    return FitResult(
        model=name, params=np.ones(n_params), log_likelihood=0.0, aicc=aicc,
        n_eff=80.0, residual_ar1_rho=0.0, success=success,
        identifiable=identifiable,
        identification_reason="" if identifiable else "rate outside window",
    )


def test_unidentified_m2_is_never_the_all_infinite_fallback() -> None:
    """With every AICc infinite ``_select_model`` keeps a historical fallback
    to the first succeeded fit. A non-identified M2 must not be it."""
    fits = {
        "M0": _fr("M0", float("inf"), success=False),
        "M2": _fr("M2", float("inf"), identifiable=False, n_params=4),
    }
    y = np.exp(-_T)
    winner, tied = rel._select_model(fits, _T, y, 1.0e-16)
    assert winner == "none"
    assert tied == ()


def test_unidentified_m2_with_a_finite_score_is_still_excluded() -> None:
    """Defence in depth: even a caller-built FitResult that carries a finite
    AICc but ``identifiable=False`` does not win."""
    fits = {
        "M0": _fr("M0", -10.0),
        "M2": _fr("M2", -1.0e6, identifiable=False, n_params=4),
    }
    y = np.exp(-_T)
    winner, _ = rel._select_model(fits, _T, y, 1.0e-16)
    assert winner == "M0"


# --------------------------------------------------------------------------
# Amplitude and time-unit invariance of the new scaling
# --------------------------------------------------------------------------


# Invariance tolerance. The solver stops at ``ftol = 1e-8`` on the cost, so
# two runs that differ only by the rounding of ``c*y`` stop at slightly
# different points. Measured on seed 11, c in {1e-150 .. 1e40}: rates agree to
# <= 1.23e-9 relative (6b1cd93 with ``x_scale="jac"``: <= 2.66e-9). The
# pre-registered 1e-9 (S6) was below that stopping level -- reported as a
# miss in docs/prereg/2026-11-08_m2_identifiability_result.md, not re-tuned
# there. 1e-7 is ~100x the measured spread and 3e6x below the failures it
# separates: ``x_scale=1.0`` at 1e+-40 (rel 0.49) and ``"jac"`` at a time
# unit of 1e9 (rel 0.30, reported as success).
_RTOL_INVARIANT = 1.0e-7


@pytest.mark.parametrize("scale", [1.0e-150, 1.0e-40, 1.0e40])
def test_m2_rates_are_amplitude_invariant(scale: float) -> None:
    t, y = _two_rate(11, 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = fit_gls_ar1(M2, t, y, initial_guess_m2(t, y))
        out = fit_gls_ar1(M2, t, scale * y, initial_guess_m2(t, scale * y))
    assert out.success
    for i in (1, 3):
        assert math.isclose(out.params[i], ref.params[i], rel_tol=_RTOL_INVARIANT)


@pytest.mark.parametrize("c", [1.0e-9, 1.0e6, 1.0e9])
def test_m2_rates_follow_a_change_of_time_unit(c: float) -> None:
    """At c = 1e9 the ``"jac"`` fit of 6b1cd93 returned rates 30 % off with
    ``success=True``: SciPy's ``xtol`` is tested in unscaled variables, where
    a rate of 5e-10 next to an amplitude of 1 has "converged" at once."""
    t, y = _two_rate(11, 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = fit_gls_ar1(M2, t, y, initial_guess_m2(t, y))
        out = fit_gls_ar1(M2, t * c, y, initial_guess_m2(t * c, y))
    assert out.success
    for i in (1, 3):
        assert math.isclose(out.params[i] * c, ref.params[i], rel_tol=_RTOL_INVARIANT)


# --------------------------------------------------------------------------
# Multistart (pre-registered, item 4)
# --------------------------------------------------------------------------


def _multistart(t: np.ndarray, y: np.ndarray, n: int = 50) -> list[FitResult]:
    rng = np.random.default_rng(20261108)
    span = float(t[-1] - t[0])
    dt0 = float(t[1] - t[0])
    lo, hi = math.log(5.0e-3 / span), math.log(10.0 / dt0)
    amp = 0.5 * float(np.max(np.abs(y)))
    out: list[FitResult] = []
    original = rel.initial_guess_m2
    try:
        for _ in range(n):
            b1, b2 = np.exp(rng.uniform(lo, hi, size=2))
            start = np.array([amp, b1, amp, b2])
            rel.initial_guess_m2 = lambda _t, _y, s=start: s.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fr, _ = rel._fit_with_model("M2", t, y)
            out.append(fr)
    finally:
        rel.initial_guess_m2 = original
    return out


@pytest.mark.slow
def test_multistart_chain1_never_finds_an_identified_m2_beating_m0(chain1) -> None:
    t = np.asarray(chain1.t_grid, dtype=float)
    y = np.asarray(chain1.relative_entropy_curve, dtype=float)
    m0_aicc = float(chain1.fits["M0"].aicc)
    fits = _multistart(t, y)
    beating = [fr.params for fr in fits if fr.identifiable and fr.aicc < m0_aicc]
    assert beating == []


@pytest.mark.slow
def test_multistart_resolved_two_rate_curve_is_start_independent() -> None:
    t, y = _two_rate(11, 3.0)
    fits = _multistart(t, y)

    def cost(fr: FitResult) -> float:
        r = y - M2(t, fr.params)
        return 0.5 * float(r @ r)

    best = min(cost(fr) for fr in fits if fr.success)
    good = [
        fr for fr in fits
        if fr.success and fr.identifiable
        and math.isclose(cost(fr), best, rel_tol=1.0e-6)
    ]
    assert len(good) >= 45, len(good)
