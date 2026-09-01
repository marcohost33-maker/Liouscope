"""Round-20 review of PR #121: three more gates that certified what they had not checked.

The three findings are the same shape as rounds 17-19, one layer deeper each
time -- a decision is published that the code never made:

* the spectral layer withheld D1 on ``not resolved`` but D3/D4 and the
  oscillating-pair flag on the NARROWER ``not certified``, so an ambiguous
  certificate still published them (and published the ABSENCE of the very
  oscillation that made it ambiguous);
* the trace-preservation gate compared a finite defect against ``tp_rtol *
  inf``, and no finite number exceeds infinity, so an operator with a defect
  of 4.1 was certified as trace preserving;
* the bootstrap retained non-converged replicates on the claim that this can
  only widen the interval -- but a failed fit returns its unchanged STARTING
  value, which piles artificial mass on the centre and NARROWS the BCa
  quantiles.

Every test pairs the failing case with a positive control, so a repair that
refuses everything cannot pass.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from liouscope import build_liouvillian

# ---------------------------------------------------------------------------
# 1. An AMBIGUOUS certificate must withhold D3/D4 and the pair flag, like D1
# ---------------------------------------------------------------------------

# A certificate can be ``certified=True`` and still ``resolved=False``: issue
# #113's ambiguous in-band mode, neither machine-zero nor decidably non-zero.
# Round 19 gave D3/D4 the predicate ``not certified``, which this state passes,
# so they were published from a spectrum the layer cannot resolve.
_AMBIGUOUS_SPECTRUM = np.array(
    [0.0 + 0.0j, -1.0e-13 + 1.0e-13j, -1.0 + 0.0j, -2.0 + 0.0j]
)
_BOUND = 1.0e-12


def _certificate(*, ambiguous_count: int):
    """A REAL production certificate, ambiguous or resolved on demand.

    Built from the dataclass rather than from a generator that happens to
    provoke the state: the contract under test is stated in terms of
    ``resolved``, and a LAPACK-dependent fixture would test the fixture.
    """
    from liouscope.numerics.linalg import ZeroModeCertificate

    return ZeroModeCertificate(
        applicable=True,
        certified=True,
        solver="zgeev",
        residual=0.0,
        bound=_BOUND,
        trace_defect=0.0,
        ambiguous_count=ambiguous_count,
    )


def _healthy_generator() -> np.ndarray:
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    return build_liouvillian(H, [lowering], [0.5])


def _spectral_with(monkeypatch, certificate):
    from liouscope.diagnostics import spectral as sp

    L = _healthy_generator()
    monkeypatch.setattr(
        sp, "certified_eigvals", lambda *a, **k: (_AMBIGUOUS_SPECTRUM, certificate)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.compute_spectral_layer(L)


def test_the_ambiguous_mode_is_the_one_d3_would_have_denied() -> None:
    """Pins WHY the old predicate was wrong, not just that it was.

    The ambiguous mode sits inside ``zero_tol`` and carries the only nonzero
    imaginary part in the spectrum. The zero-mode filter therefore deletes it
    before D3 and the pair flag ever look, and what they then report is the
    absence of the oscillation that made the certificate unresolvable. If
    this stops holding, the test below no longer exercises the failure.
    """
    from liouscope.diagnostics.spectral import oscillating_mode_gap, spectral_spread

    assert np.abs(_AMBIGUOUS_SPECTRUM[1]) < _BOUND, "the mode must be in band"
    assert np.abs(_AMBIGUOUS_SPECTRUM[1].imag) > 0.0, "and it must oscillate"
    # What the old code published from this spectrum:
    assert not bool(np.any(np.abs(np.imag(_AMBIGUOUS_SPECTRUM)) > _BOUND)), (
        "the pair flag read False -- an assertion that nothing oscillates"
    )
    assert oscillating_mode_gap(_AMBIGUOUS_SPECTRUM, atol=_BOUND) == 0.0
    assert np.isfinite(spectral_spread(_AMBIGUOUS_SPECTRUM, atol=_BOUND))


def test_ambiguous_certificate_withholds_d3_d4_and_the_pair_flag(monkeypatch) -> None:
    """THE regression: D3/D4 must travel with D1, on D1's predicate."""
    result = _spectral_with(monkeypatch, _certificate(ambiguous_count=1))

    assert np.isnan(result.gap), "D1 was already withheld here"
    assert np.isnan(result.oscillating_gap), (
        "D3 published 0.0 -- 'no oscillating gap' -- for a spectrum whose "
        "only oscillating mode the filter had just discarded"
    )
    assert np.isnan(result.spectral_spread), "D4 was read off a truncated spectrum"
    assert result.has_complex_pairs is None, (
        "False asserts 'no oscillating modes'; None says the question was "
        "not answered"
    )


def test_a_resolved_certificate_still_publishes_d3_d4_and_the_flag(monkeypatch) -> None:
    """Positive control differing in ONE field: ``ambiguous_count``.

    Same spectrum, same solver, same bound. If the repair had become a
    blanket refusal, this fails.
    """
    result = _spectral_with(monkeypatch, _certificate(ambiguous_count=0))

    assert np.isfinite(result.gap)
    assert np.isfinite(result.oscillating_gap)
    assert np.isfinite(result.spectral_spread)
    assert result.has_complex_pairs is not None


# ---------------------------------------------------------------------------
# 2. A band that accepts the whole spectrum has certified nothing
# ---------------------------------------------------------------------------

def _overflowing_norm_operator() -> np.ndarray:
    """The reviewer's counterexample, entry for entry.

    Finite 4x4, cancelling ``+-1e308`` in one trace equation. A naive
    sum-of-squares Frobenius norm overflows, but the TRUE norm is the finite
    representable value ``sqrt(2)*1e308``. The absolute trace defect is
    ``sqrt(17) = 4.12`` and the componentwise trace-equation error is 1: other
    enormous entries are not allowed to dilute the violated columns.
    Eigenvalues are exactly ``{1, 2, 3, 4}``.
    """
    L = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    L[0, 1] = 1.0e308
    L[3, 1] = -1.0e308
    return L


def _nilpotent_wide_band_operator() -> np.ndarray:
    """The same band failure WITHOUT a trace-preservation failure.

    Strictly upper triangular with ``1e10`` entries confined to row 1, so
    ``vec(I)^H L = 0`` EXACTLY (trace preserving, defect 0.0), ``||L||_F`` is
    a perfectly ordinary ``1.4e10``, and every eigenvalue is exactly zero.
    The band ``rtol * eps * ||L||_2`` is then ``3.1e-3`` and swallows all four
    -- a nilpotent generator whose band has discriminated nothing.
    """
    L = np.zeros((4, 4), dtype=complex)
    L[1, 2] = 1.0e10
    L[1, 3] = 1.0e10
    return L


def test_large_operator_measurement_and_local_tp_error_are_both_real() -> None:
    """Pin the corrected arithmetic and the independent policy precondition."""
    from liouscope.numerics.linalg import (
        trace_preservation_componentwise_error,
        trace_preservation_defect,
    )

    L = _overflowing_norm_operator()
    assert np.all(np.isfinite(L)), "the input must be FINITE for this to be the case"
    defect, fro = trace_preservation_defect(L)
    local_error = trace_preservation_componentwise_error(L)

    assert defect == pytest.approx(np.sqrt(17.0)), "a real, order-one trace defect"
    assert np.isfinite(fro), "a representable norm must not be turned into infinity"
    assert fro == pytest.approx(np.sqrt(2.0) * 1.0e308, rel=2.0e-15)
    assert local_error == pytest.approx(1.0), (
        "at least one trace equation has no cancellation at all; unrelated "
        "1e308 entries must not make that violation look relatively small"
    )
    assert np.allclose(np.sort(np.linalg.eigvals(L).real), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    ("make", "expected_applicable"),
    [(_overflowing_norm_operator, False), (_nilpotent_wide_band_operator, True)],
)
def test_a_non_tp_input_or_a_nondiscriminating_band_does_not_certify(
    make, expected_applicable: bool
) -> None:
    """Both ladders fail closed, but the two counterexamples fail differently."""
    from liouscope.numerics.linalg import certified_eig, certified_eigvals

    L = make()
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert_vals = certified_eigvals(L)
        _, cert_vec = certified_eig(L)

    for cert in (cert_vals, cert_vec):
        assert cert.certified is False
        assert cert.resolved is False
        assert cert.applicable is expected_applicable
        assert cert.zero_mode_count == 0


def test_componentwise_tp_gate_refuses_local_defect_before_zero_mode_claim() -> None:
    """The #130 regression: finite global norm must not create a fail-open.

    The true Frobenius norm is ~1.4e308, so a GLOBAL normwise quotient makes
    ``sqrt(17)/||L||`` microscopic. Trace preservation, however, consists of
    separate cancellation equations. Several columns here have order-one
    equation error, which invalidates the exact-zero-mode theorem regardless
    of unrelated entries elsewhere in the operator.
    """
    from liouscope.numerics.linalg import (
        certified_eigvals,
        trace_preservation_componentwise_error,
        trace_preservation_defect,
    )

    L = _overflowing_norm_operator()
    defect, fro = trace_preservation_defect(L)
    local_error = trace_preservation_componentwise_error(L)

    assert np.isfinite(defect)
    assert np.isfinite(fro)
    assert local_error == pytest.approx(1.0)

    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert = certified_eigvals(L)

    assert cert.applicable is False
    assert cert.certified is False
    assert cert.resolved is False


def test_a_healthy_generator_still_certifies() -> None:
    """Positive control: an ordinary band still separates and still passes."""
    from liouscope.numerics.linalg import certified_eig, certified_eigvals

    L = _healthy_generator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert_vals = certified_eigvals(L)
        _, cert_vec = certified_eig(L)

    for cert in (cert_vals, cert_vec):
        assert cert.resolved is True
        assert cert.zero_mode_count == 1


def test_the_exactly_zero_generator_is_not_caught_by_the_guard() -> None:
    """Over-correction guard, and the one case the criterion must exempt.

    ``L = 0`` has ``bound = 0``: a band of width zero accepts only exact
    zeros, so it discriminates by construction even though it contains the
    whole spectrum. "Everything is stationary" is here the correct physics,
    measured exactly. A guard phrased as "zero_count == size" alone would
    have refused it.
    """
    from liouscope.numerics.linalg import certified_eigvals

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert = certified_eigvals(np.zeros((4, 4), dtype=complex))

    assert cert.bound == 0.0
    assert cert.certified is True
    assert cert.zero_mode_count == 4


# ---------------------------------------------------------------------------
# 3. No confidence interval from a resample that contains non-fits
#    (this finding and issue #123 are the same defect, so one repair covers both)
# ---------------------------------------------------------------------------

def test_retaining_failed_replicates_narrows_the_interval() -> None:
    """The premise the previous round got backwards, measured.

    Round 19 retained non-converged replicates on the argument that keeping
    them "widens the interval -- the conservative direction". A failed
    ``least_squares`` returns its unchanged STARTING value, which in this
    resample is ``theta_hat`` itself, so every failure piles mass on the
    centre. This test computes both intervals and shows the retained one is
    the NARROWER of the two; without it the repair would rest on an argument
    rather than on a number.
    """
    from liouscope.fitting.bootstrap import bca_ci

    rng = np.random.default_rng(0)
    theta_hat = np.array([1.0, 1.3])
    good = rng.normal(theta_hat, [0.05, 0.05], size=(400, 2))
    n_fail = 160
    mixed = good.copy()
    mixed[:n_fail] = theta_hat  # what a failed fit hands back

    dropped = bca_ci(good[n_fail:], theta_hat)
    retained = bca_ci(mixed, theta_hat)
    w_dropped = dropped[:, 1] - dropped[:, 0]
    w_retained = retained[:, 1] - retained[:, 0]

    assert np.all(w_retained < w_dropped), (
        "retaining failed replicates must be shown to NARROW the interval; "
        f"widths retained={w_retained} vs dropped={w_dropped}"
    )


def test_bootstrap_refuses_an_interval_containing_failed_replicates(monkeypatch) -> None:
    """THE regression for the finding as stated.

    The failure is injected at the documented contract surface -- a fit whose
    ``success`` is False -- rather than by hunting for data that makes
    ``least_squares`` give up, which would make the test depend on SciPy's
    convergence heuristics instead of on the rule under test.
    """
    from liouscope.fitting import bootstrap as bs
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 5.0, 48)
    y = np.exp(-1.3 * t)
    p0 = np.array([1.0, 1.2])
    real_fit = bs.fit_gls_ar1
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        out = real_fit(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1 and calls["n"] % 3 == 0:  # never the BASE fit
            return replace(out, success=False)
        return out

    monkeypatch.setattr(bs, "fit_gls_ar1", flaky)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="did not converge"):
            bs.parametric_bootstrap(M0, t, y, p0, B=9)


def test_bootstrap_still_reports_when_every_replicate_converged() -> None:
    """Positive control: the refusal must not have become unconditional.

    The curve carries NOISE deliberately. A noise-free curve is fitted to
    round-off, ``sigma`` comes out at ~1e-15, every replicate lands on
    ``theta_hat`` and the interval collapses to zero width -- true to the
    parametric bootstrap's own model and useless as a control, since it
    cannot tell a working interval from a refusal.
    """
    from liouscope.fitting.bootstrap import bca_ci, parametric_bootstrap
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 5.0, 48)
    rng = np.random.default_rng(7)
    y = np.exp(-1.3 * t) + rng.normal(0.0, 0.01, size=t.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, theta_hat = parametric_bootstrap(
            M0, t, y, np.array([1.0, 1.2]), B=30
        )
        ci = bca_ci(samples, theta_hat)

    assert samples.shape == (30, 2)
    assert np.all(np.isfinite(ci))
    assert np.all(ci[:, 1] > ci[:, 0]), "a real resample has a non-degenerate CI"


# --- issue #123: the same defect one layer down -----------------------------

def test_a_fit_on_a_curve_without_variation_returns_no_parameters() -> None:
    """Issue #123: the seed came back wearing the shape of a measurement.

    With zero data variation every direction is equally optimal, so
    ``least_squares`` reports "gradient is small" at the STARTING point. The
    seed is chosen here to differ from every rate in the positive control
    below, so the regression pins that the old return value was the seed and
    not a coincidence.
    """
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 5.0, 64)
    seed = np.array([1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_gls_ar1(M0, t, np.zeros_like(t), seed)

    assert out.success is False
    assert out.degenerate is True
    assert np.all(np.isnan(out.params)), (
        "the old return was ``[~0, 1.0]`` -- the seed's rate, unchanged"
    )


def test_the_zero_curve_no_longer_yields_a_zero_width_confidence_interval() -> None:
    """The two findings meet here: measured CI width was EXACTLY 0.0.

    Perfect confidence about a parameter nothing was measured for is the
    worst available output of an uncertainty pipeline, and it arose with
    ZERO failed replicates -- which is why the resample guard alone could not
    have caught it.
    """
    from liouscope.fitting.bootstrap import parametric_bootstrap
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 5.0, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="issue #123"):
            parametric_bootstrap(M0, t, np.zeros_like(t), np.array([1.0, 1.0]), B=20)


def test_a_curve_with_signal_is_still_fitted() -> None:
    """Positive control: the guard must not have become a blanket refusal.

    The seed's rate is 1.0 and the true rate is 1.3, so returning the seed
    would be visible. Without this the degenerate-curve test above would pass
    against an implementation that refuses every curve.
    """
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 5.0, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_gls_ar1(M0, t, np.exp(-1.3 * t), np.array([1.0, 1.0]))

    assert out.success is True
    assert out.degenerate is False
    assert out.params[1] == pytest.approx(1.3, rel=1.0e-3), (
        "and it must recover the rate, not return the seed 1.0"
    )


def test_the_guard_keys_on_variation_and_not_on_smallness() -> None:
    """Anti-absolute-floor control, the #108/#111 lesson applied to a curve.

    ``1e-40 * exp(-1.3 t)`` is minuscule in absolute terms and fully
    resolvable in relative ones. An absolute floor would refuse it and
    reintroduce exactly the rate-unit dependence the spectral layer spent
    two issues removing, so ``degenerate`` must stay False.

    Only ``degenerate`` is asserted. Whether the OPTIMISER can work at that
    magnitude is a different question with a different answer -- measured,
    ``least_squares`` stops at its absolute ``ftol``/``gtol`` on a cost of
    order 1e-80 and returns the seed with ``success=True``. That is the same
    seed-passthrough shape as issue #123, arriving through SciPy's absolute
    tolerances rather than through flatness, and it is NOT repaired here;
    pinning it as expected behaviour would be pinning a defect.
    """
    from liouscope.fitting.gls import fit_gls_ar1
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 5.0, 64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fit_gls_ar1(M0, t, 1.0e-40 * np.exp(-1.3 * t), np.array([1.0e-40, 1.0]))

    assert out.degenerate is False, (
        "a relative criterion must not refuse a small but well-resolved curve"
    )


def test_neff_does_not_launder_a_nan_series_into_the_full_sample_size() -> None:
    """The fourth instance of the round's shape, found while building the third.

    ``max(1.0, min(float(n), n_eff))`` returns ``float(n)`` for ``n_eff =
    NaN``, because every comparison against NaN is false and ``min`` then
    keeps its first argument. A clamp written to BOUND a number decided that
    there WAS one -- and picked the most over-confident value in range. The
    degenerate-fit path above returns NaN residuals, so this is now reachable.
    """
    from liouscope.fitting.neff import estimate_neff_geyer

    assert np.isnan(estimate_neff_geyer(np.full(64, np.nan)))
    # Positive control: a finite, genuinely uncorrelated series is unaffected.
    rng = np.random.default_rng(1)
    n_eff = estimate_neff_geyer(rng.normal(size=64))
    assert np.isfinite(n_eff) and 1.0 <= n_eff <= 64.0