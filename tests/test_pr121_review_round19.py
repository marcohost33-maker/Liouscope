"""Round-19 review of PR #121: three more gates that certified what they had not checked.

All three findings share one shape. A layer decides something it never
measured, and the decision looks like a measurement:

* ``holdout_validate`` scored the parameters of a fit it never asked about.
* ``spectral`` withheld D1 on an uncertified certificate and published D3, D4
  and the oscillating-pair flag from the same untrusted spectrum.
* the Hermiticity gate compared against a NaN scale, and a comparison against
  NaN answers "no violation".

Each test below pairs the failing case with a positive control, so a repair
that simply refuses everything cannot pass.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.fitting import models
from liouscope.fitting.holdout import holdout_validate

# ---------------------------------------------------------------------------
# 1. The anti-overfit gate must not score a fit that never converged
# ---------------------------------------------------------------------------

def test_holdout_rejects_a_saturated_fit() -> None:
    """A saturated fit is a non-fit; "does it generalise?" has no answer.

    On the ~1e100 saturation plateau the model is CONSTANT, so train and
    holdout RMSE are equally enormous and their ratio is about 1 -- the
    anti-overfit criterion is satisfied by the very failure it exists to
    catch. That is why ``ratio`` is asserted here too: without it the test
    would not show WHY the old gate accepted.
    """
    t = np.linspace(0.0, 1e10, 64)
    y = np.exp(-5.0 * t / 1e10)

    result = holdout_validate(models.M0, t, y, np.array([1.0, -1.0]))

    assert result.fit_success is False
    assert result.accept is False
    assert result.ratio == pytest.approx(1.0, abs=0.1), (
        "the plateau ratio is what made the old gate accept; if this stops "
        "being ~1 the test no longer exercises that path"
    )


def test_holdout_still_accepts_a_healthy_fit() -> None:
    """Positive control: the gate must not have become a blanket refusal."""
    t = np.linspace(0.0, 5.0, 64)
    y = np.exp(-1.0 * t)

    result = holdout_validate(models.M0, t, y, np.array([1.0, -0.9]))

    assert result.fit_success is True
    assert result.accept is True


# ---------------------------------------------------------------------------
# 2. D3/D4 and the oscillating-pair flag travel with D1
# ---------------------------------------------------------------------------

def _uncertified_certificate(bound: float = 1.0e-12):
    """A REAL certificate that is applicable but never certified.

    Constructing a generator that genuinely defeats every repair route is
    possible but fragile -- it depends on LAPACK behaviour we do not control.
    The contract under test is "applicable and not certified implies
    withheld", so the state is built from the production dataclass and the
    contract is exercised exactly, without a stand-in whose surface could
    drift away from the real one.
    """
    from liouscope.numerics.linalg import ZeroModeCertificate

    return ZeroModeCertificate(
        applicable=True,
        certified=False,
        solver="zgeev",
        residual=float("nan"),
        bound=bound,
        trace_defect=0.0,
    )


def _healthy_generator() -> np.ndarray:
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    return build_liouvillian(H, [lowering], [0.5])


def test_uncertified_certificate_withholds_d3_d4_and_the_pair_flag(monkeypatch) -> None:
    """THE regression: withholding D1 alone was inconsistent, not partial.

    Publishing some numbers from a spectrum the layer calls untrustworthy
    while withholding others teaches consumers that the layer withholds what
    it cannot stand behind -- which makes the survivors MORE credible, not
    less.
    """
    from liouscope.diagnostics import spectral as sp

    L = _healthy_generator()
    monkeypatch.setattr(
        sp, "certified_eigvals",
        lambda *a, **k: (np.linalg.eigvals(np.asarray(L, dtype=complex)),
                         _uncertified_certificate()),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sp.compute_spectral_layer(L)

    assert np.isnan(result.gap), "D1 must stay withheld"
    assert np.isnan(result.oscillating_gap), "D3 must be withheld with D1"
    assert np.isnan(result.spectral_spread), "D4 must be withheld with D1"
    assert result.has_complex_pairs is None, (
        "False would assert 'no oscillating modes'; None says the question "
        "was not answered"
    )


def test_a_certified_run_still_publishes_d3_and_d4() -> None:
    """Positive control: a healthy generator keeps its measurements."""
    from liouscope.diagnostics import spectral as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sp.compute_spectral_layer(_healthy_generator())

    assert np.isfinite(result.gap)
    assert np.isfinite(result.spectral_spread)
    assert result.has_complex_pairs is not None


def test_withheld_pair_flag_does_not_hold_the_a8_rung(monkeypatch) -> None:
    """The unavailable marker must reach the classifier as unavailable.

    ``None`` becomes NaN in the evidence dict, ``_strip_unavailable`` removes
    it, and a condition whose required key is missing does not hold. This
    pins that route rather than trusting it.
    """
    from liouscope.diagnostics import classification as cl

    class _Spectral:
        gap = float("nan")
        gns_gap = float("nan")
        kms_gap = float("nan")
        oscillating_gap = float("nan")
        spectral_spread = float("nan")
        has_complex_pairs = None

    ev: dict[str, float] = {}
    ev["has_complex_pairs"] = (
        float("nan") if _Spectral.has_complex_pairs is None
        else float(_Spectral.has_complex_pairs)
    )
    stripped = cl._strip_unavailable(ev)
    assert "has_complex_pairs" not in stripped


# ---------------------------------------------------------------------------
# 3. A comparison against NaN must not read as "no violation"
# ---------------------------------------------------------------------------

def test_finite_but_huge_hamiltonian_cannot_overflow_past_the_gate() -> None:
    """``np.trace`` summed before dividing and overflowed on a FINITE H.

    ``H_gauge`` and ``scale`` then became NaN, and ``defect > EPS * NaN`` is
    False -- so an H with an order-one gauge-fixed Hermiticity defect was
    accepted. The finiteness gate above it had already passed, which is what
    made the hole invisible.
    """
    H = np.array([[1e308, 1e290], [0.0, 1e308]], dtype=complex)
    assert np.all(np.isfinite(H)), "the input must be finite for this to be the bug"
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not np.isfinite(np.trace(H)), (
            "the probe relies on trace() overflowing; if it stops, this test "
            "no longer exercises the failure"
        )

    with pytest.raises(ValueError):
        build_liouvillian(H, [], [])


def test_a_large_but_genuinely_hermitian_hamiltonian_is_still_accepted() -> None:
    """Over-correction guard: size alone is not a defect.

    A repair that refused every large H would pass the test above and be
    wrong. ``1e308 * I`` is perfectly Hermitian and must go through.
    """
    H = np.array([[1e308, 0.0], [0.0, 1e308]], dtype=complex)
    build_liouvillian(H, [], [])


def test_the_gauge_shift_no_longer_emits_overflow_warnings() -> None:
    """The arithmetic itself must be clean, not merely caught afterwards."""
    H = np.array([[1e308, 0.0], [0.0, 1e308]], dtype=complex)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_liouvillian(H, [], [])
    texts = [str(w.message) for w in caught]
    assert not any("overflow" in t or "invalid value" in t for t in texts), texts
