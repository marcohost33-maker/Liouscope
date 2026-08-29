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

    Finite 4x4, cancelling ``+-1e308`` in the ``vec(I)^H L`` combination, so
    the absolute trace defect is ``sqrt(17) = 4.12`` while ``||L||_F``
    overflows to ``inf``. Eigenvalues are exactly ``{1, 2, 3, 4}``: column 0
    and column 2 each carry a single diagonal entry, and the remaining 2x2 on
    indices ``{1, 3}`` is lower triangular.
    """
    L = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    L[0, 1] = 1.0e308
    L[3, 1] = -1.0e308
    return L


def _nilpotent_wide_band_operator() -> np.ndarray:
    """The same failure WITHOUT any overflow, to show the class is general.

    Strictly upper triangular with ``1e10`` entries confined to row 1, so
    ``vec(I)^H L = 0`` EXACTLY (trace preserving, defect 0.0), ``||L||_F`` is
    a perfectly ordinary ``1.4e10``, and every eigenvalue is exactly zero.
    The band ``rtol * eps * ||L||_2`` is then ``3.1e-3`` and swallows all four
    -- a nilpotent generator certified as a four-dimensional stationary
    manifold, on finite, well-scaled arithmetic.
    """
    L = np.zeros((4, 4), dtype=complex)
    L[1, 2] = 1.0e10
    L[1, 3] = 1.0e10
    return L


def test_the_overflowing_operator_is_the_state_the_reviewer_described() -> None:
    """Pins the preconditions, so a numpy change cannot hollow out the test."""
    from liouscope.numerics.linalg import trace_preservation_defect

    L = _overflowing_norm_operator()
    assert np.all(np.isfinite(L)), "the input must be FINITE for this to be the bug"
    with np.errstate(over="ignore"):
        defect, fro = trace_preservation_defect(L)
    assert defect == pytest.approx(np.sqrt(17.0)), "a real, order-one trace defect"
    assert not np.isfinite(fro), (
        "the whole failure is that ``defect > tp_rtol * fro`` compares "
        "against infinity; if ||L||_F stops overflowing the test is hollow"
    )
    assert np.allclose(np.sort(np.linalg.eigvals(L).real), [1.0, 2.0, 3.0, 4.0])


@pytest.mark.parametrize(
    "make", [_overflowing_norm_operator, _nilpotent_wide_band_operator]
)
def test_a_band_that_swallows_the_spectrum_does_not_certify(make) -> None:
    """THE regression, on both ladders.

    Before this guard the certificate read ``certified=True, resolved=True``
    with ``zero_mode_count = 4`` -- every eigenvalue declared stationary,
    which for a d=2 system asserts that the entire operator space is a steady
    state. ``resolved=True`` is the damaging half: it tells the spectral layer
    that D1/D3/D4 may be published.
    """
    from liouscope.numerics.linalg import certified_eig, certified_eigvals

    L = make()
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, cert_vals = certified_eigvals(L)
        _, cert_vec = certified_eig(L)

    for cert in (cert_vals, cert_vec):
        assert cert.applicable is True, (
            "the operator is trace preserving RELATIVE to its own scale, "
            "which is the criterion this module chose; the repair is not a "
            "trace-preservation verdict"
        )
        assert cert.certified is False
        assert cert.resolved is False
        assert cert.zero_mode_count == 0


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
