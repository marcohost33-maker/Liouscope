"""Round-22 review of PR #121: eight gates that decided from what they never had.

The findings repeat two shapes the earlier rounds already met, one variable or
one code path further along each time:

* **a comparison against a number that is not one** -- ``tp_rtol = NaN``/``inf``
  makes the trace-preservation test vacuous, exactly as the non-finite
  reference scale did in round 21 (findings 4);
* **a default published as a measurement** -- the inapplicable certificate kept
  ``zero_mode_count = 1``, the sparse builder inherited the dense builder's
  overflow, a geometric mean underflowed to 0, a spectral radius overflowed to
  inf, a jackknife fit failed unnoticed, and a missing gap was read as the
  gapless limit (findings 1, 2, 3, 6, 7, 8);
* **a report that cannot be written precisely when it matters** -- the
  unresolved run is the one whose stability report fails to serialise
  (finding 5).

Every test is paired with a positive control: a guard that refuses everything
discriminates nothing and must not be able to pass this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.numerics.linalg import certified_eig, certified_eigvals

# A genuine GKSL generator, used throughout as the over-correction control:
# whatever a new guard refuses, it must NOT refuse this.
_H = np.diag([0.0, 1.0]).astype(complex)
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _healthy_generator() -> np.ndarray:
    return build_liouvillian(_H, [_LOWER], [0.5])


# ---------------------------------------------------------------------------
# Finding 4: a non-finite tolerance certifies a non-trace-preserving operator
# ---------------------------------------------------------------------------

# vec(I)^H L = (0, 0, 0, -3) for this operator, so the trace-preservation
# defect is exactly 3.0 against ||L||_F = sqrt(14). It is not a generator by
# any reading -- yet ``3.0 > nan * sqrt(14)`` is False, and so is every other
# comparison against NaN, so the applicability gate waved it through.
_NOT_TRACE_PRESERVING = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)


@pytest.mark.parametrize("api", [certified_eigvals, certified_eig])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0e-10])
def test_non_finite_tp_rtol_is_refused_not_honoured(api, bad: float) -> None:
    """A tolerance that is NaN/inf/negative cannot decide trace preservation."""
    with pytest.raises(ValueError, match="tp_rtol"):
        api(_NOT_TRACE_PRESERVING, tp_rtol=bad)


@pytest.mark.parametrize("api", [certified_eigvals, certified_eig])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_non_finite_rtol_is_refused_too(api, bad: float) -> None:
    """Same class, the sibling parameter: the band multiplier is validated."""
    with pytest.raises(ValueError, match="rtol"):
        api(_healthy_generator(), rtol=bad)


@pytest.mark.parametrize("api", [certified_eigvals, certified_eig])
def test_valid_tolerances_still_certify_a_real_generator(api) -> None:
    """POSITIVE CONTROL: the guard must reject tolerances, not generators."""
    _, cert = api(_healthy_generator(), tp_rtol=1.0e-10)
    assert cert.applicable and cert.certified
    _, cert_wide = api(_healthy_generator(), tp_rtol=1.0)
    assert cert_wide.applicable and cert_wide.certified
    _, cert_zero = api(_healthy_generator(), tp_rtol=0.0)
    assert cert_zero.applicable, "tp_rtol = 0 is a legal (strictest) tolerance"


# ---------------------------------------------------------------------------
# Finding 6: an inapplicable certificate must not report a zero mode
# ---------------------------------------------------------------------------

# diag(1, 2, 3, 4): a spectrum containing no zero at all, and not trace
# preserving, so the certificate cannot apply. ``zero_mode_count`` nonetheless
# kept the dataclass default of 1 -- a counted stationary mode that was never
# counted, persisted into the run report as audit metadata.
_NO_ZERO_MODE = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)


@pytest.mark.parametrize("api", [certified_eigvals, certified_eig])
def test_inapplicable_certificate_counts_no_zero_modes(api) -> None:
    _, cert = api(_NO_ZERO_MODE)
    assert cert.applicable is False
    assert cert.zero_mode_count == 0, (
        f"certificate reports {cert.zero_mode_count} zero mode(s) for a "
        "spectrum with no zero eigenvalue and no applicable certificate"
    )
    assert cert.as_dict()["zero_mode_count"] == 0


@pytest.mark.parametrize("api", [certified_eigvals, certified_eig])
def test_applicable_certificate_still_counts_its_zero_mode(api) -> None:
    """POSITIVE CONTROL: the real stationary mode is still counted as one."""
    _, cert = api(_healthy_generator())
    assert cert.applicable and cert.certified
    assert cert.zero_mode_count == 1


# ---------------------------------------------------------------------------
# Finding 3: the geometric mean of two representable numbers underflowed to 0
# ---------------------------------------------------------------------------

# The refinement places the applied tolerance at the geometric mean of the two
# populations it separated. ``sqrt(lo * hi)`` forms the product first, and
# ``lo * hi`` scales as c^2 under a uniform rate rescale ``L -> cL`` while the
# refinement itself is scale free -- so below c ~ 1e-139 the product underflows
# to 0 and the tolerance becomes 0.0, which keeps the numerical stationary
# residual as a physical mode.
#
# Exact eigenvectors are supplied because ``scipy.linalg.eig`` is itself
# unreliable at this magnitude (measured here: it returns 6.72e-151 and
# 6.72e-139 for ``diag(1e-177, 1e-165)``), and this test is about the
# refinement's arithmetic, not about LAPACK's. Passing the vectors is the
# ordinary call path -- ``certified_eig`` uses it.
_IDENTITY_VECTORS = {
    "right_vectors": np.eye(2, dtype=complex),
    "left_vectors": np.eye(2, dtype=complex),
}


@pytest.mark.parametrize(
    ("lo_v", "hi_v", "expected"),
    [
        (1.0e-177, 1.0e-165, 1.0e-171),
        (1.0e-200, 1.0e-190, 1.0e-195),
        (1.0e-180, 1.0e-170, 1.0e-175),
    ],
)
def test_refined_tolerance_survives_a_rate_unit_rescale(
    lo_v: float, hi_v: float, expected: float
) -> None:
    from liouscope.numerics.linalg import refine_zero_band

    assert lo_v * hi_v == 0.0, "fixture must actually underflow the product"
    L = np.diag([lo_v, hi_v]).astype(complex)
    ev = np.array([lo_v, hi_v], dtype=complex)
    in_band, zero_tolerance = refine_zero_band(
        L, ev, np.abs(ev), 1.0e-150, **_IDENTITY_VECTORS
    )
    assert in_band.tolist() == [True, False], (
        "fixture must exercise the refinement: the slow mode has to leave the "
        "band and the stationary residual has to stay in it"
    )
    assert zero_tolerance > 0.0, (
        f"zero tolerance is {zero_tolerance}; a 0.0 cutoff admits the "
        "numerical stationary residual as a physical mode"
    )
    assert zero_tolerance == pytest.approx(expected, rel=1.0e-12)
    assert lo_v < zero_tolerance < hi_v, "the cutoff must separate the two"


def test_refined_tolerance_unchanged_at_ordinary_rate_units() -> None:
    """POSITIVE CONTROL: the healthy scale keeps the value it always had."""
    from liouscope.numerics.linalg import refine_zero_band

    lo_v, hi_v = 1.0e-16, 1.0e-15
    L = np.diag([lo_v, hi_v]).astype(complex)
    ev = np.array([lo_v, hi_v], dtype=complex)
    _, zero_tolerance = refine_zero_band(L, ev, np.abs(ev), 1.0e-13)
    assert zero_tolerance == pytest.approx(np.sqrt(lo_v * hi_v), rel=1.0e-12)


def test_refined_tolerance_is_rescale_equivariant() -> None:
    """The refinement is scale free, so the cutoff must scale exactly with c."""
    from liouscope.numerics.linalg import refine_zero_band

    lo_v, hi_v = 1.0e-16, 1.0e-15
    ref = refine_zero_band(
        np.diag([lo_v, hi_v]).astype(complex),
        np.array([lo_v, hi_v], dtype=complex),
        np.abs(np.array([lo_v, hi_v])),
        1.0e-13,
        **_IDENTITY_VECTORS,
    )[1]
    for c in (1.0e-50, 1.0e-100, 1.0e-150):
        scaled = refine_zero_band(
            np.diag([lo_v * c, hi_v * c]).astype(complex),
            np.array([lo_v * c, hi_v * c], dtype=complex),
            np.abs(np.array([lo_v * c, hi_v * c])),
            1.0e-13 * c,
            **_IDENTITY_VECTORS,
        )[1]
        assert scaled == pytest.approx(ref * c, rel=1.0e-12), (
            f"tolerance is not equivariant under L -> {c:.0e} L: "
            f"{scaled:.6e} instead of {ref * c:.6e}"
        )


# ---------------------------------------------------------------------------
# Finding 7: an overflowing spectral radius turned a huge mode into "gapless"
# ---------------------------------------------------------------------------

# |1.3e308 + 1.3e308j| is about 1.84e308 and does not fit in a double, so
# ``np.abs`` returns inf although every COMPONENT is finite and passes the
# finiteness gate. The tolerance became inf, every strict |lambda| > tol test
# was False, and D1/D3/D4 all reported 0.0 for a spectrum with a huge non-zero
# mode.
_OVERFLOWING_SPECTRUM = np.array(
    [
        0.0 + 0.0j,
        -1.3e308 - 1.3e308j,
        -1.3e308 + 1.3e308j,
        -5.0e307 + 0.0j,
    ],
    dtype=complex,
)


def test_overflowing_modulus_does_not_yield_an_infinite_tolerance() -> None:
    from liouscope.numerics.scale import spectral_zero_tolerance

    assert np.isinf(np.abs(_OVERFLOWING_SPECTRUM)).any(), (
        "fixture must actually overflow the modulus"
    )
    assert np.all(np.isfinite(_OVERFLOWING_SPECTRUM.real))
    assert np.all(np.isfinite(_OVERFLOWING_SPECTRUM.imag))
    tol = spectral_zero_tolerance(_OVERFLOWING_SPECTRUM)
    assert np.isfinite(tol) and tol > 0.0
    # ~ rtol * eps * 1.84e308; must stay far below the modes it separates.
    assert tol < 5.0e307


def test_overflowing_modulus_is_not_reported_as_gapless() -> None:
    from liouscope.diagnostics.spectral import (
        liouvillian_gap,
        oscillating_mode_gap,
        spectral_spread,
    )

    assert liouvillian_gap(_OVERFLOWING_SPECTRUM) == pytest.approx(5.0e307)
    assert oscillating_mode_gap(_OVERFLOWING_SPECTRUM) == pytest.approx(1.3e308)
    assert spectral_spread(_OVERFLOWING_SPECTRUM) == pytest.approx(8.0e307)


def test_ordinary_and_extreme_valid_spectra_are_untouched() -> None:
    """POSITIVE CONTROL + over-correction control.

    The repair may not move the finite path by a single bit, and a spectrum
    that is merely large -- but representable -- must not be refused.
    """
    from liouscope._consts import ZERO_MODE_EPS_FACTOR
    from liouscope.numerics.scale import spectral_zero_tolerance

    eps = float(np.finfo(float).eps)
    for spectrum in (
        np.array([0.0, -1.0, -2.0 + 3.0j]),
        np.array([0.0, -1.0e308]),
        np.array([0.0, -1.0e-300]),
        np.array([0.0 + 0.0j]),
    ):
        expected = ZERO_MODE_EPS_FACTOR * eps * float(np.max(np.abs(spectrum)))
        assert spectral_zero_tolerance(spectrum) == expected


def test_a_tolerance_that_cannot_be_represented_is_refused() -> None:
    """Fail-closed second line of defence: no silent infinite threshold."""
    from liouscope.numerics.scale import spectral_zero_tolerance

    with pytest.raises(ValueError, match="not finite"):
        spectral_zero_tolerance(_OVERFLOWING_SPECTRUM, rtol=1.0e300)
