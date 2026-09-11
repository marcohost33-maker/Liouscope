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

import warnings

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


def test_a_nan_tp_rtol_cannot_certify_a_non_generator() -> None:
    """The reviewer's example verbatim, un-parametrised.

    The parametrised twin above covers both APIs and all three bad values; it
    is kept for breadth. This single case exists because the mutation proof
    classifies a parametrised ``pytest.raises`` death as ART-UNBESTIMMT --
    every failure arrives as ``Failed: DID NOT RAISE``, which the tool accepts
    for a single test but not in its multi-line aggregation. That is a
    property of the measuring tool, not of the guard; a case it can read makes
    the evidence checkable without weakening anything.
    """
    with pytest.raises(ValueError, match="tp_rtol"):
        certified_eigvals(_NOT_TRACE_PRESERVING, tp_rtol=float("nan"))


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


# ---------------------------------------------------------------------------
# Finding 1: the sparse builder inherited none of the dense overflow repair
# ---------------------------------------------------------------------------

# The round-18 repair divided each diagonal entry BEFORE summing, in the dense
# builder only. The sparse twin kept ``diagonal().sum()``, which overflows to
# inf for a finite H with large entries; the gauge-fixed matrix becomes NaN,
# the scale becomes NaN, and ``defect > EPS * NaN`` is False -- so a matrix
# with a Hermiticity defect of 1e290 was accepted by one builder and rejected
# by the other, on the same input.
_OVERFLOWING_H = np.array([[1.3e308, 1.0e290], [0.0, 1.3e308]], dtype=complex)


def test_sparse_builder_rejects_what_the_dense_builder_rejects() -> None:
    import scipy.sparse as sp

    from liouscope.sparse.build import build_sparse_liouvillian

    assert np.all(np.isfinite(_OVERFLOWING_H)), "fixture H must be finite"
    assert float(np.max(np.abs(_OVERFLOWING_H - _OVERFLOWING_H.conj().T))) == 1.0e290
    # Warnings are suppressed DELIBERATELY, and the reason is a measurement:
    # the repo runs pytest with ``filterwarnings = ["error"]``, and the
    # unrepaired arithmetic emits ``RuntimeWarning: overflow encountered in
    # reduce`` on this input. Without the suppression the test dies of that
    # warning before it reaches its own judgement -- red, but by crash, so it
    # could not distinguish "the gate refused" from "the gate accepted".
    # Measured in the mutation proof: the un-suppressed form scored
    # "ROT DURCH ABSTURZ -- kein Beleg". What the test asserts is the REFUSAL;
    # the absence of the warning is asserted separately below, where it is the
    # subject rather than a side effect.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"Hermitian|Hermiticity"):
            build_liouvillian(_OVERFLOWING_H, [], [])
        with pytest.raises(ValueError, match=r"Hermitian|Hermiticity"):
            build_sparse_liouvillian(sp.csr_matrix(_OVERFLOWING_H), [], [])


def test_the_repaired_sparse_gate_does_not_overflow_on_the_way() -> None:
    """The repair is arithmetic, so the absence of the overflow is testable.

    Refusing for the right reason and refusing after an overflow are different
    states, and only the first is the fix -- the finiteness guard alone would
    still refuse, one step too late and for a reason nobody chose.

    The warnings are RECORDED and then asserted on, rather than left to
    ``filterwarnings = ["error"]`` to convert into a failure. Both forms go red
    when the overflow returns, but only this one goes red by the test's own
    judgement: a test whose death is a raised warning cannot be told apart from
    a test that crashed, which is the same conflation this whole file is about.
    Measured: the ``simplefilter("error")`` form scored "ROT DURCH ABSTURZ --
    kein Beleg" in the mutation proof although it detected the defect
    perfectly well.
    """
    import scipy.sparse as sp

    from liouscope.sparse.build import build_sparse_liouvillian

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match=r"Hermitian|Hermiticity"):
            build_sparse_liouvillian(sp.csr_matrix(_OVERFLOWING_H), [], [])
    overflows = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not overflows, (
        "the sparse gate refused, but only after overflowing on the way: "
        f"{[str(w.message) for w in overflows]}"
    )


def test_sparse_and_dense_builders_agree_on_valid_and_invalid_input() -> None:
    """POSITIVE CONTROL: parity in BOTH directions, not just refusal."""
    import scipy.sparse as sp

    from liouscope.sparse.build import build_sparse_liouvillian

    hermitian = np.array([[1.0, 0.5j], [-0.5j, 2.0]], dtype=complex)
    dense = build_liouvillian(hermitian, [_LOWER], [0.5])
    sparse = build_sparse_liouvillian(
        sp.csr_matrix(hermitian), [sp.csr_matrix(_LOWER)], [0.5]
    )
    assert np.allclose(dense, sparse.toarray())

    # A real identity offset is physically inert and must still be accepted by
    # both -- the overflow repair must not have turned the gauge shift into a
    # rejection criterion.
    shifted = hermitian + 1.0e12 * np.eye(2, dtype=complex)
    assert build_liouvillian(shifted, [], []).shape == (4, 4)
    assert build_sparse_liouvillian(sp.csr_matrix(shifted), [], []).shape == (4, 4)

    # ... and an ordinary non-Hermitian H must still be refused by both.
    not_hermitian = np.array([[1.0, 1.0], [0.0, 2.0]], dtype=complex)
    with pytest.raises(ValueError):
        build_liouvillian(not_hermitian, [], [])
    with pytest.raises(ValueError):
        build_sparse_liouvillian(sp.csr_matrix(not_hermitian), [], [])


# ---------------------------------------------------------------------------
# Finding 2: the jackknife never asked whether its fits converged
# ---------------------------------------------------------------------------


def _noisy_decay() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 5.0, 40)
    rng = np.random.default_rng(11)
    y = np.exp(-1.3 * t) + rng.normal(0.0, 0.02, size=t.size)
    return t, y, np.array([1.0, 1.2])


def test_jackknife_refuses_when_a_leave_one_out_fit_fails(monkeypatch) -> None:
    """THE regression: the round-20 bootstrap guard does not cover this path.

    Injected at the documented contract surface (``success is False``) rather
    than by hunting for data that makes ``least_squares`` give up, so the test
    depends on the rule and not on SciPy's convergence heuristics.
    """
    from dataclasses import replace

    from liouscope.fitting import bootstrap as bs
    from liouscope.fitting.models import M0

    t, y, p0 = _noisy_decay()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, theta_hat = bs.parametric_bootstrap(M0, t, y, p0, B=25)
    assert samples.shape == (25, 2), (
        "every bootstrap replicate must converge here -- the point of the "
        "finding is that the bootstrap guard passes while the jackknife fails"
    )

    real_fit = bs.fit_gls_ar1
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        out = real_fit(*args, **kwargs)
        calls["n"] += 1
        return replace(out, success=False) if calls["n"] == 4 else out

    monkeypatch.setattr(bs, "fit_gls_ar1", flaky)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="did not converge"):
            bs._jackknife(M0, t, y, theta_hat, None)


def test_a_failed_jackknife_estimate_moves_the_bca_interval() -> None:
    """Why refusing is not pedantry: the endpoints move with no data behind it.

    A failed ``least_squares`` returns its unchanged starting value, which is
    ``theta_hat`` itself, so the jackknife distribution is pulled towards its
    own centre and the acceleration -- a third moment of exactly these
    estimates -- shifts. Measured here for ONE failure out of 40: the
    amplitude interval narrows to 0.923x of its width. The direction is
    parameter dependent (the rate interval moves by 0.2 %), so the assertion
    is on the SHIFT, not on a claimed direction.
    """
    from liouscope.fitting.bootstrap import _jackknife, bca_ci, parametric_bootstrap
    from liouscope.fitting.models import M0

    t, y, p0 = _noisy_decay()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, theta_hat = parametric_bootstrap(M0, t, y, p0, B=200)
        clean = _jackknife(M0, t, y, theta_hat, None)

    corrupted = clean.copy()
    corrupted[0] = theta_hat  # what a non-converged leave-one-out fit deposits
    ci_clean = bca_ci(samples, theta_hat, jackknife_estimates=clean)
    ci_corrupt = bca_ci(samples, theta_hat, jackknife_estimates=corrupted)

    w_clean = ci_clean[:, 1] - ci_clean[:, 0]
    w_corrupt = ci_corrupt[:, 1] - ci_corrupt[:, 0]
    ratio = w_corrupt / w_clean
    assert np.all(np.isfinite(ci_corrupt)), (
        "the corrupted interval is FINITE -- that is what made it dangerous"
    )
    assert np.min(ratio) < 0.95, (
        f"width ratios {ratio.tolist()}; a single non-fit is expected to move "
        "at least one endpoint pair by more than 5 %"
    )


def test_jackknife_still_returns_estimates_when_every_fit_converges() -> None:
    """POSITIVE CONTROL: the refusal must not have become unconditional."""
    from liouscope.fitting.bootstrap import _jackknife, bca_ci, parametric_bootstrap
    from liouscope.fitting.models import M0

    t, y, p0 = _noisy_decay()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        samples, theta_hat = parametric_bootstrap(M0, t, y, p0, B=40)
        jk = _jackknife(M0, t, y, theta_hat, None)

    assert jk.shape == (t.size, theta_hat.size)
    ci = bca_ci(samples, theta_hat, jackknife_estimates=jk)
    assert np.all(np.isfinite(ci))
    assert np.all(ci[:, 1] > ci[:, 0]), "a real jackknife gives a real interval"


# ---------------------------------------------------------------------------
# Finding 5: the unresolved run was the one whose report would not serialise
# ---------------------------------------------------------------------------


def _damped_qubit() -> tuple[np.ndarray, np.ndarray]:
    from liouscope import steady_state

    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.6])
    return L, steady_state(L)


def _diagnosed():
    from liouscope import diagnose

    L, rho = _damped_qubit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = diagnose(L, rho_steady_state=rho, bootstrap_B=30, include_mpemba=False)
    return report, L, rho


def _payload(report, L, rho):
    from liouscope.io.stability_report import build_stability_report

    return build_stability_report(
        report,
        L_super=L,
        rho_ss=rho,
        claim_level="BLOCK",
        direction="slowdown",
        cp_evidence_level="construction_only",
    )


def test_a_withheld_diagnostic_does_not_break_the_report(tmp_path) -> None:
    """THE regression: D1/D3/D9 are NaN by design when a certificate is
    unresolved, and ``allow_nan=False`` then raised on exactly that run."""
    from dataclasses import replace

    from liouscope.io.stability_report import (
        UNAVAILABLE_CLAIM_STATUS,
        dump_stability_report,
        validate_stability_report,
    )

    report, L, rho = _diagnosed()
    unresolved = replace(
        report,
        spectral=replace(
            report.spectral, gap=float("nan"), oscillating_gap=float("nan")
        ),
        nonnorm=replace(report.nonnorm, petermann_max=float("nan")),
    )
    payload = _payload(unresolved, L, rho)
    for key in ("D1_gap", "D3_oscillating_gap", "D9_petermann_max"):
        entry = payload["diagnostics"][key]
        assert isinstance(entry, dict), (key, entry)
        assert entry["value"] is None
        assert entry["claim_status"] == UNAVAILABLE_CLAIM_STATUS
        assert entry["__nonfinite__"] == "nan"
    validate_stability_report(payload)
    out = tmp_path / "unresolved.json"
    dump_stability_report(payload, out)
    import json

    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["diagnostics"]["D1_gap"]["value"] is None
    assert loaded["diagnostics"]["D1_gap"]["__nonfinite__"] == "nan"


def test_an_infinite_diagnostic_stays_distinguishable_from_a_withheld_one(
    tmp_path,
) -> None:
    """+-inf is equally non-conforming under RFC 8259 and raised in the same
    place, so the encoding must not stop at NaN -- but it must not FLATTEN the
    two either.

    This assertion was ``is None`` until the PR #121 and PR #127 encodings were
    unified. Both branches kept the report writable; only the tagged one keeps
    a measured infinity (a Kreiss constant that really did diverge) apart from
    an unavailable one (a certificate that was never resolved). Once both are
    ``null`` on disk, no later reader can tell them apart again -- which is the
    conflation the whole withholding mechanism exists to prevent.
    """
    from dataclasses import replace

    from liouscope.io.stability_report import (
        UNAVAILABLE_CLAIM_STATUS,
        dump_stability_report,
    )

    report, L, rho = _diagnosed()
    infinite = replace(report, nonnorm=replace(report.nonnorm, kreiss=float("inf")))
    payload = _payload(infinite, L, rho)
    entry = payload["diagnostics"]["D10_kreiss"]
    assert entry["value"] is None
    assert entry["claim_status"] == UNAVAILABLE_CLAIM_STATUS
    assert entry["__nonfinite__"] == "inf"

    # DISCRIMINATION: the same slot, withheld rather than measured, must carry
    # a DIFFERENT token -- otherwise this encoding is a renamed ``null``.
    withheld = replace(report, nonnorm=replace(report.nonnorm, kreiss=float("nan")))
    assert _payload(withheld, L, rho)["diagnostics"]["D10_kreiss"][
        "__nonfinite__"
    ] == "nan"

    dump_stability_report(payload, tmp_path / "infinite.json")


def test_a_resolved_run_still_carries_its_numbers(tmp_path) -> None:
    """POSITIVE CONTROL: the projection must not blank out real measurements."""
    from liouscope.io.stability_report import dump_stability_report

    report, L, rho = _diagnosed()
    payload = _payload(report, L, rho)
    for key in (
        "D1_gap",
        "D2_gns_gap",
        "D2b_kms_gap",
        "D3_oscillating_gap",
        "D8_henrici",
        "D9_petermann_max",
        "D10_kreiss",
    ):
        value = payload["diagnostics"][key]
        assert isinstance(value, float) and np.isfinite(value), (
            f"{key} came back as {value!r} for a perfectly ordinary run"
        )
    assert payload["diagnostics"]["D1_gap"] == pytest.approx(report.spectral.gap)
    dump_stability_report(payload, tmp_path / "resolved.json")


# ---------------------------------------------------------------------------
# Finding 8: a missing gap was read as the gapless limit
# ---------------------------------------------------------------------------

_F5_EVIDENCE = {
    "pseudospectral_radius": 12.0,
    "henrici_eta": 3.0,
    "gns_gap": 0.1,
    "kreiss": 1.0,
    "petermann_max": 1.0,
}


def _classify(ev: dict[str, float]):
    from liouscope.diagnostics.classification import (
        _pick_a_class,
        hypothesis_evidence_matrix,
    )

    report, _, _ = _diagnosed()
    picked = _pick_a_class(ev, relaxation=report.relaxation)
    matrix = hypothesis_evidence_matrix(ev, relaxation=report.relaxation)
    f5 = next(e for e in matrix if e["a_class"] == "A10")
    return picked, f5["status"]


def test_an_unmeasured_gap_does_not_support_the_phantom_hypothesis() -> None:
    """THE regression: NaN D1 must be absence of evidence, not evidence."""
    picked, status = _classify({**_F5_EVIDENCE, "gap": float("nan")})
    assert picked != ("A10", "F5"), (
        "the classifier claimed the phantom-relaxation mechanism although its "
        "dimensionless reach could not be computed at all"
    )
    assert status == "UNEVALUABLE"


def test_an_absent_gap_key_behaves_the_same_as_a_nan_one() -> None:
    """The NaN encoding and the missing key must not decide differently."""
    assert _classify({**_F5_EVIDENCE, "gap": float("nan")}) == _classify(
        dict(_F5_EVIDENCE)
    )


def test_a_measured_zero_gap_still_fires_the_gapless_reach_leg() -> None:
    """POSITIVE CONTROL, and the over-correction control in one.

    A gap that was MEASURED as 0.0 is the documented gapless limit and must
    still count as reach evidence -- the repair distinguishes "no gap" from
    "no measurement", it does not abolish the gapless branch.
    """
    picked, status = _classify({**_F5_EVIDENCE, "gap": 0.0})
    assert picked == ("A10", "F5")
    assert status == "SUPPORTED"


def test_a_measured_positive_gap_still_decides_on_the_reach_ratio() -> None:
    """POSITIVE CONTROL: the ordinary quantitative branch is untouched."""
    fires, _ = _classify({**_F5_EVIDENCE, "gap": 1.0, "gap_to_gns_ratio": 1.0})
    assert fires == ("A10", "F5"), "radius/gap = 12 > 2 must still fire"
    quiet, status = _classify({**_F5_EVIDENCE, "gap": 12.0, "gap_to_gns_ratio": 1.0})
    assert quiet != ("A10", "F5"), "radius/gap = 1 must not fire"
    assert status == "NOT_SUPPORTED"
