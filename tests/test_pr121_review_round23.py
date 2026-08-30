"""Round-23 review of PR #121: two measurements taken from what was never there.

Both findings have the shape the earlier rounds keep producing -- a gate that
certifies something it did not measure -- one layer further out each time:

* **a reference scale lost to underflow** (finding 12). ``trace_preservation_
  defect`` squares its entries before summing, so a generator in small enough
  rate units returns ``(0.0, 0.0)``; the applicability test ``0 > tp_rtol *
  tiny`` is then False for EVERY operator and a non-trace-preserving one
  becomes certificate-applicable. The same operator fifty decades larger is
  correctly refused, so a pure change of rate unit decided whether the object
  is a legal generator.

* **a diagnostic read off a spectrum the layer already refused to stand behind**
  (finding 13). Where the zero-mode certificate is applicable but unresolved,
  D1/D3/D4 are withheld as NaN -- and D16 was still published from that very
  spectrum. Measured before the fix: ``lep_proximity = 0.0`` with 51 candidate
  pairs. ``0.0`` is not a neutral number but the STRONGEST available
  exceptional-point signal ("coalesced"), manufactured out of slow modes that
  sit below the eigensolver's backward error.

Every guard here is paired with a positive control: a guard that refuses
everything discriminates nothing and must not be able to pass this file. The
overflow twin of finding 12 has its own test for the opposite reason -- it
pins a refusal that must NOT be repaired, because the round-21 guard depends
on it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope.diagnostics.lep import compute_lep_layer
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.numerics.linalg import (
    certified_eig,
    certified_eigvals,
    trace_preservation_defect,
)

_H = np.diag([0.0, 1.0]).astype(complex)
_LOWER = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _healthy_generator() -> np.ndarray:
    """A genuine GKSL generator: the over-correction control throughout."""
    return build_liouvillian(_H, [_LOWER], [0.5])


def _non_tp_operator() -> np.ndarray:
    """The reviewer's operator: a trace defect of 3.0 at unit scale."""
    return np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)


# ---------------------------------------------------------------------------
# Finding 12: use underflow-safe norms for the TP gate.
# ---------------------------------------------------------------------------


def test_the_reference_scale_survives_the_reviewers_rate_unit() -> None:
    """``diag([0, -1e-200, -2e-200, -3e-200])`` must not lose its own scale.

    Measured before the fix: ``(defect, fro) == (0.0, 0.0)``. Two zeros are not
    a small trace defect -- they are the absence of a measurement, and the gate
    downstream cannot tell the difference.
    """
    defect, fro = trace_preservation_defect(_non_tp_operator() * 1e-200)
    assert fro > 0.0, "the Frobenius reference scale underflowed to zero"
    assert defect > 0.0, "the trace defect underflowed to zero"
    # The RATIO is the dimensionless quantity the gate actually uses, and it is
    # what must be unit-invariant: 3.0 / sqrt(14) at every scale.
    assert defect / fro == pytest.approx(3.0 / np.sqrt(14.0), rel=1e-12)


def test_a_non_tp_operator_is_refused_in_every_rate_unit() -> None:
    """The verdict must be a property of the operator, not of its units.

    This is the finding itself: at ``1e-150`` the operator was correctly
    refused and at ``1e-200`` it was certified, from the same physics.
    """
    for scale in (1.0, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300):
        op = _non_tp_operator() * scale
        for name, fn in (
            ("certified_eigvals", certified_eigvals),
            ("certified_eig", certified_eig),
        ):
            cert = fn(op)[-1]
            assert not cert.applicable, (
                f"{name} at scale {scale:g}: a trace defect of 3.0 relative to "
                f"a norm of sqrt(14) is not trace preservation, yet the "
                f"certificate was declared applicable "
                f"(certified={cert.certified})"
            )


def test_a_healthy_generator_in_the_same_rate_units_still_certifies() -> None:
    """POSITIVE CONTROL. The rescue must not become a blanket refusal.

    Without this, a guard that simply refused every small-scale operator would
    pass the two tests above.
    """
    for scale in (1.0, 1e-150, 1e-200):
        gen = _healthy_generator() * scale
        cert = certified_eigvals(gen)[-1]
        assert cert.applicable, f"a GKSL generator at scale {scale:g} was refused"
        assert cert.certified, f"a GKSL generator at scale {scale:g} was not certified"


def test_the_overflow_direction_is_deliberately_not_rescued() -> None:
    """The round-21 refusal must SURVIVE the round-23 repair.

    The two directions look symmetric and are not. ``||L||_F = inf`` for the
    round-20 counterexample (cancelling +-1e308 entries, spectrum {1,2,3,4}) is
    the input to the round-21 guard, which refuses a non-finite reference
    scale. Its true Frobenius norm is about 1.4e308 and therefore perfectly
    representable, so a scaled computation would hand back a finite number,
    readmit the operator and reopen a hole that took five interpreter versions
    of CI to close. This test goes red the moment someone makes the rescue
    symmetric.
    """
    # The round-20 operator, entry for entry (see
    # tests/test_pr121_review_round20.py::_overflowing_norm_operator).
    op = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)
    op[0, 1] = 1.0e308
    op[3, 1] = -1.0e308
    assert np.all(np.isfinite(op)), "the input must be FINITE for this to be the case"
    with np.errstate(over="ignore"):
        defect, fro = trace_preservation_defect(op)
    assert defect == pytest.approx(np.sqrt(17.0)), "a real, order-one trace defect"
    assert not np.isfinite(fro), (
        "the overflowing reference scale was made finite; the round-21 guard "
        "that refuses a non-finite scale now has nothing to refuse"
    )
    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cert = certified_eigvals(op)[-1]
    assert not cert.applicable and not cert.certified


def test_the_repair_reaches_into_the_subnormal_range() -> None:
    """The finding, continued past the scale the reviewer happened to name.

    Fixing the norms alone left the rate-unit dependence intact below ~1e-317,
    because the gate floors its reference scale at ``max(fro, tiny)`` and that
    constant is LARGER than a subnormal operator. Measured with the norms
    repaired but the floor still in place: defect 3e-320 against a floor-derived
    bound of 2.2e-318, i.e. "trace preserving" for a relative defect of 0.80.
    """
    for scale in (1e-310, 1e-315, 1e-320, 5e-324):
        op = _non_tp_operator() * scale
        if not np.any(op):
            continue
        # BOTH ladders: the floor was removed in each, and a guard that only
        # one API exercises is a guard sitting off half its path.
        for name, fn in (
            ("certified_eigvals", certified_eigvals),
            ("certified_eig", certified_eig),
        ):
            with np.errstate(under="ignore"):
                cert = fn(op)[-1]
            assert not cert.applicable, (
                f"{name} at scale {scale:g}: a relative trace defect of "
                "3/sqrt(14) was declared trace preserving; the reference scale "
                "was floored at a constant instead of measured"
            )


def test_a_healthy_generator_survives_the_subnormal_repair() -> None:
    """POSITIVE CONTROL, and a regression this run actually produced.

    The first version of the rescue divided by the largest component. That is
    COMPLEX division, and for a subnormal divisor NumPy forms an intermediate
    reciprocal that overflows: both norms came back NaN, the round-21 guard
    refused the operator, and a valid GKSL generator at 1e-310 went from
    ``applicable=True`` to ``False``. Repairing a fail-open must not install a
    fail-closed defect in its place, so the scaling is now an exact power-of-two
    shift. This test is the one that would have caught it.
    """
    for scale in (1e-308, 1e-310, 1e-315):
        gen = _healthy_generator() * scale
        with np.errstate(under="ignore"):
            defect, fro = trace_preservation_defect(gen)
            cert = certified_eigvals(gen)[-1]
        assert np.isfinite(defect) and np.isfinite(fro), (
            f"scale {scale:g}: the scaled computation produced "
            f"(defect, fro) = ({defect}, {fro}) for finite input"
        )
        assert cert.applicable and cert.certified, (
            f"a GKSL generator at scale {scale:g} was refused"
        )


def test_the_exactly_zero_operator_keeps_its_zero_scale() -> None:
    """POSITIVE CONTROL for the other edge of the rescue.

    ``fro == 0`` is a lost measurement for a nonzero operator and the correct
    answer for the zero one. Rescaling by a maximum of zero is undefined, so
    the branch must not be entered.
    """
    defect, fro = trace_preservation_defect(np.zeros((4, 4), dtype=complex))
    assert (defect, fro) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Finding 13: withhold D16 when the spectral certificate is unresolved.
# ---------------------------------------------------------------------------

_STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
_UNRESOLVED_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 1e8, 1.42e-5]
_RESOLVED_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]


def _classical_network(rates: list[float]) -> np.ndarray:
    jumps = []
    for (to, frm) in _STIFF_PAIRS:
        j = np.zeros((4, 4), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, rates)


def _population_steady_state(rates: list[float]) -> np.ndarray:
    k = np.zeros((4, 4))
    for (to, frm), g in zip(_STIFF_PAIRS, rates):
        k[to, frm] += g
        k[frm, frm] -= g
    w, v = np.linalg.eig(k)
    p = np.real(v[:, int(np.argmin(np.abs(w)))])
    return np.diag(p / p.sum()).astype(complex)


def _layers(rates: list[float]):
    lsup = _classical_network(rates)
    rho = _population_steady_state(rates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectral = compute_spectral_layer(lsup, rho)
        cert = spectral.zero_mode_certificate
        lep = compute_lep_layer(
            lsup,
            spectral.eigenvalues,
            beta_D_linear=1.0,
            gap=spectral.gap,
            rho_steady_state=rho,
            seed=7,
            n_haar=3,
            spectral_resolved=not (cert["applicable"] and not cert["resolved"]),
        )
    return spectral, lep


def test_the_fixture_really_is_an_unresolved_certificate() -> None:
    """PRECONDITION, asserted separately from the verdict it produces.

    Without this the tests below could pass because the fixture stopped being
    unresolved -- the failure mode in which a regression quietly starts testing
    nothing.
    """
    spectral, _ = _layers(_UNRESOLVED_RATES)
    cert = spectral.zero_mode_certificate
    assert cert["applicable"] is True and cert["resolved"] is False
    assert np.isnan(spectral.gap), "D1 must already be withheld here"


def test_d16_is_withheld_on_an_unresolved_spectrum() -> None:
    """D16 must not be published from a spectrum D1/D3/D4 refused to report.

    Measured before the fix: ``lep_proximity = 0.0`` with 51 candidate pairs.
    """
    _, lep = _layers(_UNRESOLVED_RATES)
    assert np.isnan(lep.lep_proximity), (
        f"D16 reported {lep.lep_proximity!r} from an unresolved spectrum"
    )
    assert lep.lep_candidate_count is None, (
        f"D16 candidate count {lep.lep_candidate_count!r} was counted on a "
        "spectrum the layer declined to report"
    )


def test_zero_is_not_the_unavailable_value_it_is_the_strongest_signal() -> None:
    """DISCRIMINATION: the withheld value must not be confusable with a measured one.

    ``0.0`` means coalesced and ``inf`` means maximally separated; both are
    measurements. Only NaN says the question was not answered, and
    ``_strip_unavailable`` in the classifier keys on exactly that.
    """
    _, lep = _layers(_UNRESOLVED_RATES)
    assert lep.lep_proximity != 0.0
    assert not np.isinf(lep.lep_proximity)


def test_a_resolved_spectrum_still_reports_d16() -> None:
    """POSITIVE CONTROL. A guard that withheld D16 always discriminates nothing."""
    spectral, lep = _layers(_RESOLVED_RATES)
    cert = spectral.zero_mode_certificate
    assert cert["resolved"] is True
    assert np.isfinite(lep.lep_proximity), "D16 was withheld on a resolved spectrum"
    assert isinstance(lep.lep_candidate_count, int)


def test_diagnose_itself_withholds_d16_on_an_unresolved_run() -> None:
    """The WIRING, not just the layer.

    ``compute_lep_layer`` can withhold correctly and still publish D16 to every
    real caller if ``diagnose`` never tells it the certificate was unresolved.
    Tests that exercise only the layer leave that line undefended -- the guard
    would sit off the path that actually runs.
    """
    lsup = _classical_network(_UNRESOLVED_RATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = diagnose(
            lsup,
            rho_steady_state=_population_steady_state(_UNRESOLVED_RATES),
            include_mpemba=False,
            bootstrap_B=8,
            seed=7,
        )
    cert = report.spectral.zero_mode_certificate
    assert cert is not None
    assert cert["applicable"] is True and cert["resolved"] is False
    assert np.isnan(report.lep.lep_proximity), (
        f"diagnose published D16 = {report.lep.lep_proximity!r} from an "
        "unresolved spectrum; the layer withholds but the caller does not"
    )
    assert report.lep.lep_candidate_count is None


def test_withholding_does_not_switch_off_the_non_finite_eigenvalue_refusal() -> None:
    """Closing one fail-open must not open another.

    ``lep_proximity`` carries the issue-#82 refusal of non-finite eigenvalues:
    solver corruption is not an exceptional-point signal. Returning early to
    withhold D16 would skip that call and, with it, that refusal -- the cheapest
    way to turn a fix into a regression.
    """
    lsup = _classical_network(_RESOLVED_RATES)
    bad = np.array([0.0, np.nan, -1.0, -2.0], dtype=complex)
    with pytest.raises(ValueError, match="finite eigenvalues"):
        compute_lep_layer(
            lsup,
            bad,
            beta_D_linear=1.0,
            gap=float("nan"),
            rho_steady_state=_population_steady_state(_RESOLVED_RATES),
            seed=7,
            n_haar=3,
            spectral_resolved=False,
        )


def test_the_default_keeps_the_layer_reporting() -> None:
    """POSITIVE CONTROL for the parameter itself: callers who say nothing measure."""
    lsup = _classical_network(_RESOLVED_RATES)
    rho = _population_steady_state(_RESOLVED_RATES)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectral = compute_spectral_layer(lsup, rho)
        lep = compute_lep_layer(
            lsup,
            spectral.eigenvalues,
            beta_D_linear=1.0,
            gap=spectral.gap,
            rho_steady_state=rho,
            seed=7,
            n_haar=3,
        )
    assert np.isfinite(lep.lep_proximity)
    assert isinstance(lep.lep_candidate_count, int)
