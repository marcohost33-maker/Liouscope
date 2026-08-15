"""Issue #112: the spectral layer must not report a gap the eigensolver lost.

For a trace-preserving generator ``vec(I)^H L = 0`` holds exactly, so ``0`` is
an exact eigenvalue. A computed spectrum without one is proof that the solve
failed -- a theorem, not a tuned threshold. These tests pin the measured stiff
failure, the repair, the no-op behaviour on healthy systems, and the
fail-closed behaviour when no repair route works.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope._consts import ZERO_MODE_AMBIGUITY_FACTOR
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.spectral import compute_spectral_layer, liouvillian_gap
from liouscope.numerics.linalg import (
    certified_eigvals,
    eig_nonhermitian,
    trace_preservation_defect,
)

# The measured #112 repro: a four-level classical jump network with a rate
# spread of ~1e10. Exactly trace preserving, and because H = 0 the generator
# block-decouples, so the population block gives an independent ground truth.
STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
STIFF_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]
STIFF_TRUE_GAP = 1.074e-5      # slowest coherence mode, -0.5*(Gamma_1 + Gamma_3)
STIFF_WRONG_GAP = 7.28e-6      # what zgeev reported before this fix


def _classical_network(pairs, rates, d=4):
    jumps = []
    for (to, frm) in pairs:
        j = np.zeros((d, d), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((d, d), dtype=complex), jumps, rates)


def _analytic_spectrum(pairs, rates, d=4):
    """Exact spectrum of a classical jump network with H = 0.

    Populations follow the d x d Markov generator K; coherences are exactly
    decoupled and diagonal with rates -0.5*(Gamma_i + Gamma_j), i != j.
    """
    k = np.zeros((d, d))
    for (to, frm), g in zip(pairs, rates):
        k[to, frm] += g
        k[frm, frm] -= g
    gamma = -np.diag(k)
    coh = [-0.5 * (gamma[i] + gamma[j]) for i in range(d) for j in range(d) if i != j]
    return np.concatenate([np.linalg.eigvals(k), np.array(coh, dtype=complex)])


def _gap(ev, rel=1e-12):
    ev = np.asarray(ev)
    radius = float(np.max(np.abs(ev))) or 1.0
    nonzero = ev[np.abs(ev) > rel * radius]
    return float(min(-e.real for e in nonzero))


# --------------------------------------------------------------------------
# The defect, and that it is a solver failure rather than a threshold choice.
# --------------------------------------------------------------------------


def test_stiff_generator_is_exactly_trace_preserving() -> None:
    """No zero mode may be blamed on a malformed generator."""
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    defect, fro = trace_preservation_defect(lsup)
    assert defect == 0.0
    assert fro > 0.0


def test_incumbent_solver_loses_the_zero_mode() -> None:
    """DISCRIMINATION: without the repair the spectrum has no zero mode at all."""
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    raw = eig_nonhermitian(lsup).eigenvalues
    smallest = float(np.min(np.abs(raw)))
    backward_error = float(np.finfo(float).eps * np.linalg.norm(lsup, 2))
    assert smallest > 1e4 * backward_error, (
        "the repro no longer exercises the #112 defect: zgeev found a zero mode"
    )
    assert liouvillian_gap(raw) == pytest.approx(STIFF_WRONG_GAP, rel=1e-6)


def test_no_zero_mode_tolerance_can_repair_it() -> None:
    """The wrong gap is identical under the relative AND the legacy filter.

    This is what makes #112 distinct from #108: the information is missing from
    the eigenvalues, so no separation threshold recovers it.
    """
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    raw = eig_nonhermitian(lsup).eigenvalues
    assert liouvillian_gap(raw) == pytest.approx(liouvillian_gap(raw, atol=1e-10))


def test_certificate_repairs_the_stiff_gap() -> None:
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    ev, cert = certified_eigvals(lsup)
    assert cert.applicable and cert.certified
    assert cert.solver != "zgeev"
    assert cert.residual <= cert.bound
    assert liouvillian_gap(ev) == pytest.approx(STIFF_TRUE_GAP, rel=1e-9)


def test_repaired_gap_matches_the_independent_analytic_spectrum() -> None:
    """Ground truth from the decoupled blocks, not from another dense solve."""
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    ev, _ = certified_eigvals(lsup)
    assert _gap(ev) == pytest.approx(_gap(_analytic_spectrum(STIFF_PAIRS, STIFF_RATES)),
                                     rel=1e-9)


def test_spectral_layer_reports_the_repaired_gap_and_the_certificate() -> None:
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    k = np.zeros((4, 4))
    for (to, frm), g in zip(STIFF_PAIRS, STIFF_RATES):
        k[to, frm] += g
        k[frm, frm] -= g
    w, v = np.linalg.eig(k)
    p = np.real(v[:, int(np.argmin(np.abs(w)))])
    rho_ss = np.diag(p / p.sum()).astype(complex)

    result = compute_spectral_layer(lsup, rho_ss)
    assert result.gap == pytest.approx(STIFF_TRUE_GAP, rel=1e-9)
    cert = result.zero_mode_certificate
    assert cert is not None
    assert cert["applicable"] is True and cert["certified"] is True
    assert cert["residual"] <= cert["bound"]


# --------------------------------------------------------------------------
# No-op on healthy systems: the incumbent result must survive untouched.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
def test_healthy_generator_keeps_the_incumbent_solver(scale: float) -> None:
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    lsup = scale * build_liouvillian(
        np.zeros((2, 2), dtype=complex), [sm], [1.0]
    )
    ev, cert = certified_eigvals(lsup)
    assert cert.certified and cert.solver == "zgeev"
    np.testing.assert_allclose(
        np.sort_complex(ev), np.sort_complex(eig_nonhermitian(lsup).eigenvalues)
    )


def test_sweep_of_random_networks_never_degrades_the_gap() -> None:
    """Certified result is at least as accurate as the incumbent, system by system."""
    rng = np.random.default_rng(20260815)
    d = 4
    offdiag = [(i, j) for i in range(d) for j in range(d) if i != j]
    checked = 0
    for trial in range(120):
        idx = rng.choice(len(offdiag), size=5, replace=False)
        pairs = [offdiag[k] for k in idx]
        if trial % 2 == 0:
            rates = list(10 ** rng.uniform(-5.5, -4.5, 5))
            rates[int(rng.integers(5))] = float(10 ** rng.uniform(4.5, 6.0))
        else:
            rates = list(10 ** rng.uniform(-0.5, 0.5, 5))
        lsup = _classical_network(pairs, rates, d)
        try:
            truth = _gap(_analytic_spectrum(pairs, rates, d))
        except ValueError:
            continue
        if not np.isfinite(truth) or truth <= 0:
            continue
        incumbent = _gap(eig_nonhermitian(lsup).eigenvalues)
        certified = _gap(certified_eigvals(lsup)[0])
        err_inc = abs(incumbent - truth) / truth
        err_cert = abs(certified - truth) / truth
        assert err_cert <= max(err_inc, 1e-9) * 1.000001, (
            f"trial {trial}: certified gap {certified:.6e} is worse than "
            f"incumbent {incumbent:.6e} against truth {truth:.6e}"
        )
        checked += 1
    assert checked > 80, f"sweep degenerated to {checked} usable systems"


# --------------------------------------------------------------------------
# Applicability and fail-closed behaviour.
# --------------------------------------------------------------------------


def test_certificate_is_not_applicable_to_a_non_trace_preserving_operator() -> None:
    """No guaranteed zero mode -> the check must say nothing rather than fire."""
    rng = np.random.default_rng(5)
    arbitrary = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    ev, cert = certified_eigvals(arbitrary)
    assert cert.applicable is False
    np.testing.assert_allclose(
        np.sort_complex(ev), np.sort_complex(eig_nonhermitian(arbitrary).eigenvalues)
    )


def test_uncertified_spectrum_warns_instead_of_reporting_silently() -> None:
    """Fail-closed: an unrepairable solve must not pass as a measurement."""
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    lsup = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])
    rho_ss = np.array([[1, 0], [0, 0]], dtype=complex)

    # Force the certificate to be unsatisfiable by demanding an impossibly
    # tight bound; this exercises the reporting path, not a real solver failure.
    import liouscope.diagnostics.spectral as spectral_mod

    def _impossible(l_super, **_kw):
        from liouscope.numerics.linalg import ZeroModeCertificate
        return (
            eig_nonhermitian(l_super).eigenvalues,
            ZeroModeCertificate(
                applicable=True, certified=False, solver="zgeev",
                residual=1.0, bound=0.0, trace_defect=0.0,
            ),
        )

    original = spectral_mod.certified_eigvals
    spectral_mod.certified_eigvals = _impossible
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = compute_spectral_layer(lsup, rho_ss)
        messages = [str(w.message) for w in caught]
        assert any("issue #112" in m for m in messages), messages
        assert result.zero_mode_certificate["certified"] is False
    finally:
        spectral_mod.certified_eigvals = original


# --------------------------------------------------------------------------
# Issue #113: unresolvable slow spectrum must not be reported as a fast gap.
# --------------------------------------------------------------------------


def _stiff_with_fast_rate(fast: float):
    return _classical_network(
        STIFF_PAIRS, [7.28e-6, 3.67e-5, 1.53e-5, fast, 1.42e-5]
    )


def _population_steady_state(rates) -> np.ndarray:
    k = np.zeros((4, 4))
    for (to, frm), g in zip(STIFF_PAIRS, rates):
        k[to, frm] += g
        k[frm, frm] -= g
    w, v = np.linalg.eig(k)
    p = np.real(v[:, int(np.argmin(np.abs(w)))])
    return np.diag(p / p.sum()).astype(complex)


@pytest.mark.parametrize("fast", [2.7e5, 1e7])
def test_in_range_spread_still_reports_the_true_gap(fast: float) -> None:
    """Below the ceiling nothing changes: exactly one zero mode, real gap."""
    lsup = _stiff_with_fast_rate(fast)
    rates = [7.28e-6, 3.67e-5, 1.53e-5, fast, 1.42e-5]
    result = compute_spectral_layer(lsup, _population_steady_state(rates))
    assert result.zero_mode_certificate["zero_mode_count"] == 1
    assert result.zero_mode_certificate["resolved"] is True
    assert result.gap == pytest.approx(STIFF_TRUE_GAP, rel=1e-9)


@pytest.mark.parametrize("fast", [1e8])
def test_unresolvable_spread_reports_nan_not_a_fast_mode(fast: float) -> None:
    """Above the ceiling D1 must refuse, not report the next surviving mode."""
    lsup = _stiff_with_fast_rate(fast)
    rates = [7.28e-6, 3.67e-5, 1.53e-5, fast, 1.42e-5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_spectral_layer(lsup, _population_steady_state(rates))
    assert np.isnan(result.gap), (
        f"D1 reported {result.gap:.6e}; the slow modes are below the solver's "
        "backward error, so no gap is measurable"
    )
    cert = result.zero_mode_certificate
    assert cert["zero_mode_count"] > 1 and cert["resolved"] is False
    assert any("issue #113" in str(w.message) for w in caught)


@pytest.mark.parametrize("fast", [1e8])
def test_unresolved_gap_is_not_reported_as_gapless(fast: float) -> None:
    """DISCRIMINATION: 0.0 would be wrong too -- it fires the gapless F5 leg."""
    lsup = _stiff_with_fast_rate(fast)
    rates = [7.28e-6, 3.67e-5, 1.53e-5, fast, 1.42e-5]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_spectral_layer(lsup, _population_steady_state(rates))
    assert result.gap != 0.0


def test_degenerate_stationary_manifold_is_not_flagged_unresolved() -> None:
    """A conserved quantity gives several GENUINE zero modes -- that is legal.

    The #113 check must key on modes that are inside the zero-mode tolerance
    without being machine-zero, not on the mere COUNT of zero modes. A
    symmetry-degenerate manifold puts its extra zeros at exactly 0, and the gap
    taken from the complement is correct. (This case is why counting alone was
    wrong: it flagged a healthy fixture with a spectral spread of only 4.)
    """
    # Two-qubit dephasing with a conserved magnetisation sector: three exact
    # zero modes, unit-scale spectrum, nothing stiff about it.
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    lsup = build_liouvillian(
        np.zeros((4, 4), dtype=complex),
        [np.kron(sz, eye), np.kron(eye, sz)],
        [0.5, 0.5],
    )
    _, cert = certified_eigvals(lsup)
    assert cert.certified
    assert cert.zero_mode_count > 1, "fixture must actually be degenerate"
    assert cert.ambiguous_count == 0
    assert cert.resolved is True, (
        "genuine degeneracy must not be reported as an unresolved spectrum"
    )


def test_ambiguity_split_keeps_margin_on_both_sides() -> None:
    """The #113 split must clear the healthy population AND catch the defect.

    The two populations are NOT cleanly separated -- measured over 83 healthy
    generators the largest genuine in-band |lambda| reaches 2.38 * eps*||L||,
    while the marginal unresolved case sits at 4.87, about 2x away. The split
    is therefore placed at 30x: well above everything healthy, still below the
    case it is meant to catch (4.87e2). This test pins both margins, so a
    future change that narrows either one fails loudly rather than silently
    turning the check into a coin flip.
    """
    def level_of(m: np.ndarray) -> float:
        return float(np.finfo(float).eps * np.linalg.norm(m, 2))

    # healthy side: must stay comfortably BELOW the split
    for fast in (2.7e5, 1e7):
        lsup = _stiff_with_fast_rate(fast)
        ev, cert = certified_eigvals(lsup)
        band = np.abs(ev)[np.abs(ev) <= cert.bound]
        ratio = band.max() / level_of(lsup)
        assert ratio < ZERO_MODE_AMBIGUITY_FACTOR / 10.0, (
            f"healthy zero mode at {ratio:.2f}x is within 10x of the split"
        )
        assert cert.resolved

    # broken side: must stay comfortably ABOVE the split
    broken = _stiff_with_fast_rate(1e8)
    ev_b, cert_b = certified_eigvals(broken)
    band_b = np.abs(ev_b)[np.abs(ev_b) <= cert_b.bound]
    ratio_b = band_b.max() / level_of(broken)
    assert ratio_b > ZERO_MODE_AMBIGUITY_FACTOR * 10.0, (
        f"unresolved mode at {ratio_b:.2f}x is within 10x of the split"
    )
    assert not cert_b.resolved


def test_the_discarded_value_really_was_a_fast_mode() -> None:
    """DISCRIMINATION: pin what the pre-#113 code would have reported.

    If this stops holding, the repro no longer exercises the defect and the
    NaN assertions above become vacuous.
    """
    lsup = _stiff_with_fast_rate(1e8)
    ev, cert = certified_eigvals(lsup)
    assert cert.certified and cert.zero_mode_count > 1
    would_have_reported = liouvillian_gap(ev)
    slowest_genuine = min(
        abs(e.real) for e in ev if abs(e) > 1e-14 * float(np.max(np.abs(ev)))
    )
    assert would_have_reported > 1e6 * slowest_genuine, (
        "the pre-#113 value was supposed to be a FAST mode, orders above the "
        f"slowest genuine one; got {would_have_reported:.3e} vs "
        f"{slowest_genuine:.3e}"
    )


def test_certificate_dict_is_json_serialisable() -> None:
    import json

    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    _, cert = certified_eigvals(lsup)
    json.dumps(cert.as_dict(), allow_nan=False)      # RFC 8259: no NaN/inf
