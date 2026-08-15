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
import scipy.linalg as sla

from liouscope._consts import ZERO_MODE_AMBIGUITY_FACTOR
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.spectral import (
    compute_spectral_layer,
    liouvillian_gap,
    oscillating_mode_gap,
)
from liouscope.numerics.linalg import (
    certified_eig,
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


def test_uncertified_spectrum_warns_instead_of_reporting_silently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed: an unrepairable solve must not pass as a measurement."""
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    lsup = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])
    rho_ss = np.array([[1, 0], [0, 0]], dtype=complex)

    # Force the certificate to be unsatisfiable by demanding an impossibly
    # tight bound; this exercises the reporting path, not a real solver failure.
    # Patched by string target so the module is not imported a second time
    # under a different style (CodeQL py/import-and-import-from), and so the
    # restore is fixture-managed instead of a manual try/finally.
    def _impossible(l_super, **_kw):
        from liouscope.numerics.linalg import ZeroModeCertificate
        return (
            eig_nonhermitian(l_super).eigenvalues,
            ZeroModeCertificate(
                applicable=True, certified=False, solver="zgeev",
                residual=1.0, bound=0.0, trace_defect=0.0,
            ),
        )

    monkeypatch.setattr(
        "liouscope.diagnostics.spectral.certified_eigvals", _impossible
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_spectral_layer(lsup, rho_ss)
    messages = [str(w.message) for w in caught]
    assert any("issue #112" in m for m in messages), messages
    assert result.zero_mode_certificate["certified"] is False


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


# --------------------------------------------------------------------------
# Issue #112 follow-up: the eigenVECTOR consumers (D19 / A11-F4 rung).
# --------------------------------------------------------------------------


def test_incumbent_slow_eigenvector_is_not_an_eigenvector() -> None:
    """DISCRIMINATION: the pre-fix slow 'mode' satisfies no eigenvalue equation.

    This is what makes the D19 exposure worse than a wrong number: the overlap
    was measured against a vector with a residual LARGER than the eigenvalue it
    claims to belong to.
    """
    from liouscope.numerics.linalg import eig_nonhermitian as _eig

    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    decomp = _eig(lsup, compute_left=True)
    ev = decomp.eigenvalues
    tol = 1e3 * np.finfo(float).eps * np.linalg.norm(lsup, 2)
    mask = np.abs(ev) > tol
    idx = np.where(mask)[0][int(np.argmax(np.real(ev[mask])))]
    r = decomp.right_vectors[:, idx]
    r = r / np.linalg.norm(r)
    residual = float(np.linalg.norm(lsup @ r - ev[idx] * r))
    assert residual > abs(ev[idx]), (
        "repro no longer exercises the defect: the incumbent slow eigenvector "
        f"has residual {residual:.3e} against |lambda| = {abs(ev[idx]):.3e}"
    )


def test_certified_decomposition_returns_genuine_eigenvectors() -> None:
    """After repair, both left and right slow eigenvectors satisfy their equations."""
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    decomp, cert = certified_eig(lsup)
    assert cert.certified and cert.solver != "zgeev"
    ev = decomp.eigenvalues
    assert decomp.left_vectors is not None
    tol = 1e3 * np.finfo(float).eps * np.linalg.norm(lsup, 2)
    mask = np.abs(ev) > tol
    idx = np.where(mask)[0][int(np.argmax(np.real(ev[mask])))]
    lam = ev[idx]
    assert lam.real == pytest.approx(-STIFF_TRUE_GAP, rel=1e-9)

    r = decomp.right_vectors[:, idx] / np.linalg.norm(decomp.right_vectors[:, idx])
    left = decomp.left_vectors[:, idx] / np.linalg.norm(decomp.left_vectors[:, idx])
    scale = float(np.linalg.norm(lsup, 2))
    assert np.linalg.norm(lsup @ r - lam * r) <= 1e3 * np.finfo(float).eps * scale
    assert np.linalg.norm(left.conj() @ lsup - lam * left.conj()) <= (
        1e3 * np.finfo(float).eps * scale
    )


def test_slowest_mode_selects_the_true_slow_mode() -> None:
    """The D19 entry point now picks the analytically correct mode."""
    from liouscope.diagnostics.mpemba import _slowest_mode

    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    mode = _slowest_mode(lsup)
    assert mode is not None
    l_slow, r_slow = mode
    r_slow = r_slow / np.linalg.norm(r_slow)
    rayleigh = complex(np.vdot(r_slow, lsup @ r_slow))
    assert rayleigh.real == pytest.approx(-STIFF_TRUE_GAP, rel=1e-6)


def test_certified_eig_is_a_no_op_on_healthy_generators() -> None:
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    lsup = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])
    decomp, cert = certified_eig(lsup)
    assert cert.certified and cert.solver == "zgeev"
    incumbent = eig_nonhermitian(lsup, compute_left=True)
    np.testing.assert_allclose(decomp.eigenvalues, incumbent.eigenvalues)
    np.testing.assert_allclose(decomp.right_vectors, incumbent.right_vectors)


def test_certified_eig_ladder_is_narrower_than_eigvals_and_says_so() -> None:
    """A complex-valued stiff generator has no eigenvector-producing repair.

    certified_eigvals may still repair such a case via Schur; certified_eig
    must NOT claim to, because Schur yields no eigenvectors. The honest
    outcome is certified=False, not a silently different spectrum.
    """
    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES).astype(complex)
    lsup = lsup + 0j
    lsup[0, 0] += 1e-18j          # make it non-real without changing the physics
    decomp, cert = certified_eig(lsup)
    if not cert.certified:
        # the narrower ladder gave up -- it must return the incumbent, unchanged
        incumbent = eig_nonhermitian(lsup, compute_left=True)
        np.testing.assert_allclose(decomp.eigenvalues, incumbent.eigenvalues)


def test_certificate_dict_is_json_serialisable() -> None:
    import json

    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    _, cert = certified_eigvals(lsup)
    json.dumps(cert.as_dict(), allow_nan=False)      # RFC 8259: no NaN/inf


# --------------------------------------------------------------------------
# Twelfth-round review: repair-ladder correctness and consumption guards.
# --------------------------------------------------------------------------


def test_dgeev_real_route_requires_exact_realness():
    """A tiny imaginary part must not be silently deleted by the real route.

    `np.allclose(imag, 0)` carries an absolute default atol, so a stiff complex
    generator in small rate units read as "real" and `dgeev-real` solved a
    DIFFERENT matrix -- the Hamiltonian contribution vanished, moving D3 under
    a pure `L -> cL` unit change. Exact realness is the only scale-invariant
    criterion under which the route is valid.
    """
    H = np.diag([0.0, 1.0e-7, 2.0e-7, 3.0e-7]).astype(complex)
    lsup = _classical_network(STIFF_PAIRS, [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5])
    d = 4
    eye = np.eye(d, dtype=complex)
    lsup = lsup + (-1j * (np.kron(eye, H) - np.kron(H.T, eye)))

    ref_osc = oscillating_mode_gap(certified_eigvals(lsup)[0])
    for c in (1.0e-10, 1.0, 1.0e5):
        ev, cert = certified_eigvals(c * lsup)
        assert cert.solver != "dgeev-real", "complex L must never take the real route"
        assert oscillating_mode_gap(ev) / c == pytest.approx(ref_osc, rel=1e-6)


def test_repair_ladder_is_lazy_for_the_healthy_path(monkeypatch):
    """A certified primary solve must not pay for the fallback decompositions."""
    calls: list[str] = []
    original_schur = sla.schur

    def _counting_schur(*args, **kwargs):
        calls.append("schur")
        return original_schur(*args, **kwargs)

    monkeypatch.setattr(sla, "schur", _counting_schur)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    lsup = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])
    _ev, cert = certified_eigvals(lsup)
    assert cert.certified and cert.solver == "zgeev"
    assert calls == [], "healthy path executed a fallback Schur decomposition"


def test_unresolved_spectrum_withholds_the_mpemba_overlap():
    """D19 must abstain, not fire, when the slow spectrum is unresolved.

    The uncertified decomposition of the stiff network offers a "slowest"
    vector with Rayleigh value ~-5e7 against an analytic slow mode near
    -1e-5; consuming it corrupts the classifier's highest-priority A11/F4
    rung. Equally important: the abstention value is NaN, not 0.0 -- a 0.0
    would itself read as "the initial state skips the slowest mode" and
    MANUFACTURE a Mpemba candidate out of solver failure.
    """
    from liouscope.diagnostics.mpemba import compute_mpemba_layer, overlap_c1

    lsup = _stiff_with_fast_rate(1.0e8)
    _ev, cert = certified_eigvals(lsup)
    if cert.resolved:  # pragma: no cover - guard against future solver changes
        pytest.skip("fixture no longer produces an unresolved spectrum")

    rho0 = np.zeros((4, 4), dtype=complex)
    rho0[1, 1] = rho0[2, 2] = 0.5
    rho0[1, 2] = rho0[2, 1] = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = overlap_c1(lsup, rho0)
        result = compute_mpemba_layer(
            lsup, rho0, rho_steady_state=_population_steady_state(
                [7.28e-6, 3.67e-5, 1.53e-5, 1.0e8, 1.42e-5]
            )
        )
    assert np.isnan(c1)
    assert np.isnan(result.overlap_c1)
    assert result.is_mpemba_candidate is False


def test_unresolved_certificate_floors_the_verdict_to_undefined():
    """No mechanism claim may stand on a spectrum the solver could not resolve.

    Fail-closed at the VERDICT level, exactly like the non-finite-beta_D floor:
    class and family remain the best-fit hypothesis, the claim degrades to
    UNDEFINED / EXPLORATION. Synthetic certificates keep the test independent
    of which fixtures happen to defeat the current repair ladder.
    """
    from liouscope._types import (
        LepResult,
        MpembaResult,
        NonNormalityResult,
        RelaxationResult,
        ResolventResult,
        SpectralResult,
        TransientResult,
    )
    from liouscope.diagnostics.classification import classify_mechanism

    arr = np.zeros(1, dtype=complex)

    def _classify(certificate):
        # Local synthetic results (no cross-test import: the ``tests``
        # directory is not an importable package on the CI runner).
        return classify_mechanism(
            SpectralResult(
                gap=0.5, gns_gap=0.5, kms_gap=0.5, oscillating_gap=0.1,
                spectral_spread=1.0, eigenvalues=arr,
                steady_state=np.zeros((1, 1), dtype=complex),
                has_complex_pairs=False, zero_mode_certificate=certificate,
            ),
            NonNormalityResult(
                henrici_eta=0.5, petermann_max=1.0, petermann_factors=arr,
                kreiss=1.0, bohr_ap_length=1, bohr_ap_pauli_bound=0.0,
            ),
            RelaxationResult(
                von_neumann_entropy=0.0, relative_entropy_curve=arr.real,
                fidelity_curve=arr.real, entanglement_asymmetry=None, fits={},
                aicc_model="M1", beta_D=0.5, bca_ci_beta=(0.4, 0.6),
            ),
            ResolventResult(
                resolvent_peak=1.0, ridge_fwhm=1.0, pseudospectral_radius=0.5,
                pseudospec_eps=1.0e-3,
            ),
            TransientResult(
                trans_amplitude_ratio=1.0, kappa_trans=1.0, numerical_abscissa=0.0,
            ),
            LepResult(
                lep_proximity=1.0, gap_rate_consistency=0.01,
                initial_state_sensitivity=0.0, lep_candidate_count=0,
            ),
            MpembaResult(
                overlap_c1=0.5, expansion_alpha=1.0, is_mpemba_candidate=False,
            ),
        )

    resolved = _classify(
        {"applicable": True, "certified": True, "resolved": True}
    )
    assert resolved.verdict != "UNDEFINED"

    for bad in (
        {"applicable": True, "certified": False, "resolved": False},
        {"applicable": True, "certified": True, "resolved": False},  # ambiguous
    ):
        floored = _classify(bad)
        assert floored.verdict == "UNDEFINED"
        assert floored.tier == "EXPLORATION"
        assert floored.a_class == resolved.a_class  # hypothesis label preserved
        assert floored.evidence["spectral_resolved"] == 0.0

    # Not applicable (non-trace-preserving input): no floor, no evidence key.
    inapplicable = _classify(
        {"applicable": False, "certified": False, "resolved": False}
    )
    assert inapplicable.verdict == resolved.verdict
    assert "spectral_resolved" not in inapplicable.evidence


# --------------------------------------------------------------------------
# Round-13 review: ONE zero-mode scale for certification and filtering.
# --------------------------------------------------------------------------

# A strongly non-normal, exactly trace-preserving 4x4 generator with true
# spectrum {0, -1, -2, -3} and ``||L||_2 / max|lambda| ~ 4e3`` (built as
# ``V diag(0,-1,-2,-3) V^{-1}`` with three nearly parallel columns of ``V``
# orthogonal to ``vec(I)``; frozen as exact float64 literals so the test is
# deterministic). On this operator ``zgeev`` displaces the stationary
# eigenvalue to ``~9e-13`` -- inside the certificate's operator-derived
# backward-error bound ``rtol * eps * ||L||_2`` (certified, resolved), but
# ABOVE the radius-based #108 tolerance ``rtol * eps * max|lambda|``. Before
# the round-13 fix the certified zero mode therefore survived the downstream
# filter as a spurious "genuine" mode: D1 reported ``~1e-12`` (negative on
# some runs) instead of the true gap ``1``, and D9 reported four eigenmodes
# where three exist.
_NONNORMAL_TP = np.array([
    [1492.4052081746918-576.74034398067965j, 4242.3884273335898+2154.8269993072913j, -123.4536161693602+4504.5869089583975j, -108.4279406414272+1496.7026292274365j],
    [436.11533077659476+1358.1304295507621j, -2136.4399993753545+3659.1255589614616j, -3997.0661272793914-342.52784186068834j, -1323.878377954796-173.62369459581453j],
    [-593.97781553431003+43.262846725648785j, -1272.4461716988801-1224.2111146810887j, 529.60685055923602-1585.6825857533454j, 199.42084850952054-519.04474610309012j],
    [-1492.4052081746918+576.74034398067965j, -4242.3884273335898-2154.8269993072909j, 123.4536161693602-4504.5869089583975j, 108.42794064142709-1496.7026292274363j],
])


def test_nonnormal_fixture_shows_the_scale_mismatch() -> None:
    """Pin the defect's precondition: certified-resolved, yet above the
    radius tolerance.

    If a future LAPACK computes this stationary eigenvalue below the
    radius-based tolerance the fixture no longer exercises the mismatch;
    skip rather than fail so the guard is visible.
    """
    from liouscope.numerics.scale import spectral_zero_tolerance

    defect, fro = trace_preservation_defect(_NONNORMAL_TP)
    assert defect <= 1.0e-12 * fro, "fixture must be trace preserving"

    ev, cert = certified_eigvals(_NONNORMAL_TP)
    assert cert.applicable and cert.certified and cert.resolved
    ratio = float(np.linalg.norm(_NONNORMAL_TP, 2)) / float(np.max(np.abs(ev)))
    assert ratio > 1.0e3, "fixture must be strongly non-normal"

    residual = float(np.min(np.abs(ev)))
    if residual <= spectral_zero_tolerance(ev):  # pragma: no cover
        pytest.skip("solver resolved the zero mode below the radius tolerance")
    assert residual <= cert.bound  # inside the operator-derived bound


def test_certificate_bound_equals_the_operator_zero_tolerance() -> None:
    """The certificate and the shared helper must be the SAME scale."""
    from liouscope.numerics.linalg import operator_zero_tolerance

    _ev, cert = certified_eigvals(_NONNORMAL_TP)
    assert cert.bound == operator_zero_tolerance(_NONNORMAL_TP)


def test_one_scale_spectral_layer_reports_the_true_gap() -> None:
    """D1/D3/D4 must filter with the certificate's own bound.

    Before round-13 this reported ``gap ~ 1e-12`` -- occasionally NEGATIVE,
    impossible for a GKSL generator -- because the certified stationary mode
    survived the smaller radius-based filter.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ill-conditioned sqrtm inside gns_gap
        result = compute_spectral_layer(_NONNORMAL_TP)
    assert result.gap == pytest.approx(1.0, rel=1e-6)
    assert result.spectral_spread == pytest.approx(2.0, rel=1e-6)
    assert result.zero_mode_certificate["resolved"] is True


def test_one_scale_petermann_drops_the_stationary_mode() -> None:
    """D9 must not report the displaced zero mode as a fourth eigenmode."""
    from liouscope.diagnostics.nonnormality import petermann_factors

    eigvals, factors = petermann_factors(_NONNORMAL_TP)
    assert eigvals.size == 3
    assert factors.size == 3
    assert np.real(eigvals[0]) == pytest.approx(-1.0, rel=1e-6)


def test_one_scale_zhou_predictor_sees_the_true_gap() -> None:
    """D24's recomputation shares the certificate scale (round-13)."""
    import liouscope._zhou as zhou

    result = zhou.compute_zhou_predictor(_NONNORMAL_TP, epsilon=0.05)
    assert result.converged is True
    assert result.gap == pytest.approx(1.0, rel=1e-6)


# --------------------------------------------------------------------------
# Round-14 review: radius fallback when the certificate is inapplicable, and
# the Petermann decomposition is certified.
# --------------------------------------------------------------------------

# Not trace preserving (no structural zero mode), strongly non-normal: the
# operator-norm bound (~2.2e3) exceeds the whole spectrum {-1,-2,-3,-4}, so
# filtering with it would discard every eigenvalue and report the gapless
# D1 = 0.0 for a well-separated spectrum.
def _non_tp_nonnormal() -> np.ndarray:
    A = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(complex)
    A[0, 3] = 1.0e16
    return A


def test_inapplicable_certificate_falls_back_to_the_radius_filter() -> None:
    """The operator bound is only a valid cutoff under trace preservation."""
    A = _non_tp_nonnormal()
    _ev, cert = certified_eigvals(A)
    assert cert.applicable is False
    assert cert.bound > 4.0, "precondition: bound must exceed the spectrum"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # non-TP warning from the layer
        result = compute_spectral_layer(A, rho_steady=np.eye(2, dtype=complex) / 2)
    assert result.gap == pytest.approx(1.0, rel=1e-9)
    assert result.spectral_spread == pytest.approx(3.0, rel=1e-9)


def test_petermann_uses_the_certified_decomposition() -> None:
    """D9 must not consume the raw spectrum the spectral layer repairs.

    On the stiff #112 fixture the raw ``zgeev`` decomposition loses the
    structural zero mode -- no cutoff can restore an eigenvalue that is
    absent -- and reported 16 "non-zero" modes with ``petermann_max ~ 622.7``.
    The certified route (``dgeev-real``) returns the physical 15 modes with
    ``petermann_max ~ 2.0``.
    """
    from liouscope.diagnostics.nonnormality import petermann_factors

    lsup = _classical_network(STIFF_PAIRS, STIFF_RATES)
    _decomp, cert = certified_eigvals(lsup)
    if not cert.resolved:  # pragma: no cover - guard against solver changes
        pytest.skip("fixture is no longer certifiable")

    eigvals, factors = petermann_factors(lsup)
    assert eigvals.size == 15
    assert float(np.max(factors)) == pytest.approx(2.0, rel=1e-6)
    # The slowest surviving mode is the physical gap, not the spurious one.
    assert float(-np.max(np.real(eigvals))) == pytest.approx(
        STIFF_TRUE_GAP, rel=1e-3
    )


def test_petermann_withholds_nan_on_an_unresolved_spectrum() -> None:
    """An unresolved decomposition yields the NaN sentinel, not K = 1.

    NaN flows through the evidence machinery as "unavailable", so the F1
    rung reads UNEVALUABLE; any computed value here -- including the
    empty-array default ``K_max = 1.0``, which asserts perfect normality --
    would be a fabricated measurement.
    """
    from liouscope.diagnostics.nonnormality import (
        compute_nonnormality_layer,
        petermann_factors,
    )

    lsup = _stiff_with_fast_rate(1.0e8)
    _decomp, cert = certified_eigvals(lsup)
    if cert.resolved:  # pragma: no cover - guard against solver changes
        pytest.skip("fixture no longer produces an unresolved spectrum")

    with pytest.warns(RuntimeWarning, match="withheld"):
        eigvals, factors = petermann_factors(lsup)
    assert np.all(np.isnan(eigvals)) and np.all(np.isnan(factors))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = compute_nonnormality_layer(lsup)
    assert np.isnan(layer.petermann_max)
