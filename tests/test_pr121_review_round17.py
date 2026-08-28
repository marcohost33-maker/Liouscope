"""Round-17 review of PR #121: one zero-mode cutoff, obtained one way.

The a posteriori refinement (issue #113 second axis) can rescue a genuine slow
mode from the raw operator band. Four consumer layers kept filtering with
``ZeroModeCertificate.bound`` afterwards and threw that mode away again. Each
test below pins ONE of those sites, and each is paired with a positive control
so it cannot pass by refusing everything.

The guard at the end is the structural half: it fails when any NEW site starts
reading the raw band, which is what made a four-fold miss possible in the first
place.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope._zhou import compute_zhou_predictor
from liouscope.diagnostics import nonnormality as nn
from liouscope.diagnostics.mpemba import expansion_alpha, overlap_c1
from liouscope.numerics.linalg import certified_eigvals

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _refined_generator() -> np.ndarray:
    """Two-level system whose slowest decay (1e-15) sits INSIDE the raw band.

    ``||L||`` is set by the Hamiltonian (omega = 1), so the operator band is
    ``2.2e-13`` while the genuine slow mode is at ``1e-15``. The a posteriori
    certificate proves that mode non-stationary, and the applied tolerance
    drops to ``5e-16`` -- 444x below the raw band.
    """
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    return build_liouvillian(H, [lowering, sigma_z], [1.0e-15, 1.0e-14])


_D11_PAIRS = [(2, 0), (0, 3), (3, 1), (1, 3), (0, 2), (1, 0), (2, 1)]
_D11_RATES = [4.452e-05, 1.005e08, 2.452e-06, 6.823e-06, 4.239e-06, 4.307e-05, 1.0e-11]


def _d11_fallback_generator() -> np.ndarray:
    """Stiff classical network: eigenvalues resolve WITH a refinement, vectors do not.

    That combination is what puts D11 on its own fallback path: the D9
    eigenvector gate withholds the Petermann factors, so the progression scan
    is recomputed from the certified spectrum -- and it is exactly there that
    the raw band discarded the rescued modes again.
    """
    jumps = []
    for to, frm in _D11_PAIRS:
        j = np.zeros((4, 4), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, _D11_RATES)


def _rescued_modes(L: np.ndarray) -> tuple[np.ndarray, object]:
    """``(|lambda| the refinement rescued, certificate)`` -- the positive control."""
    ev, cert = certified_eigvals(L)
    mag = np.abs(ev)
    return np.sort(mag[(mag > cert.applied_tolerance) & (mag <= cert.bound)]), cert


# --------------------------------------------------------------------------
# src/liouscope/_zhou.py -- D24 mixing-time window
# --------------------------------------------------------------------------


def test_d24_reads_the_refined_gap_not_the_raw_band() -> None:
    L = _refined_generator()
    rescued, cert = _rescued_modes(L)
    assert cert.resolved, "precondition: the certificate must resolve"
    assert cert.applied_tolerance < cert.bound, "precondition: a refinement happened"
    assert rescued.size == 1, "precondition: exactly one mode was rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_zhou_predictor(L)

    # Filtering by ``bound`` discards the 1e-15 mode and D24 reads the next,
    # faster one at 2.05e-14 -- a mixing-time window ~20x too short.
    assert result.gap == pytest.approx(1.0e-15, rel=1.0e-6)
    assert result.mixing_time_lower > 1.0e15


def test_d24_is_unchanged_on_a_generator_with_nothing_to_rescue() -> None:
    """Positive control: the refinement is inert on a healthy generator."""
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(H, [lowering], [0.5])
    _rescued, cert = _rescued_modes(L)
    assert cert.applied_tolerance == pytest.approx(cert.bound)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_zhou_predictor(L)
    assert result.gap == pytest.approx(0.25, rel=1.0e-9)


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/mpemba.py -- D19 overlap and the mode expansion
# --------------------------------------------------------------------------

_RHO_0 = np.array([[0.9, 0.0], [0.0, 0.1]], dtype=complex)


def test_d19_overlap_survives_the_refinement() -> None:
    """``overlap_c1 == 0.0`` is the FALSE A11/F4 trigger, on the top rung."""
    L = _refined_generator()
    rescued, _cert = _rescued_modes(L)
    assert rescued.size == 1, "precondition: exactly one mode was rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = overlap_c1(L, _RHO_0)

    assert c1 > 0.0
    assert c1 == pytest.approx(0.14142135623730953, rel=1.0e-9)


def test_expansion_alpha_uses_the_refined_zero_set() -> None:
    L = _refined_generator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = expansion_alpha(L, _RHO_0)

    # With the raw band the rescued mode is gone, the expansion is fitted on
    # the remaining fast pair and the slope collapses to ~-1e-15.
    assert alpha < -1.0
    assert alpha == pytest.approx(-33.560770643553646, rel=1.0e-6)


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/nonnormality.py -- D9 Petermann factors
# --------------------------------------------------------------------------


def test_petermann_keeps_the_refined_slow_mode() -> None:
    L = _refined_generator()
    rescued, _cert = _rescued_modes(L)
    assert rescued.size == 1, "precondition: exactly one mode was rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigvals_filt, factors = nn.petermann_factors(L)

    # The raw band leaves 2 modes with petermann_max 1.0; the rescued mode is
    # the one whose conditioning can dominate the F1 input.
    assert eigvals_filt.size == 3
    assert float(np.min(np.abs(eigvals_filt))) == pytest.approx(1.0e-15, rel=1.0e-6)
    assert float(np.max(factors)) == pytest.approx(2.0, rel=1.0e-9)


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/nonnormality.py -- D11 fallback (D9 withheld)
# --------------------------------------------------------------------------


def test_d11_fallback_scans_the_refined_zero_set(monkeypatch) -> None:
    """The fallback must hand the progression scan the REFINED complement.

    Asserted at the seam the review names: what reaches
    ``bohr_arithmetic_progression``. The measured progression length happens
    to be 1 for this (Hamiltonian-free) network either way, so an assertion on
    the reported number alone would be blind here.
    """
    L = _d11_fallback_generator()
    rescued, cert = _rescued_modes(L)
    assert cert.resolved, "precondition: the eigenvalue certificate resolves"
    assert cert.applied_tolerance < cert.bound, "precondition: a refinement happened"
    assert rescued.size == 3, "precondition: three modes were rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ev, factors = nn.petermann_factors(L)
    assert bool(np.all(np.isnan(factors))), "precondition: D9 is withheld here"

    seen: list[np.ndarray] = []
    original = nn.bohr_arithmetic_progression

    def _spy(eigvals: np.ndarray, d: int) -> tuple[float, float]:
        seen.append(np.asarray(eigvals).copy())
        return original(eigvals, d)

    monkeypatch.setattr(nn, "bohr_arithmetic_progression", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn.compute_nonnormality_layer(L)

    assert len(seen) == 1
    scanned = np.abs(seen[0])
    ev_all, _c = certified_eigvals(L)
    expected = int(np.count_nonzero(np.abs(ev_all) > cert.applied_tolerance))
    assert scanned.size == expected
    # Every rescued mode is still in the scan; the raw band would drop all three.
    for mode in rescued:
        assert np.any(np.isclose(scanned, mode, rtol=1.0e-9))


# --------------------------------------------------------------------------
# Structural guard: no NEW site may filter on the raw band
# --------------------------------------------------------------------------

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "liouscope"
_BOUND = re.compile(r"\b(?:cert|certificate|self)\.bound\b")

#: The ONLY places allowed to read the raw band: three warning texts that
#: label it "bound", and the two accessors inside the certificate itself.
_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    (
        "diagnostics/mpemba.py",
        'f"{certificate.residual:.3e}, bound {certificate.bound:.3e}). "',
    ),
    (
        "diagnostics/nonnormality.py",
        'f"{certificate.residual:.3e}, bound {certificate.bound:.3e}); "',
    ),
    (
        "diagnostics/spectral.py",
        'f"{certificate.bound:.3e}). The generator is too stiff for a dense "',
    ),
    (
        "numerics/linalg.py",
        "return self.bound if not np.isfinite(self.zero_tolerance) "
        "else self.zero_tolerance",
    ),
    ("numerics/linalg.py", '"bound": _f(self.bound),'),
})


def _bound_readers() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _BOUND.search(stripped):
                found.add((rel, stripped))
    return found


def test_only_report_sites_read_the_raw_certificate_bound() -> None:
    """A fifth filter site must fail here, not in a review six rounds later."""
    found = _bound_readers()
    # Positive control: a guard that finds nothing proves nothing.
    assert len(found) >= len(_ALLOWED) > 0
    assert found >= _ALLOWED, (
        "the allowlist has rotted -- these entries no longer exist: "
        f"{sorted(_ALLOWED - found)}"
    )
    assert found <= _ALLOWED, (
        "a new site reads the RAW certificate band; filters must use "
        f"ZeroModeCertificate.zero_set_tolerance(): {sorted(found - _ALLOWED)}"
    )


def test_the_certificate_exposes_one_way_to_get_a_filter_cutoff() -> None:
    """The applicable/inapplicable branch and the refinement live in one place."""
    L = _refined_generator()
    ev, cert = certified_eigvals(L)
    assert cert.zero_set_tolerance(ev) == pytest.approx(cert.applied_tolerance)
    assert cert.zero_set_tolerance(ev, atol=1.0e-9) == pytest.approx(1.0e-9)
    assert cert.zero_set_tolerance(ev) < cert.bound


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/spectral.py -- D1 must not be published fail-open
# --------------------------------------------------------------------------


def _forced_certificate(**kw):
    """A ``certified_eigvals`` stand-in with a caller-chosen certificate.

    A real all-routes-fail solver outcome cannot be produced on demand, and
    the existing suite already exercises this reporting path the same way
    (``test_spectral_certificate.py``). Patched by string target so the module
    is not imported twice under different styles.
    """
    from liouscope.numerics.linalg import ZeroModeCertificate, eig_nonhermitian

    def _fake(l_super, **_kw):
        return (
            eig_nonhermitian(l_super).eigenvalues,
            ZeroModeCertificate(**kw),
        )

    return _fake


def _two_level_decay() -> tuple[np.ndarray, np.ndarray]:
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])
    return L, np.array([[1, 0], [0, 0]], dtype=complex)


def test_d1_is_withheld_when_no_repair_route_certifies(monkeypatch) -> None:
    """An applicable certificate with ``certified=False`` must not publish a gap."""
    from liouscope.diagnostics.spectral import compute_spectral_layer

    L, rho_ss = _two_level_decay()
    monkeypatch.setattr(
        "liouscope.diagnostics.spectral.certified_eigvals",
        _forced_certificate(
            applicable=True, certified=False, solver="zgeev",
            residual=1.0, bound=0.0, trace_defect=0.0,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = compute_spectral_layer(L, rho_ss)

    assert np.isnan(layer.gap), (
        "a gap read off a spectrum the certificate calls untrustworthy was "
        "published to every caller of SpectralResult.gap"
    )
    assert layer.zero_mode_certificate["certified"] is False
    # The warning must still fire -- withholding replaces the silent number,
    # it does not replace the explanation.
    assert any("issue #112" in str(w.message) for w in caught)


def test_d1_is_still_published_for_a_resolved_certificate(monkeypatch) -> None:
    """Positive control: the withholding keys on the certificate, not on the run."""
    from liouscope.diagnostics.spectral import compute_spectral_layer

    L, rho_ss = _two_level_decay()
    layer = compute_spectral_layer(L, rho_ss)
    assert layer.zero_mode_certificate["resolved"] is True
    assert np.isfinite(layer.gap)
    assert layer.gap == pytest.approx(0.5, rel=1.0e-9)


# --------------------------------------------------------------------------
# src/liouscope/fitting/gls.py -- a saturated fit must not be selectable
# --------------------------------------------------------------------------


def _decay_generator() -> np.ndarray:
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])


def _saturating_fit_patch(monkeypatch, *, only: str | None):
    """Make ``fit_gls_ar1`` report a saturated fit with an IRRESISTIBLE likelihood.

    A saturated fit is convergence for the wrong reason: the magnitude guards
    return a constant, every derivative vanishes and ``least_squares`` stops on
    "gradient is small". The pre-fix code flipped ``success`` and nothing else,
    so such a fit still produced a finite AICc and could win the hierarchy.
    """
    import dataclasses

    from liouscope.diagnostics import relaxation as rx

    real = rx.fit_gls_ar1
    names = {rx.M0: "M0", rx.M1: "M1", rx.M2: "M2", rx.M3a: "M3a", rx.M3b: "M3b"}

    def _fake(model, t, y, p0, **kw):
        out = real(model, t, y, p0, **kw)
        if only is None or names.get(model) == only:
            return dataclasses.replace(
                out, success=False, saturated=("magnitude",), log_likelihood=1.0e6
            )
        return out

    monkeypatch.setattr(rx, "fit_gls_ar1", _fake)
    return rx


def test_a_saturated_fit_cannot_win_the_model_hierarchy(monkeypatch) -> None:
    rx = _saturating_fit_patch(monkeypatch, only="M0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rx.compute_relaxation_layer(_decay_generator(), bootstrap_B=20)

    assert result.fits["M0"].success is False
    # log_likelihood 1e6 would give M0 by far the smallest AICc.
    assert not np.isfinite(result.fits["M0"].aicc)
    assert result.aicc_model != "M0"
    assert np.isfinite(result.beta_D)


def test_all_models_saturated_reports_no_winner(monkeypatch) -> None:
    rx = _saturating_fit_patch(monkeypatch, only=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rx.compute_relaxation_layer(_decay_generator(), bootstrap_B=20)

    # Pre-fix: choose_model fell back to "M0" and beta_D was read off a fit
    # the guard had just declared a non-result.
    assert result.aicc_model == "none"
    assert np.isnan(result.beta_D)
    assert all(np.isnan(v) for v in result.bca_ci_beta)


def test_relaxation_layer_is_unchanged_when_nothing_saturates() -> None:
    """Positive control: the rule keys on saturation, not on the generator."""
    from liouscope.diagnostics.relaxation import compute_relaxation_layer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_relaxation_layer(_decay_generator(), bootstrap_B=20)
    assert result.aicc_model in {"M0", "M1", "M2", "M3a", "M3b"}
    assert np.isfinite(result.fits[result.aicc_model].aicc)
    assert np.isfinite(result.beta_D)


def test_bootstrap_refuses_a_saturated_base_fit() -> None:
    """A resample around a non-estimate is not an uncertainty."""
    from liouscope.fitting.bootstrap import parametric_bootstrap
    from liouscope.fitting.models import M0

    t = np.linspace(0.0, 1.0e10, 64)
    y = np.exp(-5.0 * t / 1.0e10)
    # The refusal is asserted, not merely raised: without the guard the run
    # continues and emits the retained-replicate warning instead, and a test
    # that dies of a WARNING proves nothing about the guard.
    raised: RuntimeError | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parametric_bootstrap(M0, t, y, np.array([1.0, -1.0]), B=5)
        except RuntimeError as exc:
            raised = exc
    assert raised is not None, (
        "a bootstrap was resampled around a base fit that ended on the "
        "magnitude plateau"
    )
    assert "did not converge" in str(raised)

    # Positive control: the same grid with a well-posed seed still bootstraps.
    samples, theta_hat = parametric_bootstrap(
        M0, t, y, np.array([1.0, 5.0e-10]), B=5
    )
    assert samples.shape == (5, 2)
    assert np.all(np.isfinite(theta_hat))


# --------------------------------------------------------------------------
# src/liouscope/numerics/linalg.py -- the repair ladder must survive the primary
# --------------------------------------------------------------------------

_STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
_STIFF_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]


def _real_classical_network() -> np.ndarray:
    jumps = []
    for to, frm in _STIFF_PAIRS:
        j = np.zeros((4, 4), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, _STIFF_RATES)


def _raise_nonconvergence(*_a, **_kw):
    raise np.linalg.LinAlgError("eig algorithm (zgeev) did not converge")


def test_eigvals_ladder_continues_past_a_primary_nonconvergence(monkeypatch) -> None:
    from liouscope.numerics import linalg as la

    L = _real_classical_network()
    monkeypatch.setattr(la, "eig_nonhermitian", _raise_nonconvergence)

    # The ladder failure is asserted, not crashed into: an uncaught
    # LinAlgError would kill the test without saying which mutation did it.
    ended: Exception | None = None
    ev = cert = None
    try:
        ev, cert = la.certified_eigvals(L)
    except np.linalg.LinAlgError as exc:
        ended = exc
    assert ended is None, f"the primary solve ended the repair ladder: {ended!r}"
    assert cert is not None and ev is not None
    assert cert.applicable is True
    assert cert.certified is True, "a later repair route certified this spectrum"
    assert cert.solver != "zgeev"
    assert ev.size == 16


def test_eig_ladder_continues_past_a_primary_nonconvergence(monkeypatch) -> None:
    from liouscope.numerics import linalg as la

    L = _real_classical_network()
    monkeypatch.setattr(la, "eig_nonhermitian", _raise_nonconvergence)

    ended: Exception | None = None
    decomp = cert = None
    try:
        decomp, cert = la.certified_eig(L)
    except np.linalg.LinAlgError as exc:
        ended = exc
    assert ended is None, f"the primary solve ended the repair ladder: {ended!r}"
    assert cert is not None and decomp is not None
    assert cert.solver == "dgeev-real"
    assert cert.certified is True
    assert decomp.left_vectors is not None


def test_a_total_solver_failure_still_raises(monkeypatch) -> None:
    """Fail-closed control: with no route left the original error surfaces."""
    from liouscope.numerics import linalg as la

    # A complex generator has no ``dgeev-real`` route, so ``certified_eig``
    # has exactly one candidate -- and it is the one that raises.
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(H, [lowering], [0.5])
    assert np.any(np.asarray(L, dtype=complex).imag)

    monkeypatch.setattr(la, "eig_nonhermitian", _raise_nonconvergence)
    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        la.certified_eig(L)


# --------------------------------------------------------------------------
# src/liouscope/_types.py -- the field documentation must match the contract
# --------------------------------------------------------------------------


def _certificate_field_comment() -> str:
    text = (_SRC / "_types.py").read_text(encoding="utf-8").splitlines()
    idx = next(
        i for i, ln in enumerate(text)
        if ln.strip().startswith("zero_mode_certificate:")
    )
    block: list[str] = []
    for line in reversed(text[:idx]):
        if not line.strip().startswith("#"):
            break
        block.append(line.strip())
    return "\n".join(reversed(block))


def test_certificate_field_documents_its_classifier_effect() -> None:
    block = _certificate_field_comment()
    assert block, "guard is vacuous if the comment block was not found"
    # The false CLAIM, not the word: the field does enter a verdict.
    assert "does not enter any verdict" not in block.lower()
    assert "load-bearing" in block.lower()
    assert "_apply_spectral_certificate_floor" in block

    # Positive control: the described contract must actually be in the code.
    classification = (_SRC / "diagnostics" / "classification.py").read_text(
        encoding="utf-8"
    )
    assert "verdict, tier = _apply_spectral_certificate_floor(" in classification
    assert "certificate = getattr(spectral" in classification


def test_a_successful_fit_with_a_non_finite_aicc_stays_selectable(monkeypatch) -> None:
    """A non-finite AICc is not a failed fit -- the V4 near-miss.

    Measured on validation system V4 (thermal two-level), trace-distance
    curve: all five models converge, and all five AICc values are ``inf``
    because the Geyer-corrected ``n_eff`` of that smooth residual series is
    too small for the small-sample correction. A selection rule keyed on
    finiteness withholds D17 on a system where nothing failed.
    """
    from liouscope.diagnostics import relaxation as rx

    monkeypatch.setattr(rx, "aicc", lambda *_a, **_kw: float("inf"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = rx.compute_relaxation_layer(_decay_generator(), bootstrap_B=20)

    assert all(fr.success for fr in result.fits.values())
    assert all(not np.isfinite(fr.aicc) for fr in result.fits.values())
    assert result.aicc_model in {"M0", "M1", "M2", "M3a", "M3b"}
    assert np.isfinite(result.beta_D)
