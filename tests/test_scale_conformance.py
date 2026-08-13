"""Issue #101 slice B: rate-unit conformance suite for scale-relative diagnostics.

Contract under a positive unit rescale ``L -> cL`` for
``c in {1e-10, 1e-5, 1, 1e5, 1e10}``:

* DIMENSIONLESS diagnostics (D8b ``henrici_relative``, D10b ``kreiss_scaled``,
  ``resolvent_peak_scaled``, ``ridge_fwhm_rel``, ``pseudospectral_radius_rel``,
  ``pseudospectral_abscissa_rel``) are invariant within declared tolerance;
* RATE-VALUED outputs (``rate_scale``, ``pseudospectral_abscissa``) scale by
  ``c``;
* D14 agrees after ``t -> t/c`` (metamorphic propagator identity
  ``exp((cL)(t/c)) = exp(Lt)``);
* zero, normal, gapless-normal and ordinary non-normal negatives behave as the
  documented oracles demand (normal -> ~0, Jordan -> positive, zero -> defined
  semantics, NaN/inf -> located error);
* values are invariant under unitary similarity ``L -> W L W^H`` (the basis
  transformations that preserve the Hilbert-Schmidt/2-norm geometry these
  diagnostics are defined in).

The legacy (absolute-grid) fields are exercised elsewhere and deliberately NOT
asserted scale-invariant here -- they are documented as unit-dependent.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope.diagnostics.classification import ADVISORY_EVIDENCE_KEYS
from liouscope.diagnostics.nonnormality import (
    HENRICI_REL_CLIP_TOL,
    henrici_relative,
    kreiss_grid_lower_bound,
)
from liouscope.diagnostics.resolvent import compute_resolvent_layer
from liouscope.diagnostics.transient import trans_amplitude_ratio
from liouscope.numerics.pseudospec import pseudospectrum_extent
from liouscope.numerics.scale import rate_scale

C_SET = (1.0e-10, 1.0e-5, 1.0, 1.0e5, 1.0e10)

# Rounding-level tolerance for exactly-invariant constructions: every grid
# coordinate is built dimensionlessly and re-dimensionalised per point, so the
# only drift is floating-point (verified ~1e-13 across 20 decades).
RTOL_INV = 1.0e-8
ATOL_INV = 1.0e-12


def _rabi_dephasing_L() -> np.ndarray:
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    return build_liouvillian(0.5 * X, [Z], [0.3])


def _amplitude_damping_L() -> np.ndarray:
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [0.7])


def _random_unitary(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(A)
    return np.asarray(Q * (np.diag(R) / np.abs(np.diag(R))))


# ---------------------------------------------------------------------------
# rate_scale
# ---------------------------------------------------------------------------


def test_rate_scale_is_frobenius_and_homogeneous():
    L = _rabi_dephasing_L()
    base = rate_scale(L)
    assert base == pytest.approx(float(np.linalg.norm(L, ord="fro")), rel=0.0)
    for c in C_SET:
        np.testing.assert_allclose(rate_scale(c * L), c * base, rtol=RTOL_INV)


def test_rate_scale_zero_operator_semantics():
    assert rate_scale(np.zeros((4, 4), dtype=complex)) == 0.0


def test_rate_scale_fails_closed_on_nonfinite_and_nonsquare():
    bad = np.eye(3, dtype=complex)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        rate_scale(bad)
    with pytest.raises(ValueError):
        rate_scale(np.zeros((2, 3), dtype=complex))


def test_rate_scale_unitary_similarity_invariant(rng):
    L = _rabi_dephasing_L()
    W = _random_unitary(4, rng)
    np.testing.assert_allclose(
        rate_scale(W @ L @ W.conj().T), rate_scale(L), rtol=1.0e-12
    )


# ---------------------------------------------------------------------------
# D8b henrici_relative
# ---------------------------------------------------------------------------


def test_henrici_relative_invariant_under_rate_rescale():
    for L in (_rabi_dephasing_L(), _amplitude_damping_L()):
        base = henrici_relative(L)
        assert 0.0 <= base <= 1.0
        for c in C_SET:
            np.testing.assert_allclose(
                henrici_relative(c * L), base, rtol=RTOL_INV, atol=ATOL_INV
            )


def test_henrici_relative_oracles():
    # Normal (diagonal) matrix -> exactly no departure from normality.
    assert henrici_relative(np.diag([-1.0, -2.0, -3.0]).astype(complex)) == 0.0
    # Normal with complex spectrum (skew part) is still normal.
    normal_c = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    assert henrici_relative(normal_c) <= 10.0 * HENRICI_REL_CLIP_TOL
    # Jordan / strictly-upper mass -> strictly positive, close to 1 when the
    # off-diagonal dominates the Frobenius mass.
    J = np.array([[-1.0, 50.0], [0.0, -1.0]], dtype=complex)
    rel = henrici_relative(J)
    assert 0.9 < rel <= 1.0
    # Nilpotent (all mass off-diagonal) -> exactly 1.
    N = np.array([[0.0, 3.0], [0.0, 0.0]], dtype=complex)
    assert henrici_relative(N) == pytest.approx(1.0, rel=1.0e-12)
    # Zero operator -> documented 0.0 (trivially normal, no scale).
    assert henrici_relative(np.zeros((3, 3), dtype=complex)) == 0.0


def test_henrici_relative_fails_closed_on_nonfinite():
    bad = np.eye(2, dtype=complex)
    bad[0, 1] = np.inf
    with pytest.raises(ValueError):
        henrici_relative(bad)


def test_henrici_relative_unitary_similarity_invariant(rng):
    L = _rabi_dephasing_L()
    W = _random_unitary(4, rng)
    np.testing.assert_allclose(
        henrici_relative(W @ L @ W.conj().T), henrici_relative(L), rtol=1.0e-9
    )


# ---------------------------------------------------------------------------
# D10b kreiss_grid_lower_bound
# ---------------------------------------------------------------------------


def test_kreiss_scaled_invariant_under_rate_rescale():
    L = _rabi_dephasing_L()
    base = kreiss_grid_lower_bound(L)
    assert base.value > 0.0
    assert base.value >= base.coarse_value > 0.0
    for c in C_SET:
        est = kreiss_grid_lower_bound(c * L)
        np.testing.assert_allclose(est.value, base.value, rtol=RTOL_INV)
        # Grid metadata is dimensionless and must agree exactly.
        assert est.sigma_rel_lo == base.sigma_rel_lo
        assert est.sigma_rel_hi == pytest.approx(base.sigma_rel_hi, rel=RTOL_INV)
        assert est.omega_rel_max == pytest.approx(base.omega_rel_max, rel=RTOL_INV)
        assert est.edge_maximizer == base.edge_maximizer
        # The rate scale itself is rate-valued: scales by c.
        np.testing.assert_allclose(est.rate_scale, c * base.rate_scale, rtol=RTOL_INV)


def test_kreiss_scaled_zero_operator_is_exactly_one():
    est = kreiss_grid_lower_bound(np.zeros((4, 4), dtype=complex))
    assert est.value == 1.0
    assert est.rate_scale == 0.0
    assert not est.edge_maximizer
    assert np.isnan(est.sigma_rel_lo)


def test_kreiss_scaled_nonnormal_exceeds_normal():
    # Strong Jordan coupling drives real transient amplification; the grid
    # lower bound must see it, while a normal stable matrix stays near 1.
    J = np.array([[-1.0, 50.0], [0.0, -1.0]], dtype=complex)
    D = np.diag([-1.0, -2.0]).astype(complex)
    est_j = kreiss_grid_lower_bound(J)
    est_d = kreiss_grid_lower_bound(D)
    assert est_j.value > 5.0
    assert est_d.value <= 1.0 + 1.0e-9
    # Refinement never loses the coarse bound.
    assert est_j.value >= est_j.coarse_value
    assert est_j.refinement_delta >= 0.0


def test_kreiss_scaled_parameter_guards():
    L = _rabi_dephasing_L()
    with pytest.raises(ValueError):
        kreiss_grid_lower_bound(L, n_sigma=1)
    with pytest.raises(ValueError):
        kreiss_grid_lower_bound(L, n_omega=0)
    with pytest.raises(ValueError):
        kreiss_grid_lower_bound(L, refine_levels=-1)
    bad = np.eye(4, dtype=complex)
    bad[2, 2] = np.nan
    with pytest.raises(ValueError):
        kreiss_grid_lower_bound(bad)


def test_kreiss_scaled_unitary_similarity_invariant(rng):
    L = _rabi_dephasing_L()
    W = _random_unitary(4, rng)
    est_w = kreiss_grid_lower_bound(W @ L @ W.conj().T)
    est = kreiss_grid_lower_bound(L)
    np.testing.assert_allclose(est_w.value, est.value, rtol=1.0e-8)


# ---------------------------------------------------------------------------
# Scale-relative resolvent layer (D11b/D12/D13 + abscissa)
# ---------------------------------------------------------------------------


def test_resolvent_layer_scaled_fields_invariant_under_rate_rescale():
    for L in (_rabi_dephasing_L(), _amplitude_damping_L()):
        base = compute_resolvent_layer(L)
        for c in C_SET:
            r = compute_resolvent_layer(c * L)
            np.testing.assert_allclose(
                r.resolvent_peak_scaled,
                base.resolvent_peak_scaled,
                rtol=RTOL_INV,
            )
            np.testing.assert_allclose(
                r.ridge_fwhm_rel, base.ridge_fwhm_rel, rtol=RTOL_INV, atol=ATOL_INV
            )
            np.testing.assert_allclose(
                r.pseudospectral_radius_rel,
                base.pseudospectral_radius_rel,
                rtol=RTOL_INV,
                atol=ATOL_INV,
            )
            np.testing.assert_allclose(
                r.pseudospectral_abscissa_rel,
                base.pseudospectral_abscissa_rel,
                rtol=RTOL_INV,
                atol=ATOL_INV,
            )
            # Rate-valued outputs scale by c.
            np.testing.assert_allclose(r.rate_scale, c * base.rate_scale, rtol=RTOL_INV)
            np.testing.assert_allclose(
                r.pseudospectral_abscissa,
                c * base.pseudospectral_abscissa,
                rtol=RTOL_INV,
                atol=ATOL_INV,
            )


def test_resolvent_layer_legacy_fields_unchanged_by_new_kwargs():
    # The legacy fields must be computed from the legacy absolute parameters
    # only -- passing different *_rel values must not perturb them.
    L = _rabi_dephasing_L()
    a = compute_resolvent_layer(L)
    b = compute_resolvent_layer(L, sigma_rel=7.0e-3, pseudo_eps_rel=5.0e-3)
    assert a.resolvent_peak == b.resolvent_peak
    assert a.ridge_fwhm == b.ridge_fwhm
    assert a.pseudospectral_radius == b.pseudospectral_radius
    assert a.pseudospec_eps == b.pseudospec_eps


def test_resolvent_layer_zero_operator_scaled_fields_are_nan():
    r = compute_resolvent_layer(np.zeros((4, 4), dtype=complex))
    assert r.rate_scale == 0.0
    for val in (
        r.resolvent_peak_scaled,
        r.ridge_fwhm_rel,
        r.pseudospectral_radius_rel,
        r.pseudospectral_abscissa,
        r.pseudospectral_abscissa_rel,
    ):
        assert np.isnan(val)


def test_pseudospectrum_extent_oracle_and_underresolved_marker():
    L = np.diag([0.0, -1.0, -2.0]).astype(complex)
    scale = rate_scale(L)
    span = 3.0
    grid = (-2.5, 0.5, 61)
    grid_im = (-0.5, 0.5, 21)
    radius, abscissa = pseudospectrum_extent(
        L, 1.0e-3 * scale, grid_re=grid, grid_im=grid_im
    )
    # The eps-pseudospectrum surrounds each eigenvalue; the max-modulus point
    # sits near the fastest mode, the abscissa near the slowest (z ~ 0).
    assert radius == pytest.approx(2.0, abs=0.1), f"span={span}"
    assert abscissa == pytest.approx(0.0, abs=0.1)
    # A hopeless eps with a grid that misses every pseudospectrum point must
    # return the NaN "under-resolved" marker, not a fake 0.0.
    radius_nan, abscissa_nan = pseudospectrum_extent(
        L, 1.0e-15, grid_re=(-1.7, -1.3, 5), grid_im=(0.2, 0.4, 3)
    )
    assert np.isnan(radius_nan) and np.isnan(abscissa_nan)


def test_resolvent_peak_curve_scaled_guards():
    from liouscope.diagnostics.resolvent import resolvent_peak_curve_scaled

    L = _rabi_dephasing_L()
    with pytest.raises(ValueError, match="sigma_rel"):
        resolvent_peak_curve_scaled(L, 0.0)
    with pytest.raises(ValueError, match="zero operator"):
        resolvent_peak_curve_scaled(np.zeros((4, 4), dtype=complex))


def test_pseudospectrum_extent_fails_closed_on_nonfinite():
    bad = np.eye(2, dtype=complex) * np.nan
    with pytest.raises(ValueError):
        pseudospectrum_extent(bad, 1.0e-3, grid_re=(-1, 1, 3), grid_im=(-1, 1, 3))


# ---------------------------------------------------------------------------
# D14 metamorphic rate-rescale agreement
# ---------------------------------------------------------------------------


def test_d14_transient_ratio_agrees_after_time_rescale():
    L = _rabi_dephasing_L()
    t = np.linspace(0.0, 8.0, 33)
    base = trans_amplitude_ratio(L, t_grid=t)
    for c in (1.0e-5, 1.0e-3, 1.0e3, 1.0e5):
        rescaled = trans_amplitude_ratio(c * L, t_grid=t / c)
        np.testing.assert_allclose(rescaled, base, rtol=1.0e-9)


# ---------------------------------------------------------------------------
# Gapless-normal negative control + diagnose() integration
# ---------------------------------------------------------------------------


def test_gapless_normal_control_shows_no_nonnormality():
    # Pure dephasing WITHOUT Hamiltonian: normal generator with conserved
    # populations (two zero modes -- the gapless/degenerate-manifold control
    # from the #101 addendum). The scale-relative non-normality evidence must
    # stay at the normal-operator floor: a missing gap alone is NOT phantom
    # evidence.
    Z = np.diag([1.0, -1.0]).astype(complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [Z], [0.5])
    assert henrici_relative(L) <= 10.0 * HENRICI_REL_CLIP_TOL
    est = kreiss_grid_lower_bound(L)
    assert est.value <= 1.0 + 1.0e-9


def test_diagnose_surfaces_scale_relative_advisory_evidence(pauli):
    H = 0.5 * pauli["X"]
    L = build_liouvillian(H, [pauli["Z"]], [0.3])
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    nn = report.nonnorm
    rv = report.resolvent
    assert 0.0 <= nn.henrici_relative <= 1.0
    assert np.isfinite(nn.kreiss_scaled) and nn.kreiss_scaled > 0.0
    assert np.isfinite(rv.resolvent_peak_scaled)
    assert np.isfinite(rv.pseudospectral_radius_rel)
    assert rv.rate_scale > 0.0
    ev = report.classification.evidence
    for key in (
        "henrici_relative",
        "kreiss_scaled",
        "pseudospectral_radius_rel",
        "pseudospectral_abscissa_rel",
    ):
        assert key in ev
        assert key in ADVISORY_EVIDENCE_KEYS
