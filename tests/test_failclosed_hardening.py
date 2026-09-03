"""Fail-closed hardening regressions (2026-07-12 full-repo review).

Each test here pins a defect found by the July 2026 deep review: paths where
malformed or non-finite input could silently *upgrade* a verdict, bypass a
gate, or fabricate a result instead of failing closed. Grouped by module.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, classify_mechanism
from liouscope._consts import (
    TIER_EXPLORATION,
    VERDICT_CONFIRMED,
    VERDICT_UNDEFINED,
)
from liouscope._types import (
    LepResult,
    MpembaResult,
    NonNormalityResult,
    RelaxationResult,
    ResolventResult,
    SpectralResult,
    TransientResult,
)
from liouscope.core.lindblad import steady_state
from liouscope.ensemble import ENSEMBLE_MPEMBA_CONFIRMED, EnsembleEvidence

_ARR = np.zeros(1, dtype=complex)


def _spectral(**kw) -> SpectralResult:
    base = {
        "gap": 0.5,
        "gns_gap": 0.5,
        "kms_gap": 0.5,
        "oscillating_gap": 0.1,
        "spectral_spread": 1.0,
        "eigenvalues": _ARR,
        "steady_state": np.zeros((1, 1), dtype=complex),
        "has_complex_pairs": False,
        "zero_mode_certificate": {"applicable": True, "certified": True, "resolved": True},
    }
    base.update(kw)
    return SpectralResult(**base)


def _nonnorm(**kw) -> NonNormalityResult:
    base = {
        "henrici_eta": 0.5,
        "petermann_max": 1.0,
        "petermann_factors": _ARR,
        "kreiss": 1.0,
        "bohr_ap_length": 1,
        "bohr_ap_pauli_bound": 0.0,
    }
    base.update(kw)
    return NonNormalityResult(**base)


def _relax(**kw) -> RelaxationResult:
    base = {
        "von_neumann_entropy": 0.0,
        "relative_entropy_curve": _ARR.real,
        "fidelity_curve": _ARR.real,
        "entanglement_asymmetry": None,
        "fits": {},
        "aicc_model": "M1",
        "beta_D": 0.5,
        "bca_ci_beta": (0.4, 0.6),
    }
    base.update(kw)
    return RelaxationResult(**base)


def _resolvent(**kw) -> ResolventResult:
    base = {
        "resolvent_peak": 1.0,
        "ridge_fwhm": 1.0,
        "pseudospectral_radius": 0.5,
        "pseudospec_eps": 1.0e-3,
    }
    base.update(kw)
    return ResolventResult(**base)


def _transient(**kw) -> TransientResult:
    base = {
        "trans_amplitude_ratio": 1.0,
        "kappa_trans": 1.0,
        "numerical_abscissa": 0.0,
    }
    base.update(kw)
    return TransientResult(**base)


def _lep(**kw) -> LepResult:
    base = {
        "lep_proximity": 1.0,
        "gap_rate_consistency": 1.0,
        "initial_state_sensitivity": 0.0,
        "lep_candidate_count": 0,
    }
    base.update(kw)
    return LepResult(**base)


def _mpemba(**kw) -> MpembaResult:
    base = {
        "overlap_c1": 1.0e-9,
        "expansion_alpha": 1.0,
        "is_mpemba_candidate": True,
        "trivial_overlap": False,
    }
    base.update(kw)
    return MpembaResult(**base)


def _classify(spectral=None, nonnorm=None, relaxation=None, resolvent=None,
              transient=None, lep=None, mpemba=None, **kw):
    return classify_mechanism(
        spectral or _spectral(),
        nonnorm or _nonnorm(),
        relaxation or _relax(),
        resolvent or _resolvent(),
        transient or _transient(),
        lep or _lep(),
        mpemba,
        **kw,
    )


# ---------------------------------------------------------------------------
# 1. A11 floor override must be carried by the *typed* EnsembleEvidence only.
# ---------------------------------------------------------------------------


def _maxmix_mpemba_inputs():
    """Single-state A11 pick on a maximally mixed steady state (floor case)."""
    return {
        "spectral": _spectral(steady_state=np.eye(2, dtype=complex) / 2.0),
        "mpemba": _mpemba(),
    }


def test_duck_typed_ensemble_evidence_is_rejected():
    """FAILS-BEFORE: any object with permits_claim_floor_override=True lifted
    the A11 floor, bypassing all provenance validation in ensemble.py."""
    fake = type("Fake", (), {"permits_claim_floor_override": True})()
    with pytest.raises(TypeError, match="EnsembleEvidence"):
        _classify(**_maxmix_mpemba_inputs(), ensemble_evidence=fake)


def test_none_ensemble_evidence_keeps_floor():
    res = _classify(**_maxmix_mpemba_inputs())
    assert res.a_class == "A11"
    assert res.verdict == VERDICT_UNDEFINED
    assert res.tier == TIER_EXPLORATION


def test_real_passing_evidence_still_lifts_floor():
    evidence = EnsembleEvidence(
        manifest_sha256="a" * 64,
        initial_state_family="thermal",
        ordering_parameter="beta",
        run_ids=("1" * 64, "2" * 64),
        input_hashes=("3" * 64, "4" * 64),
        relaxation_metric="trace_distance",
        comparison_test="crossing_time",
        uncertainty_method="bootstrap_bca",
        software_version="0.6.0.dev0",
        gate_status="PASS",
        reason_code=ENSEMBLE_MPEMBA_CONFIRMED,
        producer_attestation_sha256="b" * 64,
        reviewer_attestation_sha256="c" * 64,
    )
    res = _classify(**_maxmix_mpemba_inputs(), ensemble_evidence=evidence)
    assert res.a_class == "A11"
    assert res.verdict != VERDICT_UNDEFINED


# ---------------------------------------------------------------------------
# 2. Non-finite symmetrised gaps must not grant the A1 #80 certificate.
# ---------------------------------------------------------------------------


def _a1_inputs(**spectral_kw):
    """Inputs that pick A1 via the strong D17 branch."""
    return {
        "spectral": _spectral(**spectral_kw),
        "lep": _lep(gap_rate_consistency=0.01),
        "relaxation": _relax(linear_fit_model="M1"),
    }


def test_infinite_kms_gap_does_not_certify_a1():
    """FAILS-BEFORE: kms_gap=inf gave gap_to_kms_ratio=0.0 <= 1.2 and granted
    sym_gap_corroborated -> A1 CONFIRMED/PUBLICATION_GRADE."""
    res = _classify(**_a1_inputs(kms_gap=float("inf"), gns_gap=0.0))
    assert res.a_class == "A1"
    assert res.evidence["sym_gap_corroborated"] == 0.0
    assert res.verdict != VERDICT_CONFIRMED
    assert res.confidence == pytest.approx(0.70)


def test_infinite_gns_gap_does_not_certify_a1():
    """FAILS-BEFORE: gns_gap=inf passed the certified floor test (inf >= x) and
    gave gap_to_gns_ratio=0.0 <= 1.2 -> certificate granted."""
    res = _classify(**_a1_inputs(gns_gap=float("inf"), kms_gap=0.0))
    assert res.a_class == "A1"
    assert res.evidence["gns_certified"] == 0.0
    assert res.evidence["sym_gap_corroborated"] == 0.0
    assert res.verdict != VERDICT_CONFIRMED


def test_finite_matching_kms_gap_still_certifies_a1():
    # Contract preserved: a genuinely measured Delta_KMS == Delta corroborates.
    res = _classify(**_a1_inputs(kms_gap=0.5, gns_gap=0.0))
    assert res.a_class == "A1"
    assert res.evidence["sym_gap_corroborated"] == 1.0
    assert res.verdict == VERDICT_CONFIRMED


# ---------------------------------------------------------------------------
# 3. F5 gapless phantom limit fires even when the GNS gap is ALSO floored.
# ---------------------------------------------------------------------------


def test_f5_fires_gapless_with_floored_gns_gap():
    """FAILS-BEFORE: gap=0 AND gns_gap=0 made both sides inf and 'inf > inf'
    is False -- the documented gapless phantom/critical limit never fired and
    the run fell through to A12."""
    res = _classify(
        spectral=_spectral(gap=0.0, gns_gap=0.0, kms_gap=0.0),
        nonnorm=_nonnorm(henrici_eta=2.0),
        lep=_lep(gap_rate_consistency=float("inf")),
        relaxation=_relax(aicc_model="M9", linear_fit_model="M9"),
    )
    assert (res.a_class, res.f_family) == ("A10", "F5")


def test_f5_still_fires_gapless_with_measured_gns_gap():
    # The previously pinned case (gap=0, gns_gap>0) keeps firing.
    res = _classify(
        spectral=_spectral(gap=0.0, gns_gap=0.5),
        nonnorm=_nonnorm(henrici_eta=2.0),
        lep=_lep(gap_rate_consistency=float("inf")),
        relaxation=_relax(aicc_model="M9", linear_fit_model="M9"),
    )
    assert (res.a_class, res.f_family) == ("A10", "F5")


# ---------------------------------------------------------------------------
# 4. Malformed steady-state shapes must not silently disable the A11 floor.
# ---------------------------------------------------------------------------


def test_vectorised_steady_state_shape_raises():
    """FAILS-BEFORE: a still-vectorised (d^2,) maximally mixed steady state
    read as 'not maximally mixed' and skipped the floor entirely."""
    vectorised = (np.eye(2, dtype=complex) / 2.0).flatten()
    with pytest.raises(ValueError, match="square 2-D"):
        _classify(
            spectral=_spectral(steady_state=vectorised),
            mpemba=_mpemba(),
        )


def test_scalar_placeholder_steady_state_is_not_flagged():
    # The documented d < 2 placeholder path stays available for unit tests.
    res = _classify(spectral=_spectral(steady_state=np.zeros((1, 1), dtype=complex)))
    assert res.evidence["maximally_mixed_steady_state"] == 0.0


# ---------------------------------------------------------------------------
# 5. EnsembleEvidence: duplicate input hashes are not a reference family.
# ---------------------------------------------------------------------------


def test_duplicate_input_hashes_rejected():
    """FAILS-BEFORE: two paired runs with the identical input hash (same
    system + same initial state = no ordering-parameter variation) validated
    and lifted the A11 floor."""
    with pytest.raises(ValueError, match="input_hashes must be unique"):
        EnsembleEvidence(
            manifest_sha256="a" * 64,
            initial_state_family="thermal",
            ordering_parameter="beta",
            run_ids=("1" * 64, "2" * 64),
            input_hashes=("4" * 64, "4" * 64),
            relaxation_metric="trace_distance",
            comparison_test="crossing_time",
            uncertainty_method="bootstrap_bca",
            software_version="0.6.0.dev0",
            gate_status="PASS",
            reason_code=ENSEMBLE_MPEMBA_CONFIRMED,
            producer_attestation_sha256="b" * 64,
            reviewer_attestation_sha256="c" * 64,
        )


# ---------------------------------------------------------------------------
# 6. build_liouvillian / examples: non-finite input fails loudly.
# ---------------------------------------------------------------------------


def test_build_liouvillian_rejects_nan_rate(pauli):
    """FAILS-BEFORE: NaN < 0 is False, so a NaN rate produced an all-NaN
    Liouvillian with only a numpy RuntimeWarning."""
    with pytest.raises(ValueError, match="finite"):
        build_liouvillian(pauli["Z"], [pauli["Z"]], [float("nan")])


def test_build_liouvillian_rejects_infinite_rate(pauli):
    with pytest.raises(ValueError, match="finite"):
        build_liouvillian(pauli["Z"], [pauli["Z"]], [float("inf")])


def test_build_liouvillian_rejects_nonfinite_jump_op(pauli):
    bad = np.array([[0.0, np.inf], [0.0, 0.0]], dtype=complex)
    with pytest.raises(ValueError, match="non-finite"):
        build_liouvillian(pauli["Z"], [bad], [0.5])


def test_build_liouvillian_rejects_infinite_hermitian_h():
    """FAILS-BEFORE: an all-real +/-inf diagonal H is 'Hermitian' to
    np.allclose and propagated silently."""
    bad_h = np.diag([np.inf, -np.inf]).astype(complex)
    with pytest.raises(ValueError, match="non-finite"):
        build_liouvillian(bad_h, [], [])


def test_v4_thermal_rejects_nan_beta():
    """FAILS-BEFORE: nan*omega <= 0 is False, so beta=nan slipped past the
    thermal-occupation guard into silently-NaN rates."""
    from liouscope.examples import v4_thermal_two_level

    with pytest.raises(ValueError, match="finite"):
        v4_thermal_two_level(beta=float("nan"))


# ---------------------------------------------------------------------------
# 7. steady_state: no fabricated steady states, structured shape errors.
# ---------------------------------------------------------------------------


def test_steady_state_no_null_vector_warns_with_residual():
    """FAILS-BEFORE: an invertible superoperator (no steady state at all)
    returned the smallest-singular-value direction silently, with residual
    ||L rho|| = O(1) and no signal to the caller."""
    with pytest.warns(RuntimeWarning, match="NOT a verified steady state"):
        steady_state(-np.eye(4, dtype=complex))


def test_steady_state_rejects_empty_superoperator():
    """FAILS-BEFORE: a (0, 0) input passed the square-d check with d=0 and
    crashed with a bare IndexError inside the SVD post-processing."""
    with pytest.raises(ValueError, match="square-d"):
        steady_state(np.zeros((0, 0), dtype=complex))


def test_steady_state_rejects_non_2d_input():
    with pytest.raises(ValueError, match="square 2-D"):
        steady_state(np.zeros(16, dtype=complex))


# ---------------------------------------------------------------------------
# 8. diagnose(): the t_grid boundary is validated like every other input.
# ---------------------------------------------------------------------------


def _small_l(pauli):
    return build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])


def test_diagnose_rejects_nan_t_grid(pauli):
    """FAILS-BEFORE: a NaN-contaminated t_grid propagated through expm(L t)
    and still yielded a finite, confident-looking beta_D."""
    from liouscope import diagnose

    with pytest.raises(ValueError, match="non-finite"):
        diagnose(_small_l(pauli), t_grid=np.array([0.0, 1.0, np.nan]))


def test_diagnose_rejects_negative_t_grid(pauli):
    from liouscope import diagnose

    with pytest.raises(ValueError, match="non-negative"):
        diagnose(_small_l(pauli), t_grid=np.array([-1.0, 0.0, 1.0]))


def test_diagnose_rejects_unordered_t_grid(pauli):
    from liouscope import diagnose

    with pytest.raises(ValueError, match="strictly increasing"):
        diagnose(_small_l(pauli), t_grid=np.array([0.0, 2.0, 1.0]))


def test_diagnose_accepts_valid_custom_t_grid(pauli):
    from liouscope import diagnose

    # >= 41 points: the GLS AR(1) small-n advisory warning fires at n <= 40
    # and the suite runs with warnings-as-errors.
    #
    # ROUND 21: rho_initial is now given explicitly. Left to the default the
    # chosen state can BE the steady state, in which case the relaxation curve
    # is identically zero and there is nothing to fit -- and whether it lands
    # there varies with the numpy/scipy build (exactly zero on the 3.10/3.11
    # runners, non-zero elsewhere). The old code returned the optimiser's seed
    # for such a curve, so ``isfinite`` passed on a meaningless number and the
    # test looked green; issue #123 made that refusal explicit and the test
    # started failing on the two runners where the curve really was flat.
    # ``|0><0|`` genuinely relaxes towards the maximally mixed steady state, so
    # this now exercises what the test claims: that a valid custom grid is
    # accepted, not that a degenerate fit returns a number.
    rho_initial = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    report = diagnose(
        _small_l(pauli),
        rho_initial=rho_initial,
        t_grid=np.linspace(0.0, 6.0, 60),
        bootstrap_B=20,
        seed=42,
    )
    assert np.isfinite(report.relaxation.beta_D)
    assert report.relaxation.beta_D > 0.0, (
        "a state relaxing towards the steady state must yield a positive "
        "decay rate; a non-positive one would mean the curve carried no decay"
    )


# ---------------------------------------------------------------------------
# 9. Zhou predictor: unconverged record honours caller-supplied arguments.
# ---------------------------------------------------------------------------


def test_zhou_early_return_honours_supplied_gap():
    """FAILS-BEFORE: the all-zero-modes early return hard-coded gap=0.0,
    overwriting an explicitly supplied caller value in the returned record."""
    from liouscope._zhou import compute_zhou_predictor

    res = compute_zhou_predictor(np.zeros((4, 4), dtype=complex), gap=0.5)
    assert not res.converged
    assert res.gap == pytest.approx(0.5)
    res2 = compute_zhou_predictor(
        np.zeros((4, 4), dtype=complex), petermann_factor=2.5
    )
    assert not res2.converged
    assert res2.petermann_factor == pytest.approx(2.5)
    res3 = compute_zhou_predictor(np.zeros((4, 4), dtype=complex))
    assert res3.gap == 0.0 and np.isnan(res3.petermann_factor)


def test_steady_state_genuine_null_vector_stays_silent(pauli):
    # Sanity: the physical path (unique steady state) emits no warning.
    import warnings as _warnings

    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        rho_ss = steady_state(L)
    np.testing.assert_allclose(rho_ss, 0.5 * np.eye(2, dtype=complex), atol=1e-9)
