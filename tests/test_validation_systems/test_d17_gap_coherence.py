"""End-to-end D17 gap-rate coherence checks on the V1-V5 systems + a genuine
gap-controlled reference (issue #69).

Before the fix, D17 = |beta_D - Delta| / Delta compared the RELATIVE-ENTROPY fit
rate beta_D directly with the spectral gap Delta. Relative entropy near a
faithful (full-rank) steady state is quadratic in (rho - pi), so it decays at
2*Delta; near a rank-deficient pi the null-space-leakage term is linear, so it
decays at 1*Delta. That metric multiplier m in {1, 2} inflated D17 (e.g.
dephasing D17 ~ 3.3, amp-damp ~ 1.1) and made the A1 "gap-controlled" label
unreachable for every system.

The fix feeds D17 the LINEAR trace-distance decay rate beta_D_linear, which is
dimension-coherent with Delta (both bare single-mode rates). These tests are the
end-to-end regression that was previously absent (test_classification.py only
drove synthetic evidence dicts).
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import diagnose
from liouscope.core.hamiltonian import _pauli
from liouscope.core.lindblad import build_liouvillian, steady_state
from liouscope.examples import (
    v1_qutrit,
    v2_dephasing_qubit,
    v3_amplitude_damped_qubit,
    v4_thermal_two_level,
    v5_jaynes_cummings,
)
from liouscope.numerics.kronecker import unvec


def _gap_controlled_reference() -> tuple[np.ndarray, np.ndarray]:
    """A textbook gap-controlled qubit: faithful (full-rank) thermal steady
    state, fast dephasing so the POPULATION mode is the slowest (= the spectral
    gap), and a diagonal initial state that overlaps that mode. The observable
    (trace-distance) relaxation is then a single exponential at exactly Delta,
    so the dimension-coherent D17 -> 0 and the classifier can award A1.
    """
    _, _, _, _sz = _pauli()
    sz = _sz
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # lowering |0><1|
    sp = sm.conj().T
    g_down, g_up, g_dephase = 0.6, 0.2, 1.0  # g_up > 0 => faithful (full-rank) pi
    H = np.zeros((2, 2), dtype=complex)  # no coherent drive => no complex pairs
    L = build_liouvillian(H, [sm, sp, sz], [g_down, g_up, g_dephase])
    rho0 = np.array([[0.9, 0.0], [0.0, 0.1]], dtype=complex)  # diagonal => overlaps pop mode
    return L, rho0


ALL_V = [
    ("V1", v1_qutrit()),
    ("V2", v2_dephasing_qubit()),
    ("V3", v3_amplitude_damped_qubit(gamma=0.5)),
    ("V4", v4_thermal_two_level()),
    ("V5", v5_jaynes_cummings()),
]

# Golden end-to-end mechanism labels for V1-V5 (full pipeline, bootstrap_B=30,
# seed=42). Measured on main (baseline 3fef6a1) AND on this branch: IDENTICAL --
# the #69 fix corrects the D17 NUMBER, it does not reclassify any V-system (none
# is a single-mode gap-controlled system; they mode-skip or are Mpemba
# candidates). This pins that "no label change" claim, which was previously
# unpinned (the V* golden files assert only gap/beta_D, never a_class).
V_GOLDEN_ACLASS = {"V1": "A5", "V2": "A11", "V3": "A5", "V4": "A5", "V5": "A11"}


def _slow_mode_aligned_rho0(L: np.ndarray, eps_candidates=(0.06, 0.04, 0.03, 0.02)):
    """Build a valid density matrix rho0 = pi + eps * (slowest real right-eigenmode)
    so the trajectory is a clean single exponential at exactly the spectral gap
    (=> D17 -> 0, single-exp). Returns None if no eps keeps rho0 positive.
    """
    d = int(round(np.sqrt(L.shape[0])))
    pi = steady_state(L)
    w, V = np.linalg.eig(L)
    order = np.argsort(-w.real)
    slow = next(i for i in order if abs(w[i]) > 1e-9)
    if abs(w[slow].imag) > 1e-6:
        return None  # want a real slow mode for a clean single exponential
    mode = unvec(V[:, slow], d=d)
    mode = 0.5 * (mode + mode.conj().T)
    mode = mode - np.trace(mode) / d * np.eye(d)
    mode /= np.linalg.norm(mode)
    for eps in eps_candidates:
        cand = pi + eps * mode
        if np.linalg.eigvalsh(0.5 * (cand + cand.conj().T)).min() > 1e-6:
            return cand
    return None


def _phantom_ladder() -> tuple[np.ndarray, np.ndarray]:
    """A STRONGLY non-normal phantom operator (d=4 directional ladder): henrici
    ~4.2, pseudospectral radius ~2.15 > 2*gap_to_gns ~2.0 => the F5 phantom
    family fires. Paired with a slow-mode-aligned rho0 it ALSO satisfies the A1
    early-branch condition (single-exp trace distance at the gap rate). This is
    the Equalita #79 adversarial case: without the F5-first reorder it would be
    mislabelled A1/F1 CONFIRMED, shadowing the real phantom mechanism.
    """
    d = 4
    low = np.zeros((d, d), dtype=complex)
    for i in range(d - 1):
        low[i, i + 1] = 1.0  # directional lowering |i><i+1|
    up = low.conj().T
    zdiag = np.diag(np.arange(d, dtype=complex))  # weak dephasing keeps gns finite
    L = build_liouvillian(np.zeros((d, d), dtype=complex), [low, up, zdiag],
                          [1.5, 0.1, 0.15])
    rho0 = _slow_mode_aligned_rho0(L)
    assert rho0 is not None
    return L, rho0


@pytest.mark.parametrize("label,sys", ALL_V)
def test_v_systems_mechanism_label_golden(label, sys):
    """End-to-end a_class golden pins (Equalita #79 Befund 2). Previously absent:
    the V* golden files assert only gap/beta_D, and test_classification.py drives
    only synthetic evidence dicts, so nothing pinned that the #69 D17 change does
    not flip any V-system's mechanism label. Baseline (main 3fef6a1) == branch.
    """
    rep = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=30, seed=42)
    assert rep.classification.a_class == V_GOLDEN_ACLASS[label]


@pytest.mark.parametrize("label,sys", ALL_V)
def test_v_systems_d17_is_dimension_coherent(label, sys):
    """D17 must equal |beta_D_linear - Delta| / Delta (the LINEAR rate), NOT the
    relative-entropy rate. This is the core dimension-coherence contract.
    """
    rep = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    gap = rep.spectral.gap
    blin = rep.relaxation.beta_D_linear
    assert gap > 0
    assert np.isfinite(blin)
    # D17 is computed from the linear rate, exactly.
    expected = abs(blin - gap) / gap
    assert rep.lep.gap_rate_consistency == pytest.approx(expected, rel=1e-9)
    assert rep.lep.beta_D_linear == pytest.approx(blin, rel=1e-9)
    # Evidence dict exposes the inputs explicitly (auditable factor).
    ev = rep.classification.evidence
    assert ev["gap"] == pytest.approx(gap, rel=1e-9)
    assert ev["beta_D_linear"] == pytest.approx(blin, rel=1e-9)
    assert ev["beta_D"] == pytest.approx(rep.relaxation.beta_D, rel=1e-9)


@pytest.mark.parametrize("label,sys", ALL_V)
def test_v_systems_relative_entropy_multiplier_never_below_one(label, sys):
    """The relative-entropy metric multiplier m = beta_D / beta_D_linear is
    physically bounded to [~1, ~2]: 1 for a rank-deficient pi, 2 for a faithful
    pi. It must never be < ~1 (that would mean relative entropy decays SLOWER
    than a linear metric, which is impossible for a contraction toward pi).
    """
    rep = diagnose(sys.L, rho_initial=sys.rho_initial, bootstrap_B=20, seed=42)
    m = rep.classification.evidence["d17_metric_multiplier"]
    assert np.isfinite(m)
    assert m > 0.85  # allow fit noise below the ideal 1.0 floor
    assert m < 2.6   # ideal ceiling 2.0 + fit noise


def test_faithful_vs_rankdeficient_multiplier_split():
    """Faithful (full-rank) pi -> m ~ 2; rank-deficient pi -> m ~ 1. This is the
    empirical signature the fix relies on: V2/V4 (faithful) vs V3 (rank-1 pi).
    """
    v2 = diagnose(v2_dephasing_qubit().L,
                  rho_initial=v2_dephasing_qubit().rho_initial,
                  bootstrap_B=20, seed=42)
    v4s = v4_thermal_two_level()
    v4 = diagnose(v4s.L, rho_initial=v4s.rho_initial, bootstrap_B=20, seed=42)
    v3s = v3_amplitude_damped_qubit(gamma=0.5)
    v3 = diagnose(v3s.L, rho_initial=v3s.rho_initial, bootstrap_B=20, seed=42)
    m2 = v2.classification.evidence["d17_metric_multiplier"]
    m4 = v4.classification.evidence["d17_metric_multiplier"]
    m3 = v3.classification.evidence["d17_metric_multiplier"]
    assert m2 > 1.5 and m4 > 1.5   # faithful pi -> quadratic -> ~2
    assert m3 < 1.5                # rank-deficient pi -> linear leakage -> ~1


# The reference qubit is a mildly non-normal dissipator whose propagator norm
# settles onto a small transient ratio (~1.17, far below the A4 skin-effect
# threshold of 5) by a slow monotone approach; the physics-scaled transient grid
# reports that supremum as a lower bound (documented TransientGridWarning). It is
# irrelevant to gap-control classification, so we filter it BY MESSAGE (matching
# the ResolventConvergenceWarning convention in pyproject.toml -- a class-based
# filter would force a config-time import and crater coverage).
@pytest.mark.filterwarnings(
    "ignore:trans_amplitude_ratio"
)
def test_gap_controlled_reference_reaches_a1():
    """A genuine gap-controlled textbook system must now be able to earn the A1
    'gap-controlled' label -- the whole point of issue #69. Before the fix this
    was impossible for any system.
    """
    L, rho0 = _gap_controlled_reference()
    rep = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    cls = rep.classification
    assert cls.a_class == "A1"
    # issue #70 A6: A1 (gap-controlled, primitive QMS) is the NO-gap-failure case
    # and maps to family "none", not F1 (Mori-Shirai overlap gap-FAILURE). This
    # does not touch the #69 dimension-coherence logic -- only the family label.
    assert cls.f_family == "none"
    assert cls.verdict == "CONFIRMED"
    # Dimension-coherent D17 is essentially zero: the observable relaxation is a
    # single exponential at exactly the gap.
    assert rep.lep.gap_rate_consistency < 0.05
    assert rep.relaxation.linear_fit_model in ("M0", "M1")


@pytest.mark.filterwarnings(
    "ignore:trans_amplitude_ratio"
)
def test_gap_controlled_reference_old_formula_would_have_failed_a1():
    """Regression proving the fix is load-bearing: the OLD D17 (relative-entropy
    rate vs gap) exceeds the A1 threshold (0.20) for the same gap-controlled
    system, i.e. the metric multiplier alone kept A1 unreachable. The NEW D17
    (linear rate) is far below 0.05.
    """
    L, rho0 = _gap_controlled_reference()
    rep = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    gap = rep.spectral.gap
    old_d17 = abs(rep.relaxation.beta_D - gap) / gap   # pre-#69 formula
    new_d17 = rep.lep.gap_rate_consistency
    assert old_d17 > 0.20   # would have failed BOTH A1 thresholds
    assert new_d17 < 0.05   # passes the strong-A1 threshold
    assert old_d17 > new_d17 + 0.5  # the multiplier gap is large and real


@pytest.mark.filterwarnings("ignore:trans_amplitude_ratio")
def test_phantom_with_gap_matched_rho0_is_not_mislabelled_a1():
    """Equalita #79 Befund 1: a strongly non-normal PHANTOM operator (F5) with a
    slow-mode-aligned rho_0 satisfies the A1 early-branch condition (single-exp
    trace distance at the gap rate), yet MUST be classified A10/F5, not A1/F1.
    Guards against awarding a false gap-controlled PUBLICATION_GRADE label to a
    genuine gap-FAILURE mechanism.
    """
    L, rho0 = _phantom_ladder()
    rep = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    ev = rep.classification.evidence
    # Adversarial precondition: the A1 early branch WOULD fire on its own.
    assert rep.lep.gap_rate_consistency < 0.05
    assert ev["d17_linear_single_exp"] == 1.0
    # The operator is genuinely a strong F5 phantom (operator-intrinsic).
    # issue #70 A8: the F5 rule is now dimension-coherent -- the pseudospectral
    # REACH (radius / gap, dimensionless) exceeds 2 * gap_to_gns_ratio, not the
    # bare (rate-dimensioned) radius.
    assert ev["henrici_eta"] > 1.0
    assert ev["gap"] > 0.0
    assert ev["pseudospectral_radius"] / ev["gap"] > 2.0 * ev["gap_to_gns_ratio"]
    # ...so the gap-failure family wins; A1/"none" is NOT awarded.
    assert (rep.classification.a_class, rep.classification.f_family) == ("A10", "F5")
    assert rep.classification.a_class != "A1"


def test_gap_controlled_reference_steady_state_is_faithful():
    """Sanity: the reference pi is full-rank (faithful), so its multiplier is ~2
    and the D17 correction (dividing out that factor) is what enables A1.
    """
    from liouscope.core.lindblad import steady_state

    L, _rho0 = _gap_controlled_reference()
    pi = steady_state(L)
    evals = np.linalg.eigvalsh(0.5 * (pi + pi.conj().T))
    assert int(np.sum(evals > 1e-9)) == pi.shape[0]  # full rank
