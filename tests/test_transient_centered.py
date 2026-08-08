"""Issue #103 — D14 transient amplitude: projector baseline and norm geometry.

The validation plan of the issue is covered point by point; each test carries
the item number it discharges.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.spectral import liouvillian_gap
from liouscope.diagnostics.transient import (
    centered_transient_amplitude,
    compute_transient_layer,
    decaying_transient_amplitude,
    operational_trace_amplitude,
    steady_projector,
    trans_amplitude_ratio,
)

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)          # |0><1|, lowers to |0>
SP = SM.conj().T
ZERO_H = np.zeros((2, 2), dtype=complex)


def _amplitude_damping(gamma: float = 1.0) -> np.ndarray:
    return build_liouvillian(ZERO_H, [np.sqrt(gamma) * SM])


def _depolarising(gamma: float = 1.0) -> np.ndarray:
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    c = np.sqrt(gamma / 4.0)
    return build_liouvillian(ZERO_H, [c * SX, c * sy, c * sz])


def _steady_state(L: np.ndarray, d: int) -> np.ndarray:
    """Steady state from the kernel, normalised to unit trace."""
    w, v = np.linalg.eig(L)
    rho = v[:, int(np.argmin(np.abs(w)))].reshape(d, d, order="F")
    rho = 0.5 * (rho + rho.conj().T)
    return rho / np.trace(rho).real


# --- Validation item 1: closed-form rank-one projector norm ------------------

def test_projector_norm_matches_closed_form_amplitude_damping():
    """Pure steady state (|0><0|): ||P_inf||_2 = sqrt(d * Tr rho_ss^2) = sqrt(2)."""
    L = _amplitude_damping()
    proj = steady_projector(L)
    assert proj.rank == 1
    assert proj.semisimple
    rho = _steady_state(L, 2)
    expected = np.sqrt(2.0 * float(np.trace(rho @ rho).real))
    got = float(sla.svdvals(proj.projector)[0])
    assert got == pytest.approx(expected, rel=1e-9)
    assert got == pytest.approx(np.sqrt(2.0), rel=1e-9)


def test_projector_norm_is_one_for_maximally_mixed_steady_state():
    """Depolarising: rho_ss = I/2, purity 1/2 => ||P_inf||_2 = 1."""
    proj = steady_projector(_depolarising())
    assert proj.rank == 1
    assert float(sla.svdvals(proj.projector)[0]) == pytest.approx(1.0, rel=1e-9)


def test_projector_is_idempotent_and_annihilates_the_decaying_block():
    L = _amplitude_damping()
    P = steady_projector(L).projector
    assert np.allclose(P @ P, P, atol=1e-10)
    # P commutes with the semigroup and is its asymptotic limit.
    assert np.allclose(sla.expm(L * 50.0), P, atol=1e-8)


# --- Validation item 2: legacy value inflated by the baseline alone ----------

def test_legacy_d14_carries_the_projector_baseline_without_overshoot():
    """The confound the issue reports, measured.

    Amplitude damping has NO transient overshoot of the decaying part, yet the
    legacy full-space value sits at sqrt(d * purity) = sqrt(2) > 1.
    """
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    legacy = trans_amplitude_ratio(L, gap=gap)

    assert legacy == pytest.approx(np.sqrt(2.0), rel=1e-6)
    assert legacy > 1.0 + 1e-3
    assert centered_transient_amplitude(L, gap=gap).projector_norm == pytest.approx(
        np.sqrt(2.0), rel=1e-6
    )
    # Only the restricted semigroup is free of the baseline.
    assert decaying_transient_amplitude(L, gap=gap).value <= 1.0 + 1e-9


def test_centering_alone_does_not_remove_the_baseline():
    """Correction to the wording of issue #103.

    The issue offers ``sup_t ||e^{tL} - P_inf||`` "or equivalently restrict the
    semigroup to the decaying complement". They are NOT equivalent: at ``t=0``
    the centred form is ``||I - P_inf||``, and for a non-trivial oblique
    projector ``||I - P|| = ||P||``. So the centred value still equals the
    projector norm the issue set out to remove.
    """
    L = _amplitude_damping()
    P = steady_projector(L).projector
    n = P.shape[0]
    norm_P = float(sla.svdvals(P)[0])
    norm_IP = float(sla.svdvals(np.eye(n) - P)[0])
    assert norm_IP == pytest.approx(norm_P, rel=1e-9)      # the identity itself

    gap = liouvillian_gap(np.linalg.eigvals(L))
    centered = centered_transient_amplitude(L, gap=gap).value
    decaying = decaying_transient_amplitude(L, gap=gap).value
    assert centered == pytest.approx(norm_P, rel=1e-6)     # baseline still there
    assert decaying <= 1.0 + 1e-9                          # baseline gone
    assert decaying < centered - 1e-3


def test_restricted_value_is_at_most_the_legacy_value():
    for L in (_amplitude_damping(), _depolarising()):
        gap = liouvillian_gap(np.linalg.eigvals(L))
        assert decaying_transient_amplitude(L, gap=gap).value <= (
            trans_amplitude_ratio(L, gap=gap) + 1e-9
        )


# --- Validation item 3: metamorphic rate rescaling --------------------------

@pytest.mark.parametrize("c", [1e-3, 1e-1, 1.0, 10.0, 1e3])
def test_centered_amplitude_invariant_under_rate_rescale(c):
    """``exp(cL * t/c) = exp(Lt)``: D14b is dimensionless, so it must not move."""
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    base = decaying_transient_amplitude(L, gap=gap).value
    scaled = decaying_transient_amplitude(c * L, gap=c * gap).value
    assert scaled == pytest.approx(base, rel=1e-6)


@pytest.mark.parametrize("c", [1e-2, 1.0, 1e2])
def test_operational_amplitude_invariant_under_rate_rescale(c):
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    base = operational_trace_amplitude(L, gap=gap).value
    scaled = operational_trace_amplitude(c * L, gap=c * gap).value
    assert scaled == pytest.approx(base, rel=1e-6)


# --- Validation item 4: basis-unitary invariance ----------------------------

def test_centered_amplitude_invariant_under_basis_unitary():
    L = _amplitude_damping()
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    gap = liouvillian_gap(np.linalg.eigvals(L))
    a = decaying_transient_amplitude(L, gap=gap).value
    b = decaying_transient_amplitude(q.conj().T @ L @ q, gap=gap).value
    assert b == pytest.approx(a, rel=1e-8)


# --- Validation item 5: unique, degenerate and defective fixtures -----------

def test_degenerate_stationary_manifold_gives_rank_two_projector():
    """Two decoupled dephasing qubits with a two-dimensional kernel.

    A rank-one projector built from one arbitrary null vector would silently
    misrepresent this case; the Riesz construction must report rank 2.
    """
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L1 = build_liouvillian(ZERO_H, [sz])          # dephasing: kernel is 2-dim
    proj = steady_projector(L1)
    assert proj.rank == 2
    assert proj.semisimple
    assert np.allclose(proj.projector @ proj.projector, proj.projector, atol=1e-10)


def test_defective_zero_mode_fails_closed():
    """A Jordan block at zero has no such spectral projector -> NaN, not a number."""
    J = np.array(
        [[0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, -1.0, 0.0],
         [0.0, 0.0, 0.0, -2.0]],
        dtype=complex,
    )
    proj = steady_projector(J)
    assert proj.semisimple is False
    est = centered_transient_amplitude(J, gap=1.0, projector=proj)
    assert np.isnan(est.value)
    assert est.semisimple is False


def test_zero_generator_has_identity_projector():
    proj = steady_projector(np.zeros((4, 4)))
    assert proj.rank == 4
    assert np.allclose(proj.projector, np.eye(4))


def test_non_finite_input_fails_closed():
    bad = np.zeros((4, 4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        steady_projector(bad)


# --- Validation item 6: grid metadata / lower-bound labelling ---------------

def test_estimates_report_grid_window_and_edge_flag():
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    est = centered_transient_amplitude(L, gap=gap)
    assert est.n_points > 1
    assert est.t_max > est.t_min
    assert isinstance(est.edge_maximizer, bool)

    op = operational_trace_amplitude(L, gap=gap)
    assert op.n_states > 0
    assert op.seed == 0                    # sampling is recorded, not implicit


def test_narrow_grid_is_flagged_as_edge_maximizer():
    """A window that stops inside the rise must announce the lower bound.

    Uses a strongly non-normal STABLE generator with a genuine transient peak
    (no peripheral mode, so the projector is zero and the centred quantity is
    the bare propagator norm).
    """
    A = np.array([[-1.0, 20.0], [0.0, -2.0]], dtype=complex)
    est = centered_transient_amplitude(A, t_grid=np.linspace(0.0, 0.02, 8))
    assert est.rank == 0
    assert est.edge_maximizer is True


# --- Validation item 7: physical-state trace-distance contractivity ---------

@pytest.mark.parametrize("build", [_amplitude_damping, _depolarising])
def test_cptp_dynamics_never_amplify_trace_distance(build):
    """The operational diagnostic is its own control: CPTP => value <= 1."""
    L = build()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    op = operational_trace_amplitude(L, gap=gap)
    assert op.value <= 1.0 + 1e-8


def test_operational_stays_at_one_where_the_legacy_value_exceeds_it():
    """Same system, two answers — that is the point of the split."""
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    assert trans_amplitude_ratio(L, gap=gap) > 1.0 + 1e-3
    assert operational_trace_amplitude(L, gap=gap).value <= 1.0 + 1e-8


# --- Layer wiring: additive, no classifier consumption ----------------------

def test_transient_layer_exposes_the_new_fields_without_changing_legacy():
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    res = compute_transient_layer(L, gap)
    assert res.trans_amplitude_ratio == pytest.approx(
        trans_amplitude_ratio(L, gap=gap), rel=1e-12
    )
    assert res.steady_projector_rank == 1
    assert res.steady_projector_semisimple is True
    assert res.steady_projector_norm == pytest.approx(np.sqrt(2.0), rel=1e-6)
    assert res.trans_amplitude_centered == pytest.approx(np.sqrt(2.0), rel=1e-6)
    assert res.trans_amplitude_decaying <= 1.0 + 1e-9
    assert res.trans_amplitude_operational <= 1.0 + 1e-8


# --- Cross-Family-Review zu PR #105: je Befund ein Regressionstest ----------

def test_slow_decaying_mode_is_not_absorbed_into_the_projector():
    """BEFUND #222: eine sqrt(eps)-Schwelle verschluckt echte langsame Moden.

    Rates 1 und 1e-8: der stationaere Projektor hat Rang 1. Unter der alten
    Schwelle (~2.3e-7 * ||L||_F) waeren die -5e-9/-1e-8-Moden als stationaer
    eingestuft worden, und D14b/D14d haetten echte Relaxation weggerechnet.
    """
    A = np.diag([0.0, -1.0, -5e-9, -1e-8]).astype(complex)
    proj = steady_projector(A)
    assert proj.rank == 1
    assert proj.semisimple


def test_oscillatory_peripheral_mode_fails_closed():
    """BEFUND #231: rein imaginaere Moden haben keinen statischen Grenzwert.

    Ein rein hamiltonscher Qubit-Liouvillian oszilliert ewig; e^{tL} konvergiert
    nicht, also existiert kein zeitunabhaengiger asymptotischer Projektor.
    """
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(sz, [])          # nur Hamiltonteil -> Moden auf iR
    proj = steady_projector(L)
    assert proj.semisimple is False
    assert np.isnan(centered_transient_amplitude(L, gap=1.0, projector=proj).value)


def test_semisimplicity_verdict_is_invariant_under_rate_rescale():
    """BEFUND #280: die Clusterbreite darf nicht absolut sein.

    Nach dem Fix zu #231 besteht die Peripherie nur noch aus Nullmoden, womit
    der urspruenglich genannte Fall (mehrere getrennte Frequenzen) gar nicht
    mehr auftritt. Was bleibt und geprueft gehoert: das Semisimplizitaets-
    Verdikt darf sich unter L -> cL nicht aendern.
    """
    D = np.diag([0.0, 0.0, -1.0, -2.0]).astype(complex)
    base = steady_projector(D)
    assert base.semisimple is True
    assert base.rank == 2
    for c in (1e-13, 1e-6, 1e6, 1e13):
        scaled = steady_projector(D * c)
        assert scaled.semisimple is base.semisimple
        assert scaled.rank == base.rank


def test_fallback_grid_contains_time_zero():
    """BEFUND #377: ohne gap/t_grid begann das Gitter bei 0.01 und unterbot 1."""
    L = _amplitude_damping()
    est = decaying_transient_amplitude(L)          # weder gap noch t_grid
    assert est.t_min == 0.0
    assert est.value >= 1.0 - 1e-12


def test_backward_time_grid_is_rejected():
    """BEFUND #467: negative Zeiten werten die nicht-CPTP-Inverse aus."""
    L = _amplitude_damping()
    with pytest.raises(ValueError):
        operational_trace_amplitude(L, t_grid=np.array([-1.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        centered_transient_amplitude(L, t_grid=np.array([1.0, 0.5, 2.0]))
    with pytest.raises(ValueError):
        decaying_transient_amplitude(L, t_grid=np.array([0.0, np.inf]))


def test_run_seed_reaches_the_sampling_diagnostic():
    """BEFUND #525: das Manifest hielt seed=123 fest, D14c wuerfelte mit 0."""
    L = _amplitude_damping()
    gap = liouvillian_gap(np.linalg.eigvals(L))
    res = compute_transient_layer(L, gap, seed=123)
    assert res.transient_seed == 123
