"""Tests for Mpemba layer D19-D20 and the non-triviality guard (issue #68)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope import (
    ENSEMBLE_MPEMBA_CONFIRMED,
    EnsembleEvidence,
    build_liouvillian,
    diagnose,
    steady_state,
)
from liouscope.diagnostics.mpemba import (
    compute_mpemba_layer,
    expansion_alpha,
    is_trivial_overlap,
    overlap_c1,
)
from liouscope.numerics.kronecker import vec


def test_overlap_c1_zero_when_initial_is_steady(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = 0.5 * np.eye(2, dtype=complex)
    c1 = overlap_c1(L, rho0)
    assert c1 < 1e-9


def test_overlap_c1_nonzero_for_general_state(pauli):
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [pauli["Z"]], [0.3])
    rho0 = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
    c1 = overlap_c1(L, rho0)
    assert c1 >= 0


def test_overlap_c1_is_biorthogonal_coefficient(pauli):
    # overlap_c1 must be |<l1, rho0>| / |<l1, r1>|, not the old |<l1,rho0>|/||l1||.
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = np.array([[0.7, 0.2 + 0.1j], [0.2 - 0.1j, 0.3]], dtype=complex)
    eigvals, vl, vr = sla.eig(L, left=True, right=True)
    nz = np.abs(eigvals) > 1e-10
    idx = np.where(nz)[0][np.argsort(-np.real(eigvals[nz]))[0]]
    l1, r1 = vl[:, idx], vr[:, idx]
    expected = abs(np.vdot(l1, vec(rho0))) / abs(np.vdot(l1, r1))
    np.testing.assert_allclose(overlap_c1(L, rho0), expected, rtol=1e-9)


def test_expansion_alpha_finite(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
    alpha = expansion_alpha(L, rho0)
    assert np.isfinite(alpha)


def test_compute_mpemba_layer_returns_result(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    rho0 = np.array([[0.8, 0.1], [0.1, 0.2]], dtype=complex)
    res = compute_mpemba_layer(L, rho0)
    assert res.overlap_c1 >= 0
    assert np.isfinite(res.expansion_alpha)


# --- non-triviality guard (issue #68) ----------------------------------------


def _amp_damping(gamma: float = 0.5):
    H = np.zeros((2, 2), dtype=complex)
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(H, [np.sqrt(gamma) * sm])
    return L, steady_state(L)


def test_trivial_overlap_flags_commuting_diagonal_state():
    # Diagonal rho_0 commutes with the ground steady state; the amplitude-damping
    # slow modes are coherences -> the zero overlap is symmetry-protected.
    L, rss = _amp_damping()
    assert is_trivial_overlap(L, np.eye(2, dtype=complex) / 2, rss)
    assert is_trivial_overlap(L, np.array([[0, 0], [0, 1]], dtype=complex), rss)


def test_trivial_overlap_false_for_noncommuting_state():
    L, rss = _amp_damping()
    rho0 = np.array([[0.6, 0.2 + 0.1j], [0.2 - 0.1j, 0.4]], dtype=complex)
    assert not is_trivial_overlap(L, rho0, rss)


def test_canonical_amp_damping_is_not_a_mpemba_candidate():
    # The headline false positive from issue #68: default I/2 must not be flagged.
    L, rss = _amp_damping()
    res = compute_mpemba_layer(L, np.eye(2, dtype=complex) / 2, rho_steady_state=rss)
    assert res.overlap_c1 < 1e-4  # overlap does vanish ...
    assert res.trivial_overlap  # ... but it is symmetry-protected ...
    assert not res.is_mpemba_candidate  # ... so it is NOT a candidate.


def _finetuned_mpemba_qutrit():
    """A genuine strong-Mpemba oracle: a fine-tuned, non-commuting qutrit state
    whose slowest-mode coefficient vanishes exactly (c_1 ~ 0) yet which is not in
    the classical sector of rho_ss. The guard must PERMIT this one."""
    H = np.array([[0, 0.4, 0], [0.4, 0.6, 0.3], [0, 0.3, 1.1]], dtype=complex)
    j1 = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=complex)
    j2 = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]], dtype=complex)
    L = build_liouvillian(H, [np.sqrt(0.5) * j1, np.sqrt(0.3) * j2])
    rss = steady_state(L)
    eigvals, vl, _ = sla.eig(L, left=True, right=True)
    nz = np.abs(eigvals) > 1e-9
    idx = np.where(nz)[0][np.argsort(-np.real(eigvals[nz]))[0]]
    l1 = vl[:, idx]
    d = 3
    gens = []
    for i in range(d):
        for j in range(i + 1, d):
            sym = np.zeros((d, d), dtype=complex)
            sym[i, j] = sym[j, i] = 1
            gens.append(sym)
            asym = np.zeros((d, d), dtype=complex)
            asym[i, j], asym[j, i] = -1j, 1j
            gens.append(asym)
    for k in range(1, d):
        dm = np.zeros((d, d), dtype=complex)
        for m in range(k):
            dm[m, m] = 1
        dm[k, k] = -k
        gens.append(dm / np.sqrt(k * (k + 1)))
    m_map = np.array(
        [[np.vdot(l1, vec(g)).real for g in gens], [np.vdot(l1, vec(g)).imag for g in gens]]
    )
    null = np.linalg.svd(m_map)[2][2:]  # directions with zero slow-mode overlap
    rng = np.random.default_rng(7)
    best = None
    for _ in range(2000):
        direction = rng.standard_normal(null.shape[0]) @ null
        delta = sum(c * g for c, g in zip(direction, gens))
        delta = (delta + delta.conj().T) / 2
        if abs(np.vdot(l1, vec(delta))) < 1e-10:
            comm = np.linalg.norm(delta @ rss - rss @ delta)
            if best is None or comm > best[0]:
                best = (comm, delta)
    delta = best[1] / np.linalg.norm(best[1])
    for scale in (0.12, 0.1, 0.08, 0.06, 0.05, 0.04):
        rho0 = rss + scale * delta
        rho0 = (rho0 + rho0.conj().T) / 2
        rho0 = rho0 / np.trace(rho0).real
        if np.min(np.linalg.eigvalsh(rho0)) > 1e-4:
            break
    return L, rss, rho0


def test_finetuned_mpemba_is_a_candidate():
    L, rss, rho0 = _finetuned_mpemba_qutrit()
    assert np.min(np.linalg.eigvalsh(rho0)) > 0  # valid density matrix
    res = compute_mpemba_layer(L, rho0, rho_steady_state=rss)
    assert res.overlap_c1 < 1e-4  # slowest mode skipped ...
    assert not res.trivial_overlap  # ... and it is not symmetry-protected ...
    assert res.is_mpemba_candidate  # ... so this IS a genuine candidate.


# --- end-to-end classification regression (issue #68) ------------------------


def test_canonical_amp_damping_not_classified_a11():
    # Textbook amplitude damping with default inputs must NOT earn a
    # PUBLICATION_GRADE-confirmed A11 anomalous-Mpemba label. (The genuine
    # true-positive direction is covered at the layer level by
    # test_finetuned_mpemba_is_a_candidate feeding the A11 branch, which
    # test_classification::test_branch_a11_mpemba pins; a full-pipeline oracle
    # would fold in the unrelated, fragile relaxation-fit numerics.)
    L, _ = _amp_damping()
    cls = diagnose(L).classification
    assert cls.a_class != "A11"


def _readme_dephasing_mpemba_case():
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    L = build_liouvillian(0.5 * sx, [sz], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    return L, rho0


def _passing_ensemble_evidence() -> EnsembleEvidence:
    return EnsembleEvidence(
        manifest_sha256="1" * 64,
        initial_state_family="thermal Gibbs family",
        ordering_parameter="inverse_temperature_beta",
        run_ids=("2" * 64, "3" * 64),
        input_hashes=("4" * 64, "5" * 64),
        relaxation_metric="trace_distance",
        comparison_test="family_ordered_crossing_test",
        uncertainty_method="BCa bootstrap 95% CI with multiplicity control",
        software_version="0.6.0.dev0",
        gate_status="PASS",
        reason_code=ENSEMBLE_MPEMBA_CONFIRMED,
        producer_attestation_sha256="6" * 64,
        reviewer_attestation_sha256="7" * 64,
    )


def test_maximally_mixed_single_state_mpemba_is_undetermined():
    # GOLDEN FLIP (issue #78 / decision E0706-13). The README dephasing example
    # (rho_ss = I/2) is MAXIMALLY MIXED: it collapses to a single eigenprojector, so
    # the issue-#68 triviality guard structurally cannot fire. A SINGLE initial
    # state on a maximally mixed steady state is INSUFFICIENT EVIDENCE for a Mpemba
    # claim -- the honest floor is UNDEFINED (UNDETERMINED), NOT the previous
    # A11/F4 CANDIDATE (over-claim) and NOT hard-exclusion (a unital multi-rate
    # process CAN exhibit Mpemba; hard-exclusion was falsified cross-family). The
    # A11/F4 best-fit label is preserved; only the verdict/tier drop.
    L, rho0 = _readme_dephasing_mpemba_case()
    cls = diagnose(L, rho_initial=rho0, bootstrap_B=50).classification
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.verdict == "UNDEFINED"
    assert cls.tier == "EXPLORATION"
    # a fortiori: never self-certifies as a confirmed / publication-grade claim.
    assert cls.tier != "PUBLICATION_GRADE"
    assert cls.verdict != "CONFIRMED"
    assert cls.verdict != "CANDIDATE"


def test_public_diagnose_ensemble_evidence_suppresses_maxmix_floor():
    L, rho0 = _readme_dephasing_mpemba_case()
    floor = diagnose(L, rho_initial=rho0, bootstrap_B=50)
    evidence = _passing_ensemble_evidence()
    override = diagnose(
        L,
        rho_initial=rho0,
        bootstrap_B=50,
        ensemble_evidence=evidence,
    )
    cls = override.classification
    assert (cls.a_class, cls.f_family) == ("A11", "F4")
    assert cls.verdict == "CANDIDATE"
    assert cls.tier == "CONFIRMATION"
    assert cls.evidence["maximally_mixed_steady_state"] == 1.0
    assert cls.evidence["ensemble_confirmation"] == 1.0
    assert floor.governance.input_hash != override.governance.input_hash
    assert override.extras["ensemble_evidence_sha256"] == evidence.sha256
    assert override.extras["ensemble_evidence"] == evidence.to_payload()


def test_public_diagnose_rejects_legacy_boolean_override():
    L, rho0 = _readme_dephasing_mpemba_case()
    with pytest.raises(ValueError, match="EnsembleEvidence"):
        diagnose(
            L,
            rho_initial=rho0,
            bootstrap_B=50,
            ensemble_confirmation=True,
        )


def test_nonpassing_ensemble_evidence_does_not_suppress_floor():
    L, rho0 = _readme_dephasing_mpemba_case()
    passing = _passing_ensemble_evidence()
    review = EnsembleEvidence(
        **{
            **passing.to_payload(),
            "run_ids": tuple(passing.run_ids),
            "input_hashes": tuple(passing.input_hashes),
            "gate_status": "REVIEW",
        }
    )
    report = diagnose(
        L,
        rho_initial=rho0,
        bootstrap_B=50,
        ensemble_evidence=review,
    )
    assert report.classification.verdict == "UNDEFINED"
    assert report.classification.tier == "EXPLORATION"
    assert report.classification.evidence["ensemble_confirmation"] == 0.0
