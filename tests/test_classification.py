"""Tests for the A1-A12 mechanism classifier."""

from __future__ import annotations

import numpy as np

from liouscope import (
    A_CLASSES,
    DIAGNOSTIC_SCHEMA_VERSION,
    F_FAMILIES,
    TAXONOMY_VERSION,
    build_liouvillian,
    classify_mechanism,
    diagnose,
)


def test_a_class_set_size():
    assert len(A_CLASSES) == 12
    assert "A11" in A_CLASSES


def test_f_family_set_size():
    assert len(F_FAMILIES) == 6  # F1-F5 + "none"


def test_classification_stamps_versions(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert report.classification.taxonomy_version == TAXONOMY_VERSION
    assert report.classification.schema_version == DIAGNOSTIC_SCHEMA_VERSION


def test_classification_a_class_is_valid(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert report.classification.a_class in A_CLASSES
    assert report.classification.f_family in F_FAMILIES


def test_classification_confidence_bounded(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    assert 0.0 <= report.classification.confidence <= 1.0


def test_classifier_demotes_a11_when_sensitivity_too_high(pauli):
    """Mackinnon-Paternostro 2026 fragility demotion (NR-159):

    A synthetic ClassificationResult input via the public ``classify_mechanism``
    must demote A11 to A12 whenever the initial-state-sensitivity is large
    even if ``c_1`` is exactly zero.
    """
    from dataclasses import replace

    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=10, seed=42)

    # Force a "fragile" Mpemba candidate: zero c_1 but high sensitivity.
    assert report.mpemba is not None
    fragile_mpemba = replace(report.mpemba, overlap_c1=1e-12, is_mpemba_candidate=True)
    fragile_lep = replace(report.lep, initial_state_sensitivity=0.5)

    cls = classify_mechanism(
        spectral=report.spectral,
        nonnorm=report.nonnorm,
        relaxation=report.relaxation,
        resolvent=report.resolvent,
        transient=report.transient,
        lep=fragile_lep,
        mpemba=fragile_mpemba,
    )
    assert cls.a_class == "A12", f"expected A12 (demoted), got {cls.a_class}"

    # Conversely, a "robust" Mpemba candidate keeps the A11 verdict.
    robust_lep = replace(report.lep, initial_state_sensitivity=0.001)
    cls_robust = classify_mechanism(
        spectral=report.spectral,
        nonnorm=report.nonnorm,
        relaxation=report.relaxation,
        resolvent=report.resolvent,
        transient=report.transient,
        lep=robust_lep,
        mpemba=fragile_mpemba,
    )
    assert cls_robust.a_class == "A11", (
        f"expected A11 (robust), got {cls_robust.a_class}"
    )


def test_classify_mechanism_callable_directly(pauli):
    L = build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.outer(plus, plus.conj())
    report = diagnose(L, rho_initial=rho0, bootstrap_B=20, seed=42)
    cls = classify_mechanism(
        spectral=report.spectral,
        nonnorm=report.nonnorm,
        relaxation=report.relaxation,
        resolvent=report.resolvent,
        transient=report.transient,
        lep=report.lep,
        mpemba=report.mpemba,
    )
    assert cls.a_class in A_CLASSES
