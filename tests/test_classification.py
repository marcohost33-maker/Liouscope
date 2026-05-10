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
