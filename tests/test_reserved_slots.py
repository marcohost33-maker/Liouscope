"""Contract tests for reserved diagnostic slots and the U1 nominal floor.

Issue #71 C2 (reserved D21-D23 slots) and B5 (explicit U1 nominal floor):
these pin the code-level contract so the "24 diagnostics" schema name stays
honest -- D21-D23 are reserved, not silently claimed as implemented -- and so
the U1 placeholder has a single named source instead of a magic number.
"""

from __future__ import annotations

import numpy as np

from liouscope._consts import (
    RESERVED_DIAGNOSTIC_SLOTS,
    U1_NOMINAL_FLOOR,
)
from liouscope._types import RelaxationResult
from liouscope.diagnostics.uncertainty import compute_uncertainty_layer


def test_reserved_slots_cover_d21_d23_only():
    assert set(RESERVED_DIAGNOSTIC_SLOTS) == {"D21", "D22", "D23"}


def test_reserved_slots_are_marked_not_implemented():
    for note in RESERVED_DIAGNOSTIC_SLOTS.values():
        assert "reserved" in note.lower()
        assert "not implemented" in note.lower()


def test_reserved_slots_disjoint_from_implemented_ids():
    # The implemented set is D1-D20 + D24; the reserved ids must not collide.
    implemented = {f"D{i}" for i in range(1, 21)} | {"D24"}
    assert implemented.isdisjoint(RESERVED_DIAGNOSTIC_SLOTS)


def _relaxation_with_ci(lo: float, hi: float) -> RelaxationResult:
    """A minimal RelaxationResult; only ``bca_ci_beta`` drives the U-layer."""
    return RelaxationResult(
        von_neumann_entropy=0.0,
        relative_entropy_curve=np.zeros(2),
        fidelity_curve=np.zeros(2),
        entanglement_asymmetry=None,
        fits={},
        aicc_model="M0",
        beta_D=0.5,
        bca_ci_beta=(lo, hi),
    )


def test_u1_defaults_to_named_nominal_floor():
    u = compute_uncertainty_layer(_relaxation_with_ci(0.4, 0.6))
    assert u.solver_uncertainty == U1_NOMINAL_FLOOR


def test_u1_uses_supplied_solver_residual_when_given():
    u = compute_uncertainty_layer(_relaxation_with_ci(0.4, 0.6), solver_residual=3.0e-8)
    assert u.solver_uncertainty == 3.0e-8


def test_u2_is_none_unless_supplied():
    assert compute_uncertainty_layer(_relaxation_with_ci(0.4, 0.6)).size_uncertainty is None
    u = compute_uncertainty_layer(_relaxation_with_ci(0.4, 0.6), size_residual=0.01)
    assert u.size_uncertainty == 0.01
