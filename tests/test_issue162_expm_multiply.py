"""Issue #162: exponential-action trajectory backend."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.relaxation import (
    _evolve,
    _evolve_with_backend,
    _trajectory_audit,
    compute_relaxation_layer,
)
from liouscope.numerics.kronecker import unvec, vec

_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_RHO_PLUS = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=complex)


def _generator(scale: float = 1.0) -> np.ndarray:
    return build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.sqrt(0.7 * scale) * _SM],
    )


def _dense_reference(L: np.ndarray, grid: np.ndarray) -> np.ndarray:
    state = vec(_RHO_PLUS)
    out = np.empty((grid.size, 2, 2), dtype=complex)
    for k, t in enumerate(grid):
        if t == 0.0:
            out[k] = _RHO_PLUS
        else:
            out[k] = unvec(sla.expm(L * t) @ state, d=2)
    return out


@pytest.mark.parametrize(
    "grid",
    [
        np.linspace(0.0, 4.0, 17),
        np.array([0.0, 0.01, 0.1, 0.7, 2.0, 4.0]),
    ],
)
def test_action_backend_agrees_with_dense_reference(grid: np.ndarray) -> None:
    """Action propagation matches the former full-exponential formula."""
    L = _generator()
    actual = _evolve(L, _RHO_PLUS, grid)
    expected = _dense_reference(L, grid)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=2.0e-14)


def test_exact_linspace_uses_interval_backend_and_preserves_t0_bitwise() -> None:
    grid = np.linspace(0.0, 3.0, 31)
    traj, backend = _evolve_with_backend(_generator(), _RHO_PLUS, grid)
    assert backend == "expm_multiply_interval"
    np.testing.assert_array_equal(traj[0], _RHO_PLUS)


def test_nonuniform_grid_uses_pointwise_backend() -> None:
    grid = np.array([0.0, 0.01, 0.13, 0.7, 3.0])
    _traj, backend = _evolve_with_backend(_generator(), _RHO_PLUS, grid)
    assert backend == "expm_multiply_pointwise"


def test_sparse_and_dense_action_paths_agree() -> None:
    L = _generator()
    grid = np.linspace(0.0, 2.0, 21)
    dense = _evolve(L, _RHO_PLUS, grid)
    sparse = _evolve(sp.csr_matrix(L), _RHO_PLUS, grid)
    np.testing.assert_allclose(sparse, dense, rtol=3.0e-13, atol=3.0e-14)


def test_rate_unit_metamorphism_L_to_cL_t_to_t_over_c() -> None:
    """Pure rate-unit changes must leave the propagated states invariant."""
    L = _generator()
    grid = np.array([0.0, 0.02, 0.11, 0.8, 3.0])
    reference = _evolve(L, _RHO_PLUS, grid)
    for c in (1.0e-6, 1.0e6):
        changed = _evolve(c * L, _RHO_PLUS, grid / c)
        np.testing.assert_allclose(
            changed, reference, rtol=5.0e-12, atol=5.0e-13
        )


def test_trajectory_audit_measures_physicality_drift_without_repairing() -> None:
    traj = _evolve(_generator(), _RHO_PLUS, np.linspace(0.0, 3.0, 17))
    trace_error, hermiticity, min_eval = _trajectory_audit(traj)
    assert trace_error < 1.0e-12
    assert hermiticity < 1.0e-12
    assert min_eval > -1.0e-12

    perturbed = traj.copy()
    perturbed[-1, 0, 0] -= 1.0e-5
    _te, _hd, drift = _trajectory_audit(perturbed)
    assert drift <= min_eval
    # The audit records the input; it does not clip it back to positivity.
    assert perturbed[-1, 0, 0] == traj[-1, 0, 0] - 1.0e-5


def test_relaxation_result_records_backend_and_audit_fields() -> None:
    report = compute_relaxation_layer(
        _generator(),
        rho_initial=_RHO_PLUS,
        t_grid=np.linspace(0.0, 2.0, 16),
        bootstrap_B=5,
        seed=1,
    )
    assert report.trajectory_backend == "expm_multiply_interval"
    assert report.trajectory_max_trace_error < 1.0e-12
    assert report.trajectory_max_hermiticity_defect < 1.0e-12
    assert report.trajectory_min_eigenvalue > -1.0e-12
