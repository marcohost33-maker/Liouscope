"""Trajectory propagation backends (issue #162).

The relaxation layer can propagate ``rho(t)`` either through the materialised
dense ``expm(L t)`` (reference) or through the Al-Mohy & Higham exponential
action (``scipy.sparse.linalg.expm_multiply``). These tests pin the acceptance
list of #162:

* dense vs action agreement on ordinary GKSL fixtures within a declared
  tolerance;
* trace / Hermiticity on propagated states, positivity drift measured;
* rate-unit metamorphic invariance ``L -> cL, t -> t/c``;
* non-uniform, two-scale and repeated-point grids;
* dense and sparse Liouvillians;
* the extreme #156 fixture fails closed instead of hanging or returning NaN;

plus the audit contract: the ``"auto"`` rule is deterministic, keeps every
system below ``ACTION_MIN_DIM`` on the historical dense formula bit-for-bit,
and the backend actually used is recorded in ``RelaxationResult``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.relaxation import (
    UnrepresentableTrajectoryError,
    _evolve,
    compute_relaxation_layer,
)
from liouscope.numerics import propagation
from liouscope.numerics.kronecker import unvec, vec
from liouscope.numerics.propagation import (
    ACTION_MIN_DIM,
    BACKEND_ACTION,
    BACKEND_DENSE,
    action_matvec_bound,
    propagate_trajectory,
    select_trajectory_backend,
)
from liouscope.sparse.build import build_sparse_liouvillian

_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SZ = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_SP = _SM.conj().T

# Declared forward tolerance for dense-vs-action agreement on trace-one
# states (entries bounded by 1). Measured on these fixtures: <= 2e-14.
_AGREE_ATOL = 1.0e-11


def _site(op: np.ndarray, i: int, n_sites: int) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for k in range(n_sites):
        out = np.kron(out, op if k == i else np.eye(2, dtype=complex))
    return out


def _chain_operators(
    n_sites: int, *, gamma: float = 0.7, J: float = 1.0, h: float = 0.3
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Transverse-field Ising chain with site-local amplitude damping."""
    H = sum(
        (J * _site(_SZ, i, n_sites) @ _site(_SZ, i + 1, n_sites) for i in range(n_sites - 1)),
        start=np.zeros((2**n_sites, 2**n_sites), dtype=complex),
    )
    H = H + sum(h * _site(_SX, i, n_sites) for i in range(n_sites))
    jumps = [np.sqrt(gamma) * _site(_SM, i, n_sites) for i in range(n_sites)]
    return H, jumps


def _chain(n_sites: int, **kw: float) -> np.ndarray:
    H, jumps = _chain_operators(n_sites, **kw)
    return build_liouvillian(H, jumps)


def _ground(d: int) -> np.ndarray:
    rho = np.zeros((d, d), dtype=complex)
    rho[-1, -1] = 1.0  # all spins down: the damping target, far from rho_ss under h
    return rho


def _plus_state(d: int) -> np.ndarray:
    psi = np.ones(d, dtype=complex) / np.sqrt(d)
    return np.outer(psi, psi.conj())


def _dense_reference(L: np.ndarray, rho0: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The historical formula, written out independently of the module."""
    d = rho0.shape[0]
    out = np.empty((grid.size, d, d), dtype=complex)
    for k, t in enumerate(grid):
        out[k] = rho0 if t == 0.0 else unvec(sla.expm(L * t) @ vec(rho0), d=d)
    return out


# --------------------------------------------------------------------------
# Audit contract of the "auto" rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", [1, 2])
def test_auto_keeps_small_systems_on_the_historical_dense_formula_bitwise(
    n_sites: int,
) -> None:
    """Below ACTION_MIN_DIM nothing changes -- not even the last bit."""
    L = _chain(n_sites)
    assert L.shape[0] < ACTION_MIN_DIM
    d = 2**n_sites
    grid = np.linspace(0.0, 10.0, 80)
    np.testing.assert_array_equal(
        _evolve(L, _plus_state(d), grid), _dense_reference(L, _plus_state(d), grid)
    )


def test_auto_selects_the_action_for_a_moderate_generator_at_the_threshold() -> None:
    L = _chain(3)
    assert L.shape[0] == ACTION_MIN_DIM
    backend, bound = select_trajectory_backend(L, np.linspace(0.0, 10.0, 80))
    assert backend == BACKEND_ACTION
    assert 0.0 < bound < 1.0e4


def test_auto_keeps_a_stiff_generator_on_the_dense_path() -> None:
    """Action cost grows like ||L|| t; dense only like log(||L|| t)."""
    L = _chain(3, gamma=1.0e7)
    grid = np.linspace(0.0, 10.0, 80)
    backend, bound = select_trajectory_backend(L, grid)
    assert backend == BACKEND_DENSE
    assert bound > propagation.dense_matvec_estimate(L, grid)


def test_auto_prefers_dense_when_the_action_is_affordable_but_costlier() -> None:
    """Within budget is not enough: the action must also beat the dense estimate."""
    L = _chain(3, gamma=1.0e3)
    grid = np.linspace(0.0, 10.0, 80)
    bound = action_matvec_bound(L, grid)
    assert propagation.dense_matvec_estimate(L, grid) < bound
    assert bound < propagation.DEFAULT_ACTION_MATVEC_BUDGET
    assert select_trajectory_backend(L, grid)[0] == BACKEND_DENSE


def test_auto_falls_back_to_dense_on_a_grid_it_cannot_step() -> None:
    L = _chain(3)
    backend, bound = select_trajectory_backend(L, np.array([0.0, 2.0, 1.0]))
    assert backend == BACKEND_DENSE
    assert bound == np.inf


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        propagate_trajectory(_chain(1), vec(_plus_state(2)), np.array([0.0, 1.0]), backend="rk45")


def test_forced_action_refuses_a_grid_it_cannot_step() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        propagate_trajectory(
            _chain(1), vec(_plus_state(2)), np.array([0.0, 2.0, 1.0]), backend=BACKEND_ACTION
        )
    with pytest.raises(ValueError, match="non-negative"):
        propagate_trajectory(
            _chain(1), vec(_plus_state(2)), np.array([-1.0, 0.0]), backend=BACKEND_ACTION
        )


# --------------------------------------------------------------------------
# Dense vs action agreement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_sites", [1, 2, 3])
@pytest.mark.parametrize("initial", ["ground", "plus"])
def test_action_agrees_with_dense_reference_on_gksl_fixtures(
    n_sites: int, initial: str
) -> None:
    L = _chain(n_sites)
    d = 2**n_sites
    rho0 = _ground(d) if initial == "ground" else _plus_state(d)
    grid = np.linspace(0.0, 10.0, 80)
    action = _evolve(L, rho0, grid, backend=BACKEND_ACTION)
    dense = _evolve(L, rho0, grid, backend=BACKEND_DENSE)
    np.testing.assert_allclose(action, dense, rtol=0.0, atol=_AGREE_ATOL)
    assert action[0] is not dense[0]
    np.testing.assert_array_equal(action[0], rho0)  # t = 0 is rho_0 exactly


@pytest.mark.parametrize(
    "grid",
    [
        pytest.param(np.geomspace(1.0e-3, 20.0, 60), id="geometric"),
        pytest.param(
            np.concatenate([np.linspace(0.0, 0.5, 40), np.linspace(0.6, 40.0, 40)]),
            id="two-scale",
        ),
        pytest.param(np.array([0.0, 0.0, 0.3, 0.3, 0.3, 1.7, 5.0, 5.0]), id="repeated"),
        pytest.param(np.array([2.5, 3.0, 7.0]), id="starts-after-zero"),
        pytest.param(np.array([0.0]), id="single-point"),
        pytest.param(np.array([], dtype=float), id="empty"),
    ],
)
def test_action_agrees_with_dense_on_non_uniform_grids(grid: np.ndarray) -> None:
    L = _chain(3)
    rho0 = _ground(8)
    action = _evolve(L, rho0, grid, backend=BACKEND_ACTION)
    dense = _evolve(L, rho0, grid, backend=BACKEND_DENSE)
    assert action.shape == (grid.size, 8, 8)
    np.testing.assert_allclose(action, dense, rtol=0.0, atol=_AGREE_ATOL)


def test_repeated_grid_points_repeat_the_state_exactly() -> None:
    grid = np.array([0.0, 0.3, 0.3, 1.7, 1.7])
    traj = _evolve(_chain(3), _ground(8), grid, backend=BACKEND_ACTION)
    np.testing.assert_array_equal(traj[1], traj[2])
    np.testing.assert_array_equal(traj[3], traj[4])


# --------------------------------------------------------------------------
# Physical invariants, measured on the action path
# --------------------------------------------------------------------------


def test_action_states_stay_trace_one_hermitian_and_positive() -> None:
    """Trace and Hermiticity to round-off; positivity drift measured, bounded."""
    L = _chain(3)
    grid = np.linspace(0.0, 40.0, 200)  # well into the steady state
    traj = _evolve(L, _ground(8), grid, backend=BACKEND_ACTION)
    traces = np.einsum("kii->k", traj)
    herm_defect = np.max(np.abs(traj - np.conj(np.swapaxes(traj, 1, 2))))
    min_eig = min(float(np.linalg.eigvalsh(0.5 * (r + r.conj().T)).min()) for r in traj)
    assert np.max(np.abs(traces - 1.0)) < 1.0e-12
    assert herm_defect < 1.0e-12
    # The initial state is pure (seven exact zero eigenvalues), so this is the
    # tightest possible probe of how far round-off pushes the spectrum below 0.
    assert min_eig > -1.0e-12


# --------------------------------------------------------------------------
# Rate-unit metamorphic invariance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", [BACKEND_DENSE, BACKEND_ACTION])
@pytest.mark.parametrize("c", [1.0e-3, 7.0, 1.0e3])
def test_rate_unit_change_leaves_the_trajectory_invariant(backend: str, c: float) -> None:
    """``L -> cL`` with ``t -> t/c`` is a change of time unit, not of physics."""
    L = _chain(3)
    rho0 = _ground(8)
    grid = np.concatenate([np.linspace(0.0, 1.0, 20), np.linspace(1.5, 12.0, 20)])
    base = _evolve(L, rho0, grid, backend=backend)
    scaled = _evolve(c * L, rho0, grid / c, backend=backend)
    np.testing.assert_allclose(scaled, base, rtol=0.0, atol=_AGREE_ATOL)


# --------------------------------------------------------------------------
# Sparse generators
# --------------------------------------------------------------------------


def test_sparse_generator_matches_the_dense_reference() -> None:
    H, jumps = _chain_operators(3)
    L_sparse = build_sparse_liouvillian(H, jumps)
    L_dense = build_liouvillian(H, jumps)
    np.testing.assert_allclose(L_sparse.toarray(), L_dense, rtol=0.0, atol=1.0e-14)

    grid = np.geomspace(1.0e-2, 15.0, 30)
    v0 = vec(_ground(8))
    auto = propagate_trajectory(L_sparse, v0, grid)
    assert auto.backend == BACKEND_ACTION
    reference = propagate_trajectory(L_dense, v0, grid, backend=BACKEND_DENSE)
    np.testing.assert_allclose(auto.states, reference.states, rtol=0.0, atol=_AGREE_ATOL)

    forced_dense = propagate_trajectory(L_sparse, v0, grid, backend=BACKEND_DENSE)
    np.testing.assert_array_equal(forced_dense.states, reference.states)


# --------------------------------------------------------------------------
# Cost bound and fail-closed behaviour
# --------------------------------------------------------------------------


class _CountingArray(np.ndarray):
    """ndarray whose ``.dot`` counts matrix-vector products."""

    products = 0

    def dot(self, other, out=None):  # type: ignore[override]
        other_arr = np.asarray(other)
        _CountingArray.products += 1 if other_arr.ndim == 1 else other_arr.shape[1]
        return np.asarray(self).dot(other_arr)


@pytest.mark.parametrize("gamma", [0.7, 20.0])
def test_action_cost_bound_is_an_upper_bound_on_scipys_products(gamma: float) -> None:
    """The auto rule relies on the bound over-, never under-estimating SciPy.

    ``gamma = 20`` makes ``tr(L)/n`` large, so the bound is only valid if the
    trace shift it assumes is actually passed to SciPy.
    """
    L = _chain(3, gamma=gamma)
    grid = np.linspace(0.0, 10.0, 80)
    counting = L.view(_CountingArray)
    _CountingArray.products = 0
    propagate_trajectory(counting, vec(_ground(8)), grid, backend=BACKEND_ACTION)
    counted = _CountingArray.products
    assert counted > 0, "probe no longer sees SciPy's products; revisit the cost model"
    assert counted <= action_matvec_bound(L, grid)


def _extreme_thermalising_qubit() -> np.ndarray:
    """The #156 fixture: finite Liouvillian whose exponential is not representable."""
    rate = 8.5e307
    L = build_liouvillian(
        np.zeros((2, 2), dtype=complex),
        [np.sqrt(rate) * _SM, np.sqrt(rate) * _SP],
    )
    assert np.all(np.isfinite(L))
    return L


@pytest.mark.parametrize("backend", ["auto", BACKEND_DENSE, BACKEND_ACTION])
def test_extreme_fixture_fails_closed_promptly_on_every_backend(backend: str) -> None:
    L = _extreme_thermalising_qubit()
    grid = np.array([0.0, 1.1])
    start = time.perf_counter()
    with pytest.raises(UnrepresentableTrajectoryError):
        _evolve(L, _plus_state(2), grid, backend=backend)
    assert time.perf_counter() - start < 5.0


def test_forced_action_refuses_an_over_budget_stiff_generator_without_running() -> None:
    L = _chain(3, gamma=1.0e7)
    grid = np.linspace(0.0, 10.0, 80)
    start = time.perf_counter()
    with pytest.raises(UnrepresentableTrajectoryError, match="exceeds the budget"):
        propagate_trajectory(L, vec(_ground(8)), grid, backend=BACKEND_ACTION)
    assert time.perf_counter() - start < 5.0


def test_action_refuses_a_non_finite_scaled_generator() -> None:
    L = np.eye(ACTION_MIN_DIM, dtype=complex) * -1.0e308
    with pytest.raises(UnrepresentableTrajectoryError):
        propagate_trajectory(
            L,
            np.ones(ACTION_MIN_DIM, dtype=complex),
            np.array([0.0, 2.0]),
            backend=BACKEND_ACTION,
            max_action_matvecs=np.inf,
        )


def test_action_normalises_a_scipy_runtime_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    def _warn(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeWarning("synthetic overflow inside expm_multiply")

    monkeypatch.setattr(propagation, "expm_multiply", _warn)
    with pytest.raises(UnrepresentableTrajectoryError, match="expm_multiply"):
        propagate_trajectory(
            _chain(1), vec(_plus_state(2)), np.array([0.0, 1.0]), backend=BACKEND_ACTION
        )


def test_action_rejects_a_non_finite_returned_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        propagation, "expm_multiply", lambda A, v, **_kw: np.full_like(v, np.nan)
    )
    with pytest.raises(UnrepresentableTrajectoryError, match="became non-finite"):
        propagate_trajectory(
            _chain(1), vec(_plus_state(2)), np.array([0.0, 1.0]), backend=BACKEND_ACTION
        )


# --------------------------------------------------------------------------
# Relaxation layer: audit field and end-to-end agreement
# --------------------------------------------------------------------------


def test_relaxation_layer_records_the_backend_it_used() -> None:
    small = compute_relaxation_layer(_chain(1), rho_initial=_plus_state(2), bootstrap_B=20)
    assert small.trajectory_backend == BACKEND_DENSE
    moderate = compute_relaxation_layer(_chain(3), rho_initial=_ground(8), bootstrap_B=20)
    assert moderate.trajectory_backend == BACKEND_ACTION


def test_relaxation_layer_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        compute_relaxation_layer(_chain(1), trajectory_backend="rk45", bootstrap_B=20)


def test_relaxation_layer_is_backend_independent_end_to_end() -> None:
    """Curves agree to round-off, and the fitted rate far below its CI width."""
    L = _chain(3)
    rho0 = _ground(8)
    dense = compute_relaxation_layer(
        L, rho_initial=rho0, bootstrap_B=20, trajectory_backend=BACKEND_DENSE
    )
    action = compute_relaxation_layer(
        L, rho_initial=rho0, bootstrap_B=20, trajectory_backend=BACKEND_ACTION
    )
    assert dense.trajectory_backend == BACKEND_DENSE
    assert action.trajectory_backend == BACKEND_ACTION
    np.testing.assert_allclose(
        action.trace_distance_curve, dense.trace_distance_curve, rtol=0.0, atol=1.0e-10
    )
    np.testing.assert_allclose(
        action.relative_entropy_curve, dense.relative_entropy_curve, rtol=0.0, atol=1.0e-10
    )
    assert action.aicc_model == dense.aicc_model
    assert action.beta_D == pytest.approx(dense.beta_D, rel=1.0e-8)


def test_sparse_input_is_accepted_by_the_propagator_only() -> None:
    """The relaxation layer itself stays dense-only; the propagator is not."""
    H, jumps = _chain_operators(3)
    L_sparse = build_sparse_liouvillian(H, jumps)
    assert sp.issparse(L_sparse)
    result = propagate_trajectory(L_sparse, vec(_ground(8)), np.array([0.0, 1.0]))
    assert result.states.shape == (2, 64)
