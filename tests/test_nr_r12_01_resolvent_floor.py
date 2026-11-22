"""NR-R12-01: the resolvent norm abstains below the rounding floor.

Invariant: every path that returns ``||(zI - L)^{-1}||_2`` either returns a
value whose rounding error is bounded against ``1/sigma_min(zI - L)``, or
raises ``LinAlgError`` when ``sigma_min(zI - L) <= n eps ||zI - L||_F``;
NaN / inf fall into the refusal.

Before the fix a valid GKSL generator (n = 144) at the stationary shift
``z = 0`` returned ``~1e15`` from all three paths (dense inverse + SVD for
``n <= 128``, SuperLU power iteration and the shared Schur form above it).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import liouscope.numerics.resolvent as resolvent
from liouscope.core.lindblad import build_liouvillian
from liouscope.diagnostics.resolvent import (
    compute_resolvent_layer,
    resolvent_peak_curve,
    ridge_fwhm,
)

EPS = float(np.finfo(float).eps)
# Positive control tolerance. Measured relative deviation from the dense SVD
# reference: ~1e-14 (dense, SuperLU), ~2e-14 (Schur, Lanczos rtol 1e-10 on the
# squared value). 1e-9 leaves the power-iteration tolerance (1e-9) as the bound.
RTOL = 1.0e-9


def _gksl(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    jumps = [
        (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))) / d
        for _ in range(2)
    ]
    return build_liouvillian((H + H.conj().T) / np.sqrt(d), jumps)


def _direct(L: np.ndarray, z: complex) -> float:
    return resolvent.resolvent_norm(L, z)


def _schur(L: np.ndarray, z: complex) -> float:
    return resolvent.ShiftedSchurResolvent(L).norm(z)


def _evaluator(L: np.ndarray, z: complex) -> float:
    return resolvent.resolvent_norm_evaluator(L)(z)


# (dimension d, path): d = 11 -> n = 121 (dense), d = 12 -> n = 144 (SuperLU /
# Schur). The evaluator routes by n, so it covers dense AND Schur.
PATHS = [
    pytest.param(11, _direct, id="dense-n121"),
    pytest.param(12, _direct, id="superlu-n144"),
    pytest.param(12, _schur, id="schur-n144"),
    pytest.param(11, _evaluator, id="evaluator-n121"),
    pytest.param(12, _evaluator, id="evaluator-n144"),
]


def _reference(L: np.ndarray, z: complex) -> tuple[float, float]:
    A = z * np.eye(L.shape[0]) - L
    return float(sla.svdvals(A)[-1]), resolvent._resolution_floor(A)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize(("d", "norm"), PATHS)
def test_stationary_shift_of_gksl_generator_is_refused(d, norm):
    """RED before the fix: 4.4e15 (dense), 9.5e15 (SuperLU), 1.6e15 (Schur)."""
    L = _gksl(d)
    sigma_min, floor = _reference(L, 0.0)
    assert sigma_min < 1.0e-3 * floor  # the case is far below the floor
    with pytest.raises(np.linalg.LinAlgError):
        norm(L, 0.0)


@pytest.mark.parametrize(("d", "norm"), PATHS)
def test_positive_control_matches_dense_svd(d, norm):
    L = _gksl(d)
    z = 0.02 + 0.01j
    sigma_min, floor = _reference(L, z)
    assert sigma_min > 1.0e6 * floor
    assert norm(L, z) == pytest.approx(1.0 / sigma_min, rel=RTOL)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
def test_small_dense_numerically_singular_matrix_is_refused():
    # Rank 3 of 4 behind a random unitary: zI - L at z = 0 is singular in exact
    # arithmetic, but its computed pivots are rounding noise, not zero, so
    # scipy.linalg.solve does not raise by itself.
    rng = np.random.default_rng(7)
    Q, _ = np.linalg.qr(rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)))
    L = Q @ np.diag([-1.0, -2.0, -3.0, 0.0]) @ Q.conj().T
    assert float(sla.svdvals(L)[-1]) != 0.0
    with pytest.raises(np.linalg.LinAlgError):
        resolvent.resolvent_norm(L, 0.0)


@pytest.mark.parametrize("norm", [_direct, _schur], ids=["direct", "schur"])
@pytest.mark.parametrize("factor", [4.0, 0.25], ids=["above", "below"])
def test_threshold_is_pinned_on_both_sides(norm, factor):
    """sigma_min = 4 * floor is resolved, sigma_min = floor / 4 is refused.

    Upper triangular, NOT diagonal: identity with one exact coupling
    ``A[0, 1] = 2**-30`` and an isolated ``A[-1, -1] = s``. The inverse is
    exact, sigma_min = s, so only the floor decides. A floor without the
    factor n (operand mutant) would accept the "below" case.
    """
    n = 16 if norm is _direct else 130
    A = np.eye(n, dtype=complex)
    A[0, 1] = 2.0**-30
    floor = n * EPS * float(np.sqrt(n))  # ||A||_F^2 = n - 1 + 2**-60 + s^2
    s = factor * floor
    A[-1, -1] = s
    assert float(sla.svdvals(A)[-1]) == pytest.approx(s, rel=1.0e-6)
    L = -A
    if factor > 1.0:
        assert norm(L, 0.0) == pytest.approx(1.0 / s, rel=RTOL)
    else:
        with pytest.raises(np.linalg.LinAlgError):
            norm(L, 0.0)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize(("d", "norm"), PATHS)
@pytest.mark.parametrize("c", [1.0e-30, 2.0**-40, 1.0e30])
def test_unit_rescale_keeps_the_decision(d, norm, c):
    """(L, z) -> (cL, cz): same refusal at z = 0, norm scales by 1/c."""
    L = _gksl(d)
    z = 0.02 + 0.01j
    assert c * norm(c * L, c * z) == pytest.approx(norm(L, z), rel=RTOL)
    with pytest.raises(np.linalg.LinAlgError):
        norm(c * L, 0.0)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize(("d", "norm"), PATHS)
def test_permutation_and_unitary_similarity_keep_the_decision(d, norm):
    L = _gksl(d)
    n = L.shape[0]
    rng = np.random.default_rng(3)
    P = np.eye(n)[rng.permutation(n)]
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    z = 0.02 + 0.01j
    expected = norm(L, z)
    for M in (P @ L @ P.T, Q @ L @ Q.conj().T):
        assert norm(M, z) == pytest.approx(expected, rel=RTOL)
        with pytest.raises(np.linalg.LinAlgError):
            norm(M, 0.0)


def test_strict_caller_fp_policy_on_a_tiny_operator():
    # A 1e-300 operator: the floor (~1e-314) is subnormal. Computing it must not
    # raise under the caller's np.errstate(all="raise").
    L = -np.diag([1.0e-300, 2.0e-300, 3.0e-300]).astype(complex)
    with np.errstate(all="raise"):
        assert resolvent.resolvent_norm(L, 0.0) == pytest.approx(1.0e300, rel=1e-12)


@pytest.mark.parametrize(
    ("norm_inv", "floor"),
    [
        (float("nan"), 1.0e-12),
        (float("inf"), 1.0e-12),
        (0.0, 1.0e-12),
        (1.0, float("nan")),
        (1.0, float("inf")),
        (2.0**10, 2.0**-10),  # exactly ON the floor: sigma_min == floor is refused
        (1.0e300, 1.0e10),  # product overflows -> refused
    ],
)
def test_acceptance_is_fail_closed(norm_inv, floor):
    with pytest.raises(np.linalg.LinAlgError):
        resolvent._accepted_norm(norm_inv, floor, 0.0)


def test_schur_norm_refuses_non_finite_and_on_the_floor(monkeypatch):
    L = _gksl(12)
    engine = resolvent.ShiftedSchurResolvent(L)
    monkeypatch.setattr(engine, "sigma_min", lambda z: float("inf"))
    with pytest.raises(np.linalg.LinAlgError):
        engine.norm(0.5)
    # The zero operator at z = 0: sigma_min == floor == 0.0 -- refused, not 1/0.
    zero = resolvent.ShiftedSchurResolvent(np.zeros((130, 130), dtype=complex))
    with pytest.raises(np.linalg.LinAlgError):
        zero.norm(0.0)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
def test_overflowing_dense_inverse_is_a_linalg_error():
    L = -np.diag([1.0e-310, 1.0]).astype(complex)
    with pytest.raises(np.linalg.LinAlgError):
        resolvent.resolvent_norm(L, 0.0)


def test_exactly_singular_superlu_factor_is_a_linalg_error():
    with pytest.raises(np.linalg.LinAlgError):
        resolvent.resolvent_norm(np.zeros((130, 130), dtype=complex), 0.0)


def test_power_iteration_zero_estimate_is_refused(monkeypatch):
    # A solve returning exactly zero is the documented sigma == 0.0 exit of the
    # power iteration; before the fix it returned a resolvent norm of 0.0.
    class _ZeroLU:
        def solve(self, b, trans="N"):
            return np.zeros_like(b)

    monkeypatch.setattr(resolvent.spla, "splu", lambda A: _ZeroLU())
    with pytest.raises(np.linalg.LinAlgError):
        resolvent.resolvent_norm(-np.eye(130, dtype=complex), 0.5)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
def test_resolvent_layer_reports_unresolved_cells_as_nan(pauli):
    """Complete mediation through the public layer: no crash, no finite fake.

    The legacy D11b scan sits at the ABSOLUTE sigma = 1e-3 right of the
    spectrum. For a generator at rate scale 1e12 that is below the rounding
    floor n eps ||zI - L||_F ~ 3e-3, so cells refuse and the peak and width
    are unknown (NaN) instead of a number made of rounding noise.
    """
    L = 1.0e12 * build_liouvillian(0.5 * pauli["X"], [pauli["Z"]], [0.3])
    _, norms = resolvent_peak_curve(L)
    assert np.isnan(norms).any()
    res = compute_resolvent_layer(L)
    assert np.isnan(res.resolvent_peak)
    assert np.isnan(res.ridge_fwhm)
    # The scale-relative scan (sigma = 1e-3 ||L||_F) stays resolved.
    assert np.isfinite(res.resolvent_peak_scaled)


def test_ridge_fwhm_of_an_unknown_peak_is_nan():
    omegas = np.linspace(-1.0, 1.0, 5)
    norms = np.array([1.0, 2.0, float("nan"), 2.0, 1.0])
    assert np.isnan(ridge_fwhm(omegas, norms))


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
@pytest.mark.parametrize(
    ("n", "big"), [(2, 1.0e200), (130, 1.0e60)], ids=["dense", "superlu"]
)
def test_exact_diagonal_with_extreme_dynamic_range_is_resolved(n, big):
    """PR-R12-04: an exactly diagonal zI - L has the floor 0.

    diag(0, -big) at z = 1/big has the exact norm ``big`` although sigma_min
    lies far below n eps ||zI - L||_F. (The Schur path is pinned by
    test_issue178_schur_resolvent.py::test_a_shift_that_underflows_...)
    """
    L = np.diag(np.r_[0.0, -big, -np.ones(n - 2)]).astype(complex)
    assert resolvent.resolvent_norm(L, 1.0 / big) == pytest.approx(big, rel=1e-12)


def test_superlu_norm_beyond_the_squared_range_is_refused_documented_limit():
    """GRENZE: the power iteration works on ||R||^2 and its 2-norm squares again.

    Above ``||R|| ~ 1e77`` the sum of squares overflows.

    diag(0, -1e200, -1, ...) at z = 1e-200 (n = 130) has the exact norm 1e200,
    which the dense path returns; the SuperLU path refuses instead of iterating
    on inf / NaN.
    """
    L = np.diag(np.r_[0.0, -1.0e200, -np.ones(128)]).astype(complex)
    with pytest.raises(np.linalg.LinAlgError):
        resolvent.resolvent_norm(L, 1.0e-200)


@pytest.mark.filterwarnings("ignore::scipy.linalg.LinAlgWarning")
def test_one_coupling_entry_takes_the_normwise_floor_documented_limit():
    """The same dynamic range with ONE off-diagonal entry is refused (GRENZE).

    [[1e-200, 1e-300], [0, 1e200]] still has sigma_min ~ 1e-200, but the floor
    is normwise again; a structural method could resolve it, this one does not.
    """
    A = np.array([[1.0e-200, 1.0e-300], [0.0, 1.0e200]], dtype=complex)
    with pytest.raises(np.linalg.LinAlgError):
        resolvent.resolvent_norm(-A, 0.0)
