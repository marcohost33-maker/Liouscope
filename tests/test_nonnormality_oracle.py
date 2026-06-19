"""Independent-oracle cross-checks for the non-normality layer (D8-D11).

The existing ``tests/test_nonnormality.py`` only asserts *signs* and *self-
consistency* (``eta > 0``, ``K > 0``, ``length >= 1``). A sign error, a wrong
normalisation, or a missing eigenvalue filter in
:mod:`liouscope.diagnostics.nonnormality` would pass every one of those checks.
This module closes that gap exactly as ``tests/test_qutip_spectral_oracle.py``
did for the spectral layer: it pins each diagnostic to a **closed-form** or an
**independent numerical oracle**, never to the library's own machinery.

Oracles used (all derived/verified outside the implementation):

* **D8 Henrici** ``eta_N`` -- the Schur-strict-upper Frobenius norm equals the
  unitarily-invariant departure-from-normality
  ``sqrt(||A||_F^2 - sum_i |lambda_i|^2)`` (Henrici 1962; see also Higham,
  "What Is a (Non)normal Matrix?", 2020). For an upper-triangular ``[[a, b],
  [0, d]]`` with ``a != d`` this is exactly ``|b|``.
* **D9 Petermann** ``K_j`` -- for a 2x2 matrix with distinct eigenvalues the
  per-mode condition number has the closed form
  ``K = tr(B0^H B0) / |lambda_i - lambda_j|^2`` with ``B0 = adj(lambda_i I - A)``
  (the adjugate / two-level result, e.g. Phys. Rev. Research 5, 033042 (2023);
  arXiv:1911.05191). Both eigenmodes of a 2x2 share the same Petermann factor.
* **D10 Kreiss** ``K(A)`` -- the continuous-time (Hurwitz) Kreiss constant
  ``K = sup_{Re z > 0} Re(z) ||(zI - A)^{-1}||`` (Kreiss matrix theorem; LeVeque
  & Trefethen; Mitchell, SIAM J. Matrix Anal. Appl. 41(4), 2020). Reference
  facts: ``K = 1`` for a normal Hurwitz matrix, ``K >= 1`` always, and an
  independent value from a 2D Nelder-Mead maximisation of the resolvent norm.
* **D11 Bohr arithmetic progression** -- the length of the longest AP of
  imaginary parts and the Pauli depth bound ``log_2(d)`` (Basso,
  arXiv:2510.07267, 2025). Verified on hand-constructed spectra with a known
  longest AP.

A note on D10 and the implementation's reference frame: the implementation
takes the supremum of ``sigma * ||((sigma + re_max) + i omega) I - L)^{-1}||``
with ``re_max`` the spectral abscissa. For Liouvillians the steady state pins
the abscissa at ``0``, so ``re_max = 0`` and the implementation reduces to the
standard continuous-time Kreiss constant. The oracle matrices below therefore
all have spectral abscissa ``0`` (eigenvalue at 0, the rest in the open left
half-plane), exercising the implementation in exactly the regime it is used.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla
from scipy.optimize import minimize

from liouscope import build_liouvillian
from liouscope.diagnostics.nonnormality import (
    bohr_arithmetic_progression,
    henrici_eta_n,
    kreiss_constant,
    petermann_factors,
)

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# =============================================================================
# D8 -- Henrici departure-from-normality.
# Closed form: eta_N(A) = sqrt(||A||_F^2 - sum_i |lambda_i|^2). For the upper-
# triangular family [[0, b], [0, -1]] (eigenvalues 0, -1) this is exactly |b|.
# =============================================================================


@pytest.mark.parametrize("b", [1.0, 2.5, 4.0])
def test_henrici_equals_invariant_closed_form_2x2(b):
    A = np.array([[0.0, b], [0.0, -1.0]], dtype=complex)
    eta = henrici_eta_n(A)
    # Hand value: ||A||_F^2 = b^2 + 1, |lambda|^2 sum = 0 + 1 = 1 => eta = |b|.
    assert eta == pytest.approx(abs(b), abs=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_henrici_matches_unitary_invariant_random(seed):
    # The Schur-strict-upper Frobenius norm must equal the rotation-invariant
    # departure-from-normality for ANY matrix, not just triangular ones.
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    eig = sla.eigvals(A)
    invariant = float(np.sqrt(np.linalg.norm(A, "fro") ** 2 - np.sum(np.abs(eig) ** 2)))
    assert henrici_eta_n(A) == pytest.approx(invariant, abs=1e-9)


def test_henrici_zero_iff_normal_negative_control():
    # Negative control: a genuinely non-normal matrix must NOT report ~0.
    # (Guards against an implementation that always returns 0.)
    A = np.array([[0.0, 3.0], [0.0, -1.0]], dtype=complex)
    assert henrici_eta_n(A) > 1.0  # closed form is 3.0; a wrong-zero impl fails here.


# =============================================================================
# D9 -- Petermann factors (mode condition numbers).
# Closed form for 2x2 distinct-eigenvalue A: K = tr(B0^H B0) / |li - lj|^2,
# B0 = adj(li I - A). Both modes of a 2x2 share the same factor.
# =============================================================================


def _petermann_closed_form_2x2(A: np.ndarray) -> float:
    a, b = A[0, 0], A[0, 1]
    c, d = A[1, 0], A[1, 1]
    tr, det = a + d, a * d - b * c
    disc = np.sqrt(tr * tr - 4 * det)
    l1, l2 = (tr + disc) / 2, (tr - disc) / 2
    B0 = np.array([[l1 - d, b], [c, l1 - a]], dtype=complex)  # adj(l1 I - A)
    return float(np.trace(B0.conj().T @ B0).real / abs(l1 - l2) ** 2)


@pytest.mark.parametrize("b", [1.0, 3.0, 5.0])
def test_petermann_matches_adjugate_closed_form(b):
    # Distinct, both-nonzero eigenvalues so nothing is filtered by the API.
    # Eigenvalue gap is 2 (|l1 - l2|^2 = 4 != 1) so a missing-denominator bug
    # would inflate the factor 4x and fail the closed-form comparison.
    A = np.array([[-0.5, b], [0.0, -2.5]], dtype=complex)
    _, K = petermann_factors(A)
    expected = _petermann_closed_form_2x2(A)
    assert K.size == 2
    # Both 2x2 modes carry the same Petermann factor.
    np.testing.assert_allclose(np.sort(K), [expected, expected], rtol=1e-9)


def test_petermann_unity_for_normal_liouvillian():
    # Pure dephasing has a normal Liouvillian -> every Petermann factor is 1.
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_Z], [0.5])
    _, K = petermann_factors(L)
    np.testing.assert_allclose(K, np.ones_like(K), atol=1e-9)


def test_petermann_keeps_conjugate_pair_count():
    # Anchor I: complex-conjugate eigenmodes are kept (none silently dropped).
    L = build_liouvillian(0.5 * _X, [_Z], [0.3])
    eigvals, K = petermann_factors(L)
    assert K.size == eigvals.size
    assert np.any(np.abs(np.imag(eigvals)) > 1e-6)


# =============================================================================
# D10 -- Kreiss constant.
# Oracle: continuous-time K = sup_{Re z > 0} Re(z) ||(zI - A)^{-1}||.
# Independent value from a multistart 2D resolvent-norm maximisation.
# Reference facts: K = 1 (normal, abscissa 0), K >= 1 (always),
# closed form sqrt(1 + b^2) for [[0, b], [0, -1]] (verified by the oracle).
# =============================================================================


def _kreiss_oracle_standard(A: np.ndarray, *, xmax: float = 30.0, ymax: float = 40.0) -> float:
    """Independent Kreiss value: 2D max of Re(z)*||(zI-A)^-1|| over Re(z)>0."""
    n = A.shape[0]
    eye = np.eye(n, dtype=complex)

    def neg(p: np.ndarray) -> float:
        x, y = p
        if x <= 1e-9:
            return 0.0
        z = x + 1j * y
        return -float(x * sla.svdvals(np.linalg.inv(z * eye - A))[0])

    best, start = 0.0, (1.0, 0.0)
    for x0 in np.geomspace(1e-2, xmax, 60):
        for y0 in np.linspace(0.0, ymax, 60):
            v = -neg(np.array([x0, y0]))
            if v > best:
                best, start = v, (x0, y0)
    res = minimize(
        neg, np.array(start), method="Nelder-Mead",
        options={"xatol": 1e-7, "fatol": 1e-9},
    )
    return max(best, -float(res.fun))


def test_kreiss_unity_for_normal_matrix():
    # Normal Hurwitz matrix (spectral abscissa 0) -> K = 1 exactly.
    A = np.diag([0.0, -1.0, -2.0]).astype(complex)
    assert kreiss_constant(A, n_sigma=80, n_omega=81) == pytest.approx(1.0, abs=2e-3)


def test_kreiss_unity_for_normal_liouvillian():
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [_Z], [0.5])
    assert kreiss_constant(L, n_sigma=60, n_omega=61) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("b,exact", [(1.0, np.sqrt(2.0)), (2.0, np.sqrt(5.0)), (4.0, np.sqrt(17.0))])
def test_kreiss_matches_closed_form_and_oracle(b, exact):
    A = np.array([[0.0, b], [0.0, -1.0]], dtype=complex)
    k_code = kreiss_constant(A, n_sigma=160, n_omega=161)
    k_oracle = _kreiss_oracle_standard(A)
    # The independent 2D-optimised oracle must reproduce the sqrt(1+b^2) value.
    assert k_oracle == pytest.approx(exact, rel=2e-3)
    # The grid implementation must match the oracle (grid is a lower bound that
    # converges from below; tolerance is tight enough to fail a wrong formula).
    assert k_code == pytest.approx(k_oracle, rel=5e-3)
    assert k_code >= 1.0  # Kreiss resolvent condition.


def test_kreiss_negative_control_grid_cannot_exceed_oracle():
    # Non-vacuous guard: the grid value must not spuriously exceed the true sup
    # by more than grid error (would signal an over-counting bug).
    A = np.array([[0.0, 4.0], [0.0, -1.0]], dtype=complex)
    k_code = kreiss_constant(A, n_sigma=160, n_omega=161)
    k_oracle = _kreiss_oracle_standard(A)
    assert k_code <= k_oracle * (1.0 + 1e-2)


# =============================================================================
# D11 -- Bohr arithmetic progression of imaginary parts (Basso 2025).
# Oracle: hand-constructed spectra whose longest AP of |Im(lambda)| is known.
# The implementation operates on strictly-positive imaginary parts (oscillating
# modes); the oracles below therefore use nonzero-imag spectra, the documented
# and unambiguous regime.
# =============================================================================


@pytest.mark.parametrize(
    "imags,expected",
    [
        ([2.0, 4.0, 6.0, 8.0], 4),          # exact AP, common diff 2
        ([3.0, 6.0, 9.0], 3),                # exact AP, common diff 3
        ([2.0, 4.0, 5.0, 6.0, 8.0, 13.0], 4),  # AP {2,4,6,8} with distractors
        ([1.0, 10.0, 100.0], 2),             # no AP beyond a pair
        ([5.0], 1),                          # single mode
    ],
)
def test_bohr_ap_longest_progression(imags, expected):
    eigs = -1.0 + 1j * np.array(imags)
    length, _ = bohr_arithmetic_progression(eigs, d=4)
    assert length == expected


def test_bohr_ap_pauli_bound_is_log2_d():
    eigs = -1.0 + 1j * np.array([2.0, 4.0])
    for d, expected in [(2, 1.0), (4, 2.0), (8, 3.0), (16, 4.0)]:
        _, bound = bohr_arithmetic_progression(eigs, d=d)
        assert bound == pytest.approx(expected, abs=1e-12)


def test_bohr_ap_negative_control_not_overcounting():
    # A spectrum with NO three-term AP must report length 2, not more.
    eigs = -1.0 + 1j * np.array([1.0, 4.0, 100.0])
    length, _ = bohr_arithmetic_progression(eigs, d=4)
    assert length == 2
