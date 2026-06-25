"""Metamorphic-relation oracles for the spectral pipeline (LIOU-F-020).

Metamorphic testing checks relations between inputs and outputs without knowing
the absolute ground truth:

* **Scaling (B008):** ``gap(c*L) = c*gap(L)`` for ``c > 0``. Catches linearity /
  unit errors in the gap computation.
* **Unitary similarity (B009):** ``spec(U L U^dag) = spec(L)`` for any unitary
  ``U``. Catches basis-dependence bugs in the eigenvalue routine.

Both are trivially true for Liouvillian spectra, so a violation is a pure
implementation bug. They are deterministic and need no reference spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.diagnostics.spectral import liouvillian_gap
from liouscope.numerics.linalg import eig_nonhermitian

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_LOWER = (0.5 * (_X - 1j * _Y)).conj().T


def _systems() -> list[np.ndarray]:
    return [
        build_liouvillian(np.zeros((2, 2), dtype=complex), [_LOWER], [0.4]),  # amp damp
        build_liouvillian(0.5 * _X, [_Z], [0.3]),  # driven dephasing (complex pair)
        build_liouvillian(0.65 * _Z, [_LOWER, _LOWER.conj().T], [0.9, 0.2]),  # thermal
    ]


def _gap(L: np.ndarray) -> float:
    return liouvillian_gap(eig_nonhermitian(L).eigenvalues)


def _spectrum_sorted(L: np.ndarray) -> np.ndarray:
    """Eigenvalues sorted by a rounded (real, imag) key.

    A plain ``sort_complex`` is unstable for near-degenerate / conjugate-pair
    eigenvalues, so two numerically-equal spectra can come out in different
    order. Rounding the sort key makes the multiset comparison order-stable.
    """
    z = eig_nonhermitian(L).eigenvalues
    zr = np.round(z, 7)
    return z[np.lexsort((zr.imag, zr.real))]


@pytest.mark.parametrize("c", [0.5, 2.0, 7.3])
def test_gap_scales_linearly_with_generator(c):
    for L in _systems():
        base = _gap(L)
        scaled = _gap(c * L)
        assert scaled == pytest.approx(c * base, rel=1e-9, abs=1e-12)


def test_scaling_is_non_vacuous():
    # gap(2L) must actually differ from gap(L) (guards a trivially-passing test).
    L = _systems()[0]
    assert _gap(2.0 * L) == pytest.approx(2.0 * _gap(L), rel=1e-9)
    assert abs(_gap(2.0 * L) - _gap(L)) > 1e-3


def test_spectrum_invariant_under_unitary_similarity():
    rng = np.random.default_rng(2024)
    for L in _systems():
        n = L.shape[0]
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        U, _ = np.linalg.qr(a)  # Haar-ish unitary
        assert np.allclose(U @ U.conj().T, np.eye(n), atol=1e-10)
        L_transformed = U @ L @ U.conj().T
        spec_orig = _spectrum_sorted(L)
        spec_trans = _spectrum_sorted(L_transformed)
        np.testing.assert_allclose(spec_trans, spec_orig, atol=1e-7)
        # The gap, being a spectral functional, is likewise invariant.
        assert _gap(L_transformed) == pytest.approx(_gap(L), abs=1e-9)
