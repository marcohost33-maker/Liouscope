"""Documentation + sanity contract for the Petermann caveat (LIOU-NG-003).

NG-003 is a SAFE caveat (no formula change): a large Petermann factor (D9) is
NOT a direct propagator amplification ``sup_t ||e^{tL}||`` -- that growth is
bounded by the Kreiss constant (D10) and the numerical abscissa (D15). These
tests pin (a) that the caveat stays documented (so it cannot silently regress)
and (b) the conditioning-measure sanity that motivates it.
"""

from __future__ import annotations

import numpy as np

from liouscope.diagnostics import nonnormality
from liouscope.diagnostics.nonnormality import petermann_factors

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_LOWER = (0.5 * (_X - 1j * _Y)).conj().T


def test_caveat_is_documented():
    # The NG-003 caveat must be present in both the module and function docs.
    assert "LIOU-NG-003" in (nonnormality.__doc__ or "")
    assert "not sufficient" in (nonnormality.__doc__ or "")
    assert "LIOU-NG-003" in (petermann_factors.__doc__ or "")


def test_petermann_factors_are_condition_numbers_geq_one():
    # Condition numbers K_j = ||r||^2 ||l||^2 / |<l,r>|^2 >= 1 always
    # (Cauchy-Schwarz); the caveat is exactly that K_j measures conditioning,
    # not realised amplification.
    from liouscope import build_liouvillian

    L = build_liouvillian(0.65 * _Z, [_LOWER, _LOWER.conj().T], [0.9, 0.2])
    _, K = petermann_factors(L)
    assert K.size > 0
    assert np.all(K >= 1.0 - 1e-9)


def test_normal_generator_has_petermann_near_one():
    # A normal (Hermitian) superoperator has orthogonal eigenvectors =>
    # Petermann factors ~ 1 (no conditioning blow-up), independent of any gap.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 4))
    herm = a + a.T  # real symmetric => normal
    _, K = petermann_factors(herm.astype(complex), atol=0.0)
    assert np.max(K) < 1.0 + 1e-6
