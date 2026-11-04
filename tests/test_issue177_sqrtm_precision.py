"""Issue #177: ``sqrtm``'s extended-precision result must not reach NumPy linalg.

SciPy <= 1.14 returns ``complex256`` from ``scipy.linalg.sqrtm`` for complex
input (scipy/scipy#18250). NumPy's ``linalg`` refuses that dtype, so the GNS
and KMS gaps failed on the then-declared minimum ``scipy>=1.10``; the current
floor ``scipy>=1.13`` still upcasts. The tests below
emulate the old SciPy on any installed version by upcasting ``sqrtm``'s result,
so the regression stays covered even where the real old SciPy is not installed
(the ``min-deps`` CI job additionally runs the whole suite on the declared
minimum, SciPy 1.13 with NumPy 2.0, which still upcasts).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

from liouscope.core.lindblad import build_liouvillian, steady_state
from liouscope.diagnostics import spectral

_HAS_EXTENDED = np.dtype(np.clongdouble).itemsize > np.dtype(np.complex128).itemsize


def _amplitude_damped_qubit() -> tuple[np.ndarray, np.ndarray]:
    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    H = 0.4 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    L = build_liouvillian(H, [np.sqrt(0.7) * sm, np.sqrt(0.2) * sm.conj().T])
    return L, steady_state(L)


def _old_scipy_sqrtm(monkeypatch: pytest.MonkeyPatch) -> None:
    real = sla.sqrtm

    def _upcasting(A: np.ndarray, *args: object, **kwargs: object) -> np.ndarray:
        root = np.asarray(real(A, *args, **kwargs))
        return root.astype(np.clongdouble) if np.iscomplexobj(root) else root

    monkeypatch.setattr(spectral.sla, "sqrtm", _upcasting)


def test_sqrtm_double_returns_double_precision() -> None:
    A = np.diag([0.5, 0.25, 0.25]).astype(complex)
    root = spectral._sqrtm_double(A)
    assert root.dtype == np.complex128
    np.testing.assert_allclose(root @ root, A, atol=1.0e-15)
    assert spectral._sqrtm_double(np.eye(3)).dtype in (np.float64, np.complex128)


@pytest.mark.skipif(not _HAS_EXTENDED, reason="clongdouble == complex128 on this platform")
def test_the_emulated_old_scipy_really_breaks_numpy_linalg(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive control: the emulation reproduces the #177 failure mode."""
    _old_scipy_sqrtm(monkeypatch)
    root = spectral.sla.sqrtm(np.eye(2, dtype=complex))
    assert root.dtype == np.clongdouble
    with pytest.raises(TypeError, match="unsupported in linalg"):
        np.linalg.eigvalsh(root)


@pytest.mark.skipif(not _HAS_EXTENDED, reason="clongdouble == complex128 on this platform")
def test_gns_and_kms_gaps_survive_an_upcasting_sqrtm(monkeypatch: pytest.MonkeyPatch) -> None:
    L, rho_ss = _amplitude_damped_qubit()
    reference = (spectral.gns_gap(L, rho_ss), spectral.kms_gap(L, rho_ss))
    _old_scipy_sqrtm(monkeypatch)
    emulated = (spectral.gns_gap(L, rho_ss), spectral.kms_gap(L, rho_ss))
    assert all(np.isfinite(v) for v in emulated)
    np.testing.assert_allclose(emulated, reference, rtol=1.0e-12)
