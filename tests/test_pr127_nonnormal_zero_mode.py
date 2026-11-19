"""PR #127 review thread (linalg.py:1054): nonnormal zero-mode displacement.

``_zero_mode_applicable`` admits a trace defect up to ``sqrt(d) * bound``
because the nearest exactly trace-preserving operator ``L'`` is that close to
``L``. For a NONNORMAL zero mode the eigenvalue moves by much more than the
operator: the reviewer's 4x4 operator (leading block ``[[0, 1e-14], [1, 0]]``
in a basis starting with ``vec(I)/sqrt(2)``) has its exact eigenvalues nearest
zero at ``+-1e-7`` against a band of ``4.4e-13``. Every route then "failed", and
the CORRECT spectrum was reported as a failed eigensolve (``applicable=True,
certified=False`` -> D1 withheld, classification UNDEFINED).

Fix: in the no-route-certified path only, the certificate reports
``applicable=False`` when the first-order displacement of the zero mode of
``L'`` -- ``||r|| / |u^H x|`` with ``x`` from an SVD of ``L'`` -- exceeds the
band. Controls:

* a healthy generator stays certified (the fix does not touch that path);
* a healthy generator whose eigensolver is made to FAIL keeps
  ``applicable=True, certified=False`` even with its trace defect pushed to
  half the admissible maximum -- the strongest forger available to this check
  (a genuine failure must not be relabelled into "not applicable").
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import liouscope.numerics.linalg as la
from liouscope.core.lindblad import build_liouvillian

_U = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)


def _basis(seed: int) -> np.ndarray:
    """Unitary whose first column is ``vec(I)/sqrt(2)``, otherwise random."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    z[:, 0] = _U
    q, _ = np.linalg.qr(z)
    q[:, 0] *= np.vdot(q[:, 0], _U) / abs(np.vdot(q[:, 0], _U))
    return q


def _reviewer_operator(seed: int) -> np.ndarray:
    b = np.zeros((4, 4), dtype=complex)
    b[0, 1] = 1.0e-14
    b[1, 0] = 1.0
    b[2, 2] = -1.0
    b[3, 3] = -2.0
    q = _basis(seed)
    return q @ b @ q.conj().T


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_premise_is_inside_the_admitted_band(seed: int) -> None:
    """Fixture guard: the operator must sit where the bug lived -- admitted by
    ``_zero_mode_applicable`` (both gates), with no eigenvalue in the band."""
    L = _reviewer_operator(seed)
    tp_defect, fro = la.trace_preservation_defect(L)
    bound = la.ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * float(np.linalg.norm(L, 2))
    assert la._zero_mode_applicable(tp_defect, fro, bound, 4, 1.0e-10)
    assert la.trace_preservation_componentwise_error(L) <= 1.0e-10
    nearest = float(np.min(np.abs(np.linalg.eigvals(L))))
    assert nearest > 1.0e3 * bound  # ~1e-7 against ~4e-13


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_nonnormal_displacement_is_not_a_failed_eigensolve(seed: int) -> None:
    L = _reviewer_operator(seed)
    for api in (la.certified_eigvals, la.certified_eig):
        spectrum, cert = api(L)
        assert cert.applicable is False, (api.__name__, cert)
        assert cert.certified is False
        assert cert.zero_mode_count == 0
        values = spectrum if isinstance(spectrum, np.ndarray) else spectrum.eigenvalues
        # The returned spectrum is the CORRECT one, +-1e-7 near zero. Error
        # model: lambda = +-sqrt(delta + e) with delta = 1e-14 and a backward
        # error e ~ eps * ||L||_2, so the relative error is ~ e / (2 delta)
        # (~2.7e-2 here); allowed: twice that.
        rel = 2.0 * float(np.finfo(float).eps) * float(np.linalg.norm(L, 2)) / (2.0e-14)
        assert float(np.min(np.abs(values))) == pytest.approx(1.0e-7, rel=rel)


def _healthy() -> np.ndarray:
    h = np.array([[0.5, 0.2], [0.2, -0.5]], dtype=complex)
    return build_liouvillian(h, [_SIGMA_MINUS], [0.7])


def test_healthy_generator_is_still_certified() -> None:
    L = _healthy()
    for api in (la.certified_eigvals, la.certified_eig):
        _, cert = api(L)
        assert cert.applicable and cert.certified and cert.zero_mode_count == 1


def _perturb_trace(L: np.ndarray, fraction: float) -> np.ndarray:
    """Add ``u e^H`` so that ``||u^H L||`` grows by ``fraction * bound``."""
    bound = la.ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * float(np.linalg.norm(L, 2))
    e = np.zeros(L.shape[0], dtype=complex)
    e[1] = fraction * bound
    return L + np.outer(_U, e)


def _break_every_route(monkeypatch, L: np.ndarray, shift: float) -> None:
    """Every solver route returns the TRUE spectrum shifted by ``-shift``: no
    zero mode, i.e. a genuine eigensolver failure on a healthy generator.
    The decomposition is taken BEFORE patching: ``eig_nonhermitian`` itself
    calls ``scipy.linalg.eig``, which is disabled below."""
    d = la.eig_nonhermitian(L, compute_left=True)

    def _primary(A: np.ndarray, *, compute_left: bool = False) -> la.EigenDecomposition:
        return la.EigenDecomposition(
            eigenvalues=d.eigenvalues - shift,
            right_vectors=d.right_vectors,
            left_vectors=d.left_vectors if compute_left else None,
        )

    def _raise(*args: object, **kwargs: object) -> object:
        raise sla.LinAlgError("route disabled for this test")

    monkeypatch.setattr(la, "eig_nonhermitian", _primary)
    monkeypatch.setattr(la.np.linalg, "eigvals", _raise)
    monkeypatch.setattr(la.sla, "schur", _raise)
    monkeypatch.setattr(la.sla, "eig", _raise)


@pytest.mark.parametrize("fraction", [0.0, 0.5])
def test_a_genuine_eigensolver_failure_is_not_relabelled(monkeypatch, fraction: float) -> None:
    """NEGATIVE CONTROL / strongest forger: healthy generator, trace defect at
    up to half the admissible maximum, every route broken. The certificate
    must keep saying "the eigensolve failed"."""
    L = _perturb_trace(_healthy(), fraction)
    tp_defect, fro = la.trace_preservation_defect(L)
    bound = la.ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * float(np.linalg.norm(L, 2))
    assert la._zero_mode_applicable(tp_defect, fro, bound, 4, 1.0e-10)
    if fraction > 0.0:
        assert tp_defect > 0.25 * bound  # the defect really is near the edge
    _break_every_route(monkeypatch, L, shift=0.3)
    for api in (la.certified_eigvals, la.certified_eig):
        _, cert = api(L)
        assert cert.applicable is True, (api.__name__, fraction, cert)
        assert cert.certified is False


def test_helper_fails_closed_on_unusable_input() -> None:
    assert la._zero_mode_displaced_by_premise(np.zeros((3, 3), dtype=complex), 1.0) is False
    assert la._zero_mode_displaced_by_premise(_healthy(), float("nan")) is False
    assert la._zero_mode_displaced_by_premise(_healthy(), 0.0) is False
    # Exactly trace preserving: r = 0, nothing can be displaced.
    exact = np.zeros((4, 4), dtype=complex)
    exact[1, 2] = 1.0  # u^H exact = 0 because rows 0 and 3 are zero
    assert la._zero_mode_displaced_by_premise(exact, 1.0e-12) is False


def test_a_degenerate_zero_mode_is_not_explained_by_the_premise() -> None:
    """Two zero singular values: a rank-one defect leaves one zero in place to
    first order, so a spectrum without any zero is a failure, not displacement."""
    L = np.zeros((4, 4), dtype=complex)
    L[2, 2] = -1.0
    L = L + np.outer(_U, np.array([0.0, 1.0e-15, 0.0, 0.0]))
    assert la._zero_mode_displaced_by_premise(L, 1.0e-13) is False
