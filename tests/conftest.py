"""Shared pytest fixtures and markers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

QUTIP_AVAILABLE = importlib.util.find_spec("qutip") is not None


def qutip_required(func):
    """Mark a test as a QuTiP cross-check.

    Combines two things so a single decorator serves local dev *and* the
    dedicated CI job:

    * ``pytest.mark.qutip`` -- a selectable marker. The
      ``ci-qutip.yml`` job installs QuTiP and runs ``pytest -m qutip`` so the
      cross-checks execute for real (no skip) on every push.
    * ``pytest.mark.skipif(not QUTIP_AVAILABLE)`` -- keeps the default suite
      green on machines without the optional QuTiP extra installed.
    """
    func = pytest.mark.qutip(func)
    func = pytest.mark.skipif(not QUTIP_AVAILABLE, reason="QuTiP not installed")(func)
    return func


def assert_spectra_match(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    rtol: float = 1.0e-7,
    atol: float = 0.0,
    err_msg: str = "",
) -> None:
    """Assert two eigenvalue MULTISETS agree, independent of their order.

    Issue #180 (macOS probe): ``np.sort_complex`` orders by real part first,
    so a conjugate pair whose exact real parts are equal is ordered by its
    rounding. On Apple Accelerate ``-0.3000000000000003 +- 0.954i`` came back
    against ``-0.2999999999999998 -+ 0.954i`` from QuTiP, the sorted arrays
    paired ``+i`` with ``-i`` and an exact match failed by ``1.9``. Rounding the
    sort key only moves that tie to the rounding boundary.

    This helper asks the question the comparison means: is there a pairing in
    which every pair satisfies ``|a - d| <= atol + rtol * |d|``, the criterion
    of :func:`numpy.testing.assert_allclose`? That is a perfect matching in the
    bipartite feasibility graph, decided exactly by
    :func:`scipy.optimize.linear_sum_assignment` on the infeasibility
    indicator, the same primitive ``liouscope.numerics.conditioning`` uses to
    pair spectra. A minimum-total-distance pairing is NOT used for the verdict:
    it need not minimise the largest deviation. It is used only to print a
    readable diagnostic when no feasible pairing exists.
    """
    a = np.asarray(actual, dtype=complex).ravel()
    d = np.asarray(desired, dtype=complex).ravel()
    if a.shape != d.shape:
        raise AssertionError(
            f"spectra differ in size: {a.size} vs {d.size}. {err_msg}".rstrip()
        )
    if a.size == 0:
        return
    distance = np.abs(a[:, None] - d[None, :])
    feasible = distance <= atol + rtol * np.abs(d)[None, :]
    rows, cols = linear_sum_assignment((~feasible).astype(float))
    if bool(np.all(feasible[rows, cols])):
        return
    rows, cols = linear_sum_assignment(distance)
    np.testing.assert_allclose(a[rows], d[cols], rtol=rtol, atol=atol, err_msg=err_msg)
    raise AssertionError(  # pragma: no cover - the closest pairing failed above
        f"no pairing satisfies rtol={rtol}, atol={atol}. {err_msg}".rstrip()
    )


@pytest.fixture(scope="session")
def pauli() -> dict[str, np.ndarray]:
    return {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
