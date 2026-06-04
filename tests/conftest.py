"""Shared pytest fixtures and markers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

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
