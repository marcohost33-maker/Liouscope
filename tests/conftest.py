"""Shared pytest fixtures and markers."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

QUTIP_AVAILABLE = importlib.util.find_spec("qutip") is not None
qutip_required = pytest.mark.skipif(not QUTIP_AVAILABLE, reason="QuTiP not installed")


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
