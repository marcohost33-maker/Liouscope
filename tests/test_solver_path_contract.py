"""Tests for the top-level diagnose() solver-path contract."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose


def _dephased_qubit_liouvillian() -> np.ndarray:
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return build_liouvillian(0.5 * sx, [sz], [0.3])


def test_diagnose_rejects_unknown_solver_path():
    L = _dephased_qubit_liouvillian()

    with pytest.raises(ValueError, match="solver_path"):
        diagnose(L, solver_path="gpu_magic")


def test_diagnose_sparse_arpack_is_reserved_not_silent_dense():
    L = _dephased_qubit_liouvillian()

    with pytest.raises(NotImplementedError, match="sparse_arpack"):
        diagnose(L, solver_path="sparse_arpack")
