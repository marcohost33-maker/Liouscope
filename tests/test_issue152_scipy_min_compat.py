"""Regression for the SciPy 1.10 compatibility boundary of issue #152.

Liouscope declares ``scipy>=1.10``.  The public ``scipy.sparse.sparray`` base
class was introduced in SciPy 1.11, so the output guard must not require that
symbol at runtime.  These tests delete the symbol from a modern SciPy module to
model the declared minimum-version surface and prove both dense and sparse
paths continue to use the compatibility API ``scipy.sparse.issparse``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from liouscope.numerics.generator_guard import require_finite_generator


def test_generator_guard_does_not_require_sparray_symbol(monkeypatch):
    monkeypatch.delattr(sp, "sparray", raising=False)

    dense = np.eye(2, dtype=complex)
    sparse = sp.csr_matrix(dense)

    require_finite_generator(dense, builder="test")
    require_finite_generator(sparse, builder="test")


def test_generator_guard_still_refuses_nonfinite_without_sparray_symbol(monkeypatch):
    monkeypatch.delattr(sp, "sparray", raising=False)

    dense = np.eye(2, dtype=complex)
    dense[0, 1] = np.inf
    sparse = sp.csr_matrix(dense)

    with pytest.raises(ValueError, match="non-finite"):
        require_finite_generator(dense, builder="test")
    with pytest.raises(ValueError, match="non-finite"):
        require_finite_generator(sparse, builder="test")
