"""Issue #152: the Liouvillian builders must not return a non-finite generator.

Every input gate of both builders looks at the INPUT: ``H``, the jump
operators and the rates are each finite, and ``H`` is Hermitian on its
gauge-fixed scale. Nothing looked at the OUTPUT. A finite input can still
assemble into a non-finite generator, because the assembly forms quantities the
gates never see:

* the coherent part ``-i (I (x) H - H.T (x) I)`` contains the diagonal
  differences ``H_jj - H_kk``, which overflow for ``diag(1e308, -1e308)``;
* a dissipator forms ``L_k^dag L_k`` and ``L_k.conj() (x) L_k``, i.e. products
  of entries, which overflow for an entry of ``1e200``;
* a finite rate multiplies an already finite dissipator and can overflow it.

Each rejection test is paired with a control of the same shape that must stay
accepted, so a guard that simply refuses large inputs cannot pass: the controls
sit one factor of a few below the overflow boundary.

Expected exceptions are caught broadly (``except Exception``) and the class is
checked by ``isinstance``: without the fix the build does not raise
``ValueError`` -- it either returns silently or, under this suite's
``filterwarnings = error``, raises a NumPy ``RuntimeWarning`` -- and the test
must fail at an ASSERTION in both cases, not at an unexpected exception and not
at pytest's ``DID NOT RAISE`` (which is not an assertion).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from liouscope.core.lindblad import build_liouvillian
from liouscope.sparse.build import build_sparse_liouvillian

_MSG = "non-finite"


def _dense(H, jumps, rates=None):
    return build_liouvillian(H, jumps, rates)


def _sparse(H, jumps, rates=None):
    return build_sparse_liouvillian(H, jumps, rates)


BUILDERS = [
    pytest.param(_dense, id="dense"),
    pytest.param(_sparse, id="sparse"),
]


def _values(L):
    return L.data if sp.issparse(L) else np.asarray(L)


def _raised(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # broad on purpose: the class is asserted below
        return exc
    return None


def _assert_refused(build, H, jumps, rates=None):
    # Warnings are escalated explicitly, independent of the pytest config: a
    # refused build must surface as the builder's ValueError, not as a NumPy
    # overflow RuntimeWarning that escaped before the guard could speak.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        err = _raised(build, H, jumps, rates)
    assert err is not None, "builder accepted the input and returned a generator"
    assert isinstance(err, ValueError), (
        f"expected ValueError, got {type(err).__name__}: {err}"
    )
    assert _MSG in str(err), str(err)
    return err


# --- before-tests: finite input, non-finite generator ------------------------


def test_dense_rejects_overflowing_diagonal_difference():
    H = np.diag([1e308, -1e308]).astype(complex)
    _assert_refused(_dense, H, [])


def test_sparse_rejects_overflowing_diagonal_difference():
    H = np.diag([1e308, -1e308]).astype(complex)
    _assert_refused(_sparse, H, [])


def test_dense_rejects_overflowing_dissipator_product():
    H = np.zeros((2, 2), dtype=complex)
    J = np.array([[0.0, 1e200], [0.0, 0.0]], dtype=complex)
    _assert_refused(_dense, H, [J])


def test_sparse_rejects_overflowing_dissipator_product():
    H = np.zeros((2, 2), dtype=complex)
    J = np.array([[0.0, 1e200], [0.0, 0.0]], dtype=complex)
    _assert_refused(_sparse, H, [J])


@pytest.mark.parametrize("build", BUILDERS)
def test_rejects_finite_rate_overflowing_finite_dissipator(build):
    # The jump operator alone assembles to entries of 1e20; the finite rate
    # 1e300 is what pushes the product past float64.
    H = np.zeros((2, 2), dtype=complex)
    J = np.array([[0.0, 1e10], [0.0, 0.0]], dtype=complex)
    _assert_refused(build, H, [J], [1e300])


@pytest.mark.parametrize("build", BUILDERS)
def test_refusal_message_names_the_cause(build):
    H = np.diag([1e308, -1e308]).astype(complex)
    err = _assert_refused(build, H, [])
    assert "overflow" in str(err), str(err)


# --- controls: must stay accepted and finite ---------------------------------


@pytest.mark.parametrize("build", BUILDERS)
def test_control_unit_diagonal_is_accepted(build):
    H = np.diag([1.0, -1.0]).astype(complex)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L = build(H, [])
    assert np.all(np.isfinite(_values(L)))
    dense = L.toarray() if sp.issparse(L) else L
    # Closed form: [H, |j><k|] = (H_jj - H_kk) |j><k| for diagonal H, and
    # |j><k| sits at column-stacked index j + d*k. For diag(1, -1) the order
    # (jk) = (00, 10, 01, 11) gives -i * (0, -2, +2, 0) = (0, +2i, -2i, 0).
    np.testing.assert_array_equal(np.diag(dense), [0.0, 2.0j, -2.0j, 0.0])
    assert np.count_nonzero(dense - np.diag(np.diag(dense))) == 0


@pytest.mark.parametrize("build", BUILDERS)
def test_control_large_but_representable_diagonal_is_accepted(build):
    # H_jj - H_kk = 8e307 < float64 max (~1.797e308): must NOT be refused.
    H = np.diag([4e307, -4e307]).astype(complex)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L = build(H, [])
    vals = _values(L)
    assert np.all(np.isfinite(vals))
    assert np.max(np.abs(vals)) == pytest.approx(8e307)


@pytest.mark.parametrize("build", BUILDERS)
def test_control_large_but_representable_dissipator_is_accepted(build):
    # Products reach 1e300, still representable.
    H = np.zeros((2, 2), dtype=complex)
    J = np.array([[0.0, 1e150], [0.0, 0.0]], dtype=complex)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        L = build(H, [J])
    vals = _values(L)
    assert np.all(np.isfinite(vals))
    assert np.max(np.abs(vals)) == pytest.approx(1e300)


def test_controls_agree_between_dense_and_sparse():
    H = np.diag([4e307, -4e307]).astype(complex)
    J = np.array([[0.0, 1e150], [0.0, 0.0]], dtype=complex)
    for jumps, H_in in (([], H), ([J], np.zeros((2, 2), dtype=complex))):
        np.testing.assert_array_equal(
            _sparse(H_in, jumps).toarray(), _dense(H_in, jumps)
        )


# --- the shared output guard itself ------------------------------------------


def test_guard_rejects_non_canonical_sparse_duplicates_that_overflow():
    # CSR with two finite duplicates at the same position: every stored value
    # is finite, but the matrix they represent holds 1e308 + 1e308 = inf.
    # Checking ``.data`` of a non-canonical matrix would therefore miss it.
    from liouscope.numerics.generator_guard import require_finite_generator

    A = sp.csr_matrix(
        (np.array([1e308, 1e308]), np.array([0, 0]), np.array([0, 2, 2])),
        shape=(2, 2),
    )
    assert not A.has_canonical_format
    assert np.all(np.isfinite(A.data))
    err = _raised(require_finite_generator, A, builder="test")
    assert isinstance(err, ValueError), repr(err)
    # The caller's matrix is inspected, not mutated.
    assert A.nnz == 2
    assert not A.has_canonical_format


def test_guard_rejects_coo_duplicates_that_overflow():
    from liouscope.numerics.generator_guard import require_finite_generator

    A = sp.coo_matrix(
        (np.array([1e308, 1e308]), (np.array([1, 1]), np.array([0, 0]))),
        shape=(2, 2),
    )
    err = _raised(require_finite_generator, A, builder="test")
    assert isinstance(err, ValueError), repr(err)


def test_guard_accepts_finite_and_counts_non_finite_dense():
    from liouscope.numerics.generator_guard import require_finite_generator

    require_finite_generator(np.eye(4, dtype=complex), builder="test")
    require_finite_generator(sp.identity(4, format="csr"), builder="test")
    bad = np.eye(4, dtype=complex)
    bad[0, 1] = np.nan
    bad[2, 3] = np.inf
    err = _raised(require_finite_generator, bad, builder="test")
    assert isinstance(err, ValueError), repr(err)
    assert "2 of 16" in str(err), str(err)


def test_sparse_builder_guard_does_not_densify(monkeypatch):
    # The output check is O(nnz): it must never materialise d^2 x d^2.
    def boom(*_a, **_k):
        raise AssertionError("sparse generator was densified")

    for cls in (sp.csr_matrix, sp.csr_array, sp.coo_matrix, sp.coo_array):
        monkeypatch.setattr(cls, "toarray", boom)
        monkeypatch.setattr(cls, "todense", boom)
    d = 16
    rng = np.random.default_rng(152)
    X = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = sp.csr_matrix(0.5 * (X + X.conj().T))
    J = sp.csr_matrix(np.diag(np.arange(d, dtype=complex)))
    L = build_sparse_liouvillian(H, [J], [0.3])
    assert L.shape == (d * d, d * d)
    assert np.all(np.isfinite(L.data))
