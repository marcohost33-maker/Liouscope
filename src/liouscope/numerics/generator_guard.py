"""Output guard for assembled GKSL generators (issue #152).

The builders gate their INPUT: ``H``, the jump operators and the rates must be
finite, and ``H`` Hermitian on its gauge-fixed scale. A finite input can still
assemble into a NON-finite generator, because the assembly forms quantities no
input gate sees -- the diagonal differences ``H_jj - H_kk`` of the coherent
part ``-i[H, .]``, the products ``L_k^dag L_k`` and ``L_k.conj() (x) L_k`` of a
dissipator, and ``rate * dissipator``. ``diag(1e308, -1e308)`` is enough.

So the builders additionally check what they return. The check is exact, not
a heuristic bound on the input: it refuses precisely the generators that
contain a non-finite entry, and nothing else.

Why ``.data`` suffices for a sparse generator
---------------------------------------------
In a sparse matrix in CANONICAL form (sorted indices, no duplicates) every
position outside the stored index structure has the value exactly ``0.0``,
which is finite. Hence ``all(isfinite(A.data))`` is equivalent to
``all(isfinite(A.toarray()))`` -- without materialising ``d^2 x d^2``.

Two ways that equivalence could fail, and why they do not:

* **An overflow result dropped as an implicit zero.** SciPy's sparse binary
  operations prune only results that compare equal to zero. ``inf`` and
  ``NaN`` never compare equal to zero, so an overflow (and ``inf - inf =
  NaN``) is always stored explicitly. Scalar multiplication acts on ``.data``
  and prunes nothing.
* **Duplicate entries.** A NON-canonical matrix may store several finite
  values at one position whose SUM overflows, e.g. ``1e308 + 1e308``;
  ``.data`` is then all finite while the represented matrix is not. The guard
  therefore canonicalises first (on a copy, so the caller's object is not
  mutated). For an already canonical matrix this costs nothing beyond the
  O(nnz) check.

The error is a ``ValueError``, the family every other non-representable-input
gate of the builders raises (``"H contains non-finite entries"``,
``"jump_op contains non-finite entries"``).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def require_finite_generator(
    L_super: np.ndarray | sp.spmatrix | sp.sparray, *, builder: str
) -> None:
    """Refuse an assembled generator that contains a non-finite entry.

    Parameters
    ----------
    L_super
        Dense array or ``scipy.sparse`` matrix/array. Sparse input is checked
        in O(nnz) on its canonical form and is never densified.
    builder
        Name of the calling builder, used in the error message.

    Raises
    ------
    ValueError
        If any entry of the represented matrix is ``NaN`` or ``+/-inf``.
    """
    if isinstance(L_super, (sp.spmatrix, sp.sparray)):
        A = L_super if L_super.format in ("csr", "csc") else L_super.tocsr()
        if not A.has_canonical_format:
            if A is L_super:
                A = A.copy()
            A.sum_duplicates()
        values = np.asarray(A.data)
        total = int(A.shape[0]) * int(A.shape[1])
    else:
        values = np.asarray(L_super)
        total = int(values.size)
    finite = np.isfinite(values)
    if np.all(finite):
        return
    n_bad = int(values.size - np.count_nonzero(finite))
    raise ValueError(
        f"{builder}: the assembled Liouvillian has {n_bad} of {total} entries "
        "non-finite although H, jump_ops and rates passed their finiteness "
        "gates. An intermediate quantity overflowed float64 -- a diagonal "
        "difference H_jj - H_kk of the coherent part -i[H, .], a dissipator "
        "product L_k^dag L_k or L_k.conj() (x) L_k, or rate * dissipator -- so "
        "the generator is not representable and the input is refused "
        "(issue #152)."
    )


__all__ = ["require_finite_generator"]
