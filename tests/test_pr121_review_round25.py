"""PR #121 round-25 review finding: the eigenVECTOR gate's own norms.

One finding, and it is the reason this file exists at all: round 24 made the
eigenpair residuals inside ``certify_nonstationary`` underflow-safe
(``linalg.py`` B4) and left the SEPARATE ``certified_eig`` eigenvector gate on
the raw ``np.linalg.norm(..., axis=0)``. The same 1e-162 cliff therefore
survived one full review round at a second call site -- an INSTANCE was
repaired where a CLASS was reported.

Below ~1e-162 per component ``np.linalg.norm`` squares to exactly zero, so a
demonstrably wrong eigenvector comes back with residual ``0.0``. Zero is not a
small residual; it is the strongest possible claim (an exact eigenpair), and it
makes ``offending`` empty, ``certified`` True, and hands D9/D19 precisely the
pair this gate was built to withhold.

The three tests are a set and only mean something together:

* the REGRESSION -- corrupted right eigenvector at rate scale 1e-200,
* the POSITIVE CONTROL -- the identical corruption at rate scale 1, which was
  already rejected before the fix and proves the fixture is a genuinely bad
  eigenvector rather than an artefact of the small scale,
* the NO-OVER-REJECT CONTROL -- the untouched decomposition at rate scale
  1e-200, which must still certify, so the repair cannot be a blanket refusal.

The decomposition is built BY HAND rather than taken from the solver at that
scale: measured, ``zgeev`` returns eigenvalues near 6.7e-139 for a generator
whose entries are 1e-200 (a LAPACK scaling artefact of its own), and a test
that depended on those numbers would be measuring the solver, not the gate.
Eigenvectors are scale-invariant, so the exact scaled decomposition is the
c = 1 vectors with the c = 1 eigenvalues multiplied by c.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg as sla

import liouscope.numerics.linalg as la
from liouscope._consts import VECTOR_RESIDUAL_REL_MAX
from liouscope.core.lindblad import build_liouvillian

# Amplitude damping on one qubit -- the reviewer's generator.
_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_BASE = build_liouvillian(np.zeros((2, 2), dtype=complex), [_SIGMA_MINUS])


def _exact_decomposition(c: float) -> tuple[np.ndarray, la.EigenDecomposition]:
    """``(c * L, decomposition of c * L)``, exact by construction."""
    unit = la.eig_nonhermitian(_BASE, compute_left=True)
    assert unit.left_vectors is not None
    return c * _BASE, la.EigenDecomposition(
        eigenvalues=unit.eigenvalues * c,
        right_vectors=unit.right_vectors.copy(),
        left_vectors=unit.left_vectors.copy(),
    )


def _with_a_wrong_right_vector(
    decomp: la.EigenDecomposition,
) -> tuple[la.EigenDecomposition, int]:
    """Replace the fastest mode's right vector by a wrong UNIT vector."""
    assert decomp.left_vectors is not None
    vr = decomp.right_vectors.copy()
    j = int(np.argmax(np.abs(decomp.eigenvalues)))
    n = vr.shape[0]
    wrong = np.zeros(n, dtype=complex)
    wrong[(j + 1) % n] = 1.0
    vr[:, j] = wrong
    return (
        la.EigenDecomposition(
            eigenvalues=decomp.eigenvalues.copy(),
            right_vectors=vr,
            left_vectors=decomp.left_vectors.copy(),
        ),
        j,
    )


def _with_a_wrong_left_vector(
    decomp: la.EigenDecomposition,
) -> tuple[la.EigenDecomposition, int]:
    """Same corruption on the LEFT vector.

    The finding names both residuals, and the two are separate expressions
    with separate norms: repairing only ``res_r`` would leave exactly the
    round-24-to-round-25 pattern in place one line lower. Round 15 already
    measured that the LEFT vector is the one that goes bad on a real fixture
    while the right vectors are fine, so this is not a symmetry argument.
    """
    assert decomp.left_vectors is not None
    vl = decomp.left_vectors.copy()
    j = int(np.argmax(np.abs(decomp.eigenvalues)))
    n = vl.shape[0]
    wrong = np.zeros(n, dtype=complex)
    wrong[(j + 1) % n] = 1.0
    vl[:, j] = wrong
    return (
        la.EigenDecomposition(
            eigenvalues=decomp.eigenvalues.copy(),
            right_vectors=decomp.right_vectors.copy(),
            left_vectors=vl,
        ),
        j,
    )


def _only_this_decomposition(monkeypatch, decomp: la.EigenDecomposition) -> None:
    """Make ``decomp`` the ladder's ONLY candidate.

    ``certified_eig`` also offers a ``dgeev-real`` route for a real operator,
    and the amplitude-damping superoperator is real. Left in place it supplies
    a correct decomposition, the gate certifies THAT one, and the test would
    pass for a reason having nothing to do with the norms under examination.
    ``scipy.linalg.eig`` is raised out inside the ladder own
    ``contextlib.suppress``; the patched primary is what remains.
    """

    def _primary(A: np.ndarray, *, compute_left: bool = False) -> la.EigenDecomposition:
        return decomp

    def _no_second_route(*args: object, **kwargs: object) -> object:
        raise sla.LinAlgError("dgeev-real route disabled for this test")

    monkeypatch.setattr(la, "eig_nonhermitian", _primary)
    monkeypatch.setattr(la.sla, "eig", _no_second_route)


def test_column_norms_do_not_lose_a_nonzero_column() -> None:
    """The helper itself, at the seam the gate now uses.

    Third column is nonzero but every component is below the ~1.5e-162 cliff;
    the raw NumPy answer is exactly ``0.0``. The healthy columns must come back
    BIT FOR BIT unchanged, and a genuinely zero column must stay ``0.0``
    rather than being rescued into something nonzero.
    """
    m = np.zeros((2, 4), dtype=complex)
    m[:, 0] = [3.0, 4.0]  # norm 5, ordinary
    m[:, 1] = 0.0  # genuinely zero
    m[:, 2] = [3.0e-200, 4.0e-200]  # norm 5e-200, underflows when squared
    m[:, 3] = [1.0j, 0.0]  # norm 1, complex

    raw = np.linalg.norm(m, axis=0)
    assert raw[2] == 0.0, "fixture no longer exercises the underflow"

    safe = la.underflow_safe_column_norms(m)
    assert safe[0] == raw[0]
    assert safe[1] == 0.0
    assert safe[3] == raw[3]
    # ``abs=0.0`` is not decoration. ``pytest.approx(5e-200, rel=1e-12)``
    # carries a DEFAULT absolute tolerance of 1e-12, against which 0.0 and
    # 5e-200 are the same number -- so the assertion would have passed with
    # the rescue removed. Measured: the round-25 mutation run reported this
    # test BLIND until the absolute tolerance was pinned to zero. A guard
    # against a false zero must not itself be unable to see a zero.
    assert safe[2] == pytest.approx(5.0e-200, rel=1e-12, abs=0.0)
    assert safe[2] > 0.0


def test_wrong_eigenvector_is_rejected_at_tiny_rate_units(monkeypatch) -> None:
    """THE regression. Certified=True here means D9/D19 consume a wrong pair."""
    c = 1.0e-200
    lsup, exact = _exact_decomposition(c)
    corrupted, j = _with_a_wrong_right_vector(exact)

    # The fixture must actually sit on the cliff, or the test proves nothing.
    offending_column = lsup @ corrupted.right_vectors[:, j] - (
        corrupted.eigenvalues[j] * corrupted.right_vectors[:, j]
    )
    assert np.linalg.norm(offending_column) == 0.0, "raw norm no longer underflows"
    true_residual = la.underflow_safe_norm(offending_column)
    assert true_residual > 0.0
    assert true_residual > VECTOR_RESIDUAL_REL_MAX * abs(corrupted.eigenvalues[j])

    _only_this_decomposition(monkeypatch, corrupted)
    _decomp, cert = la.certified_eig(lsup)

    assert cert.applicable is True
    assert cert.certified is False, (
        "the eigenvector gate accepted a wrong eigenvector because its "
        "residual norm underflowed to zero"
    )
    assert cert.resolved is False


def test_wrong_LEFT_eigenvector_is_rejected_at_tiny_rate_units(monkeypatch) -> None:
    """The second residual expression, which has its own pair of norms.

    Without this test the ``res_l`` half of the repair is unmeasured, and an
    unmeasured half is how this finding reached round 25 in the first place.
    """
    c = 1.0e-200
    lsup, exact = _exact_decomposition(c)
    corrupted, j = _with_a_wrong_left_vector(exact)

    offending_column = lsup.conj().T @ corrupted.left_vectors[:, j] - (
        np.conj(corrupted.eigenvalues[j]) * corrupted.left_vectors[:, j]
    )
    assert np.linalg.norm(offending_column) == 0.0, "raw norm no longer underflows"
    assert la.underflow_safe_norm(offending_column) > (
        VECTOR_RESIDUAL_REL_MAX * abs(corrupted.eigenvalues[j])
    )

    _only_this_decomposition(monkeypatch, corrupted)
    _decomp, cert = la.certified_eig(lsup)

    assert cert.applicable is True
    assert cert.certified is False, (
        "the LEFT residual still uses a norm that underflows to a false zero"
    )
    assert cert.resolved is False


def test_the_same_wrong_eigenvector_is_rejected_at_rate_units_of_one(
    monkeypatch,
) -> None:
    """Positive control: the corruption is bad physics, not a small number.

    Without this the regression above could pass for any reason that refuses
    a 1e-200 generator wholesale. Here the identical construction at c = 1 --
    where no norm underflows and the gate was already correct -- must reach
    the same verdict.
    """
    lsup, exact = _exact_decomposition(1.0)
    corrupted, j = _with_a_wrong_right_vector(exact)
    offending_column = lsup @ corrupted.right_vectors[:, j] - (
        corrupted.eigenvalues[j] * corrupted.right_vectors[:, j]
    )
    assert np.linalg.norm(offending_column) > 0.0, "control must NOT underflow"

    _only_this_decomposition(monkeypatch, corrupted)
    _decomp, cert = la.certified_eig(lsup)

    assert cert.applicable is True
    assert cert.certified is False


def test_an_intact_decomposition_still_certifies_at_tiny_rate_units(
    monkeypatch,
) -> None:
    """No-over-reject control: the repair must not refuse valid physics.

    Same generator, same rate units, uncorrupted vectors. If this ever turns
    red the fix has become a blanket withhold and the two tests above stop
    being evidence about eigenvectors.
    """
    c = 1.0e-200
    lsup, exact = _exact_decomposition(c)

    _only_this_decomposition(monkeypatch, exact)
    _decomp, cert = la.certified_eig(lsup)

    assert cert.applicable is True
    assert cert.certified is True
    assert cert.resolved is True
