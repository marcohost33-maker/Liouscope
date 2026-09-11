"""PR #121 round-26 review finding: one eigenPAIR, matched to one mode.

``certified_nonzero_modes`` decided which eigenpair belongs to an in-band
candidate with ``argmin(|ref - lam|)`` -- a lookup BY VALUE. On a degenerate
spectrum that is not a bijection. ``np.argmin`` returns the FIRST index
attaining the minimum, so every mode sharing an eigenvalue borrowed the SAME
vectors, and the a-posteriori certificate for the second mode was computed from
evidence measured on the first.

The escape is not caught downstream. ``certified_eig``'s ``_vector_residual``
gate tests only modes ABOVE the raw ``bound``, and a mode this refinement
rescues is by construction below it, so a rescued in-band pair is never
revalidated before D9/D19 consume it.

Both directions of the mis-match are defects and both are pinned here:

* a CORRUPT mode is certified on a healthy neighbour's residual (the reviewer's
  case), and
* a HEALTHY mode is refused on a corrupt neighbour's residual, which happens
  whenever the corrupt duplicate comes first.

The parameter range is chosen so that it CONTAINS the boundary: multiplicity
``1`` is the non-degenerate case, where ``argmin`` was always a bijection and
the repair must change nothing, and ``2``/``3`` are the degenerate ones. The
corrupted position runs over every slot including ``None`` (no corruption at
all), so over-rejection is measured as explicitly as under-rejection. Measured
against the pre-fix code, 5 of the 9 combinations disagree with ground truth
and all 5 are degenerate; all 4 non-degenerate combinations agree, which is the
evidence that the healthy path is untouched.

The corruption is a real eigenvector of the SAME operator -- the fast mode's --
rather than a random vector, so the fixture cannot be dismissed as noise: its
residual against the slow eigenvalue is the fast rate itself, ~1.0.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import scipy.linalg as sla

import liouscope.numerics.linalg as la
from liouscope._consts import ZERO_MODE_EPS_FACTOR

_SLOW = -1.0e-14


def _fixture(
    mult: int, corrupt_pos: int | None, *, seed: int = 20260904
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Operator with ``mult`` EXACTLY equal slow eigenvalues, one of them bad.

    Built from a prescribed left-eigenvector basis rather than from a solver,
    because a solver on a degenerate spectrum returns an arbitrary basis of the
    eigenspace and the test would then be measuring LAPACK, not the matching.
    """
    rng = np.random.default_rng(seed)
    n = 2 + mult  # one zero mode, ``mult`` slow modes, one fast mode
    W = np.eye(n, dtype=complex) + 0.3 * rng.standard_normal((n, n))
    V = np.linalg.inv(W)
    lam = np.array([0.0] + [_SLOW] * mult + [-1.0], dtype=complex)
    L = V @ np.diag(lam) @ W
    vl = np.conj(W).T  # scipy convention: vl[:, i] is the left vector of lam[i]
    vr = V.copy()
    if corrupt_pos is not None:
        vr[:, 1 + corrupt_pos] = V[:, n - 1]
    return L, lam, vr, vl


def _in_band(L: np.ndarray, lam: np.ndarray) -> np.ndarray:
    bound = ZERO_MODE_EPS_FACTOR * float(np.finfo(float).eps) * float(
        np.linalg.norm(L, 2)
    )
    return np.abs(lam) <= bound


@pytest.mark.parametrize(
    ("mult", "corrupt_pos"),
    [
        (1, None), (1, 0),
        (2, None), (2, 0), (2, 1),
        (3, None), (3, 0), (3, 1), (3, 2),
    ],
)
def test_certificate_uses_each_mode_s_own_eigenpair(
    mult: int, corrupt_pos: int | None
) -> None:
    """Exactly the modes whose OWN vector is sound may be certified."""
    L, lam, vr, vl = _fixture(mult, corrupt_pos)

    # The fixture is only meaningful if the corruption is genuinely bad.
    if corrupt_pos is not None:
        k = 1 + corrupt_pos
        x = vr[:, k] / np.linalg.norm(vr[:, k])
        assert float(np.linalg.norm(L @ x - lam[k] * x)) > 0.1

    out = la.certified_nonzero_modes(
        L, lam, _in_band(L, lam), right_vectors=vr, left_vectors=vl
    )
    got = [bool(out[1 + k]) for k in range(mult)]
    assert got == [corrupt_pos != k for k in range(mult)]


def test_borrowed_decomposition_matches_one_to_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without supplied vectors, a corrupt borrowed vector must not decide anything.

    ROUND-28 CHANGED THE ANSWER HERE, and the reason is the finding this file
    exists for taken one step further. The two slow modes are EXACTLY
    degenerate, so "which of the two borrowed eigenvectors belongs to which
    mode" has no answer -- any assignment is as good as any other, and the
    pre-round-26 ``argmin`` picked one and certified both from it. Round 26 made
    the assignment one-to-one, which removed the borrowing but left the choice
    arbitrary: this test then measured which arm of a tie the search happened to
    take.

    Round 28 stops making the choice. A degenerate cluster is certified as an
    INVARIANT SUBSPACE, from the operator's own Schur basis, so the borrowed
    eigenvectors are not consulted for it at all -- which is both the LAPACK
    reading (a clustered subspace can be well conditioned while its individual
    vectors are not determined) and strictly stronger here: the verdict can no
    longer be changed by corrupting them, and both modes are certified because
    both genuinely sit at ``_SLOW``, away from the stationary mode.

    The stub therefore has to carry ``schur`` as well; it still controls the
    eigenVALUES the certificate is handed, which is what this fixture is about.
    """
    L, lam, vr, vl = _fixture(2, 1)
    fake = types.SimpleNamespace(
        eig=lambda A, left=False, right=False: (lam.copy(), vl.copy(), vr.copy()),
        schur=sla.schur,
        LinAlgError=sla.LinAlgError,
    )
    monkeypatch.setattr(la, "sla", fake)
    out = la.certified_nonzero_modes(L, lam, _in_band(L, lam))
    assert [bool(out[1]), bool(out[2])] == [True, True]

    # The load-bearing half: the same call with the borrowed right vectors
    # INTACT must give the identical verdict. If corruption could move it, the
    # cluster would still be resting on those vectors.
    L_ok, lam_ok, _vr_ok, vl_ok = _fixture(2, None)
    fake_ok = types.SimpleNamespace(
        eig=lambda A, left=False, right=False: (
            lam_ok.copy(), vl_ok.copy(), _vr_ok.copy()
        ),
        schur=sla.schur,
        LinAlgError=sla.LinAlgError,
    )
    monkeypatch.setattr(la, "sla", fake_ok)
    clean = la.certified_nonzero_modes(L_ok, lam_ok, _in_band(L_ok, lam_ok))
    assert [bool(clean[1]), bool(clean[2])] == [True, True]

    # NEGATIVE CONTROL on the same machinery: the fast mode is nowhere near the
    # zero band, so a certificate that had simply started saying True would be
    # caught here.
    assert bool(out[3]) is False and bool(out[0]) is False


def _trace_preserving_complex() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """4x4 trace-preserving generator, spectrum ``{0, -1e-14, -1e-14, -1}``.

    ``vec(I)^H L = 0`` holds exactly because ``[1, 0, 0, 1]`` is prescribed as
    the left eigenvector of the zero mode. The operator is deliberately COMPLEX
    so that ``certified_eig``'s ``dgeev-real`` route is not offered: with a
    second route available the ladder repairs the corruption by switching
    solvers, which is correct behaviour but would hide whether the gate itself
    still accepts the bad pair.
    """
    W = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.3j, 0.0],
            [0.2, 0.0, 1.0, -0.2],
            [1.0, 0.4j, 0.0, -1.0],
        ],
        dtype=complex,
    )
    V = np.linalg.inv(W)
    lam = np.array([0.0, _SLOW, _SLOW, -1.0], dtype=complex)
    return V @ np.diag(lam) @ W, lam, V.copy(), np.conj(W).T


def test_certified_eig_does_not_resolve_on_a_borrowed_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reviewer's end-to-end claim, at the API a consumer actually calls."""
    L, lam, vr, vl = _trace_preserving_complex()
    assert bool(np.any(L.imag))  # no dgeev-real fallback for this operator
    vr_bad = vr.copy()
    vr_bad[:, 2] = vr[:, 3]
    x = vr_bad[:, 2] / np.linalg.norm(vr_bad[:, 2])
    assert float(np.linalg.norm(L @ x - lam[2] * x)) > 0.5

    monkeypatch.setattr(
        la,
        "eig_nonhermitian",
        lambda A, *, compute_left=False: la.EigenDecomposition(
            eigenvalues=lam.copy(),
            right_vectors=vr_bad.copy(),
            left_vectors=vl.copy(),
        ),
    )
    _, cert = la.certified_eig(L)
    # Pre-fix this returned certified=True AND resolved=True with
    # zero_mode_count 1: the corrupt slow pair had been rescued out of the band
    # on its neighbour's residual, and every consumer was cleared to use it.
    assert cert.resolved is False


def test_certified_eig_still_resolves_the_untouched_decomposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No-over-reject control: the same degenerate operator, vectors intact."""
    L, lam, vr, vl = _trace_preserving_complex()
    monkeypatch.setattr(
        la,
        "eig_nonhermitian",
        lambda A, *, compute_left=False: la.EigenDecomposition(
            eigenvalues=lam.copy(),
            right_vectors=vr.copy(),
            left_vectors=vl.copy(),
        ),
    )
    _, cert = la.certified_eig(L)
    assert cert.certified is True
    assert cert.resolved is True
    assert cert.zero_mode_count == 1
