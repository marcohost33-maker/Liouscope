"""PR #127: the Hermiticity tolerance is relative to the GENERATOR, not to H alone.

Why a third predicate was needed at all -- measured, not argued:

Two review-pinned fixtures had to get OPPOSITE verdicts, and the two predicates
on the table gave them the SAME one:

* ``F1 = I + 2**-53 * e01`` with the jump operator sigma-, the numerically pure
  gauge Hamiltonian of PR #127 round 18/19 (``tests/test_pr127_review_round19.py``),
  which must be ACCEPTED;
* ``F2 = 1e308 * I + 1e290 * e01`` with no jump operators, the overflow fixture
  of PR #121 round 19/22 (``tests/test_pr121_review_round19.py``,
  ``tests/test_pr121_review_round22.py``), which must be REJECTED.

Both have the form ``c * I + N`` with ``N`` strictly upper triangular. For every
such matrix the gauge shift is ``c`` exactly, the gauge-fixed part is ``N``, and
``max|H - H^dag| = max|N|`` because ``N`` and ``N^dag`` have disjoint supports --
so ``defect / gauge_scale`` is exactly 1 for both, by construction. Worse, a
gauge shift (twelfth-round review) followed by a change of units (issue #109)
maps BOTH onto ``e01``. Any test that reads only ``H`` and respects both
symmetries is therefore constant on that orbit and cannot separate them: the
gauge-fixed relative gate on ``main`` rejected both, and the round-18 allowance
``d * eps * |gauge_shift|`` -- which breaks the gauge symmetry -- accepted both.

The information that separates them is not in ``H``. It is in the dissipator:
the physical object is the generator, whose only part that fails to preserve
Hermiticity is ``-i[A, .]`` with ``A`` the anti-Hermitian part of ``H`` (the
dissipator is Hermiticity preserving by construction). ``H`` and
``K / 2 = sum_k gamma_k L_k^dag L_k / 2`` enter the generator on the same
footing, as ``H_eff = H - i K / 2``, so the defect is now measured against
``max(max|H_gauge|, max|K| / 2)``. That scale is gauge invariant (``K`` does
not see ``c``), covariant under a change of units, and with no dissipation it
IS the old gauge-fixed scale -- every purely coherent verdict is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from liouscope.core.lindblad import build_liouvillian
from liouscope.sparse.build import build_sparse_liouvillian

_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_EPS = float(np.finfo(float).eps)


def _verdicts(
    H: np.ndarray, jumps: list[np.ndarray], rates: list[float] | None = None
) -> tuple[str | None, str | None]:
    """One outcome per builder: ``None`` if accepted, else the refusal.

    The TYPE and message are the measurement, so the exception is caught and
    returned rather than allowed to end the test: a red that comes from an
    uncaught throw cannot be attributed to the condition that failed.
    """
    out: list[str | None] = []
    for build, conv in (
        (build_liouvillian, lambda m: m),
        (build_sparse_liouvillian, sp.csr_matrix),
    ):
        try:
            with np.errstate(all="ignore"):
                build(conv(H), [conv(j) for j in jumps], rates)
        except Exception as exc:  # the TYPE is the finding, so catch broadly
            out.append(f"{type(exc).__name__}: {exc}")
        else:
            out.append(None)
    return out[0], out[1]


def _one_orbit_premise(H: np.ndarray) -> None:
    """The fixture must sit on the orbit that made both old predicates blind."""
    defect = float(np.max(np.abs(H - H.conj().T)))
    # Divide BEFORE summing: ``mean()`` of ``[1e308, 1e308]`` overflows, and
    # under ``filterwarnings = ["error"]`` that killed the first draft of this
    # very premise -- the overflow class this file is about, in its own guard.
    shift = float(np.sum(np.real(np.diagonal(H)) / H.shape[0]))
    gauge_scale = float(np.max(np.abs(H - shift * np.eye(H.shape[0]))))
    assert gauge_scale > 0.0
    assert defect / gauge_scale == 1.0, "fixture left the c*I + N orbit"


# ---------------------------------------------------------------------------
# The two fixtures, and the residual PR #127 recorded as OPEN
# ---------------------------------------------------------------------------


def test_the_overflow_fixture_is_rejected_although_it_shares_the_orbit() -> None:
    """F2, and it must be refused by BOTH builders -- same orbit as F1."""
    H = np.array([[1.0e308, 1.0e290], [0.0, 1.0e308]], dtype=complex)
    _one_orbit_premise(H)
    dense, sparse = _verdicts(H, [])
    assert dense is not None and "Hermitian" in dense, dense
    assert sparse is not None and "Hermitian" in sparse, sparse


def test_the_unit_defect_on_a_huge_gauge_term_is_rejected() -> None:
    """The residual PR #127 left OPEN: ``[[1e308, 1], [0, 1e308]]``.

    Its generator does not preserve Hermiticity (``||drho - drho^dag||_F =
    0.632456`` on a Hermitian rho, measured when the residual was recorded).
    The round-18 allowance excused it because ``2 * eps * 1e308 = 4.4e292``.
    Refused with and without a unit-rate dissipator: a defect of 1 is not
    small against a dissipation scale of 1/2 either.
    """
    H = np.array([[1.0e308, 1.0], [0.0, 1.0e308]], dtype=complex)
    _one_orbit_premise(H)
    for jumps in ([], [_SIGMA_MINUS]):
        dense, sparse = _verdicts(H, jumps)
        assert dense is not None and "Hermitian" in dense, (len(jumps), dense)
        assert sparse is not None and "Hermitian" in sparse, (len(jumps), sparse)


def test_the_pure_gauge_fixture_is_accepted_because_of_its_dissipator() -> None:
    """F1 with sigma-: accepted, and for a stated reason.

    ``defect = 2**-53`` against a dissipation scale of ``1/2`` is a relative
    defect of 2.2e-16, seven orders below the gate. The same H reappears in
    the negative controls below with the dissipation taken away.
    """
    H = np.eye(2, dtype=complex)
    H[0, 1] = 0.5 * _EPS
    _one_orbit_premise(H)
    assert _verdicts(H, [_SIGMA_MINUS]) == (None, None)


# ---------------------------------------------------------------------------
# Negative controls: the excuse is a MAGNITUDE, not the presence of a jump op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0, 1.0e-9])
def test_dissipation_too_weak_to_matter_excuses_nothing(rate: float) -> None:
    """Same H as F1, same sigma-, but a rate that cannot outweigh the defect.

    ``rate = 0`` means no dissipation at all, so the jump operator is present
    and inert; ``1e-9`` gives ``EPS * rate / 2 = 5e-19`` below the defect of
    ``1.1e-16``. A gate that excused on the PRESENCE of jump operators would
    accept both.
    """
    H = np.eye(2, dtype=complex)
    H[0, 1] = 0.5 * _EPS
    dense, sparse = _verdicts(H, [_SIGMA_MINUS], [rate])
    assert dense is not None and "Hermitian" in dense, dense
    assert sparse is not None and "Hermitian" in sparse, sparse


def test_an_overflowing_dissipator_excuses_nothing() -> None:
    """``L = 1e200 * sigma-`` makes ``L^dag L`` overflow to inf.

    An infinite dissipation scale would make ``defect <= EPS * scale`` true for
    every defect -- the fail-open this PR has now closed in four other places.
    The excuse must be refused, not granted, when it cannot be computed.
    """
    H = np.array([[1.0, 1.0], [0.0, -1.0]], dtype=complex)
    dense, sparse = _verdicts(H, [1.0e200 * _SIGMA_MINUS])
    assert dense is not None and "Hermitian" in dense, dense
    assert sparse is not None and "Hermitian" in sparse, sparse


# ---------------------------------------------------------------------------
# The two symmetries, asserted as verdict equality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("defect", [1.0e-12, 1.0e-6])
def test_the_verdict_does_not_move_with_the_gauge(defect: float) -> None:
    """``H + c*I`` for exactly representable ``c``: one physical H, one verdict.

    Traceless part of scale 1, off-diagonal defect ``defect``, unit-rate
    sigma-. The off-diagonal entries do not see ``c``, and ``1 + c`` is exact
    for every ``c`` listed, so the stored traceless part is bit-identical
    across the sweep. Measured before this change: with the round-18
    allowance the 1e-6 defect was rejected at ``c = 0`` and accepted from
    ``c ~ 2.25e9`` on.
    """
    base = np.array([[1.0, defect], [0.0, -1.0]], dtype=complex)
    seen = set()
    for c in (0.0, 1.0, 1.0e9, 2.0**40, 1.0e15):
        H = base + c * np.eye(2, dtype=complex)
        dense, sparse = _verdicts(H, [_SIGMA_MINUS])
        seen.add((dense is None, sparse is None))
    assert len(seen) == 1, f"verdict moved with the gauge: {seen}"
    expected_accept = defect <= 1.0e-9
    assert seen == {(expected_accept, expected_accept)}


@pytest.mark.parametrize("unit", [1.0e-6, 1.0, 1.0e6])
def test_the_verdict_does_not_move_with_the_units(unit: float) -> None:
    """Rescaling H and every rate by one factor is a change of units.

    F1 stays accepted and F2's shape (no dissipator) stays rejected at every
    unit; ``unit`` is a power of ten, so the pure-gauge diagonal is exact.
    """
    H = unit * np.eye(2, dtype=complex)
    H[0, 1] = unit * 0.5 * _EPS
    assert _verdicts(H, [_SIGMA_MINUS], [unit]) == (None, None)
    dense, sparse = _verdicts(H, [])
    assert dense is not None and sparse is not None
