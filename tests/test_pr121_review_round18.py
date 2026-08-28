"""Round-18 review of PR #121: the SECOND repair step must not end the ladder.

Round 17 guarded the primary solve, so a ``zgeev`` non-convergence no longer
kills the repair chain. The ``dgeev-real`` step that runs next was left
unguarded in ``certified_eigvals`` -- and unlike the primary it sits BEFORE two
further routes (``zgees-schur``, ``balanced-zgeev``), so its exception
propagates out of the generator and discards two repairs that could still have
certified the spectrum.

The sibling ladder in ``certified_eig`` already suppressed exactly this step.
The omission was the inconsistency, not the guard there.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian

# Same stiff, exactly real classical network as round 17: it is real, so the
# ``dgeev-real`` route exists, and stiff, so the later routes have something
# to do.
_STIFF_PAIRS = [(0, 3), (0, 2), (1, 0), (3, 2), (2, 1)]
_STIFF_RATES = [7.28e-6, 3.67e-5, 1.53e-5, 2.70e5, 1.42e-5]


def _real_classical_network() -> np.ndarray:
    jumps = []
    for to, frm in _STIFF_PAIRS:
        j = np.zeros((4, 4), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, _STIFF_RATES)


def _raise_nonconvergence(*_a, **_kw):
    raise np.linalg.LinAlgError("eig algorithm (zgeev) did not converge")


def test_the_generator_is_exactly_real_so_the_route_is_reached() -> None:
    """Precondition, asserted rather than assumed.

    ``dgeev-real`` is only yielded for an EXACTLY real generator. If this
    fixture ever stopped being exactly real, every test below would pass by
    never entering the branch under test -- a green that proves nothing.
    """
    L = np.asarray(_real_classical_network(), dtype=complex)
    assert not np.any(L.imag)


def test_eigvals_ladder_continues_past_a_failing_real_driver(monkeypatch) -> None:
    """THE regression: the second repair step must not abandon the later ones.

    Both ``np.linalg.eigvals`` call sites are made to raise, which removes the
    ``dgeev-real`` AND ``balanced-zgeev`` routes. Exactly one route is left --
    ``zgees-schur`` -- and the ladder has to reach it.
    """
    from liouscope.numerics import linalg as la

    L = _real_classical_network()
    monkeypatch.setattr(la, "eig_nonhermitian", _raise_nonconvergence)
    monkeypatch.setattr(np.linalg, "eigvals", _raise_nonconvergence)

    # Asserted, not crashed into: an uncaught LinAlgError would end the test
    # without saying which step let it through.
    ended: Exception | None = None
    ev = cert = None
    try:
        ev, cert = la.certified_eigvals(L)
    except np.linalg.LinAlgError as exc:
        ended = exc

    assert ended is None, f"the real-driver repair step ended the ladder: {ended!r}"
    assert cert is not None and ev is not None
    assert cert.applicable is True
    assert cert.certified is True
    assert cert.solver == "zgees-schur", (
        "the only route left had to be the one that certified"
    )
    assert ev.size == 16


def test_the_real_driver_route_is_genuinely_load_bearing(monkeypatch) -> None:
    """Positive control for the test above.

    Without this, the previous test could pass because ``zgees-schur`` always
    wins anyway and ``dgeev-real`` never mattered. Here only the primary is
    broken: ``dgeev-real`` is intact, and it must be the route that certifies.
    If this ever reports another solver, the test above stops being evidence
    about the real-driver step.
    """
    from liouscope.numerics import linalg as la

    L = _real_classical_network()
    monkeypatch.setattr(la, "eig_nonhermitian", _raise_nonconvergence)

    ev, cert = la.certified_eigvals(L)
    assert cert.certified is True
    assert cert.solver == "dgeev-real"
    assert ev.size == 16


def test_a_total_solver_failure_still_raises(monkeypatch) -> None:
    """Fail-closed control: suppressing the step must not swallow the verdict.

    With every route broken the original error has to surface. A guard that
    turned a total failure into a silent success would be a worse defect than
    the one being repaired.
    """
    from liouscope.numerics import linalg as la
    import scipy.linalg as sla

    L = _real_classical_network()
    monkeypatch.setattr(la, "eig_nonhermitian", _raise_nonconvergence)
    monkeypatch.setattr(np.linalg, "eigvals", _raise_nonconvergence)
    monkeypatch.setattr(sla, "schur", _raise_nonconvergence)
    monkeypatch.setattr(sla, "matrix_balance", _raise_nonconvergence)

    with pytest.raises(np.linalg.LinAlgError, match="did not converge"):
        la.certified_eigvals(L)


def test_the_two_ladders_guard_the_same_step(monkeypatch) -> None:
    """Structural half: the sibling ladders must not drift apart again.

    The defect existed because one ladder guarded the real-driver step and the
    other did not. This reads the source and fails if either ``dgeev-real``
    yield is left outside a suppression block.
    """
    import pathlib

    import liouscope.numerics.linalg as la

    quelle = pathlib.Path(la.__file__).read_text(encoding="utf-8").splitlines()
    stellen = [i for i, ln in enumerate(quelle) if '"dgeev-real"' in ln]
    assert len(stellen) == 2, f"expected two dgeev-real routes, found {len(stellen)}"

    for i in stellen:
        # Walk back over the yield's own lines to the nearest enclosing
        # statement; a guarded route has ``contextlib.suppress`` within the
        # few lines above it and at a shallower indent.
        fenster = quelle[max(0, i - 8):i]
        assert any("contextlib.suppress" in ln for ln in fenster), (
            f"the dgeev-real route at line {i + 1} is not inside a "
            f"suppression block -- an exception there ends the ladder"
        )
