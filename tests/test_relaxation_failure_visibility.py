"""D7b failure surface: a swallowed exception is not a measurement (finding E).

Background (cross-family review 2026-09-02, ported 2026-09-09)
-------------------------------------------------------------
``compute_relaxation_layer`` guarded its D7b call with a bare
``except Exception:`` that swallowed the failure WORDLESSLY -- twenty lines
below a comment block arguing, correctly, that a swallowed numerical failure is
worse than a loud one ("Fail loud, not certain").

Two distinct defects sat in that one line:

* ``Exception`` also catches ``TypeError``/``AttributeError``/``KeyError``,
  i.e. PROGRAMMING errors inside :func:`entanglement_asymmetry` itself. A typo
  in the estimator surfaced as a NaN diagnostic -- a defect in the code
  reported as a property of the physics.
* Nothing was emitted, so a run whose D7b value is UNKNOWN was
  indistinguishable from one where it was genuinely not computable (``d != 4``
  returns NaN by contract, without raising).

The NaN RESULT is deliberately unchanged by the fix; only the caught exception
surface and the visibility change. The tests below pin both halves plus a
positive control, so they cannot pass by refusing everything.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.diagnostics import relaxation as rx


def _decay_generator() -> np.ndarray:
    """Amplitude-damped qubit -- ``d = 2``, so D7b is NaN by contract, not by failure."""
    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    return build_liouvillian(np.zeros((2, 2), dtype=complex), [sm], [1.0])


def _raise(exc: BaseException):
    def _fn(_rho):
        raise exc

    return _fn


def test_a_programming_error_in_d7b_is_not_swallowed(monkeypatch) -> None:
    """The load-bearing regression: pre-fix this returned a result with NaN D7b.

    ``TypeError`` is the signature of a bug in the estimator, not of an
    ill-conditioned input. It must reach the caller as a traceback.
    """
    monkeypatch.setattr(
        rx, "entanglement_asymmetry", _raise(TypeError("bad argument in D7b"))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="bad argument in D7b"):
            rx.compute_relaxation_layer(_decay_generator(), bootstrap_B=20)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("ill-conditioned block"),
        RuntimeError("eigensolver gave up"),
        np.linalg.LinAlgError("eigenvalues did not converge"),
    ],
)
def test_a_numerical_d7b_failure_is_audible_and_still_reports_unknown(
    monkeypatch, exc: Exception
) -> None:
    """Genuine numerical failures stay caught -- but they now say so.

    The three caught types are the documented numerical surface. The returned
    value is unchanged (``None``, the report-layer encoding of NaN): this fix
    changes visibility, not verdicts.

    Recorded rather than asserted via ``pytest.warns`` ON PURPOSE: a missing
    warning kills ``pytest.warns`` with ``Failed: DID NOT WARN``, which a
    retraction probe must classify as a crash, not as the test's own verdict.
    An explicit assertion over the recorded warnings makes the death cause an
    ``AssertionError``, i.e. attributable.
    """
    monkeypatch.setattr(rx, "entanglement_asymmetry", _raise(exc))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rx.compute_relaxation_layer(_decay_generator(), bootstrap_B=20)
    matching = [
        w for w in caught if "entanglement_asymmetry failed" in str(w.message)
    ]
    assert matching, f"no D7b failure warning; got {[str(w.message) for w in caught]}"
    assert issubclass(matching[0].category, RuntimeWarning)
    assert repr(exc) in str(matching[0].message)  # the cause must be named
    assert result.entanglement_asymmetry is None


def test_positive_control_the_ordinary_path_stays_silent() -> None:
    """Without an induced failure no D7b warning is emitted.

    Without this control, "the warning fired" and "the warning always fires"
    are indistinguishable, and the two tests above would pass against a
    function that warns unconditionally. ``d = 2`` returns NaN through the
    documented contract path, i.e. WITHOUT entering the except branch.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = rx.compute_relaxation_layer(_decay_generator(), bootstrap_B=20)
    assert result.entanglement_asymmetry is None
    assert not [
        w for w in caught if "entanglement_asymmetry failed" in str(w.message)
    ]
