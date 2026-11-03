"""Self-tests for the order-independent spectrum comparison (issue #180)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import assert_spectra_match


def test_conjugate_pair_with_rounding_tied_real_parts_matches() -> None:
    """The exact macOS-probe failure: same multiset, opposite sort order."""
    ours = np.array(
        [
            -0.6 + 2.775558e-17j,
            -0.3000000000000003 + 0.9539392014169454j,
            -0.30000000000000004 - 0.9539392014169451j,
            -7.504111e-17 + 2.204258e-17j,
        ]
    )
    qutip_like = np.array(
        [
            -0.6 + 0.0j,
            -0.2999999999999998 - 0.9539392014169453j,
            -0.29999999999999977 + 0.9539392014169455j,
            1.409844e-17 - 5.622188e-17j,
        ]
    )
    # Premise: the order-dependent comparison really does fail on this input.
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.sort_complex(ours), np.sort_complex(qutip_like), atol=1e-9)
    assert_spectra_match(ours, qutip_like, atol=1e-9)


def test_genuinely_different_spectra_are_rejected() -> None:
    with pytest.raises(AssertionError):
        assert_spectra_match(np.array([0.0, -1.0 + 1j]), np.array([0.0, -1.0 + 1.1j]), atol=1e-9)


def test_multiplicity_is_respected() -> None:
    """{1, 1, 2} and {1, 2, 2} share every value but are different multisets."""
    with pytest.raises(AssertionError):
        assert_spectra_match(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 2.0]), atol=1e-9)


def test_size_mismatch_is_rejected() -> None:
    with pytest.raises(AssertionError, match="differ in size"):
        assert_spectra_match(np.array([1.0, 2.0]), np.array([1.0]), atol=1e-9)


def test_verdict_does_not_depend_on_a_tie_in_total_distance() -> None:
    """Why the verdict uses feasibility, not the minimum-total-distance pairing.

    For a=[0, 1] against d=[1, 1.9] both pairings have total distance 1.9:
    (0, 1) + (1, 1.9) has deviations 1.0 and 0.9, while (0, 1.9) + (1, 1) has
    1.9 and 0.0. A minimum-total solver may return either. At ``atol=1.0``
    only the first is feasible, so a verdict taken from the minimum-total
    pairing could fail a matching multiset; the feasibility matching cannot.
    """
    assert_spectra_match(np.array([0.0, 1.0]), np.array([1.0, 1.9]), atol=1.0)
    with pytest.raises(AssertionError):
        assert_spectra_match(np.array([0.0, 1.0]), np.array([1.0, 1.9]), atol=0.95)
