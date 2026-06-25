"""Tests for the temporal holdout / train-test split (LIOU-A-012).

Oracle = the generalisation behaviour we know analytically:

* A correct model on clean exponential data generalises: the held-out tail is
  predicted about as well as the training segment (accept).
* A model fitted on a window that does NOT reveal the full structure fails
  out-of-sample: when the held-out tail rises away from the fitted monotone
  decay, the holdout error explodes and the gate rejects (BLOCK) -- exactly the
  anti-overfit / wrong-model guard the entry targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.fitting.holdout import holdout_validate, train_holdout_split
from liouscope.fitting.models import M0


def test_split_is_temporal_tail():
    t = np.linspace(0.0, 1.0, 20)
    y = np.arange(20.0)
    t_tr, y_tr, t_ho, y_ho = train_holdout_split(t, y, holdout_frac=0.25)
    assert t_tr.size == 15 and t_ho.size == 5
    # Holdout is the contiguous final segment (no shuffling).
    np.testing.assert_array_equal(y_ho, np.arange(15.0, 20.0))
    np.testing.assert_array_equal(t_ho, t[15:])


def test_clean_exponential_generalises():
    t = np.linspace(0.0, 8.0, 120)
    y = 1.3 * np.exp(-0.7 * t) + 0.002 * np.sin(50 * t)  # tiny deterministic jitter
    res = holdout_validate(M0, t, y, np.array([1.0, 1.0]), holdout_frac=0.25)
    assert res.accept
    assert res.n_train + res.n_holdout == t.size
    assert res.holdout_rmse < 0.05


def test_wrong_model_fails_out_of_sample():
    # Train window is a clean decay; the held-out tail then RISES, which a
    # monotone single exponential cannot predict -> holdout error >> train.
    t = np.linspace(0.0, 10.0, 120)
    y = np.exp(-0.8 * t)
    n = t.size
    n_hold = int(round(n * 0.25))
    y[n - n_hold:] += np.linspace(0.0, 1.5, n_hold)  # tail departs upward
    res = holdout_validate(M0, t, y, np.array([1.0, 1.0]), holdout_frac=0.25, delta=0.5)
    assert not res.accept
    assert res.ratio > 1.5


# --- negative / edge-input gates ---------------------------------------------


@pytest.mark.parametrize("frac", [0.0, 1.0, 1.5, -0.2, float("nan")])
def test_split_rejects_bad_fraction(frac):
    t = np.linspace(0.0, 1.0, 20)
    with pytest.raises(ValueError, match="holdout_frac"):
        train_holdout_split(t, t, holdout_frac=frac)


def test_split_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        train_holdout_split(np.zeros(10), np.zeros(9))


def test_split_rejects_too_few_points():
    with pytest.raises(ValueError, match=">=2 points"):
        train_holdout_split(np.zeros(3), np.zeros(3), holdout_frac=0.5)
