"""Tests for the Ljung-Box residual-whiteness gate (LIOU-A-013).

Oracles:

* iid Gaussian noise is white: the gate must NOT reject (``p_value > 0.05``).
* A strongly autocorrelated AR(1) series (phi = 0.85) is NOT white: the gate
  rejects with a large Q and ``p_value < 0.01``.
* A constant (zero-variance) residual has no structure: ``Q = 0``, ``p = 1``.
* The reference distribution is :func:`scipy.stats.chi2`, independent of the Q
  computation, so ``p_value == chi2.sf(Q, dof)`` by construction; we also pin
  ``Q = 0 -> p = 1`` as a closed-form anchor.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from liouscope.fitting.whiteness import ljung_box_q, whiteness_gate


def test_white_noise_passes_gate():
    rng = np.random.default_rng(0)
    resid = rng.standard_normal(500)
    res = ljung_box_q(resid, m=20, n_params=2)
    assert res.is_white
    assert res.p_value > 0.05
    assert res.dof == 18


def test_ar1_series_is_not_white():
    rng = np.random.default_rng(1)
    n = 500
    phi = 0.85
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = eps[0]
    for k in range(1, n):
        x[k] = phi * x[k - 1] + eps[k]
    res = ljung_box_q(x, m=20, n_params=0)
    assert not res.is_white
    assert res.p_value < 0.01
    assert res.q_stat > stats.chi2.ppf(0.99, res.dof)


def test_constant_residual_is_white_q_zero():
    res = ljung_box_q(np.full(100, 3.14), m=10, n_params=0)
    assert res.q_stat == pytest.approx(0.0, abs=1e-12)
    assert res.p_value == pytest.approx(1.0, abs=1e-12)
    assert res.is_white


def test_p_value_matches_scipy_chi2():
    rng = np.random.default_rng(5)
    resid = rng.standard_normal(300)
    res = ljung_box_q(resid, m=15, n_params=4)
    assert res.p_value == pytest.approx(float(stats.chi2.sf(res.q_stat, res.dof)), abs=1e-12)


def test_whiteness_gate_alpha_threshold():
    rng = np.random.default_rng(2)
    resid = rng.standard_normal(400)
    base = ljung_box_q(resid, m=12, n_params=0)
    strict = whiteness_gate(resid, m=12, n_params=0, alpha=min(base.p_value + 0.5, 0.999))
    # A high alpha makes the gate stricter; with p < alpha it must reject.
    assert strict.is_white == (base.p_value > min(base.p_value + 0.5, 0.999))


# --- negative / edge-input gates ---------------------------------------------


def test_rejects_m_geq_n():
    with pytest.raises(ValueError, match="strictly less than"):
        ljung_box_q(np.zeros(10), m=10, n_params=0)


def test_rejects_m_below_one():
    with pytest.raises(ValueError, match="positive integer"):
        ljung_box_q(np.zeros(10), m=0, n_params=0)


def test_rejects_nonpositive_dof():
    with pytest.raises(ValueError, match="degrees of freedom"):
        ljung_box_q(np.zeros(50), m=3, n_params=3)


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        ljung_box_q(np.zeros((10, 2)), m=3, n_params=0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_gate_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        whiteness_gate(np.zeros(50), m=5, alpha=alpha)
