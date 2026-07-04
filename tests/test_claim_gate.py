"""Tests for the anti-overfit claim gate (LIOU-A-014, issue #71 B2).

The gate wires the whiteness (Ljung-Box) and holdout (temporal out-of-sample)
verdicts onto the SAFE/REVIEW/BLOCK claim vocabulary, fail-closed. These tests
exercise the truth table end-to-end against real gate outputs, so an overfit
residual demonstrably downgrades or blocks the claim rather than passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.fitting import (
    assess_relaxation_claim,
    holdout_validate,
    whiteness_gate,
)
from liouscope.fitting.holdout import HoldoutResult
from liouscope.fitting.models import M0
from liouscope.fitting.whiteness import WhitenessResult


def _white() -> WhitenessResult:
    """A passing (white) residual verdict."""
    rng = np.random.default_rng(0)
    return whiteness_gate(rng.standard_normal(200), m=10)


def _not_white() -> WhitenessResult:
    """A failing (structured) residual verdict: a slow ramp is strongly
    autocorrelated, so Ljung-Box rejects whiteness."""
    res = whiteness_gate(np.linspace(-1.0, 1.0, 200), m=10)
    assert not res.is_white  # guard the fixture's premise
    return res


# --- green path --------------------------------------------------------------


def test_both_gates_pass_is_safe():
    assessment = assess_relaxation_claim(whiteness=_white(), holdout=_pass_holdout())
    assert assessment.claim_level == "SAFE"
    assert assessment.reasons == ()


def _pass_holdout() -> HoldoutResult:
    t = np.linspace(0.0, 8.0, 120)
    y = 1.3 * np.exp(-0.7 * t) + 0.002 * np.sin(50 * t)
    res = holdout_validate(M0, t, y, np.array([1.0, 1.0]), holdout_frac=0.25)
    assert res.accept  # guard the fixture's premise
    return res


def _fail_holdout() -> HoldoutResult:
    t = np.linspace(0.0, 10.0, 120)
    y = np.exp(-0.8 * t)
    n = t.size
    n_hold = int(round(n * 0.25))
    y[n - n_hold:] += np.linspace(0.0, 1.5, n_hold)  # tail departs upward
    res = holdout_validate(M0, t, y, np.array([1.0, 1.0]), holdout_frac=0.25, delta=0.5)
    assert not res.accept  # guard the fixture's premise
    return res


# --- red paths (fail-closed) --------------------------------------------------


def test_holdout_reject_blocks():
    assessment = assess_relaxation_claim(whiteness=_white(), holdout=_fail_holdout())
    assert assessment.claim_level == "BLOCK"
    assert any("holdout rejected" in r for r in assessment.reasons)


def test_non_white_residuals_downgrade_to_review():
    assessment = assess_relaxation_claim(whiteness=_not_white(), holdout=_pass_holdout())
    assert assessment.claim_level == "REVIEW"
    assert any("not white" in r for r in assessment.reasons)


def test_block_dominates_review_worst_wins():
    # Holdout rejects (BLOCK) AND residuals are non-white (REVIEW): worst wins.
    assessment = assess_relaxation_claim(
        whiteness=_not_white(), holdout=_fail_holdout()
    )
    assert assessment.claim_level == "BLOCK"
    assert len(assessment.reasons) == 2


def test_missing_holdout_gate_is_review_not_safe():
    assessment = assess_relaxation_claim(whiteness=_white(), holdout=None)
    assert assessment.claim_level == "REVIEW"
    assert any("holdout gate not run" in r for r in assessment.reasons)


def test_missing_whiteness_gate_is_review_not_safe():
    assessment = assess_relaxation_claim(whiteness=None, holdout=_pass_holdout())
    assert assessment.claim_level == "REVIEW"
    assert any("whiteness gate not run" in r for r in assessment.reasons)


def test_no_gates_at_all_fails_closed():
    with pytest.raises(ValueError, match="at least one gate"):
        assess_relaxation_claim(whiteness=None, holdout=None)


def test_recommendation_feeds_stability_report_vocabulary():
    # The returned claim_level is exactly a StabilityReport ClaimLevel string.
    from liouscope.io.stability_report import _CLAIM_LEVELS

    assessment = assess_relaxation_claim(whiteness=_white(), holdout=_pass_holdout())
    assert assessment.claim_level in _CLAIM_LEVELS
