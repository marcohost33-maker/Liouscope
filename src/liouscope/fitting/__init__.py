"""Fitting pipeline (Spec Teil 11): GLS+AR(1), N_eff, AICc, Bootstrap.

Also exposes the two anti-overfit acceptance gates -- residual whiteness
(Ljung-Box) and temporal holdout -- together with
:func:`assess_relaxation_claim`, which maps their verdicts onto the
SAFE/REVIEW/BLOCK claim vocabulary used by the StabilityReport contract.
"""

from .aicc import aicc, choose_model
from .bootstrap import bca_ci, parametric_bootstrap
from .claim_gate import ClaimAssessment, ClaimLevel, assess_relaxation_claim
from .gls import fit_gls_ar1
from .holdout import HoldoutResult, holdout_validate, train_holdout_split
from .models import (
    M0,
    M1,
    M2,
    M3a,
    M3b,
    initial_guess_m0,
    initial_guess_m1,
    initial_guess_m2,
    initial_guess_m3a,
    initial_guess_m3b,
)
from .neff import estimate_neff_geyer
from .prony import prony_seed
from .whiteness import WhitenessResult, ljung_box_q, whiteness_gate

__all__ = [
    "M0",
    "M1",
    "M2",
    "ClaimAssessment",
    "ClaimLevel",
    "HoldoutResult",
    "M3a",
    "M3b",
    "WhitenessResult",
    "aicc",
    "assess_relaxation_claim",
    "bca_ci",
    "choose_model",
    "estimate_neff_geyer",
    "fit_gls_ar1",
    "holdout_validate",
    "initial_guess_m0",
    "initial_guess_m1",
    "initial_guess_m2",
    "initial_guess_m3a",
    "initial_guess_m3b",
    "ljung_box_q",
    "parametric_bootstrap",
    "prony_seed",
    "train_holdout_split",
    "whiteness_gate",
]
