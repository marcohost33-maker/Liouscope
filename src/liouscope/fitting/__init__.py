"""Fitting pipeline (Spec Teil 11): GLS+AR(1), N_eff, AICc, Bootstrap."""

from .aicc import aicc, choose_model
from .bootstrap import bca_ci, parametric_bootstrap
from .gls import fit_gls_ar1
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

__all__ = [
    "M0",
    "M1",
    "M2",
    "M3a",
    "M3b",
    "aicc",
    "bca_ci",
    "choose_model",
    "estimate_neff_geyer",
    "fit_gls_ar1",
    "initial_guess_m0",
    "initial_guess_m1",
    "initial_guess_m2",
    "initial_guess_m3a",
    "initial_guess_m3b",
    "parametric_bootstrap",
    "prony_seed",
]
