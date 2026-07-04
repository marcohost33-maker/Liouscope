"""Frozen result dataclasses for the LiouScope public API.

All result containers are frozen dataclasses with keyword-only construction.
The top-level :class:`DiagnosticReport` is what `diagnose()` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ._consts import DIAGNOSTIC_SCHEMA_VERSION, TAXONOMY_VERSION

# issue #70 A5: "EXCLUDED" was removed. A single-pass maximum-evidence
# classifier reports the best-fit A-class with its support and never emits a
# genuine active-rejection verdict, so the value was permanently unreachable
# through ``diagnose()`` (advertised a capability the pipeline lacks). Active
# exclusion requires a per-hypothesis scoring mode (deferred). See CHANGELOG.
Verdict = Literal["CONFIRMED", "CANDIDATE", "NOT_EXCLUDED", "UNDEFINED"]
Tier = Literal["PUBLICATION_GRADE", "CONFIRMATION", "EXPLORATION"]
QualityLabel = Literal["stable", "moderate", "exploratory"]
SolverPath = Literal["dense", "sparse_arpack"]


@dataclass(frozen=True, slots=True, kw_only=True)
class SpectralResult:
    """Spectral layer S: D1, D2, D2b, D3, D4."""

    gap: float                  # D1 Delta
    gns_gap: float              # D2 Delta_s (Mori-Shirai 2023)
    kms_gap: float              # D2b (Fagnola 2025)
    oscillating_gap: float      # D3 min |Im(lambda)| over complex pairs
    spectral_spread: float      # D4 max|Re| minus min|Re| (non-zero modes)
    eigenvalues: np.ndarray     # full sigma(L) sorted by real part
    steady_state: np.ndarray    # rho_ss matrix, d x d
    has_complex_pairs: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class NonNormalityResult:
    """Non-normality layer N: D8, D9, D10, D11."""

    henrici_eta: float          # D8
    petermann_max: float        # D9 max K_j
    petermann_factors: np.ndarray
    kreiss: float               # D10
    bohr_ap_length: int         # D11 Bohr arithmetic-progression depth
    bohr_ap_pauli_bound: float


@dataclass(frozen=True, slots=True, kw_only=True)
class FitResult:
    """Single-model fit summary for an entry of the M-hierarchy."""

    model: str                  # "M0", "M1", "M2", "M3a", "M3b"
    params: np.ndarray
    log_likelihood: float
    aicc: float
    n_eff: float
    residual_ar1_rho: float
    success: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class RelaxationResult:
    """Relaxation layer R: D5-D7b plus M0..M3b winner."""

    von_neumann_entropy: float            # D5 final
    relative_entropy_curve: np.ndarray    # D6 trajectory
    fidelity_curve: np.ndarray            # D7 trajectory
    entanglement_asymmetry: float | None  # D7b
    fits: dict[str, FitResult]
    aicc_model: str                       # winning model
    beta_D: float                         # fitted exponential rate of best model
    bca_ci_beta: tuple[float, float]      # BCa 95% CI
    # F-018 (LIOU-F-018): half trace-norm distance to rho_ss along the
    # trajectory, the observable relaxation metric alongside D5/D6/D7. Optional
    # and defaulted so the field is purely additive (older callers/serialised
    # reports remain valid).
    trace_distance_curve: np.ndarray | None = None
    # LIOU-#69: dominant rate of the LINEAR trace-distance curve. Unlike beta_D
    # (fit on relative entropy, which carries a metric multiplier m in {1, 2}),
    # this rate decays at the bare mode rate and is dimension-coherent with the
    # spectral gap Delta -- it is the rate the D17 gap-rate consistency check
    # uses. Additive + defaulted so older callers / serialised reports stay valid.
    beta_D_linear: float = float("nan")
    linear_fit_model: str = "none"


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolventResult:
    """Resolvent layer: D11b, D12, D13."""

    resolvent_peak: float                 # D11b
    ridge_fwhm: float                     # D12
    pseudospectral_radius: float          # D13 eps-pseudospectrum radius
    pseudospec_eps: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TransientResult:
    """Transient layer: D14, D15."""

    trans_amplitude_ratio: float          # D14 sup_t ||e^{tL}|| / ||rho_0||
    kappa_trans: float                    # D15 omega(L) / Delta
    numerical_abscissa: float             # omega(L)


@dataclass(frozen=True, slots=True, kw_only=True)
class LepResult:
    """LEP layer: D16, D17, D18.

    Note: conjugate-pair LEP candidates are INCLUDED (FIX-3, anchor I).
    """

    lep_proximity: float                  # D16 min pair separation
    gap_rate_consistency: float           # D17 |beta_D_linear - Delta| / Delta
    initial_state_sensitivity: float      # D18 std over Haar ensemble
    lep_candidate_count: int
    # LIOU-#69: the LINEAR-metric rate actually fed to D17 (dimension-coherent
    # with the gap). Additive + defaulted so synthetic callers stay valid.
    beta_D_linear: float = float("nan")


@dataclass(frozen=True, slots=True, kw_only=True)
class MpembaResult:
    """Mpemba layer: D19, D20."""

    overlap_c1: float                     # D19 slowest-mode overlap
    is_mpemba_candidate: bool
    expansion_alpha: float                # D20 Phi_n scaling exponent
    trivial_overlap: bool = False         # symmetry-protected zero overlap (not Mpemba)


@dataclass(frozen=True, slots=True, kw_only=True)
class UncertaintyResult:
    """Uncertainty layer U: U0, U1, U2."""

    fit_uncertainty: float                # U0
    solver_uncertainty: float             # U1
    size_uncertainty: float | None        # U2 (None for single-N)
    bootstrap_B: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ClassificationResult:
    """Classification layer C: A-class plus F-family plus verdict.

    Anchor L: ``taxonomy_version`` and ``schema_version`` are stamped here.
    """

    a_class: str                          # "A1" .. "A12"
    f_family: str                         # "F1" .. "F5" or "none"
    verdict: Verdict
    tier: Tier
    confidence: float                     # 0..1
    evidence: dict[str, float]
    taxonomy_version: str = TAXONOMY_VERSION
    schema_version: str = DIAGNOSTIC_SCHEMA_VERSION


@dataclass(frozen=True, slots=True, kw_only=True)
class GovernanceMetadata:
    """Governance layer G: SHA-256 run-id and reproducibility info."""

    run_id: str
    framework_version: str
    python_version: str
    platform: str
    numpy_version: str
    scipy_version: str
    seed: int
    solver_path: SolverPath
    quality_label: QualityLabel
    timestamp: str
    input_hash: str


@dataclass(frozen=True, slots=True, kw_only=True)
class DiagnosticReport:
    """Top-level report produced by :func:`liouscope.diagnose`."""

    spectral: SpectralResult
    nonnorm: NonNormalityResult
    relaxation: RelaxationResult
    resolvent: ResolventResult
    transient: TransientResult
    lep: LepResult
    uncertainty: UncertaintyResult
    classification: ClassificationResult
    governance: GovernanceMetadata
    mpemba: MpembaResult | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True, kw_only=True)
class ZhouPredictorResult:
    """v0.2.1 post-submission Zhou universal mixing-time predictor (D24)."""

    mixing_time_lower: float
    mixing_time_upper: float
    epsilon: float
    converged: bool
    gap: float = float("nan")
    petermann_factor: float = float("nan")
