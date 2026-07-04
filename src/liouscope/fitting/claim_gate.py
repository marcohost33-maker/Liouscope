"""Wire the anti-overfit gates into the claim vocabulary (LIOU-A-014).

:mod:`liouscope.fitting.whiteness` (Ljung-Box residual whiteness) and
:mod:`liouscope.fitting.holdout` (temporal out-of-sample generalisation) each
produce a boolean verdict, but nothing mapped those verdicts onto the
SAFE / REVIEW / BLOCK claim vocabulary that
:func:`liouscope.io.stability_report.build_stability_report` consumes -- the
``claim_level`` was entirely caller-supplied (issue #71 B2). This module closes
that gap: :func:`assess_relaxation_claim` turns the two gate outcomes into a
single, *fail-closed* ``claim_level`` recommendation a caller can hand straight
to ``build_stability_report(claim_level=...)``.

**Fail-closed policy (worst-wins).** A claim is only ``SAFE`` when *both* gates
ran *and* passed. Missing evidence never earns ``SAFE``:

* holdout rejected (out-of-sample error blows up)  -> ``BLOCK``
  (unmodelled structure / overfit -- the "false M3b oscillation" failure mode).
* residuals not white (leftover serial correlation) -> ``REVIEW``
  (under-specified model; the point estimate may stand but wants a human look).
* a gate was not run                                -> ``REVIEW``
  (we cannot assert generalisation we never tested).

This mirrors the fail-closed stance the rest of the library takes (e.g. the
``sparse_arpack`` solver path raising rather than silently running dense, and
the CPTP input guard). It changes no existing default: ``diagnose()`` still
emits its report unchanged; callers *opt in* by running the gates and feeding
the result here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .holdout import HoldoutResult
from .whiteness import WhitenessResult

ClaimLevel = Literal["SAFE", "REVIEW", "BLOCK"]

# Severity order used by the worst-wins reducer; higher = more restrictive.
_ORDER: dict[ClaimLevel, int] = {"SAFE": 0, "REVIEW": 1, "BLOCK": 2}


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """The claim-level recommendation derived from the anti-overfit gates.

    Attributes
    ----------
    claim_level
        ``SAFE`` / ``REVIEW`` / ``BLOCK`` -- the worst (most restrictive)
        verdict implied by the supplied gates.
    reasons
        Human-readable justifications, one per gate contribution, in the order
        the gates were evaluated (holdout, then whiteness). Empty only for an
        all-clear ``SAFE``.
    """

    claim_level: ClaimLevel
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _worst(a: ClaimLevel, b: ClaimLevel) -> ClaimLevel:
    """Return the more restrictive of two claim levels."""
    return a if _ORDER[a] >= _ORDER[b] else b


def assess_relaxation_claim(
    *,
    whiteness: WhitenessResult | None = None,
    holdout: HoldoutResult | None = None,
) -> ClaimAssessment:
    """Combine the whiteness and holdout gates into a claim-level recommendation.

    Parameters
    ----------
    whiteness
        Outcome of :func:`liouscope.fitting.whiteness.whiteness_gate` /
        :func:`~liouscope.fitting.whiteness.ljung_box_q`, or ``None`` if the
        gate was not run.
    holdout
        Outcome of :func:`liouscope.fitting.holdout.holdout_validate`, or
        ``None`` if the gate was not run.

    Returns
    -------
    ClaimAssessment
        The fail-closed ``claim_level`` and the reasons behind it.

    Raises
    ------
    ValueError
        If *both* gates are ``None`` -- there is no evidence to assess, and
        silently returning a verdict would defeat the fail-closed contract.
    """
    if whiteness is None and holdout is None:
        raise ValueError(
            "assess_relaxation_claim needs at least one gate result; both "
            "whiteness and holdout were None (nothing to assess)."
        )

    level: ClaimLevel = "SAFE"
    reasons: list[str] = []

    if holdout is None:
        level = _worst(level, "REVIEW")
        reasons.append("holdout gate not run (out-of-sample generalisation untested)")
    elif not holdout.accept:
        level = _worst(level, "BLOCK")
        reasons.append(
            "holdout rejected: out-of-sample error blew up "
            f"(ratio={holdout.ratio:.3g}, holdout_rmse={holdout.holdout_rmse:.3g} "
            f"vs train_rmse={holdout.train_rmse:.3g})"
        )

    if whiteness is None:
        level = _worst(level, "REVIEW")
        reasons.append("whiteness gate not run (residual autocorrelation untested)")
    elif not whiteness.is_white:
        level = _worst(level, "REVIEW")
        reasons.append(
            "residuals not white: leftover serial correlation "
            f"(Q={whiteness.q_stat:.3g}, dof={whiteness.dof}, p={whiteness.p_value:.3g})"
        )

    return ClaimAssessment(claim_level=level, reasons=tuple(reasons))


__all__ = ["ClaimAssessment", "ClaimLevel", "assess_relaxation_claim"]
