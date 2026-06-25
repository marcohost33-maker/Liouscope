"""Temporal holdout / train-test split for the M0-M3b fit hierarchy (LIOU-A-012).

An anti-overfit gate that complements the existing GLS/AR1/AICc/BCa stack with a
direct *out-of-sample* generalisation test. A time-ordered tail of the
relaxation trajectory is withheld from the fit and used only to score the
fitted model. A flexible model (e.g. a spurious damped oscillation M3b) that
wins in-sample by bending to noise but fails to predict the held-out tail is
flagged -- the exact failure mode the entry targets ("false M3b oscillation").

The split is **temporal, never shuffled**: relaxation residuals are
autocorrelated, so random k-fold would leak information across the AR(1)
structure and defeat the purpose. The holdout is the contiguous final segment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .gls import fit_gls_ar1

ModelFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """Outcome of a temporal holdout validation.

    Attributes
    ----------
    train_rmse
        Root-mean-square fit residual on the training segment.
    holdout_rmse
        Root-mean-square prediction error on the withheld tail.
    ratio
        ``holdout_rmse / train_rmse`` (``inf`` if ``train_rmse == 0``).
    accept
        ``True`` iff ``holdout_rmse <= train_rmse * (1 + delta)`` -- the model
        generalises. ``False`` signals overfitting / unmodelled structure and
        should trigger a fallback to a simpler nested model (BLOCK).
    n_train, n_holdout
        Segment sizes.
    """

    train_rmse: float
    holdout_rmse: float
    ratio: float
    accept: bool
    n_train: int
    n_holdout: int


def train_holdout_split(
    t: np.ndarray,
    y: np.ndarray,
    *,
    holdout_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split ``(t, y)`` temporally into ``(t_train, y_train, t_hold, y_hold)``.

    The final ``holdout_frac`` of the time-ordered samples is the holdout.

    Raises
    ------
    ValueError
        If ``holdout_frac`` is not in the open interval ``(0, 1)``, if ``t`` and
        ``y`` differ in length, if the split would leave fewer than two points in
        either segment (a one-point segment cannot constrain an exponential fit
        nor yield a meaningful RMSE), or if ``t`` is not strictly increasing.
        A strictly-increasing time grid is required: the split assumes ``t`` is
        time-ordered, so an unordered ``t`` would put arbitrary (not the latest)
        samples into the holdout and leak future information into training.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.shape != y.shape:
        raise ValueError(f"t and y must have the same shape, got {t.shape} vs {y.shape}")
    if t.ndim != 1:
        raise ValueError(f"t and y must be 1-D, got ndim {t.ndim}")
    if not np.all(np.isfinite(t)):
        raise ValueError("t must be finite (no NaN/inf) to define a temporal order")
    if t.size >= 2 and not np.all(np.diff(t) > 0.0):
        raise ValueError(
            "t must be strictly increasing (time-ordered) so the holdout is the "
            "latest segment; got a non-monotone grid, which would leak future "
            "samples into training."
        )
    if not np.isfinite(holdout_frac) or not (0.0 < holdout_frac < 1.0):
        raise ValueError(
            f"holdout_frac must lie in the open interval (0, 1), got {holdout_frac}"
        )
    n = t.size
    n_hold = int(round(n * holdout_frac))
    n_hold = max(2, min(n_hold, n - 2))
    n_train = n - n_hold
    if n_train < 2 or n_hold < 2:
        raise ValueError(
            f"holdout split needs >=2 points per segment; got n={n}, "
            f"n_train={n_train}, n_holdout={n_hold}. Provide more samples or a "
            "different holdout_frac."
        )
    return t[:n_train], y[:n_train], t[n_train:], y[n_train:]


def holdout_validate(
    model: ModelFunc,
    t: np.ndarray,
    y: np.ndarray,
    p0: np.ndarray,
    *,
    holdout_frac: float = 0.25,
    delta: float = 0.5,
) -> HoldoutResult:
    """Fit ``model`` on the training segment and score it on the held-out tail.

    Parameters
    ----------
    model
        Callable ``f(t, theta) -> y_hat`` from :mod:`liouscope.fitting.models`.
    t, y
        Full time grid and trajectory.
    p0
        Initial parameter guess for the fit.
    holdout_frac
        Fraction of the time-ordered tail reserved for validation.
    delta
        Slack on the acceptance gate. The model is accepted iff
        ``holdout_rmse <= train_rmse * (1 + delta)``.

    Returns
    -------
    HoldoutResult
    """
    t_tr, y_tr, t_ho, y_ho = train_holdout_split(t, y, holdout_frac=holdout_frac)
    fit = fit_gls_ar1(model, t_tr, y_tr, np.asarray(p0, dtype=float))
    train_resid = y_tr - model(t_tr, fit.params)
    hold_resid = y_ho - model(t_ho, fit.params)
    train_rmse = float(np.sqrt(np.mean(train_resid**2)))
    holdout_rmse = float(np.sqrt(np.mean(hold_resid**2)))
    ratio = holdout_rmse / train_rmse if train_rmse > 0.0 else float("inf")
    accept = holdout_rmse <= train_rmse * (1.0 + delta)
    return HoldoutResult(
        train_rmse=train_rmse,
        holdout_rmse=holdout_rmse,
        ratio=ratio,
        accept=accept,
        n_train=t_tr.size,
        n_holdout=t_ho.size,
    )


__all__ = ["HoldoutResult", "holdout_validate", "train_holdout_split"]
