"""Regressions for the three findings of issue #118 that survived PR #107.

Thirteen of the sixteen review findings were closed in the PR itself; these are
the three that a re-run at head 4a8ae9c still reproduced. Each test pins the
BEHAVIOUR, not the implementation, and each is paired with a healthy-path guard
so that a future change cannot satisfy it by simply refusing everything.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.core.lindblad import steady_state
from liouscope.diagnostics.spectral import compute_spectral_layer
from liouscope.fitting.gls import fit_gls_ar1
from liouscope.fitting.models import M0

# --- finding 9: a fit that ENDS on the magnitude plateau is not a success ----


def _long_grid() -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0e10, 64)
    return t, np.exp(-5.0 * t / 1.0e10)


def test_saturated_optimum_is_not_reported_as_success() -> None:
    """``p0 = [1, -1]`` puts M0 into the model cap, where every derivative is
    exactly zero, so ``least_squares`` terminates on "gradient is small" without
    having moved. The residual norm at that point is ~7.9e100."""
    t, y = _long_grid()
    out = fit_gls_ar1(M0, t, y, np.array([1.0, -1.0]))

    assert out.success is False
    assert out.saturated  # the reason is reported, not just the refusal
    assert float(np.linalg.norm(out.residuals)) > 1.0e50


def test_well_posed_fit_on_the_same_grid_still_succeeds() -> None:
    """Positive control: the guard must key on saturation, not on the grid."""
    t, y = _long_grid()
    out = fit_gls_ar1(M0, t, y, np.array([1.0, 5.0e-10]))

    assert out.success is True
    assert out.saturated == ()
    assert out.params[1] == pytest.approx(5.0e-10, rel=1.0e-6)


# --- finding 15: a sub-split decay mode is not a second zero mode -----------


def _subsplit_generator() -> np.ndarray:
    """Two-level system whose SLOWEST decay (1e-15) is far below the scale the
    Hamiltonian sets (omega = 1). The true spectrum is
    ``{0, -1e-15, -(0.5e-15 + 2e-14) +- 1i}``, so D1 = 1e-15."""
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    return build_liouvillian(H, [lowering, sigma_z], [1.0e-15, 1.0e-14])


def test_slow_mode_below_the_norm_scaled_band_survives_as_the_gap() -> None:
    """The zero-mode band is ``1e3 * eps * ||L||`` and ``||L||`` is set by the
    Hamiltonian, so the genuine 1e-15 mode falls inside it. Its a posteriori
    backward-error bound certifies it as non-stationary regardless."""
    L = _subsplit_generator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = compute_spectral_layer(L, steady_state(L, allow_degenerate=True))

    assert layer.gap == pytest.approx(1.0e-15, rel=1.0e-6)
    cert = layer.zero_mode_certificate
    assert cert["zero_mode_count"] == 1
    assert cert["resolved"] is True
    # The tolerance the gap filters applied must separate the two populations.
    assert 0.0 < cert["zero_tolerance"] < 1.0e-15


def test_healthy_generator_keeps_its_untouched_certificate() -> None:
    """Positive control: with no mode to rescue the refinement is inert and the
    applied tolerance is still the raw band."""
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(H, [lowering], [0.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        layer = compute_spectral_layer(L, steady_state(L, allow_degenerate=True))

    cert = layer.zero_mode_certificate
    assert cert["zero_mode_count"] == 1
    assert cert["zero_tolerance"] == pytest.approx(cert["bound"])
    assert layer.gap == pytest.approx(0.25, rel=1.0e-9)


# --- finding 16: the changelog must not call a load-bearing field report-only


def test_changelog_does_not_claim_the_certificate_is_report_only() -> None:
    """``classify`` reads the certificate and caps verdict/tier through
    ``_apply_spectral_certificate_floor``; D1/D3/D4 additionally filter by its
    ``zero_tolerance``. Describing it as report-only was false in both paths."""
    root = pathlib.Path(__file__).resolve().parents[1]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    source = (
        root / "src" / "liouscope" / "diagnostics" / "classification.py"
    ).read_text(encoding="utf-8")

    reads_certificate = bool(re.search(r"certificate = getattr\(spectral", source))
    assert reads_certificate, "guard is vacuous if the classifier stopped reading it"
    assert "report-only, additive field" not in changelog
