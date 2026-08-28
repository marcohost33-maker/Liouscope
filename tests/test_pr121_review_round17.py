"""Round-17 review of PR #121: one zero-mode cutoff, obtained one way.

The a posteriori refinement (issue #113 second axis) can rescue a genuine slow
mode from the raw operator band. Four consumer layers kept filtering with
``ZeroModeCertificate.bound`` afterwards and threw that mode away again. Each
test below pins ONE of those sites, and each is paired with a positive control
so it cannot pass by refusing everything.

The guard at the end is the structural half: it fails when any NEW site starts
reading the raw band, which is what made a four-fold miss possible in the first
place.
"""

from __future__ import annotations

import pathlib
import re
import warnings

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope._zhou import compute_zhou_predictor
from liouscope.diagnostics import nonnormality as nn
from liouscope.diagnostics.mpemba import expansion_alpha, overlap_c1
from liouscope.numerics.linalg import certified_eigvals

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _refined_generator() -> np.ndarray:
    """Two-level system whose slowest decay (1e-15) sits INSIDE the raw band.

    ``||L||`` is set by the Hamiltonian (omega = 1), so the operator band is
    ``2.2e-13`` while the genuine slow mode is at ``1e-15``. The a posteriori
    certificate proves that mode non-stationary, and the applied tolerance
    drops to ``5e-16`` -- 444x below the raw band.
    """
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    return build_liouvillian(H, [lowering, sigma_z], [1.0e-15, 1.0e-14])


_D11_PAIRS = [(2, 0), (0, 3), (3, 1), (1, 3), (0, 2), (1, 0), (2, 1)]
_D11_RATES = [4.452e-05, 1.005e08, 2.452e-06, 6.823e-06, 4.239e-06, 4.307e-05, 1.0e-11]


def _d11_fallback_generator() -> np.ndarray:
    """Stiff classical network: eigenvalues resolve WITH a refinement, vectors do not.

    That combination is what puts D11 on its own fallback path: the D9
    eigenvector gate withholds the Petermann factors, so the progression scan
    is recomputed from the certified spectrum -- and it is exactly there that
    the raw band discarded the rescued modes again.
    """
    jumps = []
    for to, frm in _D11_PAIRS:
        j = np.zeros((4, 4), dtype=complex)
        j[to, frm] = 1.0
        jumps.append(j)
    return build_liouvillian(np.zeros((4, 4), dtype=complex), jumps, _D11_RATES)


def _rescued_modes(L: np.ndarray) -> tuple[np.ndarray, object]:
    """``(|lambda| the refinement rescued, certificate)`` -- the positive control."""
    ev, cert = certified_eigvals(L)
    mag = np.abs(ev)
    return np.sort(mag[(mag > cert.applied_tolerance) & (mag <= cert.bound)]), cert


# --------------------------------------------------------------------------
# src/liouscope/_zhou.py -- D24 mixing-time window
# --------------------------------------------------------------------------


def test_d24_reads_the_refined_gap_not_the_raw_band() -> None:
    L = _refined_generator()
    rescued, cert = _rescued_modes(L)
    assert cert.resolved, "precondition: the certificate must resolve"
    assert cert.applied_tolerance < cert.bound, "precondition: a refinement happened"
    assert rescued.size == 1, "precondition: exactly one mode was rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_zhou_predictor(L)

    # Filtering by ``bound`` discards the 1e-15 mode and D24 reads the next,
    # faster one at 2.05e-14 -- a mixing-time window ~20x too short.
    assert result.gap == pytest.approx(1.0e-15, rel=1.0e-6)
    assert result.mixing_time_lower > 1.0e15


def test_d24_is_unchanged_on_a_generator_with_nothing_to_rescue() -> None:
    """Positive control: the refinement is inert on a healthy generator."""
    H = np.diag([0.0, 1.0]).astype(complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(H, [lowering], [0.5])
    _rescued, cert = _rescued_modes(L)
    assert cert.applied_tolerance == pytest.approx(cert.bound)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compute_zhou_predictor(L)
    assert result.gap == pytest.approx(0.25, rel=1.0e-9)


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/mpemba.py -- D19 overlap and the mode expansion
# --------------------------------------------------------------------------

_RHO_0 = np.array([[0.9, 0.0], [0.0, 0.1]], dtype=complex)


def test_d19_overlap_survives_the_refinement() -> None:
    """``overlap_c1 == 0.0`` is the FALSE A11/F4 trigger, on the top rung."""
    L = _refined_generator()
    rescued, _cert = _rescued_modes(L)
    assert rescued.size == 1, "precondition: exactly one mode was rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = overlap_c1(L, _RHO_0)

    assert c1 > 0.0
    assert c1 == pytest.approx(0.14142135623730953, rel=1.0e-9)


def test_expansion_alpha_uses_the_refined_zero_set() -> None:
    L = _refined_generator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha = expansion_alpha(L, _RHO_0)

    # With the raw band the rescued mode is gone, the expansion is fitted on
    # the remaining fast pair and the slope collapses to ~-1e-15.
    assert alpha < -1.0
    assert alpha == pytest.approx(-33.560770643553646, rel=1.0e-6)


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/nonnormality.py -- D9 Petermann factors
# --------------------------------------------------------------------------


def test_petermann_keeps_the_refined_slow_mode() -> None:
    L = _refined_generator()
    rescued, _cert = _rescued_modes(L)
    assert rescued.size == 1, "precondition: exactly one mode was rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eigvals_filt, factors = nn.petermann_factors(L)

    # The raw band leaves 2 modes with petermann_max 1.0; the rescued mode is
    # the one whose conditioning can dominate the F1 input.
    assert eigvals_filt.size == 3
    assert float(np.min(np.abs(eigvals_filt))) == pytest.approx(1.0e-15, rel=1.0e-6)
    assert float(np.max(factors)) == pytest.approx(2.0, rel=1.0e-9)


# --------------------------------------------------------------------------
# src/liouscope/diagnostics/nonnormality.py -- D11 fallback (D9 withheld)
# --------------------------------------------------------------------------


def test_d11_fallback_scans_the_refined_zero_set(monkeypatch) -> None:
    """The fallback must hand the progression scan the REFINED complement.

    Asserted at the seam the review names: what reaches
    ``bohr_arithmetic_progression``. The measured progression length happens
    to be 1 for this (Hamiltonian-free) network either way, so an assertion on
    the reported number alone would be blind here.
    """
    L = _d11_fallback_generator()
    rescued, cert = _rescued_modes(L)
    assert cert.resolved, "precondition: the eigenvalue certificate resolves"
    assert cert.applied_tolerance < cert.bound, "precondition: a refinement happened"
    assert rescued.size == 3, "precondition: three modes were rescued"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ev, factors = nn.petermann_factors(L)
    assert bool(np.all(np.isnan(factors))), "precondition: D9 is withheld here"

    seen: list[np.ndarray] = []
    original = nn.bohr_arithmetic_progression

    def _spy(eigvals: np.ndarray, d: int) -> tuple[float, float]:
        seen.append(np.asarray(eigvals).copy())
        return original(eigvals, d)

    monkeypatch.setattr(nn, "bohr_arithmetic_progression", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nn.compute_nonnormality_layer(L)

    assert len(seen) == 1
    scanned = np.abs(seen[0])
    ev_all, _c = certified_eigvals(L)
    expected = int(np.count_nonzero(np.abs(ev_all) > cert.applied_tolerance))
    assert scanned.size == expected
    # Every rescued mode is still in the scan; the raw band would drop all three.
    for mode in rescued:
        assert np.any(np.isclose(scanned, mode, rtol=1.0e-9))


# --------------------------------------------------------------------------
# Structural guard: no NEW site may filter on the raw band
# --------------------------------------------------------------------------

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "liouscope"
_BOUND = re.compile(r"\b(?:cert|certificate|self)\.bound\b")

#: The ONLY places allowed to read the raw band: three warning texts that
#: label it "bound", and the two accessors inside the certificate itself.
_ALLOWED: frozenset[tuple[str, str]] = frozenset({
    (
        "diagnostics/mpemba.py",
        'f"{certificate.residual:.3e}, bound {certificate.bound:.3e}). "',
    ),
    (
        "diagnostics/nonnormality.py",
        'f"{certificate.residual:.3e}, bound {certificate.bound:.3e}); "',
    ),
    (
        "diagnostics/spectral.py",
        'f"{certificate.bound:.3e}). The generator is too stiff for a dense "',
    ),
    (
        "numerics/linalg.py",
        "return self.bound if not np.isfinite(self.zero_tolerance) "
        "else self.zero_tolerance",
    ),
    ("numerics/linalg.py", '"bound": _f(self.bound),'),
})


def _bound_readers() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _BOUND.search(stripped):
                found.add((rel, stripped))
    return found


def test_only_report_sites_read_the_raw_certificate_bound() -> None:
    """A fifth filter site must fail here, not in a review six rounds later."""
    found = _bound_readers()
    # Positive control: a guard that finds nothing proves nothing.
    assert len(found) >= len(_ALLOWED) > 0
    assert found >= _ALLOWED, (
        "the allowlist has rotted -- these entries no longer exist: "
        f"{sorted(_ALLOWED - found)}"
    )
    assert found <= _ALLOWED, (
        "a new site reads the RAW certificate band; filters must use "
        f"ZeroModeCertificate.zero_set_tolerance(): {sorted(found - _ALLOWED)}"
    )


def test_the_certificate_exposes_one_way_to_get_a_filter_cutoff() -> None:
    """The applicable/inapplicable branch and the refinement live in one place."""
    L = _refined_generator()
    ev, cert = certified_eigvals(L)
    assert cert.zero_set_tolerance(ev) == pytest.approx(cert.applied_tolerance)
    assert cert.zero_set_tolerance(ev, atol=1.0e-9) == pytest.approx(1.0e-9)
    assert cert.zero_set_tolerance(ev) < cert.bound
