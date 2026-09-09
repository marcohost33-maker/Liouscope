"""D13 resolution contract: an unresolved sweep must not look like a measurement.

Background (cross-family review 2026-09-02, ported 2026-09-09)
-------------------------------------------------------------
``pseudospectral_radius`` initialised its accumulator to ``0.0`` and returned
it when no grid node satisfied ``sigma_min(zI - L) <= eps``. ``0.0`` is a
*plausible* pseudospectral radius ("the eps-pseudospectrum sits at the
origin"), so an unresolved sweep was indistinguishable from a measured one --
a guard producing a WRONG result rather than none.

It reached a decision. The value feeds ``pseudospectral_radius_diag`` ->
``ResolventResult`` -> the F5 phantom-relaxation rung, whose reach leg tests
``pseudospectral_radius / gap > 2 * gap_to_gns_ratio``; with a returned ``0.0``
that comparison is False, so an unmeasured D13 silently reported "no
pseudospectral intrusion" -- fail-open in a classifier.

The sibling ``pseudospectrum_extent`` was built fail-visible for issue #101, so
two implementations of the SAME quantity returned ``0.0`` and ``nan`` on
byte-identical input. These tests pin the resolved disagreement, and the last
group pins what the change does (and does not do) to the decision ladder.

The oracle is ANALYTIC, not a second implementation: for a NORMAL operator the
eps-pseudospectrum is exactly the union of closed eps-discs around the spectrum
(Trefethen & Embree, *Spectra and Pseudospectra*, Thm. 2.2), because
``sigma_min(zI - A) = dist(z, spec(A))`` there. The true pseudospectral radius
is therefore ``max|lambda| + eps``.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.diagnostics.classification import (
    _hypothesis_ladder,
    hypothesis_evidence_matrix,
)
from liouscope.numerics.pseudospec import (
    pseudospectral_radius,
    pseudospectrum_extent,
    pseudospectrum_sigma_floor,
)

_SPECTRUM = np.array([0.0, -0.2, -1.0 - 3.0j, -1.0 + 3.0j, -4.0], dtype=complex)


def _normal_matrix() -> np.ndarray:
    """Unitary conjugation of ``diag(_SPECTRUM)`` -- normal by construction."""
    rng = np.random.default_rng(20260902)
    seed = rng.standard_normal((5, 5)) + 1j * rng.standard_normal((5, 5))
    q, _ = np.linalg.qr(seed)
    a = q @ np.diag(_SPECTRUM) @ q.conj().T
    assert np.allclose(a @ a.conj().T, a.conj().T @ a), "construction is not normal"
    return a


# Deliberately misaligned with the spectrum: no node lands on an eigenvalue,
# which is the ONLY way a rectangular sweep sees a 1e-3 disc.
_FINE_GRID = {"grid_re": (-6.0, 1.0, 101), "grid_im": (-5.0, 5.0, 101)}


def test_unresolved_sweep_returns_nan_not_zero() -> None:
    """The load-bearing regression. Pre-fix this returned ``0.0``."""
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        radius = pseudospectral_radius(a, 1.0e-3, **_FINE_GRID)
    assert np.isnan(radius)


@pytest.mark.parametrize("n", [25, 51, 101, 201])
def test_refining_the_grid_does_not_rescue_a_too_small_eps(n: int) -> None:
    """Documents WHY ``0.0`` was wrong rather than merely imprecise.

    At ``eps = 1e-3`` the pseudospectrum of this normal operator is five discs
    of radius 1e-3. A rectangular sweep resolves it only when a node lands
    INSIDE one, which refinement does not achieve on a grid whose nodes are
    misaligned with the spectrum -- so the pre-fix code returned a confident
    ``0.0`` at every resolution up to 201x201 (40'401 SVDs). The old value
    measured grid geometry, not the operator.
    """
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        radius = pseudospectral_radius(
            a, 1.0e-3, grid_re=(-6.0, 1.0, n), grid_im=(-5.0, 5.0, n)
        )
    assert np.isnan(radius)


def test_the_two_estimators_agree_on_unresolved_input() -> None:
    """``pseudospectral_radius`` and ``pseudospectrum_extent`` used to disagree.

    Same operator, same grid, same eps: ``0.0`` from one, ``nan`` from the
    other. Two implementations of one quantity contradicting each other is how
    the defect stayed invisible; they now share a single sweep.
    """
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning, match="unresolved"):
        legacy = pseudospectral_radius(a, 1.0e-3, **_FINE_GRID)
    extent_radius, extent_abscissa = pseudospectrum_extent(a, 1.0e-3, **_FINE_GRID)
    assert np.isnan(legacy)
    assert np.isnan(extent_radius)
    assert np.isnan(extent_abscissa)


def test_resolved_sweep_matches_the_analytic_oracle() -> None:
    """Positive control: with an eps the grid CAN resolve, the value is right.

    Without this, the fix would be satisfied by a function returning ``nan``
    unconditionally -- "the guard fired" and "the guard always fires" would be
    indistinguishable.
    """
    a = _normal_matrix()
    eps = 0.5  # comfortably above this grid's sigma_min floor (1e-2)
    radius = pseudospectral_radius(a, eps, **_FINE_GRID)
    true_radius = float(np.max(np.abs(_SPECTRUM))) + eps  # = 4.5, analytic
    assert np.isfinite(radius)
    # A grid sweep is a LOWER bound on the true supremum (finite sampling, no
    # globality certificate), so it must not exceed the oracle, and at this
    # resolution it must come within about one grid step of it.
    assert radius <= true_radius + 1.0e-12
    assert radius >= true_radius - 0.15


def test_sigma_floor_reports_the_smallest_eps_that_would_resolve() -> None:
    """The companion that turns "found nothing" into an actionable number."""
    a = _normal_matrix()
    floor = pseudospectrum_sigma_floor(a, **_FINE_GRID)
    assert np.isfinite(floor)
    assert floor > 1.0e-3  # which is exactly why eps=1e-3 resolved nothing
    # Just above the floor the sweep resolves; just below it cannot.
    assert np.isfinite(pseudospectral_radius(a, floor * 1.01, **_FINE_GRID))
    with pytest.warns(RuntimeWarning, match="unresolved"):
        assert np.isnan(pseudospectral_radius(a, floor * 0.99, **_FINE_GRID))


def test_the_warning_names_the_achievable_eps() -> None:
    """A fail-visible message must say what to do, not merely that it failed."""
    a = _normal_matrix()
    with pytest.warns(RuntimeWarning) as record:
        pseudospectral_radius(a, 1.0e-6, **_FINE_GRID)
    message = str(record[0].message)
    assert "sigma_min" in message
    assert "nan" in message
    # The ACTIONABLE half, asserted as a whole phrase. A retraction probe on
    # 2026-09-09 showed why: asserting only that the number appears SOMEWHERE
    # in the message left this test blind -- the diagnostic half ("smallest
    # sigma_min on this grid was ...") already contains it, so the instruction
    # half could be deleted and the test stayed green. A guard that accepts
    # "something failed" in place of "do this" is not the guard we wrote.
    floor = pseudospectrum_sigma_floor(a, **_FINE_GRID)
    assert f"raise eps to at least {floor:.6e}" in message


def test_the_default_grid_still_resolves_a_liouvillian_like_spectrum() -> None:
    """Positive control on the DEFAULT path -- the one production uses.

    The default grid brackets the spectrum with half-span padding and 25 nodes,
    which places nodes exactly on ``re_min``/``re_max`` (indices 6 and 18),
    i.e. on the extremal eigenvalues where ``sigma_min`` is 0. That is why the
    defect never showed in ordinary runs; this test pins that the fix does not
    break the accidental-but-real default resolution.
    """
    a = _normal_matrix()
    radius = pseudospectral_radius(a, 1.0e-3)
    assert np.isfinite(radius)
    # max|lambda| = 4; a node sits ON the extremal eigenvalue -4, so the grid
    # lower bound reaches at least that modulus.
    assert radius >= 4.0 - 1.0e-9


# ---------------------------------------------------------------------------
# What the NaN does to the decision ladder (the reason the fix is safe -- and
# the one place where it is NOT a no-op).
# ---------------------------------------------------------------------------


class _Relaxation:
    def __init__(self, aicc_model: str = "M0", beta_D: float = 1.0) -> None:
        self.aicc_model = aicc_model
        self.beta_D = beta_D


def _ev(**over: float) -> dict[str, float]:
    base = {
        "mpemba_is_candidate": 0.0, "gap": 1.0, "pseudospectral_radius": 0.1,
        "gap_to_gns_ratio": 1.0, "henrici_eta": 0.1, "kreiss": 1.0,
        "petermann_max": 1.0, "trans_amplitude_ratio": 1.0, "kappa_trans": 1.0,
        "gns_certified": 0.0, "gap_rate_consistency": 1.0,
        "d17_linear_single_exp": 0.0, "has_complex_pairs": 0.0,
    }
    base.update(over)
    return base


@pytest.mark.parametrize("henrici", [0.1, 2.0])
def test_nan_d13_does_not_make_any_rung_fire_that_zero_did_not(henrici: float) -> None:
    """With a MEASURED gap the swap 0.0 -> nan changes no rung outcome.

    ``nan > x`` is False exactly as ``0.0 > x`` was, and a NaN required key is
    stripped, which makes the rung unevaluable rather than firing. So no
    verdict can be manufactured by the change; it can only withhold.
    """
    rel = _Relaxation()
    zero = _hypothesis_ladder(_ev(henrici_eta=henrici), relaxation=rel)
    nan = _hypothesis_ladder(
        _ev(henrici_eta=henrici, pseudospectral_radius=float("nan")), relaxation=rel
    )
    assert [r[3] for r in zero] == [r[3] for r in nan]


def test_nan_d13_is_reported_as_unevaluable_not_as_a_measured_negative() -> None:
    """The audit trail the old ``0.0`` erased.

    The rung does not fire either way -- but the evidence matrix now
    distinguishes "measured, below threshold" from "never measured", which is
    the entire point of finding A.
    """
    rel = _Relaxation()

    def _f5(psr: float) -> dict[str, object]:
        matrix = hypothesis_evidence_matrix(
            _ev(henrici_eta=2.0, pseudospectral_radius=psr), relaxation=rel
        )
        return next(e for e in matrix if e["rule_id"] == "F5_PSEUDOSPECTRAL")

    assert _f5(0.0)["status"] == "NOT_SUPPORTED"
    unresolved = _f5(float("nan"))
    assert unresolved["status"] == "UNEVALUABLE"
    assert "pseudospectral_radius" in unresolved["missing"]


def test_an_unresolved_d13_withholds_the_gapless_f5_limit() -> None:
    """The ONE outcome the swap does change -- fail-closed, and on purpose.

    ``gap == 0`` is the gapless/critical limit, where the reach leg returns
    True without reading the radius at all. Pre-fix, an UNRESOLVED sweep still
    supplied ``pseudospectral_radius = 0.0``, the required key was present, and
    F5 fired: an A10/F5 mechanism claim resting on a D13 that was never
    measured. With ``nan`` the required key is absent, the rung is unevaluable
    and the claim is withheld.

    This is a real behaviour change, not a no-op, and it is the direction the
    round-22 REQUIRED-key rule already chose for ``gap``: "no measurement" must
    not be laundered into evidence.
    """
    rel = _Relaxation()

    def _fired(psr: float) -> list[str]:
        ev = _ev(gap=0.0, henrici_eta=2.0, pseudospectral_radius=psr)
        return [r[0] for r in _hypothesis_ladder(ev, relaxation=rel) if r[3]]

    assert "F5_PSEUDOSPECTRAL" in _fired(0.0)
    assert "F5_PSEUDOSPECTRAL" not in _fired(float("nan"))
