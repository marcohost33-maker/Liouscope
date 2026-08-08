"""Issue #102 — make branch shadowing visible without changing any verdict.

The classifier resolves by priority, so a system showing several mechanisms at
once reports only the first. These tests pin that the shadow report exists, is
derived from the SAME ladder as the decision, and changes nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope.diagnostics.classification import (
    _hypothesis_ladder,
    _pick_a_class,
    triggered_hypotheses,
)


class _Rel:
    """Minimal RelaxationResult stand-in for the two fields the ladder reads."""

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


def test_ladder_and_winner_come_from_the_same_source():
    """The report must not be a second copy of the conditions.

    Whatever the ladder says fires first IS what the classifier returns; that
    is the property which keeps the shadow report from drifting away from the
    decision it describes.
    """
    cases = [
        _ev(mpemba_is_candidate=1.0),
        _ev(henrici_eta=2.0, pseudospectral_radius=10.0),
        _ev(kreiss=10.0, petermann_max=10.0),
        _ev(trans_amplitude_ratio=10.0, kappa_trans=5.0),
        _ev(gap_to_gns_ratio=2.0, gns_certified=1.0),
        _ev(gap_rate_consistency=0.01, d17_linear_single_exp=1.0),
        _ev(),
    ]
    for ev in cases:
        rel = _Rel()
        fired = [r for r in _hypothesis_ladder(ev, relaxation=rel) if r[3]]
        expected = (fired[0][1], fired[0][2]) if fired else ("A12", "none")
        assert _pick_a_class(ev, relaxation=rel) == expected


def test_concurrent_mechanisms_are_all_reported():
    """Mpemba AND phantom AND skin at once: one wins, three are recorded."""
    ev = _ev(
        mpemba_is_candidate=1.0,          # F4, highest priority
        henrici_eta=2.0, pseudospectral_radius=10.0,   # F5
        trans_amplitude_ratio=10.0, kappa_trans=5.0,   # F2
    )
    rel = _Rel()
    assert _pick_a_class(ev, relaxation=rel) == ("A11", "F4")

    fired = triggered_hypotheses(ev, relaxation=rel)
    families = [h["f_family"] for h in fired]
    assert families[:3] == ["F4", "F5", "F2"]
    assert fired[0]["shadowed"] is False
    assert all(h["shadowed"] for h in fired[1:])


def test_single_mechanism_shadows_nothing():
    ev = _ev(kreiss=10.0, petermann_max=10.0)
    fired = triggered_hypotheses(ev, relaxation=_Rel())
    assert [h["rule_id"] for h in fired] == ["F1_OVERLAP_AMPLIFICATION"]
    assert fired[0]["shadowed"] is False


def test_no_mechanism_yields_an_empty_report():
    assert triggered_hypotheses(_ev(), relaxation=_Rel()) == ()


def test_shadow_report_does_not_change_the_decision():
    """Metamorphic: the report is derived, never consulted."""
    ev = _ev(mpemba_is_candidate=1.0, henrici_eta=2.0, pseudospectral_radius=10.0)
    rel = _Rel()
    before = _pick_a_class(ev, relaxation=rel)
    triggered_hypotheses(ev, relaxation=rel)     # must be side-effect free
    assert _pick_a_class(ev, relaxation=rel) == before


def test_every_rung_has_a_stable_identifier():
    """Rule ids are the audit handle; duplicates would make the report ambiguous."""
    ids = [r[0] for r in _hypothesis_ladder(_ev(), relaxation=_Rel())]
    assert len(ids) == len(set(ids))
    assert all(rid.isupper() or "_" in rid for rid in ids)


@pytest.mark.parametrize("model,expected", [("M3a", "A10"), ("M2", "A5")])
def test_model_driven_rungs_still_fire(model, expected):
    ev = _ev()
    assert _pick_a_class(ev, relaxation=_Rel(aicc_model=model))[0] == expected
    fired = triggered_hypotheses(ev, relaxation=_Rel(aicc_model=model))
    assert fired[0]["a_class"] == expected


def test_result_carries_the_report():
    """The field is populated on a real classify_mechanism run."""
    from liouscope import diagnose
    from liouscope.core.lindblad import build_liouvillian

    sm = np.array([[0, 1], [0, 0]], dtype=complex)
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), [sm])
    rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    rep = diagnose(L, rho_initial=rho0, t_grid=np.linspace(0.0, 5.0, 64))
    assert isinstance(rep.classification.triggered_hypotheses, tuple)
    if rep.classification.triggered_hypotheses:
        first = rep.classification.triggered_hypotheses[0]
        assert first["a_class"] == rep.classification.a_class
        assert first["shadowed"] is False
