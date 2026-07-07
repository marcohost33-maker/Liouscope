"""End-to-end regression for issue #88: F3 must not fire off the GNS floor sentinel.

``diagnostics.spectral.gns_gap`` deliberately returns ~0 (a conservative floor,
the documented 2026-07 audit-A1 behaviour) for non-detailed-balance systems
whose steady state carries coherences: the GNS symmetrisation certifies no
contraction there. Before the #88 fix, ``_gather_evidence`` turned that
sentinel into ``gap_to_gns_ratio ~ 1e10..inf`` and the F3 branch plus
``_confidence`` promoted it straight to A2/F3 CONFIRMED / PUBLICATION_GRADE --
a publication-grade, specific-literature mechanism claim (Mori-Shirai 2023
symmetrised-gap reduction) keyed on the ABSENCE of a certificate.

The canonical repro is a textbook Rabi-driven, amplitude-damped qubit: unique
coherent steady state, only mild non-normality, and ``Delta_KMS == Delta``
(i.e. provably no real symmetrised-gap reduction). The golden validation set
does not cover this input class (V1/V3/V4 have diagonal steady states, V5 is
caught by the Mpemba branch first), hence this dedicated e2e guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian, diagnose
from liouscope._consts import TIER_PUBLICATION, VERDICT_CONFIRMED

_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SM = np.array([[0, 1], [0, 0]], dtype=complex)


@pytest.mark.parametrize("drive", [0.3, 0.7, 1.5])
def test_rabi_damped_qubit_is_not_a2_f3_off_the_sentinel(drive):
    L = build_liouvillian(drive * _SX, [_SM], [1.0])
    report = diagnose(L, bootstrap_B=15, seed=42)
    s, c = report.spectral, report.classification

    # Precondition: this really is the sentinel regime the issue describes --
    # floored Delta_GNS, exploded ratio, and no KMS-measured reduction.
    assert s.gns_gap < 1.0e-8 * s.gap
    assert c.evidence["gns_certified"] == 0.0
    assert c.evidence["gap_to_gns_ratio"] > 1.0e6
    assert s.kms_gap == pytest.approx(s.gap, rel=1.0e-6)

    # The fix: the uncertified sentinel buys neither the A2/F3 label nor a
    # publication-grade confirmation.
    assert (c.a_class, c.f_family) != ("A2", "F3")
    assert not (c.verdict == VERDICT_CONFIRMED and c.tier == TIER_PUBLICATION)
