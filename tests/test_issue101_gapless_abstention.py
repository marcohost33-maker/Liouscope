"""Issue #101 re-audit addendum: a gapless operator abstains on the F5 reach leg.

``claim_status: pending`` (KANON ``RESEARCH-LIOUSCOPE``).

End-to-end regression on a physical system that ``main@643337a`` mislabelled.
Qubit A is driven and damped, so it is strongly non-normal. Qubit B rotates
coherently with no dissipation. B's coherences never decay, so the spectral
gap is round-off: about -9e-16 at unit rates, and negative. The old reach leg
read ``gap <= 0`` as infinite pseudospectral reach. With the rate-dimensioned
``henrici_eta > 1`` gate this produced A10/F5 "phantom relaxation"
CANDIDATE, from nothing but an undamped spectator rotation.

Before the fix the label also depended on the rate unit: g = 0.1 was not F5,
g = 1 and g = 10 were. That was the #101 scale defect showing through.

Measured before / after (class/family, verdict):

    g=0.1   A12/none NOT_EXCLUDED   ->  unchanged
    g=1     A10/F5   CANDIDATE      ->  A8/none NOT_EXCLUDED
    g=10    A10/F5   CANDIDATE      ->  A12/none UNDEFINED

The silent direction, a genuine F5 phantom with a positive gap, is pinned by
``tests/test_validation_systems/test_d17_gap_coherence.py::
test_phantom_with_gap_matched_rho0_is_not_mislabelled_a1``. That test still
asserts A10/F5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.linalg import expm

import liouscope as ls

_I2 = np.eye(2, dtype=complex)
_SM = np.array([[0, 1], [0, 0]], dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SZ = np.diag([1.0, -1.0]).astype(complex)


def _driven_qubit_with_coherent_spectator(g: float) -> ls.DiagnosticReport:
    H = g * 5.0 * np.kron(_SX, _I2) + g * 1.0 * np.kron(_I2, _SZ)
    L = ls.build_liouvillian(H, [np.kron(_SM, _I2)], [g * 1.0])
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.kron(np.diag([1.0, 0.0]).astype(complex), np.outer(plus, plus.conj()))
    # The stationary manifold is degenerate (B's populations are conserved),
    # so the steady state reached from rho0 is supplied explicitly, as the
    # API requires, from the long-time propagator.
    v = expm(L * (400.0 / g)) @ rho0.reshape(-1, order="F")
    rss = v.reshape(4, 4, order="F")
    rss = (rss + rss.conj().T) / 2
    rss /= np.trace(rss).real
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ls.diagnose(
            L, rho_initial=rho0, rho_steady_state=rss, bootstrap_B=20, seed=1
        )


@pytest.mark.parametrize("g", [1.0, 10.0])
def test_coherent_spectator_rotation_is_not_phantom_relaxation(g: float) -> None:
    rep = _driven_qubit_with_coherent_spectator(g)
    ev = rep.classification.evidence
    # Premises, so the test cannot pass for a different reason. The gap is at
    # round-off level, at most a few hundred ulps of the rate scale g.
    assert abs(ev["gap"]) <= 1.0e3 * np.finfo(float).eps * g
    assert ev["henrici_eta"] > 1.0
    # The defect: gaplessness alone used to supply infinite reach.
    assert (rep.classification.a_class, rep.classification.f_family) != ("A10", "F5")
    f5 = next(
        e for e in rep.classification.hypothesis_matrix
        if e["rule_id"] == "F5_PSEUDOSPECTRAL"
    )
    # Abstention, not refutation: the reach is undefined, not small.
    assert f5["status"] == "UNEVALUABLE"
    assert f5["missing"] == ("pseudospectral_reach",)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_a_round_off_gap_abstains_whatever_its_sign(sign: float) -> None:
    """Gap-floor adversary from the #101 addendum, found on the arm64 CI probe.

    The coherent-spectator gap is round-off, and its sign depends on the BLAS
    stack: negative on x86-64 OpenBLAS, positive on arm64. A positive round-off
    gap of about 1e-15 gives a reach of about 1e15. Without the resolution rule
    that reach fires F5 whenever ``gap_to_gns_ratio`` is finite. This test fixes
    the sign explicitly instead of relying on the platform.
    """
    from liouscope.diagnostics.classification import (
        _hypothesis_ladder,
        hypothesis_evidence_matrix,
    )

    rel = _driven_qubit_with_coherent_spectator(1.0).relaxation
    ev = {
        "gap": sign * 1.0e-15, "gap_resolution": 2.7e-12,
        "pseudospectral_radius": 5.0, "gap_to_gns_ratio": 1.0, "henrici_eta": 2.0,
    }
    ladder = {r[0]: r[3] for r in _hypothesis_ladder(ev, relaxation=rel)}
    assert ladder["F5_PSEUDOSPECTRAL"] is False
    f5 = next(e for e in hypothesis_evidence_matrix(ev, relaxation=rel)
              if e["rule_id"] == "F5_PSEUDOSPECTRAL")
    assert f5["status"] == "UNEVALUABLE"

    # Silent direction: a gap resolved above the same resolution still decides.
    resolved = {**ev, "gap": 1.0}
    ladder = {r[0]: r[3] for r in _hypothesis_ladder(resolved, relaxation=rel)}
    assert ladder["F5_PSEUDOSPECTRAL"] is True


def test_an_inapplicable_certificate_supplies_no_gap_resolution() -> None:
    """Codex review on PR #196 (P2), reproduced and pinned.

    A non-trace-preserving input has an INAPPLICABLE zero-mode certificate,
    which still carries a finite operator-norm ``zero_tolerance`` (about 2.2e3
    here). The spectral layer refuses to use that value as a cutoff. Reading it
    as ``gap_resolution`` made the resolved gap 1.0 count as gapless and
    suppressed F5. The silent direction is that an applicable certificate still
    supplies the resolution, which the coherent-spectator tests above pin.
    """
    from liouscope.diagnostics.classification import (
        _gather_evidence,
        _pseudospectral_reach,
    )
    from liouscope.diagnostics.spectral import compute_spectral_layer

    A = np.diag([-1.0, -2.0, -3.0, -4.0]).astype(complex)
    A[0, 3] = 1.0e16
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectral = compute_spectral_layer(A, rho_steady=np.eye(2, dtype=complex) / 2)
    cert = spectral.zero_mode_certificate
    assert cert is not None
    assert cert["applicable"] is False and cert["zero_tolerance"] > spectral.gap

    rep = _driven_qubit_with_coherent_spectator(1.0)
    ev = _gather_evidence(
        spectral, rep.nonnorm, rep.relaxation, rep.resolvent,
        rep.transient, rep.lep, None,
    )
    assert np.isnan(ev["gap_resolution"])
    assert _pseudospectral_reach({**ev, "pseudospectral_radius": 10.0}) == pytest.approx(
        10.0 / spectral.gap
    )
