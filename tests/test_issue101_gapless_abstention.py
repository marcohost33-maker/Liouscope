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


def test_an_inapplicable_certificate_does_not_lend_its_operator_norm_band() -> None:
    """Codex review on PR #196 (P2), reproduced and pinned.

    A non-trace-preserving input has an INAPPLICABLE zero-mode certificate,
    which still carries a finite operator-norm ``zero_tolerance`` (about 2.2e3
    here). The spectral layer refuses to use that value as a cutoff. Reading it
    as ``gap_resolution`` made the resolved gap 1.0 count as gapless and
    suppressed F5.

    Follow-up to PR #196: the resolution is no longer absent in this state. It
    is the cutoff the layer did apply, the radius-based proxy, which is far
    below the resolved gap. The reach is therefore still decided.
    """
    from liouscope.diagnostics.classification import (
        _gather_evidence,
        _pseudospectral_reach,
    )
    from liouscope.diagnostics.spectral import compute_spectral_layer
    from liouscope.numerics.scale import spectral_zero_tolerance

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
    assert ev["gap_resolution"] == spectral_zero_tolerance(spectral.eigenvalues)
    assert ev["gap_resolution"] < 1.0e-9 * spectral.gap
    assert _pseudospectral_reach({**ev, "pseudospectral_radius": 10.0}) == pytest.approx(
        10.0 / spectral.gap
    )


def _without_trace_preservation(L: np.ndarray, s: float) -> np.ndarray:
    """A diagonal similarity ``D L D^-1``: same spectrum, no trace preservation.

    The eigenvalues are those of ``L``, so the coherent spectator's gap is
    still round-off. ``vec(I)^H`` is no longer a left null vector, so the
    zero-mode certificate is inapplicable.
    """
    d = np.ones(L.shape[0])
    d[0], d[1], d[6] = s, 1.0 / s, s * s
    return (d[:, None] * L) / d[None, :]


def _spectator_generator(g: float) -> np.ndarray:
    H = g * 5.0 * np.kron(_SX, _I2) + g * 1.0 * np.kron(_I2, _SZ)
    return ls.build_liouvillian(H, [np.kron(_SM, _I2)], [g * 1.0])


@pytest.mark.parametrize("g", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("s", [1.5, 3.0, 7.3])
def test_a_round_off_gap_abstains_without_an_applicable_certificate(
    g: float, s: float
) -> None:
    """Follow-up to PR #196 (GPT-6 F3/F4, Gemini F4/F6; post-merge review).

    Measured on ``main@10be56a`` (x86-64, NumPy 2.5.3, SciPy 1.18.1): at
    ``g = 1`` the gap of this system is ``+1.3e-17`` for ``s = 1.5`` and
    ``+3.2e-16`` for ``s = 3``, and negative for the other seven cases. With no
    resolution recorded for an inapplicable certificate the threshold was 0.
    The two positive cases gave a reach of about 1e17 to 1e18, and the matrix
    reported F5 ``NOT_SUPPORTED`` with nothing missing. The seven negative
    cases reported ``UNEVALUABLE``. The audit column followed the sign of
    round-off. F5 did not fire there only because ``gap_to_gns_ratio`` was
    infinite; with a finite ratio it fires (the sign test above).

    The assertions hold for either sign, so they do not depend on which cases
    come out positive on a given BLAS stack.
    """
    base = _driven_qubit_with_coherent_spectator(g)
    L = _without_trace_preservation(_spectator_generator(g), s)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho0 = np.kron(np.diag([1.0, 0.0]).astype(complex), np.outer(plus, plus.conj()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = ls.diagnose(
            L, rho_initial=rho0, rho_steady_state=base.spectral.steady_state,
            bootstrap_B=20, seed=1,
        )
    cert = rep.spectral.zero_mode_certificate
    ev = rep.classification.evidence
    # Premises: the certificate does not apply, the gap is round-off, and the
    # sibling non-normality gate holds, so only the reach leg is in question.
    assert cert is not None and cert["applicable"] is False
    assert abs(ev["gap"]) <= 1.0e3 * np.finfo(float).eps * g
    assert ev["henrici_eta"] > 1.0

    # The resolution is the cutoff the layer applied, in the layer's own rate
    # unit: it scales with g and exceeds the round-off gap.
    assert ev["gap_resolution"] == rep.spectral.zero_cutoff
    assert abs(ev["gap"]) < ev["gap_resolution"] < 1.0e-9 * g

    f5 = next(
        e for e in rep.classification.hypothesis_matrix
        if e["rule_id"] == "F5_PSEUDOSPECTRAL"
    )
    assert f5["status"] == "UNEVALUABLE"
    assert f5["missing"] == ("pseudospectral_reach",)


def test_the_gap_resolution_is_the_cutoff_the_spectral_layer_applied() -> None:
    """One decision, one site (round-17 rule), for both certificate states.

    ``ZeroModeCertificate.zero_set_tolerance`` is the only way a layer obtains
    a zero-mode cutoff. The classifier reads the value the spectral layer got
    from it and does not repeat the applicability branch.
    """
    from liouscope.diagnostics.classification import _gap_resolution
    from liouscope.diagnostics.spectral import compute_spectral_layer
    from liouscope.numerics.scale import spectral_zero_tolerance

    rep = _driven_qubit_with_coherent_spectator(1.0)
    applicable = rep.spectral
    cert = applicable.zero_mode_certificate
    assert cert is not None and cert["applicable"] is True
    assert applicable.zero_cutoff == cert["zero_tolerance"]
    assert _gap_resolution(applicable) == applicable.zero_cutoff

    L = _without_trace_preservation(_spectator_generator(1.0), 3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inapplicable = compute_spectral_layer(L, rho_steady=applicable.steady_state)
    cert = inapplicable.zero_mode_certificate
    assert cert is not None and cert["applicable"] is False
    assert inapplicable.zero_cutoff == spectral_zero_tolerance(inapplicable.eigenvalues)
    assert inapplicable.zero_cutoff != cert["zero_tolerance"]
    assert _gap_resolution(inapplicable) == inapplicable.zero_cutoff


@pytest.mark.parametrize(
    ("cutoff", "certificate", "expected"),
    [
        # Recorded cutoff: authoritative, whatever the certificate says.
        (2.0e-12, {"applicable": False, "zero_tolerance": 2.2e3}, 2.0e-12),
        (0.0, {"applicable": True, "zero_tolerance": 1.0e-12}, 0.0),
        # A recorded cutoff that is not a finite non-negative number.
        (float("inf"), {"applicable": True, "zero_tolerance": 1.0e-12}, float("inf")),
        (-1.0, None, float("inf")),
        # No recorded cutoff: an applicable certificate still supplies one.
        (float("nan"), {"applicable": True, "zero_tolerance": 3.0e-12}, 3.0e-12),
        # Applicable, tolerance unusable (as_dict writes None for non-finite).
        (float("nan"), {"applicable": True, "zero_tolerance": None}, float("inf")),
        (float("nan"), {"applicable": True, "zero_tolerance": -1.0}, float("inf")),
    ],
)
def test_gap_resolution_states(
    cutoff: float, certificate: dict[str, object] | None, expected: float
) -> None:
    from dataclasses import replace

    from liouscope.diagnostics.classification import _gap_resolution

    base = _driven_qubit_with_coherent_spectator(1.0).spectral
    spectral = replace(base, zero_cutoff=cutoff, zero_mode_certificate=certificate)
    assert _gap_resolution(spectral) == expected


@pytest.mark.parametrize(
    "certificate",
    [
        None,
        {"applicable": False, "zero_tolerance": 2.2e3},
        # Hand-built certificate with no tolerance entry: nothing was recorded.
        # ``ZeroModeCertificate.as_dict`` always writes the key.
        {"applicable": True, "certified": True, "resolved": True},
    ],
)
def test_gap_resolution_is_unavailable_when_nothing_was_recorded(
    certificate: dict[str, object] | None,
) -> None:
    from dataclasses import replace

    from liouscope.diagnostics.classification import _gap_resolution

    base = _driven_qubit_with_coherent_spectator(1.0).spectral
    spectral = replace(
        base, zero_cutoff=float("nan"), zero_mode_certificate=certificate
    )
    assert np.isnan(_gap_resolution(spectral))


@pytest.mark.parametrize("gap", [1.0e-15, 1.0, 1.0e300])
def test_an_unusable_resolution_resolves_no_gap(gap: float) -> None:
    """GPT-6 F3: applicable certificate, resolution claimed but not statable.

    Before, an unusable resolution degraded to the threshold 0, so
    ``radius = 1e-2`` over ``gap = 1e-15`` gave a reach of 1e13. An infinite
    resolution makes the leg abstain for every gap.
    """
    from liouscope.diagnostics.classification import (
        _hypothesis_ladder,
        _pseudospectral_reach,
    )

    ev = {
        "gap": gap, "gap_resolution": float("inf"),
        "pseudospectral_radius": 1.0e-2, "gap_to_gns_ratio": 1.0, "henrici_eta": 2.0,
    }
    assert np.isnan(_pseudospectral_reach(ev))
    rel = _driven_qubit_with_coherent_spectator(1.0).relaxation
    ladder = {r[0]: r[3] for r in _hypothesis_ladder(ev, relaxation=rel)}
    assert ladder["F5_PSEUDOSPECTRAL"] is False


@pytest.mark.parametrize("g", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("factor", [0.5, 0.99, 1.0, 1.01, 10.0])
def test_the_resolution_boundary_follows_the_rate_unit(g: float, factor: float) -> None:
    """GPT-6 test template: gap at ``factor * resolution`` across rate units.

    The decision depends on ``gap / resolution`` only, so a change of rate
    unit that scales gap, resolution and radius together cannot move it.
    """
    from liouscope.diagnostics.classification import _pseudospectral_reach

    resolution = 2.7e-12 * g
    ev = {
        "gap": factor * resolution, "gap_resolution": resolution,
        "pseudospectral_radius": 5.0 * g,
    }
    reach = _pseudospectral_reach(ev)
    if factor <= 1.0:
        assert np.isnan(reach)
    else:
        assert reach == pytest.approx(5.0 / (factor * 2.7e-12), rel=1e-12)
