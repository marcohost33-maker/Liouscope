"""Issue #101 addendum: the sign of a round-off gap decides nothing.

``claim_status: pending`` (KANON ``RESEARCH-LIOUSCOPE``).

Follow-up to PR #196, which made the F5 reach leg abstain on a round-off gap.
The same sign was still read in two other places.

* The transient layer scaled the D14 time window to ``horizon / gap`` for any
  positive gap. A round-off gap of 1e-15 gives a window of about 1e16 time
  units, where ``expm`` returns its own error. Measured on one trace-preserving
  generator, varying only the gap handed in: ``trans_amplitude_ratio`` is 1.04
  for a gap of -1e-15 or 0, and 2.3e3, 1.9e5 and 2.4e216 for +1e-15, +3e-16
  and +1e-17.
* ``sym_gap_corroborated`` and ``gns_certified`` required ``gap > 0``, so two
  round-off numbers could grant a positive certificate.

Census on ``main@10be56a`` plus PR #197's first commit (x86-64, NumPy 2.5.3,
SciPy 1.18.1): 187 randomly drawn trace-preserving spectator systems whose gap
is round-off. Of the 59 with a positive gap, 45 were A4/F2 ``CONFIRMED`` at
``PUBLICATION_GRADE``. Of the 128 with a negative gap, none was A4. After the
fix no positive case is A4, and the negative cases are unchanged one by one.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

import liouscope as ls
from liouscope.diagnostics.transient import (
    _resolved_gap_or_gapless,
    compute_transient_layer,
)

_I2 = np.eye(2, dtype=complex)
_SM = np.array([[0, 1], [0, 0]], dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SZ = np.diag([1.0, -1.0]).astype(complex)

# Fields of the transient layer that a gap can reach.
_GAP_DEPENDENT = (
    "trans_amplitude_ratio",
    "kappa_trans",
    "trans_amplitude_centered",
    "trans_amplitude_decaying",
    "trans_amplitude_operational",
)


def _same(a: float, b: float) -> bool:
    """Bit-for-bit equal, with NaN (a withheld diagnostic) equal to NaN."""
    return bool(a == b or (np.isnan(a) and np.isnan(b)))


def _spectator_generator(g: float) -> np.ndarray:
    """Driven-damped qubit A next to an undamped, coherently rotating qubit B."""
    H = g * 5.0 * np.kron(_SX, _I2) + g * 1.0 * np.kron(_I2, _SZ)
    return ls.build_liouvillian(H, [np.kron(_SM, _I2)], [g * 1.0])


def _random_spectator_system(
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Trace-preserving draw ``index`` of the census: ``(L, rho0, rho_ss, g)``.

    Qubit A has a random Hamiltonian and one random jump operator. Qubit B has
    a random Hamiltonian and no dissipation, so its coherences never decay and
    the gap is round-off.
    """
    rng = np.random.default_rng([101, index])

    def hermitian(n: int) -> np.ndarray:
        a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return (a + a.conj().T) / 2

    g = 10.0 ** rng.uniform(-2.0, 2.0)
    h_a, h_b = hermitian(2), hermitian(2)
    jump = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    H = g * (np.kron(h_a, _I2) + np.kron(_I2, h_b))
    L = ls.build_liouvillian(H, [np.kron(jump, _I2)], [g * (abs(rng.normal()) + 0.1)])
    r = hermitian(4)
    rho0 = r @ r.conj().T
    rho0 /= np.trace(rho0).real
    v = expm(L * (300.0 / g)) @ rho0.reshape(-1, order="F")
    rss = v.reshape(4, 4, order="F")
    rss = (rss + rss.conj().T) / 2
    rss /= np.trace(rss).real
    return L, rho0, rss, g


@pytest.mark.parametrize("g", [0.1, 1.0, 10.0])
@pytest.mark.parametrize("gap_over_g", [-1.0e-15, 1.0e-17, 3.0e-16, 1.0e-15, 2.0e-12])
def test_an_unresolved_gap_is_the_gapless_case_for_the_transient_layer(
    g: float, gap_over_g: float
) -> None:
    """Same generator, same resolution: only the round-off gap changes."""
    L = _spectator_generator(g)
    resolution = 2.7e-12 * g
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gapless = compute_transient_layer(L, 0.0, gap_resolution=resolution)
        got = compute_transient_layer(L, gap_over_g * g, gap_resolution=resolution)
    for name in _GAP_DEPENDENT:
        assert _same(getattr(got, name), getattr(gapless, name)), name
    # Not growth: the window is the gapless one, and the value is of order 1.
    assert got.trans_amplitude_ratio < 2.0
    assert got.kappa_trans == float("inf")


def test_a_resolved_gap_still_sets_the_transient_time_window() -> None:
    """Silent direction: the rule does not touch a gap above the resolution."""
    L = ls.build_liouvillian(0.5 * _SX, [_SM], [1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spectral = ls.diagnose(
            L, rho_initial=np.diag([1.0, 0.0]).astype(complex), bootstrap_B=20, seed=1
        ).spectral
        plain = compute_transient_layer(L, spectral.gap)
        ruled = compute_transient_layer(
            L, spectral.gap, gap_resolution=spectral.zero_cutoff
        )
    assert spectral.gap > 1.0e6 * spectral.zero_cutoff
    for name in _GAP_DEPENDENT:
        assert _same(getattr(ruled, name), getattr(plain, name)), name
    assert np.isfinite(ruled.kappa_trans)


@pytest.mark.parametrize(
    ("gap", "resolution", "expected"),
    [
        (1.0, 1.0e-12, 1.0),            # resolved
        (1.0e-15, 1.0e-12, 0.0),        # positive round-off
        (-1.0e-15, 1.0e-12, 0.0),       # negative round-off
        (1.0e-12, 1.0e-12, 0.0),        # at the resolution: not above it
        (0.0, 0.0, 0.0),
        (1.0e-15, 0.0, 1.0e-15),        # no resolution supplied: legacy
        (1.0e-15, float("nan"), 1.0e-15),
        (1.0e-15, -1.0, 1.0e-15),
        (1.0e300, float("inf"), 0.0),   # unusable resolution resolves nothing
        (float("inf"), 1.0e-12, float("inf")),
    ],
)
def test_resolved_gap_or_gapless_states(
    gap: float, resolution: float, expected: float
) -> None:
    assert _resolved_gap_or_gapless(gap, resolution) == expected


def test_a_withheld_gap_stays_withheld() -> None:
    """A NaN gap (D1 withheld) is not turned into a measured gapless 0."""
    assert np.isnan(_resolved_gap_or_gapless(float("nan"), 1.0e-12))


@pytest.mark.parametrize("index", range(12))
def test_round_off_gap_systems_do_not_report_transient_growth(index: int) -> None:
    """End to end on randomly drawn trace-preserving spectator systems.

    The sign of the gap differs between draws and between BLAS stacks. Every
    assertion holds for either sign. Measured on x86-64 OpenBLAS (NumPy 2.5.3,
    SciPy 1.18.1): draws 0, 1, 4, 7 and 8 have a positive gap. Before the fix,
    draws 0, 1, 7 and 8 were A4/F2 ``CONFIRMED`` at ``PUBLICATION_GRADE`` with
    a ``trans_amplitude_ratio`` of 5.4e8, 2.7e3, 1.4e34 and 31.9, and draw 4
    carried ``sym_gap_corroborated = 1.0``. The seven negative-gap draws had a
    ratio between 1.0 and 1.4.
    """
    L, rho0, rss, g = _random_spectator_system(index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = ls.diagnose(
            L, rho_initial=rho0, rho_steady_state=rss, bootstrap_B=20, seed=1
        )
        gapless = compute_transient_layer(L, 0.0)
    ev = rep.classification.evidence
    cert = rep.spectral.zero_mode_certificate
    # Premises: a trace-preserving generator whose gap is round-off.
    assert cert is not None and cert["applicable"] is True
    assert abs(rep.spectral.gap) <= rep.spectral.zero_cutoff

    # D14 is the gapless-window value, whatever the sign of the gap.
    assert rep.transient.trans_amplitude_ratio == gapless.trans_amplitude_ratio
    assert rep.transient.kappa_trans == float("inf")
    # A CPTP semigroup does not grow by orders of magnitude. The bound is the
    # F2 threshold: the growth leg of F2 is not satisfied.
    assert ev["trans_amplitude_ratio"] < 5.0
    assert (rep.classification.a_class, rep.classification.f_family) != ("A4", "F2")
    # No positive gap certificate from a round-off gap.
    assert ev["gns_certified"] == 0.0
    assert ev["sym_gap_corroborated"] == 0.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_round_off_gaps_grant_no_symmetrised_gap_certificate(sign: float) -> None:
    """Measured: gap = 3.5e-15 next to kms_gap = 1.1e-14 granted the certificate.

    ``gap / kms_gap <= 1.2`` holds for those two round-off numbers, and
    ``gap > 0`` was the only requirement on the gap itself.
    """
    from liouscope.diagnostics.classification import _gather_evidence

    L, rho0, rss, _ = _random_spectator_system(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = ls.diagnose(
            L, rho_initial=rho0, rho_steady_state=rss, bootstrap_B=20, seed=1
        )
    cutoff = rep.spectral.zero_cutoff

    def evidence(**fields: float) -> dict[str, float]:
        spectral = replace(rep.spectral, **fields)
        return _gather_evidence(
            spectral, rep.nonnorm, rep.relaxation, rep.resolvent,
            rep.transient, rep.lep, None,
        )

    tiny = 1.0e-4 * cutoff
    ev = evidence(gap=sign * tiny, kms_gap=3.0 * tiny, gns_gap=3.0 * tiny)
    assert ev["gns_certified"] == 0.0
    assert ev["sym_gap_corroborated"] == 0.0

    # Silent direction: resolved gaps of the same ratio still certify.
    big = 1.0e6 * cutoff
    ev = evidence(gap=big, kms_gap=big, gns_gap=big)
    assert ev["gns_certified"] == 1.0
    assert ev["sym_gap_corroborated"] == 1.0
