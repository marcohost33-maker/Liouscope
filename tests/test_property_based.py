"""Property-based (Hypothesis) oracles for the numerics core.

These generalise the fixed-fixture metamorphic relations in
``test_metamorphic_spectral.py`` to *randomly drawn* GKSL systems: Hypothesis
searches the input space for counterexamples and shrinks any failure to a
minimal reproducer. All properties are ground-truth-free invariants — a
violation is a pure implementation bug, never a physics judgement call.

Covered invariants:

* gap scaling ``gap(c*L) = c*gap(L)`` (linearity of the spectrum),
* spectrum invariance under unitary similarity,
* trace-distance axioms (range, symmetry, identity of indiscernibles,
  triangle inequality) plus CPTP contractivity under ``exp(dt*L)``,
* the Choi CPTP gate accepts every honestly constructed GKSL propagator.

CI determinism: ``derandomize=True`` makes Hypothesis derive examples from the
test body instead of a wall-clock seed, so runs are reproducible (AGENTS.md
"Reproducibility") and cannot flake nondeterministically. ``deadline=None``
because eigensolver wall time varies across CI hosts, not with input size.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from liouscope import build_liouvillian
from liouscope.diagnostics.relaxation import trace_distance
from liouscope.diagnostics.spectral import liouvillian_gap
from liouscope.numerics.cptp import cptp_choi_gate
from liouscope.numerics.kronecker import unvec, vec
from liouscope.numerics.linalg import eig_nonhermitian

_SETTINGS = settings(max_examples=30, deadline=None, derandomize=True)

_DIMS = st.sampled_from([2, 3])

# Bounded, well-scaled entries: keeps the eigenproblems benign so failures
# point at logic bugs, not float saturation.
_ENTRY = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _complex_matrix(draw, n: int) -> np.ndarray:
    re = draw(st.lists(_ENTRY, min_size=n * n, max_size=n * n))
    im = draw(st.lists(_ENTRY, min_size=n * n, max_size=n * n))
    return (np.array(re) + 1j * np.array(im)).reshape(n, n)


@st.composite
def gksl_systems(draw) -> np.ndarray:
    """A random honest GKSL generator: Hermitian H + 1-2 jump ops, rates > 0."""
    n = draw(_DIMS)
    A = _complex_matrix(draw, n)
    H = 0.5 * (A + A.conj().T)
    n_jumps = draw(st.integers(min_value=1, max_value=2))
    jumps = [_complex_matrix(draw, n) for _ in range(n_jumps)]
    rates = [
        draw(st.floats(min_value=0.05, max_value=2.0, allow_nan=False))
        for _ in range(n_jumps)
    ]
    # A jump operator that is (numerically) zero makes the dissipator vacuous;
    # the invariants still hold but the gap may collapse to a pure-Hamiltonian
    # degeneracy — filter for a non-trivial dissipative part.
    assume(all(np.linalg.norm(j) > 1e-3 for j in jumps))
    return build_liouvillian(H, jumps, rates)


@st.composite
def density_matrices(draw) -> np.ndarray:
    """A random full-rank-ish density matrix rho = A A^dag / tr(A A^dag)."""
    n = draw(_DIMS)
    A = _complex_matrix(draw, n)
    P = A @ A.conj().T
    tr = float(np.trace(P).real)
    assume(tr > 1e-6)
    return P / tr


def _density_pair(draw, n: int) -> tuple[np.ndarray, np.ndarray]:
    out = []
    for _ in range(2):
        A = _complex_matrix(draw, n)
        P = A @ A.conj().T
        tr = float(np.trace(P).real)
        assume(tr > 1e-6)
        out.append(P / tr)
    return out[0], out[1]


@st.composite
def gksl_with_states(draw) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = draw(_DIMS)
    A = _complex_matrix(draw, n)
    H = 0.5 * (A + A.conj().T)
    J = _complex_matrix(draw, n)
    assume(np.linalg.norm(J) > 1e-3)
    rate = draw(st.floats(min_value=0.05, max_value=2.0, allow_nan=False))
    L = build_liouvillian(H, [J], [rate])
    rho, sigma = _density_pair(draw, n)
    dt = draw(st.floats(min_value=0.01, max_value=5.0, allow_nan=False))
    return L, rho, sigma, dt


def _gap(L: np.ndarray) -> float:
    return liouvillian_gap(eig_nonhermitian(L).eigenvalues)


# ---------------------------------------------------------------------------
# Spectral invariants
# ---------------------------------------------------------------------------


@_SETTINGS
@given(L=gksl_systems(), c=st.floats(min_value=0.1, max_value=10.0, allow_nan=False))
def test_gap_scaling_property(L, c):
    np.testing.assert_allclose(_gap(c * L), c * _gap(L), rtol=1e-8, atol=1e-10)


@_SETTINGS
@given(L=gksl_systems(), data=st.data())
def test_spectrum_unitary_similarity_property(L, data):
    n = L.shape[0]
    B = _complex_matrix(data.draw, n)
    assume(np.linalg.norm(B) > 1e-3)
    U, _ = np.linalg.qr(B + 1e-3 * np.eye(n))
    assume(np.allclose(U @ U.conj().T, np.eye(n), atol=1e-10))
    z_orig = np.sort_complex(np.round(eig_nonhermitian(L).eigenvalues, 6))
    z_sim = np.sort_complex(np.round(eig_nonhermitian(U @ L @ U.conj().T).eigenvalues, 6))
    np.testing.assert_allclose(z_sim, z_orig, atol=1e-5)


# ---------------------------------------------------------------------------
# Trace-distance axioms + contractivity
# ---------------------------------------------------------------------------


@_SETTINGS
@given(rho=density_matrices(), sigma=density_matrices())
def test_trace_distance_axioms(rho, sigma):
    assume(rho.shape == sigma.shape)
    d = trace_distance(rho, sigma)
    assert 0.0 <= d <= 1.0 + 1e-12
    assert abs(trace_distance(sigma, rho) - d) < 1e-12  # symmetry
    assert trace_distance(rho, rho) < 1e-12  # identity of indiscernibles


@_SETTINGS
@given(data=st.data())
def test_trace_distance_triangle_inequality(data):
    n = data.draw(_DIMS)
    rho, sigma = _density_pair(data.draw, n)
    tau, _ = _density_pair(data.draw, n)
    assert trace_distance(rho, sigma) <= (
        trace_distance(rho, tau) + trace_distance(tau, sigma) + 1e-10
    )


@_SETTINGS
@given(args=gksl_with_states())
def test_trace_distance_cptp_contractivity(args):
    import scipy.linalg as sla

    L, rho, sigma, dt = args
    Phi = sla.expm(dt * L)
    rho_t = unvec(Phi @ vec(rho))
    sigma_t = unvec(Phi @ vec(sigma))
    assert trace_distance(rho_t, sigma_t) <= trace_distance(rho, sigma) + 1e-8


@_SETTINGS
@given(args=gksl_with_states(), split=st.floats(min_value=0.1, max_value=0.9))
def test_propagator_semigroup_property(args, split):
    """Phi_{t1+t2} = Phi_{t2} @ Phi_{t1} — the defining GKSL semigroup law."""
    import scipy.linalg as sla

    L, rho, _, dt = args
    t1 = split * dt
    t2 = dt - t1
    rho_direct = unvec(sla.expm(dt * L) @ vec(rho))
    rho_composed = unvec(sla.expm(t2 * L) @ (sla.expm(t1 * L) @ vec(rho)))
    np.testing.assert_allclose(rho_composed, rho_direct, atol=1e-9)


# ---------------------------------------------------------------------------
# CPTP Choi gate: no false alarms on honest GKSL input
# ---------------------------------------------------------------------------


@_SETTINGS
@given(args=gksl_with_states())
def test_cptp_gate_accepts_honest_gksl(args):
    L, _, _, dt = args
    result = cptp_choi_gate(L, dt)
    assert result.is_cptp, (
        f"honest GKSL propagator flagged non-CPTP: min_eig={result.min_eig}, "
        f"tp_residual={result.tp_residual}, herm_residual={result.choi_herm_residual}"
    )
