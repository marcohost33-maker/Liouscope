"""Mpemba layer: D19 slowest-mode overlap, D20 expansion-coefficient scaling.

D19 -- :func:`overlap_c1`. The Mpemba-overlap test (Carollo 2021). Decompose
the initial state in the left-eigenvector basis of L and test whether the
coefficient on the slowest mode (lambda_1 != 0) vanishes. If ``|c_1| << eps``,
the state relaxes anomalously fast (strong-Mpemba *candidate*).

The coefficient reported is the **biorthogonal** expansion coefficient
``c_1 = <l_1, rho_0> / <l_1, r_1>`` -- the physically meaningful weight of the
slowest right-mode ``r_1`` in the decomposition of ``rho_0`` (l_1, r_1 the left
and right eigenvectors of the slowest non-zero mode). Normalising by
``<l_1, r_1>`` rather than ``||l_1||`` (the previous behaviour, for which only
the zero test was meaningful) makes ``c_1`` the actual expansion magnitude.

**Non-triviality guard (LIOU-CLS-002, issue #68).** A vanishing ``c_1`` is only
an *anomalous* Mpemba skip when a state that could generically excite the
slowest mode is fine-tuned so it does not. A ``rho_0`` that **commutes with the
steady state** ``rho_ss`` lies in the classical (population) sector of
``rho_ss``; if the slowest mode is a *coherence* in the ``rho_ss`` eigenbasis,
such a ``rho_0`` has ``c_1 = 0`` by symmetry -- it never populates that mode --
which is trivially fast relaxation, not a Mpemba effect. The default
``rho_0 = I/d`` and any diagonal state fall in this class. :func:`is_trivial_overlap`
detects it, and :func:`compute_mpemba_layer` withholds the candidate flag so a
canonical amplitude-damping / dephasing system cannot earn an anomalous-Mpemba
label from its default inputs.

D20 -- :func:`expansion_alpha`. Scaling exponent of overlap coefficients
``|c_n|`` against the index n. Polynomial scaling ``Phi_n ~ exp(alpha L)``
signals overlap-amplification (F1).
"""

from __future__ import annotations

import warnings

import numpy as np

from .._consts import EPS_DIV, EPS_HERMITICITY
from .._types import MpembaResult
from ..numerics.kronecker import vec
from ..numerics.linalg import certified_eig
from ..numerics.scale import spectral_zero_tolerance


# Twelfth-round review sentinel: "the slow spectrum is UNRESOLVED for this
# system" -- deliberately distinct from ``None`` ("every mode is a zero mode",
# whose 0.0-overlap semantics are correct). Mapping unresolved to 0.0 would
# MANUFACTURE a Mpemba candidate out of solver failure, on the classifier's
# highest-priority rung.
class _UnresolvedType:
    """Singleton marker: the slow spectrum is unresolved for this system."""


_UNRESOLVED = _UnresolvedType()


def _certified_decomposition(
    L_super: np.ndarray, *, what: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Certified eigendecomposition, or ``None`` when it must not be consumed.

    Shared by the Mpemba-layer consumers so they cannot drift apart on whether
    the spectrum they read was trustworthy (issue #112 follow-up). Returns
    ``None`` -- abstention, not a value -- whenever the certificate is
    applicable but NOT resolved (twelfth-round review): that covers both a
    failed certification and a certified spectrum with AMBIGUOUS in-band
    modes, where the arithmetic cannot say which mode is the slow one. On a
    stiff network the uncertified decomposition offered a "slowest" vector
    with Rayleigh value ~-5e7 against an analytic slow mode near -1e-5 --
    consuming it points D19 at a mode that does not exist, on the very rung
    that fires A11/F4.
    """
    decomp, certificate = certified_eig(L_super)
    if certificate.applicable and not certificate.resolved:
        detail = (
            "no eigenvector-producing repair route recovered the structural "
            "zero mode"
            if not certificate.certified
            else f"{certificate.ambiguous_count} in-band mode(s) are ambiguous "
            "(inside the zero-mode tolerance but far above machine zero)"
        )
        warnings.warn(
            f"{what}: the eigendecomposition of this trace-preserving "
            f"generator is unresolved -- {detail} (smallest |lambda| = "
            f"{certificate.residual:.3e}, bound {certificate.bound:.3e}). "
            "The slowest mode -- and therefore the D19 overlap and the A11/F4 "
            "rung consuming it -- is withheld for this system (issues "
            "#112/#113).",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    left = decomp.left_vectors
    if left is None:  # pragma: no cover - certified_eig always sets left vectors
        raise RuntimeError("certified_eig did not return left eigenvectors")
    return decomp.eigenvalues, left, decomp.right_vectors


def _slowest_mode(
    L_super: np.ndarray, *, atol: float | None = None
) -> tuple[np.ndarray, np.ndarray] | _UnresolvedType | None:
    """Return ``(l_slow, r_slow)`` for the slowest non-zero mode, or ``None``.

    The slowest mode is the non-zero eigenvalue with the largest (least
    negative) real part. Returns ``None`` when every mode is a zero mode.

    The zero-mode separation is SCALE-RELATIVE (issue #108). With the former
    absolute floor a pure change of rate units ``L -> cL`` with small ``c``
    pushed every genuine mode below the floor, so this returned ``None``,
    :func:`overlap_c1` returned ``0.0``, and ``0.0 < overlap_threshold`` fired
    a FALSE A11/F4 Mpemba candidate on the highest-priority rung of the
    classifier -- while the issue-#68 non-triviality guard could not correct it,
    since :func:`is_trivial_overlap` depends on this same function.
    """
    L_super = np.asarray(L_super, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
    # Issue #112 follow-up. The slowest-mode EIGENVECTOR is what D19 measures
    # the overlap against, so a spurious slow mode does not merely shift a
    # number -- it points the overlap at a mode that does not exist, on the very
    # rung that fires A11/F4. Certify the decomposition against vec(I)^H L = 0
    # first, and repair it when the incumbent solve lost the zero mode.
    certified = _certified_decomposition(L_super, what="slowest mode")
    if certified is None:
        return _UNRESOLVED
    eigvals, vl, vr = certified
    tol = spectral_zero_tolerance(eigvals, atol=atol, name="eigenvalues of L_super")
    nonzero_mask = np.abs(eigvals) > tol
    if not np.any(nonzero_mask):
        return None
    eigvals_nz = eigvals[nonzero_mask]
    order = np.argsort(-np.real(eigvals_nz))
    slowest_idx = np.where(nonzero_mask)[0][order[0]]
    return vl[:, slowest_idx], vr[:, slowest_idx]


def overlap_c1(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    *,
    atol: float | None = None,
) -> float:
    """Magnitude of the biorthogonal slowest-mode overlap ``|c_1|``.

    ``c_1 = <l_1, vec(rho_0)> / <l_1, r_1>``; the denominator is floored at
    :data:`liouscope._consts.EPS_DIV` so a genuinely defective (``<l_1, r_1> = 0``)
    slow mode yields a large finite value rather than a spurious NaN.

    Returns NaN -- the library's unavailable sentinel -- when the slow
    spectrum is UNRESOLVED (issues #112/#113): a 0.0 there would read as "the
    initial state skips the slowest mode" and fire a false A11/F4 candidate
    from solver failure.
    """
    mode = _slowest_mode(L_super, atol=atol)
    if isinstance(mode, _UnresolvedType):
        return float("nan")
    if mode is None:
        return 0.0
    l_slow, r_slow = mode
    rho_vec0 = vec(np.asarray(rho_initial))
    denom = abs(np.vdot(l_slow, r_slow))
    return float(abs(np.vdot(l_slow, rho_vec0)) / max(denom, EPS_DIV))


def _steady_state_eigenprojectors(
    rho_steady_state: np.ndarray, *, tol: float = EPS_HERMITICITY
) -> list[np.ndarray]:
    """Orthogonal projectors onto the distinct eigenspaces of ``rho_ss``.

    Grouping degenerate eigenvalues into a single projector makes the sector
    decomposition well-defined even when ``rho_ss`` is degenerate or maximally
    mixed -- the regime where an individual-eigenvector basis is arbitrary.
    """
    rho_steady_state = np.asarray(rho_steady_state, dtype=complex)
    w, u = np.linalg.eigh(rho_steady_state)
    projectors: list[np.ndarray] = []
    i = 0
    n = w.size
    while i < n:
        j = i + 1
        while j < n and abs(w[j] - w[i]) <= tol:
            j += 1
        block = u[:, i:j]
        projectors.append(block @ block.conj().T)
        i = j
    return projectors


def is_trivial_overlap(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    rho_steady_state: np.ndarray,
    *,
    atol: float | None = None,
    block_tol: float = 1.0e-9,
) -> bool:
    """Is a vanishing slowest-mode overlap symmetry-protected (not Mpemba)?

    Returns ``True`` when the zero overlap is *structural*: in the sector
    decomposition set by the eigenprojectors ``{P_a}`` of ``rho_ss``, the slowest
    mode ``l_1`` and ``rho_0`` occupy **disjoint** blocks ``P_a (.) P_b``. Then
    ``rho_0`` can never populate the slowest mode -- its fast relaxation is
    trivial, not a fine-tuned Mpemba skip (e.g. a diagonal ``rho_0`` vs a
    coherence slowest mode of an amplitude-damping channel).

    Using eigen*projectors* (not individual eigenvectors) keeps the test robust
    when ``rho_ss`` is degenerate. A maximally mixed ``rho_ss`` collapses to a
    single projector (one block), so no pair of blocks is disjoint, nothing is
    structurally decoupled in the fixpoint, and the guard cannot fire -- this is
    a property of the ``rho_ss`` eigenprojectors, NOT a claim that the Liouvillian
    carries no protecting sector. The single-state maximally-mixed case is instead
    handled by the (demoted) confidence tier / the issue-#78 evidence floor.
    """
    L_super = np.asarray(L_super)
    rho_initial = np.asarray(rho_initial, dtype=complex)
    rho_steady_state = np.asarray(rho_steady_state, dtype=complex)

    mode = _slowest_mode(L_super, atol=atol)
    if isinstance(mode, _UnresolvedType) or mode is None:
        # Unresolved: no guard claim can be made either way (fail-closed).
        return False
    l_slow, _ = mode
    d = rho_steady_state.shape[0]
    l_mat = l_slow.reshape(d, d, order="F")

    projectors = _steady_state_eigenprojectors(rho_steady_state)
    l_scale = max(float(np.linalg.norm(l_mat)), 1.0)
    rho_scale = max(float(np.linalg.norm(rho_initial)), 1.0)
    for p_a in projectors:
        for p_b in projectors:
            l_block = float(np.linalg.norm(p_a @ l_mat @ p_b))
            rho_block = float(np.linalg.norm(p_a @ rho_initial @ p_b))
            if l_block > block_tol * l_scale and rho_block > block_tol * rho_scale:
                return False  # a shared block -> rho_0 can reach the slow mode
    return True


def expansion_alpha(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    *,
    n_modes: int = 6,
    atol: float | None = None,
) -> float:
    """Scaling exponent of overlap coefficients vs. mode-index.

    Fits ``log|c_n| = alpha * n + c0`` for the ``n_modes`` slowest modes.
    Returns ``alpha`` (slope). A flat distribution gives ``alpha`` near 0.
    """
    L_super = np.asarray(L_super, dtype=complex)  # complex128: scipy dispatches by dtype; the double-solve contract (#108) must hold
    certified = _certified_decomposition(L_super, what="mode expansion")
    if certified is None:
        return float("nan")  # unresolved slow spectrum: no expansion to fit
    eigvals, vl, vr = certified
    tol = spectral_zero_tolerance(eigvals, atol=atol, name="eigenvalues of L_super")
    mask = np.abs(eigvals) > tol
    eigvals_nz = eigvals[mask]
    if eigvals_nz.size < 2:
        return 0.0
    order = np.argsort(-np.real(eigvals_nz))
    nz_indices = np.where(mask)[0][order]
    rho_vec0 = vec(np.asarray(rho_initial))
    cs: list[float] = []
    for idx in nz_indices[: min(n_modes, len(nz_indices))]:
        l_n = vl[:, idx]
        denom = abs(np.vdot(l_n, vr[:, idx]))
        cs.append(float(abs(np.vdot(l_n, rho_vec0)) / max(denom, EPS_DIV)))
    if len(cs) < 2:
        return 0.0
    log_cs = np.log(np.clip(np.asarray(cs), 1.0e-30, None))
    n_arr = np.arange(1, len(cs) + 1, dtype=float)
    slope, _ = np.polyfit(n_arr, log_cs, 1)
    return float(slope)


def compute_mpemba_layer(
    L_super: np.ndarray,
    rho_initial: np.ndarray,
    *,
    rho_steady_state: np.ndarray | None = None,
    overlap_threshold: float = 1.0e-4,
) -> MpembaResult:
    """Run D19, D20 and flag Mpemba candidacy.

    ``is_mpemba_candidate`` is ``True`` only when ``|c_1| < overlap_threshold``
    **and** the vanishing overlap is not symmetry-protected. Triviality can only
    be assessed when ``rho_steady_state`` is supplied (the standalone default is
    ``None``, which preserves the raw overlap test for direct callers).
    """
    c1 = overlap_c1(L_super, rho_initial)
    alpha = expansion_alpha(L_super, rho_initial)
    trivial = (
        is_trivial_overlap(L_super, rho_initial, rho_steady_state)
        if rho_steady_state is not None
        else False
    )
    return MpembaResult(
        overlap_c1=c1,
        is_mpemba_candidate=bool(c1 < overlap_threshold and not trivial),
        expansion_alpha=alpha,
        trivial_overlap=trivial,
    )
