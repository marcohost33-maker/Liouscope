"""Tests for the core module utilities: lattice, hamiltonian, jumps."""

from __future__ import annotations

import numpy as np
import pytest

from liouscope import build_liouvillian
from liouscope.core import (
    bose_hubbard_hamiltonian,
    boundary_dephasing_jumps,
    bulk_amplitude_damping_jumps,
    bulk_dephasing_jumps,
    engineered_target_jumps,
    heisenberg_xxz_hamiltonian,
    honeycomb_lattice,
    ising_hamiltonian,
    one_d_chain,
    square_lattice,
    triangular_lattice,
    xy_hamiltonian,
)
from liouscope.core.hamiltonian import single_site_operator, two_site_operator
from liouscope.core.lattice import Lattice

# --- Lattice tests --------------------------------------------------------


def test_one_d_chain_open_pairs():
    lat = one_d_chain(4)
    assert lat.n_sites == 4
    assert set(lat.pairs) == {(0, 1), (1, 2), (2, 3)}


def test_one_d_chain_periodic():
    lat = one_d_chain(4, periodic=True)
    assert (0, 3) in lat.pairs


def test_one_d_chain_rejects_short():
    with pytest.raises(ValueError):
        one_d_chain(1)


def test_square_lattice_2x2():
    lat = square_lattice(2, 2)
    assert lat.n_sites == 4
    assert len(lat.pairs) == 4  # 2 horizontal + 2 vertical


def test_square_lattice_periodic_2x2():
    lat = square_lattice(2, 2, periodic=True)
    assert lat.n_sites == 4


def test_square_lattice_rejects_zero():
    with pytest.raises(ValueError):
        square_lattice(0, 1)


def test_honeycomb_lattice():
    lat = honeycomb_lattice(3)
    assert lat.n_sites == 6


def test_honeycomb_lattice_rejects_zero():
    with pytest.raises(ValueError):
        honeycomb_lattice(0)


def test_triangular_lattice():
    lat = triangular_lattice(2, 2)
    assert lat.n_sites == 4
    assert len(lat.pairs) > 4  # diagonals added


def test_lattice_validates_pairs():
    with pytest.raises(ValueError):
        Lattice(n_sites=2, pairs=((1, 0),))


def test_lattice_validates_range():
    with pytest.raises(ValueError):
        Lattice(n_sites=2, pairs=((0, 5),))


# --- Hamiltonian tests ----------------------------------------------------


def test_single_site_operator(pauli):
    op = single_site_operator(pauli["X"], 0, 2)
    assert op.shape == (4, 4)
    expected = np.kron(pauli["X"], np.eye(2, dtype=complex))
    np.testing.assert_allclose(op, expected, atol=1e-12)


def test_single_site_operator_rejects_out_of_range(pauli):
    with pytest.raises(ValueError):
        single_site_operator(pauli["X"], 5, 2)


def test_two_site_operator_rejects_invalid(pauli):
    with pytest.raises(ValueError):
        two_site_operator(pauli["X"], pauli["Z"], 1, 0, 2)


def test_ising_hamiltonian_is_hermitian():
    lat = one_d_chain(3)
    H = ising_hamiltonian(lat, J=1.0, h=0.5)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_xy_hamiltonian_is_hermitian():
    lat = one_d_chain(3)
    H = xy_hamiltonian(lat, J=1.0, h=0.3)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_heisenberg_xxz_hamiltonian_is_hermitian():
    lat = one_d_chain(3)
    H = heisenberg_xxz_hamiltonian(lat, J=1.0, Delta=0.7)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_bose_hubbard_hamiltonian_is_hermitian():
    H = bose_hubbard_hamiltonian(n_sites=2, n_max=1, t=1.0, U=2.0)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_bose_hubbard_rejects_single_site():
    with pytest.raises(ValueError):
        bose_hubbard_hamiltonian(n_sites=1, n_max=2)


# --- Jump-operator tests --------------------------------------------------


def test_bulk_dephasing_jumps_count():
    jumps = bulk_dephasing_jumps(3)
    assert len(jumps) == 3
    for j in jumps:
        assert j.shape == (8, 8)


def test_bulk_amplitude_damping_to_ground():
    jumps = bulk_amplitude_damping_jumps(1)
    assert len(jumps) == 1
    L = build_liouvillian(np.zeros((2, 2), dtype=complex), jumps, [0.5])
    from liouscope import steady_state
    rho_ss = steady_state(L)
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    np.testing.assert_allclose(rho_ss, expected, atol=1e-9)


def test_boundary_dephasing_jumps_count():
    jumps = boundary_dephasing_jumps(4)
    assert len(jumps) == 2
    jumps_one = boundary_dephasing_jumps(1)
    assert len(jumps_one) == 1


def test_boundary_dephasing_rejects_zero():
    with pytest.raises(ValueError):
        boundary_dephasing_jumps(0)


def test_engineered_target_jumps():
    target = np.array([1.0, 0.0], dtype=complex)
    jumps = engineered_target_jumps(target)
    assert len(jumps) == 2
    for j in jumps:
        assert j.shape == (2, 2)
