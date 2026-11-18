"""Tests for ising_ml.lattice and ising_ml.exact (Day 1)."""

import numpy as np
import pytest

from ising_ml.exact import critical_temperature, spontaneous_magnetization
from ising_ml.lattice import energy, magnetization, neighbour_sum


def random_config(L, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))


@pytest.mark.parametrize("L", [2, 4, 7, 16])
def test_ground_state(L):
    s = np.ones((L, L), dtype=np.int8)
    # 2 bonds per site, each contributes -1
    assert energy(s) == -2 * L * L
    assert magnetization(s) == 1.0
    assert magnetization(-s) == -1.0


@pytest.mark.parametrize("L", [4, 8])
def test_antiferromagnetic_state(L):
    i, j = np.indices((L, L))
    s = np.where((i + j) % 2 == 0, 1, -1).astype(np.int8)
    assert energy(s) == 2 * L * L
    assert magnetization(s) == 0.0


def test_neighbour_sum_single_spin():
    # all spins down except one: its four neighbours each see one +1
    L = 5
    s = -np.ones((L, L), dtype=np.int8)
    s[0, 0] = 1
    ns = neighbour_sum(s)
    assert ns.shape == (L, L)
    assert ns[0, 0] == -4
    for site in [(1, 0), (L - 1, 0), (0, 1), (0, L - 1)]:  # periodic wrap-around
        assert ns[site] == -2
    assert ns[2, 2] == -4


def test_energy_is_z2_invariant():
    s = random_config(10)
    assert energy(s) == energy(-s)


def test_energy_is_translation_invariant():
    s = random_config(10)
    assert energy(s) == energy(np.roll(s, shift=(3, -2), axis=(0, 1)))


def test_energy_change_of_single_flip():
    # flipping s_i changes the energy by dE = 2 s_i * (sum of neighbours)
    s = random_config(8, seed=1)
    ns = neighbour_sum(s)
    for i, j in [(0, 0), (3, 5), (7, 7)]:
        flipped = s.copy()
        flipped[i, j] *= -1
        assert energy(flipped) - energy(s) == 2 * s[i, j] * ns[i, j]


def test_energy_does_not_overflow_int8():
    # int8 sums overflow above 127 -- a classic bug for large lattices
    s = np.ones((64, 64), dtype=np.int8)
    assert energy(s) == -2 * 64 * 64


def test_critical_temperature():
    assert critical_temperature() == pytest.approx(2.269185314, rel=1e-8)


def test_spontaneous_magnetization():
    Tc = critical_temperature()
    assert spontaneous_magnetization(0.5) == pytest.approx(1.0, abs=1e-6)
    assert spontaneous_magnetization(3.0) == 0.0
    T = np.array([1.0, 2.0, 2.2, 2.5, 4.0])
    m = spontaneous_magnetization(T)
    assert m.shape == T.shape
    assert np.all(np.diff(m) <= 0)  # decreasing in T
    assert np.all(m[T > Tc] == 0)
    assert 0 < m[2] < m[1] < m[0] < 1
