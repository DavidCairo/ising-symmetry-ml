"""Tests for ising_ml.montecarlo (Day 2).

The most important test is ``test_matches_exact_enumeration``: for a tiny
4x4 lattice there are only 2^16 = 65536 configurations, so we can compute
thermal averages EXACTLY and check that the Markov chain reproduces them.
If your sampler violates detailed balance, this test catches it.
"""

import itertools

import numpy as np
import pytest

from ising_ml.montecarlo import checkerboard_masks, metropolis_sweep, simulate


def exact_averages(L, T):
    """Exact <E>/N and <|m|> by brute-force enumeration (only for tiny L)."""
    N = L * L
    states = np.array(list(itertools.product([-1, 1], repeat=N)), dtype=np.int64).reshape(-1, L, L)
    E = -np.sum(states * np.roll(states, 1, axis=1), axis=(1, 2)) - np.sum(
        states * np.roll(states, 1, axis=2), axis=(1, 2)
    )
    absm = np.abs(states.sum(axis=(1, 2))) / N
    w = np.exp(-(E - E.min()) / T)  # shift for numerical stability
    Z = w.sum()
    return (w * E).sum() / Z / N, (w * absm).sum() / Z


def test_checkerboard_masks():
    black, white = checkerboard_masks(4)
    assert black.dtype == bool and black.shape == (4, 4)
    assert np.all(black ^ white)  # complementary
    assert black.sum() == white.sum() == 8
    # no black site has a black neighbour
    assert not np.any(black & np.roll(black, 1, axis=0))
    assert not np.any(black & np.roll(black, 1, axis=1))


def test_checkerboard_odd_L_raises():
    with pytest.raises(ValueError):
        checkerboard_masks(5)


def test_sweep_preserves_shape_and_values():
    rng = np.random.default_rng(0)
    s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(8, 8))
    out = metropolis_sweep(s, beta=0.4, rng=rng)
    assert out is s  # in place
    assert s.dtype == np.int8
    assert set(np.unique(s)) <= {-1, 1}


def test_sweep_is_reproducible():
    a = simulate(8, 2.3, n_samples=5, n_therm=20, n_between=2, rng=np.random.default_rng(42))
    b = simulate(8, 2.3, n_samples=5, n_therm=20, n_between=2, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_simulate_output():
    out = simulate(8, 2.0, n_samples=7, n_therm=10, n_between=1, rng=np.random.default_rng(0))
    assert out.shape == (7, 8, 8)
    assert out.dtype == np.int8


def test_low_temperature_stays_ordered():
    out = simulate(16, 0.5, n_samples=10, n_therm=50, n_between=5, rng=np.random.default_rng(1), init="up")
    m = out.mean(axis=(1, 2))
    assert np.all(m > 0.99)


def test_high_temperature_is_disordered():
    out = simulate(32, 10.0, n_samples=50, n_therm=50, n_between=2, rng=np.random.default_rng(2))
    m = out.mean(axis=(1, 2))
    assert abs(m.mean()) < 0.05
    assert np.abs(m).mean() < 0.15


@pytest.mark.parametrize("T", [1.8, 2.5, 4.0])
def test_matches_exact_enumeration(T):
    L = 4
    E_exact, absm_exact = exact_averages(L, T)
    out = simulate(L, T, n_samples=20000, n_therm=500, n_between=2, rng=np.random.default_rng(7))
    s = out.astype(np.int64)
    E = -np.sum(s * np.roll(s, 1, axis=1), axis=(1, 2)) - np.sum(s * np.roll(s, 1, axis=2), axis=(1, 2))
    assert E.mean() / (L * L) == pytest.approx(E_exact, abs=0.02)
    assert np.abs(s.mean(axis=(1, 2))).mean() == pytest.approx(absm_exact, abs=0.02)
