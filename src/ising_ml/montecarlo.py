"""Markov-chain Monte Carlo sampling of the 2D Ising model.

Background
----------
Metropolis: propose flipping spin s_i; the energy change is

    dE = 2 * s_i * (sum of the four neighbours of i)

Accept the flip with probability min(1, exp(-beta * dE)).

Updating all sites at once is WRONG (neighbours would be updated
simultaneously, which breaks detailed balance). The trick: colour the
lattice like a chessboard. Black sites only have white neighbours, so all
black sites can be updated simultaneously, then all white sites. With NumPy
that turns a slow double loop into two vectorised steps. One "sweep" =
one black update + one white update (each site gets one attempted flip).

Requires L even (otherwise the periodic chessboard colouring breaks).

Randomness: always pass a ``np.random.Generator`` (``np.random.default_rng(seed)``)
so that every result in the repo is reproducible.
"""

from __future__ import annotations
from ising_ml.lattice import neighbour_sum

import numpy as np


def checkerboard_masks(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two boolean arrays (black, white) of shape (L, L).

    black[i, j] is True when (i + j) is even, white is its complement.
    Raise ``ValueError`` if L is odd.
    """
    # Check that L is even.
    if L % 2 != 0:
        raise ValueError("L must be even for checkerboard masks.")

    # Form the coordinate grid.
    (i, j) = np.indices((L, L))

    # Assign colour.
    black = (i + j) % 2 == 0
    white = ~black

    return black, white


def metropolis_sweep(spins: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Perform one checkerboard Metropolis sweep in place and return ``spins``.

    Parameters
    ----------
    spins : (L, L) int8 array of +-1, modified in place.
    beta  : inverse temperature 1/T.
    rng   : NumPy random Generator.

    Hint: use ``neighbour_sum`` from ``ising_ml.lattice``, recompute it
    between the black and the white half-sweep.
    """

    # Get checkerboard masks.
    L = spins.shape[0]
    black, white = checkerboard_masks(L)

    # Perform the two half-sweeps: black first, then white.
    for mask in (black, white):
        # Calculate the energy change for every lattice site.
        neighbours = neighbour_sum(spins)
        dE = 2 * spins * neighbours

        # Calculate the Metropolis acceptance probabilities.
        acceptance = np.exp(-beta * dE)

        # Generate random numbers for the lattice.
        random_values = rng.random(spins.shape)

        # Accept and perform flips only at sites belonging to this sublattice.
        flip = mask & (random_values < acceptance)
        spins[flip] *= -1

    return spins


def simulate(
    L: int,
    T: float,
    n_samples: int,
    n_therm: int = 1000,
    n_between: int = 10,
    rng: np.random.Generator | None = None,
    init: str = "random",
) -> np.ndarray:
    """Sample configurations from the Boltzmann distribution at temperature T.

    Steps:
      1. initialise an (L, L) lattice: ``init="random"`` (hot start) or
         ``init="up"`` (cold start, all +1);
      2. do ``n_therm`` sweeps to thermalise (these are discarded);
      3. then record one configuration every ``n_between`` sweeps,
         until you have ``n_samples`` of them.

    Returns
    -------
    int8 array of shape (n_samples, L, L).
    """
    # Validate the lattice size and simulation parameters.
    if L % 2 != 0:
        raise ValueError("L must be even.")
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    if n_samples < 0 or n_therm < 0 or n_between < 1:
        raise ValueError("Invalid number of samples or sweeps.")

    # Create a random number generator if none was provided.
    if rng is None:
        rng = np.random.default_rng()

    # Initialise the lattice. Ising spins can only take two values: +1 or -1. Use np.int8 to save memory.
    if init == "random":
        spins = rng.choice([-1, 1], size=(L, L)).astype(np.int8)
    elif init == "up":
        spins = np.ones((L, L), dtype=np.int8)
    else:
        raise ValueError("init must be either 'random' or 'up'.")

    # Calculate the inverse temperature.
    beta = 1.0 / T

    # Thermalise the lattice (discard these configurations).
    for _ in range(n_therm):
        metropolis_sweep(spins, beta, rng)

    # Allocate memory for the sampled configurations.
    samples = np.empty((n_samples, L, L), dtype=np.int8)

    # Collect configurations after the specified number of sweeps.
    for n in range(n_samples):
        for _ in range(n_between):
            metropolis_sweep(spins, beta, rng)

        samples[n] = spins

    return samples
