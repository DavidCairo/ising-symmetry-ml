"""Markov-chain Monte Carlo sampling of the 2D Ising model.

Day 2: implement ``checkerboard_masks``, ``metropolis_sweep`` and ``simulate``.

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

import numpy as np


def checkerboard_masks(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Return two boolean arrays (black, white) of shape (L, L).

    black[i, j] is True when (i + j) is even, white is its complement.
    Raise ``ValueError`` if L is odd.
    """
    raise NotImplementedError("Day 2")


def metropolis_sweep(spins: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Perform one checkerboard Metropolis sweep IN PLACE and return ``spins``.

    Parameters
    ----------
    spins : (L, L) int8 array of +-1, modified in place.
    beta  : inverse temperature 1/T.
    rng   : NumPy random Generator.

    Hint: use ``neighbour_sum`` from ``ising_ml.lattice``, recompute it
    between the black and the white half-sweep.
    """
    raise NotImplementedError("Day 2")


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
    raise NotImplementedError("Day 2")
