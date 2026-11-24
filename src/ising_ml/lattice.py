"""Observables of the 2D Ising model on an L x L square lattice.

Conventions used throughout this package
----------------------------------------
* A configuration is a NumPy array ``spins`` of shape ``(L, L)`` with entries
  +1 or -1 (dtype ``np.int8``).
* Boundary conditions are periodic (the lattice is a torus).
* Units: J = 1 and k_B = 1, so the Hamiltonian is

      H(s) = - sum_{<ij>} s_i s_j

  where the sum runs over nearest-neighbour pairs, each pair counted once.
"""

from __future__ import annotations

import numpy as np


def neighbour_sum(spins: np.ndarray) -> np.ndarray:
    """Return, for every site, the sum of its four nearest-neighbour spins.

    Parameters
    ----------
    spins : array of shape (L, L) with entries +-1.

    Returns
    -------
    Array of shape (L, L) (integer dtype) where entry (i, j) equals
    s[i+1, j] + s[i-1, j] + s[i, j+1] + s[i, j-1], indices taken modulo L.
    """
    # Sum the spins in the four nearest-neighbour directions.
    neighbours = (
        np.roll(spins, 1, axis=0)
        + np.roll(spins, -1, axis=0)
        + np.roll(spins, 1, axis=1)
        + np.roll(spins, -1, axis=1)
    )

    return neighbours


def energy(spins: np.ndarray) -> float:
    """Total energy H(s) of a configuration (see module docstring)."""
    # Count horizontal and vertical once.
    energy_total = -np.sum(spins * (np.roll(spins, -1, axis=0) + np.roll(spins, -1, axis=1)))

    return float(energy_total)


def magnetization(spins: np.ndarray) -> float:
    """Magnetization per site, m = (1/N) sum_i s_i, with N = L^2.

    Returns the *signed* value in [-1, 1]. (Below T_c you will mostly want
    |m|; we keep the sign here because the Z2 symmetry s -> -s flips it.)
    """
    # Average the spins over the lattice.
    return float(np.mean(spins))
