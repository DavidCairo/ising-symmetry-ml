"""Exact results for the 2D Ising model (Onsager, Yang), used to validate the simulation.

Formulas (J = k_B = 1):

* Critical temperature:   T_c = 2 / ln(1 + sqrt(2))   (about 2.269)
* Spontaneous magnetization in the thermodynamic limit (Yang, 1952):

      m(T) = (1 - sinh(2/T)^(-4))^(1/8)   for T < T_c,
      m(T) = 0                            for T >= T_c.

Make ``spontaneous_magnetization`` work for a float AND for a NumPy array
of temperatures (hint: ``np.where``, and watch out for warnings from taking
a fractional power of a negative number above T_c).
"""

from __future__ import annotations

import numpy as np


def critical_temperature() -> float:
    """Onsager's critical temperature T_c of the square-lattice Ising model."""
    return float(2.0 / np.log(1.0 + np.sqrt(2.0)))


def spontaneous_magnetization(T: float | np.ndarray) -> float | np.ndarray:
    """Exact infinite-volume magnetization per site m(T) (see module docstring)."""

    # The critical temperature.
    Tc = critical_temperature()

    # Check the type of T and convert to array if necessary.
    T = np.asarray(T)

    # Ensure non-negative argument.
    argument = 1 - np.sinh(2 / T) ** -4
    argument = np.maximum(argument, 0)

    # Compute the spontaneous magnetization using the exact formula.
    magnetization = np.where(T < Tc, argument ** (1 / 8), 0)
    return float(magnetization) if magnetization.ndim == 0 else magnetization
