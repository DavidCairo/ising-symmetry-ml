"""Produces figures/magnetization_vs_T.png:
  <|m|> versus T for L = 8, 16, 32, with the exact infinite-volume curve
  (ising_ml.exact.spontaneous_magnetization) on top and T_c marked.

What you should see: finite lattices follow the exact curve far from T_c and
get rounded off near T_c, more so for smaller L (finite-size effects).
Write one or two sentences about this in the README.

Outline:
  1. temperatures = np.linspace(1.0, 3.5, ~25)
  2. for each L and T: samples = simulate(...); record mean |m| and its
     standard error
  3. plot with error bars, add the exact curve and a vertical line at T_c
  4. save to figures/ (create the folder if needed) -- dpi=150 is enough
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ising_ml.exact import (
    critical_temperature,
    spontaneous_magnetization,
)
from ising_ml.montecarlo import simulate


def main():
    """Plot the mean absolute magnetization <|m|> versus temperature T for various lattice sizes L."""

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Plot <|m|> vs T for various L.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with fewer samples.",
    )
    args = parser.parse_args()

    # Lattice size and temperatures.
    lattice_sizes = (8, 16, 32)

    # Simulation parameters.
    if args.quick:
        # Use fewer samples for a quick test.
        temperatures = np.linspace(1.0, 3.5, 5)
        n_samples = 100
        n_therm = 100
        n_between = 5
    else:
        temperatures = np.linspace(1.0, 3.5, 25)
        n_samples = 1000
        n_therm = 1000
        n_between = 10

    # Random number generator.
    rng = np.random.default_rng(seed=42)

    # Results for each lattice size L: a tuple of (mean magnetization, standard error) for each temperature.
    results = {}

    # Start timing the simulation.
    start_time = time.perf_counter()

    # Experiment for each lattice size L.
    for L in lattice_sizes:
        m_mean = np.zeros(len(temperatures))
        errors = np.zeros(len(temperatures))

        for i, T in enumerate(temperatures):
            # Simulate the Ising model and collect samples.
            samples = simulate(
                L=L,
                T=T,
                n_samples=n_samples,
                n_therm=n_therm,
                n_between=n_between,
                rng=rng,
            )

            # Calculate magnetization of every sample.
            m_values = samples.mean(axis=(1, 2))

            # Take absolute value and average over samples.
            m_values_abs = np.abs(m_values)
            m_mean[i] = np.mean(m_values_abs)

            # Calculate standard error of the mean.
            errors[i] = np.std(m_values_abs, ddof=1) / np.sqrt(n_samples)

        results[L] = (m_mean, errors)
    # Stop timing after the simulations.
    elapsed_time = time.perf_counter() - start_time
    print(f"Simulations completed in {elapsed_time:.2f} seconds.")

    # Creating figure using the Matplotlib API.
    fig, ax = plt.subplots(figsize=(8, 5))

    for L, (m_mean, errors) in results.items():
        ax.errorbar(
            temperatures,
            m_mean,
            yerr=errors,
            fmt="o",
            markersize=3,
            capsize=2,
            label=f"L = {L}",
        )

    # Add exact curve.
    T_exact = np.linspace(1.0, 3.5, 100)
    m_exact = spontaneous_magnetization(T_exact)
    ax.plot(
        T_exact,
        m_exact,
        color="black",
        linestyle="--",
        label="Exact Solution",
    )

    # Mark critical temperature.
    Tc = critical_temperature()
    ax.axvline(
        Tc,
        color="red",
        linestyle=":",
        label="T_c",
    )

    # Set labels, title, legend, and grid.
    ax.set_xlabel("Temperature T")
    ax.set_ylabel(r"$\langle|m|\rangle$")
    ax.set_title("Magnetization vs Temperature")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()

    # Create the figures directory if it doesn't exist.
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Save the figure to the figures directory.
    output_path = figures_dir / "magnetization_vs_T.png"
    fig.savefig(output_path, dpi=150)

    print(f"Figure saved to {output_path}")

    # Release the figure memory
    plt.close(fig)


if __name__ == "__main__":
    main()
