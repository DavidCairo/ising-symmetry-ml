"""Datasets of Ising configurations for training neural networks.

Labels follow Carrasquilla & Melko (2017): label 1 = "ordered" (T < T_c),
label 0 = "disordered" (T > T_c). We use the exact T_c only to make the
labels; the point of the experiment is that the network then *recovers*
T_c from data at temperatures it was never told about.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ising_ml.exact import critical_temperature
from ising_ml.montecarlo import simulate


def generate_dataset(
    L: int,
    temperatures: np.ndarray,
    n_per_T: int,
    seed: int = 0,
    **simulate_kwargs,
) -> dict[str, np.ndarray]:
    """Run simulate at every temperature and collect the results.

    Returns a dict with
      "spins": int8 array (len(temperatures) * n_per_T, L, L)
      "T":     float array of the temperature of each sample
      "label": int array, 1 if T < T_c else 0
    """

    # Store configurations, temperatures, and labels for each temperature.
    all_spins = []
    all_T = []
    all_labels = []

    Tc = critical_temperature()

    for i, T in enumerate(temperatures):
        # Independent, reproducible random stream per temperature.
        rng = np.random.default_rng([seed, i])

        # Generate configurations at this temperature.
        samples = simulate(
            L=L,
            T=float(T),
            n_samples=n_per_T,
            rng=rng,
            **simulate_kwargs,
        )

        # Store the configurations, temperatures, and labels.
        all_spins.append(samples)
        all_T.append(np.full(n_per_T, T, dtype=np.float32))
        # Label 1 = ordered (T < Tc), label 0 = disordered (T > Tc).
        label = 1 if T < Tc else 0
        all_labels.append(np.full(n_per_T, label, dtype=np.int8))

    # Concatenate all the results into single arrays.
    spins = np.concatenate(all_spins, axis=0)
    T_values = np.concatenate(all_T, axis=0)
    label = np.concatenate(all_labels, axis=0)

    return {
        "spins": spins,
        "T": T_values,
        "label": label,
    }


def save_dataset(path: str | Path, data: dict[str, np.ndarray]) -> None:
    """Save a dataset dict to a compressed ``.npz`` file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load a dataset saved with ``save_dataset``."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


class IsingDataset(Dataset):
    """A PyTorch Dataset wrapping the dict returned by ``generate_dataset``.

    ``__getitem__`` must return a pair ``(x, y)`` where
      x : float32 tensor of shape (1, L, L)   -- one input channel
      y : float32 scalar tensor (the label)   -- float for BCEWithLogitsLoss
    """

    def __init__(self, data: dict[str, np.ndarray]):
        self.spins = torch.from_numpy(data["spins"].astype(np.float32)).unsqueeze(1)  # shape (N, 1, L, L)
        self.labels = torch.from_numpy(data["label"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.spins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.spins[idx], self.labels[idx]
