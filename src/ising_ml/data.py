"""Datasets of Ising configurations for training neural networks.

Day 3: implement ``generate_dataset`` and ``IsingDataset``.

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


def generate_dataset(
    L: int,
    temperatures: np.ndarray,
    n_per_T: int,
    seed: int = 0,
    **simulate_kwargs,
) -> dict[str, np.ndarray]:
    """Run ``simulate`` at every temperature and collect the results.

    Returns a dict with
      "spins": int8 array (len(temperatures) * n_per_T, L, L)
      "T":     float array of the temperature of each sample
      "label": int array, 1 if T < T_c else 0

    Use a different (but reproducible) random stream per temperature, e.g.
    ``np.random.default_rng([seed, i])``.
    """
    raise NotImplementedError("Day 3")


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
        raise NotImplementedError("Day 3")

    def __len__(self) -> int:
        raise NotImplementedError("Day 3")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Day 3")
