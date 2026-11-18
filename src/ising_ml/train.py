"""Training and evaluation loops.

Day 3: implement ``train_one_epoch``, ``evaluate`` and ``predict_proba``.

Keep these functions generic: they should work for any model in
``ising_ml.models`` and any DataLoader yielding (x, y) pairs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Use Apple-silicon GPU (mps) or CUDA when available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """One pass over ``loader``. Returns the mean training loss.

    The canonical loop: model.train(); for each batch: move to device,
    zero_grad, forward, loss, backward, step. Accumulate the loss weighted
    by batch size.
    """
    raise NotImplementedError("Day 3")


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> dict[str, float]:
    """Return {"loss": mean loss, "accuracy": fraction correct} on ``loader``.

    Remember model.eval(). A logit > 0 means "predicted ordered".
    """
    raise NotImplementedError("Day 3")


@torch.no_grad()
def predict_proba(
    model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 512
) -> torch.Tensor:
    """Probability of "ordered" (sigmoid of the logit) for every sample in x, shape (N,)."""
    raise NotImplementedError("Day 3")
