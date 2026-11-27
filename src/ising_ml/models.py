"""Neural network classifiers for Ising configurations.

Every model maps a batch of shape (B, 1, L, L) to logits of shape (B,).
(Return raw logits, NOT probabilities: use ``nn.BCEWithLogitsLoss`` in training.)

Design notes
------------
* Use ``padding_mode="circular"`` in ``nn.Conv2d``: the lattice is periodic,
  so the network should be too. Together with global average pooling this
  makes the model invariant under lattice translations -- a symmetry you get
  "for free" from the architecture.
* The model must work for ANY L (needed on Day 5 for finite-size scaling).
  Global pooling instead of a flatten + Linear layer makes that possible.

The Z2 symmetry
---------------
The Hamiltonian is invariant under the global flip s -> -s, and so is the
question "ordered or disordered?". A plain CNN has to *learn* this from data.
``Z2InvariantCNN`` must satisfy f(-x) = f(x) EXACTLY, by construction.
There are several ways to achieve this; think about it before looking anything up:
  (a) symmetrise: f(x) = g(x) + g(-x) for an arbitrary network g;
  (b) make the first layer odd (no bias), and apply an even function
      (e.g. |.| or square) right after it;
  (c) feed the network only Z2-invariant inputs (e.g. products of neighbouring spins).
Pick one, and be ready to explain why it is exactly invariant. Bonus: how does
the parameter count compare to PlainCNN, and which option is cheapest?
"""

from __future__ import annotations

import torch
from torch import nn


class PlainCNN(nn.Module):
    """A small CNN: a few circular-padded conv layers + ReLU, global average
    pooling, and a linear head producing one logit per sample.
    """

    def __init__(self, channels: int = 16, n_layers: int = 2, kernel_size: int = 3):
        super().__init__()

        if n_layers < 1:
            raise ValueError("n_layers must be at least 1.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        padding = kernel_size // 2  # To keep the spatial dimensions the same

        layers = []

        # First layer: input channels = 1 (for Ising spins), output channels = channels
        layers.append(
            nn.Conv2d(1, channels, kernel_size=kernel_size, padding=padding, padding_mode="circular")
        )
        layers.append(nn.ReLU())

        # Additional convolutional layers
        for _ in range(n_layers - 1):
            layers.append(
                nn.Conv2d(
                    channels, channels, kernel_size=kernel_size, padding=padding, padding_mode="circular"
                )
            )
            layers.append(nn.ReLU())

        self.features = nn.Sequential(*layers)

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Linear head to produce one logit
        self.classifier = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolutional layers
        x = self.features(x)

        # Global average pooling
        x = self.pool(x)

        # Flatten to (B, channels)
        x = torch.flatten(x, 1)

        # Linear head
        x = self.classifier(x)

        # Return shape (B,)
        return x.squeeze(-1)


class Z2InvariantCNN(nn.Module):
    """Same interface as PlainCNN, but exactly invariant under x -> -x."""

    def __init__(self, channels: int = 16, n_layers: int = 2, kernel_size: int = 3):
        super().__init__()
        raise NotImplementedError("Day 4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Day 4")


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
