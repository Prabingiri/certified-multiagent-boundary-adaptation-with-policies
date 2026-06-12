"""Flat MLP actor and critic used by CPAC."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlatEdgeActor(nn.Module):
    """Three-way actor head over signed actions {-1, 0, +1}."""

    def __init__(self, state_dim: int = 7, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.logits = nn.Linear(hidden, 3)

    def forward(self, s_ij: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.logits(self.net(s_ij))


class FlatCritic(nn.Module):
    """Per-interface value function used during training."""

    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
