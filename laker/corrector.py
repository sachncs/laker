"""Residual corrector models."""

from __future__ import annotations

import torch
import torch.nn as nn


class Corrector(nn.Module):
    """Tiny MLP that predicts the residual ``y - y_hat``.

    The corrector operates on raw spatial coordinates (or optionally on
    embedding vectors) and its output is added to the base LAKER
    prediction at inference time. Single hidden layer with Tanh + dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["Corrector"]
