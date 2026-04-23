"""
Binary classifiers: logistic regression baseline and MLP (logits output).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Linear layer mapping flattened features to a single logit."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPClassifier(nn.Module):
    """Feedforward network with at least two hidden ReLU layers; outputs logits."""

    def __init__(self, input_dim: int, hidden1: int = 64, hidden2: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
