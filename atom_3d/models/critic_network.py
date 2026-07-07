"""
Critic Network (Value Estimator) for REINFORCE with Baseline.

Estimates the expected reward V(s) given the graph summary embedding.
Used to reduce variance in the policy gradient (Paper A Eq. 31).

The critic is trained via MSE loss: L_critic = (V(s) - R_actual)²
"""

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    Critic / value network for TENMA training.

    Takes the graph-level summary embedding h_sa from the encoder
    and predicts the expected reward (scalar value).

    Architecture: h_sa → FC → ReLU → FC → ReLU → FC → scalar

    Args:
        embed_dim: Dimension of graph summary embedding.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_graph: torch.Tensor) -> torch.Tensor:
        """
        Predict expected reward from graph summary.

        Args:
            h_graph: (batch_size, embed_dim) graph summary embedding.

        Returns:
            value: (batch_size, 1) predicted reward value.
        """
        return self.network(h_graph)
