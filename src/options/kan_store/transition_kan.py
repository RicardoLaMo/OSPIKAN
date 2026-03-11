"""
Regime Transition Probability KAN Network.

Maps (regime, horizon) -> softmax probabilities over 4 regimes.
"""

import torch
import torch.nn as nn
from src.physics.kan_layers import KANNetwork, KANLayerConfig


class TransitionKAN(nn.Module):
    """
    KAN network for regime transition probabilities.

    Input: (regime[8], horizon[1]) = 9 dimensions
    Output: softmax probabilities over 4 regimes

    Architecture: [9, 32, 32, 4]
    """

    def __init__(
        self,
        layer_sizes: list = None,
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        n_regimes: int = 4,
    ):
        """
        Initialize TransitionKAN.

        Args:
            layer_sizes: Custom layer sizes (default: [9, 32, 32, 4])
            grid_size: B-spline grid size
            spline_order: B-spline order
            dropout: Dropout rate
            residual: Use residual connections
            n_regimes: Number of regimes (default 4)
        """
        super().__init__()

        self.n_regimes = n_regimes

        if layer_sizes is None:
            layer_sizes = [9, 32, 32, n_regimes]

        config = KANLayerConfig(
            grid_size=grid_size,
            spline_order=spline_order,
            grid_range=(-1.0, 1.0),
            base_activation="silu",
            grid_eps=0.02,
        )

        self.kan_network = KANNetwork(
            layer_sizes=layer_sizes,
            config=config,
            dropout=dropout,
            residual=residual,
        )

    def forward(self, regime: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            regime: Tensor of shape (batch, 8) - normalized regime features
            horizon: Tensor of shape (batch, 1) - T/252 (trading days)

        Returns:
            Softmax probabilities of shape (batch, n_regimes)
        """
        # Concatenate inputs
        x = torch.cat([regime, horizon], dim=1)

        # Forward through KAN
        logits = self.kan_network(x)  # (batch, n_regimes)

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=1)

        return probs

    def get_logits(self, regime: torch.Tensor, horizon: torch.Tensor) -> torch.Tensor:
        """
        Get raw logits before softmax.

        Args:
            regime: Tensor of shape (batch, 8)
            horizon: Tensor of shape (batch, 1)

        Returns:
            Logits of shape (batch, n_regimes)
        """
        x = torch.cat([regime, horizon], dim=1)
        return self.kan_network(x)

    def regularization_loss(self, lambda_l1: float = 1e-4) -> torch.Tensor:
        """Compute L1 regularization loss."""
        return self.kan_network.regularization_loss(lambda_l1)
