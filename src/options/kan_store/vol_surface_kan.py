"""
Volatility Surface KAN Network.

Maps (regime, moneyness, time_to_expiry) -> implied_vol using KAN.
"""

import torch
import torch.nn as nn
from src.physics.kan_layers import KANNetwork, KANLayerConfig


class VolSurfaceKAN(nn.Module):
    """
    KAN network for volatility surface.

    Input: (regime[8], log_moneyness[1], time_to_expiry[1]) = 10 dimensions
    Output: implied_vol (scalar, [0, 2] via 2*sigmoid)

    Architecture: [10, 32, 32, 16, 1]
    """

    def __init__(
        self,
        layer_sizes: list = None,
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        """
        Initialize VolSurfaceKAN.

        Args:
            layer_sizes: Custom layer sizes (default: [10, 32, 32, 16, 1])
            grid_size: B-spline grid size
            spline_order: B-spline order
            dropout: Dropout rate
            residual: Use residual connections
        """
        super().__init__()

        if layer_sizes is None:
            layer_sizes = [10, 32, 32, 16, 1]

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

    def forward(self, regime: torch.Tensor, log_moneyness: torch.Tensor,
                time_to_expiry: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            regime: Tensor of shape (batch, 8) - normalized regime features
            log_moneyness: Tensor of shape (batch, 1) - log(S/K)
            time_to_expiry: Tensor of shape (batch, 1) - T/365

        Returns:
            Implied vol tensor of shape (batch, 1), scaled to [0, 2]
        """
        # Concatenate inputs
        x = torch.cat([regime, log_moneyness, time_to_expiry], dim=1)

        # Forward through KAN
        output = self.kan_network(x)

        # Scale output to [0, 2] via 2*sigmoid
        vol = 2.0 * torch.sigmoid(output)

        return vol

    def regularization_loss(self, lambda_l1: float = 1e-4) -> torch.Tensor:
        """Compute L1 regularization loss."""
        return self.kan_network.regularization_loss(lambda_l1)
