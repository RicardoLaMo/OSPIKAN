"""
Covariance Matrix KAN Network.

Maps regime -> Cholesky factor L such that Cov = L @ L.T is a 6x6 PSD matrix.
"""

import torch
import torch.nn as nn
from src.physics.kan_layers import KANNetwork, KANLayerConfig


class CovarianceKAN(nn.Module):
    """
    KAN network for covariance matrix.

    Input: regime[8] dimensions
    Output: Cholesky factor L of shape (6, 6) lower triangular
            - 21 unique elements (6 + 5 + 4 + 3 + 2 + 1)
            - Diagonal elements via softplus to ensure positive

    Architecture: [8, 32, 32, 21]
    """

    def __init__(
        self,
        layer_sizes: list = None,
        grid_size: int = 8,
        spline_order: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        n_assets: int = 6,
    ):
        """
        Initialize CovarianceKAN.

        Args:
            layer_sizes: Custom layer sizes (default: [8, 32, 32, 21])
            grid_size: B-spline grid size
            spline_order: B-spline order
            dropout: Dropout rate
            residual: Use residual connections
            n_assets: Number of assets (default 6: silver, gold, dxy, ...)
        """
        super().__init__()

        self.n_assets = n_assets
        # Cholesky matrix has n*(n+1)/2 unique elements
        n_cholesky = n_assets * (n_assets + 1) // 2

        if layer_sizes is None:
            layer_sizes = [8, 32, 32, n_cholesky]

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

    def forward(self, regime: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            regime: Tensor of shape (batch, 8) - normalized regime features

        Returns:
            Covariance matrix of shape (batch, n_assets, n_assets)
        """
        batch_size = regime.shape[0]
        device = regime.device

        # Forward through KAN to get Cholesky elements
        cholesky_flat = self.kan_network(regime)  # (batch, n_cholesky)

        # Reconstruct Cholesky matrix (lower triangular)
        L = torch.zeros(batch_size, self.n_assets, self.n_assets, device=device)

        idx = 0
        for i in range(self.n_assets):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: softplus to ensure positive
                    L[:, i, j] = torch.nn.functional.softplus(cholesky_flat[:, idx])
                else:
                    # Off-diagonal
                    L[:, i, j] = cholesky_flat[:, idx]
                idx += 1

        # Compute covariance: Cov = L @ L.T
        cov = torch.bmm(L, L.transpose(1, 2))

        return cov

    def get_cholesky(self, regime: torch.Tensor) -> torch.Tensor:
        """
        Get Cholesky factor L.

        Args:
            regime: Tensor of shape (batch, 8)

        Returns:
            Cholesky factor L of shape (batch, n_assets, n_assets)
        """
        batch_size = regime.shape[0]
        device = regime.device

        cholesky_flat = self.kan_network(regime)

        L = torch.zeros(batch_size, self.n_assets, self.n_assets, device=device)

        idx = 0
        for i in range(self.n_assets):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = torch.nn.functional.softplus(cholesky_flat[:, idx])
                else:
                    L[:, i, j] = cholesky_flat[:, idx]
                idx += 1

        return L

    def regularization_loss(self, lambda_l1: float = 1e-4) -> torch.Tensor:
        """Compute L1 regularization loss."""
        return self.kan_network.regularization_loss(lambda_l1)
