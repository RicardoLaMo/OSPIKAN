"""
Separable Physics-Informed Kolmogorov-Arnold Network (SPIKAN) for Market Dynamics.

SPIKAN exploits the separability structure of the Kolmogorov-Arnold representation
to decompose multivariate functions into sums of products of univariate functions:

    f(x₁, x₂, ..., xₐ) ≈ Σᵢ φᵢ₁(x₁) × φᵢ₂(x₂) × ... × φᵢₐ(xₐ)

This reduces computational complexity from O(N^d) to O(d×N), making it practical
for high-dimensional market data with multiple assets and macro features.

For market dynamics, we use three separable branches:
1. Asset branch: Processes return/momentum features per asset
2. Macro branch: Processes macro indicators (DXY, rates, VIX)
3. Geometry branch: Processes Ricci curvature and MST stress features

The branches are combined via outer products and pooling, preserving the
physics-informed structure while enabling efficient computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kan_layers import KANLayer, KANLayerConfig, KANNetwork


@dataclass
class SPIKANConfig:
    """Configuration for SPIKAN architecture."""

    # Input dimensions
    n_assets: int = 6  # Number of assets in portfolio
    n_macro: int = 4  # Number of macro indicators
    n_geometric: int = 4  # Number of geometric features (Ricci, MST)

    # Architecture
    hidden_dim: int = 32  # Hidden dimension per branch
    n_terms: int = 8  # Number of separable terms (Σ in K-A representation)
    output_dim: int = 1  # Output dimension (e.g., next-step momentum)

    # KAN configuration
    grid_size: int = 5
    spline_order: int = 3
    dropout: float = 0.1

    # Physics parameters
    viscosity: float = 0.01  # Default viscosity for Burgers equation
    diffusion: float = 0.1  # Diffusion coefficient


class SeparableBranch(nn.Module):
    """
    A single separable branch processing one input modality.

    Each branch learns K univariate functions (one per separable term)
    that will be combined with other branches via outer product.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        n_terms: int,
        kan_config: Optional[KANLayerConfig] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_terms = n_terms

        # Project input to hidden dimension
        self.input_proj = KANLayer(
            in_features=in_features,
            out_features=hidden_dim,
            config=kan_config,
            dropout=dropout,
        )

        # Generate n_terms outputs (one per separable term)
        self.term_heads = nn.ModuleList([
            KANLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                config=kan_config,
                dropout=0.0,
            )
            for _ in range(n_terms)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_features)

        Returns:
            (batch, n_terms, hidden_dim) - one hidden vector per separable term
        """
        h = self.input_proj(x)  # (batch, hidden_dim)
        terms = [head(h) for head in self.term_heads]  # list of (batch, hidden_dim)
        return torch.stack(terms, dim=1)  # (batch, n_terms, hidden_dim)


class MarketSPIKAN(nn.Module):
    """
    Separable Physics-Informed KAN for Market Fluid Dynamics.

    Architecture:
        1. Three separable branches process different input modalities:
           - Asset branch: Returns, momentum, volatility per asset
           - Macro branch: DXY, rates, VIX, SPX
           - Geometry branch: Ricci curvature, MST stress, GA features

        2. Branches combine via element-wise product (separable structure)

        3. Output predicts next-step field values or PDE residuals

    The separable structure enables:
        - Efficient O(d×N) complexity instead of O(N^d)
        - Interpretable per-modality transformations
        - Natural integration with physics-informed loss
    """

    def __init__(self, config: Optional[SPIKANConfig] = None):
        super().__init__()
        self.config = config or SPIKANConfig()

        kan_config = KANLayerConfig(
            grid_size=self.config.grid_size,
            spline_order=self.config.spline_order,
        )

        # Separable branches for each input modality
        self.asset_branch = SeparableBranch(
            in_features=self.config.n_assets,
            hidden_dim=self.config.hidden_dim,
            n_terms=self.config.n_terms,
            kan_config=kan_config,
            dropout=self.config.dropout,
        )

        self.macro_branch = SeparableBranch(
            in_features=self.config.n_macro,
            hidden_dim=self.config.hidden_dim,
            n_terms=self.config.n_terms,
            kan_config=kan_config,
            dropout=self.config.dropout,
        )

        self.geometry_branch = SeparableBranch(
            in_features=self.config.n_geometric,
            hidden_dim=self.config.hidden_dim,
            n_terms=self.config.n_terms,
            kan_config=kan_config,
            dropout=self.config.dropout,
        )

        # Time embedding (for PDE time dependence)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Combine separable terms
        self.combiner = KANNetwork(
            layer_sizes=[
                self.config.hidden_dim,
                self.config.hidden_dim * 2,
                self.config.hidden_dim,
            ],
            config=kan_config,
            dropout=self.config.dropout,
        )

        # Output projection
        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(
        self,
        assets: torch.Tensor,
        macro: torch.Tensor,
        geometry: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through SPIKAN.

        Args:
            assets: (batch, n_assets) - Asset returns/momentum
            macro: (batch, n_macro) - Macro indicators
            geometry: (batch, n_geometric) - Geometric features (Ricci, MST)
            t: (batch, 1) - Optional time coordinate for PDE

        Returns:
            (batch, output_dim) - Predicted field values
        """
        # Process each branch
        h_assets = self.asset_branch(assets)  # (batch, n_terms, hidden)
        h_macro = self.macro_branch(macro)  # (batch, n_terms, hidden)
        h_geom = self.geometry_branch(geometry)  # (batch, n_terms, hidden)

        # Separable combination: element-wise product across branches
        # This is the key K-A decomposition: f(a,m,g) ≈ Σᵢ φᵢ(a) × ψᵢ(m) × χᵢ(g)
        h_combined = h_assets * h_macro * h_geom  # (batch, n_terms, hidden)

        # Sum over separable terms
        h_summed = h_combined.sum(dim=1)  # (batch, hidden)

        # Add time embedding if provided
        if t is not None:
            t_embed = self.time_embed(t)  # (batch, hidden)
            h_summed = h_summed + t_embed

        # Final processing and output
        h_out = self.combiner(h_summed)  # (batch, hidden)
        return self.output_proj(h_out)  # (batch, output_dim)

    def forward_with_derivatives(
        self,
        assets: torch.Tensor,
        macro: torch.Tensor,
        geometry: torch.Tensor,
        t: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with automatic differentiation for PDE residuals.

        Supports both scalar (output_dim=1) and multi-asset (output_dim>1) outputs.
        For multi-asset case, computes full Jacobian and Hessian diagonals.

        Returns dict with:
            - u: Output field value (batch, output_dim)
            - u_t: Time derivative (batch, output_dim)
            - u_x: Jacobian w.r.t. assets (batch, output_dim, n_assets)
            - u_xx: Hessian diagonal w.r.t. assets (batch, output_dim, n_assets)
        """
        batch_size = assets.shape[0]
        n_assets = assets.shape[1]
        output_dim = self.config.output_dim

        # Enable gradients for differentiation
        assets = assets.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)

        # Forward pass
        u = self.forward(assets, macro, geometry, t)  # (batch, output_dim)

        # For scalar output, use simple gradient computation
        if output_dim == 1:
            ones = torch.ones_like(u)

            # Time derivative: du/dt
            u_t = torch.autograd.grad(
                outputs=u, inputs=t,
                grad_outputs=ones,
                create_graph=True, retain_graph=True
            )[0]

            # Spatial gradient: du/dx (Jacobian for scalar is just gradient)
            u_x = torch.autograd.grad(
                outputs=u, inputs=assets,
                grad_outputs=ones,
                create_graph=True, retain_graph=True
            )[0]  # (batch, n_assets)

            # Second spatial derivative (Laplacian diagonal)
            # Compute d²u/dx_i² for each asset dimension
            u_xx = torch.zeros_like(u_x)
            for i in range(n_assets):
                grad_i = u_x[:, i:i+1]  # (batch, 1)
                u_xx_i = torch.autograd.grad(
                    outputs=grad_i, inputs=assets,
                    grad_outputs=torch.ones_like(grad_i),
                    create_graph=True, retain_graph=True
                )[0][:, i:i+1]  # (batch, 1) - diagonal element
                u_xx[:, i:i+1] = u_xx_i

            # Reshape for consistency: (batch, 1, n_assets)
            u_x = u_x.unsqueeze(1)
            u_xx = u_xx.unsqueeze(1)

        else:
            # Multi-output case: compute full Jacobian
            # u_t: (batch, output_dim)
            u_t = torch.zeros(batch_size, output_dim, device=u.device, dtype=u.dtype)

            # u_x: Jacobian (batch, output_dim, n_assets)
            u_x = torch.zeros(batch_size, output_dim, n_assets, device=u.device, dtype=u.dtype)

            # u_xx: Hessian diagonal (batch, output_dim, n_assets)
            u_xx = torch.zeros(batch_size, output_dim, n_assets, device=u.device, dtype=u.dtype)

            # Compute per-output derivatives
            for j in range(output_dim):
                u_j = u[:, j:j+1]  # (batch, 1)
                ones_j = torch.ones_like(u_j)

                # Time derivative for output j
                u_t_j = torch.autograd.grad(
                    outputs=u_j, inputs=t,
                    grad_outputs=ones_j,
                    create_graph=True, retain_graph=True
                )[0]
                u_t[:, j:j+1] = u_t_j

                # Spatial gradient for output j: du_j/dx
                u_x_j = torch.autograd.grad(
                    outputs=u_j, inputs=assets,
                    grad_outputs=ones_j,
                    create_graph=True, retain_graph=True
                )[0]  # (batch, n_assets)
                u_x[:, j, :] = u_x_j

                # Hessian diagonal for output j: d²u_j/dx_i²
                for i in range(n_assets):
                    grad_ji = u_x_j[:, i:i+1]  # (batch, 1)
                    u_xx_ji = torch.autograd.grad(
                        outputs=grad_ji, inputs=assets,
                        grad_outputs=torch.ones_like(grad_ji),
                        create_graph=True, retain_graph=True,
                        allow_unused=True
                    )[0]
                    if u_xx_ji is not None:
                        u_xx[:, j, i] = u_xx_ji[:, i]

        return {
            "u": u,
            "u_t": u_t,
            "u_x": u_x,
            "u_xx": u_xx,
        }

    def regularization_loss(self, lambda_l1: float = 1e-4) -> torch.Tensor:
        """Total regularization loss for all branches including term_heads."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)

        # Input projections
        total = total + self.asset_branch.input_proj.regularization_loss(lambda_l1)
        total = total + self.macro_branch.input_proj.regularization_loss(lambda_l1)
        total = total + self.geometry_branch.input_proj.regularization_loss(lambda_l1)

        # Term heads for each branch
        for branch in [self.asset_branch, self.macro_branch, self.geometry_branch]:
            for term_head in branch.term_heads:
                total = total + term_head.regularization_loss(lambda_l1)

        # Combiner network
        total = total + self.combiner.regularization_loss(lambda_l1)

        return total


class MultiscaleSPIKAN(nn.Module):
    """
    Multi-scale SPIKAN for capturing dynamics at different time horizons.

    Uses parallel SPIKAN branches at different scales (e.g., 5d, 20d, 60d)
    to capture both fast momentum dynamics and slow regime shifts.
    """

    def __init__(
        self,
        base_config: SPIKANConfig,
        scales: List[int] = [5, 20, 60],
    ):
        super().__init__()
        self.scales = scales

        # One SPIKAN per scale
        self.scale_networks = nn.ModuleDict({
            f"scale_{s}": MarketSPIKAN(base_config)
            for s in scales
        })

        # Cross-scale attention for combining
        hidden_dim = base_config.hidden_dim
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=base_config.output_dim,
            num_heads=1,
            batch_first=True,
        )

        # Final output
        self.output_proj = nn.Linear(
            base_config.output_dim * len(scales),
            base_config.output_dim,
        )

    def forward(
        self,
        assets: torch.Tensor,
        macro: torch.Tensor,
        geometry: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multi-scale SPIKAN.

        Each scale processes the same inputs but learns different dynamics.
        """
        # Get output from each scale
        scale_outputs = []
        for scale in self.scales:
            out = self.scale_networks[f"scale_{scale}"](assets, macro, geometry, t)
            scale_outputs.append(out)

        # Stack for attention: (batch, n_scales, output_dim)
        stacked = torch.stack(scale_outputs, dim=1)

        # Self-attention across scales
        attended, _ = self.cross_scale_attn(stacked, stacked, stacked)

        # Concatenate and project
        concat = attended.reshape(attended.shape[0], -1)
        return self.output_proj(concat)


def create_market_spikan(
    n_assets: int = 6,
    n_macro: int = 4,
    include_geometry: bool = True,
    multiscale: bool = False,
) -> nn.Module:
    """
    Factory function to create SPIKAN for market dynamics.

    Args:
        n_assets: Number of assets in portfolio
        n_macro: Number of macro indicators
        include_geometry: Whether to include geometric features (Ricci, MST)
        multiscale: Whether to use multi-scale architecture

    Returns:
        SPIKAN model configured for market data
    """
    config = SPIKANConfig(
        n_assets=n_assets,
        n_macro=n_macro,
        n_geometric=4 if include_geometry else 1,  # At least 1 for dummy
        hidden_dim=32,
        n_terms=8,
        output_dim=n_assets,  # Predict next-step for all assets
    )

    if multiscale:
        return MultiscaleSPIKAN(config, scales=[5, 20, 60])
    else:
        return MarketSPIKAN(config)
