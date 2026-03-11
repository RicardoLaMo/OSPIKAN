"""
Market-Specific Partial Differential Equations for Fluid Dynamics.

This module defines PDEs that model market dynamics as fluid systems:

1. Burgers Equation: Momentum transport with viscous friction
   ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
   Models: Momentum cascades, shock formation (flash crashes)

2. Advection-Diffusion: Information flow with volatility diffusion
   ∂u/∂t + v·∇u = D·∇²u + S(x,t)
   Models: Price propagation, macro transmission to assets

3. Mean Reversion with Stochastic Forcing
   ∂u/∂t = -α(u - μ) + σ·∂²u/∂x² + F(t)
   Models: Price mean reversion, equilibrium dynamics

Each PDE is implemented with:
- Residual computation for physics-informed loss
- Physical parameter encoding from market data
- Shock/discontinuity handling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class PDEConfig:
    """Configuration for market PDE models."""
    # Physical parameters
    viscosity: float = 0.01  # ν: Market friction coefficient
    diffusion: float = 0.1  # D: Volatility-driven diffusion
    mean_reversion: float = 0.1  # α: Mean reversion speed
    equilibrium: float = 0.0  # μ: Long-term equilibrium

    # Loss weights
    lambda_pde: float = 1.0  # Weight for PDE residual loss
    lambda_data: float = 1.0  # Weight for data fitting loss
    lambda_bc: float = 0.1  # Weight for boundary conditions
    lambda_ic: float = 0.1  # Weight for initial conditions
    lambda_reg: float = 1e-4  # Regularization weight


class MarketPDE(ABC):
    """Abstract base class for market PDEs."""

    @abstractmethod
    def residual(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute PDE residual (should be zero for exact solution)."""
        pass

    @abstractmethod
    def encode_physical_params(self, market_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encode market data into physical parameters."""
        pass


class BurgersEquation(MarketPDE):
    """
    Burgers Equation for Market Momentum Dynamics.

    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

    Physical Interpretation:
    - u: Price momentum field (rolling returns)
    - ν: Market friction (liquidity, transaction costs)
    - u·∂u/∂x: Nonlinear advection (momentum feeds on itself)
    - ν·∂²u/∂x²: Viscous diffusion (friction dissipates momentum)

    Key Properties:
    - Can form shocks (discontinuities) when viscosity is low
    - Shock formation → flash crash / momentum collapse
    - High viscosity → smooth momentum decay
    """

    def __init__(self, config: Optional[PDEConfig] = None):
        self.config = config or PDEConfig()

    def residual(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        viscosity: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute Burgers equation residual.

        R = ∂u/∂t + u·∂u/∂x - ν·∂²u/∂x²

        For a true solution, R → 0.
        """
        nu = viscosity if viscosity is not None else self.config.viscosity

        # Handle multi-dimensional u_x (sum across asset dimension if needed)
        if u_x.dim() > 1 and u_x.shape[-1] > 1:
            u_x_scalar = u_x.mean(dim=-1, keepdim=True)
            u_xx_scalar = u_xx.mean(dim=-1, keepdim=True)
        else:
            u_x_scalar = u_x
            u_xx_scalar = u_xx

        # Burgers residual: u_t + u*u_x - nu*u_xx = 0
        residual = u_t + u * u_x_scalar - nu * u_xx_scalar

        return residual

    def encode_physical_params(
        self, market_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode market data into Burgers physical parameters.

        Viscosity encoding:
        - High liquidity → high viscosity → smooth dynamics
        - Low liquidity → low viscosity → shock-prone
        - High volatility → higher effective viscosity (faster diffusion)
        """
        params = {}

        # Viscosity from volatility and volume
        if "realized_vol" in market_data:
            vol = market_data["realized_vol"]
            # Higher vol → higher friction (faster momentum decay)
            params["viscosity"] = self.config.viscosity * (1.0 + vol)
        else:
            params["viscosity"] = torch.tensor(self.config.viscosity)

        # Effective momentum field
        if "momentum" in market_data:
            params["u_init"] = market_data["momentum"]

        return params

    def shock_formation_indicator(
        self,
        u: torch.Tensor,
        u_x: torch.Tensor,
        viscosity: float = 0.01,
    ) -> torch.Tensor:
        """
        Estimate shock formation likelihood.

        Shocks form when the nonlinear term dominates viscous diffusion.
        Shock indicator: |u * u_x| / (ν * max(|u_x|, ε))

        High values indicate imminent shock formation (flash crash risk).
        """
        eps = 1e-6
        if u_x.dim() > 1:
            u_x_mag = u_x.abs().max(dim=-1, keepdim=True)[0]
        else:
            u_x_mag = u_x.abs()

        nonlinear = (u * u_x.mean(dim=-1, keepdim=True) if u_x.dim() > 1 else u * u_x).abs()
        diffusive = viscosity * u_x_mag.clamp(min=eps)

        return nonlinear / diffusive


class AdvectionDiffusion(MarketPDE):
    """
    Advection-Diffusion Equation for Information Flow.

    ∂u/∂t + v·∇u = D·∇²u + S(x,t)

    Physical Interpretation:
    - u: Asset price/return field
    - v: "Wind" velocity (macro momentum: DXY, rates, SPX)
    - D: Diffusion coefficient (volatility-driven spreading)
    - S: Source term (news shocks, Fed decisions)

    This models how macro information propagates to individual assets.
    """

    def __init__(self, config: Optional[PDEConfig] = None):
        self.config = config or PDEConfig()

    def residual(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        velocity: Optional[torch.Tensor] = None,
        diffusion: Optional[torch.Tensor] = None,
        source: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute advection-diffusion residual.

        R = ∂u/∂t + v·∇u - D·∇²u - S
        """
        v = velocity if velocity is not None else torch.zeros_like(u)
        D = diffusion if diffusion is not None else self.config.diffusion
        S = source if source is not None else torch.zeros_like(u)

        # Advection term: v dot grad(u)
        if u_x.dim() > 1 and v.dim() > 1:
            advection = (v * u_x).sum(dim=-1, keepdim=True)
        else:
            advection = v * u_x

        # Diffusion term
        if u_xx.dim() > 1:
            diffusion_term = D * u_xx.sum(dim=-1, keepdim=True)
        else:
            diffusion_term = D * u_xx

        # Advection-diffusion residual
        residual = u_t + advection - diffusion_term - S

        return residual

    def encode_physical_params(
        self, market_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode market data into advection-diffusion parameters.

        Velocity encoding:
        - DXY momentum → dollar strength driving asset flows
        - Rate changes → monetary policy velocity
        - SPX momentum → risk-on/off flow direction
        """
        params = {}

        # Build velocity field from macro indicators
        velocity_components = []
        for key in ["dxy_momentum", "rate_change", "spx_momentum"]:
            if key in market_data:
                velocity_components.append(market_data[key])

        if velocity_components:
            params["velocity"] = torch.stack(velocity_components, dim=-1).mean(dim=-1, keepdim=True)
        else:
            params["velocity"] = torch.zeros(1)

        # Diffusion from volatility
        if "realized_vol" in market_data:
            # Higher vol → faster diffusion (information spreads quickly)
            params["diffusion"] = self.config.diffusion * (1.0 + market_data["realized_vol"])
        else:
            params["diffusion"] = torch.tensor(self.config.diffusion)

        return params


class MeanReversionPDE(MarketPDE):
    """
    Mean Reversion PDE with Stochastic Forcing.

    ∂u/∂t = -α(u - μ) + σ·∂²u/∂x² + F(t)

    Physical Interpretation:
    - u: Asset price deviation from equilibrium
    - α: Mean reversion speed
    - μ: Equilibrium level (fair value)
    - σ: Volatility (diffusion)
    - F(t): External forcing (macro shocks)

    Models how prices revert to fundamental values while being
    perturbed by volatility and external shocks.
    """

    def __init__(self, config: Optional[PDEConfig] = None):
        self.config = config or PDEConfig()

    def residual(
        self,
        u: torch.Tensor,
        u_t: torch.Tensor,
        u_x: torch.Tensor,
        u_xx: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        forcing: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute mean reversion PDE residual.

        R = ∂u/∂t + α(u - μ) - σ·∂²u/∂x² - F
        """
        a = alpha if alpha is not None else self.config.mean_reversion
        m = mu if mu is not None else self.config.equilibrium
        s = sigma if sigma is not None else self.config.diffusion
        F = forcing if forcing is not None else torch.zeros_like(u)

        # Diffusion term
        if u_xx.dim() > 1:
            diffusion = s * u_xx.sum(dim=-1, keepdim=True)
        else:
            diffusion = s * u_xx

        # Mean reversion residual
        residual = u_t + a * (u - m) - diffusion - F

        return residual

    def encode_physical_params(
        self, market_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Encode market data into mean reversion parameters."""
        params = {}

        # Mean reversion speed from market regime
        if "ricci_curvature" in market_data:
            # Higher curvature (tighter structure) → faster mean reversion
            curv = market_data["ricci_curvature"]
            curv_normalized = (curv - curv.mean()) / (curv.std() + 1e-8)
            params["alpha"] = self.config.mean_reversion * (1.0 + 0.1 * curv_normalized)
        else:
            params["alpha"] = torch.tensor(self.config.mean_reversion)

        # Equilibrium from moving average
        if "price_ma" in market_data:
            params["mu"] = market_data["price_ma"]
        else:
            params["mu"] = torch.tensor(self.config.equilibrium)

        return params


def physics_informed_loss(
    model: nn.Module,
    assets: torch.Tensor,
    macro: torch.Tensor,
    geometry: torch.Tensor,
    t: torch.Tensor,
    target: torch.Tensor,
    pde: MarketPDE,
    config: Optional[PDEConfig] = None,
    pde_params: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute physics-informed loss combining data fitting and PDE residual.

    Total Loss = λ_data * L_data + λ_pde * L_pde + λ_reg * L_reg

    Where:
    - L_data = MSE(predicted, target)
    - L_pde = MSE(PDE_residual, 0)
    - L_reg = Regularization on model weights

    Handles both scalar (output_dim=1) and multi-asset outputs.
    For multi-asset, derivatives have shape (batch, output_dim, n_assets).

    Args:
        model: SPIKAN model with forward_with_derivatives method
        assets: Asset features (batch, n_assets)
        macro: Macro features (batch, n_macro)
        geometry: Geometric features (batch, n_geometric)
        t: Time coordinate (batch, 1)
        target: Target values (batch, output_dim)
        pde: PDE instance for residual computation
        config: Loss configuration
        pde_params: Physical parameters for PDE

    Returns:
        Dict with individual and total loss terms
    """
    config = config or PDEConfig()
    pde_params = pde_params or {}

    # Forward pass with derivatives
    derivs = model.forward_with_derivatives(assets, macro, geometry, t)
    u = derivs["u"]  # (batch, output_dim)
    u_t = derivs["u_t"]  # (batch, output_dim) or (batch, 1)
    u_x = derivs["u_x"]  # (batch, output_dim, n_assets) or (batch, 1, n_assets)
    u_xx = derivs["u_xx"]  # (batch, output_dim, n_assets) or (batch, 1, n_assets)

    # Data fitting loss
    data_loss = F.mse_loss(u, target)

    # For PDE residual, we need to reduce spatial dimensions
    # Average over asset dimension for PDE computation
    if u_x.dim() == 3:
        # Multi-output case: (batch, output_dim, n_assets)
        u_x_reduced = u_x.mean(dim=-1)  # (batch, output_dim)
        u_xx_reduced = u_xx.mean(dim=-1)  # (batch, output_dim)
    else:
        u_x_reduced = u_x
        u_xx_reduced = u_xx

    # PDE residual loss - compute per output then average
    residual = pde.residual(u, u_t, u_x_reduced, u_xx_reduced, **pde_params)
    pde_loss = torch.mean(residual ** 2)

    # Regularization loss
    if hasattr(model, "regularization_loss"):
        reg_loss = model.regularization_loss(config.lambda_reg)
    else:
        reg_loss = torch.tensor(0.0, device=u.device)

    # Total loss
    total_loss = (
        config.lambda_data * data_loss
        + config.lambda_pde * pde_loss
        + reg_loss
    )

    return {
        "total": total_loss,
        "data": data_loss,
        "pde": pde_loss,
        "reg": reg_loss,
        "residual_mean": residual.abs().mean(),
        "residual_max": residual.abs().max(),
    }


def adaptive_viscosity(
    momentum: torch.Tensor,
    volatility: torch.Tensor,
    mst_stress: torch.Tensor,
    base_viscosity: float = 0.01,
) -> torch.Tensor:
    """
    Compute adaptive viscosity based on market conditions.

    Higher viscosity when:
    - Volatility is high (more friction)
    - MST stress is high (fragmented market)
    - Momentum is extreme (near reversal)

    Lower viscosity when:
    - Markets are calm and trending
    - Low stress, smooth correlations
    """
    # Normalize inputs
    vol_norm = volatility / (volatility.mean() + 1e-8)
    stress_norm = mst_stress / (mst_stress.mean() + 1e-8)
    mom_abs = momentum.abs() / (momentum.abs().mean() + 1e-8)

    # Combine factors
    viscosity = base_viscosity * (1.0 + 0.5 * vol_norm + 0.3 * stress_norm + 0.2 * mom_abs)

    return viscosity.clamp(min=1e-4, max=1.0)
