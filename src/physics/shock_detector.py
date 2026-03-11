"""
Shock and Discontinuity Detection for Market Dynamics.

Market shocks are analogous to shocks in fluid dynamics - rapid,
discontinuous changes in the "velocity field" (momentum/returns).

This module provides:
1. Shock indicators based on PDE analysis
2. Gradient-based discontinuity detection
3. Multi-scale shock characterization
4. Integration with SPIKAN predictions

Key Concepts:
- Shock = large gradient in momentum field (∂u/∂x >> ν)
- Rarefaction = spreading/smoothing of momentum (∂²u/∂x² dominant)
- Shock front = boundary between regimes (transition zone)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class ShockConfig:
    """Configuration for shock detection."""
    # Detection thresholds
    gradient_threshold: float = 2.0  # Z-score threshold for shock gradient
    momentum_threshold: float = 2.5  # Z-score threshold for momentum spike
    vol_threshold: float = 1.5  # Volatility multiplier for shock conditions

    # Multi-scale windows
    fast_window: int = 5  # Fast dynamics
    medium_window: int = 20  # Medium dynamics
    slow_window: int = 60  # Slow dynamics

    # Shock characterization
    shock_duration_min: int = 1  # Minimum shock duration (days)
    shock_duration_max: int = 10  # Maximum shock duration


class ShockDetector:
    """
    Multi-scale shock detector for market momentum fields.

    Detects shocks using:
    1. Gradient analysis (momentum acceleration)
    2. Burgers equation stability analysis
    3. Volatility regime breaks
    4. Cross-asset shock propagation
    """

    def __init__(self, config: Optional[ShockConfig] = None):
        self.config = config or ShockConfig()

    def compute_momentum_gradient(
        self,
        momentum: pd.Series,
        window: int = 5,
    ) -> pd.Series:
        """
        Compute momentum gradient (acceleration).

        This is ∂u/∂t in the Burgers equation - rate of momentum change.
        """
        gradient = momentum.diff(window) / window
        return gradient

    def compute_spatial_gradient(
        self,
        momentum_panel: pd.DataFrame,
        window: int = 5,
    ) -> pd.DataFrame:
        """
        Compute cross-asset momentum gradient.

        This approximates ∂u/∂x - momentum dispersion across assets.
        """
        # Rolling standard deviation across assets
        spatial_grad = momentum_panel.rolling(window).std(axis=1)
        return spatial_grad

    def shock_formation_index(
        self,
        momentum: pd.Series,
        volatility: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """
        Compute shock formation index based on Burgers equation analysis.

        Index = |u · ∂u/∂t| / (ν · |∂²u/∂t²| + ε)

        High values indicate:
        - Nonlinear advection dominates viscous diffusion
        - Shock formation is likely
        - Momentum is self-reinforcing
        """
        # Momentum gradient (du/dt)
        du_dt = momentum.diff()

        # Second derivative (d²u/dt²)
        d2u_dt2 = du_dt.diff()

        # Effective viscosity from volatility
        vol_norm = volatility / volatility.rolling(252).mean()
        nu = 0.01 * (1 + vol_norm)  # Higher vol = higher viscosity

        # Nonlinear term: u * du/dt
        nonlinear = (momentum * du_dt).abs()

        # Diffusive term: nu * d²u/dt²
        diffusive = (nu * d2u_dt2).abs() + 1e-8

        # Shock formation index
        sfi = nonlinear / diffusive

        # Smooth and normalize
        sfi = sfi.rolling(window).mean()
        sfi_z = (sfi - sfi.rolling(252).mean()) / (sfi.rolling(252).std() + 1e-8)

        return sfi_z

    def detect_shocks(
        self,
        momentum: pd.Series,
        volatility: pd.Series,
        returns: pd.Series,
        threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        Detect market shocks using multiple indicators.

        Returns DataFrame with:
        - shock_indicator: Binary shock flag
        - shock_intensity: Continuous shock magnitude
        - shock_type: Classification (momentum, volatility, hybrid)
        """
        results = pd.DataFrame(index=momentum.index)

        # 1. Momentum shock: large momentum gradient
        mom_grad = self.compute_momentum_gradient(momentum, self.config.fast_window)
        mom_grad_z = (mom_grad - mom_grad.rolling(252).mean()) / (mom_grad.rolling(252).std() + 1e-8)

        # 2. Volatility shock: sudden vol spike
        vol_change = volatility.pct_change(self.config.fast_window)
        vol_change_z = (vol_change - vol_change.rolling(252).mean()) / (vol_change.rolling(252).std() + 1e-8)

        # 3. Return shock: extreme returns
        ret_z = (returns - returns.rolling(252).mean()) / (returns.rolling(252).std() + 1e-8)

        # 4. Shock formation index
        sfi = self.shock_formation_index(momentum, volatility, self.config.medium_window)

        # Combine indicators
        shock_intensity = (
            0.3 * mom_grad_z.abs()
            + 0.3 * vol_change_z.abs()
            + 0.2 * ret_z.abs()
            + 0.2 * sfi.abs()
        )

        # Binary shock indicator
        shock_indicator = shock_intensity > threshold

        # Classify shock type
        def classify_shock(row):
            if pd.isna(row["shock_indicator"]) or not row["shock_indicator"]:
                return "none"
            if row["mom_shock"] and row["vol_shock"]:
                return "hybrid"
            if row["mom_shock"]:
                return "momentum"
            if row["vol_shock"]:
                return "volatility"
            return "other"

        results["shock_indicator"] = shock_indicator
        results["shock_intensity"] = shock_intensity
        results["mom_shock"] = mom_grad_z.abs() > threshold
        results["vol_shock"] = vol_change_z.abs() > threshold
        results["shock_type"] = results.apply(classify_shock, axis=1)
        results["shock_formation_index"] = sfi

        return results

    def shock_propagation_analysis(
        self,
        returns_panel: pd.DataFrame,
        shock_dates: pd.DatetimeIndex,
        window_before: int = 5,
        window_after: int = 20,
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze how shocks propagate across assets.

        Computes cumulative returns around shock events to understand
        shock transmission dynamics.
        """
        results = {}

        for shock_date in shock_dates:
            if shock_date not in returns_panel.index:
                continue

            # Get position in index
            idx = returns_panel.index.get_loc(shock_date)

            # Extract window around shock
            start = max(0, idx - window_before)
            end = min(len(returns_panel), idx + window_after + 1)

            window_returns = returns_panel.iloc[start:end].copy()

            # Cumulative returns from shock date
            if idx - start < len(window_returns):
                shock_idx = idx - start
                cum_returns = window_returns.cumsum()
                cum_returns = cum_returns - cum_returns.iloc[shock_idx]

                results[str(shock_date.date())] = cum_returns

        return results


class NeuralShockDetector(nn.Module):
    """
    Neural network-based shock detector using SPIKAN features.

    Learns to predict shock probability from:
    - Asset momentum features
    - Geometric features (Ricci curvature, MST stress)
    - Macro indicators
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 3,  # no_shock, mild_shock, severe_shock
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Shock probability head
        self.shock_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        # Shock intensity regression head
        self.intensity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, input_dim)

        Returns:
            shock_probs: Shock class probabilities (batch, num_classes)
            shock_intensity: Shock intensity estimate (batch, 1)
        """
        h = self.encoder(x)
        shock_probs = self.shock_head(h)
        shock_intensity = self.intensity_head(h)
        return shock_probs, shock_intensity


def compute_shock_features(
    momentum: pd.Series,
    volatility: pd.Series,
    ricci_curvature: pd.Series,
    mst_stress: pd.Series,
    config: Optional[ShockConfig] = None,
) -> pd.DataFrame:
    """
    Compute comprehensive shock-related features for analysis.

    These features can be used for:
    - Shock prediction (input to NeuralShockDetector)
    - Regime classification
    - Risk management signals
    """
    config = config or ShockConfig()
    features = pd.DataFrame(index=momentum.index)

    # Momentum dynamics
    features["momentum"] = momentum
    features["momentum_grad_5d"] = momentum.diff(config.fast_window) / config.fast_window
    features["momentum_grad_20d"] = momentum.diff(config.medium_window) / config.medium_window
    features["momentum_accel"] = features["momentum_grad_5d"].diff()

    # Volatility dynamics
    features["volatility"] = volatility
    features["vol_change_5d"] = volatility.pct_change(config.fast_window)
    features["vol_regime"] = volatility / volatility.rolling(60).mean()

    # Geometric indicators
    features["ricci_curvature"] = ricci_curvature
    features["ricci_change"] = ricci_curvature.diff(config.fast_window)
    features["mst_stress"] = mst_stress
    features["mst_stress_change"] = mst_stress.diff(config.fast_window)

    # Shock formation indicators
    detector = ShockDetector(config)
    features["shock_formation_index"] = detector.shock_formation_index(
        momentum, volatility, config.medium_window
    )

    # Cross-feature interactions (nonlinear shock signals)
    features["mom_vol_interaction"] = features["momentum_grad_5d"] * features["vol_change_5d"]
    features["ricci_stress_divergence"] = (
        (ricci_curvature - ricci_curvature.rolling(60).mean())
        / (ricci_curvature.rolling(60).std() + 1e-8)
        - (mst_stress - mst_stress.rolling(60).mean())
        / (mst_stress.rolling(60).std() + 1e-8)
    )

    return features


def label_historical_shocks(
    returns: pd.Series,
    threshold_mild: float = 2.0,
    threshold_severe: float = 3.0,
    lookahead: int = 5,
) -> pd.Series:
    """
    Label historical data with shock classes for supervised learning.

    Labels:
    - 0: No shock (next lookahead days normal)
    - 1: Mild shock (return z-score crosses mild threshold)
    - 2: Severe shock (return z-score crosses severe threshold)
    """
    # Z-score of returns
    ret_z = (returns - returns.rolling(252).mean()) / (returns.rolling(252).std() + 1e-8)

    # Forward-looking max absolute z-score
    forward_max_z = ret_z.rolling(lookahead).max().shift(-lookahead)
    forward_min_z = ret_z.rolling(lookahead).min().shift(-lookahead)
    forward_extreme = pd.concat([forward_max_z.abs(), forward_min_z.abs()], axis=1).max(axis=1)

    # Classify
    labels = pd.Series(0, index=returns.index)
    labels[forward_extreme > threshold_mild] = 1
    labels[forward_extreme > threshold_severe] = 2

    return labels
