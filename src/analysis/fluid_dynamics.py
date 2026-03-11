"""
Fluid Dynamics Feature Integration for Market Analysis.

This module bridges the physics module with the existing analysis pipeline,
providing fluid dynamics features that complement geometric features.

Integration points:
- compute_silver_features(): Add shock indicators and PDE-derived features
- geometric_regime_classification(): Enhance with fluid dynamics signals
- Markov regime models: Add fluid dynamics as exogenous variables

Key Features:
- Shock formation index (Burgers equation analysis)
- Momentum decay rate (viscous dissipation)
- Flow divergence (multi-asset dynamics)
- Macro transmission index (advection-diffusion)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class FluidDynamicsConfig:
    """Configuration for fluid dynamics feature computation."""
    # Windows
    fast_window: int = 5
    medium_window: int = 20
    slow_window: int = 60

    # Physical parameters
    base_viscosity: float = 0.01
    base_diffusion: float = 0.1

    # Shock detection
    shock_threshold: float = 2.0


def compute_momentum_field(
    returns: pd.DataFrame,
    windows: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """
    Compute momentum field at multiple scales.

    Momentum = cumulative returns over window (rolling sum).
    This is the "velocity field" in fluid dynamics terms.
    """
    momentum = pd.DataFrame(index=returns.index)

    for w in windows:
        mom = returns.rolling(w).sum()
        momentum[f"momentum_{w}d"] = mom if isinstance(mom, pd.Series) else mom.mean(axis=1)

    return momentum


def compute_momentum_gradient(
    momentum: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Compute momentum gradient (du/dt).

    This is the rate of change of the momentum field.
    """
    return momentum.diff(window) / window


def compute_momentum_acceleration(
    momentum: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Compute momentum acceleration (d²u/dt²).

    Second derivative of momentum - curvature of the momentum curve.
    """
    grad = compute_momentum_gradient(momentum, window)
    return grad.diff(window) / window


def compute_effective_viscosity(
    volatility: pd.Series,
    mst_stress: Optional[pd.Series] = None,
    base_viscosity: float = 0.01,
) -> pd.Series:
    """
    Compute market-adaptive viscosity coefficient.

    Higher viscosity = more friction = faster momentum decay.

    Encoding:
    - High volatility → high viscosity (turbulent, dissipative)
    - High MST stress → high viscosity (fragmented, frictional)
    """
    vol_norm = volatility / volatility.rolling(252).mean()

    if mst_stress is not None:
        stress_norm = mst_stress / mst_stress.rolling(252).mean()
        viscosity = base_viscosity * (1 + 0.5 * vol_norm + 0.3 * stress_norm)
    else:
        viscosity = base_viscosity * (1 + 0.5 * vol_norm)

    return viscosity.clip(lower=1e-4, upper=1.0)


def compute_shock_formation_index(
    momentum: pd.Series,
    viscosity: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute shock formation index from Burgers equation analysis.

    SFI = |u · du/dt| / (ν · |d²u/dt²| + ε)

    High SFI indicates:
    - Nonlinear advection dominates viscous diffusion
    - Momentum is self-reinforcing
    - Shock (discontinuity) is likely to form
    """
    # Momentum derivatives
    du_dt = momentum.diff()
    d2u_dt2 = du_dt.diff()

    # Nonlinear term: u * du/dt
    nonlinear = (momentum * du_dt).abs()

    # Diffusive term: ν * d²u/dt²
    diffusive = (viscosity * d2u_dt2).abs() + 1e-8

    # Shock formation index
    sfi = nonlinear / diffusive

    # Rolling average for smoothing
    sfi_smooth = sfi.rolling(window).mean()

    return sfi_smooth


def compute_momentum_decay_rate(
    momentum: pd.Series,
    viscosity: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Compute momentum decay rate (how fast momentum dissipates).

    Based on diffusion term of Burgers equation: ν · d²u/dt²

    High decay rate = momentum quickly reverting
    Low decay rate = momentum persisting (trending)
    """
    d2u_dt2 = momentum.diff().diff()
    decay_rate = viscosity * d2u_dt2.abs()
    return decay_rate.rolling(window).mean()


def compute_flow_divergence(
    returns_panel: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Compute flow divergence across assets.

    Divergence measures whether capital is flowing into or out of the asset space.
    - Positive divergence: Capital inflow, coordinated buying
    - Negative divergence: Capital outflow, coordinated selling
    - Near-zero: Balanced flow, rotation between assets
    """
    # Mean return = average flow direction
    mean_ret = returns_panel.mean(axis=1)

    # Variance of returns = flow dispersion
    var_ret = returns_panel.var(axis=1)

    # Divergence = mean adjusted by dispersion
    divergence = mean_ret / (np.sqrt(var_ret) + 1e-8)

    return divergence.rolling(window).mean()


def compute_macro_transmission(
    asset_returns: pd.Series,
    macro_returns: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Compute macro-to-asset transmission coefficients.

    Based on advection-diffusion: how macro "wind" carries asset returns.

    Returns:
        - transmission_coef: Rolling regression coefficient
        - transmission_lag: Optimal lag for transmission
        - transmission_strength: R² of relationship
    """
    results = pd.DataFrame(index=asset_returns.index)

    # Rolling covariance and variance
    cov = asset_returns.rolling(window).cov(macro_returns)
    var_macro = macro_returns.rolling(window).var()

    # Transmission coefficient (like advection velocity)
    results["transmission_coef"] = cov / (var_macro + 1e-8)

    # Correlation strength
    std_asset = asset_returns.rolling(window).std()
    std_macro = macro_returns.rolling(window).std()
    results["transmission_strength"] = (cov / (std_asset * std_macro + 1e-8)).abs()

    return results


def compute_fluid_dynamics_features(
    close_panel: pd.DataFrame,
    *,
    silver_symbol: str,
    gold_symbol: Optional[str] = None,
    dxy_symbol: Optional[str] = None,
    geometric_features: Optional[pd.DataFrame] = None,
    config: FluidDynamicsConfig = FluidDynamicsConfig(),
) -> pd.DataFrame:
    """
    Compute comprehensive fluid dynamics features for market analysis.

    This is the main entry point for integrating fluid dynamics
    with the existing silver analysis pipeline.

    Args:
        close_panel: Price panel (date index, symbol columns)
        silver_symbol: Symbol for silver
        gold_symbol: Symbol for gold (for cross-asset analysis)
        dxy_symbol: Symbol for DXY (macro transmission)
        geometric_features: Existing geometric features (Ricci, MST)
        config: Configuration for computation

    Returns:
        DataFrame with fluid dynamics features:
        - shock_formation_index: Shock likelihood indicator
        - momentum_decay_rate: How fast momentum dissipates
        - flow_divergence: Capital flow direction
        - effective_viscosity: Market friction coefficient
        - macro_transmission_*: Macro-to-silver transmission
    """
    from src.analysis.features import compute_log_returns

    out = pd.DataFrame(index=close_panel.index)

    # Compute returns
    returns = close_panel.apply(compute_log_returns, axis=0)
    silver_returns = returns[silver_symbol] if silver_symbol in returns.columns else None

    if silver_returns is None:
        return out

    # Momentum at multiple scales
    momentum = compute_momentum_field(
        returns[[silver_symbol]],
        windows=[config.fast_window, config.medium_window, config.slow_window],
    )
    for col in momentum.columns:
        out[f"silver_{col}"] = momentum[col]

    # Primary momentum for subsequent computations
    mom_20d = momentum.get(f"momentum_{config.medium_window}d", silver_returns.rolling(20).sum())

    # Volatility for viscosity
    volatility = silver_returns.rolling(20).std() * np.sqrt(252)
    out["realized_vol_fluid"] = volatility

    # MST stress from geometric features (if available)
    mst_stress = None
    if geometric_features is not None:
        if "mst_stress_core_60d" in geometric_features.columns:
            mst_stress = geometric_features["mst_stress_core_60d"]
        elif "mst_stress_60d" in geometric_features.columns:
            mst_stress = geometric_features["mst_stress_60d"]

    # Effective viscosity
    viscosity = compute_effective_viscosity(
        volatility,
        mst_stress=mst_stress,
        base_viscosity=config.base_viscosity,
    )
    out["effective_viscosity"] = viscosity

    # Shock formation index (key PDE-derived feature)
    sfi = compute_shock_formation_index(mom_20d, viscosity, config.medium_window)
    out["shock_formation_index"] = sfi

    # Z-scored version for thresholding
    sfi_mean = sfi.rolling(252).mean()
    sfi_std = sfi.rolling(252).std()
    out["shock_formation_index_z"] = (sfi - sfi_mean) / (sfi_std + 1e-8)

    # Momentum decay rate
    decay_rate = compute_momentum_decay_rate(mom_20d, viscosity, config.medium_window)
    out["momentum_decay_rate"] = decay_rate

    # Flow divergence (multi-asset)
    valid_returns = returns.dropna(axis=1, how="all")
    if valid_returns.shape[1] >= 2:
        divergence = compute_flow_divergence(valid_returns, config.medium_window)
        out["flow_divergence"] = divergence

    # Momentum gradient and acceleration
    out["momentum_gradient_5d"] = compute_momentum_gradient(mom_20d, config.fast_window)
    out["momentum_gradient_20d"] = compute_momentum_gradient(mom_20d, config.medium_window)
    out["momentum_acceleration"] = compute_momentum_acceleration(mom_20d, config.fast_window)

    # Macro transmission (if DXY available)
    if dxy_symbol and dxy_symbol in returns.columns:
        dxy_returns = returns[dxy_symbol]
        transmission = compute_macro_transmission(
            silver_returns, dxy_returns, config.medium_window
        )
        out["dxy_transmission_coef"] = transmission["transmission_coef"]
        out["dxy_transmission_strength"] = transmission["transmission_strength"]

    # Cross-asset analysis with gold (if available)
    if gold_symbol and gold_symbol in returns.columns:
        gold_returns = returns[gold_symbol]
        gold_mom = gold_returns.rolling(config.medium_window).sum()

        # Silver-gold momentum divergence
        out["silver_gold_momentum_div"] = mom_20d - gold_mom

        # Correlation in momentum space
        out["momentum_corr_silver_gold"] = mom_20d.rolling(config.slow_window).corr(gold_mom)

    return out


def integrate_with_geometric_features(
    geometric_features: pd.DataFrame,
    fluid_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Integrate fluid dynamics features with existing geometric features.

    Combines:
    - Static geometry (Ricci curvature, MST stress)
    - Dynamic physics (shock formation, momentum decay)

    Avoids duplicate columns and ensures index alignment.
    """
    # Align indices
    common_idx = geometric_features.index.intersection(fluid_features.index)

    combined = geometric_features.loc[common_idx].copy()

    # Add fluid features (avoid duplicates)
    for col in fluid_features.columns:
        if col not in combined.columns:
            combined[col] = fluid_features.loc[common_idx, col]

    return combined


def fluid_regime_signals(
    features: pd.DataFrame,
    shock_threshold: float = 2.0,
    decay_threshold: float = 0.8,
) -> pd.Series:
    """
    Generate regime signals from fluid dynamics features.

    Complements geometric_regime_classification() with physics-based signals.

    Signals:
    - SHOCK_IMMINENT: High shock formation index
    - MOMENTUM_PERSISTING: Low decay rate, trending
    - MOMENTUM_DECAYING: High decay rate, mean reverting
    - NORMAL: Balanced dynamics
    """
    signals = pd.Series(index=features.index, dtype="object")

    # Get required columns with fallbacks
    sfi_z_col = "shock_formation_index_z"
    decay_col = "momentum_decay_rate"

    if sfi_z_col not in features.columns or decay_col not in features.columns:
        signals[:] = "UNKNOWN"
        return signals

    sfi_z = features[sfi_z_col]
    decay = features[decay_col]

    # Normalize decay rate
    decay_median = decay.rolling(252).median()
    decay_relative = decay / (decay_median + 1e-8)

    # Classify
    signals[:] = "NORMAL"
    signals[sfi_z > shock_threshold] = "SHOCK_IMMINENT"
    signals[(sfi_z <= shock_threshold) & (decay_relative < decay_threshold)] = "MOMENTUM_PERSISTING"
    signals[(sfi_z <= shock_threshold) & (decay_relative > 1.0 / decay_threshold)] = "MOMENTUM_DECAYING"

    return signals
