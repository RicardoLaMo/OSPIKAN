#!/usr/bin/env python3
"""
Generate thesis-ready fluid dynamics and shockwave analysis report.

This script produces comprehensive visualizations demonstrating:
1. SPIKAN's shock formation detection capabilities
2. Burgers equation components in market dynamics
3. Momentum decay patterns across geometric regimes
4. Static (Ricci curvature) vs Dynamic (SPIKAN) complementarity
5. Economic interpretation of fluid dynamics variables
6. Cross-asset capital flow analysis
7. Combined regime analysis (geometric + fluid)

Usage:
    python scripts/fluid_dynamics_report.py [--features-path PATH] [--output-dir DIR]
    python scripts/fluid_dynamics_report.py --generate-features  # Generate features with fluid dynamics

Example:
    python scripts/fluid_dynamics_report.py --features-path data/processed/silver_features_*.parquet
    python scripts/fluid_dynamics_report.py --generate-features --config configs/silver_universe.yaml

Author: Thesis Fluid Dynamics Analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Set matplotlib style for thesis-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

# Regime color scheme for consistency
REGIME_COLORS = {
    'STABLE': '#2ecc71',          # Green
    'TRANSITION': '#f39c12',       # Orange
    'STRESS': '#e74c3c',           # Red
    'RECOVERY': '#3498db',         # Blue
    'SHOCK_IMMINENT': '#9b59b6',   # Purple
    'MOMENTUM_PERSISTING': '#1abc9c',  # Teal
    'MOMENTUM_DECAYING': '#e67e22',    # Dark orange
    'NORMAL': '#95a5a6',           # Gray
    'UNKNOWN': '#bdc3c7',          # Light gray
}


def load_or_generate_features(
    features_path: Optional[str] = None,
    config_path: str = "configs/silver_universe.yaml",
    generate: bool = False,
) -> pd.DataFrame:
    """Load existing features or generate new ones with fluid dynamics."""
    if features_path and Path(features_path).exists():
        print(f"Loading features from {features_path}")
        return pd.read_parquet(features_path)

    if generate:
        print("Generating features with fluid dynamics enabled...")
        return _generate_features_with_fluid(config_path)

    # Search for existing feature files
    search_dirs = [
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "reports" / "silver" / "runs",
        PROJECT_ROOT / "reports" / "silver" / "runs_test",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            parquet_files = sorted(search_dir.glob("*features*.parquet"), reverse=True)
            if parquet_files:
                print(f"Found features at {parquet_files[0]}")
                return pd.read_parquet(parquet_files[0])

    raise FileNotFoundError(
        "No feature files found. Run with --generate-features flag or "
        "specify --features-path to an existing parquet file."
    )


def _generate_features_with_fluid(config_path: str) -> pd.DataFrame:
    """Generate features with fluid dynamics enabled."""
    import yaml
    from src.analysis.features import compute_silver_features, FeatureConfig

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load price data (this would need actual data)
    # For now, we'll raise an error with instructions
    raise NotImplementedError(
        "Feature generation requires price data. Please run the silver pipeline first:\n"
        "  python -m src.pipeline.silver_pipeline ingest --config configs/silver_universe.yaml\n"
        "  python -m src.pipeline.silver_pipeline features --config configs/silver_universe.yaml --fluid-dynamics"
    )


def compute_fluid_features_if_missing(features: pd.DataFrame) -> pd.DataFrame:
    """Add fluid dynamics features if not already present."""
    fluid_cols = ['shock_formation_index', 'shock_formation_index_z',
                  'momentum_decay_rate', 'effective_viscosity', 'flow_divergence']

    if any(col in features.columns for col in fluid_cols):
        return features  # Already has fluid features

    print("Computing fluid dynamics features...")
    from src.analysis.fluid_dynamics import compute_fluid_dynamics_features, FluidDynamicsConfig

    # Need price panel - reconstruct from features if possible
    if 'silver_close' not in features.columns:
        print("Warning: Cannot compute fluid features without price data")
        return features

    # Build minimal price panel
    price_cols = [c for c in features.columns if 'close' in c.lower() or 'level' in c.lower()]
    if not price_cols:
        return features

    price_panel = features[price_cols].copy()
    price_panel.columns = [c.replace('_close', '').replace('_level', '') for c in price_cols]

    # Compute fluid features
    config = FluidDynamicsConfig()
    fluid = compute_fluid_dynamics_features(
        price_panel,
        silver_symbol='silver',
        gold_symbol='gold' if 'gold' in price_panel.columns else None,
        dxy_symbol='dxy' if 'dxy' in price_panel.columns else None,
        geometric_features=features,
        config=config,
    )

    # Merge
    for col in fluid.columns:
        if col not in features.columns:
            features[col] = fluid[col]

    return features


def figure1_shock_formation_analysis(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 1: Shock Formation Analysis

    Shows SFI time series with shock warnings overlaid on silver price.
    Demonstrates shock prediction capability.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Silver price with shock warning shading
    ax1 = axes[0]
    price_col = 'silver_close' if 'silver_close' in features.columns else features.columns[0]
    price = features[price_col].dropna()

    ax1.plot(price.index, price.values, 'k-', linewidth=1.2, label='Silver Price')

    # Shade shock warning periods
    if 'shock_formation_index_z' in features.columns:
        sfi_z = features['shock_formation_index_z']
        shock_mask = sfi_z > 2.0

        # Find contiguous shock periods
        shock_start = None
        for i, (date, is_shock) in enumerate(shock_mask.items()):
            if is_shock and shock_start is None:
                shock_start = date
            elif not is_shock and shock_start is not None:
                ax1.axvspan(shock_start, date, alpha=0.3, color=REGIME_COLORS['SHOCK_IMMINENT'],
                           label='Shock Warning' if i < 10 else '')
                shock_start = None

    ax1.set_ylabel('Price ($)')
    ax1.set_title('Silver Price with Shock Formation Warnings')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Shock Formation Index (raw and z-scored)
    ax2 = axes[1]
    if 'shock_formation_index' in features.columns:
        sfi = features['shock_formation_index'].dropna()
        ax2.plot(sfi.index, sfi.values, 'b-', linewidth=0.8, alpha=0.7, label='SFI (raw)')

    if 'shock_formation_index_z' in features.columns:
        sfi_z = features['shock_formation_index_z'].dropna()
        ax2_twin = ax2.twinx()
        ax2_twin.plot(sfi_z.index, sfi_z.values, 'r-', linewidth=1.0, label='SFI (z-score)')
        ax2_twin.axhline(y=2.0, color='r', linestyle='--', alpha=0.5, label='Threshold (z=2)')
        ax2_twin.set_ylabel('Z-Score', color='r')
        ax2_twin.legend(loc='upper right')

    ax2.set_ylabel('SFI (raw)', color='b')
    ax2.set_title('Shock Formation Index: |u · du/dt| / (ν · |d²u/dt²| + ε)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Forward returns by shock warning status
    ax3 = axes[2]
    if 'log_return_1d' in features.columns and 'shock_formation_index_z' in features.columns:
        returns = features['log_return_1d']
        sfi_z = features['shock_formation_index_z']

        # Forward 5-day returns
        fwd_5d = returns.rolling(5).sum().shift(-5)

        shock_warn = sfi_z > 2.0
        no_warn = sfi_z <= 2.0

        warn_returns = fwd_5d[shock_warn].dropna() * 100
        no_warn_returns = fwd_5d[no_warn].dropna() * 100

        if len(warn_returns) > 0 and len(no_warn_returns) > 0:
            ax3.hist(no_warn_returns, bins=50, alpha=0.5, color='gray',
                    label=f'No Warning (n={len(no_warn_returns)}, μ={no_warn_returns.mean():.2f}%)')
            ax3.hist(warn_returns, bins=30, alpha=0.7, color=REGIME_COLORS['SHOCK_IMMINENT'],
                    label=f'Shock Warning (n={len(warn_returns)}, μ={warn_returns.mean():.2f}%)')
            ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
            ax3.set_xlabel('Forward 5-Day Return (%)')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Forward Returns by Shock Warning Status')
            ax3.legend()

    plt.tight_layout()
    fig.savefig(output_dir / 'shock_formation_analysis.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: shock_formation_analysis.png")


def figure2_burgers_equation_components(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 2: Burgers Equation Components

    Visualizes momentum field, viscosity, and SFI decomposition.
    Shows how the PDE captures market dynamics.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Momentum fields at multiple scales
    ax1 = axes[0]
    mom_cols = [c for c in features.columns if 'momentum' in c and 'd' in c and 'decay' not in c]
    for col in mom_cols[:3]:
        if col in features.columns:
            mom = features[col].dropna()
            ax1.plot(mom.index, mom.values, linewidth=0.9, alpha=0.8, label=col)

    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Momentum')
    ax1.set_title('Momentum Field u(t) at Multiple Scales')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Momentum gradient (du/dt) and acceleration (d²u/dt²)
    ax2 = axes[1]
    if 'momentum_gradient_5d' in features.columns:
        grad = features['momentum_gradient_5d'].dropna()
        ax2.plot(grad.index, grad.values, 'b-', linewidth=0.8, alpha=0.8, label='du/dt (5d)')
    if 'momentum_acceleration' in features.columns:
        accel = features['momentum_acceleration'].dropna()
        ax2_twin = ax2.twinx()
        ax2_twin.plot(accel.index, accel.values, 'r-', linewidth=0.8, alpha=0.8, label='d²u/dt²')
        ax2_twin.set_ylabel('Acceleration', color='r')
        ax2_twin.legend(loc='upper right')

    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Gradient', color='b')
    ax2.set_title('Momentum Derivatives: Gradient (du/dt) and Acceleration (d²u/dt²)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Effective viscosity ν(t)
    ax3 = axes[2]
    if 'effective_viscosity' in features.columns:
        visc = features['effective_viscosity'].dropna()
        ax3.fill_between(visc.index, 0, visc.values, alpha=0.4, color='purple')
        ax3.plot(visc.index, visc.values, 'purple', linewidth=1.0, label='ν(t)')
        ax3.set_ylabel('Viscosity ν')
        ax3.set_title('Effective Viscosity: ν = ν₀ · (1 + 0.5·σ/σ̄ + 0.3·stress/stress̄)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Panel 4: Burgers equation balance
    ax4 = axes[3]
    if all(c in features.columns for c in ['shock_formation_index', 'momentum_decay_rate']):
        sfi = features['shock_formation_index'].dropna()
        decay = features['momentum_decay_rate'].dropna()

        common_idx = sfi.index.intersection(decay.index)
        sfi = sfi.loc[common_idx]
        decay = decay.loc[common_idx]

        # Normalize for visualization
        sfi_norm = (sfi - sfi.mean()) / (sfi.std() + 1e-8)
        decay_norm = (decay - decay.mean()) / (decay.std() + 1e-8)

        ax4.fill_between(common_idx, 0, sfi_norm, where=sfi_norm > 0, alpha=0.4,
                        color=REGIME_COLORS['SHOCK_IMMINENT'], label='Nonlinear term (u·∂u/∂x)')
        ax4.fill_between(common_idx, 0, -decay_norm, where=decay_norm > 0, alpha=0.4,
                        color=REGIME_COLORS['MOMENTUM_DECAYING'], label='Diffusive term (ν·∂²u/∂x²)')
        ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('Normalized Magnitude')
        ax4.set_xlabel('Date')
        ax4.set_title('Burgers Equation Balance: ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'burgers_equation_components.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: burgers_equation_components.png")


def figure3_momentum_decay_by_regime(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 3: Momentum Decay by Geometric Regime

    Links fluid dynamics (decay rate) to geometric regimes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get regime column
    regime_col = None
    for col in ['geometric_regime', 'combined_regime', 'regime']:
        if col in features.columns:
            regime_col = col
            break

    if regime_col is None or 'momentum_decay_rate' not in features.columns:
        # Create placeholder
        fig.text(0.5, 0.5, 'Regime or decay rate data not available',
                ha='center', va='center', fontsize=14)
        fig.savefig(output_dir / 'momentum_decay_regimes.png', bbox_inches='tight')
        plt.close(fig)
        return

    regimes = features[regime_col].dropna()
    decay = features['momentum_decay_rate'].dropna()

    common_idx = regimes.index.intersection(decay.index)
    regimes = regimes.loc[common_idx]
    decay = decay.loc[common_idx]

    # Panel 1: Box plot of decay rate by regime
    ax1 = axes[0, 0]
    regime_groups = regimes.unique()
    box_data = [decay[regimes == r].dropna() for r in regime_groups]
    colors = [REGIME_COLORS.get(r, 'gray') for r in regime_groups]

    bp = ax1.boxplot(box_data, labels=regime_groups, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_ylabel('Momentum Decay Rate')
    ax1.set_title('Decay Rate Distribution by Regime')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Time series with regime shading
    ax2 = axes[0, 1]
    ax2.plot(decay.index, decay.values, 'k-', linewidth=0.8, alpha=0.8)

    # Shade by regime
    current_regime = regimes.iloc[0]
    start_idx = regimes.index[0]
    for i, (date, regime) in enumerate(regimes.items()):
        if regime != current_regime or i == len(regimes) - 1:
            color = REGIME_COLORS.get(current_regime, 'gray')
            ax2.axvspan(start_idx, date, alpha=0.2, color=color)
            start_idx = date
            current_regime = regime

    ax2.set_ylabel('Decay Rate')
    ax2.set_title('Momentum Decay Rate with Regime Background')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Decay rate vs returns scatter by regime
    ax3 = axes[1, 0]
    if 'log_return_1d' in features.columns:
        returns = features['log_return_1d'].loc[common_idx]
        for regime in regime_groups:
            mask = regimes == regime
            x = decay[mask].values
            y = returns[mask].values * 100
            ax3.scatter(x, y, alpha=0.3, c=REGIME_COLORS.get(regime, 'gray'),
                       label=regime, s=10)

        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Momentum Decay Rate')
        ax3.set_ylabel('Daily Return (%)')
        ax3.set_title('Decay Rate vs Returns by Regime')
        ax3.legend(loc='upper right', ncol=2, fontsize=8)
        ax3.grid(True, alpha=0.3)

    # Panel 4: Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_data = []
    for regime in regime_groups:
        mask = regimes == regime
        decay_regime = decay[mask]
        stats_data.append([
            regime,
            f"{mask.sum()}",
            f"{decay_regime.mean():.4f}",
            f"{decay_regime.std():.4f}",
            f"{decay_regime.median():.4f}",
        ])

    table = ax4.table(
        cellText=stats_data,
        colLabels=['Regime', 'Count', 'Mean', 'Std', 'Median'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax4.set_title('Momentum Decay Statistics by Regime', pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'momentum_decay_regimes.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: momentum_decay_regimes.png")


def figure4_static_vs_dynamic(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 4: Static (Ricci) vs Dynamic (SPIKAN) Features

    Demonstrates complementary nature of approaches.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Find Ricci curvature column
    ricci_col = None
    for col in ['ricci_mean_core_60d', 'ricci_mean_60d', 'ricci_p10_core_60d']:
        if col in features.columns:
            ricci_col = col
            break

    # Panel 1: Ricci curvature (static geometry)
    ax1 = axes[0]
    if ricci_col:
        ricci = features[ricci_col].dropna()
        ax1.plot(ricci.index, ricci.values, 'b-', linewidth=1.0, alpha=0.8)
        ax1.fill_between(ricci.index, ricci.values, 0, where=ricci.values < 0,
                        alpha=0.3, color='red', label='Negative (stressed)')
        ax1.fill_between(ricci.index, ricci.values, 0, where=ricci.values >= 0,
                        alpha=0.3, color='green', label='Positive (stable)')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Ricci Curvature')
        ax1.set_title('STATIC: Forman-Ricci Curvature (Market Topology)')
        ax1.legend(loc='upper right')
    else:
        ax1.text(0.5, 0.5, 'Ricci curvature data not available',
                ha='center', va='center', transform=ax1.transAxes)
    ax1.grid(True, alpha=0.3)

    # Panel 2: SFI (dynamic physics)
    ax2 = axes[1]
    if 'shock_formation_index_z' in features.columns:
        sfi_z = features['shock_formation_index_z'].dropna()
        ax2.plot(sfi_z.index, sfi_z.values, color=REGIME_COLORS['SHOCK_IMMINENT'],
                linewidth=1.0, alpha=0.8)
        ax2.fill_between(sfi_z.index, sfi_z.values, 2, where=sfi_z.values > 2,
                        alpha=0.5, color=REGIME_COLORS['SHOCK_IMMINENT'], label='Shock warning')
        ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('SFI (z-score)')
        ax2.set_title('DYNAMIC: Shock Formation Index (Burgers PDE)')
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'SFI data not available',
                ha='center', va='center', transform=ax2.transAxes)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Complementarity - scatter plot
    ax3 = axes[2]
    if ricci_col and 'shock_formation_index_z' in features.columns:
        ricci = features[ricci_col]
        sfi_z = features['shock_formation_index_z']

        common_idx = ricci.dropna().index.intersection(sfi_z.dropna().index)
        ricci_c = ricci.loc[common_idx]
        sfi_c = sfi_z.loc[common_idx]

        # Color by forward returns if available
        if 'log_return_1d' in features.columns:
            fwd_ret = features['log_return_1d'].rolling(5).sum().shift(-5).loc[common_idx]
            scatter = ax3.scatter(ricci_c, sfi_c, c=fwd_ret * 100, cmap='RdYlGn',
                                 alpha=0.5, s=5, vmin=-5, vmax=5)
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Forward 5d Return (%)')
        else:
            ax3.scatter(ricci_c, sfi_c, alpha=0.3, s=5, c='blue')

        # Quadrant labels
        ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5)
        ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        ax3.text(ricci_c.min(), 3, 'Stressed + Shock\n(Crisis)', fontsize=9, color='red')
        ax3.text(ricci_c.max() * 0.7, 3, 'Stable + Shock\n(Breakout)', fontsize=9, color='purple')
        ax3.text(ricci_c.min(), 0, 'Stressed + Calm\n(Recovery)', fontsize=9, color='blue')
        ax3.text(ricci_c.max() * 0.7, 0, 'Stable + Calm\n(Trending)', fontsize=9, color='green')

        ax3.set_xlabel('Ricci Curvature (Static Geometry)')
        ax3.set_ylabel('SFI Z-Score (Dynamic Physics)')
        ax3.set_title('Static vs Dynamic: Complementary Market Views')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'static_vs_dynamic.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: static_vs_dynamic.png")


def figure5_economic_interpretation(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 5: Economic Interpretation of Fluid Variables

    Maps physics variables to economic concepts.
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid for subplots and interpretation table
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    # Economic variable mapping (for annotation)
    mapping = {
        'velocity': ('momentum_10d', 'Capital Velocity', 'Rate of price change'),
        'viscosity': ('effective_viscosity', 'Market Friction', 'Transaction costs, illiquidity'),
        'density': ('realized_vol_fluid', 'Liquidity Density', 'Inverse of volatility'),
        'pressure': ('shock_formation_index', 'Market Pressure', 'Imbalance intensity'),
        'reynolds': ('shock_formation_index_z', 'Disruption Index', 'Turbulence indicator'),
    }

    # Panel 1: Velocity (Momentum)
    ax1 = fig.add_subplot(gs[0, 0])
    if 'silver_momentum_20d' in features.columns:
        mom = features['silver_momentum_20d'].dropna()
        ax1.plot(mom.index, mom.values * 100, 'b-', linewidth=0.8)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Momentum (%)')
    ax1.set_title('VELOCITY: Capital Flow Rate', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Viscosity (Friction)
    ax2 = fig.add_subplot(gs[0, 1])
    if 'effective_viscosity' in features.columns:
        visc = features['effective_viscosity'].dropna()
        ax2.fill_between(visc.index, 0, visc.values, alpha=0.4, color='purple')
        ax2.plot(visc.index, visc.values, 'purple', linewidth=0.8)
    ax2.set_ylabel('Viscosity ν')
    ax2.set_title('VISCOSITY: Market Friction', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Pressure (SFI)
    ax3 = fig.add_subplot(gs[0, 2])
    if 'shock_formation_index' in features.columns:
        sfi = features['shock_formation_index'].dropna()
        ax3.plot(sfi.index, sfi.values, color='red', linewidth=0.8, alpha=0.8)
    ax3.set_ylabel('SFI')
    ax3.set_title('PRESSURE: Market Imbalance', fontsize=10, fontweight='bold')
    ax3.tick_params(axis='x', rotation=30)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Decay Rate
    ax4 = fig.add_subplot(gs[1, 0])
    if 'momentum_decay_rate' in features.columns:
        decay = features['momentum_decay_rate'].dropna()
        ax4.plot(decay.index, decay.values, 'orange', linewidth=0.8)
    ax4.set_ylabel('Decay Rate')
    ax4.set_title('DIFFUSION: Momentum Dissipation', fontsize=10, fontweight='bold')
    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Flow Divergence
    ax5 = fig.add_subplot(gs[1, 1])
    if 'flow_divergence' in features.columns:
        div = features['flow_divergence'].dropna()
        ax5.fill_between(div.index, div.values, 0, where=div.values > 0,
                        alpha=0.5, color='green', label='Inflow')
        ax5.fill_between(div.index, div.values, 0, where=div.values < 0,
                        alpha=0.5, color='red', label='Outflow')
        ax5.plot(div.index, div.values, 'k-', linewidth=0.5)
    ax5.set_ylabel('Divergence')
    ax5.set_title('DIVERGENCE: Capital Flow Direction', fontsize=10, fontweight='bold')
    ax5.tick_params(axis='x', rotation=30)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: DXY Transmission
    ax6 = fig.add_subplot(gs[1, 2])
    if 'dxy_transmission_coef' in features.columns:
        trans = features['dxy_transmission_coef'].dropna()
        ax6.plot(trans.index, trans.values, 'teal', linewidth=0.8)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('Transmission β')
    ax6.set_title('ADVECTION: Macro Transmission (DXY→Silver)', fontsize=10, fontweight='bold')
    ax6.tick_params(axis='x', rotation=30)
    ax6.grid(True, alpha=0.3)

    # Bottom panel: Interpretation table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    table_data = [
        ['Physical Variable', 'Market Interpretation', 'Equation Role', 'Feature Column'],
        ['Velocity u', 'Capital flow rate', '∂u/∂t + u·∂u/∂x', 'momentum_*d'],
        ['Viscosity ν', 'Transaction costs, illiquidity', '= ν·∂²u/∂x²', 'effective_viscosity'],
        ['Pressure p', 'Order imbalance, stress', '-∇p/ρ term', 'shock_formation_index'],
        ['Density ρ', 'Liquidity, market depth', 'In pressure term', 'realized_vol (inverse)'],
        ['Temperature T', 'Volatility regime', 'Drives viscosity', 'realized_vol_20d'],
        ['External Force F', 'Macro shocks (DXY, rates)', 'Source term S(x,t)', 'dxy_transmission_coef'],
    ]

    table = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)

    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_text_props(weight='bold')
        table[(0, i)].set_facecolor('#d4e6f1')

    ax_table.set_title('Physical-Economic Variable Mapping', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    fig.savefig(output_dir / 'economic_interpretation.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: economic_interpretation.png")


def figure6_flow_divergence(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 6: Cross-Asset Capital Flow Analysis

    Shows flow divergence with regime overlays.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Panel 1: Flow divergence with price overlay
    ax1 = axes[0]
    price_col = 'silver_close' if 'silver_close' in features.columns else None

    if 'flow_divergence' in features.columns:
        div = features['flow_divergence'].dropna()
        ax1.fill_between(div.index, div.values, 0, where=div.values > 0,
                        alpha=0.5, color='green', label='Net Inflow')
        ax1.fill_between(div.index, div.values, 0, where=div.values < 0,
                        alpha=0.5, color='red', label='Net Outflow')
        ax1.plot(div.index, div.values, 'k-', linewidth=0.5)
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Flow Divergence')
        ax1.legend(loc='upper left')

        if price_col:
            ax1_twin = ax1.twinx()
            price = features[price_col].loc[div.index]
            ax1_twin.plot(price.index, price.values, 'b-', linewidth=1.0, alpha=0.7, label='Silver')
            ax1_twin.set_ylabel('Price ($)', color='b')
            ax1_twin.legend(loc='upper right')

    ax1.set_title('Capital Flow Divergence: Σ(asset flows) / √Var(flows)')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rolling correlation of divergence with returns
    ax2 = axes[1]
    if 'flow_divergence' in features.columns and 'log_return_1d' in features.columns:
        div = features['flow_divergence']
        ret = features['log_return_1d']

        # Rolling correlation
        corr = div.rolling(60).corr(ret)

        ax2.fill_between(corr.index, corr.values, 0, where=corr.values > 0,
                        alpha=0.5, color='green')
        ax2.fill_between(corr.index, corr.values, 0, where=corr.values < 0,
                        alpha=0.5, color='red')
        ax2.plot(corr.index, corr.values, 'k-', linewidth=0.5)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Correlation')
        ax2.set_xlabel('Date')
        ax2.set_title('Rolling 60d Correlation: Flow Divergence vs Silver Returns')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'flow_divergence.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: flow_divergence.png")


def figure7_combined_regime_analysis(
    features: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Figure 7: Combined Regime Analysis (Geometric + Fluid)

    Complete picture for thesis showing unified regime view.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    # Panel 1: Silver price with combined regime shading
    ax1 = axes[0]
    price_col = 'silver_close' if 'silver_close' in features.columns else features.columns[0]
    price = features[price_col].dropna()
    ax1.plot(price.index, price.values, 'k-', linewidth=1.2)

    # Find regime column
    regime_col = None
    for col in ['combined_regime', 'geometric_regime', 'regime']:
        if col in features.columns:
            regime_col = col
            break

    if regime_col:
        regimes = features[regime_col].loc[price.index]

        # Shade by regime
        current_regime = regimes.iloc[0] if len(regimes) > 0 else None
        start_idx = regimes.index[0]
        for i, (date, regime) in enumerate(regimes.items()):
            if regime != current_regime or i == len(regimes) - 1:
                if current_regime is not None:
                    color = REGIME_COLORS.get(str(current_regime), 'gray')
                    ax1.axvspan(start_idx, date, alpha=0.2, color=color)
                start_idx = date
                current_regime = regime

        # Add legend
        handles = [plt.Rectangle((0,0), 1, 1, fc=REGIME_COLORS.get(r, 'gray'), alpha=0.5)
                  for r in regimes.unique() if pd.notna(r)]
        labels = [str(r) for r in regimes.unique() if pd.notna(r)]
        ax1.legend(handles, labels, loc='upper left', ncol=3, fontsize=8)

    ax1.set_ylabel('Price ($)')
    ax1.set_title('Silver Price with Unified Regime Classification')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Ricci curvature (geometric view)
    ax2 = axes[1]
    ricci_col = None
    for col in ['ricci_mean_core_60d', 'ricci_mean_60d']:
        if col in features.columns:
            ricci_col = col
            break

    if ricci_col:
        ricci = features[ricci_col].dropna()
        ax2.fill_between(ricci.index, ricci.values, 0, where=ricci.values < 0,
                        alpha=0.3, color='red')
        ax2.fill_between(ricci.index, ricci.values, 0, where=ricci.values >= 0,
                        alpha=0.3, color='green')
        ax2.plot(ricci.index, ricci.values, 'b-', linewidth=0.8)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Ricci Curvature')
    ax2.set_title('Static Geometry: Forman-Ricci Curvature')
    ax2.grid(True, alpha=0.3)

    # Panel 3: SFI (fluid dynamics view)
    ax3 = axes[2]
    if 'shock_formation_index_z' in features.columns:
        sfi_z = features['shock_formation_index_z'].dropna()
        ax3.fill_between(sfi_z.index, sfi_z.values, 2, where=sfi_z.values > 2,
                        alpha=0.5, color=REGIME_COLORS['SHOCK_IMMINENT'])
        ax3.plot(sfi_z.index, sfi_z.values, color='purple', linewidth=0.8)
        ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('SFI (z-score)')
    ax3.set_title('Dynamic Physics: Shock Formation Index')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Momentum decay rate
    ax4 = axes[3]
    if 'momentum_decay_rate' in features.columns:
        decay = features['momentum_decay_rate'].dropna()
        ax4.fill_between(decay.index, 0, decay.values, alpha=0.4, color='orange')
        ax4.plot(decay.index, decay.values, 'darkorange', linewidth=0.8)
    ax4.set_ylabel('Decay Rate')
    ax4.set_xlabel('Date')
    ax4.set_title('Momentum Dissipation: ν · |∂²u/∂t²|')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / 'combined_regime_analysis.png', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: combined_regime_analysis.png")


def compute_statistics(features: pd.DataFrame) -> Dict:
    """Compute summary statistics for the report."""
    stats = {}

    # Shock statistics
    if 'shock_formation_index_z' in features.columns:
        sfi_z = features['shock_formation_index_z']
        shock_warnings = (sfi_z > 2.0).sum()
        total_obs = sfi_z.notna().sum()
        stats['shock_warnings'] = int(shock_warnings)
        stats['shock_warning_rate'] = float(shock_warnings / total_obs) if total_obs > 0 else 0

        # Forward returns analysis
        if 'log_return_1d' in features.columns:
            ret = features['log_return_1d']
            fwd_5d = ret.rolling(5).sum().shift(-5)

            warn_mask = sfi_z > 2.0
            no_warn_mask = sfi_z <= 2.0

            warn_ret = fwd_5d[warn_mask].dropna()
            no_warn_ret = fwd_5d[no_warn_mask].dropna()

            if len(warn_ret) > 0:
                stats['mean_fwd_5d_warn'] = float(warn_ret.mean() * 100)
                stats['std_fwd_5d_warn'] = float(warn_ret.std() * 100)
            if len(no_warn_ret) > 0:
                stats['mean_fwd_5d_no_warn'] = float(no_warn_ret.mean() * 100)
                stats['std_fwd_5d_no_warn'] = float(no_warn_ret.std() * 100)

    # Regime statistics
    for col in ['geometric_regime', 'combined_regime', 'regime']:
        if col in features.columns:
            regime_counts = features[col].value_counts()
            stats[f'{col}_distribution'] = regime_counts.to_dict()
            break

    # Feature correlations with returns
    if 'log_return_1d' in features.columns:
        ret = features['log_return_1d']
        fwd_5d = ret.rolling(5).sum().shift(-5)

        corr_features = ['shock_formation_index_z', 'momentum_decay_rate',
                        'effective_viscosity', 'flow_divergence']
        stats['correlations'] = {}
        for feat in corr_features:
            if feat in features.columns:
                corr = features[feat].corr(fwd_5d)
                stats['correlations'][feat] = float(corr) if pd.notna(corr) else None

    return stats


def write_markdown_report(
    stats: Dict,
    output_dir: Path,
    features: pd.DataFrame,
) -> None:
    """Write markdown summary report."""
    report_path = output_dir / 'FLUID_DYNAMICS_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Fluid Dynamics Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Executive Summary\n\n")
        f.write("This report demonstrates the application of **SPIKAN (Separable Physics-Informed ")
        f.write("Kolmogorov-Arnold Networks)** to market regime analysis. The key insight is that ")
        f.write("SPIKAN's fluid dynamics approach **complements** Ricci curvature by providing:\n\n")
        f.write("- **Static Geometry (Ricci)**: Current market structure/topology\n")
        f.write("- **Dynamic Physics (SPIKAN)**: How the market will evolve\n\n")

        f.write("## Key Findings\n\n")

        if 'shock_warnings' in stats:
            f.write(f"### Shock Detection\n\n")
            f.write(f"- Total shock warnings: **{stats['shock_warnings']}**\n")
            f.write(f"- Warning rate: **{stats['shock_warning_rate']*100:.1f}%** of observations\n")

            if 'mean_fwd_5d_warn' in stats:
                f.write(f"\n### Forward Returns by Shock Warning\n\n")
                f.write(f"| Condition | Mean 5d Return | Std Dev |\n")
                f.write(f"|-----------|----------------|----------|\n")
                f.write(f"| Shock Warning | {stats.get('mean_fwd_5d_warn', 'N/A'):.2f}% | {stats.get('std_fwd_5d_warn', 'N/A'):.2f}% |\n")
                f.write(f"| No Warning | {stats.get('mean_fwd_5d_no_warn', 'N/A'):.2f}% | {stats.get('std_fwd_5d_no_warn', 'N/A'):.2f}% |\n\n")

        if 'correlations' in stats:
            f.write("### Feature Correlations with Forward 5d Returns\n\n")
            f.write("| Feature | Correlation |\n")
            f.write("|---------|-------------|\n")
            for feat, corr in stats['correlations'].items():
                corr_str = f"{corr:.3f}" if corr is not None else "N/A"
                f.write(f"| {feat} | {corr_str} |\n")
            f.write("\n")

        f.write("## Burgers Equation Framework\n\n")
        f.write("The market dynamics are modeled using the Burgers equation:\n\n")
        f.write("$$\\frac{\\partial u}{\\partial t} + u \\cdot \\frac{\\partial u}{\\partial x} = \\nu \\cdot \\frac{\\partial^2 u}{\\partial x^2}$$\n\n")
        f.write("Where:\n")
        f.write("- **u**: Momentum field (capital velocity)\n")
        f.write("- **ν**: Effective viscosity (market friction)\n")
        f.write("- Left side: Nonlinear advection (momentum cascades)\n")
        f.write("- Right side: Viscous diffusion (momentum decay)\n\n")

        f.write("## Shock Formation Index (SFI)\n\n")
        f.write("The SFI measures the balance between nonlinear and diffusive terms:\n\n")
        f.write("$$SFI = \\frac{|u \\cdot du/dt|}{\\nu \\cdot |d^2u/dt^2| + \\epsilon}$$\n\n")
        f.write("High SFI indicates shock formation (momentum is self-reinforcing faster than it dissipates).\n\n")

        f.write("## Figures\n\n")
        figures = [
            ("shock_formation_analysis.png", "Shock Formation Analysis with forward return validation"),
            ("burgers_equation_components.png", "Burgers Equation component decomposition"),
            ("momentum_decay_regimes.png", "Momentum decay patterns by geometric regime"),
            ("static_vs_dynamic.png", "Ricci curvature vs SFI complementarity"),
            ("economic_interpretation.png", "Physical-economic variable mapping"),
            ("flow_divergence.png", "Cross-asset capital flow analysis"),
            ("combined_regime_analysis.png", "Unified regime classification"),
        ]

        for filename, description in figures:
            f.write(f"### {filename.replace('.png', '').replace('_', ' ').title()}\n")
            f.write(f"![{description}](figures/{filename})\n\n")
            f.write(f"*{description}*\n\n")

        f.write("## Conclusion\n\n")
        f.write("The fluid dynamics framework provides actionable insights:\n\n")
        f.write("1. **Shock warnings** identify potential regime transitions before they occur\n")
        f.write("2. **Momentum decay rate** distinguishes trending vs mean-reverting periods\n")
        f.write("3. **Flow divergence** reveals cross-asset capital movements\n")
        f.write("4. Combined with Ricci curvature, provides complete market state assessment\n\n")

        f.write("---\n")
        f.write("*Report generated by SPIKAN Fluid Dynamics Analysis Pipeline*\n")

    print(f"  Saved: FLUID_DYNAMICS_REPORT.md")


def save_tables(stats: Dict, features: pd.DataFrame, output_dir: Path) -> None:
    """Save statistics tables as CSV."""
    tables_dir = output_dir / 'tables'
    tables_dir.mkdir(exist_ok=True)

    # Shock warning stats
    if 'shock_warnings' in stats:
        shock_stats = pd.DataFrame([{
            'total_shock_warnings': stats['shock_warnings'],
            'warning_rate': stats['shock_warning_rate'],
            'mean_fwd_5d_warn': stats.get('mean_fwd_5d_warn', np.nan),
            'std_fwd_5d_warn': stats.get('std_fwd_5d_warn', np.nan),
            'mean_fwd_5d_no_warn': stats.get('mean_fwd_5d_no_warn', np.nan),
            'std_fwd_5d_no_warn': stats.get('std_fwd_5d_no_warn', np.nan),
        }])
        shock_stats.to_csv(tables_dir / 'shock_warning_stats.csv', index=False)

    # Feature correlations
    if 'correlations' in stats:
        corr_df = pd.DataFrame([stats['correlations']])
        corr_df.to_csv(tables_dir / 'feature_correlations.csv', index=False)

    # Regime distribution
    for key in stats:
        if 'distribution' in key:
            dist_df = pd.DataFrame([stats[key]]).T
            dist_df.columns = ['count']
            dist_df.index.name = 'regime'
            dist_df.to_csv(tables_dir / f'{key}.csv')

    # Forward returns by shock status
    if 'shock_formation_index_z' in features.columns and 'log_return_1d' in features.columns:
        sfi_z = features['shock_formation_index_z']
        ret = features['log_return_1d']
        fwd_5d = ret.rolling(5).sum().shift(-5)

        fwd_df = pd.DataFrame({
            'date': features.index,
            'shock_warning': sfi_z > 2.0,
            'forward_5d_return': fwd_5d * 100,
        })
        fwd_df = fwd_df.dropna()
        fwd_df.to_csv(tables_dir / 'forward_returns_by_shock.csv', index=False)

    print(f"  Saved tables to {tables_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Generate fluid dynamics and shockwave analysis report"
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to features parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/silver/fluid_dynamics",
        help="Output directory for report",
    )
    parser.add_argument(
        "--generate-features",
        action="store_true",
        help="Generate features with fluid dynamics if not present",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/silver_universe.yaml",
        help="Config file for feature generation",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FLUID DYNAMICS ANALYSIS REPORT GENERATOR")
    print("=" * 70)

    # Setup output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'tables').mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Load features
    print("\n1. Loading features...")
    try:
        features = load_or_generate_features(
            features_path=args.features_path,
            config_path=args.config,
            generate=args.generate_features,
        )
        print(f"   Loaded {len(features):,} observations, {len(features.columns)} features")
        print(f"   Date range: {features.index.min()} to {features.index.max()}")
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        print("\n   To generate features, run:")
        print("   python -m src.pipeline.silver_pipeline features --config configs/silver_universe.yaml --fluid-dynamics")
        sys.exit(1)

    # Add fluid features if missing
    print("\n2. Checking for fluid dynamics features...")
    features = compute_fluid_features_if_missing(features)

    fluid_cols = [c for c in features.columns if any(x in c for x in
                  ['shock', 'momentum', 'viscosity', 'divergence', 'decay'])]
    print(f"   Found {len(fluid_cols)} fluid dynamics features")

    # Generate figures
    print("\n3. Generating figures...")
    figure1_shock_formation_analysis(features, output_dir / 'figures')
    figure2_burgers_equation_components(features, output_dir / 'figures')
    figure3_momentum_decay_by_regime(features, output_dir / 'figures')
    figure4_static_vs_dynamic(features, output_dir / 'figures')
    figure5_economic_interpretation(features, output_dir / 'figures')
    figure6_flow_divergence(features, output_dir / 'figures')
    figure7_combined_regime_analysis(features, output_dir / 'figures')

    # Compute statistics
    print("\n4. Computing statistics...")
    stats = compute_statistics(features)

    # Save tables
    print("\n5. Saving tables...")
    save_tables(stats, features, output_dir)

    # Write markdown report
    print("\n6. Writing report...")
    write_markdown_report(stats, output_dir, features)

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput saved to: {output_dir}")
    print("\nFiles generated:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")


if __name__ == "__main__":
    main()
