#!/usr/bin/env python3
"""
Compute sectional Ricci curvatures for different asset manifolds.

This script extends the base geometric analysis with manifold-specific curvatures:
- Monetary manifold (gold, silver, TIP, DXY)
- Industrial manifold (silver, copper, China, EM, energy)
- Miner manifold (silver mining equities)

Cross-manifold divergence signals regime transitions.
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.geometry.graph_curvature import (
    rolling_sectional_ricci_curvature,
    compute_manifold_stress_differential,
)


def load_manifolds_from_config(config_path: str) -> dict:
    """Load manifold definitions from YAML config."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'manifolds' not in config:
        raise ValueError(f"No 'manifolds' section found in {config_path}")

    # Extract symbol lists from manifold definitions
    manifolds = {}
    for name, spec in config['manifolds'].items():
        if isinstance(spec, dict) and 'symbols' in spec:
            manifolds[name] = spec['symbols']
        elif isinstance(spec, list):
            manifolds[name] = spec
        else:
            print(f"Warning: Skipping manifold '{name}' with invalid format", file=sys.stderr)

    return manifolds


def plot_sectional_curvatures(
    sectional_features: pd.DataFrame,
    manifold_names: list,
    silver_close: pd.Series,
    out_path: str
):
    """Plot sectional curvatures for all manifolds."""
    n_manifolds = len(manifold_names)
    fig, axes = plt.subplots(n_manifolds + 1, 1, figsize=(14, 3 * (n_manifolds + 1)), sharex=True)

    # Plot silver price
    ax_price = axes[0]
    ax_price.plot(silver_close.index, silver_close, color='silver', linewidth=1.5, label='Silver Price')
    ax_price.set_ylabel('Silver Price (USD)', fontsize=10)
    ax_price.set_title('Silver Price and Sectional Ricci Curvatures', fontsize=12, fontweight='bold')
    ax_price.legend(loc='upper left')
    ax_price.grid(True, alpha=0.3)

    # Plot each manifold's curvature
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    for i, manifold_name in enumerate(manifold_names):
        ax = axes[i + 1]
        col_mean = f'ricci_mean_{manifold_name}_60d'
        col_min = f'ricci_min_{manifold_name}_60d'

        if col_mean in sectional_features.columns:
            ax.plot(
                sectional_features.index,
                sectional_features[col_mean],
                color=colors[i % len(colors)],
                linewidth=1.5,
                label=f'{manifold_name.capitalize()} Mean'
            )

        if col_min in sectional_features.columns:
            ax.plot(
                sectional_features.index,
                sectional_features[col_min],
                color=colors[i % len(colors)],
                linewidth=1,
                alpha=0.5,
                linestyle='--',
                label=f'{manifold_name.capitalize()} Min'
            )

        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
        ax.set_ylabel(f'{manifold_name.capitalize()}\nCurvature', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_curvature_divergence(
    sectional_features: pd.DataFrame,
    silver_returns: pd.Series,
    out_path: str
):
    """Plot cross-manifold curvature divergence signals."""
    # Find all divergence columns
    div_cols = [c for c in sectional_features.columns if 'curvature_divergence_' in c and '_zscore_' not in c]

    if not div_cols:
        print("Warning: No divergence columns found, skipping divergence plot", file=sys.stderr)
        return

    n_divs = len(div_cols)
    fig, axes = plt.subplots(n_divs + 1, 1, figsize=(14, 3 * (n_divs + 1)), sharex=True)

    if n_divs == 1:
        axes = [axes[0], axes[1]]  # Ensure axes is always a list

    # Plot returns
    ax_ret = axes[0]
    ax_ret.plot(silver_returns.index, silver_returns.cumsum(), color='silver', linewidth=1.5, label='Cumulative Returns')
    ax_ret.set_ylabel('Cumulative\nLog Returns', fontsize=10)
    ax_ret.set_title('Curvature Divergence Signals vs Silver Returns', fontsize=12, fontweight='bold')
    ax_ret.legend(loc='upper left')
    ax_ret.grid(True, alpha=0.3)

    # Plot each divergence
    colors = ['purple', 'orange', 'brown', 'pink', 'cyan']

    for i, col in enumerate(div_cols):
        ax = axes[i + 1]

        # Extract manifold names from column
        # Format: curvature_divergence_{name1}_{name2}_60d
        parts = col.replace('curvature_divergence_', '').replace('_60d', '').split('_')
        if len(parts) >= 2:
            name1, name2 = parts[0], parts[1]
            label = f'{name1.capitalize()} - {name2.capitalize()}'
        else:
            label = col

        div_series = sectional_features[col]

        # Plot divergence
        ax.plot(div_series.index, div_series, color=colors[i % len(colors)], linewidth=1.5, label=label)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

        # Highlight extreme divergence (±2σ)
        div_mean = div_series.rolling(252).mean()
        div_std = div_series.rolling(252).std()
        upper = div_mean + 2 * div_std
        lower = div_mean - 2 * div_std

        ax.fill_between(div_series.index, lower, upper, alpha=0.1, color=colors[i % len(colors)])

        ax.set_ylabel(f'{label}\nDivergence', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Date', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute sectional Ricci curvatures for asset manifolds"
    )
    parser.add_argument("--features", required=True, help="Path to silver_features_*.parquet")
    parser.add_argument("--config", required=True, help="Path to enhanced config YAML with manifold definitions")
    parser.add_argument("--out-dir", default="reports/silver", help="Output directory")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size")
    args = parser.parse_args()

    # Load features
    features = pd.read_parquet(args.features)
    print(f"Loaded features: {features.shape}")

    # Load manifold definitions from config
    manifolds = load_manifolds_from_config(args.config)
    print(f"\nManifolds defined: {list(manifolds.keys())}")

    # Load panel to get returns for all assets
    # Infer panel path from features path
    features_basename = os.path.basename(args.features)
    run_id = features_basename.replace('silver_features_', '').replace('.parquet', '')
    panel_path = os.path.join('data', 'interim', f'silver_panel_close_{run_id}.csv')

    if not os.path.exists(panel_path):
        print(f"Error: Panel file not found: {panel_path}", file=sys.stderr)
        print("Sectional curvature requires the full panel CSV (not just features)", file=sys.stderr)
        sys.exit(1)

    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    print(f"Loaded panel: {panel.shape}")

    # Compute log returns
    returns = panel.apply(lambda x: np.log(x / x.shift(1)), axis=0)
    returns = returns.dropna(how='all', axis=1)

    print(f"\n=== Computing Sectional Ricci Curvatures ===")

    # Compute sectional curvatures
    sectional_features = rolling_sectional_ricci_curvature(
        returns,
        manifolds=manifolds,
        window=args.window,
        min_assets=3,
    )

    print(f"Sectional features computed: {sectional_features.shape}")
    print(f"Columns: {list(sectional_features.columns)}")

    # Compute MST stress differentials
    print(f"\n=== Computing Manifold Stress Differentials ===")
    stress_features = compute_manifold_stress_differential(
        returns,
        manifolds=manifolds,
        window=args.window,
        min_assets=3,
    )

    print(f"Stress features computed: {stress_features.shape}")

    # Merge with existing features
    features_enhanced = pd.concat([features, sectional_features, stress_features], axis=1)

    # Save enhanced features
    out_features_path = args.features.replace('.parquet', '_sectional.parquet')
    features_enhanced.to_parquet(out_features_path)
    print(f"\nSaved enhanced features: {out_features_path}")

    # Generate visualizations
    out_fig_dir = os.path.join(args.out_dir, "figures")
    os.makedirs(out_fig_dir, exist_ok=True)

    print(f"\n=== Generating Visualizations ===")

    # Plot sectional curvatures
    manifold_names = list(manifolds.keys())
    if 'silver_close' in features.columns:
        plot_sectional_curvatures(
            sectional_features,
            manifold_names,
            features['silver_close'],
            out_path=os.path.join(out_fig_dir, "sectional_curvature_timeseries.png")
        )

    # Plot divergence signals
    if 'log_return_1d' in features.columns:
        plot_curvature_divergence(
            sectional_features,
            features['log_return_1d'],
            out_path=os.path.join(out_fig_dir, "curvature_divergence_regimes.png")
        )

    # Save summary statistics
    out_table_dir = os.path.join(args.out_dir, "tables")
    os.makedirs(out_table_dir, exist_ok=True)

    summary = sectional_features.describe()
    summary.to_csv(os.path.join(out_table_dir, "sectional_curvature_summary.csv"))
    print(f"Saved summary: {out_table_dir}/sectional_curvature_summary.csv")

    print(f"\n=== Sectional Curvature Analysis Complete ===")
    print(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
