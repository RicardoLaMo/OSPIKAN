#!/usr/bin/env python3
"""
Ricci Flow Shock Propagation Scenarios

Simulates how shocks to different assets propagate through the market
using Ricci flow dynamics. Answers: "If asset X crashes, which assets
are most affected through geometric deformation?"

This uses TRUE differential geometry, not just correlation.
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.geometry.ricci_flow import ricci_flow_shock_propagation


def run_shock_scenarios(panel: pd.DataFrame, shock_assets: list, shock_magnitude: float = -0.05):
    """Run shock propagation for multiple assets."""
    results = {}

    for shock_asset in shock_assets:
        if shock_asset not in panel.columns:
            print(f"Warning: {shock_asset} not in panel, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Simulating {shock_magnitude*100:+.1f}% shock to {shock_asset}")
        print(f"{'='*60}")

        try:
            # Compute log returns
            returns = panel.apply(lambda x: np.log(x / x.shift(1)), axis=0)
            returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="all")

            # Run shock propagation
            distances = ricci_flow_shock_propagation(
                returns,
                shock_asset=shock_asset,
                shock_magnitude=shock_magnitude,
                window=60,
                flow_steps=50
            )

            results[shock_asset] = distances

            # Display top affected assets
            sorted_dist = sorted(distances.items(), key=lambda x: x[1])
            print(f"\nTop 10 Most Affected Assets (by geometric distance):")
            for i, (asset, dist) in enumerate(sorted_dist[:10], 1):
                if asset != shock_asset:
                    print(f"  {i:2d}. {asset:15s} - distance = {dist:.4f}")

        except Exception as e:
            print(f"Error in shock propagation for {shock_asset}: {e}")
            results[shock_asset] = {}

    return results


def plot_shock_propagation_heatmap(results: dict, out_path: str):
    """Plot heatmap of shock propagation across scenarios."""
    if not results:
        print("No results to plot")
        return

    # Build distance matrix
    all_assets = set()
    for distances in results.values():
        all_assets.update(distances.keys())

    all_assets = sorted(all_assets)
    shocked_assets = list(results.keys())

    matrix = np.zeros((len(shocked_assets), len(all_assets)))

    for i, shock_asset in enumerate(shocked_assets):
        distances = results[shock_asset]
        for j, asset in enumerate(all_assets):
            matrix[i, j] = distances.get(asset, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(14, len(shocked_assets) * 0.5 + 2))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')

    ax.set_xticks(range(len(all_assets)))
    ax.set_xticklabels(all_assets, rotation=90, fontsize=8)
    ax.set_yticks(range(len(shocked_assets)))
    ax.set_yticklabels(shocked_assets, fontsize=9)

    ax.set_xlabel('Affected Assets')
    ax.set_ylabel('Shocked Assets')
    ax.set_title('Ricci Flow Shock Propagation: Geometric Distance After Flow')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Geometric Distance (lower = more affected)', rotation=270, labelpad=20)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ricci flow shock propagation scenarios")
    parser.add_argument('--panel', required=True, help='Path to silver_panel_close_*.csv')
    parser.add_argument('--shock-assets', nargs='+', default=['SI=F', 'GC=F', 'DX-Y.NYB', 'SPY'],
                       help='Assets to shock (space-separated)')
    parser.add_argument('--shock-magnitude', type=float, default=-0.05,
                       help='Shock size (e.g., -0.05 = -5%% return)')
    parser.add_argument('--out-dir', default='reports/silver/ricci_flow', help='Output directory')
    args = parser.parse_args()

    panel = pd.read_csv(args.panel, index_col=0, parse_dates=True)
    print(f"Loaded panel: {panel.shape}")
    print(f"Symbols: {list(panel.columns)}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'tables'), exist_ok=True)

    # Run shock scenarios
    print(f"\n{'#'*60}")
    print("RICCI FLOW SHOCK PROPAGATION SCENARIOS")
    print(f"{'#'*60}")

    results = run_shock_scenarios(panel, args.shock_assets, args.shock_magnitude)

    # Plot heatmap
    if results:
        plot_shock_propagation_heatmap(
            results,
            out_path=os.path.join(args.out_dir, 'figures', 'shock_propagation_heatmap.png')
        )

        # Save detailed results
        for shock_asset, distances in results.items():
            df = pd.DataFrame(list(distances.items()), columns=['asset', 'geometric_distance'])
            df = df.sort_values('geometric_distance')
            out_path = os.path.join(args.out_dir, 'tables', f'shock_{shock_asset.replace("/", "_")}_distances.csv')
            df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

    print(f"\n{'#'*60}")
    print("RICCI FLOW ANALYSIS COMPLETE")
    print(f"{'#'*60}")
    print(f"Outputs in: {args.out_dir}")


if __name__ == '__main__':
    main()
