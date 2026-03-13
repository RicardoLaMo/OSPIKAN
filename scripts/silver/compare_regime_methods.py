#!/usr/bin/env python3
"""
Compare Regime Detection Methods

Analyzes and compares three regime detection approaches:
1. Baseline: MA crossover x Volatility buckets
2. Geometric: Ricci curvature + GA + MST stress
3. Markov: HMM with geometric features (if available)

Generates comparative statistics and visualizations.
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.regimes import baseline_regime_table, geometric_regime_classification


def plot_regime_comparison(features: pd.DataFrame, baseline_regimes: pd.Series,
                           geometric_regimes: pd.Series, out_path: str):
    """Plot time series comparison of regime methods."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    # Silver price
    axes[0].plot(features.index, features['silver_close'], color='black', linewidth=0.8)
    axes[0].set_ylabel('Silver Price')
    axes[0].set_title('Silver Price Time Series')
    axes[0].grid(True, alpha=0.3)

    # Baseline regimes (colored background)
    ax1 = axes[1]
    ax1.plot(features.index, features['silver_close'], color='gray', linewidth=0.5, alpha=0.5)

    regime_colors_baseline = {
        'bull_low': 'lightgreen', 'bull_medium': 'green', 'bull_high': 'darkgreen',
        'bear_low': 'lightcoral', 'bear_medium': 'red', 'bear_high': 'darkred',
        'neutral_low': 'lightgray', 'neutral_medium': 'gray', 'neutral_high': 'darkgray'
    }

    for regime, color in regime_colors_baseline.items():
        mask = baseline_regimes == regime
        if mask.any():
            ax1.fill_between(features.index, features['silver_close'].min(),
                            features['silver_close'].max(),
                            where=mask.reindex(features.index, fill_value=False),
                            alpha=0.3, color=color, label=regime)

    ax1.set_ylabel('Silver (Baseline Regimes)')
    ax1.set_title('Baseline: MA Crossover x Volatility')
    ax1.legend(loc='upper left', fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Geometric regimes
    ax2 = axes[2]
    ax2.plot(features.index, features['silver_close'], color='gray', linewidth=0.5, alpha=0.5)

    regime_colors_geom = {
        'STABLE': 'green',
        'STRESS': 'red',
        'RECOVERY': 'blue',
        'TRANSITION': 'orange',
        'UNKNOWN': 'gray'
    }

    for regime, color in regime_colors_geom.items():
        mask = geometric_regimes == regime
        if mask.any():
            ax2.fill_between(features.index, features['silver_close'].min(),
                            features['silver_close'].max(),
                            where=mask.reindex(features.index, fill_value=False),
                            alpha=0.4, color=color, label=regime)

    ax2.set_ylabel('Silver (Geometric Regimes)')
    ax2.set_title('Geometric: Ricci Curvature + GA Rotors + MST Stress')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def analyze_regime_transitions(baseline: pd.Series, geometric: pd.Series, out_path: str):
    """Analyze regime transition dynamics."""
    # Transition counts
    baseline_trans = (baseline != baseline.shift(1)).sum()
    geometric_trans = (geometric != geometric.shift(1)).sum()

    # Average duration
    baseline_durations = []
    geometric_durations = []

    for regime in baseline.unique():
        if pd.isna(regime):
            continue
        mask = (baseline == regime).astype(int)
        runs = mask.diff().ne(0).cumsum()
        durations = mask.groupby(runs).sum()
        baseline_durations.extend(durations[durations > 0])

    for regime in geometric.unique():
        if pd.isna(regime):
            continue
        mask = (geometric == regime).astype(int)
        runs = mask.diff().ne(0).cumsum()
        durations = mask.groupby(runs).sum()
        geometric_durations.extend(durations[durations > 0])

    report = f"""
=== Regime Transition Analysis ===

Baseline Method (MA x Vol):
  - Total transitions: {baseline_trans}
  - Avg regime duration: {np.mean(baseline_durations):.1f} days
  - Median regime duration: {np.median(baseline_durations):.1f} days

Geometric Method (Curvature + GA):
  - Total transitions: {geometric_trans}
  - Avg regime duration: {np.mean(geometric_durations):.1f} days
  - Median regime duration: {np.median(geometric_durations):.1f} days

Interpretation:
  - Fewer transitions = more stable regime classification
  - Geometric method captures structural changes, not just price patterns
"""

    print(report)
    with open(out_path, 'w') as f:
        f.write(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Compare regime detection methods")
    parser.add_argument('--features', required=True, help='Path to silver_features_*.parquet')
    parser.add_argument('--out-dir', default='reports/silver/analysis', help='Output directory')
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    print(f"Loaded features: {features.shape}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'tables'), exist_ok=True)

    # Compute baseline regimes
    print("\n=== Computing Baseline Regimes ===")
    baseline_df = baseline_regime_table(features['silver_close'])
    baseline_regimes = baseline_df['regime']

    # Compute geometric regimes
    print("\n=== Computing Geometric Regimes ===")
    geometric_regimes = geometric_regime_classification(features)

    # Plot comparison
    print("\n=== Plotting Regime Comparison ===")
    plot_regime_comparison(
        features,
        baseline_regimes,
        geometric_regimes,
        out_path=os.path.join(args.out_dir, 'figures', 'regime_method_comparison.png')
    )

    # Analyze transitions
    print("\n=== Analyzing Regime Transitions ===")
    analyze_regime_transitions(
        baseline_regimes,
        geometric_regimes,
        out_path=os.path.join(args.out_dir, 'tables', 'transition_analysis.txt')
    )

    # Return statistics comparison
    print("\n=== Return Statistics by Method ===")

    # Baseline
    baseline_stats = features.groupby(baseline_regimes)['log_return_1d'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('sharpe', lambda x: x.mean() / x.std() if x.std() > 0 else np.nan),
        ('count', 'count')
    ]).sort_values('mean', ascending=False)
    baseline_stats['annualized_return'] = baseline_stats['mean'] * 252
    baseline_stats['annualized_vol'] = baseline_stats['std'] * np.sqrt(252)

    # Geometric
    geometric_stats = features.groupby(geometric_regimes)['log_return_1d'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('sharpe', lambda x: x.mean() / x.std() if x.std() > 0 else np.nan),
        ('count', 'count')
    ]).sort_values('mean', ascending=False)
    geometric_stats['annualized_return'] = geometric_stats['mean'] * 252
    geometric_stats['annualized_vol'] = geometric_stats['std'] * np.sqrt(252)

    print("\nBaseline Regimes (Top 5 by return):")
    print(baseline_stats.head())

    print("\nGeometric Regimes:")
    print(geometric_stats)

    # Save
    baseline_stats.to_csv(os.path.join(args.out_dir, 'tables', 'baseline_regime_stats.csv'))
    geometric_stats.to_csv(os.path.join(args.out_dir, 'tables', 'geometric_regime_stats.csv'))

    print(f"\n=== Analysis Complete ===")
    print(f"Outputs in: {args.out_dir}")


if __name__ == '__main__':
    main()
