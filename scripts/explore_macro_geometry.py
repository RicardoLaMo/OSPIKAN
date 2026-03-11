#!/usr/bin/env python3
"""
Quick exploration script for macro-geometry interactions.

This script helps you analyze the key thesis contribution: how credit spreads,
treasury yields, and other macro signals interact with geometric regime measures
(Ricci curvature, MST stress).

Usage:
    python scripts/explore_macro_geometry.py

Author: Enhanced silver analysis framework
Date: 2026-01-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
FEATURES_FILE = "data/processed/silver_features_20260112-205924_5c4f0d4fec.parquet"
OUTPUT_DIR = Path("reports/silver/macro_geometry_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """Load enhanced feature set."""
    print(f"Loading features from {FEATURES_FILE}...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"  Loaded {len(df):,} observations, {len(df.columns)} features")
    print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    return df

def compute_lead_lag_correlation(df, signal_col, target_col, max_lag=60):
    """
    Compute lead-lag correlation between a signal and target.
    Positive lag means signal leads target.

    Returns:
        pd.Series: Correlations at each lag
    """
    correlations = []
    lags = range(-max_lag, max_lag + 1)

    for lag in lags:
        if lag < 0:
            # Signal lags target
            corr = df[signal_col].corr(df[target_col].shift(-lag))
        else:
            # Signal leads target
            corr = df[signal_col].shift(lag).corr(df[target_col])
        correlations.append(corr)

    return pd.Series(correlations, index=lags)

def analyze_credit_geometry_interaction(df):
    """
    Key thesis analysis: Do credit spreads lead geometric regime changes?

    Investment banking hypothesis: Credit markets lead equity/commodity by 2-4 weeks.
    Geometric hypothesis: Ricci curvature detects regime fragmentation.
    Combined: Credit spread widening should precede Ricci curvature decline.
    """
    print("\n" + "=" * 80)
    print("CREDIT-GEOMETRY INTERACTION ANALYSIS")
    print("=" * 80)

    if 'ig_hy_spread' not in df.columns or 'ricci_mean_60d' not in df.columns:
        print("  ❌ Missing required features (ig_hy_spread or ricci_mean_60d)")
        return None

    # Compute lead-lag
    print("\n1. Computing lead-lag correlation (ig_hy_spread vs ricci_mean_60d)...")
    lead_lag = compute_lead_lag_correlation(
        df,
        signal_col='ig_hy_spread',
        target_col='ricci_mean_60d',
        max_lag=60
    )

    # Find optimal lag
    optimal_lag = lead_lag.idxmax()
    optimal_corr = lead_lag.max()

    print(f"   Optimal lag: {optimal_lag} days (positive = credit leads geometry)")
    print(f"   Correlation at optimal lag: {optimal_corr:.3f}")

    if optimal_lag > 0:
        print(f"   ✅ Credit spreads lead Ricci curvature by {optimal_lag} days")
    else:
        print(f"   ℹ️  Credit spreads lag Ricci curvature by {-optimal_lag} days")

    # Plot lead-lag
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(lead_lag.index, lead_lag.values, linewidth=2)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Zero lag')
    ax.axvline(optimal_lag, color='red', linestyle='--', alpha=0.7,
               label=f'Optimal lag: {optimal_lag}d')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Lag (days, positive = credit leads)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Credit Spread vs Ricci Curvature: Lead-Lag Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "credit_geometry_lead_lag.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    return {
        'optimal_lag': optimal_lag,
        'optimal_corr': optimal_corr,
        'lead_lag_series': lead_lag,
    }

def analyze_macro_regime_transitions(df):
    """
    Analyze how macro signals behave around geometric regime transitions.

    Define regime transition as large change in Ricci curvature.
    """
    print("\n" + "=" * 80)
    print("MACRO SIGNALS AROUND REGIME TRANSITIONS")
    print("=" * 80)

    if 'ricci_mean_60d' not in df.columns:
        print("  ❌ Missing ricci_mean_60d")
        return None

    # Detect regime transitions (large curvature changes)
    ricci_change = df['ricci_mean_60d'].diff()
    ricci_change_std = ricci_change.std()

    # Stress transitions: curvature drops sharply (becomes more negative)
    stress_transitions = ricci_change < -2 * ricci_change_std

    # Recovery transitions: curvature rises sharply (becomes less negative)
    recovery_transitions = ricci_change > 2 * ricci_change_std

    print(f"\n   Detected {stress_transitions.sum()} stress transitions")
    print(f"   Detected {recovery_transitions.sum()} recovery transitions")

    # Analyze macro signals around transitions
    macro_signals = [c for c in df.columns if any(x in c for x in
        ['ig_hy_spread', 'term_premium', 'gold_oil_ratio', 'oil_level', 'vix_level'])]

    print(f"\n   Analyzing {len(macro_signals)} macro signals around transitions...")

    results = {}
    for signal in macro_signals:
        if signal not in df.columns:
            continue

        # Average signal value 20 days before stress transitions
        stress_dates = df.index[stress_transitions]
        signal_before_stress = []
        for date in stress_dates:
            idx = df.index.get_loc(date)
            if idx >= 20:
                signal_before_stress.append(df[signal].iloc[idx-20:idx].mean())

        avg_before_stress = np.nanmean(signal_before_stress) if signal_before_stress else np.nan
        results[signal] = {'before_stress': avg_before_stress}

    # Show results
    print("\n   Macro signals 20 days before stress transitions:")
    for signal, vals in sorted(results.items(), key=lambda x: x[0]):
        val = vals['before_stress']
        if not np.isnan(val):
            print(f"     {signal:30s}: {val:>10.4f}")

    return results

def analyze_energy_silver_relationship(df):
    """
    Analyze energy complex relationship with silver.

    Hypothesis: Oil price affects silver through:
    1. Inflation expectations (oil ↑ → silver ↑)
    2. Industrial demand (oil ↓ → recession → silver ↓)
    """
    print("\n" + "=" * 80)
    print("ENERGY-SILVER RELATIONSHIP")
    print("=" * 80)

    if 'oil_level' not in df.columns or 'log_return_1d' not in df.columns:
        print("  ❌ Missing required features")
        return None

    # Compute rolling correlation
    if 'oil_log_return_1d' in df.columns:
        oil_returns = df['oil_log_return_1d']
    else:
        oil_returns = np.log(df['oil_level'] / df['oil_level'].shift(1))

    rolling_corr = df['log_return_1d'].rolling(window=60).corr(oil_returns)

    print(f"\n   Silver-Oil correlation (60-day rolling):")
    print(f"     Mean:   {rolling_corr.mean():.3f}")
    print(f"     Median: {rolling_corr.median():.3f}")
    print(f"     Std:    {rolling_corr.std():.3f}")
    print(f"     Min:    {rolling_corr.min():.3f}")
    print(f"     Max:    {rolling_corr.max():.3f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Panel 1: Prices
    ax1_twin = ax1.twinx()
    ax1.plot(df.index, df['silver_close'], color='gray', alpha=0.7, label='Silver')
    ax1_twin.plot(df.index, df['oil_level'], color='orange', alpha=0.7, label='Oil')
    ax1.set_ylabel('Silver Price', color='gray', fontsize=11)
    ax1_twin.set_ylabel('Oil Price (WTI)', color='orange', fontsize=11)
    ax1.set_title('Silver vs Oil Prices', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Rolling correlation
    ax2.plot(df.index, rolling_corr, color='purple', linewidth=2)
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax2.fill_between(df.index, 0, rolling_corr, where=(rolling_corr > 0),
                      color='green', alpha=0.3, label='Positive correlation')
    ax2.fill_between(df.index, 0, rolling_corr, where=(rolling_corr < 0),
                      color='red', alpha=0.3, label='Negative correlation')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('60-day Rolling Correlation', fontsize=11)
    ax2.set_title('Silver-Oil Return Correlation', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / "energy_silver_relationship.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   Saved: {output_path}")
    plt.close()

    return rolling_corr

def main():
    """Run all analyses."""
    print("=" * 80)
    print("MACRO-GEOMETRY INTERACTION EXPLORER")
    print("=" * 80)
    print("\nThis script analyzes the key thesis contribution:")
    print("How do macro signals (credit, treasury, energy) interact with")
    print("geometric regime measures (Ricci curvature, MST stress)?")

    # Load data
    df = load_features()

    # Run analyses
    credit_geo = analyze_credit_geometry_interaction(df)
    macro_regime = analyze_macro_regime_transitions(df)
    energy_silver = analyze_energy_silver_relationship(df)

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("\nKey findings:")

    if credit_geo:
        lag = credit_geo['optimal_lag']
        corr = credit_geo['optimal_corr']
        if lag > 0:
            print(f"  ✅ Credit spreads LEAD geometric regimes by {lag} days (r={corr:.3f})")
            print(f"     → Use credit signals as early warning for regime shifts")
        else:
            print(f"  ℹ️  Credit spreads LAG geometric regimes by {-lag} days")
            print(f"     → Geometry may be capturing faster market dynamics")

    print("\nFor your thesis:")
    print("  1. Emphasize multi-asset intelligence (43 assets vs 14)")
    print("  2. Highlight credit-geometry interaction (novel contribution)")
    print("  3. Position as 'Differential Geometry Meets Macro Finance'")
    print("  4. Use macro signals for regime confirmation/timing")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
