#!/usr/bin/env python3
"""
Geometric Signal Analysis

Investigates key questions:
1. When does Ricci curvature turn negative? (market stress events)
2. Does GA rotor magnitude lead price drawdowns? (early warning)
3. Does MST stress spike before crashes?
4. Correlation structure: Are geometric features redundant or complementary?

Generates predictive signal analysis and timing analysis.
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_curvature_turning_points(features: pd.DataFrame, out_path: str):
    """
    Detect curvature "stress events" using distributional thresholds.

    For distance-weighted Forman–Ricci curvature, the mean can remain negative
    for most/all of the sample. Zero-crossings are then non-discriminative.

    Here, we define "stress" as the worst decile of a curvature signal
    (preferring tail metrics like p10/min, and preferring the core manifold
    if present).
    """
    ricci_col = _pick_col(
        features,
        [
            "ricci_p10_core_60d",
            "ricci_min_core_60d",
            "ricci_p10_60d",
            "ricci_min_60d",
            "ricci_mean_core_60d",
            "ricci_mean_60d",
        ],
    )
    if ricci_col is None:
        print("\n=== Ricci Curvature Turning Points ===")
        print("Skipped: no ricci curvature columns found.")
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"Saved empty results: {out_path}")
        return [], []

    ricci = features[ricci_col].dropna()
    price = features["silver_close"].reindex(ricci.index)

    stress_q = 0.10
    stress_threshold = float(ricci.quantile(stress_q))
    stress_mask = ricci <= stress_threshold

    stress_onsets = ricci.index[stress_mask & ~stress_mask.shift(1, fill_value=False)]
    stress_reliefs = ricci.index[~stress_mask & stress_mask.shift(1, fill_value=False)]

    print(f"\n=== Ricci Curvature Stress Events ({ricci_col}) ===")
    print(f"Stress threshold (worst {int(stress_q*100)}%): {stress_threshold:.4f}")
    print(f"Stress onsets (enter stress): {len(stress_onsets)} events")
    print(f"Stress reliefs (exit stress): {len(stress_reliefs)} events")

    # Analyze forward returns after stress onset
    forward_returns = []
    horizons = [5, 10, 20, 60]

    for date in stress_onsets:
        if date not in price.index:
            continue
        idx = price.index.get_loc(date)
        for horizon in horizons:
            if idx + horizon < len(price):
                fwd_ret = (price.iloc[idx + horizon] / price.iloc[idx]) - 1
                forward_returns.append({
                    'date': date,
                    'horizon': horizon,
                    'forward_return': fwd_ret
                })

    if len(forward_returns) > 0:
        fwd_df = pd.DataFrame(forward_returns)

        print(f"\nForward Returns After Stress Onset:")
        for h in horizons:
            subset = fwd_df[fwd_df['horizon'] == h]
            if len(subset) > 0:
                mean_ret = subset['forward_return'].mean()
                print(f"  {h:2d}-day forward return: {mean_ret:+.4f} ({mean_ret*100:+.2f}%)")

        fwd_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")
    else:
        print("\nNote: No stress onsets detected (thresholding produced no valid forward-return windows).")
        pd.DataFrame().to_csv(out_path, index=False)
        print(f"Saved empty results: {out_path}")

    return stress_onsets, stress_reliefs


def analyze_drawdown_prediction(features: pd.DataFrame, out_path: str):
    """Check if geometric features lead major drawdowns."""
    price = features['silver_close']
    drawdown = features['drawdown']

    # Define major drawdowns (> 15%)
    major_dd = drawdown < -0.15
    dd_events = major_dd[major_dd].index

    print(f"\n=== Drawdown Prediction Analysis ===")
    print(f"Major drawdowns (>15%, rolling): {len(dd_events)} events")

    if len(dd_events) == 0:
        print("No major drawdowns in dataset")
        return

    # For each drawdown, look back at geometric features
    lookback_days = [5, 10, 20, 30]
    results = []

    ricci_col = _pick_col(
        features,
        [
            "ricci_p10_core_60d",
            "ricci_min_core_60d",
            "ricci_p10_60d",
            "ricci_min_60d",
            "ricci_mean_core_60d",
            "ricci_mean_60d",
        ],
    )
    mst_col = _pick_col(features, ["mst_stress_core_60d", "mst_stress_60d"])

    for dd_date in dd_events:
        idx = features.index.get_loc(dd_date)

        for lb in lookback_days:
            if idx - lb < 0:
                continue

            lookback_idx = idx - lb
            lookback_date = features.index[lookback_idx]

            results.append({
                'drawdown_date': dd_date,
                'lookback_days': lb,
                'lookback_date': lookback_date,
                'ricci_signal': features.loc[lookback_date, ricci_col] if ricci_col else np.nan,
                'ricci_signal_col': ricci_col or "",
                'mst_signal': features.loc[lookback_date, mst_col] if mst_col else np.nan,
                'mst_signal_col': mst_col or "",
                'ga_rotor': features.loc[lookback_date, 'ga_rotor_magnitude_60d'],
                'realized_vol': features.loc[lookback_date, 'realized_vol_20d'],
                'actual_dd': features.loc[dd_date, 'drawdown']
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_path, index=False)

    # Correlation analysis
    print("\nCorrelation: Geometric Features vs Future Drawdown")
    for lb in lookback_days:
        subset = results_df[results_df['lookback_days'] == lb]
        if len(subset) > 5:
            corr_ricci = subset[['ricci_signal', 'actual_dd']].corr().iloc[0, 1]
            corr_mst = subset[['mst_signal', 'actual_dd']].corr().iloc[0, 1]
            print(f"  {lb:2d}-day lookback: ricci_signal corr = {corr_ricci:+.3f}, mst_signal corr = {corr_mst:+.3f}")

    print(f"Saved: {out_path}")


def feature_correlation_analysis(features: pd.DataFrame, out_path: str):
    """Analyze correlation structure of geometric features."""
    geom_features = [
        # Full manifold
        'mst_stress_60d', 'ricci_mean_60d', 'ricci_min_60d', 'ricci_p10_60d', 'ricci_std_60d',
        # Core manifold
        'mst_stress_core_60d', 'ricci_mean_core_60d', 'ricci_min_core_60d', 'ricci_p10_core_60d', 'ricci_std_core_60d',
        # Pairwise
        'dist_silver_gold_60d', 'corr_silver_gold_60d', 'dist_silver_dxy_60d', 'corr_silver_dxy_60d',
        # GA
        'ga_rotor_magnitude_60d', 'ga_bivector_energy_60d',
    ]

    existing = [c for c in geom_features if c in features.columns]
    corr_matrix = features[existing].corr()

    print(f"\n=== Geometric Feature Correlation Matrix ===")
    print(corr_matrix)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Geometric Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(out_path.replace('.csv', '.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path.replace('.csv', '.png')}")

    corr_matrix.to_csv(out_path)
    print(f"Saved: {out_path}")


def plot_geometric_signal_timeline(features: pd.DataFrame, out_path: str):
    """Plot all geometric signals on one chart for visual inspection."""
    has_sg = "dist_silver_gold_60d" in features.columns
    nrows = 6 if has_sg else 5
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 13 if has_sg else 12), sharex=True)

    # Price + drawdown zones
    ax0 = axes[0]
    ax0.plot(features.index, features['silver_close'], color='black', linewidth=0.8)
    major_dd = features['drawdown'] < -0.15
    if major_dd.any():
        ax0.fill_between(features.index, features['silver_close'].min(), features['silver_close'].max(),
                        where=major_dd, alpha=0.3, color='red', label='Major Drawdown >15% (rolling)')
    ax0.set_ylabel('Silver Price')
    ax0.set_title('Silver Price & Geometric Signals Timeline')
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)

    ricci_col = _pick_col(
        features,
        [
            "ricci_p10_core_60d",
            "ricci_min_core_60d",
            "ricci_p10_60d",
            "ricci_min_60d",
            "ricci_mean_core_60d",
            "ricci_mean_60d",
        ],
    )
    mst_col = _pick_col(features, ["mst_stress_core_60d", "mst_stress_60d"])

    # Ricci curvature (distributional stress shading)
    ax1 = axes[1]
    if ricci_col:
        ricci = features[ricci_col]
        ax1.plot(features.index, ricci, color="blue", linewidth=0.8)
        stress_q = 0.10
        thr = float(ricci.dropna().quantile(stress_q)) if ricci.notna().any() else np.nan
        if np.isfinite(thr):
            ax1.axhline(thr, color="red", linestyle="--", linewidth=0.9, label=f"Stress threshold (q={stress_q:.0%})")
            ax1.fill_between(
                features.index,
                ricci,
                thr,
                where=(ricci <= thr).fillna(False),
                color="red",
                alpha=0.25,
                label="Stress (tail curvature)",
            )
        ax1.set_ylabel("Ricci")
        ax1.set_title(f"Forman-Ricci Curvature (signal: {ricci_col})")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.set_axis_off()

    # MST Stress (high-stress shading)
    ax2 = axes[2]
    if mst_col:
        mst = features[mst_col]
        ax2.plot(features.index, mst, color="purple", linewidth=0.8)
        high_q = 0.90
        thr = float(mst.dropna().quantile(high_q)) if mst.notna().any() else np.nan
        if np.isfinite(thr):
            ax2.axhline(thr, color="red", linestyle="--", linewidth=0.9, label=f"High-stress threshold (q={high_q:.0%})")
            ax2.fill_between(
                features.index,
                thr,
                mst,
                where=(mst >= thr).fillna(False),
                color="red",
                alpha=0.20,
                label="High MST stress",
            )
        ax2.set_ylabel("MST Stress")
        ax2.set_title(f"Minimum Spanning Tree Stress (signal: {mst_col})")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.set_axis_off()

    next_ax = 3

    # Silver–Gold pairwise distance (if available)
    if has_sg:
        ax_sg = axes[next_ax]
        dist = features["dist_silver_gold_60d"]
        ax_sg.plot(features.index, dist, color="teal", linewidth=0.8)
        ax_sg.set_ylabel("Distance")
        ax_sg.set_title("Silver–Gold Correlation Distance (60d)")
        ax_sg.grid(True, alpha=0.3)
        next_ax += 1

    # GA Bivector Energy
    ax3 = axes[next_ax]
    ax3.plot(features.index, features['ga_bivector_energy_60d'], color='orange', linewidth=0.8)
    ax3.set_ylabel('Bivector Energy')
    ax3.set_title('Geometric Algebra: Bivector Energy (Rotational Activity)')
    ax3.grid(True, alpha=0.3)
    next_ax += 1

    # Realized Vol (for comparison)
    ax4 = axes[next_ax]
    ax4.plot(features.index, features['realized_vol_20d'], color='green', linewidth=0.8)
    ax4.set_ylabel('Realized Vol (20d)')
    ax4.set_title('Traditional: Realized Volatility')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze geometric signals")
    parser.add_argument('--features', required=True, help='Path to silver_features_*.parquet')
    parser.add_argument('--out-dir', default='reports/silver/signals', help='Output directory')
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    print(f"Loaded features: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'tables'), exist_ok=True)

    # 1. Curvature turning points
    print("\n" + "="*60)
    print("ANALYSIS 1: Ricci Curvature Turning Points")
    print("="*60)
    detect_curvature_turning_points(
        features,
        out_path=os.path.join(args.out_dir, 'tables', 'curvature_turning_points.csv')
    )

    # 2. Drawdown prediction
    print("\n" + "="*60)
    print("ANALYSIS 2: Geometric Features as Drawdown Predictors")
    print("="*60)
    analyze_drawdown_prediction(
        features,
        out_path=os.path.join(args.out_dir, 'tables', 'drawdown_prediction_analysis.csv')
    )

    # 3. Feature correlation
    print("\n" + "="*60)
    print("ANALYSIS 3: Geometric Feature Correlation Structure")
    print("="*60)
    feature_correlation_analysis(
        features,
        out_path=os.path.join(args.out_dir, 'tables', 'geometric_feature_correlations.csv')
    )

    # 4. Timeline plot
    print("\n" + "="*60)
    print("VISUALIZATION: Geometric Signal Timeline")
    print("="*60)
    plot_geometric_signal_timeline(
        features,
        out_path=os.path.join(args.out_dir, 'figures', 'geometric_signals_timeline.png')
    )

    print(f"\n{'='*60}")
    print("GEOMETRIC SIGNAL ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Outputs in: {args.out_dir}")


if __name__ == '__main__':
    main()
