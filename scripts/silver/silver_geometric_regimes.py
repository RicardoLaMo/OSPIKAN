#!/usr/bin/env python3
"""
Silver Geometric Regime Analysis

Uses TRUE differential geometry (Forman-Ricci curvature) and Geometric Algebra
(Clifford rotors) to detect market regimes - not just linear algebra.

This script demonstrates the difference between statistical regime detection
and geometric regime detection.
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.regimes import geometric_regime_classification
from src.analysis.visualization import plot_price_with_regimes


def plot_geometric_features(features: pd.DataFrame, out_path: str):
    """
    Plots the geometric features time series to visualize their behavior.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    mst_col = "mst_stress_core_60d" if "mst_stress_core_60d" in features.columns else "mst_stress_60d"
    ricci_mean_col = "ricci_mean_core_60d" if "ricci_mean_core_60d" in features.columns else "ricci_mean_60d"
    ricci_tail_col = None
    for c in ["ricci_p10_core_60d", "ricci_min_core_60d", "ricci_p10_60d", "ricci_min_60d"]:
        if c in features.columns:
            ricci_tail_col = c
            break

    # MST Stress (graph topology)
    if mst_col in features.columns:
        axes[0].plot(features.index, features[mst_col], color="purple", linewidth=1)
        axes[0].set_ylabel("MST Stress")
        axes[0].set_title(f"Graph Topology: Minimum Spanning Tree Stress ({mst_col})")
        axes[0].grid(True, alpha=0.3)

    # Forman-Ricci Curvature (TRUE differential geometry)
    if ricci_mean_col in features.columns:
        axes[1].plot(features.index, features[ricci_mean_col], label="Mean Curvature", color="blue", linewidth=1)
        if ricci_tail_col:
            axes[1].plot(
                features.index,
                features[ricci_tail_col],
                label=f"Tail Curvature ({ricci_tail_col})",
                color="red",
                linewidth=1,
                alpha=0.7,
            )
            thr = float(features[ricci_tail_col].dropna().quantile(0.10)) if features[ricci_tail_col].notna().any() else None
            if thr is not None:
                axes[1].axhline(thr, color="red", linestyle="--", alpha=0.5, linewidth=0.8, label="Stress threshold (q=10%)")
        axes[1].set_ylabel("Ricci Curvature")
        axes[1].set_title("Forman-Ricci Curvature (Differential Geometry)")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

    # GA Rotor Magnitude (Clifford algebra rotation)
    if "ga_rotor_magnitude_60d" in features.columns:
        axes[2].plot(features.index, features["ga_rotor_magnitude_60d"], color="green", linewidth=1)
        axes[2].set_ylabel("Rotor Magnitude")
        axes[2].set_title("Geometric Algebra: Cl(4,0) Rotor Magnitude (Regime Rotation)")
        axes[2].grid(True, alpha=0.3)

    # GA Bivector Energy (rotation plane dynamics)
    if "ga_bivector_energy_60d" in features.columns:
        axes[3].plot(features.index, features["ga_bivector_energy_60d"], color="orange", linewidth=1)
        axes[3].set_ylabel("Bivector Energy")
        axes[3].set_title("Geometric Algebra: Bivector Energy (Rotational Activity)")
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def compare_regimes_table(features: pd.DataFrame, baseline_regimes: pd.Series, geometric_regimes: pd.Series, out_path: str):
    """
    Creates a comparison table of baseline vs geometric regime distributions.
    """
    comparison = pd.DataFrame({
        "date": features.index,
        "silver_close": features["silver_close"],
        "baseline_regime": baseline_regimes.reindex(features.index),
        "geometric_regime": geometric_regimes.reindex(features.index),
    })

    comparison.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print distribution summary
    print("\n=== Regime Distribution Comparison ===")
    print("\nBaseline Regimes (MA crossover x Volatility):")
    if not baseline_regimes.empty:
        print(baseline_regimes.value_counts())
    else:
        print("  (Not available)")

    print("\nGeometric Regimes (Ricci Curvature + GA Rotors):")
    print(geometric_regimes.value_counts())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Silver geometric regime analysis using Ricci curvature and GA rotors"
    )
    parser.add_argument("--features", required=True, help="Path to silver_features_*.parquet")
    parser.add_argument("--out-dir", default="reports/silver", help="Output directory")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    print(f"Loaded features: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    out_fig = os.path.join(args.out_dir, "figures")
    out_tbl = os.path.join(args.out_dir, "tables")
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_tbl, exist_ok=True)

    # Check for required geometric features (prefer core manifold if available)
    mst_col = "mst_stress_core_60d" if "mst_stress_core_60d" in df.columns else "mst_stress_60d"
    ricci_mean_col = "ricci_mean_core_60d" if "ricci_mean_core_60d" in df.columns else "ricci_mean_60d"
    ricci_tail_col = None
    for c in ["ricci_p10_core_60d", "ricci_min_core_60d", "ricci_p10_60d", "ricci_min_60d"]:
        if c in df.columns:
            ricci_tail_col = c
            break

    required_geom = [ricci_mean_col, mst_col, "ga_rotor_magnitude_60d", "ga_bivector_energy_60d"]
    if ricci_tail_col:
        required_geom.append(ricci_tail_col)

    missing_geom = [c for c in required_geom if c not in df.columns]

    if missing_geom:
        print(f"\nERROR: Missing geometric features: {missing_geom}")
        print("Please re-run the pipeline with updated feature computation.")
        sys.exit(1)

    # Plot geometric features
    print("\n=== Plotting Geometric Features ===")
    plot_geometric_features(
        df,
        out_path=os.path.join(out_fig, "geometric_features_timeseries.png")
    )

    # Compute geometric regime classification
    print("\n=== Computing Geometric Regime Classification ===")
    geometric_regimes = geometric_regime_classification(df)
    print(f"Geometric regimes computed: {len(geometric_regimes)} dates")

    # Save geometric regime labels
    labels_path = os.path.join(out_tbl, "geometric_regime_labels.csv")
    geometric_regimes.to_csv(labels_path)
    print(f"Saved: {labels_path}")

    # Plot price with geometric regimes
    if "silver_close" in df.columns:
        plot_price_with_regimes(
            df["silver_close"],
            geometric_regimes,
            title="Silver Price with Geometric Regimes (Ricci + GA)",
            out_path=os.path.join(out_fig, "silver_price_geometric_regimes.png"),
            ylabel="Silver Price",
        )

    # Baseline regime comparison (if available)
    from src.analysis.regimes import baseline_regime_table

    if "silver_close" in df.columns:
        print("\n=== Computing Baseline Regimes for Comparison ===")
        baseline_df = baseline_regime_table(df["silver_close"])
        baseline_regimes = baseline_df["regime"]

        compare_regimes_table(
            df,
            baseline_regimes,
            geometric_regimes,
            out_path=os.path.join(out_tbl, "regime_comparison.csv")
        )

    # Summary statistics by geometric regime
    print("\n=== Silver Returns by Geometric Regime ===")
    if "log_return_1d" in df.columns:
        regime_stats = df.groupby(geometric_regimes)["log_return_1d"].agg([
            ("mean_return", "mean"),
            ("volatility", "std"),
            ("sharpe", lambda x: x.mean() / x.std() if x.std() > 0 else np.nan),
            ("count", "count"),
        ])
        regime_stats["mean_return_annualized"] = regime_stats["mean_return"] * 252
        regime_stats["volatility_annualized"] = regime_stats["volatility"] * np.sqrt(252)

        stats_path = os.path.join(out_tbl, "geometric_regime_return_stats.csv")
        regime_stats.to_csv(stats_path)
        print(regime_stats)
        print(f"Saved: {stats_path}")

    print("\n=== Geometric Regime Analysis Complete ===")
    print(f"Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
