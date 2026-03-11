import argparse
import os
import sys

import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.regime_impacts import conditional_beta, conditional_return_stats, transition_events
from src.analysis.regimes import (
    baseline_regime_table,
    geometric_regime_classification,
    enhanced_regime_with_fluid_dynamics,
    regime_transition_matrix,
)
from src.analysis.visualization import plot_price_with_regimes, plot_timeseries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a first-pass silver regime report from features.")
    parser.add_argument("--features", required=True, help="Path to `data/processed/silver_features_*.parquet`")
    parser.add_argument("--out-dir", default="reports/silver", help="Report output folder.")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    if "silver_close" not in df.columns:
        raise KeyError("Expected `silver_close` in features file.")

    out_fig = os.path.join(args.out_dir, "figures")
    out_tbl = os.path.join(args.out_dir, "tables")
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_tbl, exist_ok=True)

    # Baseline regimes (trend x vol) on the silver close.
    baseline = baseline_regime_table(df["silver_close"])
    regimes = baseline["regime"]

    # Figures
    plot_price_with_regimes(
        df["silver_close"],
        regimes,
        title="Silver Price with Baseline Regimes",
        out_path=os.path.join(out_fig, "silver_price_regimes.png"),
        ylabel="Silver",
    )

    if "mst_stress_60d" in df.columns:
        plot_timeseries(
            df["mst_stress_60d"],
            title="Market Topology Stress (Rolling MST length)",
            out_path=os.path.join(out_fig, "mst_stress_60d.png"),
            ylabel="Stress",
        )

    if "gsr" in df.columns:
        plot_price_with_regimes(
            df["gsr"],
            regimes,
            title="Gold:Silver Ratio with Baseline Regimes",
            out_path=os.path.join(out_fig, "gsr_regimes.png"),
            ylabel="GSR",
        )

    # Tables
    stats = conditional_return_stats(df["log_return_1d"], regimes)
    stats.to_csv(os.path.join(out_tbl, "conditional_return_stats.csv"))

    if "dxy_log_return_1d" in df.columns:
        betas = conditional_beta(df["log_return_1d"], df["dxy_log_return_1d"], regimes)
        betas.to_csv(os.path.join(out_tbl, "conditional_beta_dxy.csv"))

    transitions = transition_events(regimes)
    transitions.to_csv(os.path.join(out_tbl, "regime_transitions.csv"))

    # Geometric regime classification (if Ricci features available)
    ricci_cols = [c for c in df.columns if "ricci" in c.lower()]
    if ricci_cols:
        print("Computing geometric regime classification...")
        geo_regimes = geometric_regime_classification(df)

        plot_price_with_regimes(
            df["silver_close"],
            geo_regimes,
            title="Silver Price with Geometric Regimes",
            out_path=os.path.join(out_fig, "silver_price_geometric_regimes.png"),
            ylabel="Silver",
        )

        # Save geometric regime stats
        geo_stats = conditional_return_stats(df["log_return_1d"], geo_regimes)
        geo_stats.to_csv(os.path.join(out_tbl, "geometric_regime_return_stats.csv"))

        # Geometric regime transitions
        geo_trans = regime_transition_matrix(geo_regimes)
        geo_trans.to_csv(os.path.join(out_tbl, "geometric_regime_transitions.csv"))

    # Fluid dynamics features (if available)
    fluid_cols = [c for c in df.columns if any(f in c.lower() for f in ["shock", "momentum_decay", "viscosity", "flow"])]
    if fluid_cols:
        print("Generating fluid dynamics plots...")

        # Shock formation index
        if "shock_formation_index_z" in df.columns:
            plot_timeseries(
                df["shock_formation_index_z"],
                title="Shock Formation Index (Z-score)",
                out_path=os.path.join(out_fig, "shock_formation_index.png"),
                ylabel="SFI Z-score",
            )

        # Momentum decay rate
        if "momentum_decay_rate" in df.columns:
            plot_timeseries(
                df["momentum_decay_rate"],
                title="Momentum Decay Rate",
                out_path=os.path.join(out_fig, "momentum_decay_rate.png"),
                ylabel="Decay Rate",
            )

        # Effective viscosity
        if "effective_viscosity" in df.columns:
            plot_timeseries(
                df["effective_viscosity"],
                title="Effective Market Viscosity",
                out_path=os.path.join(out_fig, "effective_viscosity.png"),
                ylabel="Viscosity",
            )

        # Enhanced regime with fluid dynamics
        if ricci_cols:
            print("Computing enhanced regime with fluid dynamics...")
            enhanced = enhanced_regime_with_fluid_dynamics(df)

            plot_price_with_regimes(
                df["silver_close"],
                enhanced["combined_regime"],
                title="Silver Price with Combined Geometry+Fluid Regimes",
                out_path=os.path.join(out_fig, "silver_price_combined_regimes.png"),
                ylabel="Silver",
            )

            # Save enhanced regime stats
            enhanced_stats = conditional_return_stats(df["log_return_1d"], enhanced["combined_regime"])
            enhanced_stats.to_csv(os.path.join(out_tbl, "combined_regime_return_stats.csv"))

            # Shock warnings summary
            shock_dates = enhanced[enhanced["shock_warning"]].index
            if len(shock_dates) > 0:
                shock_summary = pd.DataFrame({
                    "date": shock_dates,
                    "silver_close": df.loc[shock_dates, "silver_close"],
                    "log_return_1d": df.loc[shock_dates, "log_return_1d"],
                })
                shock_summary.to_csv(os.path.join(out_tbl, "shock_warnings.csv"), index=False)
                print(f"  Found {len(shock_dates)} shock warning dates")

    print(f"Wrote figures to {out_fig}")
    print(f"Wrote tables to {out_tbl}")


if __name__ == "__main__":
    main()
