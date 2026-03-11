import argparse
import os
import sys

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.analysis.regimes import fit_markov_regression, markov_regime_labels, markov_smoothed_probabilities
from src.analysis.visualization import plot_price_with_regimes, plot_regime_probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Markov-switching regimes and export probabilities/plots.")
    parser.add_argument("--features", required=True, help="Path to `data/processed/silver_features_*.parquet`")
    parser.add_argument("--k", type=int, default=3, help="Number of regimes.")
    parser.add_argument("--out-dir", default="reports/silver", help="Output directory under reports.")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    if "log_return_1d" not in df.columns or "silver_close" not in df.columns:
        raise KeyError("Expected `log_return_1d` and `silver_close` columns in features.")

    out_fig = os.path.join(args.out_dir, "figures")
    out_tbl = os.path.join(args.out_dir, "tables")
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_tbl, exist_ok=True)

    endog = df["log_return_1d"]

    # ENHANCEMENT: Include geometric features as exogenous variables
    # Traditional macro features
    macro_cols = [c for c in ["dxy_log_return_1d", "y10_change_1d", "spx_log_return_1d"] if c in df.columns]

    # Geometric features (differential geometry + Clifford algebra)
    geom_cols = []

    # Prefer core-manifold geometry (silver, gold, macro) if present.
    if "mst_stress_core_60d" in df.columns:
        geom_cols.append("mst_stress_core_60d")
    elif "mst_stress_60d" in df.columns:
        geom_cols.append("mst_stress_60d")

    if "ricci_p10_core_60d" in df.columns:
        geom_cols.append("ricci_p10_core_60d")
    elif "ricci_p10_60d" in df.columns:
        geom_cols.append("ricci_p10_60d")
    elif "ricci_min_core_60d" in df.columns:
        geom_cols.append("ricci_min_core_60d")
    elif "ricci_min_60d" in df.columns:
        geom_cols.append("ricci_min_60d")

    # Pairwise manifold distances (aligns with silver–gold narrative).
    for c in ["dist_silver_gold_60d", "dist_silver_dxy_60d"]:
        if c in df.columns:
            geom_cols.append(c)

    # GA features.
    for c in ["ga_rotor_magnitude_60d", "ga_bivector_energy_60d"]:
        if c in df.columns:
            geom_cols.append(c)

    exog_cols = macro_cols + geom_cols
    exog = df[exog_cols].dropna(axis=1, how="all") if exog_cols else None

    if exog is not None and not exog.empty:
        print(f"Using {len(exog_cols)} exogenous features:")
        print(f"  Macro: {macro_cols}")
        print(f"  Geometric: {geom_cols}")
    else:
        print("Warning: No exogenous features available (check feature computation)")

    res = fit_markov_regression(
        endog,
        exog=exog,
        k_regimes=args.k,
        switching_variance=True,
        trend="c",
        maxiter=200,
    )

    probs = markov_smoothed_probabilities(res)
    labels = markov_regime_labels(probs)

    probs_path = os.path.join(out_tbl, f"markov_probs_k{args.k}.csv")
    labels_path = os.path.join(out_tbl, f"markov_labels_k{args.k}.csv")
    params_path = os.path.join(out_tbl, f"markov_params_k{args.k}.txt")

    probs.to_csv(probs_path)
    labels.to_csv(labels_path)
    with open(params_path, "w", encoding="utf-8") as f:
        f.write(str(res.summary()) + "\n")

    plot_regime_probabilities(
        probs,
        title=f"Markov Regime Probabilities (k={args.k})",
        out_path=os.path.join(out_fig, f"markov_probs_k{args.k}.png"),
    )

    plot_price_with_regimes(
        df["silver_close"],
        labels.reindex(df.index),
        title=f"Silver Price with Markov Regimes (k={args.k})",
        out_path=os.path.join(out_fig, f"silver_price_markov_k{args.k}.png"),
        ylabel="Silver",
    )

    print(probs_path)
    print(labels_path)
    print(params_path)


if __name__ == "__main__":
    main()
