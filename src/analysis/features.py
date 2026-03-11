from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.analysis.trend import drawdown, rolling_drawdown
from src.geometry.graph_curvature import (
    rolling_mst_stress,
    rolling_ricci_curvature,
    rolling_sectional_ricci_curvature,
    compute_manifold_stress_differential,
)

# Import macro features (optional, for enhanced analysis)
try:
    from src.analysis.macro_features import compute_all_macro_features
    HAS_MACRO_FEATURES = True
except ImportError:
    HAS_MACRO_FEATURES = False

# Import fluid dynamics features (optional, for SPIKAN integration)
try:
    from src.analysis.fluid_dynamics import (
        compute_fluid_dynamics_features,
        FluidDynamicsConfig,
    )
    HAS_FLUID_DYNAMICS = True
except ImportError:
    HAS_FLUID_DYNAMICS = False


@dataclass(frozen=True)
class FeatureConfig:
    realized_vol_window_days: int = 20
    momentum_window_days: int = 10
    beta_window_days: int = 60
    mst_window_days: int = 60


def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    prev = close.shift(1)
    valid = (close > 0) & (prev > 0)
    out = pd.Series(index=close.index, dtype="float64")
    out.loc[valid] = np.log(close.loc[valid] / prev.loc[valid])
    return out.replace([np.inf, -np.inf], np.nan)


def rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS slope beta of y ~ a + b*x, computed via cov/var.
    """
    y = y.astype(float)
    x = x.astype(float)
    cov = y.rolling(window=window, min_periods=window).cov(x)
    var = x.rolling(window=window, min_periods=window).var()
    beta = cov / var
    beta = beta.replace([np.inf, -np.inf], np.nan)
    return beta


def realized_volatility(log_returns: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(annualization)


def compute_silver_features(
    close_panel: pd.DataFrame,
    *,
    silver_symbol: str,
    gold_symbol: Optional[str] = None,
    dxy_symbol: Optional[str] = None,
    y10_symbol: Optional[str] = None,
    spx_symbol: Optional[str] = None,
    vix_symbol: Optional[str] = None,
    config: FeatureConfig = FeatureConfig(),
    include_macro_features: bool = False,
    include_fluid_dynamics: bool = False,
    manifolds: Optional[Dict[str, list[str]]] = None,
) -> pd.DataFrame:
    """
    Computes a thesis-friendly feature set for silver trend + regime work.

    `close_panel` is expected to be an aligned close-price matrix (date index, symbol columns).
    """
    panel = close_panel.copy()
    panel.index = pd.to_datetime(panel.index, errors="coerce").normalize()
    panel = panel.sort_index()

    out = pd.DataFrame(index=panel.index)

    if silver_symbol not in panel.columns:
        raise KeyError(f"silver_symbol `{silver_symbol}` not found in close_panel columns")

    silver_close = panel[silver_symbol].astype(float).rename("silver_close")
    out["silver_close"] = silver_close

    silver_lr = compute_log_returns(silver_close).rename("log_return_1d")
    out["log_return_1d"] = silver_lr

    out["realized_vol_20d"] = realized_volatility(silver_lr, window=config.realized_vol_window_days)
    out["momentum_10d"] = np.log(silver_close / silver_close.shift(config.momentum_window_days))
    # Drawdowns:
    # - `drawdown`: rolling 252d drawdown (more diagnostic for regime timing)
    # - `drawdown_ath`: all-time-high drawdown (structural perspective)
    out["drawdown_ath"] = drawdown(silver_close)
    out["drawdown_63d"] = rolling_drawdown(silver_close, 63)
    out["drawdown_126d"] = rolling_drawdown(silver_close, 126)
    out["drawdown_252d"] = rolling_drawdown(silver_close, 252)
    out["drawdown"] = out["drawdown_252d"]

    if gold_symbol and gold_symbol in panel.columns:
        gold_close = panel[gold_symbol].astype(float).rename("gold_close")
        out["gold_close"] = gold_close
        out["gsr"] = gold_close / silver_close
        out["gsr_log"] = np.log(out["gsr"])
        gold_lr = compute_log_returns(gold_close).rename("gold_log_return_1d")
        out["gold_log_return_1d"] = gold_lr

        # Pairwise geometry: silver vs gold (event narrative alignment).
        corr_sg = silver_lr.rolling(window=config.mst_window_days, min_periods=config.mst_window_days).corr(gold_lr)
        corr_sg = corr_sg.clip(lower=-1.0, upper=1.0)
        out["corr_silver_gold_60d"] = corr_sg
        out["dist_silver_gold_60d"] = np.sqrt(2.0 * (1.0 - corr_sg))

    if dxy_symbol and dxy_symbol in panel.columns:
        dxy = panel[dxy_symbol].astype(float).rename("dxy_level")
        out["dxy_level"] = dxy
        out["dxy_log_return_1d"] = compute_log_returns(dxy)
        out["dxy_beta_60d"] = rolling_beta(out["log_return_1d"], out["dxy_log_return_1d"], config.beta_window_days)
        corr_sdxy = silver_lr.rolling(window=config.mst_window_days, min_periods=config.mst_window_days).corr(
            out["dxy_log_return_1d"]
        )
        corr_sdxy = corr_sdxy.clip(lower=-1.0, upper=1.0)
        out["corr_silver_dxy_60d"] = corr_sdxy
        out["dist_silver_dxy_60d"] = np.sqrt(2.0 * (1.0 - corr_sdxy))

    if y10_symbol and y10_symbol in panel.columns:
        y10 = panel[y10_symbol].astype(float).rename("y10_level")
        out["y10_level"] = y10
        out["y10_change_1d"] = y10.diff()

    if spx_symbol and spx_symbol in panel.columns:
        spx = panel[spx_symbol].astype(float).rename("spx_level")
        out["spx_level"] = spx
        out["spx_log_return_1d"] = compute_log_returns(spx)

    if vix_symbol and vix_symbol in panel.columns:
        vix = panel[vix_symbol].astype(float).rename("vix_level")
        out["vix_level"] = vix
        out["vix_log_return_1d"] = compute_log_returns(vix)

    # Market topology stress (geometry proxy). Use available log returns across symbols.
    returns = panel.apply(compute_log_returns, axis=0)
    returns = returns.dropna(axis=1, how="all")
    if returns.shape[1] >= 3:
        # MST stress: graph topology measure (sum of MST edge weights)
        out["mst_stress_60d"] = rolling_mst_stress(
            returns,
            window=config.mst_window_days,
            min_assets=3,
        )

        # Forman-Ricci curvature: TRUE differential geometric curvature
        # Positive = stable/clustered, Negative = stressed/bottleneck
        ricci_df = rolling_ricci_curvature(
            returns,
            window=config.mst_window_days,
            min_assets=3,
        )
        out["ricci_mean_60d"] = ricci_df["ricci_mean"]
        out["ricci_min_60d"] = ricci_df["ricci_min"]
        if "ricci_p10" in ricci_df.columns:
            out["ricci_p10_60d"] = ricci_df["ricci_p10"]
        if "ricci_p90" in ricci_df.columns:
            out["ricci_p90_60d"] = ricci_df["ricci_p90"]
        out["ricci_std_60d"] = ricci_df["ricci_std"]

    # Core-manifold geometry (aligns better with silver–gold–macro narratives).
    core_candidates = [silver_symbol, gold_symbol, dxy_symbol, y10_symbol, spx_symbol, vix_symbol]
    core_cols = [c for c in core_candidates if c and c in panel.columns]
    core_cols = list(dict.fromkeys(core_cols))  # dedupe, preserve order

    if len(core_cols) >= 2:
        core_returns = panel[core_cols].apply(compute_log_returns, axis=0).dropna(axis=1, how="all")
        out["mst_stress_core_60d"] = rolling_mst_stress(
            core_returns,
            window=config.mst_window_days,
            min_assets=2,
        )

        ricci_core = rolling_ricci_curvature(
            core_returns,
            window=config.mst_window_days,
            min_assets=2,
        )
        out["ricci_mean_core_60d"] = ricci_core["ricci_mean"]
        out["ricci_min_core_60d"] = ricci_core["ricci_min"]
        if "ricci_p10" in ricci_core.columns:
            out["ricci_p10_core_60d"] = ricci_core["ricci_p10"]
        if "ricci_p90" in ricci_core.columns:
            out["ricci_p90_core_60d"] = ricci_core["ricci_p90"]
        out["ricci_std_core_60d"] = ricci_core["ricci_std"]

    # Optional: Sectional curvature for manifold-specific analysis (Phase 2)
    if manifolds is not None and len(manifolds) > 0:
        print("Computing sectional curvature for asset manifolds...", file=__import__('sys').stderr)
        returns = panel.apply(compute_log_returns, axis=0).dropna(axis=1, how="all")

        if returns.shape[1] >= 3:
            # Sectional Ricci curvature
            sectional_ricci = rolling_sectional_ricci_curvature(
                returns,
                manifolds=manifolds,
                window=config.mst_window_days,
                min_assets=2,
            )

            # Add to features (avoid duplicates) - concat once to avoid fragmentation
            sec_cols = [c for c in sectional_ricci.columns if c not in out.columns]
            if sec_cols:
                out = pd.concat([out, sectional_ricci[sec_cols]], axis=1)

            # MST stress differentials
            stress_diff = compute_manifold_stress_differential(
                returns,
                manifolds=manifolds,
                window=config.mst_window_days,
                min_assets=2,
            )

            # Add stress differentials (avoid duplicates) - concat once to avoid fragmentation
            diff_cols = [c for c in stress_diff.columns if c not in out.columns]
            if diff_cols:
                out = pd.concat([out, stress_diff[diff_cols]], axis=1)

            print(
                f"  Added {len(sec_cols)} sectional curvature features "
                f"and {len(diff_cols)} stress-differential features",
                file=__import__('sys').stderr,
            )

    # Optional: Enhanced macro features (treasury spreads, credit spreads, ratios)
    if include_macro_features and HAS_MACRO_FEATURES:
        print("Computing enhanced macro features (treasury spreads, credit, ratios)...", file=__import__('sys').stderr)
        macro_features = compute_all_macro_features(panel)

        # Remove duplicate columns (base features take precedence)
        duplicate_cols = [col for col in macro_features.columns if col in out.columns]
        if duplicate_cols:
            print(f"  Removing {len(duplicate_cols)} duplicate columns from macro features: {duplicate_cols[:5]}...", file=__import__('sys').stderr)
            macro_features = macro_features.drop(columns=duplicate_cols)

        out = pd.concat([out, macro_features], axis=1)
    elif include_macro_features and not HAS_MACRO_FEATURES:
        print("Warning: Macro features requested but src.analysis.macro_features not available", file=__import__('sys').stderr)

    # Optional: Fluid dynamics features (SPIKAN integration)
    if include_fluid_dynamics and HAS_FLUID_DYNAMICS:
        print("Computing fluid dynamics features (shock formation, momentum decay)...", file=__import__('sys').stderr)
        fluid_config = FluidDynamicsConfig(
            fast_window=5,
            medium_window=20,
            slow_window=config.mst_window_days,
        )
        fluid_features = compute_fluid_dynamics_features(
            close_panel=panel,
            silver_symbol=silver_symbol,
            gold_symbol=gold_symbol,
            dxy_symbol=dxy_symbol,
            geometric_features=out,  # Pass existing features for Ricci/MST integration
            config=fluid_config,
        )

        # Remove duplicate columns (base features take precedence)
        duplicate_cols = [col for col in fluid_features.columns if col in out.columns]
        if duplicate_cols:
            fluid_features = fluid_features.drop(columns=duplicate_cols)

        out = pd.concat([out, fluid_features], axis=1)
        print(f"  Added {len(fluid_features.columns)} fluid dynamics features", file=__import__('sys').stderr)
    elif include_fluid_dynamics and not HAS_FLUID_DYNAMICS:
        print("Warning: Fluid dynamics requested but src.analysis.fluid_dynamics not available", file=__import__('sys').stderr)

    return out


def feature_metadata(
    *,
    silver_symbol: str,
    gold_symbol: Optional[str] = None,
    dxy_symbol: Optional[str] = None,
    y10_symbol: Optional[str] = None,
    spx_symbol: Optional[str] = None,
    vix_symbol: Optional[str] = None,
    config: FeatureConfig = FeatureConfig(),
    include_macro_features: bool = False,
    include_fluid_dynamics: bool = False,
    manifolds: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, object]:
    return {
        "symbols": {
            "silver": silver_symbol,
            "gold": gold_symbol,
            "dxy": dxy_symbol,
            "y10": y10_symbol,
            "spx": spx_symbol,
            "vix": vix_symbol,
        },
        "windows": {
            "realized_vol_days": config.realized_vol_window_days,
            "momentum_days": config.momentum_window_days,
            "beta_days": config.beta_window_days,
            "mst_days": config.mst_window_days,
        },
        "enhancements": {
            "macro_features": include_macro_features,
            "fluid_dynamics": include_fluid_dynamics,
            "manifolds": list(manifolds.keys()) if manifolds else None,
        },
    }
