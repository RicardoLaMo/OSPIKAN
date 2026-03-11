from __future__ import annotations

import numpy as np
import pandas as pd

from .trend import compute_log_returns, trend_regime_ma_cross


def realized_volatility(log_returns: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    return log_returns.rolling(window=window, min_periods=window).std() * np.sqrt(annualization)


def vol_regime_buckets(
    realized_vol: pd.Series,
    *,
    low: float = 0.15,
    high: float = 0.30,
) -> pd.Series:
    """
    Buckets annualized realized volatility into {low, medium, high}.
    """
    out = pd.Series(index=realized_vol.index, dtype="object")
    out.loc[realized_vol < low] = "low"
    out.loc[(realized_vol >= low) & (realized_vol <= high)] = "medium"
    out.loc[realized_vol > high] = "high"
    out = out.fillna("unknown")
    return out


def baseline_regime_table(
    close: pd.Series,
    *,
    fast_ma_days: int = 20,
    slow_ma_days: int = 60,
    realized_vol_window_days: int = 20,
    vol_low: float = 0.15,
    vol_high: float = 0.30,
) -> pd.DataFrame:
    """
    Minimal baseline regime table (trend x vol).
    """
    close = close.astype(float)
    log_ret_1d = compute_log_returns(close).rename("log_return_1d")
    rv_20d = realized_volatility(log_ret_1d, window=realized_vol_window_days).rename("realized_vol_20d")

    trend = trend_regime_ma_cross(close, fast_window=fast_ma_days, slow_window=slow_ma_days).rename("trend_regime")
    vol = vol_regime_buckets(rv_20d, low=vol_low, high=vol_high).rename("vol_regime")

    combined = (trend.astype(str) + "_" + vol.astype(str)).rename("regime")

    return pd.concat([close.rename("close"), log_ret_1d, rv_20d, trend, vol, combined], axis=1)


def fit_markov_regression(
    endog: pd.Series,
    *,
    exog: pd.DataFrame | None = None,
    k_regimes: int = 3,
    switching_variance: bool = True,
    trend: str = "c",
    maxiter: int = 200,
    standardize_exog: bool = True,
):
    """
    Fits a Markov-switching regression using statsmodels.

    Returns the fitted statsmodels results object.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    if exog is not None:
        df = pd.concat([endog.rename("_y"), exog], axis=1).dropna()
        y = df["_y"]
        X = df.drop(columns=["_y"]).astype(float)
        if bool(standardize_exog) and not X.empty:
            std = X.std(ddof=0).replace(0.0, np.nan)
            X = (X - X.mean()) / (std + 1e-12)
    else:
        y = endog.dropna()
        X = None

    model = MarkovRegression(
        y.astype(float),
        k_regimes=int(k_regimes),
        exog=X if X is not None else None,
        switching_variance=bool(switching_variance),
        trend=str(trend),
    )
    res = model.fit(disp=False, maxiter=int(maxiter))
    return res


def markov_smoothed_probabilities(results) -> pd.DataFrame:
    probs = results.smoothed_marginal_probabilities
    probs = probs.rename(columns={c: f"p_regime_{c}" for c in probs.columns})
    return probs


def markov_regime_labels(probs: pd.DataFrame) -> pd.Series:
    """
    Converts regime probabilities into hard labels via argmax.
    """
    if probs.empty:
        return pd.Series(dtype="int64")
    cols = list(probs.columns)
    labels = probs[cols].values.argmax(axis=1)
    return pd.Series(labels, index=probs.index, name="regime")


def geometric_regime_classification(
    features: pd.DataFrame,
    *,
    ricci_mean_col: str = "ricci_mean_60d",
    ricci_min_col: str = "ricci_min_60d",
    ricci_tail_col: str | None = None,
    rotor_col: str = "ga_rotor_magnitude_60d",
    mst_col: str = "mst_stress_60d",
    threshold_method: str = "quantile",
    stress_quantile: float = 0.10,
    transition_quantile: float = 0.90,
    mst_stress_quantile: float = 0.90,
    stress_threshold: float = -0.5,
    bottleneck_threshold: float = -1.0,
    transition_threshold: float = 0.8,
    mst_stress_threshold: float = 1.0,
) -> pd.Series:
    """
    Classifies regimes using pure differential geometric features.

    This is TRUE geometric regime detection, not just statistical clustering.
    Uses Forman-Ricci curvature + GA rotors to identify market states.

    Regime definitions:
        - STABLE: High curvature (positive), low MST stress, low rotor magnitude
                 = Tightly clustered, stable correlations, no major rotations
        - TRANSITION: Rising rotor magnitude, falling curvature
                     = Active regime shift in progress (rotation in feature space)
        - STRESS: Negative curvature, high MST stress, unstable geometry
                 = Fragmented market, bottlenecks, crisis-like structure
        - RECOVERY: Rising curvature from negative, decreasing stress
                   = Market structure healing after stress

    Args:
        features: DataFrame with geometric features
        ricci_mean_col: Column name for mean Ricci curvature
        ricci_min_col: Column name for min Ricci curvature (bottleneck detector)
        ricci_tail_col: Optional curvature tail column (preferred for stress), e.g. `ricci_p10_core_60d`
        rotor_col: Column name for GA rotor magnitude
        mst_col: Column name for MST stress
        threshold_method: `quantile` (recommended) or `zscore`
        stress_quantile: Stress threshold quantile for curvature tail (default 10%)
        transition_quantile: Transition threshold quantile for rotor magnitude (default 90%)
        mst_stress_quantile: High-stress threshold quantile for MST stress (default 90%)
        stress_threshold: Z-score threshold for negative curvature (default -0.5)
        bottleneck_threshold: Z-score threshold for bottleneck detection (default -1.0)
        transition_threshold: Z-score threshold for high rotor magnitude (default 0.8)
        mst_stress_threshold: Z-score threshold for high MST stress (default 1.0)

    Returns:
        Series of regime labels {STABLE, TRANSITION, STRESS, RECOVERY}

    Note:
        `quantile` thresholds are robust when curvature is negative throughout the
        sample (common with distance-weighted graphs). `zscore` is retained for
        backwards-compatibility and experimentation.
    """
    # Prefer core-manifold geometry if present (unless the caller overrides columns).
    if ricci_mean_col == "ricci_mean_60d" and "ricci_mean_core_60d" in features.columns:
        ricci_mean_col = "ricci_mean_core_60d"
    if ricci_min_col == "ricci_min_60d" and "ricci_min_core_60d" in features.columns:
        ricci_min_col = "ricci_min_core_60d"
    if mst_col == "mst_stress_60d" and "mst_stress_core_60d" in features.columns:
        mst_col = "mst_stress_core_60d"

    if ricci_tail_col is None:
        for c in ["ricci_p10_core_60d", "ricci_min_core_60d", "ricci_p10_60d", "ricci_min_60d"]:
            if c in features.columns:
                ricci_tail_col = c
                break
    ricci_tail_col = ricci_tail_col or ricci_min_col

    cols = list(dict.fromkeys([ricci_mean_col, ricci_min_col, ricci_tail_col, rotor_col, mst_col]))
    df = features[cols].copy()
    df = df.dropna()

    if df.empty:
        return pd.Series(index=features.index, dtype="object", name="geometric_regime")

    regime = pd.Series(index=df.index, dtype="object", name="geometric_regime")

    method = str(threshold_method).strip().lower()

    if method == "quantile":
        # Distributional thresholds (recommended).
        tail_thr = float(df[ricci_tail_col].quantile(stress_quantile))
        mst_thr = float(df[mst_col].quantile(mst_stress_quantile))
        rotor_thr = float(df[rotor_col].quantile(transition_quantile))

        stress = (df[ricci_tail_col] <= tail_thr) | (df[mst_col] >= mst_thr)
        transition = df[rotor_col] >= rotor_thr
        tail_change = df[ricci_tail_col].diff().fillna(0.0)
        tail_median = float(df[ricci_tail_col].quantile(0.50))

        for i in range(len(df)):
            if bool(stress.iloc[i]):
                regime.iloc[i] = "STRESS"
            elif bool(transition.iloc[i]):
                regime.iloc[i] = "TRANSITION"
            elif float(tail_change.iloc[i]) > 0 and float(df[ricci_tail_col].iloc[i]) < tail_median:
                regime.iloc[i] = "RECOVERY"
            else:
                regime.iloc[i] = "STABLE"

    else:
        # Z-score thresholds (legacy - DEPRECATED).
        import warnings
        warnings.warn(
            "threshold_method='zscore' is deprecated and may be removed in a future version. "
            "Use threshold_method='quantile' (default) for robust regime classification.",
            DeprecationWarning,
            stacklevel=2,
        )
        ricci_mean_z = (df[ricci_mean_col] - df[ricci_mean_col].mean()) / (df[ricci_mean_col].std() + 1e-8)
        ricci_min_z = (df[ricci_min_col] - df[ricci_min_col].mean()) / (df[ricci_min_col].std() + 1e-8)
        ricci_tail_z = (df[ricci_tail_col] - df[ricci_tail_col].mean()) / (df[ricci_tail_col].std() + 1e-8)
        rotor_z = (df[rotor_col] - df[rotor_col].mean()) / (df[rotor_col].std() + 1e-8)
        mst_z = (df[mst_col] - df[mst_col].mean()) / (df[mst_col].std() + 1e-8)

        ricci_change = df[ricci_mean_col].diff().fillna(0.0)

        for i in range(len(df)):
            r_mean = float(ricci_mean_z.iloc[i])
            r_min = float(ricci_min_z.iloc[i])
            r_tail = float(ricci_tail_z.iloc[i])
            rotor = float(rotor_z.iloc[i])
            mst = float(mst_z.iloc[i])
            curv_chg = float(ricci_change.iloc[i])

            if r_mean < stress_threshold or r_min < bottleneck_threshold or r_tail < bottleneck_threshold or mst > mst_stress_threshold:
                regime.iloc[i] = "STRESS"
            elif rotor > transition_threshold:
                regime.iloc[i] = "TRANSITION"
            elif curv_chg > 0 and r_mean > stress_threshold and r_mean < 0.5:
                regime.iloc[i] = "RECOVERY"
            else:
                regime.iloc[i] = "STABLE"

    # Forward-fill regime for missing dates in original index
    regime = regime.reindex(features.index).ffill().fillna("UNKNOWN")

    return regime


def enhanced_regime_with_fluid_dynamics(
    features: pd.DataFrame,
    *,
    base_regime_col: str = "geometric_regime",
    shock_formation_col: str = "shock_formation_index_z",
    momentum_decay_col: str = "momentum_decay_rate",
    shock_threshold: float = 2.0,
    decay_percentile_low: float = 0.25,
    decay_percentile_high: float = 0.75,
) -> pd.DataFrame:
    """
    Enhances regime classification with fluid dynamics signals.

    Adds shock and momentum indicators to existing geometric regimes:
    - SHOCK_IMMINENT: High shock formation index (from Burgers equation analysis)
    - MOMENTUM_PERSISTING: Low momentum decay rate (trending)
    - MOMENTUM_DECAYING: High momentum decay rate (mean-reverting)

    Args:
        features: DataFrame with geometric and fluid dynamics features
        base_regime_col: Column with base geometric regime (or pass None to compute)
        shock_formation_col: Column with shock formation index z-score
        momentum_decay_col: Column with momentum decay rate
        shock_threshold: Z-score threshold for shock detection
        decay_percentile_low: Percentile for "persisting" momentum
        decay_percentile_high: Percentile for "decaying" momentum

    Returns:
        DataFrame with enhanced regime columns:
        - geometric_regime: Base geometric regime
        - fluid_regime: Fluid dynamics regime signal
        - combined_regime: Combined geometric + fluid regime
        - shock_warning: Boolean shock warning flag
    """
    out = pd.DataFrame(index=features.index)

    # Get or compute base regime
    if base_regime_col in features.columns:
        out["geometric_regime"] = features[base_regime_col]
    else:
        out["geometric_regime"] = geometric_regime_classification(features)

    # Initialize fluid regime
    out["fluid_regime"] = "NORMAL"
    out["shock_warning"] = False

    # Check for fluid dynamics columns
    has_shock = shock_formation_col in features.columns
    has_decay = momentum_decay_col in features.columns

    if has_shock:
        shock_z = features[shock_formation_col]
        shock_mask = shock_z > shock_threshold
        out.loc[shock_mask, "fluid_regime"] = "SHOCK_IMMINENT"
        out.loc[shock_mask, "shock_warning"] = True

    if has_decay:
        decay = features[momentum_decay_col]
        decay_low = decay.quantile(decay_percentile_low)
        decay_high = decay.quantile(decay_percentile_high)

        # Only set momentum regime if not already shock
        persist_mask = (decay < decay_low) & (out["fluid_regime"] == "NORMAL")
        decay_mask = (decay > decay_high) & (out["fluid_regime"] == "NORMAL")

        out.loc[persist_mask, "fluid_regime"] = "MOMENTUM_PERSISTING"
        out.loc[decay_mask, "fluid_regime"] = "MOMENTUM_DECAYING"

    # Create combined regime
    def combine_regimes(row):
        geo = str(row["geometric_regime"])
        fluid = str(row["fluid_regime"])

        if fluid == "SHOCK_IMMINENT":
            # Shock signal overrides other states
            return "SHOCK_STRESS"
        elif geo == "STRESS":
            # Geometric stress takes priority
            if fluid == "MOMENTUM_DECAYING":
                return "STRESS_REVERTING"
            return "STRESS"
        elif geo == "TRANSITION":
            if fluid == "MOMENTUM_PERSISTING":
                return "TRANSITION_TRENDING"
            return "TRANSITION"
        elif geo == "RECOVERY":
            if fluid == "MOMENTUM_PERSISTING":
                return "RECOVERY_STRONG"
            return "RECOVERY"
        elif geo == "STABLE":
            if fluid == "MOMENTUM_PERSISTING":
                return "STABLE_TRENDING"
            elif fluid == "MOMENTUM_DECAYING":
                return "STABLE_REVERTING"
            return "STABLE"
        else:
            return geo

    out["combined_regime"] = out.apply(combine_regimes, axis=1)

    return out


def regime_transition_matrix(regime_series: pd.Series, normalize: bool = True) -> pd.DataFrame:
    """
    Computes regime transition matrix from a regime series.

    Args:
        regime_series: Series of regime labels
        normalize: If True, normalize to transition probabilities

    Returns:
        DataFrame with transition counts or probabilities
    """
    regimes = regime_series.dropna()
    if len(regimes) < 2:
        return pd.DataFrame()

    states = sorted(regimes.unique())
    transitions = pd.DataFrame(0, index=states, columns=states, dtype=float)

    for i in range(len(regimes) - 1):
        from_state = regimes.iloc[i]
        to_state = regimes.iloc[i + 1]
        transitions.loc[from_state, to_state] += 1

    if normalize:
        row_sums = transitions.sum(axis=1)
        transitions = transitions.div(row_sums.replace(0, 1), axis=0)

    return transitions
