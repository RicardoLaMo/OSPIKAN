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

