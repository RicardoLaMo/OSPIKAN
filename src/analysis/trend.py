from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    prev = close.shift(1)
    valid = (close > 0) & (prev > 0)
    out = pd.Series(index=close.index, dtype="float64")
    out.loc[valid] = np.log(close.loc[valid] / prev.loc[valid])
    return out.replace([np.inf, -np.inf], np.nan)


def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def trend_regime_ma_cross(
    close: pd.Series,
    *,
    fast_window: int = 20,
    slow_window: int = 60,
) -> pd.Series:
    """
    Simple baseline trend regime:
    - bull: fast MA > slow MA
    - bear: fast MA < slow MA
    - neutral: insufficient data / tie
    """
    close = close.astype(float)
    fast = moving_average(close, fast_window)
    slow = moving_average(close, slow_window)
    diff = fast - slow

    out = pd.Series(index=close.index, dtype="object")
    out.loc[diff > 0] = "bull"
    out.loc[diff < 0] = "bear"
    out.loc[diff == 0] = "neutral"
    out = out.fillna("neutral")
    return out


def drawdown(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    peak = close.cummax()
    return close / peak - 1.0


def rolling_drawdown(close: pd.Series, window: int) -> pd.Series:
    """
    Drawdown from the trailing rolling high over `window` periods.

    This is more useful for regime timing than all-time-high drawdown, since it
    resets as the rolling window advances.
    """
    close = close.astype(float)
    peak = close.rolling(window=int(window), min_periods=1).max()
    return close / peak - 1.0
