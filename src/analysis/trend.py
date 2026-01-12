from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float)
    return np.log(close / close.shift(1))


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

