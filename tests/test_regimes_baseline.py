import os
import sys

import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.regimes import baseline_regime_table, vol_regime_buckets
from src.analysis.trend import trend_regime_ma_cross


def test_trend_regime_ma_cross_bull_bear():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")

    up = pd.Series(np.linspace(10.0, 20.0, len(idx)), index=idx)
    down = pd.Series(np.linspace(20.0, 10.0, len(idx)), index=idx)

    up_reg = trend_regime_ma_cross(up, fast_window=20, slow_window=60)
    down_reg = trend_regime_ma_cross(down, fast_window=20, slow_window=60)

    assert up_reg.iloc[-1] == "bull"
    assert down_reg.iloc[-1] == "bear"


def test_vol_regime_buckets():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    vol = pd.Series([0.10, 0.20, 0.40], index=idx)

    out = vol_regime_buckets(vol, low=0.15, high=0.30)
    assert list(out) == ["low", "medium", "high"]


def test_baseline_regime_table_schema():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    close = pd.Series(np.linspace(10.0, 20.0, len(idx)), index=idx)

    df = baseline_regime_table(close)
    assert set(["close", "log_return_1d", "realized_vol_20d", "trend_regime", "vol_regime", "regime"]).issubset(
        df.columns
    )

