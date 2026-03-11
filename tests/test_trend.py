import numpy as np
import pandas as pd

from src.analysis.trend import drawdown, rolling_drawdown


def test_rolling_drawdown_resets_relative_to_window():
    # ATH drawdown stays negative after an early peak that isn't revisited.
    close = pd.Series([100.0, 50.0, 60.0, 70.0], index=pd.date_range("2020-01-01", periods=4, freq="D"))

    dd_ath = drawdown(close)
    assert dd_ath.iloc[-1] < 0

    # Rolling drawdown can reset as the trailing window's peak moves forward.
    dd_2 = rolling_drawdown(close, window=2)
    assert np.isclose(dd_2.iloc[-1], 0.0)

