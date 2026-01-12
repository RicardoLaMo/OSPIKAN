import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.graph_curvature import rolling_mst_stress


def test_rolling_mst_stress_perfect_correlation_zero_stress():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    base = np.random.default_rng(0).normal(0.0, 0.01, size=len(idx))

    returns = pd.DataFrame(
        {
            "A": base,
            "B": base,
            "C": base,
        },
        index=idx,
    )

    stress = rolling_mst_stress(returns, window=60, min_assets=3)
    last = float(stress.dropna().iloc[-1])
    assert abs(last - 0.0) < 1e-12

