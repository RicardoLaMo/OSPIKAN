import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.features import compute_silver_features


def test_compute_silver_features_columns():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(0)

    # Create simple positive prices with small noise.
    silver = 25.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=len(idx))))
    gold = 2000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.005, size=len(idx))))
    dxy = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, size=len(idx))))

    panel = pd.DataFrame({"SI=F": silver, "GC=F": gold, "DX-Y.NYB": dxy}, index=idx)

    feats = compute_silver_features(panel, silver_symbol="SI=F", gold_symbol="GC=F", dxy_symbol="DX-Y.NYB")

    expected = {
        "silver_close",
        "log_return_1d",
        "realized_vol_20d",
        "momentum_10d",
        "drawdown",
        "drawdown_252d",
        "drawdown_ath",
        "gold_close",
        "gsr",
        "gsr_log",
        "gold_log_return_1d",
        "corr_silver_gold_60d",
        "dist_silver_gold_60d",
        "dxy_level",
        "dxy_log_return_1d",
        "dxy_beta_60d",
        "corr_silver_dxy_60d",
        "dist_silver_dxy_60d",
        "mst_stress_60d",
        "mst_stress_core_60d",
        "ricci_p10_60d",
        "ricci_p10_core_60d",
    }

    assert expected.issubset(set(feats.columns))
    assert feats.index.is_monotonic_increasing
