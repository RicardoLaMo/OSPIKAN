import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.regime_impacts import (
    conditional_beta,
    conditional_return_stats,
    event_study_mean_path,
    transition_events,
)


def test_conditional_return_stats_two_regimes():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    r = pd.Series([0.01] * 5 + [-0.01] * 5, index=idx)
    regimes = pd.Series([0] * 5 + [1] * 5, index=idx)

    out = conditional_return_stats(r, regimes)
    assert set(out.index) == {0, 1}
    assert out.loc[0, "n"] == 5
    assert out.loc[1, "n"] == 5


def test_transition_events_detects_switches():
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    regimes = pd.Series([0, 0, 1, 1, 0], index=idx)

    events = transition_events(regimes)
    assert list(events.columns) == ["from", "to"]
    assert len(events) == 2
    assert events.iloc[0]["from"] == 0
    assert events.iloc[0]["to"] == 1


def test_conditional_beta_runs():
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    x = pd.Series(np.linspace(-1, 1, len(idx)), index=idx)
    y = 2.0 * x + 0.1
    regimes = pd.Series([0] * 10 + [1] * 10, index=idx)

    out = conditional_beta(y, x, regimes)
    assert set(out.index) == {0, 1}
    assert np.isfinite(out.loc[0, "beta"])


def test_event_study_mean_path_has_zero_at_event_day_when_zero_return():
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    r = pd.Series(0.0, index=idx)
    events = [idx[25]]
    path = event_study_mean_path(r, events, pre=5, post=5)

    assert 0 in path.index
    assert path.loc[0] == 0.0

