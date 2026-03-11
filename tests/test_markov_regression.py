import os
import sys

import numpy as np
import pandas as pd
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.analysis.regimes import fit_markov_regression, markov_regime_labels, markov_smoothed_probabilities


def test_fit_markov_regression_runs_and_outputs_probs():
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    rng = np.random.default_rng(0)
    n = 240
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    states = np.zeros(n, dtype=int)
    states[n // 2 :] = 1
    means = np.where(states == 0, 0.001, -0.001)
    y = means + rng.normal(0.0, 0.01, size=n)
    endog = pd.Series(y, index=idx, name="silver_lr")

    res = fit_markov_regression(endog, k_regimes=2, switching_variance=True, trend="c", maxiter=50)
    probs = markov_smoothed_probabilities(res)

    assert probs.shape[1] == 2
    assert probs.index.is_monotonic_increasing

    labels = markov_regime_labels(probs)
    assert labels.name == "regime"
    assert set(labels.unique()).issubset({0, 1})
