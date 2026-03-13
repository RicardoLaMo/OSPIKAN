import numpy as np

from src.paper.benchmark_utils import calibration_table, expected_calibration_error, summarize_binary_metrics


def test_calibration_table_has_counts():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    rows = calibration_table(y_true, y_prob, n_bins=4)
    assert rows
    assert sum(int(row["count"]) for row in rows) == 4


def test_expected_calibration_error_is_bounded():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.2, 0.7, 0.3, 0.6, 0.9, 0.1])
    ece = expected_calibration_error(y_true, y_prob, n_bins=3)
    assert 0.0 <= ece <= 1.0


def test_summarize_binary_metrics_returns_expected_fields():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.2, 0.7, 0.3, 0.6, 0.9, 0.1])
    metrics = summarize_binary_metrics(y_true, y_prob).as_dict()
    assert set(metrics) == {"auc", "precision_at_k", "brier", "ece", "n_obs", "n_positive"}
    assert metrics["n_obs"] == 6
    assert metrics["n_positive"] == 3
