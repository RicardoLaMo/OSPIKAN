import os
import sys

import pandas as pd

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.data_quality import missingness_summary, validate_panel


def test_missingness_summary_counts():
    df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [None, None, 1.0]})
    s = missingness_summary(df)
    assert s.rows == 3
    assert s.cols == 2
    assert s.missing_cells == 3
    assert abs(s.missing_rate - 0.5) < 1e-12
    assert s.per_column_missing_rate["a"] == 1 / 3
    assert s.per_column_missing_rate["b"] == 2 / 3


def test_validate_panel_required_columns_and_threshold():
    panel = pd.DataFrame(
        {"AAA": [1.0, None, 3.0], "BBB": [10.0, 11.0, 12.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    out = validate_panel(panel, required_columns=["AAA", "BBB"], max_missing_rate=0.10)
    assert out["ok"] is False

    # Now allow missingness.
    out2 = validate_panel(panel, required_columns=["AAA", "BBB"], max_missing_rate=0.20)
    assert out2["ok"] is True
