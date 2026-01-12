import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MissingnessSummary:
    rows: int
    cols: int
    missing_cells: int
    missing_rate: float
    per_column_missing_rate: Dict[str, float]


def normalize_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = pd.to_datetime(df.index, errors="coerce")
    df = df.loc[~idx.isna()].copy()
    df.index = idx[~idx.isna()].normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = df.index.name or "date"
    return df


def missingness_summary(df: pd.DataFrame) -> MissingnessSummary:
    if df.empty:
        return MissingnessSummary(
            rows=0,
            cols=0,
            missing_cells=0,
            missing_rate=0.0,
            per_column_missing_rate={},
        )

    missing = df.isna()
    missing_cells = int(missing.values.sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_rate = float(missing_cells / total_cells) if total_cells else 0.0

    per_col = {str(c): float(missing[c].mean()) for c in df.columns}
    return MissingnessSummary(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        missing_cells=missing_cells,
        missing_rate=missing_rate,
        per_column_missing_rate=per_col,
    )


def validate_panel(
    panel: pd.DataFrame,
    *,
    required_columns: Optional[List[str]] = None,
    max_missing_rate: Optional[float] = None,
) -> Dict[str, object]:
    """
    Validates an aligned price panel (e.g., close-price matrix).

    Returns a dict suitable for JSON logging.
    """
    results: Dict[str, object] = {"ok": True, "errors": [], "warnings": [], "summary": {}}

    if panel is None or panel.empty:
        results["ok"] = False
        results["errors"].append("panel_empty")
        return results

    panel = normalize_datetime_index(panel)
    summary = missingness_summary(panel)
    results["summary"] = {
        "rows": summary.rows,
        "cols": summary.cols,
        "missing_cells": summary.missing_cells,
        "missing_rate": summary.missing_rate,
        "per_column_missing_rate": summary.per_column_missing_rate,
    }

    if required_columns:
        missing_cols = [c for c in required_columns if c not in panel.columns]
        if missing_cols:
            results["ok"] = False
            results["errors"].append({"missing_required_columns": missing_cols})

    if max_missing_rate is not None and summary.missing_rate > max_missing_rate:
        results["ok"] = False
        results["errors"].append(
            {"missing_rate_exceeds_threshold": {"missing_rate": summary.missing_rate, "max": max_missing_rate}}
        )

    # Simple sanity checks for obviously broken series.
    zero_var = [str(c) for c in panel.columns if panel[c].nunique(dropna=True) <= 1]
    if zero_var:
        results["warnings"].append({"near_constant_columns": zero_var})

    return results


def realized_volatility(log_returns: pd.Series, window: int = 20, annualization: int = 252) -> pd.Series:
    return log_returns.rolling(window=window).std() * np.sqrt(annualization)


def write_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

