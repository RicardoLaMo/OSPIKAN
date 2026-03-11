from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnStats:
    n: int
    mean: float
    vol: float
    skew: float
    kurt: float
    sharpe: float


def _annualize_sharpe(mean_daily: float, vol_daily: float, annualization: int = 252) -> float:
    if not np.isfinite(vol_daily) or vol_daily <= 0:
        return float("nan")
    return float((mean_daily / vol_daily) * np.sqrt(annualization))


def conditional_return_stats(
    returns: pd.Series,
    regimes: pd.Series,
    *,
    annualization: int = 252,
) -> pd.DataFrame:
    """
    Computes distribution stats of returns conditional on regime labels.
    """
    df = pd.concat([returns.rename("r"), regimes.rename("regime")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for regime, g in df.groupby("regime"):
        r = g["r"].astype(float)
        mean = float(r.mean())
        vol = float(r.std(ddof=1))
        rows.append(
            {
                "regime": regime,
                "n": int(r.shape[0]),
                "mean": mean,
                "vol": vol,
                "skew": float(r.skew()),
                "kurt": float(r.kurt()),
                "sharpe": _annualize_sharpe(mean, vol, annualization=annualization),
            }
        )

    out = pd.DataFrame(rows).set_index("regime").sort_index()
    return out


def conditional_beta(
    y: pd.Series,
    x: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Computes OLS beta of y ~ a + b*x within each regime.
    """
    df = pd.concat([y.rename("y"), x.rename("x"), regimes.rename("regime")], axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    rows = []
    for regime, g in df.groupby("regime"):
        xv = g["x"].astype(float)
        yv = g["y"].astype(float)
        var = float(xv.var(ddof=1))
        cov = float(xv.cov(yv))
        beta = cov / var if np.isfinite(var) and var > 0 else float("nan")
        alpha = float(yv.mean() - beta * xv.mean()) if np.isfinite(beta) else float("nan")
        rows.append({"regime": regime, "n": int(len(g)), "alpha": alpha, "beta": beta})

    return pd.DataFrame(rows).set_index("regime").sort_index()


def transition_events(regimes: pd.Series) -> pd.DataFrame:
    """
    Lists regime transition timestamps (t where regime[t] != regime[t-1]).
    """
    r = regimes.dropna()
    if r.empty:
        return pd.DataFrame(columns=["from", "to"])
    prev = r.shift(1)
    mask = (r != prev) & prev.notna()
    out = pd.DataFrame({"from": prev[mask], "to": r[mask]})
    out.index.name = "date"
    return out


def event_study_mean_path(
    returns: pd.Series,
    event_dates: Iterable[pd.Timestamp],
    *,
    pre: int = 20,
    post: int = 20,
) -> pd.Series:
    """
    Mean cumulative return path around events, indexed by relative day.
    """
    r = returns.dropna().astype(float)
    if r.empty:
        return pd.Series(dtype=float)

    event_dates = [pd.Timestamp(d).normalize() for d in event_dates]
    event_dates = [d for d in event_dates if d in r.index]
    if not event_dates:
        return pd.Series(dtype=float)

    paths = []
    for d in event_dates:
        loc = r.index.get_loc(d)
        start = max(0, loc - pre)
        end = min(len(r.index) - 1, loc + post)
        window = r.iloc[start : end + 1]

        # Align by relative day.
        rel = np.arange(start - loc, end - loc + 1)
        cum = (1.0 + window).cumprod() - 1.0
        paths.append(pd.Series(cum.values, index=rel))

    # Average across events (outer join).
    aligned = pd.concat(paths, axis=1)
    return aligned.mean(axis=1).sort_index()

