from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MSTResult:
    stress: float
    n_nodes: int
    n_edges: int


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """
    Converts correlation to a Euclidean distance metric:
      d(x, y) = sqrt(2 * (1 - corr(x, y)))
    """
    corr = corr.clip(lower=-1.0, upper=1.0)
    dist = np.sqrt(2.0 * (1.0 - corr))
    return dist


def mst_stress_index(distance_matrix: pd.DataFrame) -> MSTResult:
    """
    Returns the MST length (sum of edge weights) as a global stress proxy.
    """
    import networkx as nx

    dm = distance_matrix.copy()
    dm = dm.loc[dm.index.intersection(dm.columns), dm.columns.intersection(dm.index)]
    dm = dm.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if dm.shape[0] < 2:
        return MSTResult(stress=float("nan"), n_nodes=int(dm.shape[0]), n_edges=0)

    # Build fully-connected weighted graph.
    assets = list(dm.columns)
    G = nx.Graph()
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            w = float(dm.iloc[i, j])
            if np.isfinite(w):
                G.add_edge(a, b, weight=w)

    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return MSTResult(stress=float("nan"), n_nodes=G.number_of_nodes(), n_edges=G.number_of_edges())

    mst = nx.minimum_spanning_tree(G, weight="weight")
    stress = float(sum(d["weight"] for _, _, d in mst.edges(data=True)))
    return MSTResult(stress=stress, n_nodes=mst.number_of_nodes(), n_edges=mst.number_of_edges())


def rolling_mst_stress(
    returns: pd.DataFrame,
    *,
    window: int = 60,
    min_assets: int = 3,
    method: str = "pearson",
) -> pd.Series:
    """
    Computes a rolling MST stress index on a returns panel.
    """
    returns = returns.copy().astype(float)
    returns = returns.dropna(axis=1, how="all")
    returns = returns.dropna(axis=0, how="all")
    returns = returns.sort_index()

    out = pd.Series(index=returns.index, dtype=float, name="mst_stress")

    if returns.shape[1] < min_assets:
        return out

    for end in range(window - 1, len(returns.index)):
        w = returns.iloc[end - window + 1 : end + 1].dropna(axis=1, how="any")
        if w.shape[1] < min_assets:
            continue
        corr = w.corr(method=method)
        dist = correlation_distance(corr)
        res = mst_stress_index(dist)
        out.iloc[end] = res.stress

    return out

