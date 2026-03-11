from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MSTResult:
    stress: float
    n_nodes: int
    n_edges: int


@dataclass(frozen=True)
class RicciCurvatureResult:
    """
    Result container for Ricci curvature computation on a graph.

    Attributes:
        mean_curvature: Average edge curvature (positive = stable/clustered)
        min_curvature: Most negative curvature (identifies bottlenecks)
        max_curvature: Most positive curvature (identifies tight clusters)
        curvature_std: Standard deviation of edge curvatures
        edge_curvatures: Dict mapping edge (u, v) to its curvature value
        n_edges: Number of edges analyzed
    """
    mean_curvature: float
    min_curvature: float
    max_curvature: float
    curvature_std: float
    edge_curvatures: Dict[Tuple[str, str], float]
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


def forman_ricci_curvature(G, weight: str = "weight") -> RicciCurvatureResult:
    """
    Computes Forman-Ricci curvature for all edges in a weighted graph.

    Forman-Ricci curvature is a discrete analog of Ricci curvature from differential geometry.
    For an edge e = (v, w):

        F(e) = w(e) * [ 4 - deg(v) - deg(w) + (sum of weights in triangles containing e) ]

    Interpretation:
        - Positive curvature: Edge is in a tightly clustered region (stable, low stress)
        - Negative curvature: Edge is a bottleneck or bridge (fragile, high stress)
        - Zero curvature: Flat geometry (random-like connectivity)

    This is TRUE differential geometric curvature, not just graph topology.

    Args:
        G: networkx.Graph with weighted edges
        weight: Edge attribute name for weights (default "weight")

    Returns:
        RicciCurvatureResult with summary statistics and per-edge curvatures
    """
    import networkx as nx

    if G.number_of_edges() == 0:
        return RicciCurvatureResult(
            mean_curvature=float("nan"),
            min_curvature=float("nan"),
            max_curvature=float("nan"),
            curvature_std=float("nan"),
            edge_curvatures={},
            n_edges=0,
        )

    edge_curvatures = {}

    for u, v, data in G.edges(data=True):
        w_uv = data.get(weight, 1.0)

        # Degree of nodes (weighted or unweighted)
        deg_u = G.degree(u, weight=weight) if weight else G.degree(u)
        deg_v = G.degree(v, weight=weight) if weight else G.degree(v)

        # Find triangles: common neighbors of u and v
        neighbors_u = set(G.neighbors(u))
        neighbors_v = set(G.neighbors(v))
        common = neighbors_u & neighbors_v

        # Triangle contribution: sum of weights for edges forming triangles
        triangle_weight = 0.0
        for z in common:
            w_uz = G[u][z].get(weight, 1.0)
            w_vz = G[v][z].get(weight, 1.0)
            # Contribution from triangle (u, v, z): geometric mean of edge weights
            triangle_weight += np.sqrt(w_uz * w_vz * w_uv)

        # Forman-Ricci curvature formula (normalized by edge weight)
        # Positive = clustered, Negative = bottleneck
        F_uv = w_uv * (4.0 - deg_u - deg_v) + triangle_weight

        edge_curvatures[(u, v)] = F_uv

    # Summary statistics
    curvs = np.array(list(edge_curvatures.values()))
    return RicciCurvatureResult(
        mean_curvature=float(np.mean(curvs)),
        min_curvature=float(np.min(curvs)),
        max_curvature=float(np.max(curvs)),
        curvature_std=float(np.std(curvs)),
        edge_curvatures=edge_curvatures,
        n_edges=len(curvs),
    )


def rolling_ricci_curvature(
    returns: pd.DataFrame,
    *,
    window: int = 60,
    min_assets: int = 3,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Computes rolling Forman-Ricci curvature statistics on correlation graphs.

    Returns a DataFrame with columns:
        - ricci_mean: Average edge curvature (often negative for distance-weighted graphs)
        - ricci_min: Most negative curvature (worst bottleneck)
        - ricci_p10: 10th percentile curvature (tail/breadth of bottlenecks)
        - ricci_p90: 90th percentile curvature (upper tail)
        - ricci_std: Curvature dispersion (uniformity of market structure)

    Higher mean curvature = tightly clustered market (stable regime)
    Lower/negative mean curvature = fragmented market (transition/crisis regime)
    Rising ricci_min (less negative) = stress relief
    Falling ricci_min (more negative) = stress concentration

    This captures differential geometric structure, not just topology.
    """
    returns = returns.copy().astype(float)
    returns = returns.dropna(axis=1, how="all")
    returns = returns.dropna(axis=0, how="all")
    returns = returns.sort_index()

    out = pd.DataFrame(
        index=returns.index,
        columns=["ricci_mean", "ricci_min", "ricci_p10", "ricci_p90", "ricci_std"],
        dtype=float,
    )

    if returns.shape[1] < min_assets:
        return out

    for end in range(window - 1, len(returns.index)):
        w = returns.iloc[end - window + 1 : end + 1].dropna(axis=1, how="any")
        if w.shape[1] < min_assets:
            continue

        corr = w.corr(method=method)
        dist = correlation_distance(corr)

        # Build weighted graph from distance matrix
        import networkx as nx

        assets = list(dist.columns)
        G = nx.Graph()
        for i, a in enumerate(assets):
            for j in range(i + 1, len(assets)):
                b = assets[j]
                d = float(dist.iloc[i, j])
                if np.isfinite(d):
                    # Weight = distance (larger distance = weaker connection)
                    G.add_edge(a, b, weight=d)

        if G.number_of_nodes() < min_assets or G.number_of_edges() == 0:
            continue

        res = forman_ricci_curvature(G, weight="weight")
        curvs = np.array(list(res.edge_curvatures.values()), dtype=float)
        if curvs.size == 0:
            continue

        out.loc[returns.index[end], "ricci_mean"] = res.mean_curvature
        out.loc[returns.index[end], "ricci_min"] = res.min_curvature
        out.loc[returns.index[end], "ricci_p10"] = float(np.quantile(curvs, 0.10))
        out.loc[returns.index[end], "ricci_p90"] = float(np.quantile(curvs, 0.90))
        out.loc[returns.index[end], "ricci_std"] = res.curvature_std

    return out


def rolling_sectional_ricci_curvature(
    returns: pd.DataFrame,
    manifolds: Dict[str, list[str]],
    *,
    window: int = 60,
    min_assets: int = 3,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Computes manifold-specific Forman-Ricci curvature for different asset subsets.

    NOTE ON TERMINOLOGY: This function computes Forman-Ricci curvature on
    subgraphs (asset subsets), NOT true "sectional curvature" in the differential
    geometry sense. The name is retained for backwards compatibility.

    True sectional curvature would require a smooth Riemannian manifold and
    measures how geodesics spread in 2D plane sections. Our discrete graph
    approach uses Forman-Ricci curvature which measures local connectivity and
    edge clustering. For thesis clarity, prefer "manifold-specific curvature"
    or "subgraph Ricci curvature" in documentation.

    This is a key enhancement for regime analysis: different market segments
    (monetary vs industrial vs miners) can exhibit different geometric structures.

    For financial markets, this enables:
    - Monetary manifold curvature → haven demand regime
    - Industrial manifold curvature → growth cycle regime
    - Divergence between manifolds → regime transition signal

    Args:
        returns: Full returns panel (all assets)
        manifolds: Dict mapping manifold name to list of asset symbols
                   Example: {'monetary': ['SI=F', 'GC=F', 'TIP'],
                            'industrial': ['SI=F', 'HG=F', 'FXI']}
        window: Rolling window size
        min_assets: Minimum assets required for valid curvature
        method: Correlation method ('pearson', 'spearman')

    Returns:
        DataFrame with columns:
            - ricci_mean_{manifold_name}_60d
            - ricci_min_{manifold_name}_60d
            - ricci_p10_{manifold_name}_60d
            - curvature_divergence_60d (if multiple manifolds exist)

    Example:
        >>> manifolds = {
        ...     'monetary': ['SI=F', 'GC=F', 'GLD', 'TIP'],
        ...     'industrial': ['SI=F', 'HG=F', 'COPX', 'FXI', 'CL=F']
        ... }
        >>> sectional = rolling_sectional_ricci_curvature(returns, manifolds)
        >>> # High divergence → monetary and industrial drivers decoupling
    """
    import sys

    returns_full = returns.copy().astype(float)
    returns_full = returns_full.dropna(axis=1, how="all")
    returns_full = returns_full.sort_index()

    # Initialize output DataFrame
    column_names = []
    for name in manifolds.keys():
        column_names.extend([
            f'ricci_mean_{name}_{window}d',
            f'ricci_min_{name}_{window}d',
            f'ricci_p10_{name}_{window}d',
            f'ricci_std_{name}_{window}d',
        ])

    out = pd.DataFrame(index=returns_full.index, columns=column_names, dtype=float)

    # Compute curvature for each manifold
    manifold_curvatures = {}

    for manifold_name, symbols in manifolds.items():
        # Filter returns to manifold symbols
        available_symbols = [s for s in symbols if s in returns_full.columns]

        if len(available_symbols) < min_assets:
            print(f"Warning: Manifold '{manifold_name}' has only {len(available_symbols)} "
                  f"assets (< {min_assets} required), skipping", file=sys.stderr)
            continue

        manifold_returns = returns_full[available_symbols]

        # Compute rolling curvature for this manifold
        ricci_df = rolling_ricci_curvature(
            manifold_returns,
            window=window,
            min_assets=min_assets,
            method=method,
        )

        # Store in output with manifold-specific column names
        out[f'ricci_mean_{manifold_name}_{window}d'] = ricci_df['ricci_mean']
        out[f'ricci_min_{manifold_name}_{window}d'] = ricci_df['ricci_min']
        out[f'ricci_p10_{manifold_name}_{window}d'] = ricci_df['ricci_p10']
        out[f'ricci_std_{manifold_name}_{window}d'] = ricci_df['ricci_std']

        manifold_curvatures[manifold_name] = ricci_df['ricci_mean']

    # Compute cross-manifold divergence signals
    if len(manifold_curvatures) >= 2:
        manifold_names = list(manifold_curvatures.keys())

        # Pairwise divergences
        for i, name1 in enumerate(manifold_names):
            for name2 in manifold_names[i+1:]:
                curv1 = manifold_curvatures[name1]
                curv2 = manifold_curvatures[name2]

                # Divergence = difference in curvatures (regime decoupling signal)
                divergence = curv1 - curv2
                out[f'curvature_divergence_{name1}_{name2}_{window}d'] = divergence

                # Z-scored divergence (for threshold signals)
                divergence_mean = divergence.rolling(window=252).mean()
                divergence_std = divergence.rolling(window=252).std() + 1e-8
                z_divergence = (divergence - divergence_mean) / divergence_std
                out[f'curvature_divergence_{name1}_{name2}_zscore_{window}d'] = z_divergence

    return out


def compute_manifold_stress_differential(
    returns: pd.DataFrame,
    manifolds: Dict[str, list[str]],
    *,
    window: int = 60,
    min_assets: int = 3,
) -> pd.DataFrame:
    """
    Computes MST stress differentials between manifolds.

    Complements sectional curvature with topological stress measures.
    High stress differential → one segment under pressure while other stable.

    Args:
        returns: Full returns panel
        manifolds: Dict mapping manifold name to asset symbols
        window: Rolling window size
        min_assets: Minimum assets for valid MST

    Returns:
        DataFrame with:
            - mst_stress_{manifold_name}_60d
            - stress_differential_{name1}_{name2}_60d
    """
    returns_full = returns.copy().astype(float)
    returns_full = returns_full.dropna(axis=1, how="all")
    returns_full = returns_full.sort_index()

    out = pd.DataFrame(index=returns_full.index, dtype=float)
    manifold_stresses = {}

    # Compute MST stress for each manifold
    for manifold_name, symbols in manifolds.items():
        available_symbols = [s for s in symbols if s in returns_full.columns]

        if len(available_symbols) < min_assets:
            continue

        manifold_returns = returns_full[available_symbols]

        # Compute rolling MST stress
        stress = rolling_mst_stress(
            manifold_returns,
            window=window,
            min_assets=min_assets,
        )

        out[f'mst_stress_{manifold_name}_{window}d'] = stress
        manifold_stresses[manifold_name] = stress

    # Compute stress differentials (cross-manifold)
    if len(manifold_stresses) >= 2:
        manifold_names = list(manifold_stresses.keys())

        for i, name1 in enumerate(manifold_names):
            for name2 in manifold_names[i+1:]:
                stress1 = manifold_stresses[name1]
                stress2 = manifold_stresses[name2]

                # Stress differential (absolute difference)
                diff = np.abs(stress1 - stress2)
                out[f'stress_differential_{name1}_{name2}_{window}d'] = diff

                # Relative stress (which manifold more stressed)
                relative = (stress1 - stress2) / (stress1 + stress2 + 1e-8)
                out[f'stress_relative_{name1}_{name2}_{window}d'] = relative

    return out
