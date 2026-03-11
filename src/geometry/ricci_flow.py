"""
Discrete Ricci Flow for Market Graph Evolution

Implements discrete Ricci flow to simulate how market correlation structure
evolves under curvature-driven dynamics. This is TRUE differential geometry,
not linear algebra.

Ricci flow: dg/dt = -2 * Ric(g)
In discrete setting: edge weights evolve based on their Forman-Ricci curvature.

References:
- Ollivier, "Ricci curvature of Markov chains on metric spaces" (2009)
- Forman, "Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature" (2003)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from src.geometry.graph_curvature import correlation_distance, forman_ricci_curvature, RicciCurvatureResult


@dataclass(frozen=True)
class RicciFlowResult:
    """
    Result of Ricci flow evolution.

    Attributes:
        iterations: Number of flow iterations performed
        final_curvature: Final curvature statistics after flow
        curvature_history: List of curvature results at each step
        converged: Whether flow converged (curvature change < threshold)
    """
    iterations: int
    final_curvature: RicciCurvatureResult
    curvature_history: list[RicciCurvatureResult]
    converged: bool


def ricci_flow_step(G: nx.Graph, curvature: RicciCurvatureResult, dt: float = 0.01, weight: str = "weight") -> nx.Graph:
    """
    Performs one step of discrete Ricci flow on graph G.

    Updates edge weights according to: w_new = w_old * exp(-2 * dt * F(e))
    where F(e) is the Forman-Ricci curvature of edge e.

    Interpretation:
        - Positive curvature (stable): edges get shorter (stronger correlation)
        - Negative curvature (bottleneck): edges get longer (weaker correlation)

    This flow naturally smooths out curvature, moving toward more homogeneous structure.

    Args:
        G: networkx.Graph with weighted edges
        curvature: Current Forman-Ricci curvature result
        dt: Time step size (smaller = more stable, larger = faster but risky)
        weight: Edge attribute name

    Returns:
        New graph with updated edge weights
    """
    G_new = G.copy()

    for (u, v), F_uv in curvature.edge_curvatures.items():
        if G_new.has_edge(u, v):
            w_old = G_new[u][v].get(weight, 1.0)
            # Ricci flow: dw/dt = -2 * F(e) * w
            # Discretized: w_new = w_old * exp(-2 * dt * F)
            w_new = w_old * np.exp(-2.0 * dt * F_uv)
            # Clamp to avoid numerical issues
            w_new = np.clip(w_new, 1e-6, 1e6)
            G_new[u][v][weight] = float(w_new)

    return G_new


def simulate_ricci_flow(
    G: nx.Graph,
    *,
    max_iterations: int = 100,
    dt: float = 0.01,
    convergence_threshold: float = 1e-4,
    weight: str = "weight",
) -> RicciFlowResult:
    """
    Simulates discrete Ricci flow until convergence or max iterations.

    Args:
        G: Initial graph
        max_iterations: Maximum number of flow steps
        dt: Time step size
        convergence_threshold: Stop when curvature change < threshold
        weight: Edge weight attribute name

    Returns:
        RicciFlowResult with evolution history
    """
    G_current = G.copy()
    curvature_history = []

    # Compute initial curvature
    curv = forman_ricci_curvature(G_current, weight=weight)
    curvature_history.append(curv)

    # Early return for empty graph
    if G_current.number_of_edges() == 0:
        return RicciFlowResult(
            iterations=0,
            final_curvature=curv,
            curvature_history=curvature_history,
            converged=True,
        )

    converged = False

    for i in range(max_iterations):
        # Flow step
        G_current = ricci_flow_step(G_current, curv, dt=dt, weight=weight)

        # Recompute curvature
        curv_new = forman_ricci_curvature(G_current, weight=weight)
        curvature_history.append(curv_new)

        # Check convergence: mean curvature change
        curv_change = abs(curv_new.mean_curvature - curv.mean_curvature)

        if curv_change < convergence_threshold:
            converged = True
            break

        curv = curv_new

    return RicciFlowResult(
        iterations=len(curvature_history) - 1,
        final_curvature=curvature_history[-1],
        curvature_history=curvature_history,
        converged=converged,
    )


def ricci_flow_shock_propagation(
    returns: pd.DataFrame,
    shock_asset: str,
    shock_magnitude: float = -0.05,
    window: int = 60,
    flow_steps: int = 50,
) -> Dict[str, float]:
    """
    Simulates how a shock to one asset propagates through the market via Ricci flow.

    This answers: "If asset X crashes, how does the correlation structure deform,
    and which assets are most affected?"

    Args:
        returns: DataFrame of asset returns
        shock_asset: Asset to shock (must be in returns columns)
        shock_magnitude: Return shock size (e.g., -0.05 = -5% return)
        window: Rolling window for correlation estimation
        flow_steps: Number of Ricci flow iterations

    Returns:
        Dict mapping asset -> final distance from shocked asset after flow
    """
    if shock_asset not in returns.columns:
        raise ValueError(f"Shock asset '{shock_asset}' not in returns columns")

    # Get latest window
    latest_returns = returns.iloc[-window:].copy()

    # Apply shock to target asset
    latest_returns.loc[latest_returns.index[-1], shock_asset] += shock_magnitude

    # Build correlation distance graph
    corr = latest_returns.corr(method="pearson")
    dist = correlation_distance(corr)

    # Construct graph
    assets = list(dist.columns)
    G = nx.Graph()
    for i, a in enumerate(assets):
        for j in range(i + 1, len(assets)):
            b = assets[j]
            d = float(dist.iloc[i, j])
            if np.isfinite(d):
                G.add_edge(a, b, weight=d)

    # Simulate Ricci flow
    flow_result = simulate_ricci_flow(G, max_iterations=flow_steps, dt=0.01)

    # Extract final distances from shocked asset
    G_final = G.copy()
    for (u, v), F in flow_result.final_curvature.edge_curvatures.items():
        if G_final.has_edge(u, v):
            # Update weight from flow
            w_init = G[u][v]["weight"]
            w_final = w_init * np.exp(-2.0 * 0.01 * F * flow_steps)
            G_final[u][v]["weight"] = w_final

    # Compute shortest path distances from shock_asset
    try:
        distances = nx.single_source_dijkstra_path_length(G_final, shock_asset, weight="weight")
    except nx.NetworkXNoPath:
        distances = {a: float("inf") for a in assets}

    return distances


def rolling_ricci_flow_stability(
    returns: pd.DataFrame,
    *,
    window: int = 60,
    flow_steps: int = 20,
    min_assets: int = 3,
) -> pd.Series:
    """
    Computes rolling "flow stability" metric: how much does curvature change under flow?

    High stability = curvature stays similar (market structure is robust)
    Low stability = curvature changes drastically (market structure is fragile)

    Args:
        returns: Multi-asset return DataFrame
        window: Rolling window size
        flow_steps: Number of flow iterations per window
        min_assets: Minimum assets required

    Returns:
        Series of flow stability scores (0-1, higher = more stable)
    """
    returns = returns.copy().astype(float)
    returns = returns.dropna(axis=1, how="all").dropna(axis=0, how="all").sort_index()

    stability = pd.Series(index=returns.index, dtype=float, name="ricci_flow_stability")

    if returns.shape[1] < min_assets:
        return stability

    for end in range(window - 1, len(returns.index)):
        w = returns.iloc[end - window + 1 : end + 1].dropna(axis=1, how="any")
        if w.shape[1] < min_assets:
            continue

        corr = w.corr(method="pearson")
        dist = correlation_distance(corr)

        # Build graph
        assets = list(dist.columns)
        G = nx.Graph()
        for i, a in enumerate(assets):
            for j in range(i + 1, len(assets)):
                b = assets[j]
                d = float(dist.iloc[i, j])
                if np.isfinite(d):
                    G.add_edge(a, b, weight=d)

        if G.number_of_edges() < min_assets:
            continue

        # Simulate flow
        flow_result = simulate_ricci_flow(G, max_iterations=flow_steps, dt=0.01)

        # Stability = 1 / (1 + curvature_change)
        curv_init = flow_result.curvature_history[0].mean_curvature
        curv_final = flow_result.final_curvature.mean_curvature
        curv_change = abs(curv_final - curv_init)

        stability_score = 1.0 / (1.0 + curv_change)
        stability.iloc[end] = stability_score

    return stability
