"""
Tests for Forman-Ricci curvature computation.

Validates that:
1. Curvature is positive for clustered (triangle-rich) graphs
2. Curvature is negative for bottleneck/bridge structures
3. Rolling computation produces valid output
"""
import numpy as np
import pandas as pd
import networkx as nx
import pytest

from src.geometry.graph_curvature import (
    forman_ricci_curvature,
    rolling_ricci_curvature,
    correlation_distance,
)


def test_forman_ricci_triangle():
    """
    Test: Triangle graph should have POSITIVE curvature (stable, clustered).
    """
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)

    result = forman_ricci_curvature(G, weight="weight")

    # In a perfect triangle, all edges are in triangles → positive curvature
    assert result.n_edges == 3
    assert result.mean_curvature > 0, "Triangle should have positive curvature"
    print(f"Triangle mean curvature: {result.mean_curvature:.4f} (expected > 0)")


def test_forman_ricci_bridge():
    """
    Test: Bridge edge (bottleneck) should have NEGATIVE curvature.
    """
    # Two triangles connected by a bridge edge
    G = nx.Graph()
    # Triangle 1
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)
    # Bridge
    G.add_edge("C", "D", weight=1.0)
    # Triangle 2
    G.add_edge("D", "E", weight=1.0)
    G.add_edge("E", "F", weight=1.0)
    G.add_edge("F", "D", weight=1.0)

    result = forman_ricci_curvature(G, weight="weight")

    # Bridge edge (C-D) should have most negative curvature
    bridge_curvature = result.edge_curvatures.get(("C", "D")) or result.edge_curvatures.get(("D", "C"))
    assert bridge_curvature is not None, "Bridge edge should exist"
    assert bridge_curvature < 0, "Bridge should have negative curvature"

    # Overall min curvature should be negative
    assert result.min_curvature < 0, "Graph with bridge should have negative min curvature"
    print(f"Bridge curvature: {bridge_curvature:.4f} (expected < 0)")
    print(f"Min curvature: {result.min_curvature:.4f}")


def test_forman_ricci_star():
    """
    Test: Star graph (hub-and-spoke) should have negative curvature at center.
    """
    G = nx.star_graph(5)  # 1 center node connected to 5 spokes
    nx.set_edge_attributes(G, 1.0, "weight")

    result = forman_ricci_curvature(G, weight="weight")

    # Star has no triangles → all edges have negative curvature
    assert result.mean_curvature < 0, "Star graph should have negative mean curvature (no triangles)"
    print(f"Star mean curvature: {result.mean_curvature:.4f} (expected < 0)")


def test_rolling_ricci_curvature():
    """
    Test: Rolling Ricci curvature on synthetic return data.
    """
    # Create synthetic returns: 3 assets, 100 days
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    returns = pd.DataFrame(
        {
            "A": np.random.normal(0, 0.01, 100),
            "B": np.random.normal(0, 0.01, 100),
            "C": np.random.normal(0, 0.01, 100),
        },
        index=dates,
    )

    # Make A and B highly correlated (positive curvature expected)
    returns["B"] = 0.9 * returns["A"] + 0.1 * np.random.normal(0, 0.01, 100)

    result = rolling_ricci_curvature(returns, window=30, min_assets=3)

    assert isinstance(result, pd.DataFrame), "Should return DataFrame"
    assert "ricci_mean" in result.columns
    assert "ricci_min" in result.columns
    assert "ricci_std" in result.columns

    # Should have non-NaN values after window
    assert result["ricci_mean"].notna().sum() > 0, "Should have valid curvature values"

    print(f"Computed {result['ricci_mean'].notna().sum()} valid curvature values")
    print(f"Mean curvature range: [{result['ricci_mean'].min():.4f}, {result['ricci_mean'].max():.4f}]")


def test_ricci_curvature_empty_graph():
    """
    Test: Empty graph should return NaN statistics.
    """
    G = nx.Graph()
    result = forman_ricci_curvature(G)

    assert np.isnan(result.mean_curvature)
    assert np.isnan(result.min_curvature)
    assert np.isnan(result.max_curvature)
    assert result.n_edges == 0


def test_correlation_distance_to_curvature():
    """
    Test: High correlation → small distance → positive curvature (triangle formation).
    """
    # Perfect correlation: should form tight cluster
    corr = pd.DataFrame(
        [[1.0, 0.95, 0.95], [0.95, 1.0, 0.95], [0.95, 0.95, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    dist = correlation_distance(corr)
    assert dist.min().min() >= 0, "Distance should be non-negative"
    assert dist.loc["A", "B"] < 0.5, "High correlation should give small distance"

    # Build graph and compute curvature
    G = nx.Graph()
    G.add_edge("A", "B", weight=dist.loc["A", "B"])
    G.add_edge("B", "C", weight=dist.loc["B", "C"])
    G.add_edge("C", "A", weight=dist.loc["C", "A"])

    result = forman_ricci_curvature(G, weight="weight")
    # Tight cluster → triangles → positive curvature
    assert result.mean_curvature > 0, "High correlation cluster should have positive curvature"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
