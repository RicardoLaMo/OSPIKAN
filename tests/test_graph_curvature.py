import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.geometry.graph_curvature import (
    rolling_mst_stress, 
    rolling_ricci_curvature, 
    forman_ricci_curvature
)
import networkx as nx


def test_rolling_mst_stress_perfect_correlation_zero_stress():
    idx = pd.date_range("2024-01-01", periods=80, freq="D")
    base = np.random.default_rng(0).normal(0.0, 0.01, size=len(idx))

    returns = pd.DataFrame(
        {
            "A": base,
            "B": base,
            "C": base,
        },
        index=idx,
    )

    stress = rolling_mst_stress(returns, window=60, min_assets=3)
    last = float(stress.dropna().iloc[-1])
    assert abs(last - 0.0) < 1e-12


def test_forman_ricci_curvature_triangle():
    """
    Test Forman-Ricci curvature on a simple triangle graph.
    In a triangle with equal weights w=1:
    F(e) = w * (4 - deg(u) - deg(v)) + triangle_contribution
    deg(u)=2, deg(v)=2
    triangle_contribution = sqrt(1*1*1) = 1
    F(e) = 1 * (4 - 2 - 2) + 1 = 1
    """
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("B", "C", weight=1.0)
    G.add_edge("C", "A", weight=1.0)

    res = forman_ricci_curvature(G, weight="weight")
    
    assert res.n_edges == 3
    assert abs(res.mean_curvature - 1.0) < 1e-6
    assert abs(res.min_curvature - 1.0) < 1e-6
    
    # Check specific edge
    assert abs(res.edge_curvatures[("A", "B")] - 1.0) < 1e-6


def test_forman_ricci_curvature_star():
    """
    Test curvature on a star graph (center A, leaves B, C, D).
    Edges (A,B), (A,C), (A,D). No triangles.
    deg(A)=3, deg(leaf)=1.
    F(A, leaf) = 1 * (4 - 3 - 1) + 0 = 0
    """
    G = nx.Graph()
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("A", "C", weight=1.0)
    G.add_edge("A", "D", weight=1.0)
    
    res = forman_ricci_curvature(G, weight="weight")
    
    assert res.n_edges == 3
    assert abs(res.mean_curvature - 0.0) < 1e-6


def test_rolling_ricci_curvature_integration():
    """Test rolling curvature on random data."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    rng = np.random.default_rng(42)
    
    # Create 4 correlated assets
    base = rng.normal(0, 0.01, size=len(idx))
    returns = pd.DataFrame({
        "A": base + rng.normal(0, 0.002, len(idx)),
        "B": base + rng.normal(0, 0.002, len(idx)),
        "C": base + rng.normal(0, 0.002, len(idx)),
        "D": rng.normal(0, 0.01, len(idx)), # Uncorrelated
    }, index=idx)

    # Compute rolling curvature
    df = rolling_ricci_curvature(returns, window=30, min_assets=3)
    
    assert isinstance(df, pd.DataFrame)
    assert "ricci_mean" in df.columns
    assert "ricci_min" in df.columns
    
    # Check we have values
    valid = df.dropna()
    assert len(valid) > 0
    
    # Mean curvature should likely be positive due to cluster A,B,C
    assert valid["ricci_mean"].mean() > -5.0 # Loose bound check


