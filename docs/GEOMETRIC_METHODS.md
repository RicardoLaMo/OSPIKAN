# Geometric Methods in Silver Regime Analysis

## Overview

This document explains the **differential geometric** and **Clifford algebraic** methods used in the silver price regime analysis, contrasting them with traditional **linear algebra** approaches.

**Key Distinction:**
- **Linear Algebra**: Correlation matrices, PCA, covariance, linear regression
- **Differential Geometry**: Ricci curvature, geodesics, metric tensors, manifold deformation
- **Geometric Algebra (GA)**: Clifford algebra Cl(4,0), rotors, bivectors, multivector operations

---

## 1. Forman-Ricci Curvature (TRUE Geometry)

### Mathematical Foundation

Forman-Ricci curvature is a **discrete analog** of Ricci curvature from differential geometry. For an edge \( e = (v, w) \) in a weighted graph:

$$
F(e) = w(e) \cdot \left[ 4 - \deg(v) - \deg(w) + \sum_{\text{triangles}} f_{\triangle}(e) \right]
$$

Where:
- \( w(e) \) = edge weight (correlation distance)
- \( \deg(v) \) = weighted degree of node \( v \)
- \( f_{\triangle}(e) \) = contribution from triangles containing edge \( e \)

### Interpretation

| Curvature Value | Geometric Meaning | Market Interpretation |
|----------------|-------------------|----------------------|
| **Positive** | Edge is in a **tightly clustered region** (many triangles) | Stable correlations, low stress, co-movement |
| **Negative** | Edge is a **bottleneck or bridge** (few/no triangles) | Fragmented market, structural stress, breakdown |
| **Zero** | **Flat geometry** (random-like connectivity) | Neutral, no dominant structure |

### Implementation: `src/geometry/graph_curvature.py`

```python
def forman_ricci_curvature(G, weight="weight") -> RicciCurvatureResult:
    """
    Computes Forman-Ricci curvature for all edges in graph G.

    Returns:
        - mean_curvature: Average across all edges
        - min_curvature: Most negative (identifies bottlenecks)
        - max_curvature: Most positive (identifies tight clusters)
        - edge_curvatures: Dict of (u,v) -> curvature value
    """
```

**Key Features:**
- Computes **sectional curvature** of each edge based on local triangle structure
- Identifies **bottlenecks** (negative curvature) and **clusters** (positive curvature)
- NOT just graph topology (MST stress) - true differential geometric quantity

### Why This Is NOT Linear Algebra

**Linear Algebra** (e.g., correlation matrix):
- Pairwise linear relationships
- No higher-order structure (no triangle information)
- Eigenvalues measure variance, not curvature

**Differential Geometry** (Forman-Ricci):
- Captures **local neighborhood structure** (triangles)
- Measures how **geodesics diverge** in curved space
- Curvature is a **second-order geometric invariant**, not a first-order statistic

---

## 2. Ricci Flow (Metric Evolution)

### Mathematical Foundation

Ricci flow is a PDE that evolves a Riemannian metric \( g \) over time:

$$
\frac{\partial g}{\partial t} = -2 \text{Ric}(g)
$$

In discrete setting (graph edges):

$$
\frac{dw(e)}{dt} = -2 F(e) \cdot w(e)
$$

**Discretized update:**

$$
w_{\text{new}} = w_{\text{old}} \cdot \exp(-2 \Delta t \cdot F(e))
$$

### Interpretation

- **Positive curvature edges**: Get **shorter** (stronger correlation) → Clustering
- **Negative curvature edges**: Get **longer** (weaker correlation) → Fragmentation
- **Convergence**: Flow smooths out curvature → homogeneous geometry

### Implementation: `src/geometry/ricci_flow.py`

```python
def simulate_ricci_flow(G, max_iterations=100, dt=0.01) -> RicciFlowResult:
    """
    Simulates discrete Ricci flow until convergence.

    Returns:
        - curvature_history: Evolution of curvature over time
        - converged: Whether flow reached stable state
    """
```

**Use Cases:**
- **Shock propagation**: How does a market shock deform correlation structure?
- **Stability analysis**: Does the geometry converge (stable) or diverge (unstable)?
- **Regime transitions**: Flow acceleration indicates structural changes

### Why This Is NOT Linear Algebra

**Linear methods** (e.g., rolling correlation):
- Static snapshots at each timestep
- No notion of "flow" or metric evolution
- Cannot simulate geometric deformation

**Ricci flow**:
- **Dynamic evolution** of entire metric tensor
- Captures **non-linear feedback** between curvature and distance
- Models how market structure **deforms under stress**

---

## 3. Geometric Algebra (Clifford Rotors)

### Mathematical Foundation

Geometric Algebra Cl(4,0) embeds 4 features into a **multivector**:

$$
M = \alpha_0 \cdot 1 + \alpha_1 e_1 + \alpha_2 e_2 + \alpha_3 e_3 + \alpha_4 e_4 + \ldots + \alpha_{15} e_{1234}
$$

**Rotor** \( R \) represents a **rotation** in feature space:

$$
v' = R \tilde{v} R^{-1}
$$

Where:
- \( R \) = even-grade multivector (scalar + bivector)
- \( \tilde{v} \) = reverse of \( v \)

### Bivector Decomposition

A bivector \( B = B_1 e_{12} + B_2 e_{13} + \ldots \) encodes:
- **Rotation planes**: Which features are rotating relative to each other
- **Rotation magnitude**: How large the regime shift is

### Implementation: `src/geometry/tensor_ga.py`, `src/geometry/ga_regime_features.py`

```python
def compute_rotor_magnitude(features, window=60) -> pd.Series:
    """
    Computes rotor magnitude between consecutive feature windows.

    Large rotor magnitude = significant regime transition.
    """
```

**Key Features:**
- **Rotor estimation** between two multivectors representing market states
- **Bivector energy** measures rotational activity
- Handles **anti-parallel** and **orthogonal** cases correctly

### Why This Is NOT Linear Algebra

**Linear Algebra** (e.g., rotation matrices):
- Requires specifying basis explicitly
- Rotations in 3D require 3×3 matrices (9 parameters)
- No natural extension to higher dimensions

**Geometric Algebra**:
- **Basis-independent** rotations via rotors
- Rotations in 4D via **single bivector** (6 parameters)
- **Algebraically closed**: geometric product combines inner and outer products
- **Invariant under coordinate changes**

**Example:**
```python
# Linear Algebra: Rotate vector v by matrix R
v_new = R @ v  # Requires explicit matrix

# Geometric Algebra: Rotate multivector v by rotor R
v_new = R * v * ~R  # Sandwich product, coordinate-free
```

---

## 4. Comparison Table: Methods Used in Pipeline

| Feature | Type | Mathematical Foundation | Implementation |
|---------|------|------------------------|----------------|
| **MST Stress** | Graph Topology | Kruskal's MST algorithm, sum of edge weights | `graph_curvature.py:mst_stress_index()` |
| **Forman-Ricci Curvature** | **Differential Geometry** | Discrete Ricci curvature via triangle count | `graph_curvature.py:forman_ricci_curvature()` |
| **Ricci Flow** | **Geometric PDE** | Heat equation on metric tensor | `ricci_flow.py:simulate_ricci_flow()` |
| **GA Rotors** | **Clifford Algebra** | Cl(4,0) rotor estimation | `ga_regime_features.py:compute_rotor_magnitude()` |
| **Correlation** | Linear Algebra | Pearson correlation coefficient | `pandas.DataFrame.corr()` |
| **PCA** | Linear Algebra | Eigendecomposition of covariance | (not used in current pipeline) |

---

## 5. Regime Detection: Geometric vs Statistical

### Traditional Statistical Methods

**Markov-Switching Regression** (`src/analysis/regimes.py:fit_markov_regression()`):
- Uses **linear regression** with regime-dependent coefficients
- Probabilistic state transitions via HMM
- Exogenous variables: macro returns (DXY, yields, SPX)

**Now Enhanced with Geometric Features:**
```python
exog_cols = macro_cols + geom_cols
# geom_cols = [mst_stress_60d, ricci_mean_60d, ricci_min_60d,
#              ga_rotor_magnitude_60d, ga_bivector_energy_60d]
```

### Pure Geometric Regime Detection

**`geometric_regime_classification()`** (`src/analysis/regimes.py`):

Uses **only geometric features** to classify regimes:

| Regime | Geometric Signature |
|--------|-------------------|
| **STABLE** | High curvature (positive), low MST stress, low rotor magnitude |
| **TRANSITION** | Rising rotor magnitude, falling curvature |
| **STRESS** | Negative curvature, high MST stress, unstable geometry |
| **RECOVERY** | Rising curvature from negative, decreasing stress |

**No linear regression, no HMM** - purely geometric thresholds.

---

## 6. Running Geometric Analysis

### Step 1: Compute Features with Geometry
```bash
python src/pipeline/silver_pipeline.py features \
    --panel data/interim/silver_panel_close_<RUN_ID>.csv
# This now computes:
# - ricci_mean_60d, ricci_min_60d, ricci_std_60d
# - ga_rotor_magnitude_60d, ga_bivector_energy_60d
```

### Step 2: Enhanced Markov Regimes (uses geometry as exog vars)
```bash
python scripts/silver_markov_regimes.py \
    --features data/processed/silver_features_<RUN_ID>.parquet --k 3
```

### Step 3: Pure Geometric Regimes
```bash
python scripts/silver_geometric_regimes.py \
    --features data/processed/silver_features_<RUN_ID>.parquet
```

**Outputs:**
- `reports/silver/figures/geometric_features_timeseries.png`: Curvature evolution
- `reports/silver/figures/silver_price_geometric_regimes.png`: Price with geometric regimes
- `reports/silver/tables/geometric_regime_labels.csv`: Regime labels
- `reports/silver/tables/geometric_regime_return_stats.csv`: Returns by geometric regime

---

## 7. Validation

### Tests for Geometric Methods

1. **Ricci Curvature** (`tests/test_ricci_curvature.py`):
   - Triangle graph → positive curvature ✓
   - Bridge/bottleneck → negative curvature ✓
   - Star graph → negative curvature ✓

2. **Ricci Flow** (`tests/test_ricci_flow.py`):
   - Flow converges for symmetric graphs ✓
   - Positive curvature edges get shorter ✓
   - Negative curvature edges get longer ✓

3. **GA Rotors** (`tests/test_ga_regime_features.py`):
   - Rotor magnitude is non-negative ✓
   - Bivector energy captures rotational activity ✓
   - Rotor is invariant under feature shifts ✓

Run tests:
```bash
pytest tests/test_ricci_curvature.py -v
pytest tests/test_ricci_flow.py -v
pytest tests/test_ga_regime_features.py -v
```

---

## 8. Key Takeaways

### What Makes This "Geometric" and Not Just "Linear Algebra"?

1. **Curvature is a Second-Order Invariant**
   - Linear algebra: First-order (correlation, distance)
   - Differential geometry: Second-order (how distances change along geodesics)

2. **Manifold Structure**
   - Linear algebra: Flat Euclidean space
   - Differential geometry: Curved Riemannian manifold with metric tensor

3. **Coordinate-Free Representations**
   - Linear algebra: Matrix operations depend on chosen basis
   - Geometric algebra: Rotors are intrinsic geometric objects

4. **Dynamic Evolution**
   - Linear algebra: Static snapshots
   - Ricci flow: PDE-driven metric evolution over time

5. **Topological Features**
   - Linear algebra: Pairwise relationships
   - Differential geometry: Higher-order structure (triangles, curvature, flow)

---

## 9. References

**Differential Geometry:**
- Forman, R. (2003). "Bochner's Method for Cell Complexes and Combinatorial Ricci Curvature"
- Hamilton, R. (1982). "Three-manifolds with positive Ricci curvature"
- Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces"

**Geometric Algebra:**
- Hestenes, D. (1999). "New Foundations for Classical Mechanics"
- Dorst, L. (2007). "Geometric Algebra for Computer Science"

**Financial Applications:**
- Sandhu, R. et al. (2016). "Graph curvature for differentiating cancer networks"
- Samal, A. et al. (2018). "Comparative analysis of two discretizations of Ricci curvature"

---

## 10. Future Enhancements

### Potential Next Steps

1. **Ollivier-Ricci Curvature**
   - More sophisticated than Forman-Ricci
   - Based on optimal transport theory
   - Library: `GraphRicciCurvature`

2. **Stiefel Manifold Optimization**
   - Cointegration search on manifold of orthogonal matrices
   - Cayley transform for retraction
   - Riemannian gradient descent

3. **Curvature-Based Signals**
   - Alert when `ricci_min_60d` drops below threshold
   - Trade when `ga_rotor_magnitude_60d` spikes
   - Use Ricci flow convergence as stability indicator

4. **Multi-Scale Geometry**
   - Persistent homology for topological features
   - Spectral geometry (Laplacian eigenmaps)
   - Geodesic distance vs correlation distance comparison

---

**End of Geometric Methods Documentation**
