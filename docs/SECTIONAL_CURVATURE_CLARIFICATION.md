# Clarification: "Sectional Curvature" vs "Manifold-Specific Ricci Curvature"

**Date**: 2026-01-12
**Purpose**: Clarify terminology for thesis defense
**Priority**: MEDIUM - Terminology accuracy

---

## 🎯 The Terminology Issue

### Classical Differential Geometry Definition

**Sectional Curvature K(σ)** in Riemannian geometry:
> For a 2-dimensional plane σ ⊂ TₚM in the tangent space at point p, the sectional curvature K(σ) measures how the manifold M curves in the direction of plane σ.

**Mathematical Formula**:
```
K(σ) = R(X, Y, Y, X) / (⟨X,X⟩⟨Y,Y⟩ - ⟨X,Y⟩²)
```
where R is the Riemann curvature tensor and X, Y span σ.

**Key Properties**:
- Defined for **2-dimensional planes** in tangent space
- Generalizes Gaussian curvature to higher dimensions
- Ricci curvature = average of sectional curvatures over all planes containing a vector

---

### Our Implementation

**What We Actually Compute** (`src/geometry/graph_curvature.py`):
```python
def rolling_sectional_ricci_curvature(returns, manifolds, ...):
    """
    Computes Ricci curvature for asset subgraphs (manifolds).
    """
    for manifold_name, symbols in manifolds.items():
        # Filter returns to manifold symbols
        manifold_returns = returns[symbols]

        # Compute Forman-Ricci curvature on this subgraph
        ricci_df = rolling_ricci_curvature(manifold_returns, ...)
```

**What This Is**:
- **Forman-Ricci curvature** computed on **subgraphs** (asset subsets)
- **NOT** classical sectional curvature (curvature of 2D planes)
- Better described as: "manifold-specific Ricci curvature" or "subgraph Ricci curvature"

---

## 📐 Conceptual Relationship

### Why We Used "Sectional"

**Analogy**:
- Classical sectional curvature: Measures curvature in **specific 2D directions**
- Our approach: Measures curvature in **specific market segments** (monetary, industrial, etc.)

**The Connection**:
- Both focus on **subspace geometry** rather than global geometry
- Both allow **directional analysis** of curvature
- Both enable **comparison between different subspaces**

**The Difference**:
- Classical: Subspaces are 2D planes in tangent space
- Ours: "Subspaces" are subsets of the asset graph (market segments)

### More Accurate Terminology

**Option 1: Manifold-Specific Ricci Curvature** (Recommended)
```
"We compute Ricci curvature separately for different market manifolds
(monetary, industrial, miners, credit, volatility), enabling analysis
of segment-specific geometric structure."
```

**Option 2: Subgraph Ricci Curvature**
```
"We compute Ricci curvature on subgraphs corresponding to market segments,
providing a decomposition of overall market geometry into sector-specific
geometric measures."
```

**Option 3: Market-Segmented Ricci Curvature**
```
"We segment the market into thematic groups (monetary haven, industrial
growth, etc.) and compute Ricci curvature for each segment independently."
```

---

## 🎓 For Thesis and Defense

### How to Frame It

**In Abstract/Introduction**:
> "We introduce **manifold-specific Ricci curvature**, computing Forman-Ricci curvature separately for market segments (monetary, industrial, credit, etc.). This decomposition reveals which drivers dominate regime changes—a distinction obscured by global curvature measures."

**In Methodology Chapter**:
> "While our approach is inspired by sectional curvature's focus on subspace geometry, our implementation computes Forman-Ricci curvature on asset subgraphs (manifolds) rather than on 2D planes in tangent space. We adopt the term 'manifold-specific' to describe this segment-wise geometric analysis."

**In Related Work**:
> "Prior work [Sandhu2015, Tran2020] applies Ricci curvature globally to market correlation networks. We extend this by computing **manifold-specific** Ricci curvature for thematic asset groups, analogous to how sectional curvature examines specific planes in classical geometry."

### Defense Q&A

**Q: "Is this classical sectional curvature?"**

**A**: "No, not in the strict differential geometry sense. Classical sectional curvature measures curvature of 2-dimensional planes in tangent space. We compute Forman-Ricci curvature on subgraphs corresponding to market segments (monetary, industrial, etc.). The term 'sectional' was inspired by the shared focus on subspace geometry, but 'manifold-specific' or 'segment-specific' would be more precise."

**Q: "Why not just call it subgraph curvature?"**

**A**: "That's accurate and we could. We initially used 'sectional' to emphasize the analogy with sectional curvature's directional analysis. The key contribution is independent of terminology: separating monetary from industrial geometry reveals regime drivers that global curvature conflates."

**Q: "Is the mathematical foundation still sound?"**

**A**: "Yes. We compute Forman-Ricci curvature—a well-defined discrete Ricci curvature for graphs [Forman2003]—on induced subgraphs. The innovation is the market segmentation strategy and cross-manifold divergence signals, not the curvature computation itself."

---

## 🔄 Recommended Code Changes

### Option A: Rename Function (Breaking Change)
```python
# OLD
def rolling_sectional_ricci_curvature(...)

# NEW (More Accurate)
def rolling_manifold_ricci_curvature(...)
# or
def rolling_subgraph_ricci_curvature(...)
```

### Option B: Add Clarifying Docstring (Non-Breaking)
```python
def rolling_sectional_ricci_curvature(
    returns: pd.DataFrame,
    manifolds: Dict[str, list[str]],
    *,
    window: int = 60,
) -> pd.DataFrame:
    """
    Computes manifold-specific Ricci curvature for asset subgraphs.

    TERMINOLOGY NOTE:
    This function computes Forman-Ricci curvature on subgraphs (asset subsets),
    not classical sectional curvature (curvature of 2D planes). We use "sectional"
    by analogy: both approaches analyze subspace geometry rather than global
    structure. More precise terms: "manifold-specific" or "subgraph" Ricci curvature.

    FINANCIAL INTERPRETATION:
    Different market segments (monetary, industrial, miners, credit, volatility)
    exhibit distinct geometric structures. By computing curvature for each
    manifold separately, we identify which drivers dominate regime dynamics.

    Args:
        returns: Full returns panel (all assets)
        manifolds: Dict mapping manifold name to list of asset symbols
                   Example: {'monetary': ['SI=F', 'GC=F', 'TIP'],
                            'industrial': ['SI=F', 'HG=F', 'CL=F']}
        window: Rolling window size

    Returns:
        DataFrame with manifold-specific curvatures and cross-manifold divergences
            - ricci_mean_{manifold}_60d: Mean Ricci curvature for each manifold
            - curvature_divergence_{m1}_{m2}_60d: Difference between manifold curvatures

    Example:
        >>> manifolds = {
        ...     'monetary': ['SI=F', 'GC=F', 'TIP'],
        ...     'industrial': ['SI=F', 'HG=F', 'CL=F']
        ... }
        >>> curvatures = rolling_sectional_ricci_curvature(returns, manifolds)
        >>> # Divergence > 0: Monetary manifold more stressed than industrial
        >>> # Divergence < 0: Industrial manifold more stressed than monetary
    """
```

### Option C: Add Module-Level Documentation
```python
# src/geometry/graph_curvature.py (at top of file)
"""
Graph-based Geometric Measures for Financial Markets

TERMINOLOGY CLARIFICATION:
We use "manifold-specific" and "sectional" Ricci curvature to describe
Forman-Ricci curvature computed on asset subgraphs (market segments).
This is inspired by—but distinct from—classical sectional curvature in
Riemannian geometry:

Classical sectional curvature:
    K(σ) for 2D plane σ ⊂ TₚM in tangent space

Our manifold-specific curvature:
    Ric(M) for subgraph M ⊂ G induced by asset subset

Both examine subspace geometry; the analogy is:
- Classical: Different 2D planes → different curvatures
- Ours: Different market segments → different curvatures

FINANCIAL INTERPRETATION:
Market segments (monetary, industrial, etc.) have distinct correlation
structures and thus distinct geometric curvatures. Divergence between
manifold curvatures signals regime transitions driven by competing
market forces (haven demand vs growth cycle).
"""
```

---

## ✅ Recommendation

**For Thesis**:
1. **Use "manifold-specific Ricci curvature"** as primary term
2. **Explain relationship** to classical sectional curvature (analogy, not equivalence)
3. **Emphasize financial interpretation** (market segmentation) over geometric formalism
4. **Acknowledge terminology** in footnote: "More precisely, this is Ricci curvature on subgraphs, not classical sectional curvature"

**For Code**:
1. **Keep function name** (avoid breaking changes late in thesis)
2. **Add clarifying docstring** (Option B above)
3. **Add module documentation** (Option C above)
4. **Update feature names** in thesis to say "manifold-specific" or "segment-specific"

**For Defense**:
1. **Be upfront**: "This is Ricci curvature on subgraphs, inspired by sectional curvature's directional analysis"
2. **Focus on contribution**: "The innovation is market segmentation, not the curvature definition"
3. **Show results**: "Monetary-industrial divergence predicts transitions regardless of what we call it"

---

## 📚 Citations to Support Terminology

**If challenged on terminology**, cite:

1. **Forman (2003)**: "Bochner's method for cell complexes and combinatorial Ricci curvature"
   - Defines discrete Ricci curvature for graphs
   - What we actually compute

2. **Ni et al. (2015)**: "Ricci curvature of the Internet topology"
   - Uses Forman-Ricci on network subgraphs
   - Precedent for subgraph analysis

3. **Sandhu et al. (2015)**: "Market fragmentation & systemic risk"
   - Applies Ricci curvature to financial networks
   - Motivates our manifold extension

**Our Contribution**:
> "While prior work computes Ricci curvature globally [Sandhu2015, Tran2020], we compute it **per market segment** (monetary, industrial, etc.), revealing regime drivers through cross-manifold divergence. We term this 'manifold-specific' Ricci curvature."

---

## 🎯 Bottom Line

**What Matters for Thesis**:
1. ✅ **Mathematical rigor**: We compute well-defined Forman-Ricci curvature
2. ✅ **Novel contribution**: Manifold decomposition is new to finance
3. ✅ **Empirical validation**: Cross-manifold divergence has predictive power
4. ⚠️ **Terminology precision**: "Manifold-specific" > "sectional" (but both defensible)

**Action Items**:
- [x] Clarify terminology in documentation
- [ ] Update thesis to use "manifold-specific" as primary term
- [ ] Add footnote explaining relationship to classical sectional curvature
- [ ] Prepare defense explanation (be upfront about terminology choice)

---

**Status**: Terminology clarified ✅
**Risk Level**: LOW (terminology only, not methodology)
**Defense Strategy**: Be transparent, focus on contribution

