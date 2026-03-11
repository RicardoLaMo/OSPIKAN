# Silver Geometric Analysis: Cross-Validation & Enhancement Plan
## Investment Banking + Ricci Flow Perspective

**Author**: Senior Quant with IB Background + Geometric Analysis Expertise
**Date**: 2026-01-12
**Objective**: Cross-validate existing geometric silver analysis and enhance with multi-asset intelligence

---

## Executive Summary

Your current geometric framework is **mathematically rigorous** and demonstrates:
- ✅ **Persistent negative Ricci curvature** (-19.3 mean) → structural market fragmentation
- ✅ **Strong predictive power** (0.59-0.76 correlation with drawdowns)
- ✅ **Geometric Algebra rotors** working correctly (magnitude 0.30 ± 0.20)
- ✅ **MST stress highly anti-correlated** with Ricci curvature (-0.76)

### Critical Findings from Current Analysis

1. **Regime Performance Differential**:
   - STABLE: +27.6% annual return (56% of time)
   - STRESS: +13.0% annual return (19% of time)
   - TRANSITION: **-110% annual return** (8% of time) ← Key risk signal!

2. **Silver-Gold Correlation**: 0.80 mean (stable monetary link)
3. **Silver-DXY Beta**: -1.56 (strong dollar sensitivity)

---

## Part I: Cross-Validation of Current Methods

### 1.1 Geometric Feature Validation

#### Current Implementation Assessment

**Forman-Ricci Curvature** (implemented correctly):
```python
ricci_mean_60d: μ = -19.29, σ = 1.77
ricci_min_60d:  μ = -39.58, σ = 2.22
```

**Validation Checks**:
- ✅ Curvature always negative (structural stress persistent)
- ✅ Correlation with MST stress: -0.76 (complementary metrics)
- ✅ Predicts drawdowns: +0.59 correlation

**Investment Banking Interpretation**:
> Persistent negative curvature indicates **structural inefficiency** in silver markets.
> Unlike equities (which show positive curvature in bull markets), silver exhibits
> **bottleneck dynamics** even during rallies. This suggests:
> - Fragmented liquidity pools (futures, ETPs, physical, miners)
> - Information asymmetry between industrial and monetary demand
> - Regime-dependent correlation structure (not stationary)

#### Geometric Algebra Rotor Validation

**Current GA Implementation** (Cl(4,0)):
```python
ga_rotor_magnitude_60d: μ = 0.30, σ = 0.20
ga_bivector_energy_60d: μ = 33.0, σ = 7.2
```

**Validation**:
- ✅ Rotor magnitude now varies (fixed from previous zero-issue)
- ✅ Bivector energy shows rotational dynamics
- ⚠️ Low correlation with curvature metrics (orthogonal information)

**Enhancement Opportunity**:
The GA rotors are capturing **feature space rotation** but not explicitly linked to
**regime attractors**. Recommendation: Implement **geodesic distance to regime centroids**
using Riemannian metric induced by feature covariance.

---

### 1.2 Statistical Validation Requirements

#### Out-of-Sample Testing (Missing)

**Current Issue**: All regimes computed in-sample on full dataset.

**Recommendation**: Implement walk-forward regime prediction:
```python
# Pseudocode for validation
for train_end in pd.date_range('2015-01-01', '2025-01-01', freq='1Y'):
    train_data = features[:train_end]
    test_data = features[train_end:train_end + pd.DateOffset(months=6)]

    # Fit regime thresholds on training data
    stress_threshold = train_data['ricci_min_60d'].quantile(0.10)

    # Predict regimes on test data
    test_regimes = classify_regimes(test_data, stress_threshold)

    # Evaluate: Do predicted regimes actually predict future returns?
    evaluate_regime_predictive_power(test_data, test_regimes)
```

**Success Metrics**:
- Regime return differential holds out-of-sample (STABLE > STRESS)
- Transition regime predicts volatility spikes (lead time 5-10 days)
- Curvature thresholds stable across sub-periods

---

### 1.3 Sensitivity Analysis

#### Threshold Robustness

**Current Thresholds** (appears hardcoded):
```python
stress_threshold = ricci_p10_60d.quantile(0.10)  # Bottom 10%
transition_threshold = 0.8  # GA rotor magnitude
```

**Test Required**:
1. Vary stress threshold: [5%, 10%, 15%, 20%] percentile
2. Vary transition threshold: [0.6, 0.7, 0.8, 0.9]
3. Measure impact on:
   - Regime return differential
   - Transition frequency
   - False positive rate for TRANSITION regime

**Expected Outcome**:
- Main findings robust to ±5% threshold variation
- Transition regime highly sensitive (requires calibration)

---

## Part II: Multi-Asset Expansion Strategy

### 2.1 Current Asset Universe Limitations

**You have only 13 assets**:
- 5 silver (SI=F, SLV, SIVR, SIL, SILJ)
- 3 gold (GC=F, GLD)
- 1 copper (HG=F)
- 4 macro (DXY, TNX, VIX, SPY, QQQ)

**Critical Missing Assets**:

#### A. Industrial Metals Complex
```yaml
industrial_metals:
  - PA=F      # Palladium futures (industrial + auto catalyst)
  - PL=F      # Platinum futures (industrial + jewelry)
  - ALI=F     # Aluminum futures (construction cycle)
  - COPX      # Copper miners ETF (amplified cycle signal)
  - PALL      # Palladium ETF
  - PPLT      # Platinum ETF
```

**Why Critical**: Silver is **56% industrial demand** (per World Silver Survey 2025).
Current analysis treats it purely as monetary metal (gold correlation focus).

#### B. China/EM Growth Proxies
```yaml
growth_cycle:
  - FXI       # China Large-Cap ETF (70% of silver demand from EM)
  - EWZ       # Brazil ETF (commodity cycle)
  - GXC       # China Industrials ETF (direct manufacturing link)
  - ASHR      # China A-shares (policy-driven demand)
  - EEM       # Emerging Markets (broad EM risk)
```

**Why Critical**: Industrial demand driven by EM capex, especially China.

#### C. Real Assets / Inflation Complex
```yaml
real_assets:
  - TIP       # TIPS ETF (real yield proxy)
  - DBC       # Commodities ETF (broad commodity beta)
  - CRB       # CRB Index (inflation expectations)
  - BCI       # BCOM Index (commodity momentum)
```

**Why Critical**: Silver has **dual nature** (monetary + commodity). Need to separate
inflation-driven demand from industrial cycle.

#### D. Energy Complex
```yaml
energy:
  - CL=F      # WTI Crude Oil futures
  - USO       # Oil ETF
  - XLE       # Energy sector ETF
```

**Why Critical**: Energy costs impact mining economics (40% of silver is byproduct of
base metal mining). Oil also correlates with industrial activity.

#### E. Additional Silver Sector Depth
```yaml
silver_ecosystem:
  - PAAS      # Pan American Silver (top producer)
  - AG        # First Majestic Silver (pure play)
  - HL        # Hecla Mining (US-based)
  - MAG       # MAG Silver (development-stage signal)
  - SILV      # SilverCrest Metals (high-grade)
```

**Why Critical**: Miner stock prices lead physical silver (equity markets discount faster).
Spread between SIL (seniors) and SILJ (juniors) signals risk appetite.

---

### 2.2 Enhanced Universe Configuration

**Proposed `configs/silver_universe_enhanced.yaml`**:

```yaml
version: 2

defaults:
  provider: yfinance
  start_date: "2010-01-01"
  end_date: null
  interval: 1d

universe:
  silver_core:
    - SI=F
    - SLV
    - SIVR

  silver_miners:
    - SIL
    - SILJ
    - PAAS
    - AG
    - HL

  precious_metals:
    - GC=F
    - GLD
    - PA=F      # NEW
    - PL=F      # NEW
    - PALL      # NEW
    - PPLT      # NEW

  base_metals:
    - HG=F
    - COPX      # NEW
    - ALI=F     # NEW (if available)

  growth_proxies:
    - FXI       # NEW
    - GXC       # NEW
    - EWZ       # NEW
    - EEM       # NEW

  real_assets:
    - TIP       # NEW
    - DBC       # NEW

  energy:
    - CL=F      # NEW
    - USO       # NEW
    - XLE       # NEW

  macro:
    - DX-Y.NYB
    - ^TNX
    - ^IRX
    - ^VIX
    - SPY
    - QQQ

# Asset groupings for sectional curvature analysis
manifolds:
  monetary:
    description: "Assets driven by monetary/haven demand"
    symbols: [SI=F, SLV, GC=F, GLD, TIP]

  industrial:
    description: "Assets driven by industrial cycle"
    symbols: [SI=F, HG=F, COPX, PA=F, PL=F, FXI, GXC, CL=F]

  miners:
    description: "Silver mining equities (leading indicator)"
    symbols: [SIL, SILJ, PAAS, AG, HL]
```

**Expected Impact**:
- Asset count: 13 → **~40 assets**
- Curvature signals become **regime-specific** (monetary vs industrial)
- Cross-manifold divergence signals regime transitions

---

## Part III: Advanced Geometric Enhancements

### 3.1 Sectional Curvature Analysis

**Current**: Single Ricci curvature computed across all assets (mixed regime signals).

**Enhancement**: Compute **sectional curvatures** for asset subsets:

```python
def compute_sectional_curvatures(
    returns: pd.DataFrame,
    manifolds: Dict[str, List[str]],
    window: int = 60
) -> pd.DataFrame:
    """
    Compute Ricci curvature for different asset subsets.

    Sectional curvature captures geometry of specific market segments.
    Divergence between monetary and industrial curvature → regime shift.
    """
    results = pd.DataFrame(index=returns.index)

    for name, symbols in manifolds.items():
        subset_returns = returns[symbols]
        ricci = rolling_ricci_curvature(subset_returns, window=window)
        results[f'ricci_mean_{name}_60d'] = ricci['ricci_mean']
        results[f'ricci_min_{name}_60d'] = ricci['ricci_min']

    # Compute cross-manifold divergence (key regime signal)
    if 'ricci_mean_monetary_60d' in results and 'ricci_mean_industrial_60d' in results:
        results['curvature_divergence_60d'] = (
            results['ricci_mean_monetary_60d'] - results['ricci_mean_industrial_60d']
        )

    return results
```

**Investment Banking Insight**:
> When monetary curvature > industrial curvature: Haven demand dominates (risk-off).
> When industrial curvature > monetary curvature: Growth demand dominates (risk-on).
> **Divergence spikes** predict regime transitions.

---

### 3.2 Normalized Ricci Flow

**Current**: Ricci flow exhibits overflow issues (extreme negative curvatures).

**Enhancement**: Implement **normalized Ricci flow** (stable evolution):

```python
def normalized_ricci_flow(
    metric: np.ndarray,
    ricci_tensor: np.ndarray,
    dt: float = 0.01,
    steps: int = 50
) -> Tuple[np.ndarray, List[float]]:
    """
    Normalized Ricci flow: dg/dt = -2(Ric - r·g)
    where r = average scalar curvature (volume-preserving).

    This prevents collapse and enables stable regime comparison.
    """
    g = metric.copy()
    scalar_curvatures = []

    for _ in range(steps):
        # Compute scalar curvature (trace of Ricci tensor)
        r = np.trace(ricci_tensor @ np.linalg.inv(g)) / g.shape[0]
        scalar_curvatures.append(r)

        # Normalized flow (volume-preserving)
        dg_dt = -2 * (ricci_tensor - r * g)
        g += dt * dg_dt

        # Recompute Ricci tensor for next step
        ricci_tensor = compute_ricci_tensor(g)

    return g, scalar_curvatures
```

**Key Metrics**:
1. **Convergence rate**: How fast does metric stabilize?
2. **Fixed point curvature**: What is the limiting scalar curvature?
3. **Soliton detection**: Does flow converge to Ricci soliton (stable attractor)?

**Regime Interpretation**:
- Fast convergence → Stable regime
- Slow/no convergence → Transition regime
- Soliton detection → Persistent regime attractor

---

### 3.3 Geodesic Distance Metrics

**Current**: Euclidean distance used for correlation distance.

**Enhancement**: Compute **geodesic distances** on Riemannian manifold:

```python
def geodesic_distance_regimes(
    features: pd.DataFrame,
    regime_labels: pd.Series,
    metric_cols: List[str]
) -> pd.DataFrame:
    """
    Compute geodesic distance between regime centroids.

    Uses Riemannian metric induced by feature covariance.
    Captures true "market distance" between regimes.
    """
    results = []

    for regime in regime_labels.unique():
        mask = (regime_labels == regime)
        regime_features = features.loc[mask, metric_cols]

        # Regime centroid
        centroid = regime_features.mean()

        # Covariance = Riemannian metric
        metric = regime_features.cov().values + 1e-6 * np.eye(len(metric_cols))
        metric_inv = np.linalg.inv(metric)

        # Geodesic distance to centroid for all points
        for idx in features.index:
            point = features.loc[idx, metric_cols].values
            diff = point - centroid.values

            # Geodesic distance: sqrt(diff^T · metric^{-1} · diff)
            dist = np.sqrt(diff @ metric_inv @ diff)

            results.append({
                'date': idx,
                'regime': regime,
                'geodesic_distance_to_centroid': dist
            })

    return pd.DataFrame(results)
```

**Trading Signal**:
```python
# When geodesic distance exceeds threshold → regime exit imminent
if geodesic_distance > 2.0:  # 2 standard deviations in Riemannian sense
    signal = "REGIME_TRANSITION_WARNING"
```

---

### 3.4 Parallel Transport for Beta Consistency

**Insight**: Rolling betas (e.g., `dxy_beta_60d`) are **path-dependent** in curved space.

**Enhancement**: Use **parallel transport** to adjust betas for curvature:

```python
def parallel_transport_beta(
    beta_series: pd.Series,
    ricci_curvature: pd.Series,
    dt: float = 1.0
) -> pd.Series:
    """
    Adjust rolling beta for Ricci curvature using parallel transport.

    In flat space: beta is frame-independent.
    In curved space: beta depends on path taken (connection).

    Parallel transport gives "intrinsic beta" invariant to regime path.
    """
    corrected_beta = beta_series.copy()

    for i in range(1, len(beta_series)):
        # Christoffel symbol approximation (proportional to curvature)
        curvature = ricci_curvature.iloc[i]

        # Parallel transport correction
        # dbeta = -Gamma * beta * dt where Gamma ~ curvature
        correction = -0.5 * curvature * beta_series.iloc[i-1] * dt

        corrected_beta.iloc[i] = beta_series.iloc[i] + correction

    return corrected_beta
```

**Investment Banking Application**:
> Traditional rolling betas assume flat space (stationary correlations).
> In regime-switching markets, **curvature-adjusted betas** better capture true sensitivity.
> Use for **risk management**: DXY beta is understated during STRESS regimes.

---

## Part IV: Investment Banking Feature Engineering

### 4.1 Futures Curve Structure (Roll Yield)

**Data Source**: Silver futures term structure (SI=F, SIZ26, SIH27, etc.)

**Features**:
```python
def compute_futures_curve_features(futures_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze silver futures term structure (forward curve).

    Features:
    - roll_yield: (F2 - F1) / F1 (contango vs backwardation)
    - curve_slope: regression slope across maturities
    - curve_curvature: second derivative (butterfly)
    - basis: Spot - Front month (convenience yield)
    """
    features = pd.DataFrame(index=futures_prices.index)

    # Roll yield (1-2 month spread)
    features['roll_yield'] = (
        futures_prices['SIZ26'] - futures_prices['SI=F']
    ) / futures_prices['SI=F']

    # Curve slope (OLS across maturities)
    maturities = [1, 3, 6, 12]  # months
    for idx in futures_prices.index:
        prices = futures_prices.loc[idx, ['SIF26', 'SIG26', 'SIN26', 'SIF27']]
        slope, _ = np.polyfit(maturities, prices, deg=1)
        features.loc[idx, 'curve_slope'] = slope

    # Backwardation indicator (spot > future = supply tightness)
    features['backwardation'] = (features['roll_yield'] < 0).astype(int)

    return features
```

**Regime Signal**:
- **Backwardation** (negative roll) → Supply shortage → Bullish for silver
- **Steep contango** → Excess supply → Bearish

---

### 4.2 ETP Flow Analysis

**Data Source**: SLV holdings (available daily), SIVR holdings, ETF.com flows

**Features**:
```python
def compute_etp_flow_features(
    etf_prices: pd.DataFrame,
    etf_shares_outstanding: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze investor sentiment via ETP flows.

    Silver ETPs (SLV, SIVR) are **retail investor proxy**.
    Miners (SIL, SILJ) are **institutional risk appetite**.
    """
    features = pd.DataFrame(index=etf_prices.index)

    # Daily flow estimate (assumes constant NAV per share)
    for etf in ['SLV', 'SIVR']:
        shares_chg = etf_shares_outstanding[etf].diff()
        features[f'{etf}_flow_usd'] = shares_chg * etf_prices[etf]

    # Cumulative flows (investor positioning)
    features['slv_cum_flow_60d'] = features['SLV_flow_usd'].rolling(60).sum()

    # Flow divergence (SLV flows vs price) = positioning extreme
    ret = etf_prices['SLV'].pct_change()
    features['flow_divergence'] = (
        features['SLV_flow_usd'].rolling(20).mean() - ret.rolling(20).mean()
    )

    return features
```

**Investment Banking Signal**:
> When **flow_divergence > 2σ**: Retail investors buying aggressively despite flat price → Potential reversal setup (contrarian signal).

---

### 4.3 Supply/Demand Imbalance Metrics

**Data Source**: World Silver Survey 2025 (annual data)

**Features**:
```python
def incorporate_survey_data(
    daily_features: pd.DataFrame,
    survey_path: str = "data/external/world_silver_survey.csv"
) -> pd.DataFrame:
    """
    Blend annual supply/demand data with daily price features.

    Key metrics from 2025 survey:
    - Market deficit: 148.9 Moz (2024), 117.6 Moz (2025E)
    - Industrial demand: 680.5 Moz (record high)
    - PV demand: growing despite "thrifting"
    """
    survey = pd.read_csv(survey_path, parse_dates=['year'])

    # Annual metrics (broadcast to daily)
    for year in survey['year'].dt.year.unique():
        year_data = survey[survey['year'].dt.year == year].iloc[0]
        mask = daily_features.index.year == year

        daily_features.loc[mask, 'market_deficit_moz'] = year_data['deficit_moz']
        daily_features.loc[mask, 'industrial_pct'] = year_data['industrial_demand_moz'] / year_data['total_demand_moz']
        daily_features.loc[mask, 'investment_pct'] = year_data['investment_demand_moz'] / year_data['total_demand_moz']

    # Deficit surprise (actual vs consensus forecast)
    daily_features['deficit_surprise'] = (
        daily_features['market_deficit_moz'] - daily_features['market_deficit_moz'].shift(252)
    )

    return daily_features
```

**Regime Overlay**:
```python
# Fundamental regime score
fundamental_score = (
    2.0 * deficit_moz / 150.0 +  # Weight by typical deficit level
    1.0 * industrial_pct +
    0.5 * investment_pct
)

# Combine with geometric regime
if fundamental_score > 2.5 and geometric_regime == 'STABLE':
    regime = 'STRUCTURAL_BULL'
elif fundamental_score < 1.5 and geometric_regime == 'STRESS':
    regime = 'STRUCTURAL_BEAR'
```

---

### 4.4 Cross-Sectional Momentum (Silver Complex)

**Intuition**: Relative strength within silver ecosystem signals regime conviction.

**Implementation**:
```python
def compute_cross_sectional_momentum(
    silver_assets: pd.DataFrame,
    window: int = 20
) -> pd.DataFrame:
    """
    Rank silver assets by momentum (physical, ETPs, miners).

    Strong cross-sectional trend = regime conviction.
    Divergence = warning signal.
    """
    returns = silver_assets.pct_change()
    momentum = returns.rolling(window).mean()

    # Z-score ranks (cross-sectional)
    ranks = momentum.rank(axis=1, pct=True)

    features = pd.DataFrame(index=silver_assets.index)

    # Miner leadership (SIL/SILJ rank > 0.7) = risk-on
    features['miner_leadership'] = ranks[['SIL', 'SILJ']].mean(axis=1)

    # Physical premium (SI=F rank > SLV rank) = supply tightness
    features['physical_premium'] = ranks['SI=F'] - ranks['SLV']

    # Dispersion (high = divergence = regime uncertainty)
    features['cross_sectional_dispersion'] = momentum.std(axis=1)

    return features
```

---

## Part V: Implementation Roadmap

### Phase 1: Multi-Asset Expansion (Week 1)

**Tasks**:
1. Update `configs/silver_universe_enhanced.yaml` with 40 assets
2. Re-run pipeline: `ingest → align → features`
3. Validate new asset quality (missing rate < 5%)

**Deliverables**:
- `data/processed/silver_features_enhanced_<RUN_ID>.parquet`
- New columns: `ricci_mean_monetary_60d`, `ricci_mean_industrial_60d`, `curvature_divergence_60d`

---

### Phase 2: Sectional Curvature (Week 2)

**Tasks**:
1. Implement `compute_sectional_curvatures()` in `src/geometry/graph_curvature.py`
2. Add to feature pipeline
3. Create visualization: Curvature divergence vs silver returns

**Deliverables**:
- New figure: `sectional_curvature_regimes.png`
- Analysis: Does curvature divergence predict regime transitions?

---

### Phase 3: Investment Banking Features (Week 3)

**Tasks**:
1. Implement futures curve, ETP flows, survey integration
2. Add to `src/analysis/features.py` as optional module
3. Validate: Do IB features add predictive power?

**Deliverables**:
- New features: `roll_yield`, `slv_flow_60d`, `market_deficit_moz`
- Regression: `regime_transition ~ curvature + roll_yield + flows`

---

### Phase 4: Advanced Geometry (Week 4)

**Tasks**:
1. Implement normalized Ricci flow (fix overflow issues)
2. Add geodesic distance metrics
3. Parallel transport for beta adjustment

**Deliverables**:
- `src/geometry/advanced_ricci_flow.py`
- New metrics: `geodesic_dist_to_stable`, `curvature_adjusted_dxy_beta`

---

### Phase 5: Out-of-Sample Validation (Week 5)

**Tasks**:
1. Implement walk-forward regime prediction
2. Sensitivity analysis (threshold robustness)
3. Comparison with HMM, Markov-switching baselines

**Deliverables**:
- `reports/silver/validation/out_of_sample_results.csv`
- Figures: Regime prediction accuracy, Sharpe improvement

---

## Part VI: Expected Improvements

### Quantitative Targets

| Metric | Current | Enhanced Target |
|--------|---------|-----------------|
| Regime return differential | STABLE: +27.6% vs STRESS: +13.0% | STABLE: +30%+ vs STRESS: +10% (better separation) |
| Drawdown prediction R² | 0.35 (0.59² correlation) | 0.50+ (multi-asset curvature) |
| Transition regime false positive | Unknown (not validated) | < 20% (out-of-sample) |
| Feature dimensionality | 40 features | 60+ features (IB + advanced geometry) |
| Asset coverage | 13 assets (gold-heavy) | 40 assets (balanced monetary/industrial) |

### Qualitative Improvements

1. **Regime Narratives**: Move from "STABLE/STRESS/TRANSITION" to:
   - **Monetary Haven Regime** (gold-driven, risk-off)
   - **Industrial Growth Regime** (China/EM-driven, risk-on)
   - **Supply Shock Regime** (deficit-driven, backwardation)
   - **Speculative Frenzy Regime** (ETP flows spike, divergence)

2. **Real-Time Applicability**:
   - Current: Regimes known only post-hoc (rolling window)
   - Enhanced: Leading indicators (curvature divergence, miner leadership)

3. **Risk Management**:
   - Current: Binary regime signals
   - Enhanced: Continuous geodesic distance (proximity to TRANSITION)

---

## Part VII: Academic Contributions

### Novel Methods for Finance Literature

1. **Sectional Ricci Curvature for Asset Regimes**
   - First application of sectional curvature to financial manifolds
   - Captures regime-specific geometry (monetary vs industrial)

2. **Geometric Algebra for Regime Transitions**
   - Rotors as regime shift operators (continuous vs discrete HMM)
   - Bivector energy as transition activity metric

3. **Normalized Ricci Flow for Market Stability**
   - Detect Ricci solitons (persistent regime attractors)
   - Convergence rate as stability proxy

4. **Parallel Transport for Beta Adjustment**
   - Curvature-corrected betas (non-stationary correlations)
   - Applications to risk management in curved markets

---

## Part VIII: Immediate Next Steps

### Priority 1: Multi-Asset Expansion (Do First)

```bash
# 1. Update config
vim configs/silver_universe_enhanced.yaml

# 2. Ingest new assets
python src/pipeline/silver_pipeline.py ingest \
    --config configs/silver_universe_enhanced.yaml

# 3. Align and compute features
LATEST_RUN=$(cat data/raw/_runs/LATEST)
python src/pipeline/silver_pipeline.py align \
    --config configs/silver_universe_enhanced.yaml \
    --run-id $LATEST_RUN

python src/pipeline/silver_pipeline.py features \
    --panel data/interim/silver_panel_close_${LATEST_RUN}.csv \
    --out data/processed/silver_features_enhanced_${LATEST_RUN}.parquet
```

### Priority 2: Sectional Curvature Implementation

```python
# Add to src/geometry/graph_curvature.py

def rolling_sectional_ricci(
    returns: pd.DataFrame,
    manifolds: Dict[str, List[str]],
    window: int = 60,
    min_assets: int = 3
) -> pd.DataFrame:
    """New function for sectional curvature analysis."""
    # Implementation as described in Part III
    pass
```

### Priority 3: Validation Notebook

Create `notebooks/geometric_validation.ipynb`:
1. Load enhanced features
2. Compare sectional curvatures
3. Measure curvature divergence vs returns
4. Out-of-sample regime prediction test

---

## Conclusion

Your current geometric framework is **mathematically sound and empirically strong**.
The proposed enhancements will:

1. ✅ **Cross-validate** with out-of-sample testing (currently missing)
2. ✅ **Expand** multi-asset intelligence (13 → 40 assets)
3. ✅ **Separate** monetary vs industrial drivers (sectional curvature)
4. ✅ **Add** investment banking domain features (flows, curve, fundamentals)
5. ✅ **Advance** geometric methods (normalized flow, geodesics, parallel transport)

**Bottom Line**: You've built a strong foundation. Now we're adding **regime-specific
intelligence** and **multi-asset context** to make it production-grade for investment
decisions.

---

**Next Action**: Shall I proceed with implementing the multi-asset expansion and
sectional curvature analysis?
