# Silver Analysis: Quick Command Reference

**Date**: 2026-01-12
**Status**: ✅ Ready to Execute

---

## 🚀 Quickest Path: Add Macro to Existing Data

```bash
# Use your existing panel, add macro features
python src/pipeline/silver_pipeline.py features \
    --panel data/interim/silver_panel_close_20260112-165823_ff97dd3902.csv \
    --macro-features

# Run analysis with new features
FEATURES_PATH=$(ls -t data/processed/silver_features_*.parquet | head -1)
FEATURES_PATH=$FEATURES_PATH sh scripts/run_silver_end_to_end.sh

# View results
open reports/silver/runs/*/figures/silver_price_geometric_regimes.png
```

**Time**: ~5-10 minutes
**New features**: +30-40 (treasury spreads, credit, ratios)

---

## 🎯 Full Enhanced Pipeline (50+ Assets)

```bash
# One-command full pipeline with macro enhancement
sh scripts/run_silver_end_to_end.sh \
    configs/silver_universe_macro_enhanced.yaml \
    reports/silver/runs_macro_full

# View results
open reports/silver/runs_macro_full/*/figures/*.png
```

**Time**: ~15-20 minutes (data download + computation)
**Assets**: 14 → 50+
**Features**: 40 → 80+

---

## 📊 Compare Base vs Enhanced

```bash
# Base run (your original)
PANEL_PATH=data/interim/silver_panel_close_20260112-165823_ff97dd3902.csv \
FEATURES_PATH=data/processed/silver_features_20260112-165823_ff97dd3902.parquet \
sh scripts/run_silver_end_to_end.sh

# Enhanced run (with macro)
sh scripts/run_silver_end_to_end.sh \
    configs/silver_universe_macro_enhanced.yaml \
    reports/silver/runs_comparison

# Compare outputs
diff -r reports/silver/runs/20260112-165823_ff97dd3902 \
         reports/silver/runs_comparison/*/
```

---

## 🔍 Verify New Features

```python
import pandas as pd

# Load enhanced features
df = pd.read_parquet('data/processed/silver_features_<RUN_ID>.parquet')

# Check macro features exist
macro_features = [
    'ig_hy_spread',          # Credit cycle
    'term_premium',          # Yield curve
    'gold_oil_ratio',        # Monetary vs energy
    'copper_gold_ratio',     # Dr. Copper
    'equity_bond_ratio',     # Risk-on/off
    'vix_term_slope',        # VIX structure
    'oil_level',             # Energy
]

for feat in macro_features:
    if feat in df.columns:
        print(f"✅ {feat}: {df[feat].notna().sum()} valid values")
    else:
        print(f"❌ {feat}: Not found (asset may be missing from panel)")
```

---

## 📚 Documentation Quick Links

| Document | Purpose |
|----------|---------|
| `INTEGRATION_SUMMARY.md` | **START HERE** - Overview of enhancements |
| `docs/MACRO_FEATURES_GUIDE.md` | Treasury, credit, ratios explained |
| `docs/CROSS_VALIDATION_SUMMARY.md` | Validation results + targets |
| `docs/SILVER_GEOMETRIC_ENHANCEMENT_PLAN.md` | Full 5-phase roadmap |
| `configs/silver_universe_macro_enhanced.yaml` | Enhanced asset universe (50+ assets) |

---

## ⚙️ Configuration Options

### Base Configuration (Your Original)
```yaml
# configs/silver_universe.yaml
- 14 assets (silver, gold, copper, macro)
- No credit spreads, no energy, no EM
- Use for: Baseline comparison
```

### Macro-Enhanced Configuration (New)
```yaml
# configs/silver_universe_macro_enhanced.yaml
- 50+ assets (adds energy, credit, treasuries, EM, currencies)
- Full macro features (treasury spreads, credit, ratios)
- Use for: Production analysis
```

---

## 🎛️ Feature Flags

### Base Features (Default, No Flags)
```bash
python src/pipeline/silver_pipeline.py features --panel <PANEL>
```
- Computes: Ricci curvature, GA rotors, MST stress, basic macro
- Output: ~40 features

### With Macro Features
```bash
python src/pipeline/silver_pipeline.py features \
    --panel <PANEL> \
    --macro-features  # ADD THIS
```
- Computes: Everything above + treasury spreads, credit spreads, cross-asset ratios
- Output: ~80 features

### With Deep Geometry
```bash
python src/pipeline/silver_pipeline.py features \
    --panel <PANEL> \
    --deep-geometry  # Ricci flow stability (slow)
```
- Computes: Everything above + Ricci flow stability metric
- Runtime: +10-15 minutes

### All Enhancements
```bash
python src/pipeline/silver_pipeline.py features \
    --panel <PANEL> \
    --macro-features \
    --deep-geometry
```
- Maximum feature set
- Runtime: +15-20 minutes

---

## 📈 Expected Outputs

### Base Run (14 assets, no macro)
```
data/processed/silver_features_<RUN_ID>.parquet
- 40 columns
- Includes: log_return_1d, realized_vol_20d, momentum_10d, drawdown,
            gsr, gold_log_return_1d, dxy_beta_60d,
            mst_stress_60d, ricci_mean_60d, ricci_min_60d,
            ga_rotor_magnitude_60d, ga_bivector_energy_60d
```

### Enhanced Run (50+ assets, with macro)
```
data/processed/silver_features_<RUN_ID>.parquet
- 80+ columns
- Base features PLUS:
  - Treasury: slope_5y_10y, term_premium, yield_curve_level
  - Credit: ig_hy_spread, hy_treasury_spread, em_ig_spread, hyg_equity_beta_60d
  - Ratios: gold_oil_ratio, copper_gold_ratio, equity_bond_ratio, em_dm_equity_ratio
  - Volatility: vix_term_slope, equity_bond_vol_ratio, vix_vol_20d
  - Energy: oil_level, oil_return_20d, brent_wti_spread
```

---

## 🐛 Troubleshooting

### Issue: "Asset not found" during ingest
```bash
# Some assets may not be available in yfinance (e.g., ^FVX, BZ=F)
# Solution: Pipeline continues with available assets (graceful degradation)
# Check: data/raw/_runs/<RUN_ID>/run_metadata.json for errors
```

### Issue: "Macro features not computed"
```bash
# If HYG, LQD, CL=F etc. not in panel, features skipped
# Check asset coverage:
python -c "
import pandas as pd
df = pd.read_csv('data/interim/silver_panel_close_<RUN_ID>.csv', index_col=0)
print('Available assets:', df.columns.tolist())
print('Missing rates:', df.isna().mean().sort_values(ascending=False).head(10))
"
```

### Issue: "Features computation slow"
```bash
# Normal: 50+ assets with 60-day rolling windows takes 5-10 minutes
# Speed up: Use smaller window in config (e.g., 30d instead of 60d)
# Or: Skip --deep-geometry flag (saves 10 minutes)
```

---

## 🎯 Key Metrics to Check

### After running enhanced pipeline, verify improvements:

```python
import pandas as pd

# Load base vs enhanced features
base = pd.read_parquet('data/processed/silver_features_20260112-165823_ff97dd3902.parquet')
enhanced = pd.read_parquet('data/processed/silver_features_<NEW_RUN_ID>.parquet')

print(f"Base features: {len(base.columns)}")
print(f"Enhanced features: {len(enhanced.columns)}")
print(f"New features: {len(enhanced.columns) - len(base.columns)}")

# Check macro feature coverage
macro_cols = [c for c in enhanced.columns if any(x in c for x in
    ['spread', 'ratio', 'premium', 'slope', 'oil', 'energy'])]
print(f"\nMacro features: {len(macro_cols)}")
for col in macro_cols[:10]:  # Show first 10
    coverage = enhanced[col].notna().mean()
    print(f"  {col}: {coverage:.1%} coverage")
```

---

## 🔄 Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                     WORKFLOW OPTIONS                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option A: Quickest (Use Existing Data)                     │
│  ─────────────────────────────────────                      │
│  1. Add --macro-features flag to features computation       │
│  2. Run analysis on enhanced features                       │
│  Time: ~5 minutes  |  New features: +30-40                  │
│                                                              │
│  Option B: Full Enhancement (Recommended)                   │
│  ────────────────────────────────────────                   │
│  1. Use configs/silver_universe_macro_enhanced.yaml         │
│  2. Run full pipeline (ingest → align → features → analyze) │
│  Time: ~15 minutes  |  New assets: 14→50+  |  Features: 80+ │
│                                                              │
│  Option C: Comparison Study (For Thesis)                    │
│  ─────────────────────────────────────────                  │
│  1. Keep base run as is                                     │
│  2. Run enhanced pipeline in parallel                       │
│  3. Compare regime classifications, predictions             │
│  Time: ~20 minutes  |  Output: Side-by-side comparison      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📞 Support

- **Questions about features**: See `docs/MACRO_FEATURES_GUIDE.md`
- **Questions about integration**: See `INTEGRATION_SUMMARY.md`
- **Questions about validation**: See `docs/CROSS_VALIDATION_SUMMARY.md`
- **Questions about roadmap**: See `docs/SILVER_GEOMETRIC_ENHANCEMENT_PLAN.md`

---

**Last Updated**: 2026-01-12
**Status**: ✅ All enhancements complete and tested
**Compatibility**: Backward-compatible with all existing code
