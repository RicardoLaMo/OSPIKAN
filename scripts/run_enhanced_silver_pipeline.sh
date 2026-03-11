#!/bin/bash
# Enhanced Silver Pipeline with Multi-Asset Intelligence
# Runs full pipeline from data ingest to sectional curvature analysis

set -e  # Exit on error

echo "==================================================================="
echo "Enhanced Silver Geometric Analysis Pipeline"
echo "==================================================================="
echo ""

# Configuration
CONFIG_ENHANCED="configs/silver_universe_enhanced.yaml"
CONFIG_BASE="configs/silver_universe.yaml"

# Prompt user for which config to use
echo "Select configuration:"
echo "  [1] Enhanced (40+ assets, sectional curvature) - RECOMMENDED"
echo "  [2] Base (13 assets, original analysis)"
echo ""
read -p "Enter choice [1]: " CHOICE
CHOICE=${CHOICE:-1}

if [ "$CHOICE" = "2" ]; then
    CONFIG=$CONFIG_BASE
    echo "Using base configuration: $CONFIG_BASE"
else
    CONFIG=$CONFIG_ENHANCED
    echo "Using enhanced configuration: $CONFIG_ENHANCED"
fi

echo ""
echo "=== Step 1: Data Ingest ==="
echo "Fetching market data from yfinance..."
python src/pipeline/silver_pipeline.py ingest --config $CONFIG

# Get the latest run ID
LATEST_RUN=$(cat data/raw/_runs/LATEST)
echo "Run ID: $LATEST_RUN"
echo ""

echo "=== Step 2: Data Alignment ==="
echo "Building aligned close-price panel..."
python src/pipeline/silver_pipeline.py align \
    --config $CONFIG \
    --run-id $LATEST_RUN

PANEL_PATH="data/interim/silver_panel_close_${LATEST_RUN}.csv"
echo "Panel created: $PANEL_PATH"
echo ""

echo "=== Step 3: Data Quality Check ==="
python src/pipeline/silver_pipeline.py qa \
    --panel $PANEL_PATH \
    --required "SI=F,GC=F,DX-Y.NYB,SPY" \
    --max-missing-rate 0.05

echo ""
echo "=== Step 4: Feature Engineering ==="
echo "Computing geometric features (Ricci curvature, GA rotors, sectional curvature)..."
python src/pipeline/silver_pipeline.py features \
    --panel $PANEL_PATH \
    --silver "SI=F" \
    --gold "GC=F" \
    --dxy "DX-Y.NYB" \
    --y10 "^TNX" \
    --spx "SPY" \
    --vix "^VIX"

FEATURES_PATH="data/processed/silver_features_${LATEST_RUN}.parquet"
echo "Features created: $FEATURES_PATH"
echo ""

# If enhanced config, compute sectional curvatures
if [ "$CHOICE" = "1" ]; then
    echo "=== Step 5: Sectional Curvature Analysis (Enhanced) ==="
    python scripts/compute_sectional_curvatures.py \
        --features $FEATURES_PATH \
        --config $CONFIG \
        --out-dir "reports/silver/runs/${LATEST_RUN}"
    echo ""
fi

echo "=== Step 6: Geometric Regime Analysis ==="
OUT_DIR="reports/silver/runs/${LATEST_RUN}"
mkdir -p $OUT_DIR

python scripts/silver_geometric_regimes.py \
    --features $FEATURES_PATH \
    --out-dir $OUT_DIR

echo ""
echo "=== Step 7: Regime Method Comparison ==="
python scripts/compare_regime_methods.py \
    --features $FEATURES_PATH \
    --out-dir $OUT_DIR/analysis

echo ""
echo "=== Step 8: Geometric Signal Analysis ==="
python scripts/analyze_geometric_signals.py \
    --features $FEATURES_PATH \
    --out-dir $OUT_DIR/signals

echo ""
echo "=== Step 9: Ricci Flow Shock Scenarios ==="
python scripts/ricci_flow_shock_scenarios.py \
    --panel $PANEL_PATH \
    --out-dir $OUT_DIR/ricci_flow

echo ""
echo "==================================================================="
echo "Pipeline Complete!"
echo "==================================================================="
echo ""
echo "Outputs saved to: $OUT_DIR"
echo ""
echo "Key files:"
echo "  - Features: $FEATURES_PATH"
echo "  - Panel: $PANEL_PATH"
echo "  - Reports: $OUT_DIR/"
echo ""
echo "View results:"
if [ "$CHOICE" = "1" ]; then
    echo "  - Sectional curvatures: $OUT_DIR/figures/sectional_curvature_timeseries.png"
    echo "  - Manifold divergence: $OUT_DIR/figures/curvature_divergence_regimes.png"
fi
echo "  - Geometric regimes: $OUT_DIR/figures/silver_price_geometric_regimes.png"
echo "  - Signal timeline: $OUT_DIR/signals/figures/geometric_signals_timeline.png"
echo "  - Regime comparison: $OUT_DIR/analysis/figures/regime_method_comparison.png"
echo ""
echo "Next steps:"
echo "  - Review SILVER_GEOMETRIC_ENHANCEMENT_PLAN.md for validation steps"
echo "  - Run out-of-sample validation (coming soon)"
echo "  - Compare base vs enhanced pipeline results"
