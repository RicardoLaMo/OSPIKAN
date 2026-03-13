#!/usr/bin/env python3
"""
Run FIXED out-of-sample validation (no data leakage).

CRITICAL FIXES IMPLEMENTED:
1. Regime thresholds computed per-fold on training data only
2. Label-defining features excluded from predictors (no circularity)
3. Predictions deduplicated and Sharpe computed on predicted regimes
4. All models aligned to common date windows

Usage:
    python scripts/run_validation_fixed.py <FEATURES_FILE>

Example:
    python scripts/run_validation_fixed.py data/processed/silver_features_*.parquet

Author: Phase 3 - Fixed Validation
Date: 2026-01-12
"""

import os
import sys
from pathlib import Path

# Add repo root to path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

from src.validation.out_of_sample_fixed import (
    ValidationConfig,
    compare_models,
    print_comparison_summary,
    save_validation_report,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_validation_fixed.py <FEATURES_FILE>")
        sys.exit(1)

    features_path = sys.argv[1]
    output_dir = Path("reports/silver/validation_fixed")

    print("=" * 80)
    print("OUT-OF-SAMPLE VALIDATION (FIXED - No Data Leakage)")
    print("=" * 80)
    print("\nMETHODOLOGICAL FIXES:")
    print("  1. Regime thresholds computed per-fold on training data only")
    print("  2. Label-defining features excluded from predictors")
    print("  3. Predictions deduplicated before metrics computation")
    print("  4. All models validated on identical date windows")
    print("  5. Sharpe ratios computed on PREDICTED regimes (trading reality)")
    print("\n" + "=" * 80)

    print(f"\nFeatures: {features_path}")
    print(f"Output:   {output_dir}/")

    # Load features
    print("\nLoading features...")
    features = pd.read_parquet(features_path)
    print(f"  Loaded {len(features):,} observations, {len(features.columns)} features")
    print(f"  Date range: {features.index.min().strftime('%Y-%m-%d')} to "
          f"{features.index.max().strftime('%Y-%m-%d')}")

    # Configuration
    config = ValidationConfig(
        train_window_days=756,   # ~3 years
        test_window_days=252,    # ~1 year
        step_size_days=63,       # ~3 months (quarterly refit)
        regime_col=None,         # Use heuristic (computed per-fold)
        returns_col='log_return_1d',
        # FIXED: Configurable thresholds
        ricci_percentile=50.0,
        drawdown_low_percentile=33.3,
        drawdown_high_percentile=66.7,
        # FIXED: Exclude label-defining features
        exclude_label_features=True,
    )

    print(f"\nValidation Configuration:")
    print(f"  Train window: {config.train_window_days} days (~{config.train_window_days/252:.1f} years)")
    print(f"  Test window:  {config.test_window_days} days (~{config.test_window_days/252:.1f} year)")
    print(f"  Step size:    {config.step_size_days} days (~{config.step_size_days/21:.1f} months)")
    print(f"  Ricci threshold: {config.ricci_percentile}th percentile (computed per-fold)")
    print(f"  Drawdown thresholds: {config.drawdown_low_percentile}th / {config.drawdown_high_percentile}th percentile")
    print(f"  Exclude label-defining features: {config.exclude_label_features}")

    # Run validation
    results = compare_models(features, config, regime_col=None)

    # Print summary
    print_comparison_summary(results)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_validation_report(results, output_dir, features_path)

    # Feature importance comparison
    print("\n" + "=" * 80)
    print("TOP 10 FEATURES BY IMPORTANCE")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name.upper()} Model:")
        top_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:10]
        for i, (feat, imp) in enumerate(top_features, 1):
            print(f"  {i:2d}. {feat:40s}: {imp:.4f}")

    # Comparison analysis
    print("\n" + "=" * 80)
    print("METHODOLOGY VALIDATION")
    print("=" * 80)

    base_acc = results['base'].accuracy_scores
    macro_acc = results['macro'].accuracy_scores if 'macro' in results else None
    sect_acc = results['sectional'].accuracy_scores if 'sectional' in results else None

    print("\nKey Findings:")
    print(f"  BASE model:      {sum(base_acc)/len(base_acc):.4f} accuracy "
          f"({len(results['base'].predictions_df_deduplicated):,} deduplicated predictions)")

    if macro_acc:
        improvement_macro = (sum(macro_acc) / len(macro_acc)) - (sum(base_acc) / len(base_acc))
        print(f"  MACRO model:     {sum(macro_acc)/len(macro_acc):.4f} accuracy "
              f"({improvement_macro:+.4f} vs BASE)")

    if sect_acc:
        improvement_sect = (sum(sect_acc) / len(sect_acc)) - (sum(base_acc) / len(base_acc))
        print(f"  SECTIONAL model: {sum(sect_acc)/len(sect_acc):.4f} accuracy "
              f"({improvement_sect:+.4f} vs BASE)")

    print(f"\nTransition Detection (Deduplicated):")
    for name, result in results.items():
        rate = result.regime_transitions_detected / result.regime_transitions_total \
               if result.regime_transitions_total > 0 else 0
        print(f"  {name.upper():12s}: {rate:>6.1%} of transitions detected "
              f"({result.regime_transitions_detected}/{result.regime_transitions_total})")

    print("\nSharpe Ratios (Predicted Regimes - Trading Reality):")
    for name, result in results.items():
        print(f"  {name.upper():12s}:")
        for regime, sharpe in result.sharpe_by_regime_predicted.items():
            if not pd.isna(sharpe):
                print(f"    {regime:12s}: {sharpe:>7.3f}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE (No Data Leakage)")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")

    print("\nFor thesis:")
    print("  • Use results from validation_fixed/ (not original validation/)")
    print("  • Emphasize methodological rigor (no data leakage)")
    print("  • Compare accuracy on truly out-of-sample data")
    print("  • Use predicted Sharpe ratios (trading reality)")
    print("  • Highlight that label-defining features were excluded")

    print("\nNext steps:")
    print("  1. Compare fixed vs original results (expect lower accuracy)")
    print("  2. Analyze feature importance (non-label features)")
    print("  3. Create figures from deduplicated predictions")
    print("  4. Document methodology in thesis")
    print("=" * 80)


if __name__ == "__main__":
    main()
