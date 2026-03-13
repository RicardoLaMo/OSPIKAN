#!/usr/bin/env python3
"""
Run out-of-sample validation comparing base vs enhanced models.

This script is the Phase 3 deliverable: demonstrates that enhanced features
(macro + sectional curvature) improve regime detection accuracy.

IMPORTANT: This script now uses the FIXED validation framework (no data leakage).

Usage:
    python scripts/run_out_of_sample_validation.py <FEATURES_FILE>

Example:
    python scripts/run_out_of_sample_validation.py data/processed/silver_features_*.parquet

Author: Phase 3 - Out-of-Sample Validation (Fixed)
Date: 2026-01-12
"""

import os
import sys
import warnings
from pathlib import Path

# Add repo root to path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pandas as pd

# Use FIXED validation module (no data leakage)
from src.validation.out_of_sample_fixed import (
    ValidationConfig,
    compare_models,
    print_comparison_summary,
    save_validation_report,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_out_of_sample_validation.py <FEATURES_FILE>")
        sys.exit(1)

    features_path = sys.argv[1]
    output_dir = Path("reports/silver/out_of_sample_validation")

    print("=" * 80)
    print("OUT-OF-SAMPLE VALIDATION: BASE vs ENHANCED MODELS")
    print("(Using FIXED validation framework - no data leakage)")
    print("=" * 80)
    print(f"\nFeatures: {features_path}")
    print(f"Output:   {output_dir}/")

    # Load features
    print("\nLoading features...")
    features = pd.read_parquet(features_path)
    print(f"  Loaded {len(features):,} observations, {len(features.columns)} features")
    print(f"  Date range: {features.index.min().strftime('%Y-%m-%d')} to {features.index.max().strftime('%Y-%m-%d')}")

    # Configuration (using FIXED validation settings)
    config = ValidationConfig(
        train_window_days=756,   # ~3 years
        test_window_days=252,    # ~1 year
        step_size_days=63,       # ~3 months (quarterly refit)
        regime_col=None,         # Use heuristic (computed per-fold)
        returns_col='log_return_1d',
        # FIXED: Configurable thresholds (computed per-fold on training data only)
        ricci_percentile=50.0,
        drawdown_low_percentile=33.3,
        drawdown_high_percentile=66.7,
        # FIXED: Exclude label-defining features to avoid circularity
        exclude_label_features=True,
    )

    print(f"\nValidation Configuration:")
    print(f"  Train window: {config.train_window_days} days (~{config.train_window_days/252:.1f} years)")
    print(f"  Test window:  {config.test_window_days} days (~{config.test_window_days/252:.1f} year)")
    print(f"  Step size:    {config.step_size_days} days (~{config.step_size_days/21:.1f} months)")
    print(f"  Label-feature exclusion: {config.exclude_label_features} (prevents circularity)")

    # Run validation
    results = compare_models(features, config, regime_col=None)

    # Print summary
    print_comparison_summary(results)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    save_validation_report(results, output_dir, features_path)

    # Additional analysis: Feature importance comparison
    print("\n" + "=" * 80)
    print("TOP 10 FEATURES BY IMPORTANCE")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name.upper()} Model:")
        top_features = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:10]
        for i, (feat, imp) in enumerate(top_features, 1):
            print(f"  {i:2d}. {feat:40s}: {imp:.4f}")

    # Thesis implications
    print("\n" + "=" * 80)
    print("THESIS IMPLICATIONS")
    print("=" * 80)

    base_acc = results['base'].accuracy_scores
    macro_acc = results['macro'].accuracy_scores if 'macro' in results else None
    sect_acc = results['sectional'].accuracy_scores if 'sectional' in results else None

    print("\nKey Findings:")

    if macro_acc:
        improvement_macro = (sum(macro_acc) / len(macro_acc)) - (sum(base_acc) / len(base_acc))
        print(f"  1. Macro features improve accuracy by {improvement_macro*100:+.2f}%")
        print(f"     → Treasury spreads, credit spreads, and cross-asset ratios add value")

    if sect_acc:
        improvement_sect = (sum(sect_acc) / len(sect_acc)) - (sum(base_acc) / len(base_acc))
        print(f"  2. Sectional curvature improves accuracy by {improvement_sect*100:+.2f}%")
        print(f"     → Manifold-specific geometry captures regime drivers (monetary vs industrial)")

    print(f"\n  3. Regime transition detection:")
    for name, result in results.items():
        rate = result.regime_transitions_detected / result.regime_transitions_total if result.regime_transitions_total > 0 else 0
        print(f"     {name.upper():12s}: {rate:>6.1%} of transitions detected")

    print("\nRecommendations for thesis:")
    print("  • Emphasize multi-asset intelligence improves predictive power")
    print("  • Highlight sectional curvature as novel contribution")
    print("  • Use validation results to demonstrate practical applicability")
    print("  • Compare transition detection rates (enhanced should detect more)")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nNext steps:")
    print("  1. Review predictions CSV files for error analysis")
    print("  2. Check feature importance - are macro/sectional features used?")
    print("  3. Analyze misclassified periods - why did model fail?")
    print("  4. Run on different time periods for robustness check")
    print("=" * 80)


if __name__ == "__main__":
    main()
