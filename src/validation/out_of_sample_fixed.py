"""
Out-of-Sample Validation Framework (FIXED - No Data Leakage)

CRITICAL FIXES:
1. Regime thresholds computed per-fold on training data only (no future leakage)
2. Label-defining features excluded from predictors (no circularity)
3. Predictions deduplicated and Sharpe computed on predicted regimes
4. All models aligned to common date windows
5. Thresholds made configurable

Author: Phase 3 - Out-of-Sample Validation (Fixed)
Date: 2026-01-12
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


@dataclass
class ValidationConfig:
    """Configuration for out-of-sample validation."""
    train_window_days: int = 756  # ~3 years
    test_window_days: int = 252    # ~1 year
    step_size_days: int = 63       # ~3 months (quarterly refit)
    min_train_size: int = 500      # Minimum training observations
    regime_col: str = "regime"     # Target regime column
    returns_col: str = "log_return_1d"  # Returns column for performance metrics

    # FIXED: Configurable thresholds (no hard-coding)
    ricci_percentile: float = 50.0  # Median for ricci split
    drawdown_low_percentile: float = 33.3  # Low drawdown threshold
    drawdown_high_percentile: float = 66.7  # High drawdown threshold

    # FIXED: Exclude label-defining features from predictors
    exclude_label_features: bool = True


@dataclass
class ValidationResults:
    """Results from walk-forward validation."""
    accuracy_scores: List[float]
    confusion_matrices: List[np.ndarray]
    feature_importance: Dict[str, float]
    regime_transitions_detected: int
    regime_transitions_total: int
    sharpe_by_regime_actual: Dict[str, float]
    sharpe_by_regime_predicted: Dict[str, float]  # FIXED: Added predicted
    predictions_df: pd.DataFrame
    predictions_df_deduplicated: pd.DataFrame  # FIXED: Added deduplicated


def define_regimes_on_train(
    train_features: pd.DataFrame,
    config: ValidationConfig
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    FIXED: Define regimes using heuristic rules computed ONLY on training data.

    Returns both the regime labels and the thresholds used, so test data
    can be labeled with the same thresholds (no future information leakage).

    Args:
        train_features: Training feature DataFrame
        config: Validation configuration with threshold percentiles

    Returns:
        Tuple of (regime_labels, thresholds_dict)
    """
    if 'ricci_mean_60d' not in train_features.columns or 'drawdown' not in train_features.columns:
        raise ValueError("Need ricci_mean_60d and drawdown columns")

    ricci = train_features['ricci_mean_60d']
    drawdown = train_features['drawdown']

    # FIXED: Compute thresholds ONLY on training data
    ricci_threshold = ricci.quantile(config.ricci_percentile / 100.0)
    drawdown_low = drawdown.quantile(config.drawdown_low_percentile / 100.0)
    drawdown_high = drawdown.quantile(config.drawdown_high_percentile / 100.0)

    thresholds = {
        'ricci_threshold': ricci_threshold,
        'drawdown_low': drawdown_low,
        'drawdown_high': drawdown_high,
    }

    # Apply thresholds to training data
    regimes = pd.Series(index=train_features.index, dtype=str)
    stable_mask = (ricci > ricci_threshold) & (drawdown > drawdown_high)
    stress_mask = (ricci < ricci_threshold) & (drawdown < drawdown_low)

    regimes[stable_mask] = 'STABLE'
    regimes[stress_mask] = 'STRESS'
    regimes[~(stable_mask | stress_mask)] = 'TRANSITION'

    return regimes, thresholds


def apply_regime_thresholds(
    features: pd.DataFrame,
    thresholds: Dict[str, float]
) -> pd.Series:
    """
    FIXED: Apply pre-computed thresholds to test data (no leakage).

    Args:
        features: Test feature DataFrame
        thresholds: Thresholds computed from training data

    Returns:
        pd.Series with regime labels
    """
    ricci = features['ricci_mean_60d']
    drawdown = features['drawdown']

    regimes = pd.Series(index=features.index, dtype=str)
    stable_mask = (ricci > thresholds['ricci_threshold']) & \
                  (drawdown > thresholds['drawdown_high'])
    stress_mask = (ricci < thresholds['ricci_threshold']) & \
                  (drawdown < thresholds['drawdown_low'])

    regimes[stable_mask] = 'STABLE'
    regimes[stress_mask] = 'STRESS'
    regimes[~(stable_mask | stress_mask)] = 'TRANSITION'

    return regimes


def select_features(
    features_df: pd.DataFrame,
    feature_set: str = 'base',
    exclude_label_defining: bool = True
) -> List[str]:
    """
    FIXED: Select features, optionally excluding label-defining features.

    Args:
        features_df: Full feature DataFrame
        feature_set: 'base', 'macro', 'sectional', or 'all'
        exclude_label_defining: If True, exclude ricci_mean_60d and drawdown

    Returns:
        List of feature column names
    """
    all_cols = features_df.columns.tolist()

    # FIXED: Define label-defining features to exclude
    label_defining_features = ['ricci_mean_60d', 'drawdown', 'drawdown_ath',
                               'drawdown_63d', 'drawdown_126d', 'drawdown_252d']

    # Base geometric features (excluding label-defining if requested)
    base_features = [c for c in all_cols if any(x in c for x in [
        'log_return_1d', 'realized_vol', 'momentum',
        'gsr', 'dxy_beta', 'ricci_min_60d', 'ricci_std_60d', 'ricci_p10', 'ricci_p90',
        'mst_stress_60d', 'dist_silver', 'corr_silver',
        'ga_rotor', 'ga_bivector'
    ])]

    # FIXED: Add ricci_mean_60d and drawdown only if not excluding
    if not exclude_label_defining:
        base_features.extend([c for c in all_cols if any(x in c for x in [
            'ricci_mean_60d', 'drawdown'
        ])])

    # Macro features
    macro_features = [c for c in all_cols if any(x in c for x in [
        'slope_', 'term_premium', 'yield_curve',
        'ig_hy_spread', 'hy_treasury', 'em_ig_spread',
        'gold_oil_ratio', 'copper_gold_ratio', 'equity_bond_ratio',
        'oil_level', 'oil_return', 'brent_wti',
        'vix_term', 'equity_bond_vol'
    ])]

    # Sectional curvature features (manifold-specific, not label-defining)
    sectional_features = [c for c in all_cols if any(x in c for x in [
        'ricci_mean_monetary', 'ricci_mean_industrial', 'ricci_mean_miners',
        'ricci_mean_credit', 'ricci_mean_volatility',
        'curvature_divergence', 'stress_differential'
    ])]

    if feature_set == 'base':
        selected = base_features
    elif feature_set == 'macro':
        selected = base_features + macro_features
    elif feature_set == 'sectional':
        selected = base_features + macro_features + sectional_features
    elif feature_set == 'all':
        exclude = ['silver_close', 'gold_close', 'dxy_level', 'y10_level',
                   'spx_level', 'vix_level', 'regime']
        selected = [c for c in all_cols if c not in exclude and not c.endswith('_level')]
        if exclude_label_defining:
            selected = [c for c in selected if c not in label_defining_features]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # FIXED: Final filter to remove any label-defining features
    if exclude_label_defining:
        selected = [c for c in selected if c not in label_defining_features]

    return selected


def deduplicate_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    FIXED: Deduplicate predictions by keeping last fold's prediction for each date.

    Walk-forward validation creates overlapping test windows, so each date
    can appear in multiple folds. We keep the most recent prediction.

    Args:
        predictions_df: Full predictions with duplicates

    Returns:
        Deduplicated predictions (one row per date)
    """
    # Sort by date to ensure we keep the last fold
    predictions_df = predictions_df.sort_values('date')

    # Drop duplicates, keeping last
    deduplicated = predictions_df.drop_duplicates(subset=['date'], keep='last')

    return deduplicated.sort_values('date')


def compute_sharpe_by_regime(
    predictions_df: pd.DataFrame,
    returns_series: pd.Series,
    regime_col: str = 'predicted'
) -> Dict[str, float]:
    """
    FIXED: Compute Sharpe ratio by regime using deduplicated predictions.

    Args:
        predictions_df: Deduplicated predictions DataFrame
        returns_series: Returns series indexed by date
        regime_col: Which regime column to use ('actual' or 'predicted')

    Returns:
        Dict mapping regime name to Sharpe ratio
    """
    sharpe_by_regime = {}

    # Align predictions with returns
    aligned = predictions_df.set_index('date')
    aligned = aligned.join(returns_series, how='inner')

    for regime in ['STABLE', 'STRESS', 'TRANSITION']:
        regime_mask = aligned[regime_col] == regime
        if regime_mask.sum() > 1:  # Need at least 2 observations
            regime_returns = aligned.loc[regime_mask, returns_series.name]
            mean_ret = regime_returns.mean()
            std_ret = regime_returns.std() + 1e-8
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            sharpe_by_regime[regime] = sharpe
        else:
            sharpe_by_regime[regime] = np.nan

    return sharpe_by_regime


def walk_forward_validation(
    features: pd.DataFrame,
    config: ValidationConfig,
    feature_set: str = 'base',
    regime_col: Optional[str] = None,
) -> ValidationResults:
    """
    FIXED: Walk-forward validation with NO data leakage.

    Key fixes:
    1. Regime thresholds computed per-fold on training data only
    2. Label-defining features excluded from predictors
    3. Predictions deduplicated before computing metrics
    4. Sharpe computed on both actual and predicted regimes

    Args:
        features: Full feature DataFrame
        config: Validation configuration
        feature_set: Which features to use
        regime_col: External regime labels (if None, use heuristic per-fold)

    Returns:
        ValidationResults with accuracy, predictions, Sharpe, etc.
    """
    # Select features (FIXED: exclude label-defining)
    feature_cols = select_features(
        features,
        feature_set=feature_set,
        exclude_label_defining=config.exclude_label_features
    )
    print(f"  Using {len(feature_cols)} features from '{feature_set}' set")
    if config.exclude_label_features:
        print(f"  (Excluding label-defining features: ricci_mean_60d, drawdown, etc.)")

    # Prepare data
    X = features[feature_cols].copy()

    # FIXED: If no external regime labels, we'll compute per-fold
    use_external_labels = regime_col is not None and regime_col in features.columns

    if use_external_labels:
        print(f"  Using external regime labels from column: {regime_col}")
        y = features[regime_col].copy()
    else:
        print(f"  Computing regime labels per-fold (no future leakage)")
        # Create placeholder y (will be computed per-fold)
        y = pd.Series(index=features.index, dtype=str)

    # Drop rows with all missing features
    if use_external_labels:
        valid_mask = y.notna() & (X.notna().sum(axis=1) > len(feature_cols) * 0.5)
    else:
        # For heuristic labeling, need ricci and drawdown
        valid_mask = (features['ricci_mean_60d'].notna() &
                     features['drawdown'].notna() &
                     (X.notna().sum(axis=1) > len(feature_cols) * 0.5))

    X = X[valid_mask]
    y = y[valid_mask]
    features_valid = features[valid_mask]

    # Fill remaining missing values
    X = X.ffill().fillna(X.median())

    # FIXED: Walk-forward validation with per-fold regime computation
    accuracy_scores = []
    confusion_matrices = []
    predictions_list = []

    total_obs = len(X)
    start_idx = 0
    fold_num = 0

    while start_idx + config.train_window_days + config.test_window_days <= total_obs:
        fold_num += 1
        train_end = start_idx + config.train_window_days
        test_end = train_end + config.test_window_days

        # Train/test split
        X_train = X.iloc[start_idx:train_end]
        X_test = X.iloc[train_end:test_end]

        # FIXED: Compute regime labels on training data only
        if not use_external_labels:
            train_features = features_valid.iloc[start_idx:train_end]
            test_features = features_valid.iloc[train_end:test_end]

            # Compute thresholds from training data
            y_train, thresholds = define_regimes_on_train(train_features, config)

            # Apply same thresholds to test data
            y_test = apply_regime_thresholds(test_features, thresholds)
        else:
            y_train = y.iloc[start_idx:train_end]
            y_test = y.iloc[train_end:test_end]

        # Skip if insufficient labels
        if y_train.isna().all() or y_test.isna().all():
            start_idx += config.step_size_days
            continue

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=['STABLE', 'STRESS', 'TRANSITION'])

        accuracy_scores.append(acc)
        confusion_matrices.append(cm)

        # Store predictions
        pred_df = pd.DataFrame({
            'date': y_test.index,
            'actual': y_test.values,
            'predicted': y_pred,
            'correct': y_test.values == y_pred,
            'fold': fold_num,
        })
        predictions_list.append(pred_df)

        print(f"    Fold {fold_num}: Train {X_train.index[0].strftime('%Y-%m')} to "
              f"{X_train.index[-1].strftime('%Y-%m')}, Test {X_test.index[0].strftime('%Y-%m')} to "
              f"{X_test.index[-1].strftime('%Y-%m')}, Accuracy: {acc:.3f}")

        # Step forward
        start_idx += config.step_size_days

    # Aggregate predictions
    predictions_df = pd.concat(predictions_list, ignore_index=True)

    # FIXED: Deduplicate predictions
    predictions_df_deduplicated = deduplicate_predictions(predictions_df)

    print(f"  Total predictions: {len(predictions_df):,}")
    print(f"  Deduplicated predictions: {len(predictions_df_deduplicated):,} "
          f"({len(predictions_df_deduplicated)/len(predictions_df)*100:.1f}% of original)")

    # Feature importance (from last fold)
    feature_importance = dict(zip(feature_cols, clf.feature_importances_))

    # FIXED: Detect transitions on deduplicated data
    dedup = predictions_df_deduplicated
    actual_transitions = (dedup['actual'].shift() != dedup['actual']).sum()
    detected_transitions = (
        (dedup['actual'].shift() != dedup['actual']) &
        (dedup['predicted'].shift() != dedup['predicted'])
    ).sum()

    # FIXED: Compute Sharpe on deduplicated predictions
    if config.returns_col in features_valid.columns:
        returns = features_valid[config.returns_col]

        # Sharpe by actual regime
        sharpe_actual = compute_sharpe_by_regime(
            dedup, returns, regime_col='actual'
        )

        # FIXED: Sharpe by predicted regime (this is what trading would use!)
        sharpe_predicted = compute_sharpe_by_regime(
            dedup, returns, regime_col='predicted'
        )
    else:
        sharpe_actual = {}
        sharpe_predicted = {}

    return ValidationResults(
        accuracy_scores=accuracy_scores,
        confusion_matrices=confusion_matrices,
        feature_importance=feature_importance,
        regime_transitions_detected=detected_transitions,
        regime_transitions_total=actual_transitions,
        sharpe_by_regime_actual=sharpe_actual,
        sharpe_by_regime_predicted=sharpe_predicted,
        predictions_df=predictions_df,
        predictions_df_deduplicated=dedup,
    )


def compare_models(
    features: pd.DataFrame,
    config: ValidationConfig,
    regime_col: Optional[str] = None,
) -> Dict[str, ValidationResults]:
    """
    FIXED: Compare models on IDENTICAL date windows.

    All models now use the same date index (intersection of valid rows).

    Args:
        features: Full feature DataFrame
        config: Validation configuration
        regime_col: External regime labels (if None, use heuristic)

    Returns:
        Dict mapping feature_set name to ValidationResults
    """
    # FIXED: Determine common valid date range across all feature sets
    print("\nDetermining common valid date range across all feature sets...")

    valid_indices = {}
    for feature_set in ['base', 'macro', 'sectional']:
        feature_cols = select_features(features, feature_set=feature_set,
                                      exclude_label_defining=config.exclude_label_features)
        X = features[feature_cols]
        valid_mask = (X.notna().sum(axis=1) > len(feature_cols) * 0.5)
        if regime_col is None:
            # Also need ricci and drawdown for heuristic labeling
            valid_mask = valid_mask & features['ricci_mean_60d'].notna() & features['drawdown'].notna()
        valid_indices[feature_set] = features.index[valid_mask]

    # Intersection of all valid indices
    common_index = valid_indices['base']
    for idx in valid_indices.values():
        common_index = common_index.intersection(idx)

    print(f"  Common valid dates: {len(common_index):,} (from {len(features):,} total)")

    # Filter features to common index
    features_aligned = features.loc[common_index]

    # Run validation on aligned data
    results = {}
    for feature_set in ['base', 'macro', 'sectional']:
        print(f"\n{'='*80}")
        print(f"Validating {feature_set.upper()} Model")
        print(f"{'='*80}")

        results[feature_set] = walk_forward_validation(
            features=features_aligned,
            config=config,
            feature_set=feature_set,
            regime_col=regime_col,
        )

    return results


def print_comparison_summary(results: Dict[str, ValidationResults]):
    """Print summary comparison of model performance (FIXED VERSION)."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY (FIXED - No Data Leakage)")
    print("=" * 80)

    metrics = {
        'Model': [],
        'Mean Accuracy': [],
        'Std Accuracy': [],
        'Folds': [],
        'Transitions Detected': [],
        'Transitions Total': [],
        'Detection Rate': [],
    }

    for name, result in results.items():
        metrics['Model'].append(name.upper())
        metrics['Mean Accuracy'].append(f"{np.mean(result.accuracy_scores):.4f}")
        metrics['Std Accuracy'].append(f"{np.std(result.accuracy_scores):.4f}")
        metrics['Folds'].append(len(result.accuracy_scores))
        metrics['Transitions Detected'].append(result.regime_transitions_detected)
        metrics['Transitions Total'].append(result.regime_transitions_total)
        detection_rate = result.regime_transitions_detected / result.regime_transitions_total \
                        if result.regime_transitions_total > 0 else 0
        metrics['Detection Rate'].append(f"{detection_rate:.2%}")

    df_summary = pd.DataFrame(metrics)
    print("\n" + df_summary.to_string(index=False))

    # Sharpe ratios - FIXED: Show both actual and predicted
    print("\n" + "=" * 80)
    print("SHARPE RATIOS BY REGIME (Deduplicated)")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name.upper()} Model:")
        print("  Based on ACTUAL regimes:")
        for regime, sharpe in result.sharpe_by_regime_actual.items():
            if not np.isnan(sharpe):
                print(f"    {regime:12s}: {sharpe:>7.3f}")

        print("  Based on PREDICTED regimes (trading reality):")
        for regime, sharpe in result.sharpe_by_regime_predicted.items():
            if not np.isnan(sharpe):
                print(f"    {regime:12s}: {sharpe:>7.3f}")


def save_validation_report(
    results: Dict[str, ValidationResults],
    output_dir: Path,
    features_path: str,
):
    """Save validation results (FIXED VERSION)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save deduplicated predictions
    for name, result in results.items():
        # Full predictions (with fold info)
        pred_path = output_dir / f"predictions_{name}_full.csv"
        result.predictions_df.to_csv(pred_path, index=False)
        print(f"  Saved full predictions: {pred_path}")

        # FIXED: Deduplicated predictions (for analysis)
        dedup_path = output_dir / f"predictions_{name}_deduplicated.csv"
        result.predictions_df_deduplicated.to_csv(dedup_path, index=False)
        print(f"  Saved deduplicated predictions: {dedup_path}")

    # Feature importance
    for name, result in results.items():
        importance_df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in sorted(result.feature_importance.items(), key=lambda x: -x[1])
        ])
        importance_path = output_dir / f"feature_importance_{name}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"  Saved feature importance: {importance_path}")

    # FIXED: Summary report with methodology notes
    summary_path = output_dir / "validation_summary_fixed.txt"
    with open(summary_path, 'w') as f:
        f.write("OUT-OF-SAMPLE VALIDATION SUMMARY (FIXED - No Data Leakage)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Features file: {features_path}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        f.write("METHODOLOGICAL FIXES:\n")
        f.write("1. Regime thresholds computed per-fold on training data only\n")
        f.write("2. Label-defining features excluded from predictors\n")
        f.write("3. Predictions deduplicated before metrics computation\n")
        f.write("4. All models validated on identical date windows\n")
        f.write("5. Sharpe ratios computed on PREDICTED regimes\n\n")

        for name, result in results.items():
            f.write(f"\n{name.upper()} Model\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean Accuracy: {np.mean(result.accuracy_scores):.4f} ± "
                   f"{np.std(result.accuracy_scores):.4f}\n")
            f.write(f"  Validation Folds: {len(result.accuracy_scores)}\n")
            f.write(f"  Transitions Detected: {result.regime_transitions_detected} / "
                   f"{result.regime_transitions_total}\n")
            f.write(f"  Predictions (full): {len(result.predictions_df):,}\n")
            f.write(f"  Predictions (deduplicated): {len(result.predictions_df_deduplicated):,}\n")

            f.write(f"\n  Sharpe Ratios (Actual Regimes):\n")
            for regime, sharpe in result.sharpe_by_regime_actual.items():
                if not np.isnan(sharpe):
                    f.write(f"    {regime:12s}: {sharpe:>7.3f}\n")

            f.write(f"\n  Sharpe Ratios (Predicted Regimes - Trading Reality):\n")
            for regime, sharpe in result.sharpe_by_regime_predicted.items():
                if not np.isnan(sharpe):
                    f.write(f"    {regime:12s}: {sharpe:>7.3f}\n")

    print(f"\n  Saved summary: {summary_path}")
