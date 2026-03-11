"""
Out-of-Sample Validation Framework (Phase 3)

Walk-forward validation for regime detection models.
Compares base (geometry only) vs enhanced (geometry + macro + sectional curvature) models.

Key metrics:
- Regime classification accuracy
- Transition timing precision (lead/lag)
- Sharpe ratio improvement
- Drawdown prediction accuracy

Author: Phase 3 - Out-of-Sample Validation
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


@dataclass
class ValidationResults:
    """Results from walk-forward validation."""
    accuracy_scores: List[float]
    confusion_matrices: List[np.ndarray]
    feature_importance: Dict[str, float]
    regime_transitions_detected: int
    regime_transitions_total: int
    sharpe_by_regime: Dict[str, float]
    predictions_df: pd.DataFrame


def define_regimes_heuristic(features: pd.DataFrame) -> pd.Series:
    """
    Define regimes using heuristic rules (for baseline when no existing labels).

    Uses Ricci curvature and drawdown to classify:
    - STABLE: Positive curvature, low drawdown
    - STRESS: Negative curvature, high drawdown
    - TRANSITION: Mixed signals or recent regime change

    Args:
        features: Feature DataFrame with ricci_mean_60d and drawdown

    Returns:
        pd.Series with regime labels
    """
    regimes = pd.Series(index=features.index, dtype=str)

    if 'ricci_mean_60d' not in features.columns or 'drawdown' not in features.columns:
        raise ValueError("Need ricci_mean_60d and drawdown columns for heuristic regime definition")

    ricci = features['ricci_mean_60d']
    drawdown = features['drawdown']

    # Define thresholds (data-driven)
    ricci_median = ricci.median()
    drawdown_q33 = drawdown.quantile(0.33)
    drawdown_q67 = drawdown.quantile(0.67)

    # Classify
    stable_mask = (ricci > ricci_median) & (drawdown > drawdown_q67)
    stress_mask = (ricci < ricci_median) & (drawdown < drawdown_q33)

    regimes[stable_mask] = 'STABLE'
    regimes[stress_mask] = 'STRESS'
    regimes[~(stable_mask | stress_mask)] = 'TRANSITION'

    return regimes


def select_features(features_df: pd.DataFrame, feature_set: str = 'base') -> List[str]:
    """
    Select feature columns based on feature set type.

    Args:
        features_df: Full feature DataFrame
        feature_set: 'base', 'macro', 'sectional', or 'all'

    Returns:
        List of feature column names
    """
    all_cols = features_df.columns.tolist()

    # Base geometric features
    base_features = [c for c in all_cols if any(x in c for x in [
        'log_return_1d', 'realized_vol', 'momentum', 'drawdown',
        'gsr', 'dxy_beta', 'ricci_mean_60d', 'ricci_min_60d', 'ricci_std_60d',
        'mst_stress_60d', 'dist_silver', 'corr_silver',
        'ga_rotor', 'ga_bivector'
    ])]

    # Macro features (Phase 1)
    macro_features = [c for c in all_cols if any(x in c for x in [
        'slope_', 'term_premium', 'yield_curve',
        'ig_hy_spread', 'hy_treasury', 'em_ig_spread',
        'gold_oil_ratio', 'copper_gold_ratio', 'equity_bond_ratio',
        'oil_level', 'oil_return', 'brent_wti',
        'vix_term', 'equity_bond_vol'
    ])]

    # Sectional curvature features (Phase 2)
    sectional_features = [c for c in all_cols if any(x in c for x in [
        'ricci_mean_monetary', 'ricci_mean_industrial', 'ricci_mean_miners',
        'ricci_mean_credit', 'ricci_mean_volatility',
        'curvature_divergence', 'stress_differential'
    ])]

    if feature_set == 'base':
        return base_features
    elif feature_set == 'macro':
        return base_features + macro_features
    elif feature_set == 'sectional':
        return base_features + macro_features + sectional_features
    elif feature_set == 'all':
        # Exclude non-feature columns
        exclude = ['silver_close', 'gold_close', 'dxy_level', 'y10_level',
                   'spx_level', 'vix_level', 'regime']
        return [c for c in all_cols if c not in exclude and not c.endswith('_level')]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


def walk_forward_validation(
    features: pd.DataFrame,
    config: ValidationConfig,
    feature_set: str = 'base',
    regime_col: Optional[str] = None,
) -> ValidationResults:
    """
    Walk-forward validation with expanding window.

    Trains model on historical data, tests on future period, then rolls forward.

    Args:
        features: Full feature DataFrame with regime labels
        config: Validation configuration
        feature_set: Which features to use ('base', 'macro', 'sectional', 'all')
        regime_col: Column name with regime labels (if None, use heuristic)

    Returns:
        ValidationResults with accuracy, confusion matrices, predictions
    """
    # Define regimes if not provided
    if regime_col is None or regime_col not in features.columns:
        print(f"  Defining regimes using heuristic (no {regime_col} column found)...")
        features = features.copy()
        features['regime'] = define_regimes_heuristic(features)
        regime_col = 'regime'

    # Select features
    feature_cols = select_features(features, feature_set=feature_set)
    print(f"  Using {len(feature_cols)} features from '{feature_set}' set")

    # Prepare data
    X = features[feature_cols].copy()
    y = features[regime_col].copy()

    # Drop rows with missing target or all missing features
    valid_mask = y.notna() & (X.notna().sum(axis=1) > len(feature_cols) * 0.5)
    X = X[valid_mask]
    y = y[valid_mask]

    # Fill remaining missing values with forward fill then median
    X = X.ffill().fillna(X.median())

    # Walk-forward validation
    accuracy_scores = []
    confusion_matrices = []
    predictions_list = []

    total_obs = len(X)
    start_idx = 0

    while start_idx + config.train_window_days + config.test_window_days <= total_obs:
        train_end = start_idx + config.train_window_days
        test_end = train_end + config.test_window_days

        # Train/test split
        X_train = X.iloc[start_idx:train_end]
        y_train = y.iloc[start_idx:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model (Random Forest for regime classification)
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
        })
        predictions_list.append(pred_df)

        # Step forward
        start_idx += config.step_size_days

        print(f"    Fold {len(accuracy_scores)}: Train {X_train.index[0].strftime('%Y-%m')} to {X_train.index[-1].strftime('%Y-%m')}, "
              f"Test {X_test.index[0].strftime('%Y-%m')} to {X_test.index[-1].strftime('%Y-%m')}, "
              f"Accuracy: {acc:.3f}")

    # Aggregate predictions
    predictions_df = pd.concat(predictions_list, ignore_index=False)

    # Compute feature importance (from last fold)
    feature_importance = dict(zip(feature_cols, clf.feature_importances_))

    # Detect regime transitions (work with categorical labels)
    actual_transitions = (predictions_df['actual'].shift() != predictions_df['actual']).sum()
    detected_transitions = (
        (predictions_df['actual'].shift() != predictions_df['actual']) &
        (predictions_df['predicted'].shift() != predictions_df['predicted'])
    ).sum()

    # Compute Sharpe by regime (if returns available)
    sharpe_by_regime = {}
    if config.returns_col in features.columns:
        for regime in ['STABLE', 'STRESS', 'TRANSITION']:
            regime_mask = predictions_df['actual'] == regime
            if regime_mask.sum() > 0:
                regime_returns = features.loc[predictions_df[regime_mask]['date'], config.returns_col]
                if len(regime_returns) > 1:
                    sharpe = regime_returns.mean() / (regime_returns.std() + 1e-8) * np.sqrt(252)
                    sharpe_by_regime[regime] = sharpe

    return ValidationResults(
        accuracy_scores=accuracy_scores,
        confusion_matrices=confusion_matrices,
        feature_importance=feature_importance,
        regime_transitions_detected=detected_transitions,
        regime_transitions_total=actual_transitions,
        sharpe_by_regime=sharpe_by_regime,
        predictions_df=predictions_df,
    )


def compare_models(
    features: pd.DataFrame,
    config: ValidationConfig,
    regime_col: Optional[str] = None,
) -> Dict[str, ValidationResults]:
    """
    Compare base vs macro vs sectional feature sets.

    Args:
        features: Full feature DataFrame
        config: Validation configuration
        regime_col: Regime label column (if None, use heuristic)

    Returns:
        Dict mapping feature_set name to ValidationResults
    """
    results = {}

    for feature_set in ['base', 'macro', 'sectional']:
        print(f"\n{'='*80}")
        print(f"Validating {feature_set.upper()} Model")
        print(f"{'='*80}")

        results[feature_set] = walk_forward_validation(
            features=features,
            config=config,
            feature_set=feature_set,
            regime_col=regime_col,
        )

    return results


def print_comparison_summary(results: Dict[str, ValidationResults]):
    """Print summary comparison of model performance."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
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
        detection_rate = result.regime_transitions_detected / result.regime_transitions_total if result.regime_transitions_total > 0 else 0
        metrics['Detection Rate'].append(f"{detection_rate:.2%}")

    df_summary = pd.DataFrame(metrics)
    print("\n" + df_summary.to_string(index=False))

    # Sharpe ratios
    print("\n" + "=" * 80)
    print("SHARPE RATIOS BY REGIME")
    print("=" * 80)

    for name, result in results.items():
        print(f"\n{name.upper()} Model:")
        for regime, sharpe in result.sharpe_by_regime.items():
            print(f"  {regime:12s}: {sharpe:>7.3f}")


def save_validation_report(
    results: Dict[str, ValidationResults],
    output_dir: Path,
    features_path: str,
):
    """Save validation results to CSV and generate summary report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions for each model
    for name, result in results.items():
        pred_path = output_dir / f"predictions_{name}.csv"
        result.predictions_df.to_csv(pred_path, index=False)
        print(f"  Saved predictions: {pred_path}")

    # Save feature importance
    for name, result in results.items():
        importance_df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in sorted(result.feature_importance.items(), key=lambda x: -x[1])
        ])
        importance_path = output_dir / f"feature_importance_{name}.csv"
        importance_df.to_csv(importance_path, index=False)
        print(f"  Saved feature importance: {importance_path}")

    # Summary report
    summary_path = output_dir / "validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("OUT-OF-SAMPLE VALIDATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Features file: {features_path}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        for name, result in results.items():
            f.write(f"\n{name.upper()} Model\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Mean Accuracy: {np.mean(result.accuracy_scores):.4f} ± {np.std(result.accuracy_scores):.4f}\n")
            f.write(f"  Validation Folds: {len(result.accuracy_scores)}\n")
            f.write(f"  Transitions Detected: {result.regime_transitions_detected} / {result.regime_transitions_total}\n")
            f.write(f"\n  Sharpe Ratios by Regime:\n")
            for regime, sharpe in result.sharpe_by_regime.items():
                f.write(f"    {regime:12s}: {sharpe:>7.3f}\n")

    print(f"\n  Saved summary: {summary_path}")
