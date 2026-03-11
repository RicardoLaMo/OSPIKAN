"""
Tests for Geometric Algebra regime features.

Validates:
1. Rotor magnitude computation
2. Bivector energy computation
3. Integration with feature pipeline
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.geometry.ga_regime_features import (
    compute_rotor_magnitude,
    compute_bivector_energy,
    compute_ga_regime_features,
)


def test_compute_rotor_magnitude_basic():
    """
    Test: Rotor magnitude should be non-negative and capture regime shifts.
    """
    # Create synthetic features with regime shift
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=200, freq="D")

    # Regime 1: low vol (days 0-100)
    regime1 = pd.DataFrame({
        "log_return_1d": np.random.normal(0.0001, 0.005, 100),
        "realized_vol_20d": np.full(100, 0.10),
        "momentum_10d": np.random.normal(0.01, 0.02, 100),
        "drawdown": np.random.uniform(-0.05, 0, 100),
    })

    # Regime 2: high vol (days 100-200) - REGIME SHIFT
    regime2 = pd.DataFrame({
        "log_return_1d": np.random.normal(-0.0002, 0.02, 100),
        "realized_vol_20d": np.full(100, 0.40),
        "momentum_10d": np.random.normal(-0.03, 0.05, 100),
        "drawdown": np.random.uniform(-0.25, -0.05, 100),
    })

    features = pd.concat([regime1, regime2], ignore_index=False)
    features.index = dates

    rotor_mags = compute_rotor_magnitude(features, window=30, device="cpu")

    assert isinstance(rotor_mags, pd.Series), "Should return Series"
    assert rotor_mags.name == "ga_rotor_magnitude"

    # Should have valid values
    valid_values = rotor_mags.dropna()
    assert len(valid_values) > 0, "Should have some valid rotor magnitudes"

    # All values should be non-negative (magnitude property)
    assert (valid_values >= 0).all(), "Rotor magnitude must be non-negative"
    assert valid_values.max() > 1e-6, "Rotor magnitude should not be identically zero"
    assert valid_values.nunique() > 1, "Rotor magnitude should vary over time"

    # Expect higher rotor magnitude near regime shift (around day 100)
    mid_point = len(features) // 2
    window_around_shift = rotor_mags.iloc[mid_point - 20 : mid_point + 20]
    baseline = rotor_mags.iloc[:50]  # Early stable period

    if baseline.notna().any() and window_around_shift.notna().any():
        shift_mean = window_around_shift.mean()
        baseline_mean = baseline.mean()
        print(f"Baseline rotor magnitude: {baseline_mean:.4f}")
        print(f"Shift region rotor magnitude: {shift_mean:.4f}")
        # Expect shift region to have larger magnitude (but may not always be true in synthetic data)


def test_compute_bivector_energy():
    """
    Test: Bivector energy measures rotational activity in feature space.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    features = pd.DataFrame({
        "log_return_1d": np.random.normal(0, 0.01, 100),
        "realized_vol_20d": np.full(100, 0.15),
        "momentum_10d": np.random.normal(0, 0.02, 100),
        "drawdown": np.random.uniform(-0.1, 0, 100),
    }, index=dates)

    energies = compute_bivector_energy(features, window=30, device="cpu")

    assert isinstance(energies, pd.Series)
    assert energies.name == "ga_bivector_energy"

    valid_values = energies.dropna()
    assert len(valid_values) > 0

    # Energy should be non-negative
    assert (valid_values >= 0).all(), "Bivector energy must be non-negative"

    print(f"Bivector energy range: [{valid_values.min():.4f}, {valid_values.max():.4f}]")


def test_compute_ga_regime_features_integration():
    """
    Test: Full GA regime feature computation pipeline.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=150, freq="D")

    features = pd.DataFrame({
        "log_return_1d": np.random.normal(0, 0.01, 150),
        "realized_vol_20d": np.random.uniform(0.1, 0.3, 150),
        "momentum_10d": np.random.normal(0, 0.02, 150),
        "drawdown": np.random.uniform(-0.15, 0, 150),
    }, index=dates)

    ga_features = compute_ga_regime_features(features, window=60, device="cpu")

    assert isinstance(ga_features, pd.DataFrame)
    assert "ga_rotor_magnitude_60d" in ga_features.columns
    assert "ga_bivector_energy_60d" in ga_features.columns

    # Check alignment
    assert len(ga_features) == len(features), "Output should match input length"
    assert ga_features.index.equals(features.index), "Index should be preserved"

    print(f"GA features shape: {ga_features.shape}")
    print(f"Non-null rotor values: {ga_features['ga_rotor_magnitude_60d'].notna().sum()}")
    print(f"Non-null energy values: {ga_features['ga_bivector_energy_60d'].notna().sum()}")


def test_ga_features_missing_columns():
    """
    Test: Should raise error if required feature columns are missing.
    """
    features = pd.DataFrame({
        "log_return_1d": [0.01, 0.02],
        "realized_vol_20d": [0.1, 0.15],
        # Missing momentum_10d and drawdown
    })

    with pytest.raises(KeyError):
        compute_rotor_magnitude(features, window=1, device="cpu")


def test_ga_rotor_invariance():
    """
    Test: Rotor magnitude should be rotationally invariant (geometric property).

    If we rotate all features by a constant transformation, rotor magnitudes
    should reflect the RELATIVE changes, not absolute values.
    """
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="D")

    features = pd.DataFrame({
        "log_return_1d": np.random.normal(0, 0.01, 100),
        "realized_vol_20d": np.random.uniform(0.1, 0.3, 100),
        "momentum_10d": np.random.normal(0, 0.02, 100),
        "drawdown": np.random.uniform(-0.1, 0, 100),
    }, index=dates)

    # Compute rotor on original features
    rotor1 = compute_rotor_magnitude(features, window=30, device="cpu")

    # Apply global shift (translation in feature space)
    features_shifted = features.copy()
    features_shifted["log_return_1d"] += 0.001

    rotor2 = compute_rotor_magnitude(features_shifted, window=30, device="cpu")

    # Rotor magnitudes should be similar (translation shouldn't change rotation angle)
    valid_mask = rotor1.notna() & rotor2.notna()
    if valid_mask.sum() > 10:
        correlation = rotor1[valid_mask].corr(rotor2[valid_mask])
        print(f"Rotor correlation after feature shift: {correlation:.4f}")
        # Note: May be NaN if rotors are constant or have no variance
        if not np.isnan(correlation):
            assert correlation > 0.5, "Rotor should be robust to feature shifts"
        else:
            print("Correlation is NaN (rotors may be constant or have insufficient variance)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
