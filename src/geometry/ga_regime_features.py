"""
Geometric Algebra regime features using Clifford algebra rotors.

This module integrates the TensorGA engine (Cl(4,0)) into the feature pipeline
to detect regime shifts as rotations in feature space.

Regime transitions are modeled as rotors (geometric rotations) rather than
simple statistical changes, capturing the differential geometric structure
of market state evolution.
"""
from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import pandas as pd
import torch

from src.geometry.tensor_ga import TensorGA


def compute_rotor_magnitude(
    features: pd.DataFrame,
    *,
    window: int = 60,
    feature_cols: Optional[list[str]] = None,
    device: str = "cpu",
) -> pd.Series:
    """
    Computes a rolling rotor magnitude between consecutive *state vectors*.

    We define a 4D "state vector" at each date as the feature vector z-scored
    over the trailing `window`. We then compute the rotor that maps the state
    vector at t-1 to the state vector at t, and return the bivector magnitude
    (rotation strength proxy).

    This is causal (no look-ahead): only trailing data is used to build each
    state vector.

    Args:
        features: DataFrame with time-indexed feature columns
        window: Rolling window size for z-scoring the state (e.g., 60 days)
        feature_cols: List of 4 feature columns to use (must be exactly 4)
                     If None, uses: ['log_return_1d', 'realized_vol_20d',
                                     'momentum_10d', 'drawdown']
        device: 'cpu' or 'cuda'

    Returns:
        Series of rotor magnitudes (larger = bigger regime shift)
    """
    if feature_cols is None:
        # Default: 4D feature space for Cl(4,0)
        feature_cols = ["log_return_1d", "realized_vol_20d", "momentum_10d", "drawdown"]

    if len(feature_cols) != 4:
        raise ValueError(f"GA Cl(4,0) requires exactly 4 features, got {len(feature_cols)}")

    # Check columns exist
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    df = features[feature_cols].copy()
    df = df.dropna()

    # Need at least `window` points to compute the first z-scored state, plus one
    # additional point to compute a t-1 -> t rotor.
    if len(df) < window + 1:
        return pd.Series(index=features.index, dtype=float, name="ga_rotor_magnitude")

    # Rolling z-score (causal state).
    mean = df.rolling(window=window, min_periods=window).mean()
    std = df.rolling(window=window, min_periods=window).std(ddof=0) + 1e-8
    z = (df - mean) / std
    z = z.dropna()

    if len(z) < 2:
        return pd.Series(index=features.index, dtype=float, name="ga_rotor_magnitude")

    # Initialize GA engine.
    ga = TensorGA(n_features=4, device=device)

    data_t = torch.tensor(z.values, dtype=torch.float32, device=device)
    mvs = ga.embed_data(data_t)

    # Estimate rotors between consecutive dates (batch).
    rotors = ga.estimate_rotor(mvs[:-1], mvs[1:])

    # Rotor magnitude = norm of bivector part (rotation strength proxy).
    bivector = ga.extract_grade(rotors, grade=2)
    mag_sq = ga.magnitude_sq(bivector)
    mag = torch.sqrt(torch.clamp(mag_sq, min=0.0)).squeeze(1).detach().cpu().numpy()

    rotor_mags = pd.Series(mag, index=z.index[1:], name="ga_rotor_magnitude", dtype=float)
    return rotor_mags


def compute_bivector_energy(
    features: pd.DataFrame,
    *,
    window: int = 60,
    feature_cols: Optional[list[str]] = None,
    device: str = "cpu",
) -> pd.Series:
    """
    Computes the bivector energy (rotation plane magnitude) over rolling windows.

    This measures the "rotational content" of the feature space dynamics,
    capturing regime transitions as changes in the dominant rotation plane.

    High bivector energy = market is in active rotation/transition state.
    Low bivector energy = market is stable (no rotation).

    Args:
        features: DataFrame with time-indexed feature columns
        window: Rolling window size
        feature_cols: List of 4 feature columns (default same as compute_rotor_magnitude)
        device: 'cpu' or 'cuda'

    Returns:
        Series of bivector energies
    """
    if feature_cols is None:
        feature_cols = ["log_return_1d", "realized_vol_20d", "momentum_10d", "drawdown"]

    if len(feature_cols) != 4:
        raise ValueError(f"GA Cl(4,0) requires exactly 4 features, got {len(feature_cols)}")

    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    df = features[feature_cols].copy()
    df = df.dropna()

    if len(df) < window:
        return pd.Series(index=features.index, dtype=float, name="ga_bivector_energy")

    ga = TensorGA(n_features=4, device=device)
    energies = pd.Series(index=df.index, dtype=float, name="ga_bivector_energy")

    for i in range(window - 1, len(df)):
        window_data = df.iloc[i - window + 1 : i + 1].values
        mean = np.mean(window_data, axis=0)
        std = np.std(window_data, axis=0) + 1e-8

        # Normalize
        normalized = (window_data - mean) / std

        # Embed into multivectors
        data_t = torch.tensor(normalized, dtype=torch.float32, device=device)
        mvs = ga.embed_data(data_t)

        # Compute wedge products between consecutive days (bivector dynamics)
        bivector_sum = torch.zeros(1, ga.dim, device=device)
        for j in range(len(mvs) - 1):
            bv = ga.wedge_product(mvs[j : j + 1], mvs[j + 1 : j + 2])
            bivector_sum += bv

        # Energy = magnitude of accumulated bivector
        energy_sq = ga.magnitude_sq(bivector_sum)
        energy = float(torch.sqrt(energy_sq).cpu().item())
        energies.iloc[i] = energy

    return energies


def compute_ga_regime_features(
    features: pd.DataFrame,
    *,
    window: int = 60,
    feature_cols: Optional[list[str]] = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Computes all GA-based regime features for integration into pipeline.

    Returns a DataFrame with columns:
        - ga_rotor_magnitude: Rotation magnitude between windows (regime shift size)
        - ga_bivector_energy: Rotational content within window (transition activity)

    Args:
        features: Input feature DataFrame
        window: Rolling window size (default 60)
        feature_cols: 4 feature columns for Cl(4,0) embedding
        device: 'cpu' or 'cuda'

    Returns:
        DataFrame with GA regime features aligned to input index
    """
    out = pd.DataFrame(index=features.index)

    try:
        out["ga_rotor_magnitude_60d"] = compute_rotor_magnitude(
            features, window=window, feature_cols=feature_cols, device=device
        )
    except Exception as e:
        print(f"Warning: GA rotor magnitude computation failed: {e}", file=sys.stderr)
        out["ga_rotor_magnitude_60d"] = np.nan

    try:
        out["ga_bivector_energy_60d"] = compute_bivector_energy(
            features, window=window, feature_cols=feature_cols, device=device
        )
    except Exception as e:
        print(f"Warning: GA bivector energy computation failed: {e}", file=sys.stderr)
        out["ga_bivector_energy_60d"] = np.nan

    return out
