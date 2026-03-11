"""
Feature bridge for KAN Knowledge Store.

Converts regime feature dictionaries to PyTorch tensors for KAN inputs.
"""

from typing import Dict, List, Tuple
import torch

# Regime feature columns (8-element vector)
REGIME_FEATURE_COLS = [
    "ricci_mean_core_60d",      # Forman-Ricci curvature (negative = stress)
    "ricci_min_core_60d",       # Minimum edge curvature
    "mst_stress_core_60d",      # MST stress index
    "ga_rotor_magnitude_60d",   # GA rotor magnitude (regime shift size)
    "realized_vol_20d",         # 20-day realized vol (annualized)
    "momentum_10d",             # Log momentum 10-day
    "p_regime_0",               # Markov smoothed prob regime 0
    "p_regime_1",               # Markov smoothed prob regime 1
]


def features_to_tensor(features: Dict[str, float], device: str = "cpu") -> torch.Tensor:
    """
    Convert regime features dictionary to tensor.

    Args:
        features: Dict with keys from REGIME_FEATURE_COLS
        device: torch device ("cpu" or "cuda")

    Returns:
        Tensor of shape (1, 8) with regime features
    """
    values = []
    for col in REGIME_FEATURE_COLS:
        if col not in features:
            raise KeyError(f"Missing feature: {col}")
        values.append(float(features[col]))

    return torch.tensor([values], dtype=torch.float32, device=device)


def normalize_features(features: torch.Tensor, mean: torch.Tensor = None,
                      std: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize regime features to zero mean, unit variance.

    Args:
        features: Tensor of shape (batch, 8)
        mean: Pre-computed mean (if None, compute from features)
        std: Pre-computed std (if None, compute from features)

    Returns:
        (normalized_features, mean, std)
    """
    if mean is None:
        mean = features.mean(dim=0, keepdim=True)
    if std is None:
        std = features.std(dim=0, keepdim=True).clamp(min=1e-8)

    normalized = (features - mean) / std
    return normalized, mean, std


def denormalize_features(features: torch.Tensor, mean: torch.Tensor,
                        std: torch.Tensor) -> torch.Tensor:
    """
    Denormalize regime features.

    Args:
        features: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Original scale features
    """
    return features * std + mean


class FeatureBridge:
    """Bridge between regime features and KAN model inputs."""

    def __init__(self, feature_cols: List[str] = None):
        """
        Initialize feature bridge.

        Args:
            feature_cols: List of feature column names (default: REGIME_FEATURE_COLS)
        """
        self.feature_cols = feature_cols or REGIME_FEATURE_COLS
        self.mean = None
        self.std = None

    def fit(self, features: torch.Tensor) -> None:
        """
        Fit normalization statistics.

        Args:
            features: Tensor of shape (batch, n_features)
        """
        self.mean = features.mean(dim=0, keepdim=True)
        self.std = features.std(dim=0, keepdim=True).clamp(min=1e-8)

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features.

        Args:
            features: Tensor of shape (batch, n_features)

        Returns:
            Normalized tensor
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before transform()")
        return (features - self.mean) / self.std

    def inverse_transform(self, features: torch.Tensor) -> torch.Tensor:
        """
        Denormalize features.

        Args:
            features: Normalized tensor

        Returns:
            Original scale tensor
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        return features * self.std + self.mean
