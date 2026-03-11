"""
KAN Knowledge Store: unified interface for all KAN networks.

Manages training, inference, and persistence of vol surface, covariance,
and transition KAN models.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import json

from .config import KANStoreConfig
from .feature_bridge import FeatureBridge, REGIME_FEATURE_COLS
from .vol_surface_kan import VolSurfaceKAN
from .covariance_kan import CovarianceKAN
from .transition_kan import TransitionKAN


class KANKnowledgeStore(nn.Module):
    """
    Unified KAN Knowledge Store for options pricing.

    Contains three learned KAN networks:
    - vol_surface: (regime, moneyness, time) -> vol
    - covariance: regime -> covariance matrix
    - transition: (regime, horizon) -> transition probabilities
    """

    def __init__(
        self,
        config: Optional[KANStoreConfig] = None,
        device: str = "cpu",
    ):
        """
        Initialize KANKnowledgeStore.

        Args:
            config: KANStoreConfig (default: KANStoreConfig())
            device: torch device ("cpu" or "cuda")
        """
        super().__init__()

        self.config = config or KANStoreConfig()
        self.device = device

        # Feature bridge for normalization
        self.feature_bridge = FeatureBridge(REGIME_FEATURE_COLS)

        # Initialize KAN networks
        self.vol_surface = VolSurfaceKAN(
            layer_sizes=self.config.vol_surface_layers,
            grid_size=self.config.grid_size,
            spline_order=self.config.spline_order,
            dropout=self.config.dropout,
            residual=self.config.residual,
        ).to(device)

        self.covariance = CovarianceKAN(
            layer_sizes=self.config.covariance_layers,
            grid_size=self.config.grid_size,
            spline_order=self.config.spline_order,
            dropout=self.config.dropout,
            residual=self.config.residual,
            n_assets=6,
        ).to(device)

        self.transition = TransitionKAN(
            layer_sizes=self.config.transition_layers,
            grid_size=self.config.grid_size,
            spline_order=self.config.spline_order,
            dropout=self.config.dropout,
            residual=self.config.residual,
            n_regimes=4,
        ).to(device)

    def forward(self):
        """Forward pass not defined for store; use specific query methods."""
        raise NotImplementedError("Use query_vol_surface, query_covariance, or query_transition")

    def query_vol_surface(
        self,
        regime_features: Dict[str, float],
        log_moneyness: float,
        time_to_expiry: float,
        normalize: bool = True,
    ) -> float:
        """
        Query volatility surface.

        Args:
            regime_features: Dict with regime feature values
            log_moneyness: log(S/K)
            time_to_expiry: T/365 (years)
            normalize: Whether to use feature normalization

        Returns:
            Implied volatility (scalar)
        """
        with torch.no_grad():
            # Prepare regime features
            regime_tensor = torch.tensor(
                [[regime_features.get(col, 0.0) for col in REGIME_FEATURE_COLS]],
                dtype=torch.float32,
                device=self.device,
            )

            if normalize and self.feature_bridge.mean is not None:
                regime_tensor = self.feature_bridge.transform(regime_tensor)

            # Prepare moneyness and time
            moneyness_tensor = torch.tensor(
                [[log_moneyness]], dtype=torch.float32, device=self.device
            )
            time_tensor = torch.tensor(
                [[time_to_expiry]], dtype=torch.float32, device=self.device
            )

            # Forward pass
            vol = self.vol_surface(regime_tensor, moneyness_tensor, time_tensor)

            return vol.squeeze().item()

    def query_covariance(
        self,
        regime_features: Dict[str, float],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Query covariance matrix.

        Args:
            regime_features: Dict with regime feature values
            normalize: Whether to use feature normalization

        Returns:
            Covariance matrix of shape (6, 6)
        """
        with torch.no_grad():
            # Prepare regime features
            regime_tensor = torch.tensor(
                [[regime_features.get(col, 0.0) for col in REGIME_FEATURE_COLS]],
                dtype=torch.float32,
                device=self.device,
            )

            if normalize and self.feature_bridge.mean is not None:
                regime_tensor = self.feature_bridge.transform(regime_tensor)

            # Forward pass
            cov = self.covariance(regime_tensor)

            return cov.squeeze(0)

    def query_transition(
        self,
        regime_features: Dict[str, float],
        horizon: float,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Query transition probabilities.

        Args:
            regime_features: Dict with regime feature values
            horizon: Time horizon in years (converted to trading days)
            normalize: Whether to use feature normalization

        Returns:
            Softmax probabilities over 4 regimes of shape (4,)
        """
        with torch.no_grad():
            # Prepare regime features
            regime_tensor = torch.tensor(
                [[regime_features.get(col, 0.0) for col in REGIME_FEATURE_COLS]],
                dtype=torch.float32,
                device=self.device,
            )

            if normalize and self.feature_bridge.mean is not None:
                regime_tensor = self.feature_bridge.transform(regime_tensor)

            # Prepare horizon (in trading days)
            horizon_trading_days = horizon * 252
            horizon_tensor = torch.tensor(
                [[horizon_trading_days / 252.0]], dtype=torch.float32, device=self.device
            )

            # Forward pass
            probs = self.transition(regime_tensor, horizon_tensor)

            return probs.squeeze(0)

    def fit_normalization(self, features_tensor: torch.Tensor) -> None:
        """
        Fit feature normalization statistics.

        Args:
            features_tensor: Tensor of shape (batch, n_features)
        """
        self.feature_bridge.fit(features_tensor)

    def save(self, path: str) -> None:
        """
        Save store to disk.

        Args:
            path: Directory path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.vol_surface.state_dict(), path / "vol_surface.pt")
        torch.save(self.covariance.state_dict(), path / "covariance.pt")
        torch.save(self.transition.state_dict(), path / "transition.pt")

        # Save normalization statistics
        if self.feature_bridge.mean is not None:
            torch.save(self.feature_bridge.mean, path / "norm_mean.pt")
            torch.save(self.feature_bridge.std, path / "norm_std.pt")

        # Save config
        config_dict = {
            "vol_surface_layers": self.config.vol_surface_layers,
            "covariance_layers": self.config.covariance_layers,
            "transition_layers": self.config.transition_layers,
            "grid_size": self.config.grid_size,
            "spline_order": self.config.spline_order,
            "dropout": self.config.dropout,
            "residual": self.config.residual,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load store from disk.

        Args:
            path: Directory path to load from
        """
        path = Path(path)

        # Load model weights
        self.vol_surface.load_state_dict(torch.load(path / "vol_surface.pt", map_location=self.device))
        self.covariance.load_state_dict(torch.load(path / "covariance.pt", map_location=self.device))
        self.transition.load_state_dict(torch.load(path / "transition.pt", map_location=self.device))

        # Load normalization statistics
        if (path / "norm_mean.pt").exists():
            self.feature_bridge.mean = torch.load(path / "norm_mean.pt", map_location=self.device)
            self.feature_bridge.std = torch.load(path / "norm_std.pt", map_location=self.device)

    def to(self, device):
        """Move store to device."""
        self.device = device
        self.vol_surface.to(device)
        self.covariance.to(device)
        self.transition.to(device)
        if self.feature_bridge.mean is not None:
            self.feature_bridge.mean = self.feature_bridge.mean.to(device)
            self.feature_bridge.std = self.feature_bridge.std.to(device)
        return self

    def eval(self):
        """Set all networks to eval mode."""
        self.vol_surface.eval()
        self.covariance.eval()
        self.transition.eval()
        return self

    def train(self, mode: bool = True):
        """Set all networks to train mode."""
        self.vol_surface.train(mode)
        self.covariance.train(mode)
        self.transition.train(mode)
        return self

    def get_total_regularization_loss(self, lambda_l1: float = 1e-4) -> torch.Tensor:
        """
        Get total regularization loss from all networks.

        Args:
            lambda_l1: L1 regularization coefficient

        Returns:
            Total regularization loss
        """
        total_loss = (
            self.vol_surface.regularization_loss(lambda_l1) +
            self.covariance.regularization_loss(lambda_l1) +
            self.transition.regularization_loss(lambda_l1)
        )
        return total_loss
