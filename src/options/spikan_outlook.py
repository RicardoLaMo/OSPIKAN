"""
SPIKAN-backed market outlook adapter for the options DSL.

This keeps the user-facing DSL finance-native while isolating checkpoint loading,
feature mapping, and score calibration away from the parser/executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import math
import sys

import torch

from ..physics.spikan import MarketSPIKAN, SPIKANConfig


class _LegacyTrainingConfig:
    """Placeholder for legacy torch checkpoints that stored __main__.TrainingConfig."""


@dataclass
class SPIKANOutlookResult:
    """Trader-facing summary derived from SPIKAN outputs."""

    asset: str
    horizon: float
    regime: str
    direction: str
    outlook_score: float
    confidence: float
    shock_risk: str
    flow_regime: str
    model_quality: float
    trend_signal: float
    shock_score: float
    raw_signal: float


class SPIKANOutlookEngine:
    """Thin inference adapter that turns SPIKAN outputs into trader-facing signals."""

    DEFAULT_CHECKPOINTS = (
        "reports/silver/spikan_smoke_amp2/spikan_shock.pt",
        "reports/silver/spikan_smoke_amp/spikan_shock.pt",
        "reports/silver/spikan_smoke/spikan_shock.pt",
    )

    ASSET_INDEX = {
        "silver": 0,
        "gold": 1,
        "dxy": 2,
    }

    REGIME_PRESETS = {
        "STABLE": {
            "ricci_mean_core_60d": 0.10,
            "ricci_min_core_60d": 0.02,
            "mst_stress_core_60d": 0.20,
            "ga_rotor_magnitude_60d": 0.08,
            "realized_vol_20d": 0.16,
            "momentum_10d": 0.03,
            "p_regime_0": 0.70,
            "p_regime_1": 0.20,
        },
        "TRANSITION": {
            "ricci_mean_core_60d": -0.05,
            "ricci_min_core_60d": -0.10,
            "mst_stress_core_60d": 0.48,
            "ga_rotor_magnitude_60d": 0.28,
            "realized_vol_20d": 0.24,
            "momentum_10d": 0.00,
            "p_regime_0": 0.40,
            "p_regime_1": 0.40,
        },
        "STRESS": {
            "ricci_mean_core_60d": -0.30,
            "ricci_min_core_60d": -0.45,
            "mst_stress_core_60d": 0.85,
            "ga_rotor_magnitude_60d": 0.42,
            "realized_vol_20d": 0.42,
            "momentum_10d": -0.05,
            "p_regime_0": 0.15,
            "p_regime_1": 0.65,
        },
        "RECOVERY": {
            "ricci_mean_core_60d": -0.08,
            "ricci_min_core_60d": -0.12,
            "mst_stress_core_60d": 0.36,
            "ga_rotor_magnitude_60d": 0.18,
            "realized_vol_20d": 0.22,
            "momentum_10d": 0.04,
            "p_regime_0": 0.55,
            "p_regime_1": 0.25,
        },
        "current": {},
    }

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self.model, self.model_quality = self._load_model(self.checkpoint_path)

    @classmethod
    def discover_checkpoint(cls, preferred_path: Optional[str] = None) -> Optional[str]:
        """Return the first available SPIKAN checkpoint path."""
        candidates = []
        if preferred_path:
            candidates.append(preferred_path)
        candidates.extend(cls.DEFAULT_CHECKPOINTS)

        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return str(candidate)
        return None

    def _load_model(self, checkpoint_path: str) -> tuple[MarketSPIKAN, float]:
        """Load a SPIKAN checkpoint saved by the training script."""
        sys.modules["__main__"].TrainingConfig = _LegacyTrainingConfig
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        config = checkpoint.get("config")
        model_config = SPIKANConfig(
            n_assets=getattr(config, "n_assets", 6),
            n_macro=getattr(config, "n_macro", 4),
            n_geometric=getattr(config, "n_geometric", 4),
            hidden_dim=getattr(config, "hidden_dim", 32),
            n_terms=getattr(config, "n_terms", 8),
            output_dim=1,
        )

        model = MarketSPIKAN(model_config).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_metrics = checkpoint.get("test_metrics", {})
        model_quality = float(test_metrics.get("auc", 0.5))
        return model, model_quality

    def outlook(
        self,
        asset: str,
        horizon: float,
        regime: str = "current",
        regime_features: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float | str]:
        """
        Generate a trader-facing market outlook.

        The adapter uses KAN-style regime features as the state description and
        feeds those signals into SPIKAN's asset/macro/geometry branches.
        """
        backdrop = regime or "current"
        features = self._merge_features(backdrop, regime_features or {})
        assets, macro, geometry, t = self._build_inputs(asset, horizon, features)

        with torch.no_grad():
            raw_signal = float(self.model(assets, macro, geometry, t).squeeze().item())

        diagnostics = self.model.forward_with_derivatives(assets, macro, geometry, t)
        grad_mag = float(diagnostics["u_x"].detach().abs().mean().item())
        curvature_mag = float(diagnostics["u_xx"].detach().abs().mean().item())
        time_mag = float(diagnostics["u_t"].detach().abs().mean().item())

        trend_signal = math.tanh(raw_signal)
        shock_score = 1.0 - math.exp(
            -0.15 * (
                math.log1p(grad_mag)
                + 0.5 * math.log1p(curvature_mag)
                + 0.5 * math.log1p(time_mag)
            )
        )
        shock_score = max(0.0, min(1.0, shock_score))

        if trend_signal >= 0.45:
            direction = "bullish"
        elif trend_signal >= 0.15:
            direction = "lean bullish"
        elif trend_signal <= -0.45:
            direction = "bearish"
        elif trend_signal <= -0.15:
            direction = "lean bearish"
        else:
            direction = "neutral"

        if shock_score >= 0.70:
            shock_risk = "HIGH"
            flow_regime = "SHOCK_PRONE"
        elif abs(trend_signal) >= 0.35:
            shock_risk = "MODERATE"
            flow_regime = "TRENDING"
        elif shock_score >= 0.40:
            shock_risk = "MODERATE"
            flow_regime = "UNSETTLED"
        else:
            shock_risk = "LOW"
            flow_regime = "RANGE_BOUND"

        confidence = 0.15 + 0.55 * abs(trend_signal) + 0.30 * max(0.0, self.model_quality - 0.50) * 2.0
        confidence = max(0.10, min(0.90, confidence))

        result = SPIKANOutlookResult(
            asset=asset,
            horizon=horizon,
            regime=backdrop,
            direction=direction,
            outlook_score=trend_signal * 100.0,
            confidence=confidence,
            shock_risk=shock_risk,
            flow_regime=flow_regime,
            model_quality=self.model_quality,
            trend_signal=trend_signal,
            shock_score=shock_score,
            raw_signal=raw_signal,
        )
        return {
            "query_type": "OUTLOOK",
            "asset": result.asset,
            "horizon": result.horizon,
            "horizon_days": max(1, int(round(result.horizon * 365))),
            "regime": result.regime,
            "direction": result.direction,
            "outlook_score": result.outlook_score,
            "confidence": result.confidence,
            "shock_risk": result.shock_risk,
            "flow_regime": result.flow_regime,
            "model_quality": result.model_quality,
            "trend_signal": result.trend_signal,
            "shock_score": result.shock_score,
            "raw_signal": result.raw_signal,
            "source_model": Path(self.checkpoint_path).name,
        }

    def _merge_features(self, regime: str, regime_features: Dict[str, float]) -> Dict[str, float]:
        merged = dict(self.REGIME_PRESETS.get(regime, self.REGIME_PRESETS["STABLE"]))
        merged.update(regime_features)
        return merged

    def _build_inputs(
        self,
        asset: str,
        horizon: float,
        features: Dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        momentum = float(features.get("momentum_10d", 0.0))
        vol = float(features.get("realized_vol_20d", 0.20))
        stress = float(features.get("mst_stress_core_60d", 0.0))
        rotor = float(features.get("ga_rotor_magnitude_60d", 0.0))
        ricci = float(features.get("ricci_mean_core_60d", 0.0))
        p0 = float(features.get("p_regime_0", 0.5))
        p1 = float(features.get("p_regime_1", 0.5))

        focus_idx = self.ASSET_INDEX.get(asset, 0)
        n_assets = self.model.config.n_assets
        asset_bias = momentum - 0.25 * stress + 0.10 * (p0 - p1)
        assets = torch.full((1, n_assets), asset_bias / 4.0, dtype=torch.float32, device=self.device)
        assets[0, focus_idx % n_assets] = asset_bias
        if n_assets > 3:
            assets[0, 3] = vol - 0.20
        if n_assets > 4:
            assets[0, 4] = rotor
        if n_assets > 5:
            assets[0, 5] = stress

        macro_values = [
            stress,
            vol - 0.20,
            p0 - 0.50,
            p1 - 0.50,
        ]
        macro = torch.tensor(
            [self._fit_dim(macro_values, self.model.config.n_macro)],
            dtype=torch.float32,
            device=self.device,
        )

        geometry_values = [
            ricci,
            stress,
            rotor,
            momentum,
        ]
        geometry = torch.tensor(
            [self._fit_dim(geometry_values, self.model.config.n_geometric)],
            dtype=torch.float32,
            device=self.device,
        )

        t = torch.tensor([[max(horizon, 1.0 / 365.0)]], dtype=torch.float32, device=self.device)
        return assets, macro, geometry, t

    @staticmethod
    def _fit_dim(values: list[float], size: int) -> list[float]:
        if len(values) >= size:
            return values[:size]
        return values + [0.0] * (size - len(values))
