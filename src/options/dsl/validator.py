"""
Semantic validator for the options pricing DSL.

Checks constraints on parsed AST nodes (e.g., valid regimes, asset names, ranges).
"""

from typing import Set
from .ast_nodes import (
    ASTNode,
    PriceQuery,
    RegimeCurrentQuery,
    RegimeProbQuery,
    CovarianceQuery,
    TransitionQuery,
    WhatIfQuery,
    ExplainQuery,
    SurfaceQuery,
    OutlookQuery,
)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


class DSLValidator:
    """Semantic validator for DSL AST nodes."""

    VALID_REGIMES = {"STABLE", "TRANSITION", "STRESS", "RECOVERY", "current"}
    VALID_OPTION_TYPES = {"call", "put"}
    VALID_ASSETS = {"silver", "gold", "dxy"}  # Extensible
    VALID_FEATURES = {
        "ricci_curvature",
        "ricci_mean_core_60d",
        "ricci_min_core_60d",
        "mst_stress",
        "mst_stress_core_60d",
        "ga_rotor",
        "ga_rotor_magnitude_60d",
        "realized_vol",
        "realized_vol_20d",
        "momentum",
        "momentum_10d",
    }
    VALID_SHOW_FIELDS = {"delta", "gamma", "vega", "theta", "rho", "vol", "price"}

    def validate(self, node: ASTNode) -> None:
        """
        Validate an AST node.

        Args:
            node: AST node to validate

        Raises:
            ValidationError: If validation fails
        """
        if isinstance(node, PriceQuery):
            self._validate_price(node)
        elif isinstance(node, RegimeCurrentQuery):
            self._validate_regime_current(node)
        elif isinstance(node, RegimeProbQuery):
            self._validate_regime_prob(node)
        elif isinstance(node, CovarianceQuery):
            self._validate_covariance(node)
        elif isinstance(node, TransitionQuery):
            self._validate_transition(node)
        elif isinstance(node, WhatIfQuery):
            self._validate_what_if(node)
        elif isinstance(node, ExplainQuery):
            self._validate_explain(node)
        elif isinstance(node, SurfaceQuery):
            self._validate_surface(node)
        elif isinstance(node, OutlookQuery):
            self._validate_outlook(node)

    def _validate_price(self, node: PriceQuery) -> None:
        """Validate PriceQuery."""
        if node.option_type.lower() not in self.VALID_OPTION_TYPES:
            raise ValidationError(f"Invalid option type: {node.option_type}")

        if node.spot_price <= 0:
            raise ValidationError(f"Spot price must be positive, got {node.spot_price}")

        if node.strike_price <= 0:
            raise ValidationError(f"Strike price must be positive, got {node.strike_price}")

        if node.time_to_expiry <= 0:
            raise ValidationError(f"Time to expiry must be positive, got {node.time_to_expiry}")

        # Volatility can be float or string
        if isinstance(node.volatility, float):
            if node.volatility < 0:
                raise ValidationError(f"Volatility must be non-negative, got {node.volatility}")
        elif isinstance(node.volatility, str):
            if node.volatility not in ("kan_regime",) and not node.volatility.startswith("hist_"):
                raise ValidationError(f"Invalid volatility spec: {node.volatility}")

        if node.regime and node.regime not in self.VALID_REGIMES:
            raise ValidationError(f"Invalid regime: {node.regime}")

        if node.risk_free_rate < -0.5 or node.risk_free_rate > 1.0:
            raise ValidationError(f"Risk-free rate out of reasonable range: {node.risk_free_rate}")

        if node.dividend_yield < 0 or node.dividend_yield > 1.0:
            raise ValidationError(f"Dividend yield out of range: {node.dividend_yield}")

    def _validate_regime_current(self, node: RegimeCurrentQuery) -> None:
        """Validate RegimeCurrentQuery."""
        if node.asset not in self.VALID_ASSETS:
            raise ValidationError(f"Unknown asset: {node.asset}")

    def _validate_regime_prob(self, node: RegimeProbQuery) -> None:
        """Validate RegimeProbQuery."""
        if node.from_regime not in self.VALID_REGIMES:
            raise ValidationError(f"Invalid from_regime: {node.from_regime}")
        if node.to_regime not in self.VALID_REGIMES:
            raise ValidationError(f"Invalid to_regime: {node.to_regime}")
        if node.horizon <= 0:
            raise ValidationError(f"Horizon must be positive: {node.horizon}")

    def _validate_covariance(self, node: CovarianceQuery) -> None:
        """Validate CovarianceQuery."""
        for asset in node.assets:
            if asset not in self.VALID_ASSETS:
                raise ValidationError(f"Unknown asset: {asset}")
        if node.regime not in self.VALID_REGIMES:
            raise ValidationError(f"Invalid regime: {node.regime}")
        if node.window <= 0:
            raise ValidationError(f"Window must be positive: {node.window}")

    def _validate_transition(self, node: TransitionQuery) -> None:
        """Validate TransitionQuery."""
        if node.asset not in self.VALID_ASSETS:
            raise ValidationError(f"Unknown asset: {node.asset}")

    def _validate_what_if(self, node: WhatIfQuery) -> None:
        """Validate WhatIfQuery."""
        if node.to_regime not in self.VALID_REGIMES:
            raise ValidationError(f"Invalid to_regime: {node.to_regime}")
        if node.asset not in self.VALID_ASSETS:
            raise ValidationError(f"Unknown asset: {node.asset}")

        if node.spot_price is not None and node.spot_price <= 0:
            raise ValidationError(f"Spot price must be positive: {node.spot_price}")
        if node.strike_price is not None and node.strike_price <= 0:
            raise ValidationError(f"Strike price must be positive: {node.strike_price}")
        if node.time_to_expiry is not None and node.time_to_expiry <= 0:
            raise ValidationError(f"Time to expiry must be positive: {node.time_to_expiry}")

        if node.show:
            for field in node.show:
                if field not in self.VALID_SHOW_FIELDS:
                    raise ValidationError(f"Invalid show field: {field}")

    def _validate_explain(self, node: ExplainQuery) -> None:
        """Validate ExplainQuery."""
        if node.regime not in self.VALID_REGIMES and node.regime != "all":
            raise ValidationError(f"Invalid regime: {node.regime}")

        for feature in node.features:
            if feature not in self.VALID_FEATURES:
                raise ValidationError(f"Unknown feature: {feature}")

    def _validate_surface(self, node: SurfaceQuery) -> None:
        """Validate SurfaceQuery."""
        if node.asset not in self.VALID_ASSETS:
            raise ValidationError(f"Unknown asset: {node.asset}")
        if node.regime not in self.VALID_REGIMES and node.regime != "all":
            raise ValidationError(f"Invalid regime: {node.regime}")

        for strike in node.strikes:
            if strike <= 0:
                raise ValidationError(f"Strikes must be positive: {strike}")

        if node.time_to_expiry <= 0:
            raise ValidationError(f"Time to expiry must be positive: {node.time_to_expiry}")

        if node.vol_type not in ("implied", "realized", "kan"):
            raise ValidationError(f"Invalid vol_type: {node.vol_type}")

    def _validate_outlook(self, node: OutlookQuery) -> None:
        """Validate OutlookQuery."""
        if node.asset not in self.VALID_ASSETS:
            raise ValidationError(f"Unknown asset: {node.asset}")
        if node.horizon <= 0:
            raise ValidationError(f"Horizon must be positive: {node.horizon}")
        if node.regime and node.regime not in self.VALID_REGIMES:
            raise ValidationError(f"Invalid regime: {node.regime}")
