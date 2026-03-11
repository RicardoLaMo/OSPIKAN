"""
DSL Executor: dispatches parsed DSL queries to appropriate backends.

Connects DSL queries to KAN knowledge store, pricing models, and regime analysis.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import math

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
)
from ..pricing.regime_adjusted import RegimeAdjustedPricer


@dataclass
class ExecutionContext:
    """Context for DSL query execution."""
    kan_store: Optional[Any] = None  # KANKnowledgeStore
    regime_classifier: Optional[Any] = None  # geometric_regime_classification function
    transition_matrix_fn: Optional[Any] = None  # regime_transition_matrix function
    current_regime_features: Optional[Dict[str, float]] = None  # Current market state
    current_regime_label: Optional[str] = None  # Current regime name


class DSLExecutor:
    """Executes parsed DSL queries."""

    # Mapping from regime names to feature dicts (placeholder - would come from data)
    REGIME_NAMES = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]

    def __init__(self, context: Optional[ExecutionContext] = None):
        """
        Initialize executor.

        Args:
            context: ExecutionContext with backends and data
        """
        self.context = context or ExecutionContext()
        self.pricer = RegimeAdjustedPricer(kan_store=self.context.kan_store)

    def execute(self, node: ASTNode) -> Dict[str, Any]:
        """
        Execute a DSL query node.

        Args:
            node: Parsed AST node

        Returns:
            Result dictionary
        """
        if isinstance(node, PriceQuery):
            return self._execute_price(node)
        elif isinstance(node, RegimeCurrentQuery):
            return self._execute_regime_current(node)
        elif isinstance(node, RegimeProbQuery):
            return self._execute_regime_prob(node)
        elif isinstance(node, CovarianceQuery):
            return self._execute_covariance(node)
        elif isinstance(node, TransitionQuery):
            return self._execute_transition(node)
        elif isinstance(node, WhatIfQuery):
            return self._execute_what_if(node)
        elif isinstance(node, ExplainQuery):
            return self._execute_explain(node)
        elif isinstance(node, SurfaceQuery):
            return self._execute_surface(node)
        else:
            raise ValueError(f"Unknown query type: {type(node)}")

    def _execute_price(self, query: PriceQuery) -> Dict[str, Any]:
        """Execute PRICE query."""
        # Determine regime features
        regime_features = self._get_regime_features(query.regime)

        # Price the option
        result = self.pricer.price(
            spot=query.spot_price,
            strike=query.strike_price,
            time_to_expiry=query.time_to_expiry,
            volatility=query.volatility,
            risk_free_rate=query.risk_free_rate,
            dividend_yield=query.dividend_yield,
            option_type=query.option_type,
            regime_features=regime_features,
        )

        return {
            "query_type": "PRICE",
            "option_type": query.option_type,
            "spot": query.spot_price,
            "strike": query.strike_price,
            "time_to_expiry": query.time_to_expiry,
            "risk_free_rate": query.risk_free_rate,
            "dividend_yield": query.dividend_yield,
            "volatility_used": result.used_volatility,
            "price": result.price,
            "delta": result.delta,
            "gamma": result.gamma,
            "vega": result.vega,
            "theta": result.theta,
            "rho": result.rho,
            "regime": query.regime or "current",
        }

    def _execute_regime_current(self, query: RegimeCurrentQuery) -> Dict[str, Any]:
        """Execute REGIME current query."""
        # Would call regime classifier with asset data
        # For now, return placeholder
        return {
            "query_type": "REGIME_CURRENT",
            "asset": query.asset,
            "current_regime": self.context.current_regime_label or "STABLE",
            "regime_features": self.context.current_regime_features or {},
        }

    def _execute_regime_prob(self, query: RegimeProbQuery) -> Dict[str, Any]:
        """Execute REGIME prob query."""
        if not self.context.kan_store:
            raise RuntimeError("KANKnowledgeStore required for regime probability query")

        # Get regime features (use current as baseline)
        regime_features = self.context.current_regime_features or {
            col: 0.0 for col in ["ricci_mean_core_60d", "ricci_min_core_60d",
                                 "mst_stress_core_60d", "ga_rotor_magnitude_60d",
                                 "realized_vol_20d", "momentum_10d",
                                 "p_regime_0", "p_regime_1"]
        }

        # Query transition probabilities
        probs = self.context.kan_store.query_transition(
            regime_features,
            horizon=query.horizon,
            normalize=True,
        )

        # Map to regime names
        prob_dict = {
            self.REGIME_NAMES[i]: float(probs[i].item()) if hasattr(probs[i], 'item')
            else float(probs[i])
            for i in range(len(self.REGIME_NAMES))
        }

        return {
            "query_type": "REGIME_PROB",
            "from_regime": query.from_regime,
            "to_regime": query.to_regime,
            "horizon": query.horizon,
            "horizon_days": int(query.horizon * 365),
            "all_probs": prob_dict,
            "target_prob": prob_dict.get(query.to_regime, None),
        }

    def _execute_covariance(self, query: CovarianceQuery) -> Dict[str, Any]:
        """Execute COVARIANCE query."""
        if not self.context.kan_store:
            raise RuntimeError("KANKnowledgeStore required for covariance query")

        # Get regime features
        regime_features = self._get_regime_features(query.regime)

        # Query covariance
        cov_matrix = self.context.kan_store.query_covariance(
            regime_features,
            normalize=True,
        )

        # Convert to dict for output
        cov_dict = {}
        asset_list = query.assets + ["dxy"]  # Assume 6 assets max
        for i, asset1 in enumerate(asset_list[:6]):
            for j, asset2 in enumerate(asset_list[:6]):
                if i <= j:
                    key = f"{asset1}_{asset2}"
                    val = cov_matrix[i, j].item() if hasattr(cov_matrix[i, j], 'item') else float(cov_matrix[i, j])
                    cov_dict[key] = val

        return {
            "query_type": "COVARIANCE",
            "assets": query.assets,
            "regime": query.regime,
            "window": query.window,
            "window_days": int(query.window * 365),
            "covariance": cov_dict,
        }

    def _execute_transition(self, query: TransitionQuery) -> Dict[str, Any]:
        """Execute TRANSITION query."""
        # Would call regime_transition_matrix function
        # For now, return placeholder
        return {
            "query_type": "TRANSITION",
            "asset": query.asset,
            "normalize": query.normalize,
            "transition_matrix": "Not yet implemented - requires historical data",
        }

    def _execute_what_if(self, query: WhatIfQuery) -> Dict[str, Any]:
        """Execute WHAT_IF query."""
        if not self.context.kan_store:
            raise RuntimeError("KANKnowledgeStore required for what_if analysis")

        # Use current regime as baseline
        current_features = self.context.current_regime_features or self._default_regime_features()

        # Get target regime features (placeholder - would need regime database)
        target_features = self._get_regime_features(query.to_regime)

        # Use provided prices or defaults
        spot = query.spot_price or 100.0
        strike = query.strike_price or 100.0
        time = query.time_to_expiry or 0.25

        # Analyze what-if
        what_if_result = self.pricer.what_if_regime_shift(
            spot=spot,
            strike=strike,
            time_to_expiry=time,
            current_regime_features=current_features,
            target_regime_features=target_features,
            show_fields=query.show or ["price", "delta", "vega", "vol"],
        )

        return {
            "query_type": "WHAT_IF",
            "to_regime": query.to_regime,
            "asset": query.asset,
            "spot": spot,
            "strike": strike,
            "time_to_expiry": time,
            "analysis": what_if_result,
        }

    def _execute_explain(self, query: ExplainQuery) -> Dict[str, Any]:
        """Execute EXPLAIN query."""
        return {
            "query_type": "EXPLAIN",
            "regime": query.regime,
            "features": query.features,
            "explanation": f"Characteristics of {query.regime} regime: {', '.join(query.features)}",
        }

    def _execute_surface(self, query: SurfaceQuery) -> Dict[str, Any]:
        """Execute SURFACE query."""
        if not self.context.kan_store:
            raise RuntimeError("KANKnowledgeStore required for surface query")

        regime_features = self._get_regime_features(query.regime)

        # Generate surface
        surface_result = self.pricer.surface(
            spot=100.0,  # Use baseline spot
            regime_features=regime_features,
            strikes=query.strikes,
            time_to_expiry=query.time_to_expiry,
        )

        return {
            "query_type": "SURFACE",
            "asset": query.asset,
            "regime": query.regime,
            "time_to_expiry": query.time_to_expiry,
            "vol_type": query.vol_type,
            "strikes": surface_result["strikes"],
            "vols": surface_result["vols"],
        }

    def _get_regime_features(self, regime_name: Optional[str]) -> Dict[str, float]:
        """
        Get feature dict for a regime name.

        Args:
            regime_name: Regime name or None for current

        Returns:
            Feature dict
        """
        if regime_name is None or regime_name == "current":
            return self.context.current_regime_features or self._default_regime_features()

        # Return regime-specific features (placeholder)
        # In production, would lookup from database
        regime_map = {
            "STABLE": {"ricci_mean_core_60d": 0.1, "mst_stress_core_60d": 0.2, "realized_vol_20d": 0.15},
            "TRANSITION": {"ricci_mean_core_60d": -0.05, "mst_stress_core_60d": 0.5, "realized_vol_20d": 0.25},
            "STRESS": {"ricci_mean_core_60d": -0.3, "mst_stress_core_60d": 0.8, "realized_vol_20d": 0.40},
            "RECOVERY": {"ricci_mean_core_60d": -0.1, "mst_stress_core_60d": 0.4, "realized_vol_20d": 0.22},
        }

        base_features = self._default_regime_features()
        if regime_name in regime_map:
            base_features.update(regime_map[regime_name])

        return base_features

    def _default_regime_features(self) -> Dict[str, float]:
        """Get default (zero) regime features."""
        return {
            "ricci_mean_core_60d": 0.0,
            "ricci_min_core_60d": 0.0,
            "mst_stress_core_60d": 0.0,
            "ga_rotor_magnitude_60d": 0.0,
            "realized_vol_20d": 0.2,
            "momentum_10d": 0.0,
            "p_regime_0": 0.5,
            "p_regime_1": 0.5,
        }
