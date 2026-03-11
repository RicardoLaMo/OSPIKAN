"""
Regime-Adjusted Option Pricer.

Wraps Black-Scholes pricing with regime-specific volatility from KAN network.
"""

from typing import Optional, Dict
from dataclasses import dataclass

from .black_scholes import black_scholes, black_scholes_greeks, BSResult


@dataclass
class RegimeAdjustedResult:
    """Result from regime-adjusted pricing."""
    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    used_volatility: float
    regime: Optional[str] = None


class RegimeAdjustedPricer:
    """
    Option pricer with regime-adjusted volatility.

    Uses a KAN knowledge store to look up regime-specific volatility
    before delegating to Black-Scholes pricing.
    """

    def __init__(self, kan_store=None):
        """
        Initialize pricer.

        Args:
            kan_store: KANKnowledgeStore instance (can be None for basic pricing)
        """
        self.kan_store = kan_store

    def price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        option_type: str = "call",
        regime_features: Optional[Dict[str, float]] = None,
    ) -> RegimeAdjustedResult:
        """
        Price an option with optional regime adjustment.

        Args:
            spot: Current spot price (S)
            strike: Strike price (K)
            time_to_expiry: Time to expiry in years (T)
            volatility: Base volatility or 'kan_regime' for lookup
            risk_free_rate: Risk-free rate (r)
            dividend_yield: Dividend yield (q)
            option_type: 'call' or 'put'
            regime_features: Regime feature dict for KAN vol lookup

        Returns:
            RegimeAdjustedResult with price and Greeks
        """
        # Determine volatility to use
        if isinstance(volatility, str):
            if volatility == "kan_regime" and self.kan_store and regime_features:
                # Look up vol from KAN network
                log_moneyness = __import__('math').log(spot / strike)
                used_vol = self.kan_store.query_vol_surface(
                    regime_features,
                    log_moneyness=log_moneyness,
                    time_to_expiry=time_to_expiry,
                    normalize=True,
                )
            elif volatility.startswith("hist_"):
                # Historical vol (not implemented - would need data)
                raise NotImplementedError(f"Volatility type {volatility} not implemented")
            else:
                raise ValueError(f"Unknown volatility type: {volatility}")
        else:
            used_vol = float(volatility)

        # Price using Black-Scholes
        result = black_scholes_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=used_vol,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            option_type=option_type,
        )

        return RegimeAdjustedResult(
            price=result.price,
            delta=result.delta,
            gamma=result.gamma,
            vega=result.vega,
            theta=result.theta,
            rho=result.rho,
            used_volatility=used_vol,
        )

    def what_if_regime_shift(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        current_regime_features: Dict[str, float],
        target_regime_features: Dict[str, float],
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
        option_type: str = "call",
        show_fields: Optional[list] = None,
    ) -> dict:
        """
        Analyze Greeks sensitivity to regime shift.

        Args:
            spot: Current spot price
            strike: Strike price
            time_to_expiry: Time to expiry in years
            current_regime_features: Current regime features
            target_regime_features: Target regime features
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield
            option_type: 'call' or 'put'
            show_fields: List of fields to return (default: all)

        Returns:
            Dict with pricing results for both regimes and differences
        """
        if not self.kan_store:
            raise RuntimeError("KANKnowledgeStore required for what_if analysis")

        if show_fields is None:
            show_fields = ["price", "delta", "gamma", "vega", "theta", "rho", "vol"]

        # Price in current regime
        import math
        log_moneyness = math.log(spot / strike)

        current_vol = self.kan_store.query_vol_surface(
            current_regime_features,
            log_moneyness=log_moneyness,
            time_to_expiry=time_to_expiry,
            normalize=True,
        )
        current_result = self.price(
            spot, strike, time_to_expiry, current_vol, risk_free_rate,
            dividend_yield, option_type
        )

        # Price in target regime
        target_vol = self.kan_store.query_vol_surface(
            target_regime_features,
            log_moneyness=log_moneyness,
            time_to_expiry=time_to_expiry,
            normalize=True,
        )
        target_result = self.price(
            spot, strike, time_to_expiry, target_vol, risk_free_rate,
            dividend_yield, option_type
        )

        # Build result
        result = {
            "current": {},
            "target": {},
            "delta": {},
        }

        # Populate fields
        field_map = {
            "price": "price",
            "delta": "delta",
            "gamma": "gamma",
            "vega": "vega",
            "theta": "theta",
            "rho": "rho",
            "vol": "used_volatility",
        }

        for field in show_fields:
            if field in field_map:
                attr = field_map[field]
                current_val = getattr(current_result, attr)
                target_val = getattr(target_result, attr)

                result["current"][field] = current_val
                result["target"][field] = target_val
                result["delta"][field] = target_val - current_val

        return result

    def surface(
        self,
        spot: float,
        regime_features: Dict[str, float],
        strikes: list,
        time_to_expiry: float,
        risk_free_rate: float = 0.05,
        dividend_yield: float = 0.0,
    ) -> dict:
        """
        Generate volatility surface for given regime.

        Args:
            spot: Current spot price (used to compute moneyness)
            regime_features: Regime feature dict
            strikes: List of strike prices or moneyness ratios
            time_to_expiry: Time to expiry in years
            risk_free_rate: Risk-free rate
            dividend_yield: Dividend yield

        Returns:
            Dict with strikes and implied vols
        """
        if not self.kan_store:
            raise RuntimeError("KANKnowledgeStore required for surface query")

        import math

        result = {"strikes": [], "vols": []}

        for strike in strikes:
            # Determine if strike is absolute or moneyness ratio
            if strike < 0.1:
                # Likely a moneyness ratio (e.g., 0.9, 0.95, 1.0)
                abs_strike = spot * strike
                moneyness = strike
            else:
                # Absolute strike
                abs_strike = strike
                moneyness = strike / spot

            log_moneyness = math.log(moneyness)

            vol = self.kan_store.query_vol_surface(
                regime_features,
                log_moneyness=log_moneyness,
                time_to_expiry=time_to_expiry,
                normalize=True,
            )

            result["strikes"].append(abs_strike)
            result["vols"].append(vol)

        return result
