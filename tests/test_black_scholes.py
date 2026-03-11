"""
Tests for Black-Scholes option pricing.
"""

import pytest
import math
from src.options.pricing.black_scholes import (
    black_scholes,
    black_scholes_greeks,
    compute_delta,
    compute_gamma,
    compute_vega,
    compute_theta,
    compute_rho,
)


class TestBlackScholesBasic:
    """Test basic Black-Scholes pricing."""

    def test_atm_call_at_expiry(self):
        """At expiry (T->0), ATM call should be worth ~0."""
        price = black_scholes(spot=100, strike=100, time_to_expiry=0.001, volatility=0.2,
                             risk_free_rate=0.05, option_type="call")
        # Very close to expiry, ATM call near intrinsic (0)
        assert price >= 0
        assert price < 1

    def test_itm_call_at_expiry(self):
        """At expiry, ITM call should be worth intrinsic value."""
        intrinsic = 10.0
        price = black_scholes(spot=110, strike=100, time_to_expiry=0.001, volatility=0.2,
                             risk_free_rate=0.05, option_type="call")
        # At expiry, price should be close to intrinsic (discounted)
        assert price > 9.9

    def test_otm_call_at_expiry(self):
        """At expiry, OTM call should be worth 0."""
        price = black_scholes(spot=90, strike=100, time_to_expiry=0.001, volatility=0.2,
                             risk_free_rate=0.05, option_type="call")
        assert abs(price) < 1e-6

    def test_atm_put_at_expiry(self):
        """At expiry, ATM put should be worth ~0."""
        price = black_scholes(spot=100, strike=100, time_to_expiry=0.001, volatility=0.2,
                             risk_free_rate=0.05, option_type="put")
        assert price >= 0
        assert price < 1

    def test_atm_call_value_range(self):
        """ATM call with time value should be positive and less than strike."""
        price = black_scholes(spot=100, strike=100, time_to_expiry=1.0, volatility=0.2,
                             risk_free_rate=0.05, option_type="call")
        # ATM call should have positive time value
        assert price > 0
        # Should be less than spot price
        assert price < 100

    def test_put_call_parity(self):
        """Test put-call parity: C - P = S*e^(-q*T) - K*e^(-r*T)"""
        spot = 100
        strike = 100
        time_to_expiry = 0.25
        volatility = 0.2
        risk_free_rate = 0.05
        dividend_yield = 0.02

        call = black_scholes(spot, strike, time_to_expiry, volatility, risk_free_rate,
                            dividend_yield, "call")
        put = black_scholes(spot, strike, time_to_expiry, volatility, risk_free_rate,
                           dividend_yield, "put")

        # C - P = S*e^(-q*T) - K*e^(-r*T)
        expected_diff = spot * math.exp(-dividend_yield * time_to_expiry) - \
                       strike * math.exp(-risk_free_rate * time_to_expiry)
        actual_diff = call - put

        assert abs(actual_diff - expected_diff) < 0.01  # tolerance for numerical precision

    def test_call_put_symmetry_no_dividend(self):
        """For non-dividend paying, test call-put relationship."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.25
        rate = 0.03

        call = black_scholes(spot, strike, time_to_expiry, volatility, rate, 0, "call")
        put = black_scholes(spot, strike, time_to_expiry, volatility, rate, 0, "put")

        # For ATM, call should be higher than put with positive rate
        assert call > put

    def test_zero_volatility_call(self):
        """With zero volatility, call price should be intrinsic value discounted."""
        spot = 110
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05

        price = black_scholes(spot, strike, time_to_expiry, 0.0, rate, 0, "call")

        # Should be close to (110 - 100*e^(-0.05)) = 10 - 95.12 ≈ intrinsic discounted
        expected = spot - strike * math.exp(-rate * time_to_expiry)
        assert abs(price - expected) < 0.01

    def test_zero_volatility_put(self):
        """With zero volatility, put price should be intrinsic value discounted."""
        spot = 90
        strike = 100
        time_to_expiry = 1.0
        rate = 0.05

        price = black_scholes(spot, strike, time_to_expiry, 0.0, rate, 0, "put")

        # Should be close to (100*e^(-0.05) - 90) = 95.12 - 90 = 5.12
        expected = strike * math.exp(-rate * time_to_expiry) - spot
        assert abs(price - expected) < 0.01

    def test_increasing_volatility_call(self):
        """Increasing volatility should increase call value."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        rate = 0.05

        price_low_vol = black_scholes(spot, strike, time_to_expiry, 0.1, rate, 0, "call")
        price_mid_vol = black_scholes(spot, strike, time_to_expiry, 0.2, rate, 0, "call")
        price_high_vol = black_scholes(spot, strike, time_to_expiry, 0.4, rate, 0, "call")

        assert price_low_vol < price_mid_vol < price_high_vol

    def test_increasing_volatility_put(self):
        """Increasing volatility should increase put value."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        rate = 0.05

        price_low_vol = black_scholes(spot, strike, time_to_expiry, 0.1, rate, 0, "put")
        price_mid_vol = black_scholes(spot, strike, time_to_expiry, 0.2, rate, 0, "put")
        price_high_vol = black_scholes(spot, strike, time_to_expiry, 0.4, rate, 0, "put")

        assert price_low_vol < price_mid_vol < price_high_vol

    def test_increasing_time_call(self):
        """Generally, increasing time should increase call value."""
        spot = 100
        strike = 100
        volatility = 0.2
        rate = 0.05

        price_short = black_scholes(spot, strike, 0.1, volatility, rate, 0, "call")
        price_medium = black_scholes(spot, strike, 0.5, volatility, rate, 0, "call")
        price_long = black_scholes(spot, strike, 1.0, volatility, rate, 0, "call")

        assert price_short < price_medium < price_long

    def test_dividend_effect_call(self):
        """Dividend yield should decrease call value."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        price_no_div = black_scholes(spot, strike, time_to_expiry, volatility, rate, 0, "call")
        price_with_div = black_scholes(spot, strike, time_to_expiry, volatility, rate, 0.02, "call")

        assert price_with_div < price_no_div

    def test_dividend_effect_put(self):
        """Dividend yield should increase put value."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        price_no_div = black_scholes(spot, strike, time_to_expiry, volatility, rate, 0, "put")
        price_with_div = black_scholes(spot, strike, time_to_expiry, volatility, rate, 0.02, "put")

        assert price_with_div > price_no_div


class TestGreeks:
    """Test Greeks computation."""

    def test_delta_call_bounds(self):
        """Call delta should be in [0, 1]."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        delta = compute_delta(spot, strike, time_to_expiry, volatility, rate, 0, "call")
        assert 0 <= delta <= 1
        # For ATM, delta should be close to 0.5 with positive rate
        assert 0.4 < delta < 0.6

    def test_delta_put_bounds(self):
        """Put delta should be in [-1, 0]."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        delta = compute_delta(spot, strike, time_to_expiry, volatility, rate, 0, "put")
        assert -1 <= delta <= 0
        # For ATM, put delta should be close to -0.5
        assert -0.6 < delta < -0.4

    def test_gamma_always_positive(self):
        """Gamma should always be positive (for calls and puts)."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        gamma = compute_gamma(spot, strike, time_to_expiry, volatility, rate, 0)
        assert gamma > 0

    def test_gamma_highest_atm(self):
        """Gamma should be highest for ATM options."""
        spot = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        gamma_otm = compute_gamma(spot, strike=120, time_to_expiry=time_to_expiry,
                                 volatility=volatility, risk_free_rate=rate)
        gamma_atm = compute_gamma(spot, strike=100, time_to_expiry=time_to_expiry,
                                 volatility=volatility, risk_free_rate=rate)
        gamma_itm = compute_gamma(spot, strike=80, time_to_expiry=time_to_expiry,
                                 volatility=volatility, risk_free_rate=rate)

        assert gamma_atm > gamma_otm
        assert gamma_atm > gamma_itm

    def test_vega_always_positive(self):
        """Vega should always be positive (for calls and puts)."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        vega = compute_vega(spot, strike, time_to_expiry, volatility, rate, 0)
        assert vega > 0

    def test_vega_decreases_with_time(self):
        """Vega should decrease as time to expiry decreases."""
        spot = 100
        strike = 100
        volatility = 0.2
        rate = 0.05

        vega_long = compute_vega(spot, strike, time_to_expiry=1.0,
                                volatility=volatility, risk_free_rate=rate)
        vega_short = compute_vega(spot, strike, time_to_expiry=0.1,
                                 volatility=volatility, risk_free_rate=rate)

        assert vega_long > vega_short

    def test_rho_call_positive(self):
        """Call rho should be positive (higher rates increase call value)."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        rho = compute_rho(spot, strike, time_to_expiry, volatility, rate, 0, "call")
        assert rho > 0

    def test_rho_put_negative(self):
        """Put rho should be negative (higher rates decrease put value)."""
        spot = 100
        strike = 100
        time_to_expiry = 0.5
        volatility = 0.2
        rate = 0.05

        rho = compute_rho(spot, strike, time_to_expiry, volatility, rate, 0, "put")
        assert rho < 0

    def test_theta_call_expiring(self):
        """For long call positions, theta is usually negative (time decay)."""
        spot = 100
        strike = 100
        volatility = 0.2
        rate = 0.05

        theta = compute_theta(spot, strike, time_to_expiry=0.1,
                             volatility=volatility, risk_free_rate=rate, option_type="call")
        # Near expiry, ATM call theta is typically negative
        assert theta < 0

    def test_greek_consistency_call(self):
        """Test numerical consistency of Greeks for call."""
        spot = 100
        strike = 100
        time_to_expiry = 0.25
        volatility = 0.2
        rate = 0.05

        result = black_scholes_greeks(spot, strike, time_to_expiry, volatility, rate, 0, "call")

        assert result.price > 0
        assert 0 < result.delta < 1
        assert result.gamma > 0
        assert result.vega > 0
        assert result.rho > 0

    def test_greek_consistency_put(self):
        """Test numerical consistency of Greeks for put."""
        spot = 100
        strike = 100
        time_to_expiry = 0.25
        volatility = 0.2
        rate = 0.05

        result = black_scholes_greeks(spot, strike, time_to_expiry, volatility, rate, 0, "put")

        assert result.price > 0
        assert -1 < result.delta < 0
        assert result.gamma > 0
        assert result.vega > 0
        assert result.rho < 0


class TestBSResultDataclass:
    """Test BSResult dataclass."""

    def test_bs_result_creation(self):
        """Test creating BSResult."""
        result = black_scholes_greeks(100, 100, 0.25, 0.2, 0.05, 0, "call")

        assert isinstance(result.price, float)
        assert isinstance(result.delta, float)
        assert isinstance(result.gamma, float)
        assert isinstance(result.vega, float)
        assert isinstance(result.theta, float)
        assert isinstance(result.rho, float)

    def test_bs_result_values_reasonable(self):
        """Test that BSResult values are reasonable."""
        result = black_scholes_greeks(100, 100, 1.0, 0.3, 0.05, 0, "call")

        # For 1-year ATM with 30% vol
        assert 10 < result.price < 20  # Reasonable range
        assert 0 < result.delta < 1
        assert 0 < result.gamma < 0.05
        assert 0 < result.vega < 30
        assert -1 < result.theta < 0  # theta usually negative for long calls
        assert 0 < result.rho < 50


class TestKnownValues:
    """Test against known reference values."""

    def test_atm_call_1year_20pct(self):
        """Test ATM call with standard parameters."""
        # S=100, K=100, T=1, sigma=0.2, r=0.05, q=0
        # From standard BS calculators, roughly 10.45
        price = black_scholes(100, 100, 1.0, 0.2, 0.05, 0, "call")
        # Allow some tolerance due to numerical precision
        assert 10 < price < 11

    def test_itm_call(self):
        """Test ITM call."""
        # S=110, K=100, T=0.5, sigma=0.2, r=0.05, q=0
        # Should be higher than ATM
        price = black_scholes(110, 100, 0.5, 0.2, 0.05, 0, "call")
        assert price > 10

    def test_otm_call(self):
        """Test OTM call."""
        # S=90, K=100, T=0.5, sigma=0.2, r=0.05, q=0
        # Should be lower than ATM
        price = black_scholes(90, 100, 0.5, 0.2, 0.05, 0, "call")
        assert price < 5
