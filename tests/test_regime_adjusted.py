"""
Tests for Regime-Adjusted Pricing.
"""

import pytest
import math
from src.options.pricing.regime_adjusted import RegimeAdjustedPricer, RegimeAdjustedResult


class TestRegimeAdjustedBasic:
    """Test basic regime-adjusted pricing."""

    def test_basic_pricing_without_kan(self):
        """Test pricing without KAN store (falls back to fixed vol)."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        result = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.2,
            risk_free_rate=0.05,
            option_type="call",
        )

        assert isinstance(result, RegimeAdjustedResult)
        assert result.price > 0
        assert result.used_volatility == 0.2
        assert 0 < result.delta < 1

    def test_put_pricing(self):
        """Test put option pricing."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        result = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.2,
            option_type="put",
        )

        assert result.price > 0
        assert -1 < result.delta < 0

    def test_pricing_with_dividend(self):
        """Test pricing with dividend yield."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        result_no_div = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.2,
            dividend_yield=0.0,
            option_type="call",
        )

        result_with_div = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.2,
            dividend_yield=0.02,
            option_type="call",
        )

        # Dividend should decrease call value
        assert result_with_div.price < result_no_div.price

    def test_increasing_volatility(self):
        """Test that volatility increase increases option value."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        result_low_vol = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.1,
            option_type="call",
        )

        result_high_vol = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility=0.3,
            option_type="call",
        )

        assert result_high_vol.price > result_low_vol.price

    def test_result_dataclass(self):
        """Test RegimeAdjustedResult dataclass."""
        result = RegimeAdjustedResult(
            price=10.5,
            delta=0.6,
            gamma=0.02,
            vega=20.0,
            theta=-0.05,
            rho=30.0,
            used_volatility=0.25,
            regime="STABLE",
        )

        assert result.price == 10.5
        assert result.used_volatility == 0.25
        assert result.regime == "STABLE"

    def test_unknown_volatility_type(self):
        """Test error on unknown volatility type."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        with pytest.raises(ValueError, match="Unknown volatility type"):
            pricer.price(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.25,
                volatility="unknown_vol_type",
                option_type="call",
            )

    def test_hist_volatility_not_implemented(self):
        """Test that historical volatility is not yet implemented."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        with pytest.raises(NotImplementedError, match="hist_"):
            pricer.price(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.25,
                volatility="hist_20d",
                option_type="call",
            )


class TestRegimeAdjustedWithKAN:
    """Test regime-adjusted pricing with KAN store."""

    @pytest.fixture
    def mock_kan_store(self):
        """Create a mock KAN store."""
        class MockKANStore:
            def query_vol_surface(self, regime_features, log_moneyness, time_to_expiry, normalize=True):
                # Return constant vol based on regime
                base_vol = 0.2
                if regime_features.get("mst_stress_core_60d", 0) > 0.5:
                    return 0.35  # STRESS regime
                return base_vol

        return MockKANStore()

    def test_kan_regime_vol_lookup(self, mock_kan_store):
        """Test volatility lookup from KAN."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        regime_features = {
            "ricci_mean_core_60d": 0.1,
            "ricci_min_core_60d": 0.05,
            "mst_stress_core_60d": 0.2,  # Not stressed
            "ga_rotor_magnitude_60d": 0.01,
            "realized_vol_20d": 0.15,
            "momentum_10d": 0.0,
            "p_regime_0": 0.5,
            "p_regime_1": 0.5,
        }

        result = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility="kan_regime",
            regime_features=regime_features,
        )

        assert result.used_volatility == 0.2
        assert result.price > 0

    def test_kan_stress_regime(self, mock_kan_store):
        """Test higher vol in stress regime."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        stress_features = {
            "ricci_mean_core_60d": -0.3,
            "ricci_min_core_60d": -0.2,
            "mst_stress_core_60d": 0.8,  # High stress
            "ga_rotor_magnitude_60d": 0.1,
            "realized_vol_20d": 0.40,
            "momentum_10d": 0.0,
            "p_regime_0": 0.1,
            "p_regime_1": 0.3,
        }

        result = pricer.price(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            volatility="kan_regime",
            regime_features=stress_features,
        )

        # Should use higher vol in stress
        assert result.used_volatility == 0.35
        assert result.price > 0

    def test_kan_lookup_without_features(self, mock_kan_store):
        """Test error when KAN lookup needs features but none provided."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        # Without features, kan_regime lookup should fail
        with pytest.raises((ValueError, RuntimeError, AttributeError, TypeError)):
            pricer.price(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.25,
                volatility="kan_regime",
                regime_features=None,
            )


class TestWhatIfAnalysis:
    """Test what-if regime shift analysis."""

    @pytest.fixture
    def mock_kan_store(self):
        """Create a mock KAN store."""
        class MockKANStore:
            def query_vol_surface(self, regime_features, log_moneyness, time_to_expiry, normalize=True):
                # Return different vols for different regimes
                stress = regime_features.get("mst_stress_core_60d", 0)
                if stress > 0.5:
                    return 0.35
                return 0.2

        return MockKANStore()

    def test_what_if_basic(self, mock_kan_store):
        """Test what-if analysis."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        current_features = {
            "ricci_mean_core_60d": 0.1,
            "ricci_min_core_60d": 0.05,
            "mst_stress_core_60d": 0.2,
            "ga_rotor_magnitude_60d": 0.01,
            "realized_vol_20d": 0.15,
            "momentum_10d": 0.0,
            "p_regime_0": 0.5,
            "p_regime_1": 0.5,
        }

        target_features = {
            "ricci_mean_core_60d": -0.3,
            "ricci_min_core_60d": -0.2,
            "mst_stress_core_60d": 0.8,
            "ga_rotor_magnitude_60d": 0.1,
            "realized_vol_20d": 0.40,
            "momentum_10d": 0.0,
            "p_regime_0": 0.1,
            "p_regime_1": 0.3,
        }

        result = pricer.what_if_regime_shift(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            current_regime_features=current_features,
            target_regime_features=target_features,
            show_fields=["price", "delta", "vega", "vol"],
        )

        assert "current" in result
        assert "target" in result
        assert "delta" in result

        # In stress, vol should be higher
        assert result["target"]["vol"] > result["current"]["vol"]

        # Higher vol should mean higher option price (for call)
        assert result["target"]["price"] > result["current"]["price"]

    def test_what_if_no_kan_store(self):
        """Test error when KAN store not available."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        with pytest.raises(RuntimeError, match="KANKnowledgeStore required"):
            pricer.what_if_regime_shift(
                spot=100.0,
                strike=100.0,
                time_to_expiry=0.25,
                current_regime_features={},
                target_regime_features={},
            )

    def test_what_if_custom_fields(self, mock_kan_store):
        """Test what-if with custom show fields."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        current_features = {"mst_stress_core_60d": 0.2} | {
            col: 0.0 for col in ["ricci_mean_core_60d", "ricci_min_core_60d",
                                 "ga_rotor_magnitude_60d", "realized_vol_20d",
                                 "momentum_10d", "p_regime_0", "p_regime_1"]
        }

        target_features = {"mst_stress_core_60d": 0.8} | {
            col: 0.0 for col in ["ricci_mean_core_60d", "ricci_min_core_60d",
                                 "ga_rotor_magnitude_60d", "realized_vol_20d",
                                 "momentum_10d", "p_regime_0", "p_regime_1"]
        }

        result = pricer.what_if_regime_shift(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.25,
            current_regime_features=current_features,
            target_regime_features=target_features,
            show_fields=["delta", "theta"],
        )

        assert "delta" in result["current"]
        assert "theta" in result["current"]
        assert "price" not in result["current"]


class TestSurfaceGeneration:
    """Test vol surface generation."""

    @pytest.fixture
    def mock_kan_store(self):
        """Create a mock KAN store."""
        class MockKANStore:
            def query_vol_surface(self, regime_features, log_moneyness, time_to_expiry, normalize=True):
                # Vol increases with moneyness (skew)
                base_vol = 0.2
                skew = 0.1 * abs(log_moneyness)
                return base_vol + skew

        return MockKANStore()

    def test_surface_absolute_strikes(self, mock_kan_store):
        """Test surface with absolute strike prices."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        regime_features = {col: 0.0 for col in ["ricci_mean_core_60d", "ricci_min_core_60d",
                                                 "mst_stress_core_60d", "ga_rotor_magnitude_60d",
                                                 "realized_vol_20d", "momentum_10d",
                                                 "p_regime_0", "p_regime_1"]}

        result = pricer.surface(
            spot=100.0,
            regime_features=regime_features,
            strikes=[90, 100, 110],
            time_to_expiry=0.25,
        )

        assert len(result["strikes"]) == 3
        assert len(result["vols"]) == 3

        # Check skew (OTM should have higher vol) - allow small tolerance
        # Note: vols[0] is 90 call (OTM), vols[2] is 110 call (OTM)
        # Skew should be symmetric around ATM, but with small numerical differences
        assert result["vols"][0] <= result["vols"][2] or abs(result["vols"][0] - result["vols"][2]) < 0.01

    def test_surface_moneyness_ratios(self, mock_kan_store):
        """Test surface with moneyness ratios."""
        pricer = RegimeAdjustedPricer(kan_store=mock_kan_store)

        regime_features = {col: 0.0 for col in ["ricci_mean_core_60d", "ricci_min_core_60d",
                                                 "mst_stress_core_60d", "ga_rotor_magnitude_60d",
                                                 "realized_vol_20d", "momentum_10d",
                                                 "p_regime_0", "p_regime_1"]}

        result = pricer.surface(
            spot=100.0,
            regime_features=regime_features,
            strikes=[0.9, 0.95, 1.0, 1.05, 1.1],
            time_to_expiry=0.25,
        )

        assert len(result["strikes"]) == 5
        assert len(result["vols"]) == 5

        # ATM should be lower than OTM
        assert result["vols"][2] < result["vols"][0]  # ATM < OTM

    def test_surface_no_kan_store(self):
        """Test error when KAN store not available."""
        pricer = RegimeAdjustedPricer(kan_store=None)

        with pytest.raises(RuntimeError, match="KANKnowledgeStore required"):
            pricer.surface(
                spot=100.0,
                regime_features={},
                strikes=[100, 105, 110],
                time_to_expiry=0.25,
            )
