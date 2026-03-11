"""
Tests for the options DSL parser.
"""

import pytest
from src.options.dsl.parser import parse_dsl
from src.options.dsl.ast_nodes import (
    PriceQuery,
    RegimeCurrentQuery,
    RegimeProbQuery,
    CovarianceQuery,
    TransitionQuery,
    WhatIfQuery,
    ExplainQuery,
    SurfaceQuery,
)


class TestPriceQueryParsing:
    """Test parsing PRICE queries."""

    def test_simple_price_query(self):
        """Test parsing simple PRICE query."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=0.25"
        node = parse_dsl(dsl)

        assert isinstance(node, PriceQuery)
        assert node.option_type == "call"
        assert node.spot_price == 30.0
        assert node.strike_price == 32.0
        assert abs(node.time_to_expiry - 45 / 365.0) < 1e-6
        assert node.volatility == 0.25
        assert node.regime is None
        assert node.risk_free_rate == 0.05

    def test_price_query_with_regime(self):
        """Test PRICE query with regime."""
        dsl = "PRICE option type=put S=50 K=48 T=60d sigma=0.35 regime=STRESS r=0.03"
        node = parse_dsl(dsl)

        assert isinstance(node, PriceQuery)
        assert node.option_type == "put"
        assert node.spot_price == 50.0
        assert node.strike_price == 48.0
        assert node.regime == "STRESS"
        assert node.risk_free_rate == 0.03

    def test_price_query_kan_regime_vol(self):
        """Test PRICE query with kan_regime volatility."""
        dsl = "PRICE option type=call S=100 K=105 T=30d sigma=kan_regime"
        node = parse_dsl(dsl)

        assert isinstance(node, PriceQuery)
        assert node.volatility == "kan_regime"

    def test_price_query_hist_vol(self):
        """Test PRICE query with historical volatility."""
        dsl = "PRICE option type=call S=100 K=105 T=30d sigma=hist_20d"
        node = parse_dsl(dsl)

        assert isinstance(node, PriceQuery)
        assert node.volatility == "hist_20d"

    def test_price_query_with_dividend(self):
        """Test PRICE query with dividend yield."""
        dsl = "PRICE option type=call S=100 K=100 T=365d sigma=0.2 r=0.05 q=0.02"
        node = parse_dsl(dsl)

        assert isinstance(node, PriceQuery)
        assert node.dividend_yield == 0.02

    def test_price_query_time_suffixes(self):
        """Test time suffix parsing (d, w, m)."""
        # Days
        dsl1 = "PRICE option type=call S=100 K=100 T=30d sigma=0.2"
        node1 = parse_dsl(dsl1)
        assert abs(node1.time_to_expiry - 30 / 365.0) < 1e-6

        # Weeks
        dsl2 = "PRICE option type=call S=100 K=100 T=4w sigma=0.2"
        node2 = parse_dsl(dsl2)
        assert abs(node2.time_to_expiry - 28 / 365.0) < 1e-6

        # Months
        dsl3 = "PRICE option type=call S=100 K=100 T=6m sigma=0.2"
        node3 = parse_dsl(dsl3)
        assert abs(node3.time_to_expiry - 180 / 365.0) < 1e-6


class TestRegimeQueryParsing:
    """Test parsing REGIME queries."""

    def test_regime_current_query(self):
        """Test REGIME current query."""
        dsl = "REGIME current asset=silver"
        node = parse_dsl(dsl)

        assert isinstance(node, RegimeCurrentQuery)
        assert node.asset == "silver"

    def test_regime_prob_query(self):
        """Test REGIME prob query."""
        dsl = "REGIME prob from=STABLE to=STRESS horizon=10d"
        node = parse_dsl(dsl)

        assert isinstance(node, RegimeProbQuery)
        assert node.from_regime == "STABLE"
        assert node.to_regime == "STRESS"
        assert abs(node.horizon - 10 / 365.0) < 1e-6

    def test_regime_prob_with_weeks(self):
        """Test REGIME prob with week horizon."""
        dsl = "REGIME prob from=TRANSITION to=RECOVERY horizon=2w"
        node = parse_dsl(dsl)

        assert isinstance(node, RegimeProbQuery)
        assert node.from_regime == "TRANSITION"
        assert node.to_regime == "RECOVERY"
        assert abs(node.horizon - 14 / 365.0) < 1e-6


class TestCovarianceQueryParsing:
    """Test parsing COVARIANCE queries."""

    def test_covariance_query(self):
        """Test COVARIANCE query."""
        dsl = "COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d"
        node = parse_dsl(dsl)

        assert isinstance(node, CovarianceQuery)
        assert node.assets == ["silver", "gold", "dxy"]
        assert node.regime == "STRESS"
        assert abs(node.window - 60 / 365.0) < 1e-6

    def test_covariance_single_asset(self):
        """Test COVARIANCE with single asset."""
        dsl = "COVARIANCE assets=[silver] regime=STABLE window=90d"
        node = parse_dsl(dsl)

        assert isinstance(node, CovarianceQuery)
        assert node.assets == ["silver"]

    def test_covariance_two_assets(self):
        """Test COVARIANCE with two assets."""
        dsl = "COVARIANCE assets=[gold,dxy] regime=TRANSITION window=30d"
        node = parse_dsl(dsl)

        assert isinstance(node, CovarianceQuery)
        assert node.assets == ["gold", "dxy"]


class TestTransitionQueryParsing:
    """Test parsing TRANSITION queries."""

    def test_transition_query(self):
        """Test TRANSITION query."""
        dsl = "TRANSITION matrix asset=silver normalize=true"
        node = parse_dsl(dsl)

        assert isinstance(node, TransitionQuery)
        assert node.asset == "silver"
        assert node.normalize is True

    def test_transition_no_normalize(self):
        """Test TRANSITION without normalization."""
        dsl = "TRANSITION matrix asset=gold normalize=false"
        node = parse_dsl(dsl)

        assert isinstance(node, TransitionQuery)
        assert node.asset == "gold"
        assert node.normalize is False


class TestWhatIfQueryParsing:
    """Test parsing WHAT_IF queries."""

    def test_what_if_basic(self):
        """Test basic WHAT_IF query."""
        dsl = "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega]"
        node = parse_dsl(dsl)

        assert isinstance(node, WhatIfQuery)
        assert node.to_regime == "STRESS"
        assert node.asset == "silver"
        assert node.show == ["delta", "vega"]

    def test_what_if_with_prices(self):
        """Test WHAT_IF with spot and strike prices."""
        dsl = "WHAT_IF regime_shift to=RECOVERY asset=gold S=50 K=52 T=30d show=[price,delta,gamma,theta]"
        node = parse_dsl(dsl)

        assert isinstance(node, WhatIfQuery)
        assert node.spot_price == 50.0
        assert node.strike_price == 52.0
        assert abs(node.time_to_expiry - 30 / 365.0) < 1e-6

    def test_what_if_all_greeks(self):
        """Test WHAT_IF requesting all Greeks."""
        dsl = "WHAT_IF regime_shift to=STABLE asset=dxy show=[delta,gamma,vega,theta,rho,price]"
        node = parse_dsl(dsl)

        assert len(node.show) == 6


class TestExplainQueryParsing:
    """Test parsing EXPLAIN queries."""

    def test_explain_query(self):
        """Test EXPLAIN query."""
        dsl = "EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]"
        node = parse_dsl(dsl)

        assert isinstance(node, ExplainQuery)
        assert node.regime == "TRANSITION"
        assert node.features == ["ricci_curvature", "mst_stress"]

    def test_explain_single_feature(self):
        """Test EXPLAIN with single feature."""
        dsl = "EXPLAIN regime=STRESS features=[ga_rotor]"
        node = parse_dsl(dsl)

        assert isinstance(node, ExplainQuery)
        assert node.regime == "STRESS"
        assert node.features == ["ga_rotor"]

    def test_explain_all_regimes(self):
        """Test EXPLAIN with 'all' regime."""
        dsl = "EXPLAIN regime=all features=[ricci_mean_core_60d,realized_vol_20d]"
        node = parse_dsl(dsl)

        assert isinstance(node, ExplainQuery)
        assert node.regime == "all"


class TestSurfaceQueryParsing:
    """Test parsing SURFACE queries."""

    def test_surface_vol_query(self):
        """Test SURFACE vol query."""
        dsl = "SURFACE vol asset=silver regime=all strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"
        node = parse_dsl(dsl)

        assert isinstance(node, SurfaceQuery)
        assert node.asset == "silver"
        assert node.regime == "all"
        assert node.strikes == [0.9, 0.95, 1.0, 1.05, 1.1]
        assert abs(node.time_to_expiry - 30 / 365.0) < 1e-6

    def test_surface_specific_regime(self):
        """Test SURFACE with specific regime."""
        dsl = "SURFACE vol asset=gold regime=STRESS strikes=[0.8,1.0,1.2] T=60d"
        node = parse_dsl(dsl)

        assert isinstance(node, SurfaceQuery)
        assert node.asset == "gold"
        assert node.regime == "STRESS"

    def test_surface_single_strike(self):
        """Test SURFACE with single strike."""
        dsl = "SURFACE vol asset=dxy regime=STABLE strikes=[1.0] T=45d"
        node = parse_dsl(dsl)

        assert isinstance(node, SurfaceQuery)
        assert node.strikes == [1.0]


class TestParsingErrors:
    """Test parsing error handling."""

    def test_invalid_command(self):
        """Test error on invalid command."""
        dsl = "INVALID asset=silver"
        with pytest.raises(SyntaxError):
            parse_dsl(dsl)

    def test_missing_required_parameter(self):
        """Test error on missing required parameter."""
        dsl = "PRICE option type=call S=30"  # Missing K, T, sigma
        with pytest.raises(SyntaxError):
            parse_dsl(dsl)

    def test_malformed_list(self):
        """Test error on malformed list."""
        dsl = "COVARIANCE assets=silver,gold regime=STABLE window=60d"  # Missing brackets
        with pytest.raises((SyntaxError, ValueError)):
            parse_dsl(dsl)


class TestParameterOrdering:
    """Test that parameter order is flexible."""

    def test_reordered_parameters(self):
        """Test that parameter order doesn't matter."""
        dsl1 = "PRICE option type=call S=30 K=32 T=45d sigma=0.25"
        dsl2 = "PRICE option sigma=0.25 T=45d K=32 S=30 type=call"

        node1 = parse_dsl(dsl1)
        node2 = parse_dsl(dsl2)

        assert node1.spot_price == node2.spot_price
        assert node1.strike_price == node2.strike_price
        assert node1.time_to_expiry == node2.time_to_expiry
        assert node1.volatility == node2.volatility
        assert node1.option_type == node2.option_type

    def test_optional_parameters_omitted(self):
        """Test that optional parameters can be omitted."""
        dsl_full = "PRICE option type=call S=30 K=32 T=45d sigma=0.25 r=0.05"
        dsl_partial = "PRICE option type=call S=30 K=32 T=45d sigma=0.25"

        node_full = parse_dsl(dsl_full)
        node_partial = parse_dsl(dsl_partial)

        # r should use default for partial
        assert node_partial.risk_free_rate == 0.05  # default
        assert node_full.risk_free_rate == 0.05
