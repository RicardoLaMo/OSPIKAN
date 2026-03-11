"""
Tests for the options DSL validator.
"""

import pytest
from src.options.dsl.parser import parse_dsl
from src.options.dsl.validator import DSLValidator, ValidationError
from src.options.dsl.ast_nodes import (
    PriceQuery,
    RegimeCurrentQuery,
    RegimeProbQuery,
    CovarianceQuery,
)


class TestValidatorPriceQuery:
    """Test validation of PRICE queries."""

    def test_valid_price_query(self):
        """Test that valid PRICE query passes validation."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=0.25"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_option_type(self):
        """Test that invalid option type fails validation."""
        dsl = "PRICE option type=straddle S=30 K=32 T=45d sigma=0.25"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid option type"):
            validator.validate(node)

    def test_negative_spot_price(self):
        """Test that negative spot price fails validation."""
        node = PriceQuery(
            option_type="call",
            spot_price=-30.0,
            strike_price=32.0,
            time_to_expiry=45 / 365.0,
            volatility=0.25,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Spot price must be positive"):
            validator.validate(node)

    def test_zero_spot_price(self):
        """Test that zero spot price fails validation."""
        node = PriceQuery(
            option_type="call",
            spot_price=0.0,
            strike_price=32.0,
            time_to_expiry=45 / 365.0,
            volatility=0.25,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Spot price must be positive"):
            validator.validate(node)

    def test_negative_strike_price(self):
        """Test that negative strike price fails validation."""
        node = PriceQuery(
            option_type="call",
            spot_price=30.0,
            strike_price=-32.0,
            time_to_expiry=45 / 365.0,
            volatility=0.25,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Strike price must be positive"):
            validator.validate(node)

    def test_negative_time_to_expiry(self):
        """Test that negative time to expiry fails validation."""
        node = PriceQuery(
            option_type="call",
            spot_price=30.0,
            strike_price=32.0,
            time_to_expiry=-45 / 365.0,
            volatility=0.25,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Time to expiry must be positive"):
            validator.validate(node)

    def test_negative_volatility(self):
        """Test that negative volatility fails validation."""
        node = PriceQuery(
            option_type="call",
            spot_price=30.0,
            strike_price=32.0,
            time_to_expiry=45 / 365.0,
            volatility=-0.25,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Volatility must be non-negative"):
            validator.validate(node)

    def test_kan_regime_volatility(self):
        """Test that kan_regime volatility is accepted."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=kan_regime"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_hist_volatility(self):
        """Test that hist_Nd volatility is accepted."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=hist_20d"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_volatility_string(self):
        """Test that invalid volatility string fails validation."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=invalid_vol"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid volatility spec"):
            validator.validate(node)

    def test_invalid_regime(self):
        """Test that invalid regime fails validation."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=0.25 regime=INVALID"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid regime"):
            validator.validate(node)

    def test_valid_regimes(self):
        """Test that all valid regime names are accepted."""
        for regime in ["STABLE", "TRANSITION", "STRESS", "RECOVERY", "current"]:
            dsl = f"PRICE option type=call S=30 K=32 T=45d sigma=0.25 regime={regime}"
            node = parse_dsl(dsl)
            validator = DSLValidator()
            # Should not raise
            validator.validate(node)

    def test_risk_free_rate_out_of_range(self):
        """Test that unreasonable risk-free rate fails validation."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=0.25 r=1.5"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Risk-free rate out of reasonable range"):
            validator.validate(node)

    def test_dividend_yield_out_of_range(self):
        """Test that dividend yield out of range fails validation."""
        dsl = "PRICE option type=call S=30 K=32 T=45d sigma=0.25 q=1.5"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Dividend yield out of range"):
            validator.validate(node)


class TestValidatorRegimeQuery:
    """Test validation of REGIME queries."""

    def test_valid_regime_current(self):
        """Test valid REGIME current query."""
        dsl = "REGIME current asset=silver"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_asset(self):
        """Test that invalid asset fails validation."""
        dsl = "REGIME current asset=bitcoin"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Unknown asset"):
            validator.validate(node)

    def test_valid_regime_prob(self):
        """Test valid REGIME prob query."""
        dsl = "REGIME prob from=STABLE to=STRESS horizon=10d"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_from_regime(self):
        """Test that invalid from_regime fails validation."""
        dsl = "REGIME prob from=INVALID to=STRESS horizon=10d"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid from_regime"):
            validator.validate(node)

    def test_negative_horizon(self):
        """Test that negative horizon fails validation."""
        node = RegimeProbQuery(
            from_regime="STABLE",
            to_regime="STRESS",
            horizon=-10 / 365.0,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Horizon must be positive"):
            validator.validate(node)


class TestValidatorCovarianceQuery:
    """Test validation of COVARIANCE queries."""

    def test_valid_covariance_query(self):
        """Test valid COVARIANCE query."""
        dsl = "COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_asset_in_list(self):
        """Test that invalid asset in list fails validation."""
        dsl = "COVARIANCE assets=[silver,bitcoin] regime=STRESS window=60d"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Unknown asset"):
            validator.validate(node)

    def test_invalid_regime_covariance(self):
        """Test that invalid regime fails validation."""
        dsl = "COVARIANCE assets=[silver,gold] regime=INVALID window=60d"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid regime"):
            validator.validate(node)

    def test_negative_window(self):
        """Test that negative window fails validation."""
        node = CovarianceQuery(
            assets=["silver"],
            regime="STRESS",
            window=-60 / 365.0,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Window must be positive"):
            validator.validate(node)


class TestValidatorWhatIfQuery:
    """Test validation of WHAT_IF queries."""

    def test_valid_what_if_query(self):
        """Test valid WHAT_IF query."""
        dsl = "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega]"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_to_regime(self):
        """Test that invalid to_regime fails validation."""
        dsl = "WHAT_IF regime_shift to=INVALID asset=silver show=[delta]"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid to_regime"):
            validator.validate(node)

    def test_invalid_show_field(self):
        """Test that invalid show field fails validation."""
        dsl = "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,invalid_field]"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid show field"):
            validator.validate(node)

    def test_negative_spot_in_what_if(self):
        """Test that negative spot price fails validation."""
        from src.options.dsl.ast_nodes import WhatIfQuery
        node = WhatIfQuery(
            to_regime="STRESS",
            asset="silver",
            spot_price=-30.0,
            show=["delta"],
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Spot price must be positive"):
            validator.validate(node)


class TestValidatorExplainQuery:
    """Test validation of EXPLAIN queries."""

    def test_valid_explain_query(self):
        """Test valid EXPLAIN query."""
        dsl = "EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_invalid_feature(self):
        """Test that invalid feature fails validation."""
        dsl = "EXPLAIN regime=STRESS features=[invalid_feature]"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Unknown feature"):
            validator.validate(node)

    def test_explain_all_regimes(self):
        """Test EXPLAIN with 'all' regime."""
        dsl = "EXPLAIN regime=all features=[ricci_mean_core_60d]"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)


class TestValidatorSurfaceQuery:
    """Test validation of SURFACE queries."""

    def test_valid_surface_query(self):
        """Test valid SURFACE query."""
        dsl = "SURFACE vol asset=silver regime=all strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"
        node = parse_dsl(dsl)
        validator = DSLValidator()
        # Should not raise
        validator.validate(node)

    def test_negative_strike(self):
        """Test that negative strike fails validation."""
        from src.options.dsl.ast_nodes import SurfaceQuery
        node = SurfaceQuery(
            asset="silver",
            regime="STRESS",
            strikes=[-0.9, 1.0, 1.1],
            time_to_expiry=30 / 365.0,
        )
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Strikes must be positive"):
            validator.validate(node)

    def test_invalid_vol_type(self):
        """Test that invalid vol_type fails validation."""
        dsl = "SURFACE vol asset=silver regime=STRESS strikes=[1.0] T=30d vol_type=invalid"
        node = parse_dsl(dsl)
        validator = DSLValidator()

        with pytest.raises(ValidationError, match="Invalid vol_type"):
            validator.validate(node)
