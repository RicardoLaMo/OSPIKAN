"""
Tests for DSL Executor.
"""

import pytest
from src.options.dsl.parser import parse_dsl
from src.options.dsl.executor import DSLExecutor, ExecutionContext
from src.options.dsl.ast_nodes import (
    PriceQuery,
    RegimeCurrentQuery,
    RegimeProbQuery,
    CovarianceQuery,
)


class TestExecutorBasic:
    """Test basic executor functionality."""

    def test_executor_initialization(self):
        """Test executor initialization."""
        executor = DSLExecutor()
        assert executor.context is not None
        assert executor.pricer is not None

    def test_executor_with_context(self):
        """Test executor with custom context."""
        context = ExecutionContext(current_regime_label="STABLE")
        executor = DSLExecutor(context=context)
        assert executor.context.current_regime_label == "STABLE"

    def test_execute_unknown_query_type(self):
        """Test error on unknown query type."""
        executor = DSLExecutor()

        class UnknownQuery:
            pass

        with pytest.raises(ValueError, match="Unknown query type"):
            executor.execute(UnknownQuery())


class TestExecutorPrice:
    """Test PRICE query execution."""

    def test_execute_price_simple(self):
        """Test simple PRICE execution."""
        dsl = "PRICE option type=call S=100 K=100 T=0.25 sigma=0.2"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["query_type"] == "PRICE"
        assert result["option_type"] == "call"
        assert result["spot"] == 100.0
        assert result["strike"] == 100.0
        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert "vega" in result

    def test_execute_price_put(self):
        """Test PUT option execution."""
        dsl = "PRICE option type=put S=100 K=100 T=0.25 sigma=0.2"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["option_type"] == "put"
        assert result["price"] > 0
        assert -1 < result["delta"] < 0

    def test_execute_price_with_regime(self):
        """Test PRICE with regime specification."""
        dsl = "PRICE option type=call S=100 K=100 T=0.25 sigma=0.2 regime=STRESS"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["regime"] == "STRESS"

    def test_execute_price_with_dividend(self):
        """Test PRICE with dividend yield."""
        dsl = "PRICE option type=call S=100 K=100 T=0.25 sigma=0.2 q=0.02"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["dividend_yield"] == 0.02

    def test_execute_price_itm(self):
        """Test ITM call pricing."""
        dsl = "PRICE option type=call S=110 K=100 T=0.25 sigma=0.2"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        # ITM call should have price > intrinsic value
        assert result["price"] > 10.0
        assert result["delta"] > 0.5

    def test_execute_price_otm(self):
        """Test OTM call pricing."""
        dsl = "PRICE option type=call S=90 K=100 T=0.25 sigma=0.2"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        # OTM call should have low price
        assert 0 < result["price"] < 5.0
        assert result["delta"] < 0.5


class TestExecutorRegime:
    """Test REGIME query execution."""

    def test_execute_regime_current(self):
        """Test REGIME current execution."""
        dsl = "REGIME current asset=silver"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["query_type"] == "REGIME_CURRENT"
        assert result["asset"] == "silver"
        assert "current_regime" in result

    def test_execute_regime_prob_with_mock_store(self):
        """Test REGIME prob execution with mock KAN store."""
        class MockKANStore:
            def query_transition(self, regime_features, horizon, normalize=True):
                import torch
                # Return uniform probabilities
                return torch.tensor([0.25, 0.25, 0.25, 0.25])

        dsl = "REGIME prob from=STABLE to=STRESS horizon=10d"
        node = parse_dsl(dsl)

        context = ExecutionContext(kan_store=MockKANStore())
        executor = DSLExecutor(context=context)

        result = executor.execute(node)

        assert result["query_type"] == "REGIME_PROB"
        assert result["from_regime"] == "STABLE"
        assert result["to_regime"] == "STRESS"
        assert "all_probs" in result
        assert "target_prob" in result


class TestExecutorCovariance:
    """Test COVARIANCE query execution."""

    def test_execute_covariance_with_mock_store(self):
        """Test COVARIANCE execution with mock KAN store."""
        import torch

        class MockKANStore:
            def query_covariance(self, regime_features, normalize=True):
                # Return identity-like covariance
                return torch.eye(6)

        dsl = "COVARIANCE assets=[silver,gold] regime=STABLE window=60d"
        node = parse_dsl(dsl)

        context = ExecutionContext(kan_store=MockKANStore())
        executor = DSLExecutor(context=context)

        result = executor.execute(node)

        assert result["query_type"] == "COVARIANCE"
        assert result["assets"] == ["silver", "gold"]
        assert result["regime"] == "STABLE"
        assert "covariance" in result

    def test_execute_covariance_without_store(self):
        """Test COVARIANCE error without KAN store."""
        dsl = "COVARIANCE assets=[silver,gold] regime=STABLE window=60d"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        with pytest.raises(RuntimeError, match="KANKnowledgeStore required"):
            executor.execute(node)


class TestExecutorTransition:
    """Test TRANSITION query execution."""

    def test_execute_transition(self):
        """Test TRANSITION execution."""
        dsl = "TRANSITION matrix asset=silver normalize=true"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["query_type"] == "TRANSITION"
        assert result["asset"] == "silver"
        assert result["normalize"] is True


class TestExecutorWhatIf:
    """Test WHAT_IF query execution."""

    def test_execute_what_if_with_mock_store(self):
        """Test WHAT_IF execution with mock KAN store."""
        class MockKANStore:
            def query_vol_surface(self, regime_features, log_moneyness, time_to_expiry, normalize=True):
                # Return different vols based on stress
                if regime_features.get("mst_stress_core_60d", 0) > 0.5:
                    return 0.35
                return 0.2

        dsl = "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,price]"
        node = parse_dsl(dsl)

        context = ExecutionContext(kan_store=MockKANStore())
        executor = DSLExecutor(context=context)

        result = executor.execute(node)

        assert result["query_type"] == "WHAT_IF"
        assert result["to_regime"] == "STRESS"
        assert result["asset"] == "silver"
        assert "analysis" in result

    def test_execute_what_if_without_store(self):
        """Test WHAT_IF error without KAN store."""
        dsl = "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega]"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        with pytest.raises(RuntimeError, match="KANKnowledgeStore required"):
            executor.execute(node)

    def test_execute_what_if_with_prices(self):
        """Test WHAT_IF with custom spot and strike."""
        class MockKANStore:
            def query_vol_surface(self, regime_features, log_moneyness, time_to_expiry, normalize=True):
                return 0.2

        dsl = "WHAT_IF regime_shift to=STRESS asset=silver S=110 K=100 T=30d show=[price,delta]"
        node = parse_dsl(dsl)

        context = ExecutionContext(kan_store=MockKANStore())
        executor = DSLExecutor(context=context)

        result = executor.execute(node)

        assert result["spot"] == 110.0
        assert result["strike"] == 100.0


class TestExecutorExplain:
    """Test EXPLAIN query execution."""

    def test_execute_explain(self):
        """Test EXPLAIN execution."""
        dsl = "EXPLAIN regime=STRESS features=[ricci_curvature,mst_stress]"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        assert result["query_type"] == "EXPLAIN"
        assert result["regime"] == "STRESS"
        assert result["features"] == ["ricci_curvature", "mst_stress"]


class TestExecutorSurface:
    """Test SURFACE query execution."""

    def test_execute_surface_with_mock_store(self):
        """Test SURFACE execution with mock KAN store."""
        class MockKANStore:
            def query_vol_surface(self, regime_features, log_moneyness, time_to_expiry, normalize=True):
                # Return vol with skew
                return 0.2 + 0.1 * abs(log_moneyness)

        dsl = "SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d"
        node = parse_dsl(dsl)

        context = ExecutionContext(kan_store=MockKANStore())
        executor = DSLExecutor(context=context)

        result = executor.execute(node)

        assert result["query_type"] == "SURFACE"
        assert result["asset"] == "silver"
        assert result["regime"] == "STABLE"
        assert len(result["strikes"]) == 3
        assert len(result["vols"]) == 3

    def test_execute_surface_without_store(self):
        """Test SURFACE error without KAN store."""
        dsl = "SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        with pytest.raises(RuntimeError, match="KANKnowledgeStore required"):
            executor.execute(node)


class TestExecutorIntegration:
    """Integration tests for complete DSL → Execution flow."""

    def test_end_to_end_price_query(self):
        """Test complete DSL parsing and execution for pricing."""
        dsl = "PRICE option type=call S=100 K=100 T=0.5 sigma=0.25 r=0.03"
        node = parse_dsl(dsl)
        executor = DSLExecutor()

        result = executor.execute(node)

        # Verify result structure
        assert isinstance(result, dict)
        assert result["query_type"] == "PRICE"
        assert result["price"] > 0
        assert "volatility_used" in result
        assert result["volatility_used"] == 0.25
        assert result["risk_free_rate"] == 0.03

    def test_end_to_end_with_time_suffixes(self):
        """Test DSL with various time suffixes."""
        test_cases = [
            ("PRICE option type=call S=100 K=100 T=30d sigma=0.2", 30 / 365.0),
            ("PRICE option type=call S=100 K=100 T=4w sigma=0.2", 28 / 365.0),
            ("PRICE option type=call S=100 K=100 T=6m sigma=0.2", 180 / 365.0),
        ]

        executor = DSLExecutor()

        for dsl, expected_T in test_cases:
            node = parse_dsl(dsl)
            result = executor.execute(node)

            # Check time matches expected
            assert abs(result["time_to_expiry"] - expected_T) < 1e-6

    def test_end_to_end_regime_names(self):
        """Test DSL with various regime names."""
        regimes = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]
        executor = DSLExecutor()

        for regime in regimes:
            dsl = f"PRICE option type=call S=100 K=100 T=30d sigma=0.2 regime={regime}"
            node = parse_dsl(dsl)
            result = executor.execute(node)

            assert result["regime"] == regime
            assert result["price"] > 0
