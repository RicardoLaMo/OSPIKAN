"""
Unit tests for Codex Agent tools.

Tests the 6 tool functions without requiring Anthropic API calls.
"""

import pytest
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from scripts.codex_agent import CodexAgent
from src.options.dsl.validator import DSLValidator


@pytest.mark.skipif(not HAS_ANTHROPIC, reason="anthropic not installed")
class TestCodexAgentTools:
    """Test Codex Agent tool functions."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        store_path = Path(__file__).parent.parent / "reports" / "options" / "kan_store"
        if not store_path.exists():
            pytest.skip("KAN store not found")

        try:
            return CodexAgent(store_path=str(store_path), verbose=False)
        except Exception as e:
            pytest.skip(f"Could not load KAN store: {e}")

    def test_list_available_verbs(self, agent):
        """Test list_available_verbs tool."""
        result = agent._list_available_verbs()

        assert "verbs" in result
        assert "regimes" in result
        assert "assets" in result
        assert "option_types" in result
        assert "valid_features" in result
        assert "valid_show_fields" in result

        # Check content
        assert len(result["verbs"]) >= 7
        assert set(result["regimes"]) == {"STABLE", "TRANSITION", "STRESS", "RECOVERY"}
        assert "call" in result["option_types"]
        assert "put" in result["option_types"]

    def test_get_regime_preset(self, agent):
        """Test get_regime_preset tool."""
        for regime in ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]:
            features = agent._get_regime_preset(regime)

            assert isinstance(features, dict)
            assert "ricci_mean_core_60d" in features
            assert "mst_stress_core_60d" in features
            assert "realized_vol_20d" in features
            assert len(features) == 8

    def test_get_regime_preset_unknown(self, agent):
        """Test get_regime_preset with unknown regime."""
        features = agent._get_regime_preset("UNKNOWN")
        assert features == {}

    def test_run_dsl_query_pricing(self, agent):
        """Test run_dsl_query with PRICE query."""
        result = agent._run_dsl_query(
            "PRICE option type=call S=100 K=100 T=365d sigma=0.2"
        )

        assert result.get("success")
        assert "price" in result
        assert result["price"] > 0
        assert "delta" in result
        assert 0 <= result["delta"] <= 1

    def test_run_dsl_query_invalid(self, agent):
        """Test run_dsl_query with invalid query."""
        result = agent._run_dsl_query("INVALID query")

        assert not result.get("success")
        assert "error" in result

    def test_check_math_property_put_call_parity(self, agent):
        """Test check_math_property for put-call parity."""
        result = agent._check_math_property(
            "put_call_parity",
            {"spot": 100, "strike": 100, "T": 365/365, "sigma": 0.2, "r": 0.05, "q": 0.0},
        )

        assert result.get("success")
        assert "property" in result
        assert result["property"] == "put_call_parity"
        assert "error" in result
        assert "passed" in result
        # Parity error should be very small
        assert result["error"] < 0.01

    def test_check_math_property_prob_sum(self, agent):
        """Test check_math_property for prob sum."""
        result = agent._check_math_property(
            "prob_sum_one",
            {"from_regime": "STABLE", "horizon": 10/365},
        )

        assert result.get("success")
        assert "property" in result
        assert result["property"] == "prob_sum_one"
        assert "total" in result
        assert "passed" in result
        # Total should sum to ~1.0
        assert abs(result["total"] - 1.0) < 0.01

    def test_check_math_property_greek_bounds(self, agent):
        """Test check_math_property for Greek bounds."""
        result = agent._check_math_property(
            "greek_bounds",
            {"spot": 100, "strike": 100, "T": 365/365, "sigma": 0.2, "type": "call"},
        )

        assert result.get("success")
        assert "property" in result
        assert "checks" in result
        assert "passed" in result
        # Call delta should be in [0,1]
        assert result["checks"].get("delta_in_[0,1]", False)

    def test_check_math_property_psd_covariance(self, agent):
        """Test check_math_property for PSD covariance."""
        result = agent._check_math_property(
            "psd_covariance",
            {"regime": "STABLE", "assets": ["silver", "gold", "dxy"]},
        )

        assert result.get("success")
        assert "property" in result
        assert "min_eigenvalue" in result
        assert "passed" in result
        # Min eigenvalue should be >= 0 for PSD
        assert result["min_eigenvalue"] >= -1e-8

    def test_compare_queries_same(self, agent):
        """Test compare_queries with identical queries."""
        dsl = "PRICE option type=call S=100 K=100 T=365d sigma=0.2"
        result = agent._compare_queries(dsl, dsl, "price")

        assert result.get("success")
        assert "price" in result.get("field", "")
        assert result.get("difference", 0) < 0.001

    def test_compare_queries_different(self, agent):
        """Test compare_queries with different queries."""
        dsl_a = "PRICE option type=call S=100 K=100 T=365d sigma=0.2"
        dsl_b = "PRICE option type=call S=100 K=100 T=365d sigma=0.3"

        result = agent._compare_queries(dsl_a, dsl_b, "price")

        assert result.get("success")
        assert result["value_b"] > result["value_a"]  # Higher vol = higher price

    def test_generate_test_report(self, agent):
        """Test generate_test_report."""
        findings = [
            {"category": "Pricing", "test": "ATM Call", "status": "PASS", "details": "10.45 ≈ expected"},
            {"category": "Greeks", "test": "Delta bounds", "status": "PASS", "details": "0 <= delta <= 1"},
            {"category": "Regimes", "test": "Vol ordering", "status": "FAIL", "details": "STRESS vol < STABLE"},
        ]

        report = agent._generate_test_report(findings)

        assert isinstance(report, str)
        assert "2 / 3 PASSED" in report
        assert "Pricing" in report
        assert "Greeks" in report
        assert "Regimes" in report

    def test_dispatch_tool_run_dsl_query(self, agent):
        """Test dispatch_tool for run_dsl_query."""
        result = agent.dispatch_tool(
            "run_dsl_query",
            {"dsl_str": "PRICE option type=call S=100 K=100 T=365d sigma=0.2"}
        )

        assert result.get("success")
        assert "price" in result

    def test_dispatch_tool_list_available_verbs(self, agent):
        """Test dispatch_tool for list_available_verbs."""
        result = agent.dispatch_tool("list_available_verbs", {})

        assert "verbs" in result
        assert "regimes" in result

    def test_dispatch_tool_get_regime_preset(self, agent):
        """Test dispatch_tool for get_regime_preset."""
        result = agent.dispatch_tool(
            "get_regime_preset",
            {"regime": "STABLE"}
        )

        assert isinstance(result, dict)
        assert "ricci_mean_core_60d" in result

    def test_dispatch_tool_compare_queries(self, agent):
        """Test dispatch_tool for compare_queries."""
        result = agent.dispatch_tool(
            "compare_queries",
            {
                "dsl_a": "PRICE option type=call S=100 K=100 T=365d sigma=0.2",
                "dsl_b": "PRICE option type=call S=100 K=100 T=365d sigma=0.2",
                "field": "price",
            }
        )

        assert result.get("success")

    def test_dispatch_tool_check_math_property(self, agent):
        """Test dispatch_tool for check_math_property."""
        result = agent.dispatch_tool(
            "check_math_property",
            {
                "property_type": "prob_sum_one",
                "params": {"from_regime": "STABLE", "horizon": 10/365},
            }
        )

        assert result.get("success")

    def test_dispatch_tool_generate_test_report(self, agent):
        """Test dispatch_tool for generate_test_report."""
        result = agent.dispatch_tool(
            "generate_test_report",
            {
                "findings": [
                    {"category": "Test", "test": "Example", "status": "PASS", "details": "OK"}
                ]
            }
        )

        assert "report" in result
        assert isinstance(result["report"], str)

    def test_dispatch_tool_unknown(self, agent):
        """Test dispatch_tool with unknown tool."""
        result = agent.dispatch_tool("unknown_tool", {})

        assert "error" in result

    def test_tools_defined(self, agent):
        """Test that all 6 tools are defined."""
        assert len(agent.tools) == 6

        tool_names = {tool["name"] for tool in agent.tools}
        expected = {
            "run_dsl_query",
            "list_available_verbs",
            "get_regime_preset",
            "compare_queries",
            "check_math_property",
            "generate_test_report",
        }
        assert tool_names == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
