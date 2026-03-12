"""
Unit tests for DSL REPL Rich formatters.

Tests Rich terminal output for all DSL query types.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from scripts.option_dsl_repl import OptionDSLREPL


@pytest.mark.skipif(not HAS_RICH, reason="rich library not installed")
class TestDSLFormatters:
    """Test Rich formatters for DSL results."""

    @pytest.fixture
    def repl(self):
        """Create REPL instance with mock store."""
        # Use minimal REPL without KAN store loading for testing
        repl = OptionDSLREPL.__new__(OptionDSLREPL)
        repl.verbose = False
        repl.mode = "dsl"
        repl.history = []
        repl.current_regime = "STABLE"
        return repl

    def test_format_price_result(self, repl):
        """Test PRICE formatter."""
        result = {
            "query_type": "PRICE",
            "price": 10.4506,
            "spot": 100.0,
            "strike": 100.0,
            "time_to_expiry": 1.0,
            "option_type": "call",
            "delta": 0.6368,
            "gamma": 0.0099,
            "vega": 39.45,
            "theta": -0.0639,
            "rho": 53.24,
            "volatility_used": 0.2,
            "regime": "STABLE",
        }

        # Should not raise
        repl._format_price_result(result)

    def test_format_regime_current_result(self, repl):
        """Test REGIME_CURRENT formatter."""
        result = {
            "query_type": "REGIME_CURRENT",
            "asset": "silver",
            "current_regime": "STABLE",
            "regime_features": {"ricci_mean_core_60d": 0.1},
        }

        repl._format_regime_current_result(result)
        assert repl.current_regime == "STABLE"

    def test_format_regime_prob_result(self, repl):
        """Test REGIME_PROB formatter."""
        result = {
            "query_type": "REGIME_PROB",
            "from_regime": "STABLE",
            "to_regime": "STRESS",
            "horizon": 0.1,
            "all_probs": {
                "STABLE": 0.6,
                "TRANSITION": 0.2,
                "STRESS": 0.1,
                "RECOVERY": 0.1,
            },
        }

        repl._format_regime_prob_result(result)

    def test_format_covariance_result(self, repl):
        """Test COVARIANCE formatter."""
        result = {
            "query_type": "COVARIANCE",
            "assets": ["silver", "gold"],
            "regime": "STABLE",
            "covariance": {
                "silver_silver": 1.0,
                "silver_gold": 0.5,
                "gold_gold": 1.0,
            },
        }

        repl._format_covariance_result(result)

    def test_format_covariance_empty(self, repl):
        """Test COVARIANCE formatter with empty dict."""
        result = {
            "query_type": "COVARIANCE",
            "assets": ["silver"],
            "regime": "STABLE",
            "covariance": {},
        }

        repl._format_covariance_result(result)

    def test_format_transition_string(self, repl):
        """Test TRANSITION formatter with string stub."""
        result = {
            "query_type": "TRANSITION",
            "asset": "silver",
            "transition_matrix": "Not yet implemented - requires historical data",
        }

        repl._format_transition_result(result)

    def test_format_what_if_result(self, repl):
        """Test WHAT_IF formatter."""
        result = {
            "query_type": "WHAT_IF",
            "to_regime": "STRESS",
            "asset": "silver",
            "analysis": {
                "price": {"current": 10.45, "target": 12.50},
                "delta": {"current": 0.636, "target": 0.650},
            },
        }

        repl._format_what_if_result(result)

    def test_format_surface_result(self, repl):
        """Test SURFACE formatter."""
        result = {
            "query_type": "SURFACE",
            "asset": "silver",
            "regime": "STABLE",
            "strikes": [90, 100, 110],
            "vols": [0.15, 0.18, 0.16],
        }

        repl._format_surface_result(result)

    def test_format_surface_empty(self, repl):
        """Test SURFACE formatter with empty vols."""
        result = {
            "query_type": "SURFACE",
            "asset": "silver",
            "regime": "STABLE",
            "strikes": [],
            "vols": [],
        }

        repl._format_surface_result(result)

    def test_format_explain_result(self, repl):
        """Test EXPLAIN formatter with list features."""
        result = {
            "query_type": "EXPLAIN",
            "regime": "STRESS",
            "features": ["ricci_curvature", "mst_stress", "realized_vol"],
        }

        repl._format_explain_result(result)

    def test_format_outlook_result(self, repl):
        """Test OUTLOOK formatter."""
        result = {
            "query_type": "OUTLOOK",
            "asset": "silver",
            "horizon_days": 10,
            "regime": "STRESS",
            "direction": "lean bearish",
            "outlook_score": -22.5,
            "confidence": 0.41,
            "shock_risk": "MODERATE",
            "flow_regime": "UNSETTLED",
        }

        repl._format_outlook_result(result)

    def test_format_result_dispatcher(self, repl):
        """Test _format_result dispatcher for each verb."""
        test_cases = [
            {
                "query_type": "PRICE",
                "price": 10.45,
                "delta": 0.636,
                "spot": 100.0,
                "strike": 100.0,
                "option_type": "call",
                "time_to_expiry": 1.0,
                "volatility_used": 0.2,
                "regime": "STABLE",
            },
            {
                "query_type": "REGIME_CURRENT",
                "asset": "silver",
                "current_regime": "STABLE",
            },
            {
                "query_type": "REGIME_PROB",
                "all_probs": {"STABLE": 0.5, "TRANSITION": 0.3, "STRESS": 0.1, "RECOVERY": 0.1},
            },
            {
                "query_type": "COVARIANCE",
                "assets": ["silver", "gold"],
                "regime": "STABLE",
                "covariance": {"silver_gold": 0.5},
            },
            {
                "query_type": "TRANSITION",
                "asset": "silver",
                "transition_matrix": "Stub",
            },
            {
                "query_type": "WHAT_IF",
                "analysis": {"price": {"current": 10.0, "target": 11.0}},
                "to_regime": "STRESS",
            },
            {
                "query_type": "SURFACE",
                "asset": "silver",
                "regime": "STABLE",
                "strikes": [100],
                "vols": [0.18],
            },
            {
                "query_type": "EXPLAIN",
                "regime": "STABLE",
                "features": ["ricci_curvature"],
            },
            {
                "query_type": "OUTLOOK",
                "asset": "silver",
                "horizon_days": 10,
                "regime": "STRESS",
                "direction": "lean bearish",
                "outlook_score": -22.5,
                "confidence": 0.41,
                "shock_risk": "MODERATE",
                "flow_regime": "UNSETTLED",
            },
        ]

        for result in test_cases:
            repl._format_result(result)

    def test_dynamic_prompt_colors(self, repl):
        """Test dynamic prompt color changes."""
        # Update regime and check color
        repl.current_regime = "STRESS"
        prompt = repl._get_dynamic_prompt()
        assert "STRESS" in prompt or "red" in prompt or "[" in prompt

        repl.current_regime = "STABLE"
        prompt = repl._get_dynamic_prompt()
        assert "STABLE" in prompt or "green" in prompt or "[" in prompt

    def test_regime_color_mapping(self, repl):
        """Test regime color constants."""
        assert repl.REGIME_COLORS["STABLE"] == "green"
        assert repl.REGIME_COLORS["STRESS"] == "red"
        assert repl.REGIME_COLORS["TRANSITION"] == "yellow"
        assert repl.REGIME_COLORS["RECOVERY"] == "blue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
