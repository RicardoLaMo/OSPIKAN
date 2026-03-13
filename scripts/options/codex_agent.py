#!/usr/bin/env python
"""
Codex CLI Agent: Autonomous QA tester for Options DSL powered by Claude.

Uses Anthropic SDK to drive DSL testing with a tool-use loop.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.options.dsl.parser import parse_dsl
from src.options.dsl.validator import DSLValidator
from src.options.dsl.executor import DSLExecutor, ExecutionContext
from src.options.kan_store.store import KANKnowledgeStore
import numpy as np

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("Error: anthropic SDK not installed. Install with: pip install anthropic")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

console = Console() if HAS_RICH else None


# System prompt for Claude
SYSTEM_PROMPT = """
You are a rigorous financial software QA agent testing an Options Pricing DSL system.

Your task is to comprehensively test the system's mathematical correctness and consistency.

## Ground Truth & Invariants

**BS Pricing (ATM call baseline):**
- Spot=100, Strike=100, Time=1 year, Volatility=0.2, Rate=0.05
- Expected price: ≈ 10.4506
- Delta: ≈ 0.6368
- Gamma: ≈ 0.0099
- Vega: ≈ 39.45

**Mathematical Invariants:**
1. **Put-Call Parity**: C - P = S·e^(-q·T) - K·e^(-r·T)
   - Test: |C(S,K,T,σ) - P(S,K,T,σ) - (S - K·e^(-r·T))| < 1e-4

2. **Option Greeks (Call)**:
   - Delta ∈ [0, 1] and increasing with S
   - Gamma > 0 and decreases away from ATM
   - Vega > 0 and highest near ATM
   - Theta < 0 (time decay)
   - Call Vega = Put Vega (same underlying)

3. **Regime Transitions**:
   - All 4 regime probabilities sum to 1.0
   - STRESS vol > STABLE vol at all strikes
   - Covariance matrices must be positive semi-definite

4. **Vol Surface**:
   - OTM/ITM skew: strikes farther from ATM may have higher vol
   - All vols > 0 and < 2.0
   - Vol increases with volatility regime

## Testing Strategy

1. Start with basic BS pricing verification
2. Verify put-call parity across regimes
3. Test all 4 regimes give distinct outputs
4. Validate Greeks behavior (monotonicity, bounds)
5. Check vol surface structure
6. Verify transition probabilities

Always end by calling generate_test_report with all findings.
"""

# Scenario prompts
SCENARIO_PROMPTS = {
    "pricing": "Test BS pricing accuracy: ATM call should be ≈10.45, verify put-call parity holds, check prices increase with vol and S.",
    "greeks": "Verify all Greeks: call delta in [0,1], put delta in [-1,0], gamma>0, vega>0, theta<0. Test ATM and OTM cases. Confirm vega and gamma are equal for calls/puts at same strike.",
    "regimes": "Test all 4 regime presets give distinct outputs. Verify STABLE < STRESS vol. Check all transition prob rows sum to 1.0.",
    "vol-surface": "Test vol surface structure: STRESS vols > STABLE vols. Verify OTM skew (0.9 strike ≥ 1.1 strike in STRESS). All vols positive.",
    "full": "Run comprehensive tests: pricing accuracy, put-call parity, Greeks bounds, regime vol ordering, transition probs sum=1, PSD covariance. Collect all findings.",
}


class CodexAgent:
    """Autonomous QA agent powered by Claude."""

    def __init__(self, store_path: str = "reports/options/kan_store", verbose: bool = False):
        """Initialize agent."""
        self.verbose = verbose
        self.client = anthropic.Anthropic()
        self.model = "claude-haiku-4-5-20251001"

        # Load DSL system
        self._load_dsl_system(store_path)

        # Tool definitions
        self.tools = self._define_tools()

    def _load_dsl_system(self, store_path: str):
        """Load KAN store and DSL components."""
        store_path = Path(store_path)
        if not store_path.exists():
            console.print(f"[red]Error: KAN store not found at {store_path}[/red]") if HAS_RICH else print(f"Error: KAN store not found at {store_path}")
            sys.exit(1)

        self.store = KANKnowledgeStore()
        self.store.load(str(store_path))
        self.store.eval()

        self.validator = DSLValidator()
        context = ExecutionContext(kan_store=self.store)
        self.executor = DSLExecutor(context=context)

    def _define_tools(self) -> List[Dict[str, Any]]:
        """Define tool schemas."""
        return [
            {
                "name": "run_dsl_query",
                "description": "Parse, validate, and execute a DSL query. Returns result dict.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dsl_str": {
                            "type": "string",
                            "description": "DSL query string, e.g., 'PRICE option type=call S=100 K=100 T=1y sigma=0.2'"
                        }
                    },
                    "required": ["dsl_str"]
                }
            },
            {
                "name": "list_available_verbs",
                "description": "List all DSL verbs, valid assets, regimes, and features.",
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "name": "get_regime_preset",
                "description": "Get the 8-feature vector for a regime name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "regime": {
                            "type": "string",
                            "enum": ["STABLE", "TRANSITION", "STRESS", "RECOVERY"],
                            "description": "Regime name"
                        }
                    },
                    "required": ["regime"]
                }
            },
            {
                "name": "compare_queries",
                "description": "Run two DSL queries and compare a named field.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "dsl_a": {
                            "type": "string",
                            "description": "First DSL query"
                        },
                        "dsl_b": {
                            "type": "string",
                            "description": "Second DSL query"
                        },
                        "field": {
                            "type": "string",
                            "description": "Field to compare, e.g., 'price' or 'delta'"
                        }
                    },
                    "required": ["dsl_a", "dsl_b", "field"]
                }
            },
            {
                "name": "check_math_property",
                "description": "Verify mathematical invariants (put-call parity, prob sum, PSD, Greeks bounds).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "property_type": {
                            "type": "string",
                            "enum": ["put_call_parity", "prob_sum_one", "psd_covariance", "greek_bounds"],
                            "description": "Invariant to check"
                        },
                        "params": {
                            "type": "object",
                            "description": "Property-specific params, e.g., {'spot': 100, 'strike': 100, 'T': 1, 'sigma': 0.2} for parity"
                        }
                    },
                    "required": ["property_type", "params"]
                }
            },
            {
                "name": "generate_test_report",
                "description": "Generate markdown report from test findings list.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "category": {"type": "string"},
                                    "test": {"type": "string"},
                                    "status": {"type": "string", "enum": ["PASS", "FAIL"]},
                                    "details": {"type": "string"}
                                }
                            },
                            "description": "List of test findings"
                        }
                    },
                    "required": ["findings"]
                }
            }
        ]

    def _run_dsl_query(self, dsl_str: str) -> Dict[str, Any]:
        """Execute DSL query."""
        try:
            node = parse_dsl(dsl_str)
            self.validator.validate(node)
            result = self.executor.execute(node)
            result["success"] = True
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _list_available_verbs(self) -> Dict[str, Any]:
        """List DSL grammar."""
        return {
            "verbs": [
                "PRICE option type=call S=100 K=100 T=1y sigma=0.2",
                "REGIME current asset=silver",
                "REGIME prob from=STABLE to=STRESS horizon=30d",
                "COVARIANCE assets=[silver,gold] regime=STABLE window=60d",
                "TRANSITION matrix asset=silver normalize=true",
                "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega]",
                "EXPLAIN regime=STABLE features=[ricci_curvature,mst_stress]",
                "SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d",
            ],
            "regimes": ["STABLE", "TRANSITION", "STRESS", "RECOVERY"],
            "assets": ["silver", "gold", "dxy"],
            "option_types": ["call", "put"],
            "valid_features": list(self.validator.VALID_FEATURES),
            "valid_show_fields": list(self.validator.VALID_SHOW_FIELDS),
        }

    def _get_regime_preset(self, regime: str) -> Dict[str, float]:
        """Get regime feature vector."""
        regime_map = {
            "STABLE": {
                "ricci_mean_core_60d": 0.1,
                "ricci_min_core_60d": 0.05,
                "mst_stress_core_60d": 0.2,
                "ga_rotor_magnitude_60d": 0.1,
                "realized_vol_20d": 0.15,
                "momentum_10d": 0.0,
                "p_regime_0": 0.8,
                "p_regime_1": 0.2,
            },
            "TRANSITION": {
                "ricci_mean_core_60d": -0.05,
                "ricci_min_core_60d": -0.1,
                "mst_stress_core_60d": 0.5,
                "ga_rotor_magnitude_60d": 0.3,
                "realized_vol_20d": 0.25,
                "momentum_10d": 0.05,
                "p_regime_0": 0.4,
                "p_regime_1": 0.6,
            },
            "STRESS": {
                "ricci_mean_core_60d": -0.3,
                "ricci_min_core_60d": -0.5,
                "mst_stress_core_60d": 0.8,
                "ga_rotor_magnitude_60d": 0.6,
                "realized_vol_20d": 0.40,
                "momentum_10d": -0.2,
                "p_regime_0": 0.1,
                "p_regime_1": 0.9,
            },
            "RECOVERY": {
                "ricci_mean_core_60d": -0.1,
                "ricci_min_core_60d": -0.15,
                "mst_stress_core_60d": 0.4,
                "ga_rotor_magnitude_60d": 0.2,
                "realized_vol_20d": 0.22,
                "momentum_10d": 0.1,
                "p_regime_0": 0.5,
                "p_regime_1": 0.5,
            },
        }
        return regime_map.get(regime, {})

    def _compare_queries(self, dsl_a: str, dsl_b: str, field: str) -> Dict[str, Any]:
        """Compare two queries."""
        result_a = self._run_dsl_query(dsl_a)
        result_b = self._run_dsl_query(dsl_b)

        if not result_a.get("success") or not result_b.get("success"):
            return {"success": False, "error": "One or both queries failed"}

        val_a = result_a.get(field, "N/A")
        val_b = result_b.get(field, "N/A")

        if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
            diff = val_b - val_a
            pct_diff = (diff / val_a * 100) if val_a != 0 else 0
            return {
                "success": True,
                "field": field,
                "value_a": val_a,
                "value_b": val_b,
                "difference": diff,
                "pct_difference": pct_diff,
            }
        else:
            return {
                "success": True,
                "field": field,
                "value_a": val_a,
                "value_b": val_b,
                "comparable": False,
            }

    def _check_math_property(self, property_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check mathematical invariants."""
        if property_type == "put_call_parity":
            # C - P = S·e^(-q·T) - K·e^(-r·T)
            spot = params.get("spot", 100)
            strike = params.get("strike", 100)
            T = params.get("T", 1.0)
            sigma = params.get("sigma", 0.2)
            r = params.get("r", 0.05)
            q = params.get("q", 0.0)

            # Convert T (years) to days for DSL format
            T_days = int(T * 365)
            call_query = f"PRICE option type=call S={spot} K={strike} T={T_days}d sigma={sigma} r={r} q={q}"
            put_query = f"PRICE option type=put S={spot} K={strike} T={T_days}d sigma={sigma} r={r} q={q}"

            call_result = self._run_dsl_query(call_query)
            put_result = self._run_dsl_query(put_query)

            if not call_result.get("success") or not put_result.get("success"):
                return {"success": False, "error": "Pricing failed"}

            C = call_result.get("price", 0)
            P = put_result.get("price", 0)

            lhs = C - P
            rhs = spot * np.exp(-q * T) - strike * np.exp(-r * T)
            error = abs(lhs - rhs)
            passed = error < 1e-4

            return {
                "success": True,
                "property": "put_call_parity",
                "lhs": lhs,
                "rhs": rhs,
                "error": error,
                "passed": passed,
            }

        elif property_type == "prob_sum_one":
            # Transition probs sum to 1
            from_regime = params.get("from_regime", "STABLE")
            horizon = params.get("horizon", 10/365)

            # Convert horizon (years) to days for DSL format
            horizon_days = int(horizon * 365)
            query = f"REGIME prob from={from_regime} to=STABLE horizon={horizon_days}d"
            result = self._run_dsl_query(query)

            if not result.get("success"):
                return {"success": False, "error": "Regime query failed"}

            all_probs = result.get("all_probs", {})
            total = sum(all_probs.values())
            error = abs(total - 1.0)
            passed = error < 1e-4

            return {
                "success": True,
                "property": "prob_sum_one",
                "probabilities": all_probs,
                "total": total,
                "error": error,
                "passed": passed,
            }

        elif property_type == "psd_covariance":
            # Covariance must be PSD
            regime = params.get("regime", "STABLE")
            assets = params.get("assets", ["silver", "gold", "dxy"])

            query = f"COVARIANCE assets={assets} regime={regime} window=60d"
            result = self._run_dsl_query(query)

            if not result.get("success"):
                return {"success": False, "error": "Covariance query failed"}

            cov_dict = result.get("covariance", {})
            # Reconstruct matrix from dict
            n = len(assets)
            cov_matrix = np.eye(n)
            for key, val in cov_dict.items():
                parts = key.split("_")
                if len(parts) == 2:
                    i = assets.index(parts[0]) if parts[0] in assets else 0
                    j = assets.index(parts[1]) if parts[1] in assets else 0
                    if i < n and j < n:
                        cov_matrix[i, j] = val
                        cov_matrix[j, i] = val

            eigenvals = np.linalg.eigvalsh(cov_matrix)
            min_eigenval = eigenvals.min()
            passed = min_eigenval >= -1e-8

            return {
                "success": True,
                "property": "psd_covariance",
                "min_eigenvalue": float(min_eigenval),
                "passed": passed,
            }

        elif property_type == "greek_bounds":
            # Greeks must be in bounds
            spot = params.get("spot", 100)
            strike = params.get("strike", 100)
            T = params.get("T", 1.0)
            sigma = params.get("sigma", 0.2)
            option_type = params.get("type", "call")

            # Convert T (years) to days for DSL format
            T_days = int(T * 365)
            query = f"PRICE option type={option_type} S={spot} K={strike} T={T_days}d sigma={sigma}"
            result = self._run_dsl_query(query)

            if not result.get("success"):
                return {"success": False, "error": "Pricing failed"}

            delta = result.get("delta", 0)
            gamma = result.get("gamma", 0)
            vega = result.get("vega", 0)
            theta = result.get("theta", 0)

            checks = {}
            if option_type == "call":
                checks["delta_in_[0,1]"] = 0 <= delta <= 1
            else:
                checks["delta_in_[-1,0]"] = -1 <= delta <= 0
            checks["gamma_positive"] = gamma > 0
            checks["vega_positive"] = vega > 0
            checks["theta_negative"] = theta < 0

            passed = all(checks.values())

            return {
                "success": True,
                "property": "greek_bounds",
                "checks": checks,
                "passed": passed,
            }

        return {"success": False, "error": "Unknown property type"}

    def _generate_test_report(self, findings: List[Dict[str, Any]]) -> str:
        """Generate markdown report."""
        report = "# Test Report\n\n"

        # Count results
        passed = sum(1 for f in findings if f.get("status") == "PASS")
        failed = sum(1 for f in findings if f.get("status") == "FAIL")
        total = len(findings)

        report += f"**Results:** {passed} / {total} PASSED, {failed} / {total} FAILED\n\n"

        # Categorize findings
        by_category = {}
        for finding in findings:
            cat = finding.get("category", "Other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(finding)

        for category, items in sorted(by_category.items()):
            report += f"## {category}\n\n"
            for item in items:
                status_emoji = "✓" if item.get("status") == "PASS" else "✗"
                report += f"- {status_emoji} **{item.get('test', 'Unknown')}**: {item.get('details', '')}\n"
            report += "\n"

        return report

    def dispatch_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool call."""
        if tool_name == "run_dsl_query":
            return self._run_dsl_query(tool_input["dsl_str"])
        elif tool_name == "list_available_verbs":
            return self._list_available_verbs()
        elif tool_name == "get_regime_preset":
            return self._get_regime_preset(tool_input["regime"])
        elif tool_name == "compare_queries":
            return self._compare_queries(tool_input["dsl_a"], tool_input["dsl_b"], tool_input["field"])
        elif tool_name == "check_math_property":
            return self._check_math_property(tool_input["property_type"], tool_input["params"])
        elif tool_name == "generate_test_report":
            report = self._generate_test_report(tool_input["findings"])
            return {"report": report}
        else:
            return {"error": f"Unknown tool: {tool_name}"}

    def run_scenario(self, scenario: str, goal: Optional[str] = None, max_steps: int = 20) -> str:
        """Run testing scenario."""
        user_goal = goal or SCENARIO_PROMPTS.get(scenario, "Run comprehensive tests")

        if self.verbose and HAS_RICH:
            console.print(f"[cyan]Goal: {user_goal}[/cyan]\n")

        messages = [{"role": "user", "content": user_goal}]
        final_report = ""

        for step in range(max_steps):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=self.tools,
                messages=messages,
            )

            if self.verbose and HAS_RICH:
                console.print(f"[dim]Step {step + 1}: {response.stop_reason}[/dim]")

            # Add assistant response
            messages.append({"role": "assistant", "content": response.content})

            # Check for end
            if response.stop_reason == "end_turn":
                # Extract any final text
                for block in response.content:
                    if hasattr(block, "text"):
                        final_report = block.text
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    if self.verbose and HAS_RICH:
                        console.print(f"[yellow]→ Tool: {block.name}[/yellow]")

                    result = self.dispatch_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })

                    if self.verbose and HAS_RICH:
                        console.print(f"[green]← Result: {json.dumps(result, default=str)[:100]}...[/green]")

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        return final_report

    def run(self, scenario: str = "full", goal: Optional[str] = None, verbose: bool = False, max_steps: int = 20) -> int:
        """Main entry point."""
        self.verbose = verbose

        if HAS_RICH:
            console.print(Panel(f"[bold cyan]Codex Agent - {scenario.upper()} Scenario[/bold cyan]", border_style="blue"))

        # Run scenario
        report = self.run_scenario(scenario, goal, max_steps)

        # Display report
        if HAS_RICH and report:
            try:
                console.print(Markdown(report))
            except Exception:
                console.print(report)
        elif report:
            print(report)

        # Determine exit code
        if "FAIL" in report or "Error" in report:
            return 1
        return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Codex Agent: Claude-powered QA tester")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIO_PROMPTS.keys()) + ["full"],
        default="pricing",
        help="Test scenario",
    )
    parser.add_argument(
        "--goal",
        type=str,
        help="Custom test goal (overrides scenario)",
    )
    parser.add_argument(
        "--store",
        default="reports/options/kan_store",
        help="Path to KAN store",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Max tool-use loop iterations",
    )

    args = parser.parse_args()

    agent = CodexAgent(store_path=args.store, verbose=args.verbose)
    exit_code = agent.run(
        scenario=args.scenario,
        goal=args.goal,
        verbose=args.verbose,
        max_steps=args.max_steps,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
