#!/usr/bin/env python
"""
Interactive REPL for Option DSL + KAN Knowledge Store.

Supports two modes:
  1. --dsl-mode: Direct DSL command input (default)
  2. --llm-mode: Natural language input via Ollama/Qwen

Usage:
    python scripts/option_dsl_repl.py --store reports/options/kan_store
    python scripts/option_dsl_repl.py --mode llm --store reports/options/kan_store
    python scripts/option_dsl_repl.py --dsl-mode --verbose
"""

import argparse
import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.options.dsl.parser import parse_dsl
from src.options.dsl.validator import DSLValidator, ValidationError
from src.options.dsl.executor import DSLExecutor, ExecutionContext
from src.options.kan_store.store import KANKnowledgeStore
from src.options.kan_store.config import KANStoreConfig
from src.options.llm.client import OllamaClient
from src.options.llm.nl_to_dsl import NLToDSL
from src.analysis.regimes import geometric_regime_classification, regime_transition_matrix
import numpy as np


class OptionDSLREPL:
    """Interactive DSL REPL."""

    def __init__(
        self,
        mode: str = "dsl",
        store_path: str = "reports/options/kan_store",
        verbose: bool = False,
        config_path: str = "configs/options_dsl.yaml",
    ):
        """
        Initialize REPL.

        Args:
            mode: 'dsl' or 'llm'
            store_path: Path to KAN store checkpoint
            verbose: Print debug info
            config_path: Config file path
        """
        self.mode = mode
        self.verbose = verbose
        self.config = self._load_config(config_path)
        self.validator = DSLValidator()

        # Load or create KAN store
        self.store = self._load_kan_store(store_path)

        # Initialize executor context
        from src.options.dsl.executor import ExecutionContext
        context = ExecutionContext(kan_store=self.store)
        self.executor = DSLExecutor(context=context)

        # Initialize LLM if needed
        self.translator = None
        if mode == "llm":
            self._init_llm()

        self.history = []

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config."""
        if not Path(config_path).exists():
            print(f"Warning: Config not found at {config_path}, using defaults")
            return {"ollama": {}, "repl": {}}
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_kan_store(self, store_path: str) -> KANKnowledgeStore:
        """Load KAN store from checkpoint directory."""
        store_path = Path(store_path)

        if not store_path.exists():
            print(f"Error: KAN store checkpoint not found at {store_path}")
            print(f"Run: python scripts/train_kan_store.py --checkpoint {store_path}")
            sys.exit(1)

        # Check for required model files
        required_files = ["vol_surface.pt", "covariance.pt", "transition.pt"]
        missing_files = [f for f in required_files if not (store_path / f).exists()]

        if missing_files:
            print(f"Error: Missing model files: {missing_files}")
            print(f"Run: python scripts/train_kan_store.py --checkpoint {store_path}")
            sys.exit(1)

        print(f"Loading KAN store from {store_path}...")
        store = KANKnowledgeStore()
        store.load(str(store_path))
        store.eval()  # Set to eval mode
        print("KAN store loaded successfully!")

        return store

    def _init_llm(self):
        """Initialize LLM translator."""
        ollama_config = self.config.get("ollama", {})
        client = OllamaClient(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("model", "qwen2.5-coder:7b"),
        )

        # Health check
        print("Checking Ollama health...")
        if not client.health_check():
            print("Error: Ollama server not available")
            print(f"  Start Ollama with: ollama serve")
            print(f"  Pull model with: ollama pull {client.model}")
            sys.exit(1)

        print("Ollama connected!")

        self.translator = NLToDSL(
            client=client,
            max_retries=ollama_config.get("max_retries", 3),
            temperature=ollama_config.get("temperature", 0.5),
        )

    def _format_result(self, result: dict) -> str:
        """Pretty-print result dict."""
        output = []

        query_type = result.get("query_type", "?")
        output.append(f"Query Type: {query_type}")

        if query_type == "PRICE":
            output.append(f"  Price: {result.get('price', 'N/A'):.6f}")
            greeks = result.get("greeks", {})
            if greeks:
                output.append(f"  Delta: {greeks.get('delta', 'N/A'):.4f}")
                output.append(f"  Gamma: {greeks.get('gamma', 'N/A'):.6f}")
                output.append(f"  Vega: {greeks.get('vega', 'N/A'):.6f}")
                output.append(f"  Theta: {greeks.get('theta', 'N/A'):.6f}")
                output.append(f"  Rho: {greeks.get('rho', 'N/A'):.4f}")

        elif query_type == "REGIME":
            if "current_regime" in result:
                output.append(f"  Current Regime: {result['current_regime']}")
            elif "transition_prob" in result:
                probs = result["transition_prob"]
                output.append(f"  Transition Probabilities:")
                for regime, prob in probs.items():
                    output.append(f"    {regime}: {prob:.4f}")

        elif query_type == "COVARIANCE":
            cov = result.get("covariance", None)
            if cov is not None:
                output.append(f"  Covariance Matrix ({cov.shape}):")
                for i, row in enumerate(cov):
                    output.append(f"    {row}")

        elif query_type == "TRANSITION":
            matrix = result.get("transition_matrix", None)
            if matrix is not None:
                output.append(f"  Transition Matrix ({matrix.shape}):")
                for i, row in enumerate(matrix):
                    output.append(f"    {row}")

        elif query_type == "WHAT_IF":
            output.append(f"  Scenario: {result.get('scenario', 'N/A')}")
            results = result.get("results", {})
            for key, val in results.items():
                if isinstance(val, dict):
                    output.append(f"  {key}:")
                    for k, v in val.items():
                        output.append(f"    {k}: {v:.6f}")
                else:
                    output.append(f"  {key}: {val}")

        elif query_type == "EXPLAIN":
            output.append(f"  Regime: {result.get('regime', 'N/A')}")
            features = result.get("features", {})
            if features:
                output.append(f"  Features:")
                for feature, value in features.items():
                    output.append(f"    {feature}: {value:.4f}")

        elif query_type == "SURFACE":
            output.append(f"  Surface Type: {result.get('surface_type', 'N/A')}")
            output.append(f"  Asset: {result.get('asset', 'N/A')}")
            output.append(f"  Regimes: {result.get('regimes', [])}")
            vols = result.get("vols", [])
            strikes = result.get("strikes", [])
            if vols and strikes:
                output.append(f"  Strikes: {strikes}")
                output.append(f"  Vols: {vols}")

        return "\n".join(output)

    def _print_help(self):
        """Print help message."""
        help_text = f"""
╔════════════════════════════════════════════════════════════════╗
║          Option Pricing DSL + KAN Knowledge Store              ║
║                       Interactive REPL                         ║
╚════════════════════════════════════════════════════════════════╝

Current mode: {self.mode.upper()}

COMMANDS:
  help        - Show this help message
  mode        - Show current mode
  history     - Show command history
  save <file> - Save history to file
  clear       - Clear history
  exit        - Exit REPL

DSL QUERY SYNTAX (use 'help dsl' for details):
  PRICE option type=call S=100 K=105 T=30d sigma=0.2
  REGIME current asset=silver
  COVARIANCE assets=[silver,gold,dxy] regime=STABLE window=60d
  WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega]

NL QUERY (LLM mode):
  "Price a call with spot 100, strike 105, 30 days, 20% vol"
  "What's the current regime for silver?"
  "Show vol surface for gold in stress"

Type 'help dsl' for detailed DSL syntax guide.
"""
        print(help_text)

    def _print_help_dsl(self):
        """Print DSL syntax help."""
        help_text = """
╔════════════════════════════════════════════════════════════════╗
║                      DSL SYNTAX GUIDE                          ║
╚════════════════════════════════════════════════════════════════╝

7 Query Types:

1. PRICE OPTION
   PRICE option type=call S=<spot> K=<strike> T=<time> sigma=<vol>
   Optional: regime=<regime> r=<rate> q=<dividend>
   Example: PRICE option type=call S=100 K=105 T=30d sigma=0.2

2. REGIME CURRENT
   REGIME current asset=<asset>
   Example: REGIME current asset=silver

3. REGIME PROBABILITY
   REGIME prob from=<regime> to=<regime> horizon=<time>
   Example: REGIME prob from=STABLE to=STRESS horizon=10d

4. COVARIANCE
   COVARIANCE assets=[asset1,asset2,...] regime=<regime> window=<time>
   Example: COVARIANCE assets=[silver,gold,dxy] regime=STABLE window=60d

5. TRANSITION
   TRANSITION matrix asset=<asset> normalize=true/false
   Example: TRANSITION matrix asset=silver normalize=true

6. WHAT_IF
   WHAT_IF regime_shift to=<regime> asset=<asset> show=[field1,field2,...]
   Optional: S=<spot> K=<strike> T=<time>
   Example: WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,price]

7. EXPLAIN
   EXPLAIN regime=<regime> features=[feature1,feature2,...]
   Example: EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]

8. SURFACE
   SURFACE vol asset=<asset> regime=<regime> strikes=[k1,k2,...] T=<time>
   Example: SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d

PARAMETERS:
  Time:    30d (days), 4w (weeks), 6m (months), 0.5 (years)
  Regimes: STABLE, TRANSITION, STRESS, RECOVERY, current
  Assets:  silver, gold, dxy
  Vol:     <float> or kan_regime (regime-adjusted)
  Lists:   [item1,item2,item3] (no spaces)

REGIMES: STABLE, TRANSITION, STRESS, RECOVERY
"""
        print(help_text)

    def _show_history(self):
        """Show command history."""
        if not self.history:
            print("No history yet.")
            return

        for i, (cmd, result) in enumerate(self.history, 1):
            print(f"{i}. {cmd[:60]}..." if len(cmd) > 60 else f"{i}. {cmd}")
            if result.get("success"):
                print(f"   ✓ {result.get('query_type', 'Unknown')}")
            else:
                print(f"   ✗ Error: {result.get('error', 'Unknown error')}")

    def _save_history(self, filepath: str):
        """Save history to file."""
        with open(filepath, "w") as f:
            for cmd, result in self.history:
                f.write(f"# {datetime.now().isoformat()}\n")
                f.write(f"CMD: {cmd}\n")
                f.write(f"RESULT:\n{json.dumps(result, indent=2, default=str)}\n")
                f.write("-" * 60 + "\n")
        print(f"History saved to {filepath}")

    def process_command(self, user_input: str):
        """Process user input."""
        user_input = user_input.strip()

        # Handle help commands
        if user_input.lower() == "help":
            self._print_help()
            return
        if user_input.lower() == "help dsl":
            self._print_help_dsl()
            return
        if user_input.lower() == "mode":
            print(f"Current mode: {self.mode.upper()}")
            return
        if user_input.lower() == "history":
            self._show_history()
            return
        if user_input.lower().startswith("save"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                self._save_history(parts[1])
            else:
                print("Usage: save <filepath>")
            return
        if user_input.lower() == "clear":
            self.history.clear()
            print("History cleared.")
            return
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        # Process DSL/NL query
        try:
            if self.mode == "llm":
                dsl_query = self._translate_nl_to_dsl(user_input)
            else:
                dsl_query = user_input

            if self.verbose:
                print(f"DSL: {dsl_query}")

            result = self._execute_dsl(dsl_query)

            # Store in history
            self.history.append((user_input, result))

            # Print result
            if result.get("success"):
                print(self._format_result(result))
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "query_type": "ERROR",
            }
            self.history.append((user_input, error_result))
            print(f"Error: {e}")

    def _translate_nl_to_dsl(self, nl_query: str) -> str:
        """Translate NL to DSL."""
        if not self.translator:
            raise RuntimeError("LLM translator not initialized")

        print("Translating to DSL...")
        dsl = self.translator.translate(nl_query, verbose=self.verbose)
        return dsl

    def _execute_dsl(self, dsl_query: str) -> dict:
        """Parse, validate, and execute DSL."""
        try:
            # Parse
            node = parse_dsl(dsl_query)

            # Validate
            self.validator.validate(node)

            # Execute
            result = self.executor.execute(node)

            result["success"] = True
            return result

        except (SyntaxError, ValidationError) as e:
            return {
                "success": False,
                "error": str(e),
                "query_type": "PARSE_ERROR",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query_type": "EXECUTION_ERROR",
            }

    def run(self):
        """Main REPL loop."""
        self._print_help()
        print()

        prompt = self.config.get("repl", {}).get("prompt", "options> ")

        while True:
            try:
                user_input = input(prompt).strip()

                if not user_input:
                    continue

                self.process_command(user_input)
                print()

            except KeyboardInterrupt:
                print("\nInterrupted. Type 'exit' to quit.")
            except EOFError:
                print("\nGoodbye!")
                break


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive Option DSL REPL with KAN Knowledge Store"
    )
    parser.add_argument(
        "--mode",
        choices=["dsl", "llm"],
        default="dsl",
        help="Query mode: dsl (direct DSL) or llm (natural language via Ollama)",
    )
    parser.add_argument(
        "--dsl-mode",
        action="store_const",
        const="dsl",
        dest="mode",
        help="Shorthand for --mode dsl",
    )
    parser.add_argument(
        "--llm-mode",
        action="store_const",
        const="llm",
        dest="mode",
        help="Shorthand for --mode llm",
    )
    parser.add_argument(
        "--store",
        default="reports/options/kan_store",
        help="Path to KAN store checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/options_dsl.yaml",
        help="Config file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print debug info",
    )

    args = parser.parse_args()

    repl = OptionDSLREPL(
        mode=args.mode,
        store_path=args.store,
        verbose=args.verbose,
        config_path=args.config,
    )
    repl.run()


if __name__ == "__main__":
    main()
