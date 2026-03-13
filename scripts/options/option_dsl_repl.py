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
import readline
from difflib import get_close_matches

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.options.dsl.parser import parse_dsl
from src.options.dsl.validator import DSLValidator, ValidationError
from src.options.dsl.executor import DSLExecutor, ExecutionContext
from src.options.kan_store.store import KANKnowledgeStore
from src.options.kan_store.config import KANStoreConfig
from src.options.spikan_outlook import SPIKANOutlookEngine
from src.options.llm.client import OllamaClient
from src.options.llm.nl_to_dsl import NLToDSL
from src.analysis.regimes import geometric_regime_classification, regime_transition_matrix
import numpy as np

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Warning: rich library not installed. Install with: pip install rich")

# Global console instance
console = Console() if HAS_RICH else None


class OptionDSLREPL:
    """Interactive DSL REPL."""

    # Regime color mapping for prompt
    REGIME_COLORS = {
        "STABLE": "green",
        "TRANSITION": "yellow",
        "STRESS": "red",
        "RECOVERY": "blue",
    }

    def __init__(
        self,
        mode: str = "dsl",
        store_path: str = "reports/options/kan_store",
        spikan_path: str | None = None,
        verbose: bool = False,
        config_path: str = "configs/options_dsl.yaml",
    ):
        """
        Initialize REPL.

        Args:
            mode: 'dsl' or 'llm'
            store_path: Path to KAN store checkpoint
            spikan_path: Optional path to SPIKAN checkpoint
            verbose: Print debug info
            config_path: Config file path
        """
        self.mode = mode
        self.verbose = verbose
        self.config = self._load_config(config_path)
        self.validator = DSLValidator()

        # Load or create KAN store
        self.store = self._load_kan_store(store_path)
        self.spikan_engine = self._load_spikan_engine(spikan_path)

        # Initialize executor context
        from src.options.dsl.executor import ExecutionContext
        context = ExecutionContext(kan_store=self.store, spikan_engine=self.spikan_engine)
        self.executor = DSLExecutor(context=context)

        # Initialize LLM if needed
        self.translator = None
        if mode == "llm":
            self._init_llm()

        self.history = []
        self.current_regime = "STABLE"  # For dynamic prompt coloring

        # Setup readline history
        self._setup_readline()

    def _setup_readline(self):
        """Setup readline with history persistence."""
        history_file = Path.home() / ".options_dsl_history"
        if history_file.exists():
            try:
                readline.read_history_file(str(history_file))
            except Exception:
                pass

    def _save_readline_history(self):
        """Save readline history on exit."""
        try:
            history_file = Path.home() / ".options_dsl_history"
            readline.write_history_file(str(history_file))
        except Exception:
            pass

    def _load_config(self, config_path: str) -> dict:
        """Load YAML config."""
        if not Path(config_path).exists():
            if HAS_RICH:
                console.print(f"[yellow]Warning: Config not found at {config_path}, using defaults[/yellow]")
            return {"ollama": {}, "repl": {}}
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_kan_store(self, store_path: str) -> KANKnowledgeStore:
        """Load KAN store from checkpoint directory."""
        store_path = Path(store_path)

        if not store_path.exists():
            if HAS_RICH:
                console.print(f"[red]Error: KAN store checkpoint not found at {store_path}[/red]")
                console.print(f"[cyan]Run: python scripts/train_kan_store.py --checkpoint {store_path}[/cyan]")
            else:
                print(f"Error: KAN store checkpoint not found at {store_path}")
            sys.exit(1)

        # Check for required model files
        required_files = ["vol_surface.pt", "covariance.pt", "transition.pt"]
        missing_files = [f for f in required_files if not (store_path / f).exists()]

        if missing_files:
            if HAS_RICH:
                console.print(f"[red]Error: Missing model files: {missing_files}[/red]")
                console.print(f"[cyan]Run: python scripts/train_kan_store.py --checkpoint {store_path}[/cyan]")
            else:
                print(f"Error: Missing model files: {missing_files}")
            sys.exit(1)

        if HAS_RICH:
            console.print(f"[cyan]Loading KAN store from {store_path}...[/cyan]")
        else:
            print(f"Loading KAN store from {store_path}...")
        store = KANKnowledgeStore()
        store.load(str(store_path))
        store.eval()  # Set to eval mode
        if HAS_RICH:
            console.print("[green]✓ KAN store loaded successfully![/green]")
        else:
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
        if HAS_RICH:
            console.print("[cyan]Checking Ollama health...[/cyan]")
        else:
            print("Checking Ollama health...")
        if not client.health_check():
            if HAS_RICH:
                console.print("[red]Error: Ollama server not available[/red]")
                console.print("[yellow]Start Ollama with: ollama serve[/yellow]")
                console.print(f"[yellow]Pull model with: ollama pull {client.model}[/yellow]")
            else:
                print("Error: Ollama server not available")
            sys.exit(1)

        if HAS_RICH:
            console.print("[green]✓ Ollama connected![/green]")
        else:
            print("Ollama connected!")

        self.translator = NLToDSL(
            client=client,
            max_retries=ollama_config.get("max_retries", 3),
            temperature=ollama_config.get("temperature", 0.5),
        )

    def _load_spikan_engine(self, spikan_path: str | None):
        """Load SPIKAN outlook engine if a checkpoint is available."""
        checkpoint = SPIKANOutlookEngine.discover_checkpoint(spikan_path)
        if not checkpoint:
            if self.verbose and HAS_RICH:
                console.print("[yellow]No SPIKAN checkpoint found. OUTLOOK queries will be unavailable.[/yellow]")
            elif self.verbose:
                print("No SPIKAN checkpoint found. OUTLOOK queries will be unavailable.")
            return None

        try:
            if HAS_RICH:
                console.print(f"[cyan]Loading SPIKAN model from {checkpoint}...[/cyan]")
            engine = SPIKANOutlookEngine(checkpoint)
            if HAS_RICH:
                console.print("[green]✓ SPIKAN outlook engine loaded.[/green]")
            return engine
        except Exception as exc:
            if HAS_RICH:
                console.print(f"[yellow]SPIKAN load skipped: {exc}[/yellow]")
            elif self.verbose:
                print(f"SPIKAN load skipped: {exc}")
            return None

    def _print_startup_banner(self):
        """Print startup banner with Rich formatting."""
        if not HAS_RICH:
            print("\n" + "="*70)
            print("Option Pricing DSL + KAN Knowledge Store - Interactive REPL")
            print("="*70 + "\n")
            return

        banner_text = """
╔════════════════════════════════════════════════════════════════╗
║    ⓞ Option Pricing DSL + KAN Knowledge Store                 ║
║                     Interactive REPL v2.0                      ║
╚════════════════════════════════════════════════════════════════╝
"""
        console.print(banner_text)

        # Mode badge
        mode_badge = f"[bold green]🟢 DSL MODE[/bold green]" if self.mode == "dsl" else f"[bold cyan]🔵 LLM MODE[/bold cyan]"
        kan_status = "[bold green]✓ KAN Store[/bold green]"
        status_line = f"{mode_badge}  {kan_status}"

        console.print(Panel(
            status_line,
            title="Status",
            border_style="blue",
            expand=False,
        ))

        # Quick examples
        examples = """
[bold cyan]Quick Examples:[/bold cyan]

[yellow]QUOTE:[/yellow]     QUOTE call spot=100 strike=105 expiry=30d vol=20%
[yellow]SCENARIO:[/yellow]  SCENARIO asset=silver target=STRESS metrics=[delta,vega,price]
[yellow]OUTLOOK:[/yellow]   OUTLOOK asset=silver horizon=10d backdrop=STRESS

Type [bold]help[/bold] for full documentation · [bold]exit[/bold] to quit
"""
        console.print(Panel(
            examples,
            title="Help",
            border_style="cyan",
            expand=False,
        ))
        console.print()

    def _format_price_result(self, result: dict) -> str:
        """Format PRICE query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        # Create table
        table = Table(title="Option Pricing", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        price = result.get("price", 0.0)
        spot = result.get("spot", 0.0)
        strike = result.get("strike", 0.0)
        option_type = result.get("option_type", "call")

        # Determine if ITM/OTM
        is_itm = (spot > strike) if option_type == "call" else (spot < strike)
        moneyness = spot / strike if strike > 0 else 1.0
        itm_status = "[green]ITM[/green]" if is_itm else "[red]OTM[/red]"

        table.add_row("Option Type", f"{option_type.upper()} {itm_status}")
        table.add_row("Price", f"${price:.6f}")
        table.add_row("Spot / Strike", f"{spot:.2f} / {strike:.2f} (${moneyness:.4f})")

        # Add Greeks if available
        delta = result.get("delta", None)
        if delta is not None:
            delta_color = "green" if delta > 0 else "red"
            table.add_row("Delta", f"[{delta_color}]{delta:.4f}[/{delta_color}]")
            table.add_row("Gamma", f"{result.get('gamma', 0.0):.6f}")
            table.add_row("Vega", f"{result.get('vega', 0.0):.6f}")
            table.add_row("Theta", f"{result.get('theta', 0.0):.6f}")
            table.add_row("Rho", f"{result.get('rho', 0.0):.4f}")

        table.add_row("Time to Expiry", f"{result.get('time_to_expiry', 0.0):.4f} years")
        table.add_row("Volatility", f"{result.get('volatility_used', 0.0):.4f}")
        table.add_row("Regime", result.get("regime", "current"))

        console.print(table)
        self._print_narrative(self._narrative_price(result))
        return ""

    def _format_regime_current_result(self, result: dict) -> str:
        """Format REGIME_CURRENT query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        regime = result.get("current_regime", "STABLE")
        color = self.REGIME_COLORS.get(regime, "white")
        self.current_regime = regime

        # Create regime badge
        regime_text = Text(f"  {regime}  ", style=f"bold white on {color}", justify="center")

        panel_content = f"""
[bold cyan]Asset:[/bold cyan] {result.get('asset', 'N/A')}
[bold cyan]Regime:[/bold cyan] {regime}
"""
        console.print(Panel(
            panel_content.strip(),
            title="Current Regime",
            border_style=color,
            expand=False,
        ))
        self._print_narrative(self._narrative_regime_current(result))
        return ""

    def _format_regime_prob_result(self, result: dict) -> str:
        """Format REGIME_PROB query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        table = Table(title="Transition Probabilities", show_header=True, header_style="bold cyan")
        table.add_column("Target Regime", style="cyan")
        table.add_column("Probability", justify="right")
        table.add_column("Bar", width=20)

        all_probs = result.get("all_probs", {})
        for regime in ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]:
            prob = all_probs.get(regime, 0.0)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            color = self.REGIME_COLORS.get(regime, "white")
            table.add_row(f"[{color}]{regime}[/{color}]", f"{prob:.4f}", bar)

        console.print(table)
        self._print_narrative(self._narrative_regime_prob(result))
        return ""

    def _format_covariance_result(self, result: dict) -> str:
        """Format COVARIANCE query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        assets = result.get("assets", [])
        regime = result.get("regime", "STABLE")
        cov_dict = result.get("covariance", {})

        table = Table(title=f"Covariance Matrix ({regime} regime)", show_header=True, header_style="bold cyan")
        table.add_column("Pair", style="cyan")
        table.add_column("Covariance", justify="right")

        for key, val in cov_dict.items():
            # Color code: positive = green, negative = red
            val_color = "green" if val > 0 else "red"
            table.add_row(key, f"[{val_color}]{val:.6f}[/{val_color}]")

        console.print(table)
        self._print_narrative(self._narrative_covariance(result))
        return ""

    def _format_transition_result(self, result: dict) -> str:
        """Format TRANSITION query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        matrix = result.get("transition_matrix", None)

        # Guard against string stub
        if isinstance(matrix, str):
            console.print(Panel(
                f"[yellow]{matrix}[/yellow]",
                title="Transition Matrix",
                border_style="yellow",
                expand=False,
            ))
        else:
            table = Table(title="Regime Transition Matrix", show_header=True, header_style="bold cyan")
            regimes = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]
            table.add_column("From", style="cyan")
            for regime in regimes:
                table.add_column(regime, justify="right")

            if matrix is not None:
                for i, from_regime in enumerate(regimes):
                    row = [from_regime]
                    for j in range(len(regimes)):
                        val = matrix[i, j] if hasattr(matrix, '__getitem__') else 0
                        row.append(f"{val:.4f}")
                    table.add_row(*row)

            console.print(table)

        return ""

    def _format_what_if_result(self, result: dict) -> str:
        """Format WHAT_IF query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        table = Table(title="Regime Shift Analysis", show_header=True, header_style="bold cyan")
        table.add_column("Field", style="cyan")
        table.add_column("Current", justify="right")
        table.add_column("Target", justify="right")

        analysis = result.get("analysis", {})
        target_regime = result.get("to_regime", "?")

        for field, values in analysis.items():
            if isinstance(values, dict):
                current = values.get("current", "N/A")
                target = values.get("target", "N/A")
                if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    change_pct = ((target - current) / current * 100) if current != 0 else 0
                    change_str = f"{target:.6f} ({change_pct:+.1f}%)"
                    change_color = "green" if change_pct > 0 else "red"
                    table.add_row(field, f"{current:.6f}", f"[{change_color}]{change_str}[/{change_color}]")
                else:
                    table.add_row(field, str(current), str(target))
            else:
                table.add_row(field, "—", str(values))

        console.print(table)
        self._print_narrative(self._narrative_what_if(result))
        return ""

    def _format_surface_result(self, result: dict) -> str:
        """Format SURFACE query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        table = Table(title="Volatility Surface", show_header=True, header_style="bold cyan")
        table.add_column("Strike", justify="center", style="cyan")
        table.add_column("Moneyness", justify="center")
        table.add_column("Vol", justify="right")
        table.add_column("Sparkline", width=15)

        strikes = result.get("strikes", [])
        vols = result.get("vols", [])
        asset = result.get("asset", "?")
        regime = result.get("regime", "STABLE")

        if vols:
            max_vol = max(vols) if vols else 1.0
            sparkline_chars = "▁▂▃▄▅▆▇█"
            for i, (strike, vol) in enumerate(zip(strikes, vols)):
                moneyness = strike  # Assuming normalized
                sparkline_idx = min(7, int(vol / max_vol * 8)) if max_vol > 0 else 0
                sparkline = sparkline_chars[sparkline_idx]
                table.add_row(f"{strike:.3f}", f"{moneyness:.3f}", f"{vol:.4f}", sparkline)

        console.print(table)
        self._print_narrative(self._narrative_surface(result))
        return ""

    def _format_explain_result(self, result: dict) -> str:
        """Format EXPLAIN query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        regime = result.get("regime", "?")
        features = result.get("features", [])

        color = self.REGIME_COLORS.get(regime, "white")
        content = f"[bold cyan]Features:[/bold cyan]\n"
        if isinstance(features, list):
            for feature in features:
                content += f"  • {feature}\n"
        else:
            for feature, value in features.items():
                content += f"  • {feature}: {value}\n"

        console.print(Panel(
            content.strip(),
            title=f"Regime: {regime}",
            border_style=color,
            expand=False,
        ))
        self._print_narrative(self._narrative_explain(result))
        return ""

    def _format_outlook_result(self, result: dict) -> str:
        """Format OUTLOOK query with Rich."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        table = Table(title="Market Outlook", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        score = result.get("outlook_score", 0.0)
        confidence = result.get("confidence", 0.0)
        score_color = "green" if score > 15 else "red" if score < -15 else "yellow"
        risk_color = {
            "LOW": "green",
            "MODERATE": "yellow",
            "HIGH": "red",
        }.get(result.get("shock_risk", "LOW"), "white")

        table.add_row("Asset", str(result.get("asset", "N/A")))
        table.add_row("Backdrop", str(result.get("regime", "current")))
        table.add_row("Horizon", f"{result.get('horizon_days', 0)} days")
        table.add_row("Direction", f"[{score_color}]{result.get('direction', 'neutral')}[/{score_color}]")
        table.add_row("Outlook Score", f"[{score_color}]{score:+.1f}[/{score_color}]")
        table.add_row("Shock Risk", f"[{risk_color}]{result.get('shock_risk', 'LOW')}[/{risk_color}]")
        table.add_row("Flow State", str(result.get("flow_regime", "RANGE_BOUND")))
        table.add_row("Confidence", f"{confidence:.0%}")

        console.print(table)
        self._print_narrative(self._narrative_outlook(result))
        return ""

    # ------------------------------------------------------------------
    # Narrative generators — plain-English interpretation of KAN results
    # ------------------------------------------------------------------

    def _narrative_price(self, result: dict) -> str:
        """Build narrative for PRICE result."""
        lines = []
        price    = result.get("price", 0.0)
        spot     = result.get("spot", 0.0)
        strike   = result.get("strike", 0.0)
        otype    = result.get("option_type", "call").lower()
        T        = result.get("time_to_expiry", 0.0)
        sigma    = result.get("volatility_used", 0.0)
        regime   = result.get("regime", "current")
        delta    = result.get("delta")
        gamma    = result.get("gamma")
        vega     = result.get("vega")
        theta    = result.get("theta")

        # Moneyness characterisation
        moneyness = spot / strike if strike else 1.0
        if abs(moneyness - 1.0) < 0.01:
            money_desc = "at-the-money (ATM)"
        elif otype == "call":
            money_desc = f"{'deep ' if abs(moneyness-1) > 0.05 else ''}{'in' if spot > strike else 'out-of'}-the-money"
        else:
            money_desc = f"{'deep ' if abs(moneyness-1) > 0.05 else ''}{'in' if spot < strike else 'out-of'}-the-money"

        T_days = int(T * 365)
        lines.append(
            f"This is a [bold]{money_desc}[/bold] {otype} option with [bold]{T_days} days[/bold] "
            f"to expiry, priced at [bold cyan]${price:.4f}[/bold cyan] using "
            f"a KAN-adjusted vol of [bold]{sigma:.1%}[/bold] under the "
            f"[bold]{regime}[/bold] regime."
        )

        if delta is not None:
            dollar_delta = abs(delta) * spot
            lines.append(
                f"[bold]Delta ({delta:.3f}):[/bold] A $1 move in the underlying "
                f"changes the option value by ~${abs(delta):.2f}. "
                f"The position has the directional exposure of [bold]{dollar_delta:.1f} notional units[/bold]."
            )

        if gamma is not None:
            lines.append(
                f"[bold]Gamma ({gamma:.5f}):[/bold] Delta itself shifts by "
                f"[bold]{gamma:.5f}[/bold] per $1 move — "
                + ("[yellow]low curvature, delta is relatively stable[/yellow]." if gamma < 0.005
                   else "[green]significant curvature, gamma scalping viable near expiry[/green].")
            )

        if vega is not None:
            lines.append(
                f"[bold]Vega ({vega:.4f}):[/bold] A 1 pp rise in implied vol "
                f"adds [bold cyan]${vega/100:.4f}[/bold cyan] to the option price. "
                + ("[green]Elevated vol sensitivity — consider vega hedging.[/green]" if vega > 20
                   else "Vol sensitivity is modest.")
            )

        if theta is not None:
            daily_decay = abs(theta) / 365
            lines.append(
                f"[bold]Theta ({theta:.5f}):[/bold] Time decay costs "
                f"[bold red]${daily_decay:.4f}/day[/bold red]. "
                + (f"Over the remaining {T_days} days that totals "
                   f"~${daily_decay*T_days:.2f} in carry cost." if T_days > 0 else "")
            )

        # Regime colour commentary
        regime_comments = {
            "STABLE":     "The KAN vol surface under STABLE regime prices in lower tail risk — premiums are fair-value.",
            "TRANSITION": "Vol is elevated due to regime uncertainty; the KAN store has widened the smile.",
            "STRESS":     "[red]STRESS regime: KAN vol surface applies a significant risk premium. Vega and skew are both elevated.[/red]",
            "RECOVERY":   "Recovery regime: vol is declining from stress peak but the KAN store still carries a residual risk premium.",
        }
        if regime in regime_comments:
            lines.append(regime_comments[regime])

        return "\n".join(lines)

    def _narrative_regime_current(self, result: dict) -> str:
        """Build narrative for REGIME_CURRENT result."""
        regime = result.get("current_regime", "STABLE")
        asset  = result.get("asset", "the asset")
        descriptions = {
            "STABLE": (
                f"[green]{asset.capitalize()} is in a [bold]STABLE[/bold] regime.[/green] "
                "Network topology is coherent: Ricci curvature is positive, MST stress is low, "
                "and cross-asset correlations are holding steady. "
                "The geometric structure of the market resembles a well-connected, low-stress graph. "
                "Options are priced with lower implied vol premiums — "
                "a favourable environment for theta-positive and short-vol strategies."
            ),
            "TRANSITION": (
                f"[yellow]{asset.capitalize()} is in a [bold]TRANSITION[/bold] regime.[/yellow] "
                "The correlation network is reorganising: curvature is near-zero or mildly negative, "
                "and the MST is undergoing structural changes. "
                "Vol is repricing between the stable and stress levels. "
                "Directional bets carry elevated regime-flip risk — "
                "consider delta-hedging and reducing position size until regime stabilises."
            ),
            "STRESS": (
                f"[red]{asset.capitalize()} is in a [bold]STRESS[/bold] regime.[/red] "
                "Systemic stress is elevated: Forman-Ricci curvature is significantly negative, "
                "MST stress is high, and cross-asset co-movement is rising sharply. "
                "The geometric network is contracting toward a highly connected, fragile state "
                "— a classic pre-crisis topology. "
                "Expect wider bid/ask spreads, elevated skew, and higher implied vol across all strikes. "
                "[bold red]Caution: long gamma and protective put positions are recommended.[/bold red]"
            ),
            "RECOVERY": (
                f"[blue]{asset.capitalize()} is in a [bold]RECOVERY[/bold] regime.[/blue] "
                "Market stress is declining from its peak: curvature is recovering toward zero, "
                "and MST stress is falling. Vol is decreasing but the KAN surface still "
                "carries a residual risk premium from the prior stress episode. "
                "The regime is fragile — a positive environment for risk-on positioning, "
                "but watch curvature metrics for signs of relapse."
            ),
        }
        return descriptions.get(regime, f"Regime: {regime}.")

    def _narrative_regime_prob(self, result: dict) -> str:
        """Build narrative for REGIME_PROB result."""
        all_probs    = result.get("all_probs", {})
        horizon_days = result.get("horizon_days", 0)
        from_regime  = result.get("from_regime", "current")
        to_regime    = result.get("to_regime", "?")
        target_prob  = result.get("target_prob", all_probs.get(to_regime, 0.0))

        if not all_probs:
            return "No probability data available."

        most_likely   = max(all_probs, key=all_probs.get)
        most_prob     = all_probs[most_likely]
        stress_prob   = all_probs.get("STRESS", 0.0)
        stable_prob   = all_probs.get("STABLE", 0.0)

        lines = []
        lines.append(
            f"Over the next [bold]{horizon_days} day{'s' if horizon_days != 1 else ''}[/bold], "
            f"the KAN transition network estimates [bold]{from_regime}[/bold] → "
            f"[bold]{most_likely}[/bold] is the most likely outcome "
            f"([bold cyan]{most_prob:.1%}[/bold cyan])."
        )

        if stress_prob > 0.30:
            lines.append(
                f"[bold red]Warning: {stress_prob:.1%} probability of entering STRESS.[/bold red] "
                "Consider protective positions — long put or long volatility strategies."
            )
        elif stress_prob > 0.15:
            lines.append(
                f"[yellow]Moderate tail risk: {stress_prob:.1%} chance of STRESS.[/yellow] "
                "Monitor curvature metrics daily."
            )
        else:
            lines.append(f"Stress tail risk is low at {stress_prob:.1%}.")

        if stable_prob > 0.60:
            lines.append(
                f"High {stable_prob:.1%} probability of remaining in STABLE — "
                "favourable for short-vol and carry strategies."
            )

        lines.append(
            f"Specific target [bold]{to_regime}[/bold] probability: "
            f"[bold cyan]{target_prob:.1%}[/bold cyan]."
        )
        return "\n".join(lines)

    def _narrative_covariance(self, result: dict) -> str:
        """Build narrative for COVARIANCE result."""
        cov_dict = result.get("covariance", {})
        assets   = result.get("assets", [])
        regime   = result.get("regime", "STABLE")

        if not cov_dict or not assets:
            return "No covariance data available."

        lines = []

        # Find most and least correlated off-diagonal pairs
        off_diag = {k: v for k, v in cov_dict.items() if k.split("_")[0] != k.split("_")[-1]}
        if off_diag:
            most_corr   = max(off_diag, key=lambda k: abs(off_diag[k]))
            least_corr  = min(off_diag, key=lambda k: abs(off_diag[k]))
            most_val    = off_diag[most_corr]
            least_val   = off_diag[least_corr]

            lines.append(
                f"Under the [bold]{regime}[/bold] regime, "
                f"[bold cyan]{most_corr.replace('_',' / ')}[/bold cyan] are the most correlated pair "
                f"({most_val:+.4f}). "
                f"[bold]{least_corr.replace('_',' / ')}[/bold] show the weakest co-movement "
                f"({least_val:+.4f}) — the best diversification pair in this regime."
            )

        regime_cov_notes = {
            "STABLE":     "Covariances are structurally low — reflecting uncrowded, diversified market conditions. A long multi-asset portfolio benefits from genuine diversification here.",
            "TRANSITION": "Covariances are drifting upward as the regime shifts — diversification is becoming less effective. Watch for correlation clustering.",
            "STRESS":     "[red]In STRESS, covariances spike as assets co-move in flight-to-safety selling. Diversification benefits collapse — portfolio risk is understated by stable-regime models.[/red]",
            "RECOVERY":   "Covariances are declining from stress peaks but remain above STABLE levels. Diversification is partially recovering.",
        }
        if regime in regime_cov_notes:
            lines.append(regime_cov_notes[regime])

        return "\n".join(lines)

    def _narrative_what_if(self, result: dict) -> str:
        """Build narrative for WHAT_IF result."""
        analysis     = result.get("analysis", {})
        to_regime    = result.get("to_regime", "?")
        asset        = result.get("asset", "the asset")

        if not analysis:
            return "No what-if analysis available."

        lines = []
        largest_change_field = None
        largest_change_pct   = 0.0

        for field, values in analysis.items():
            if isinstance(values, dict):
                current = values.get("current", 0)
                target  = values.get("target", 0)
                if current and isinstance(current, (int, float)):
                    pct = abs((target - current) / current * 100)
                    if pct > largest_change_pct:
                        largest_change_pct   = pct
                        largest_change_field = field

        lines.append(
            f"Shifting [bold]{asset}[/bold] to the [bold]{to_regime}[/bold] regime "
            f"produces the following changes to your position:"
        )

        if largest_change_field:
            vals    = analysis[largest_change_field]
            current = vals.get("current", 0)
            target  = vals.get("target", 0)
            direction = "increases" if target > current else "decreases"
            lines.append(
                f"The most significant impact is on [bold cyan]{largest_change_field}[/bold cyan], "
                f"which {direction} by [bold]{largest_change_pct:.1f}%[/bold] "
                f"({current:.4f} → {target:.4f})."
            )

        regime_shift_notes = {
            "STRESS": (
                "[red]Moving to STRESS: expect higher implied vol, wider skew, "
                "and a meaningful jump in option premiums. Long gamma positions gain; "
                "short gamma positions face increased mark-to-market losses.[/red]"
            ),
            "STABLE": (
                "[green]Moving to STABLE: vol compresses, skew flattens, "
                "and option premiums decline. Short vol and carry strategies benefit.[/green]"
            ),
            "TRANSITION": (
                "[yellow]Moving to TRANSITION: mixed signals — vol is rising "
                "but not yet at peak. Regime direction is uncertain.[/yellow]"
            ),
            "RECOVERY": (
                "[blue]Moving to RECOVERY: vol is declining from stress peak. "
                "Theta strategies begin to look attractive again.[/blue]"
            ),
        }
        if to_regime in regime_shift_notes:
            lines.append(regime_shift_notes[to_regime])

        return "\n".join(lines)

    def _narrative_surface(self, result: dict) -> str:
        """Build narrative for SURFACE result."""
        strikes = result.get("strikes", [])
        vols    = result.get("vols", [])
        regime  = result.get("regime", "STABLE")
        asset   = result.get("asset", "the asset")

        if not vols or not strikes:
            return "No surface data available."

        min_vol  = min(vols)
        max_vol  = max(vols)
        vol_range = max_vol - min_vol
        atm_idx  = min(range(len(strikes)), key=lambda i: abs(strikes[i] - 1.0))
        atm_vol  = vols[atm_idx]

        lines = []
        lines.append(
            f"The KAN vol surface for [bold]{asset}[/bold] under [bold]{regime}[/bold] "
            f"spans [bold cyan]{min_vol:.2%}[/bold cyan] – [bold cyan]{max_vol:.2%}[/bold cyan]. "
            f"ATM vol is [bold]{atm_vol:.2%}[/bold]."
        )

        # Skew detection
        if len(vols) >= 3:
            low_vol  = vols[0]
            high_vol = vols[-1]
            if low_vol > high_vol + 0.005:
                lines.append(
                    "[bold]Negative skew (put skew / smirk):[/bold] "
                    "Downside strikes carry a vol premium over upside — "
                    "the market is pricing in asymmetric downside risk. "
                    "Buying downside puts is relatively expensive; selling upside calls is cheap."
                )
            elif high_vol > low_vol + 0.005:
                lines.append(
                    "[bold]Positive skew:[/bold] "
                    "Upside strikes are priced at a premium — "
                    "the market is anticipating upside supply-side shocks (common in commodities)."
                )
            else:
                lines.append(
                    "[bold]Flat surface:[/bold] "
                    "The smile is approximately flat across strikes — "
                    "no strong directional tail-risk premium. "
                    "Risk reversals near zero."
                )

        if vol_range > 0.05:
            lines.append(
                f"[yellow]Wide vol range ({vol_range:.2%}) suggests active skew trading.[/yellow] "
                "Risk reversals and butterfly spreads are likely to be well-bid."
            )

        regime_surface_notes = {
            "STRESS": "[red]STRESS surface: vol is elevated across all strikes, "
                      "skew is pronounced, and the KAN network has steepened the smile "
                      "to reflect correlation-driven tail risk.[/red]",
            "STABLE": "[green]STABLE surface: vol is near historical lows, "
                      "skew is muted. A good environment for short-premium strategies.[/green]",
        }
        if regime in regime_surface_notes:
            lines.append(regime_surface_notes[regime])

        return "\n".join(lines)

    def _narrative_explain(self, result: dict) -> str:
        """Build narrative for EXPLAIN result."""
        regime   = result.get("regime", "?")
        features = result.get("features", [])

        feature_explanations = {
            "ricci_curvature":        "measures the curvature of the correlation graph — negative values signal network contraction and stress",
            "ricci_mean_core_60d":    "60-day average Forman-Ricci curvature across the core asset network — the primary geometric stress indicator",
            "ricci_min_core_60d":     "minimum edge curvature across the network — a single highly negative edge often precedes contagion",
            "mst_stress":             "Minimum Spanning Tree stress index — how tightly the MST is compressed relative to normal conditions",
            "mst_stress_core_60d":    "60-day MST stress — rising values indicate the correlation network is tightening toward systemic risk",
            "ga_rotor":               "Geometric Algebra rotor magnitude — encodes rotational changes in the asset return manifold",
            "ga_rotor_magnitude_60d": "60-day GA rotor magnitude — large values indicate rapid rotation in the return space, signalling regime change",
            "realized_vol":           "short-term realised volatility — provides the local vol anchor for the KAN vol surface",
            "realized_vol_20d":       "20-day realised vol — the core volatility input; STRESS regime shows values 2–3× higher than STABLE",
            "momentum":               "return momentum — negative in STRESS (sell-off), positive in RECOVERY",
            "momentum_10d":           "10-day price momentum — short-term directional signal feeding the KAN regime classifier",
            "p_regime_0":             "Markov-chain probability of being in regime 0 (STABLE) — high values reinforce STABLE classification",
            "p_regime_1":             "Markov-chain probability of being in regime 1 (TRANSITION/STRESS) — high values signal elevated risk",
        }

        regime_summary = {
            "STABLE":     "STABLE is characterised by positive curvature, low MST stress, and modest realised vol. The network is well-connected and resilient.",
            "TRANSITION": "TRANSITION sees curvature near zero, rising MST stress, and increasing vol. The system is moving between attractors.",
            "STRESS":     "STRESS is defined by strongly negative curvature, peak MST stress, and elevated realised vol. The network is fragile and co-movement is maximal.",
            "RECOVERY":   "RECOVERY shows curvature recovering toward zero, declining MST stress, and vol falling from its peak. Momentum is turning positive.",
        }

        lines = []
        if regime in regime_summary:
            lines.append(regime_summary[regime])

        feature_list = features if isinstance(features, list) else list(features.keys())
        if feature_list:
            lines.append("\n[bold]Feature explanations:[/bold]")
            for feat in feature_list:
                explanation = feature_explanations.get(feat, "custom feature")
                lines.append(f"  • [bold cyan]{feat}[/bold cyan]: {explanation}.")

        return "\n".join(lines)

    def _narrative_outlook(self, result: dict) -> str:
        """Build narrative for OUTLOOK result."""
        asset = result.get("asset", "the asset")
        regime = result.get("regime", "current")
        score = float(result.get("outlook_score", 0.0))
        direction = result.get("direction", "neutral")
        shock_risk = result.get("shock_risk", "LOW")
        flow_regime = result.get("flow_regime", "RANGE_BOUND")
        horizon_days = result.get("horizon_days", 0)
        confidence = float(result.get("confidence", 0.0))

        lines = [
            f"SPIKAN reads [bold]{asset}[/bold] as [bold]{direction}[/bold] over the next "
            f"[bold]{horizon_days} days[/bold] in a [bold]{regime}[/bold] backdrop. "
            f"The outlook score is [bold]{score:+.1f}[/bold] with [bold]{confidence:.0%}[/bold] confidence."
        ]

        if shock_risk == "HIGH":
            lines.append(
                "[red]Shock risk is elevated.[/red] Favor convex hedges, tighter sizing, and avoid leaning on short gamma."
            )
        elif shock_risk == "MODERATE":
            lines.append(
                "[yellow]Shock risk is moderate.[/yellow] Trend-following setups are tradable, but hedges should stay on."
            )
        else:
            lines.append(
                "[green]Shock risk is contained.[/green] The setup is better suited to carry and range-trading structures."
            )

        flow_comments = {
            "TRENDING": "Flow conditions are trending rather than mean-reverting, so directional expressions should hold better.",
            "SHOCK_PRONE": "Flow conditions are unstable and could gap rather than trend smoothly.",
            "UNSETTLED": "Flow conditions are unsettled; expect follow-through to be less reliable.",
            "RANGE_BOUND": "Flow conditions look range-bound, so premium-selling or relative-value trades may fit better.",
        }
        lines.append(flow_comments.get(flow_regime, f"Flow state: {flow_regime}."))
        return "\n".join(lines)

    def _print_narrative(self, narrative: str) -> None:
        """Render a narrative string as a Rich interpretation panel."""
        if not narrative or not HAS_RICH:
            return
        console.print(Panel(
            narrative,
            title="[bold dim]Interpretation[/bold dim]",
            border_style="dim",
            expand=False,
            padding=(0, 1),
        ))

    def _format_result_legacy(self, result: dict) -> str:
        """Legacy text formatter (fallback when Rich not available)."""
        output = []
        query_type = result.get("query_type", "?")
        output.append(f"Query Type: {query_type}")

        if query_type == "PRICE":
            output.append(f"  Price: {result.get('price', 'N/A'):.6f}")
            delta = result.get('delta', 'N/A')
            if delta != 'N/A':
                output.append(f"  Delta: {delta:.4f}")
                output.append(f"  Gamma: {result.get('gamma', 'N/A'):.6f}")
                output.append(f"  Vega: {result.get('vega', 'N/A'):.6f}")
                output.append(f"  Theta: {result.get('theta', 'N/A'):.6f}")
                output.append(f"  Rho: {result.get('rho', 'N/A'):.4f}")

        elif query_type == "REGIME_CURRENT":
            output.append(f"  Current Regime: {result.get('current_regime', 'N/A')}")

        elif query_type == "REGIME_PROB":
            all_probs = result.get("all_probs", {})
            output.append(f"  Transition Probabilities:")
            for regime, prob in all_probs.items():
                output.append(f"    {regime}: {prob:.4f}")

        elif query_type == "COVARIANCE":
            cov = result.get("covariance", {})
            output.append(f"  Covariance ({result.get('regime', 'N/A')} regime):")
            for key, val in cov.items():
                output.append(f"    {key}: {val:.6f}")

        elif query_type == "TRANSITION":
            matrix = result.get("transition_matrix", None)
            if isinstance(matrix, str):
                output.append(f"  {matrix}")
            else:
                output.append(f"  Transition Matrix")

        elif query_type == "WHAT_IF":
            output.append(f"  Regime Shift: → {result.get('to_regime', 'N/A')}")
            analysis = result.get("analysis", {})
            for key, val in analysis.items():
                output.append(f"  {key}: {val}")

        elif query_type == "EXPLAIN":
            output.append(f"  Regime: {result.get('regime', 'N/A')}")
            features = result.get("features", [])
            if features:
                output.append(f"  Features:")
                if isinstance(features, list):
                    for feature in features:
                        output.append(f"    - {feature}")
                else:
                    for feature, value in features.items():
                        output.append(f"    {feature}: {value}")

        elif query_type == "SURFACE":
            output.append(f"  Surface: {result.get('asset', 'N/A')} vol ({result.get('regime', 'N/A')})")
            output.append(f"  Strikes: {result.get('strikes', [])}")
            output.append(f"  Vols: {result.get('vols', [])}")

        elif query_type == "OUTLOOK":
            output.append(f"  Asset: {result.get('asset', 'N/A')}")
            output.append(f"  Direction: {result.get('direction', 'neutral')}")
            output.append(f"  Score: {result.get('outlook_score', 0.0):+.1f}")
            output.append(f"  Shock Risk: {result.get('shock_risk', 'LOW')}")
            output.append(f"  Flow State: {result.get('flow_regime', 'RANGE_BOUND')}")

        return "\n".join(output)

    def _format_result(self, result: dict) -> str:
        """Pretty-print result dict with Rich formatting."""
        if not HAS_RICH:
            return self._format_result_legacy(result)

        query_type = result.get("query_type", "?")

        if query_type == "PRICE":
            self._format_price_result(result)
        elif query_type == "REGIME_CURRENT":
            self._format_regime_current_result(result)
        elif query_type == "REGIME_PROB":
            self._format_regime_prob_result(result)
        elif query_type == "COVARIANCE":
            self._format_covariance_result(result)
        elif query_type == "TRANSITION":
            self._format_transition_result(result)
        elif query_type == "WHAT_IF":
            self._format_what_if_result(result)
        elif query_type == "SURFACE":
            self._format_surface_result(result)
        elif query_type == "EXPLAIN":
            self._format_explain_result(result)
        elif query_type == "OUTLOOK":
            self._format_outlook_result(result)
        else:
            console.print(f"[yellow]Unknown query type: {query_type}[/yellow]")

        return ""

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
  QUOTE call spot=100 strike=105 expiry=30d vol=20%
  REGIME current asset=silver
  RISK basket=[silver,gold,dxy] backdrop=STABLE lookback=60d
  SCENARIO asset=silver target=STRESS metrics=[delta,vega]
  OUTLOOK asset=silver horizon=10d backdrop=STRESS

NL QUERY (LLM mode):
  "Price a call with spot 100, strike 105, 30 days, 20% vol"
  "What's the current regime for silver?"
  "How does silver look over the next two weeks in stress?"

Type 'help dsl' for detailed DSL syntax guide.
"""
        if HAS_RICH:
            console.print(help_text)
        else:
            print(help_text)

    def _print_help_dsl(self):
        """Print DSL syntax help."""
        help_text = """
╔════════════════════════════════════════════════════════════════╗
║                      DSL SYNTAX GUIDE                          ║
╚════════════════════════════════════════════════════════════════╝

Finance-native commands (legacy equivalents still work):

1. QUOTE / PRICE
   QUOTE call spot=<spot> strike=<strike> expiry=<time> vol=<vol>
   Optional: backdrop=<regime> rate=<rate> dividend=<dividend>
   Legacy: PRICE option type=call S=<spot> K=<strike> T=<time> sigma=<vol>
   Example: QUOTE call spot=100 strike=105 expiry=30d vol=20%

2. REGIME CURRENT
   REGIME current asset=<asset>
   Example: REGIME current asset=silver

3. REGIME PROBABILITY / ODDS
   REGIME odds from=<regime> toward=<regime> horizon=<time>
   Legacy: REGIME prob from=<regime> to=<regime> horizon=<time>
   Example: REGIME odds from=STABLE toward=STRESS horizon=10d

4. RISK / COVARIANCE
   RISK basket=[asset1,asset2,...] backdrop=<regime> lookback=<time>
   Legacy: COVARIANCE assets=[asset1,asset2,...] regime=<regime> window=<time>
   Example: RISK basket=[silver,gold,dxy] backdrop=STABLE lookback=60d

5. TRANSITION
   TRANSITION matrix asset=<asset> normalize=true/false
   Example: TRANSITION matrix asset=silver normalize=true

6. SCENARIO / WHAT_IF
   SCENARIO asset=<asset> target=<regime> metrics=[field1,field2,...]
   Optional: spot=<spot> strike=<strike> expiry=<time>
   Legacy: WHAT_IF regime_shift to=<regime> asset=<asset> show=[field1,field2,...]
   Example: SCENARIO asset=silver target=STRESS metrics=[delta,vega,price]

7. EXPLAIN
   EXPLAIN regime=<regime> features=[feature1,feature2,...]
   Example: EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]

8. SURFACE
   SURFACE vol asset=<asset> regime=<regime> strikes=[k1,k2,...] T=<time>
   Example: SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d

9. OUTLOOK
   OUTLOOK asset=<asset> horizon=<time> [backdrop=<regime>]
   Example: OUTLOOK asset=silver horizon=10d backdrop=STRESS

PARAMETERS:
  Time:    30d (days), 4w (weeks), 6m (months), 0.5 (years)
  Regimes: STABLE, TRANSITION, STRESS, RECOVERY, current
  Assets:  silver, gold, dxy
  Vol:     <float> or kan_regime (regime-adjusted)
  Lists:   [item1,item2,item3] (no spaces)

REGIMES: STABLE, TRANSITION, STRESS, RECOVERY
"""
        if HAS_RICH:
            console.print(help_text)
        else:
            print(help_text)

    def _show_history(self):
        """Show command history."""
        if not self.history:
            if HAS_RICH:
                console.print("[yellow]No history yet.[/yellow]")
            else:
                print("No history yet.")
            return

        for i, (cmd, result) in enumerate(self.history, 1):
            cmd_display = cmd[:60] + "..." if len(cmd) > 60 else cmd
            if result.get("success"):
                query_type = result.get('query_type', 'Unknown')
                if HAS_RICH:
                    console.print(f"{i}. [cyan]{cmd_display}[/cyan]")
                    console.print(f"   [green]✓ {query_type}[/green]")
                else:
                    print(f"{i}. {cmd_display}")
                    print(f"   ✓ {query_type}")
            else:
                error = result.get('error', 'Unknown error')
                if HAS_RICH:
                    console.print(f"{i}. [cyan]{cmd_display}[/cyan]")
                    console.print(f"   [red]✗ Error: {error}[/red]")
                else:
                    print(f"{i}. {cmd_display}")
                    print(f"   ✗ Error: {error}")

    def _save_history(self, filepath: str):
        """Save history to file."""
        with open(filepath, "w") as f:
            for cmd, result in self.history:
                f.write(f"# {datetime.now().isoformat()}\n")
                f.write(f"CMD: {cmd}\n")
                f.write(f"RESULT:\n{json.dumps(result, indent=2, default=str)}\n")
                f.write("-" * 60 + "\n")
        if HAS_RICH:
            console.print(f"[green]✓ History saved to {filepath}[/green]")
        else:
            print(f"History saved to {filepath}")

    def _get_dynamic_prompt(self) -> str:
        """Get color-coded prompt based on current regime."""
        if not HAS_RICH:
            return "options> "

        color = self.REGIME_COLORS.get(self.current_regime, "white")
        return f"[bold {color}]options[{self.current_regime}]>[/bold {color}] "

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
            if HAS_RICH:
                console.print(f"[cyan]Current mode: {self.mode.upper()}[/cyan]")
            else:
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
                if HAS_RICH:
                    console.print("[yellow]Usage: save <filepath>[/yellow]")
                else:
                    print("Usage: save <filepath>")
            return
        if user_input.lower() == "clear":
            self.history.clear()
            if HAS_RICH:
                console.print("[green]✓ History cleared.[/green]")
            else:
                print("History cleared.")
            return
        if user_input.lower() in ("exit", "quit"):
            self._save_readline_history()
            if HAS_RICH:
                console.print("[cyan]Goodbye![/cyan]")
            else:
                print("Goodbye!")
            sys.exit(0)

        # Process DSL/NL query
        try:
            if self.mode == "llm":
                dsl_query = self._translate_nl_to_dsl(user_input)
            else:
                dsl_query = user_input

            if self.verbose:
                if HAS_RICH:
                    console.print(f"[dim]DSL: {dsl_query}[/dim]")
                else:
                    print(f"DSL: {dsl_query}")

            result = self._execute_dsl(dsl_query)

            # Store in history
            self.history.append((user_input, result))

            # Print result
            if result.get("success"):
                self._format_result(result)
            else:
                error = result.get('error', 'Unknown error')
                if HAS_RICH:
                    console.print(f"[red]✗ Error: {error}[/red]")
                else:
                    print(f"Error: {error}")

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "query_type": "ERROR",
            }
            self.history.append((user_input, error_result))
            if HAS_RICH:
                console.print(f"[red]✗ Error: {e}[/red]")
            else:
                print(f"Error: {e}")

    def _translate_nl_to_dsl(self, nl_query: str) -> str:
        """Translate NL to DSL."""
        if not self.translator:
            raise RuntimeError("LLM translator not initialized")

        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]Translating to DSL...[/cyan]"),
                transient=True,
            ) as progress:
                progress.add_task("translate", total=None)
                dsl = self.translator.translate(nl_query, verbose=self.verbose)
        else:
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
            error_str = str(e)
            return {
                "success": False,
                "error": error_str,
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
        self._print_startup_banner()

        # Setup tab completion
        self._setup_completer()

        while True:
            try:
                prompt = self._get_dynamic_prompt()
                if HAS_RICH:
                    user_input = input(prompt).strip()
                else:
                    user_input = input("options> ").strip()

                if not user_input:
                    continue

                self.process_command(user_input)
                if HAS_RICH:
                    console.print()
                else:
                    print()

            except KeyboardInterrupt:
                if HAS_RICH:
                    console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
                else:
                    print("\nInterrupted. Type 'exit' to quit.")
            except EOFError:
                self._save_readline_history()
                if HAS_RICH:
                    console.print("[cyan]Goodbye![/cyan]")
                else:
                    print("Goodbye!")
                break

    def _setup_completer(self):
        """Setup readline completer for tab completion."""
        verbs = [
            "QUOTE",
            "PRICE",
            "REGIME",
            "RISK",
            "COVARIANCE",
            "TRANSITION",
            "SCENARIO",
            "WHAT_IF",
            "EXPLAIN",
            "SURFACE",
            "OUTLOOK",
        ]
        regimes = ["STABLE", "TRANSITION", "STRESS", "RECOVERY"]
        assets = ["silver", "gold", "dxy"]
        option_types = ["call", "put"]
        features = list(self.validator.VALID_FEATURES)
        show_fields = list(self.validator.VALID_SHOW_FIELDS)

        all_completions = verbs + regimes + assets + option_types + features + show_fields

        def completer(text, state):
            if state == 0:
                needle = text.upper()
                self.matches = [item for item in all_completions if item.upper().startswith(needle)]
            if state < len(self.matches):
                return self.matches[state]
            return None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")


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
        "--spikan-model",
        default=None,
        help="Optional path to SPIKAN checkpoint for OUTLOOK queries",
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
        spikan_path=args.spikan_model,
        verbose=args.verbose,
        config_path=args.config,
    )
    repl.run()


if __name__ == "__main__":
    main()
