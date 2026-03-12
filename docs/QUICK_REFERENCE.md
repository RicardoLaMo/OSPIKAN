# Phase F Quick Reference Card

## REPL Quick Start

```bash
# Launch REPL
python scripts/option_dsl_repl.py --dsl-mode

# With verbose output
python scripts/option_dsl_repl.py --dsl-mode --verbose
```

## DSL Queries Cheat Sheet

### PRICE - Option Pricing with Greeks
```
PRICE option type=call S=100 K=100 T=365d sigma=0.2
```
Returns: price, delta, gamma, vega, theta, rho

### REGIME - Current Regime
```
REGIME current asset=silver
```
Returns: current_regime, regime_features

### REGIME - Transition Probability
```
REGIME prob from=STABLE to=STRESS horizon=30d
```
Returns: all_probs (dict of 4 regimes)

### COVARIANCE - Asset Correlations
```
COVARIANCE assets=[silver,gold,dxy] regime=STABLE window=60d
```
Returns: covariance dict

### WHAT_IF - Regime Shift Analysis
```
WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,price]
```
Returns: analysis dict with current/target values

### SURFACE - Volatility Surface
```
SURFACE vol asset=silver regime=STABLE strikes=[90,100,110] T=30d
```
Returns: strikes, vols arrays

### EXPLAIN - Regime Characteristics
```
EXPLAIN regime=STRESS features=[ricci_curvature,mst_stress]
```
Returns: feature list

## Built-in Commands

| Command | Purpose |
|---------|---------|
| `help` | Main help |
| `help dsl` | DSL syntax |
| `history` | Show history |
| `save <file>` | Save history |
| `clear` | Clear history |
| `exit` | Quit REPL |

## Tab Completion

Press `[TAB]` for autocomplete:
- Verbs: `PRICE`, `REGIME`, `COVARIANCE`, `WHAT_IF`, `SURFACE`, `EXPLAIN`
- Regimes: `STABLE`, `TRANSITION`, `STRESS`, `RECOVERY`
- Assets: `silver`, `gold`, `dxy`
- Types: `call`, `put`

## Dynamic Prompt Colors

```
options[STABLE]>       # Green
options[STRESS]>       # Red
options[TRANSITION]>   # Yellow
options[RECOVERY]>     # Blue
```

## Time Suffixes

- `d` = days (365d = 1 year)
- `w` = weeks (4w ≈ 1 month)
- `m` = months (6m ≈ 6 months)

## Codex CLI Agent

```bash
# Pricing test
python scripts/codex_agent.py --scenario pricing

# Greeks test
python scripts/codex_agent.py --scenario greeks

# Regimes test
python scripts/codex_agent.py --scenario regimes

# Vol surface test
python scripts/codex_agent.py --scenario vol-surface

# Full comprehensive test
python scripts/codex_agent.py --scenario full

# Verbose output
python scripts/codex_agent.py --scenario full --verbose

# Custom goal
python scripts/codex_agent.py --goal "Check STRESS vol > STABLE vol"
```

## Agent Scenarios

| Scenario | Tests |
|----------|-------|
| `pricing` | BS accuracy, put-call parity, monotonicity |
| `greeks` | Delta/gamma/vega/theta bounds |
| `regimes` | Vol ordering, prob sums |
| `vol-surface` | OTM skew, positive vols |
| `full` | All scenarios combined |

## Unit Tests

```bash
# REPL formatters
pytest tests/test_dsl_formatters.py -v

# Agent tools
pytest tests/test_codex_agent_tools.py -v

# All tests
pytest tests/test_dsl_formatters.py tests/test_codex_agent_tools.py -v
```

## Common Issues

### Rich formatting missing?
```bash
pip install rich>=13.0
```

### Tab completion not working?
```bash
pip install pyreadline  # Windows only
```

### Agent needs API key?
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### DSL syntax errors?

| Error | Fix |
|-------|-----|
| `T=1y` → Parse error | Use `T=365d` |
| `assets=[a, b]` → Parse error | Use `assets=[a,b]` (no spaces) |
| `PRICE type=call` → Error | Use `PRICE option type=call` |

## Parameter Ranges

| Param | Type | Example | Valid Range |
|-------|------|---------|-------------|
| S | float | 100 | > 0 |
| K | float | 100 | > 0 |
| T | time | 365d | > 0 |
| sigma | float | 0.2 | ≥ 0 |
| r | float | 0.05 | [-0.5, 1.0] |
| q | float | 0.02 | [0, 1.0] |
| regime | enum | STABLE | {STABLE, TRANSITION, STRESS, RECOVERY} |
| asset | enum | silver | {silver, gold, dxy} |
| type | enum | call | {call, put} |

## Magic Features (EXPLAIN)

- `ricci_curvature` - Network curvature metric
- `mst_stress` - Minimum spanning tree stress
- `realized_vol` - Historical volatility
- `momentum` - Price momentum
- `p_regime_0`, `p_regime_1` - Markov probabilities

## Show Fields (WHAT_IF)

- `delta`, `gamma`, `vega`, `theta`, `rho` - Greeks
- `vol` - Volatility
- `price` - Option price

## Regime Feature Vector (8D)

```
ricci_mean_core_60d     # Ricci curvature
ricci_min_core_60d      # Min edge curvature
mst_stress_core_60d     # MST stress
ga_rotor_magnitude_60d  # GA rotor
realized_vol_20d        # 20-day vol
momentum_10d            # 10-day momentum
p_regime_0              # Regime 0 prob
p_regime_1              # Regime 1 prob
```

## Mathematical Invariants

### Put-Call Parity
```
C - P = S·e^(-q·T) - K·e^(-r·T)
```

### Probability Sum
```
Σ(P(regime)) = 1.0
```

### Greek Bounds
```
Call: Delta ∈ [0,1], Gamma > 0, Vega > 0, Theta < 0
Put:  Delta ∈ [-1,0], Gamma > 0, Vega > 0, Theta < 0
```

### Covariance
```
Min eigenvalue ≥ 0 (Positive Semi-Definite)
```

## Performance

- REPL startup: 2-3 seconds
- Query execution: 100-500ms
- Agent run: 10-30 seconds
- Agent cost: ~$0.01-0.05 per run

## Files & Paths

```
scripts/option_dsl_repl.py       # REPL
scripts/codex_agent.py            # QA Agent
tests/test_dsl_formatters.py     # REPL tests
tests/test_codex_agent_tools.py  # Agent tests
reports/options/kan_store/        # KAN models
~/.options_dsl_history            # Command history
```

## API Keys

```bash
# Set Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Verify
python -c "import anthropic; print('✓ Ready')"
```

## Exit Codes

- `0`: Success / All tests passed
- `1`: Failure / Some tests failed

---

**Print this page for quick reference!**

*Phase F | Options Pricing DSL | March 2026*
