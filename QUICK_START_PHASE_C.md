# Quick Start: DSL Parsing & Option Pricing (Phase C)

## Basic Usage

### 1. Simple Option Pricing

```python
from src.options.dsl.parser import parse_dsl
from src.options.dsl.executor import DSLExecutor, ExecutionContext

# Parse DSL query
dsl = "PRICE option type=call S=100 K=100 T=0.25 sigma=0.2"
node = parse_dsl(dsl)

# Execute
executor = DSLExecutor()
result = executor.execute(node)

# Results
print(result)
# {
#   'query_type': 'PRICE',
#   'option_type': 'call',
#   'spot': 100.0,
#   'strike': 100.0,
#   'time_to_expiry': 0.25,
#   'volatility_used': 0.2,
#   'price': 2.78,
#   'delta': 0.54,
#   'gamma': 0.019,
#   'vega': 19.8,
#   'theta': -0.009,
#   'rho': 19.4,
# }
```

### 2. Pricing with Regime-Adjusted Volatility

```python
from src.options.kan_store import KANKnowledgeStore
from src.options.dsl.executor import DSLExecutor, ExecutionContext

# Load KAN store (from Phase B)
kan_store = KANKnowledgeStore()
kan_store.load("reports/options/kan_store/")

# Create context with KAN store
context = ExecutionContext(kan_store=kan_store)
executor = DSLExecutor(context=context)

# Parse regime-adjusted pricing
dsl = "PRICE option type=call S=100 K=100 T=0.25 sigma=kan_regime regime=STRESS"
node = parse_dsl(dsl)

# Execute (will use KAN vol surface)
result = executor.execute(node)
print(f"Price: {result['price']}")
print(f"Vol used: {result['volatility_used']}")
```

### 3. What-If Analysis

```python
dsl = "WHAT_IF regime_shift to=STRESS asset=silver show=[price,delta,vega,vol]"
node = parse_dsl(dsl)
result = executor.execute(node)

# Output:
# {
#   'query_type': 'WHAT_IF',
#   'to_regime': 'STRESS',
#   'asset': 'silver',
#   'analysis': {
#     'current': {'price': 2.78, 'delta': 0.54, 'vega': 19.8, 'vol': 0.2},
#     'target': {'price': 4.12, 'delta': 0.61, 'vega': 24.3, 'vol': 0.35},
#     'delta': {'price': 1.34, 'delta': 0.07, 'vega': 4.5, 'vol': 0.15}
#   }
# }
```

### 4. Volatility Surface

```python
dsl = "SURFACE vol asset=silver regime=STABLE strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"
node = parse_dsl(dsl)
result = executor.execute(node)

# Output:
# {
#   'query_type': 'SURFACE',
#   'asset': 'silver',
#   'regime': 'STABLE',
#   'strikes': [90, 95, 100, 105, 110],
#   'vols': [0.22, 0.21, 0.20, 0.21, 0.22]
# }
```

---

## DSL Commands Reference

### PRICE - Option Pricing
```
PRICE option type=call S=30 K=32 T=45d sigma=0.25 [regime=STRESS] [r=0.05] [q=0.02]
```
- `type`: call or put
- `S`: spot price
- `K`: strike price
- `T`: time (30d, 4w, 6m, or 0.25 for years)
- `sigma`: volatility (float, "kan_regime", or "hist_Nd")
- `regime`: STABLE, TRANSITION, STRESS, RECOVERY, or current (optional)
- `r`: risk-free rate (optional, default 0.05)
- `q`: dividend yield (optional, default 0.0)

### REGIME current - Current Regime
```
REGIME current asset=silver
```

### REGIME prob - Transition Probability
```
REGIME prob from=STABLE to=STRESS horizon=10d
```

### COVARIANCE - Regime Covariance
```
COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d
```
Requires KAN store for execution.

### WHAT_IF - Regime Shift Analysis
```
WHAT_IF regime_shift to=STRESS asset=silver [S=100] [K=100] [T=30d] show=[delta,vega,price,vol]
```

### SURFACE - Volatility Surface
```
SURFACE vol asset=silver regime=STABLE strikes=[0.9,0.95,1.0,1.05,1.1] T=30d
```

### EXPLAIN - Regime Characteristics
```
EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress,ga_rotor]
```

---

## Time Formats

- Days: `30d` → 30/365 years
- Weeks: `4w` → 28/365 years
- Months: `6m` → 180/365 years
- Years: `0.5` → 0.5 years (decimal)

---

## Direct Pricer Usage (No DSL)

```python
from src.options.pricing.regime_adjusted import RegimeAdjustedPricer

pricer = RegimeAdjustedPricer(kan_store=None)

# Simple pricing
result = pricer.price(
    spot=100,
    strike=100,
    time_to_expiry=0.25,
    volatility=0.2,
    option_type="call"
)
print(result.price)  # 2.78
print(result.delta)  # 0.54

# What-if analysis
result = pricer.what_if_regime_shift(
    spot=100, strike=100, time_to_expiry=0.25,
    current_regime_features={...},
    target_regime_features={...},
    show_fields=["price", "delta", "vega"]
)

# Vol surface
result = pricer.surface(
    spot=100,
    regime_features={...},
    strikes=[90, 100, 110],
    time_to_expiry=0.25
)
```

---

## Full Example Script

```python
#!/usr/bin/env python3
"""Example: Price a call option with regime adjustment."""

from src.options.dsl.parser import parse_dsl
from src.options.dsl.executor import DSLExecutor, ExecutionContext
from src.options.kan_store import KANKnowledgeStore

# Load KAN store
kan_store = KANKnowledgeStore()
kan_store.load("reports/options/kan_store/")

# Create executor with KAN store
context = ExecutionContext(kan_store=kan_store)
executor = DSLExecutor(context=context)

# Query 1: Basic pricing
print("=== Basic Pricing ===")
dsl1 = "PRICE option type=call S=100 K=100 T=0.25 sigma=0.2"
result1 = executor.execute(parse_dsl(dsl1))
print(f"Call price: ${result1['price']:.2f}")
print(f"Delta: {result1['delta']:.4f}")

# Query 2: Regime-adjusted pricing
print("\n=== Regime-Adjusted Pricing ===")
dsl2 = "PRICE option type=call S=100 K=100 T=0.25 sigma=kan_regime regime=STABLE"
result2 = executor.execute(parse_dsl(dsl2))
print(f"Call price (STABLE): ${result2['price']:.2f}")
print(f"Vol used: {result2['volatility_used']:.4f}")

# Query 3: What-if analysis
print("\n=== What-If Analysis ===")
dsl3 = "WHAT_IF regime_shift to=STRESS asset=silver show=[price,delta,vega,vol]"
result3 = executor.execute(parse_dsl(dsl3))
print(f"Price change: ${result3['analysis']['delta']['price']:+.2f}")
print(f"Delta change: {result3['analysis']['delta']['delta']:+.4f}")
print(f"Vol change: {result3['analysis']['delta']['vol']:+.4f}")

# Query 4: Vol surface
print("\n=== Vol Surface ===")
dsl4 = "SURFACE vol asset=silver regime=STRESS strikes=[90,100,110] T=30d"
result4 = executor.execute(parse_dsl(dsl4))
for strike, vol in zip(result4['strikes'], result4['vols']):
    print(f"Strike ${strike}: {vol:.4f}")
```

---

## Testing Your Queries

```bash
# Run all executor tests
python -m pytest tests/test_executor.py -v

# Run specific test
python -m pytest tests/test_executor.py::TestExecutorPrice::test_execute_price_simple -v

# Run with more verbosity
python -m pytest tests/test_executor.py -vv -s
```

---

## Troubleshooting

### Error: "Unknown volatility type"
- Make sure you're using `kan_regime`, not `kanregime` or other variations
- Check that `KANKnowledgeStore` is provided in `ExecutionContext` if using `kan_regime`

### Error: "KANKnowledgeStore required"
- COVARIANCE, WHAT_IF, and SURFACE queries require a KAN store
- Load one with: `kan_store.load("path/to/checkpoint/")`

### Error: "Unknown time suffix"
- Use one of: `d` (days), `w` (weeks), `m` (months)
- Or use decimal years without suffix: `0.5` means 0.5 years

### Wrong results
- Check that regime parameters are correct (STABLE, TRANSITION, STRESS, RECOVERY)
- Verify spot price, strike, and time are positive
- Make sure volatility is in annualized decimal form (0.2 = 20%)

---

## Next: Phase D (LLM Integration)

Once Phase D is complete, you'll be able to skip the DSL syntax and use natural language:

```python
from src.options.llm import NLToDSL
from src.options.dsl.executor import DSLExecutor

translator = NLToDSL()
executor = DSLExecutor()

# User input in English
user_query = "Price a silver call with spot 100, strike 100, 90 days, 20% vol"

# Translate to DSL
dsl = translator.translate(user_query)

# Execute
result = executor.execute(parse_dsl(dsl))
print(result['price'])
```

