# State Contract Specification

## 1. Purpose

This document defines the canonical market-state record that should sit between:

- `src/pipeline/`
- `src/analysis/`
- `src/geometry/`
- `src/options/dsl/`
- `src/options/spikan_outlook.py`
- `scripts/silver/train_spikan.py`

The immediate reason for this contract is that the current repo uses three
different state conventions:

1. geometric regime functions operate on feature tables
2. `SPIKANOutlookEngine` uses a handcrafted 8-feature regime vector
3. `train_spikan.py` selects features heuristically by substring

That is workable for experiments, but it is not a paper-grade architecture.

## 2. Design Principles

The state contract should be:

- canonical
  - one shared record shape across the system
- typed
  - each field has a name, meaning, and unit
- layered
  - raw inputs, derived state, and labels are separated
- backend-agnostic
  - a model can consume a projected view of state without redefining state itself
- provenance-aware
  - the state record must say where it came from and when it is valid

## 3. Canonical State Record

The proposed unit is a `MarketStateSnapshot`.

Each record represents one asset, one timestamp, and one view of the market
state available at that timestamp.

## 4. Top-Level Fields

### Identity

- `timestamp`
  - trading date or datetime
- `asset`
  - canonical asset key such as `silver`
- `state_version`
  - schema version string
- `source_run_id`
  - pipeline or benchmark run id

### Base Market Features

- `spot_close`
- `log_return_1d`
- `realized_vol_20d`
- `momentum_10d`
- `drawdown_63d`
- `drawdown_252d`

These are stable and already aligned with the silver pipeline.

### Macro Features

- `dxy_log_return_1d`
- `dxy_beta_60d`
- `y10_change_1d`
- `spx_log_return_1d`
- `vix_log_return_1d`

These represent the macro state accessible at time `t`.

### Geometry Features

The contract should prefer the current core-manifold columns when available.

- `ricci_mean_core_60d`
- `ricci_min_core_60d`
- `ricci_tail_core_60d`
  - canonical alias for whichever tail column is selected
- `mst_stress_core_60d`
- `ga_rotor_magnitude_60d`
- `ga_bivector_energy_60d`

Why:

- `geometric_regime_classification()` already prefers core Ricci and core MST
- the paper should not rely on ambiguous fallback column names in the main schema

### Regime Features

- `geometric_regime`
  - one of `STABLE`, `TRANSITION`, `STRESS`, `RECOVERY`, `UNKNOWN`
- `fluid_regime`
  - one of `NORMAL`, `SHOCK_IMMINENT`, `MOMENTUM_PERSISTING`, `MOMENTUM_DECAYING`
- `combined_regime`
- `p_regime_0`
- `p_regime_1`
- `p_regime_2`
- `p_regime_3`
- `regime_confidence`

Notes:

- the current repo already uses `p_regime_*` naming from Markov outputs
- `SPIKANOutlookEngine` currently only consumes `p_regime_0` and `p_regime_1`
- the canonical state should keep the full probability vector when available

### Fluid And Shock Features

- `shock_formation_index_z`
- `momentum_decay_rate`
- `flow_divergence`
- `shock_warning`
  - boolean

These align with the existing enhanced regime path.

### Manifold-Tension Features

These are now first-class because `ManifoldTensionValidator` already exposes:

- `structural_tension_gw`
- `level_tension_w2`

These should be treated as optional but strongly preferred fields for the paper
system because they connect narrative priors to the market-state geometry.

### Label Fields

These do not belong in the online state used at inference time, but they should
exist in the benchmark state table or adjacent evaluation record.

- `shock_label_h_plus_k`
- `direction_label_h_plus_k`
- `regime_transition_label_h_plus_k`
- `option_target_mid_h_plus_k`
- `option_target_iv_h_plus_k`

## 5. Canonical State Groups

For implementation, the state should be grouped into views:

### `base`

- returns
- volatility
- momentum
- drawdowns

### `macro`

- FX, rates, equity, vol

### `geometry`

- Ricci
- MST stress
- GA rotor and energy

### `regime`

- hard labels
- soft probabilities
- confidence

### `fluid`

- shock and decay diagnostics

### `tension`

- structural tension
- level tension

### `provenance`

- run id
- feature version
- timestamp validity

This grouping matters because backends should consume projections from these
groups rather than re-parsing raw flat DataFrames ad hoc.

## 6. Minimum Required State For Each Existing Module

### Geometric regime classification

Required now:

- `ricci_mean_core_60d`
- `ricci_min_core_60d`
- `ricci_tail_core_60d`
- `mst_stress_core_60d`
- `ga_rotor_magnitude_60d`

### Enhanced regime with fluid dynamics

Required now:

- `geometric_regime`
- `shock_formation_index_z`
- `momentum_decay_rate`

### SPIKAN outlook adapter

Current implied minimum:

- `momentum_10d`
- `realized_vol_20d`
- `mst_stress_core_60d`
- `ga_rotor_magnitude_60d`
- `ricci_mean_core_60d`
- `p_regime_0`
- `p_regime_1`

Recommended canonical minimum:

- all of the above
- plus `ricci_min_core_60d`
- plus optional tension features

### SPIKAN training

Current training path uses substring heuristics. The contract should replace
that with explicit projections:

- asset tensor view
- macro tensor view
- geometry tensor view
- time coordinate
- target record

## 7. Projection Rules

The canonical state must support deterministic projections into the current
backend interfaces.

### Projection A. State -> SPIKAN outlook tensor bundle

Asset projection:

- focus asset momentum
- stress-adjusted asset bias
- optional asset-local derived slots

Macro projection:

- stress level
- volatility deviation
- regime probability spread

Geometry projection:

- Ricci mean
- MST stress
- rotor magnitude
- momentum or other chosen bridge signal

Time projection:

- normalized horizon

Important:

The projection may remain model-specific. The source state must not.

### Projection B. State -> Regime classifier input

Use canonical geometry fields directly rather than fallback name inference.

### Projection C. State -> DSL execution context

The executor should receive:

- current state snapshot
- current option snapshot
- backend registry

not a loose collection of partially overlapping dicts.

## 8. Example Record

```json
{
  "timestamp": "2024-05-01",
  "asset": "silver",
  "state_version": "v1",
  "source_run_id": "silver_20240501",
  "spot_close": 26.41,
  "log_return_1d": 0.012,
  "realized_vol_20d": 0.244,
  "momentum_10d": 0.031,
  "drawdown_63d": -0.048,
  "drawdown_252d": -0.091,
  "dxy_log_return_1d": -0.004,
  "dxy_beta_60d": -0.37,
  "y10_change_1d": 0.02,
  "spx_log_return_1d": 0.006,
  "vix_log_return_1d": -0.018,
  "ricci_mean_core_60d": -0.07,
  "ricci_min_core_60d": -0.18,
  "ricci_tail_core_60d": -0.16,
  "mst_stress_core_60d": 0.51,
  "ga_rotor_magnitude_60d": 0.29,
  "ga_bivector_energy_60d": 0.11,
  "geometric_regime": "TRANSITION",
  "fluid_regime": "NORMAL",
  "combined_regime": "TRANSITION",
  "p_regime_0": 0.34,
  "p_regime_1": 0.41,
  "p_regime_2": 0.17,
  "p_regime_3": 0.08,
  "regime_confidence": 0.41,
  "shock_formation_index_z": 1.2,
  "momentum_decay_rate": 0.03,
  "flow_divergence": 0.18,
  "shock_warning": false,
  "structural_tension_gw": 0.27,
  "level_tension_w2": 0.12
}
```

## 9. Online Versus Offline Fields

The contract should distinguish:

### Online fields

Known at time `t` and allowed during live inference.

### Offline labels

Constructed using future windows and allowed only in benchmark/evaluation mode.

This is essential because the current shock-label path in training uses
lookahead labels, which must never be confused with online state.

## 10. Versioning

The contract should be versioned from the start.

Recommended first versions:

- `state/v1`
- `option_snapshot/v1`
- `backend_output/v1`

If field names must change later, adapters should handle migration explicitly.

## 11. Acceptance Criteria

The state contract is ready only when:

1. one schema doc exists
2. one typed example record exists
3. regime logic can consume the canonical fields directly
4. SPIKAN outlook can consume a projection from the canonical state
5. SPIKAN training can consume explicit field groups rather than substring matching

Until then, the paper should describe the current state path as heuristic, not final.
