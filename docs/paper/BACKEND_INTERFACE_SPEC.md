# Backend Interface Specification

## 1. Purpose

This document defines the common backend interface that should sit between:

- the DSL executor in `src/options/dsl/executor.py`
- SPIKAN inference in `src/options/spikan_outlook.py`
- the regime-adjusted pricing stack in `src/options/pricing/`
- comparison backends such as HMM, boosting, or simpler pricing baselines

The goal is to prove a systems claim:

> The DSL is the stable machine contract, and specialist models can be swapped
> behind that contract without re-architecting the full workflow.

The current repo does not expose that abstraction explicitly yet.

## 2. Current Reality

The current execution path is workable, but tightly coupled.

### Executor coupling

`DSLExecutor` currently:

- dispatches by AST node type
- directly owns a `RegimeAdjustedPricer`
- reads `kan_store` and `spikan_engine` directly from `ExecutionContext`
- mixes:
  - query parsing decisions
  - feature defaults
  - routing logic
  - output shaping

### Backend coupling

The backends are not yet described as a shared family.

In practice, the repo already has at least these backend types:

- `price` backend
  - Black-Scholes or regime-adjusted pricing
- `surface` backend
  - KAN vol surface
- `transition` backend
  - KAN transition model or Markov-style baseline
- `outlook` backend
  - SPIKAN outlook adapter
- `regime` backend
  - geometric regime functions

That is a real backend ecosystem. It just is not formalized yet.

## 3. Design Principles

The backend interface should be:

- capability-driven
  - a backend declares what it can answer
- schema-stable
  - outputs are typed and predictable
- state-driven
  - backend input should come from canonical state and option snapshots
- backend-neutral
  - the router should not care whether the implementation is SPIKAN, HMM, KAN, or a tree model
- auditable
  - output must include provenance and model identity

## 4. Capability Model

Each backend should declare one or more capabilities.

Recommended canonical capability set:

- `price_option`
- `price_surface`
- `regime_current`
- `regime_transition_prob`
- `covariance_estimate`
- `scenario_shift`
- `market_outlook`
- `regime_explain`

Not every backend needs every capability.

Examples:

- SPIKAN backend
  - `market_outlook`
  - optionally `regime_transition_prob`
- regime-adjusted pricing backend
  - `price_option`
  - `price_surface`
  - `scenario_shift`
- KAN store backend
  - `price_surface`
  - `covariance_estimate`
  - `regime_transition_prob`
- HMM baseline backend
  - `regime_current`
  - `regime_transition_prob`
- tabular baseline backend
  - `market_outlook`
  - possibly `regime_current`

## 5. Canonical Backend Input

All backend calls should be driven from canonical records, not ad hoc dicts.

Minimum input objects:

- `MarketStateSnapshot`
- `OptionSnapshot`
- `BackendRequest`

The `BackendRequest` should include:

- `query_kind`
- `asset` or `underlying`
- `timestamp`
- optional `regime_override`
- optional scenario parameters
- optional strike grid or tenor
- request metadata

This keeps the router separate from parser-specific implementation details.

## 6. Canonical Backend Output

Every backend response should map to a standard `BackendResult`.

Required top-level fields:

- `query_type`
- `backend_id`
- `backend_family`
- `backend_version`
- `capability`
- `status`
  - `ok`, `abstain`, `error`, `fallback`
- `payload`
  - structured result object
- `provenance`
  - model name, checkpoint, data timestamp, state version
- `warnings`
  - optional list

The current repo already hints at this pattern:

- `OUTLOOK` returns `source_model`
- all formatter paths rely on stable `query_type`

That should become explicit across all verbs.

## 7. Query-Specific Payload Schemas

### `price_option`

Payload should include:

- `option_type`
- `spot`
- `strike`
- `time_to_expiry`
- `volatility_used`
- `price`
- `delta`
- `gamma`
- `vega`
- `theta`
- `rho`
- `regime`

This aligns with current `PRICE` output.

### `regime_transition_prob`

Payload should include:

- `from_regime`
- `to_regime`
- `horizon`
- `all_probs`
- `target_prob`

This aligns with current `REGIME_PROB` output.

### `covariance_estimate`

Payload should include:

- `assets`
- `regime`
- `window`
- `covariance`

This aligns with current `COVARIANCE` output.

### `scenario_shift`

Payload should include:

- `to_regime`
- `asset`
- `spot`
- `strike`
- `time_to_expiry`
- `analysis`

This aligns with current `WHAT_IF` output.

### `price_surface`

Payload should include:

- `asset`
- `regime`
- `time_to_expiry`
- `vol_type`
- `strikes`
- `vols`

This aligns with current `SURFACE` output.

### `market_outlook`

Payload should include:

- `asset`
- `horizon`
- `horizon_days`
- `regime`
- `direction`
- `outlook_score`
- `confidence`
- `shock_risk`
- `flow_regime`
- `model_quality`
- `trend_signal`
- `shock_score`
- `raw_signal`

This aligns with current `OUTLOOK` output.

## 8. Backend Families

For comparisons, each backend should identify its family:

- `spikan`
- `kan_store`
- `pricing_baseline`
- `markov_regime`
- `tabular_ml`
- `sequence_ml`
- `rule_based`

This matters for both benchmarking and audit trails.

## 9. Recommended Interface Shape

The repo does not need a complex framework.

A minimal interface is enough.

Recommended conceptual shape:

- `BackendRegistry`
  - resolve backend by capability and policy
- `BackendHandle`
  - advertises:
    - `backend_id`
    - `backend_family`
    - `capabilities`
  - implements:
    - `execute(request, state_snapshot, option_snapshot=None)`

The exact language-level implementation can be a Python protocol, abstract base
class, or duck-typed adapter. The architecture matters more than the mechanism.

## 10. Routing Policies

The router should support simple explicit policies:

- `preferred_backend`
- `fallback_backend`
- `capability_required`
- `allow_abstain`

Examples:

- `OUTLOOK`
  - preferred backend: SPIKAN
  - fallback: tabular or rule-based outlook
- `PRICE`
  - preferred backend: regime-adjusted pricer
  - fallback: pure Black-Scholes
- `REGIME_PROB`
  - preferred backend: KAN transition
  - comparison backend: Markov model

## 11. Compatibility With Current Executor

The current executor should be understood as a transitional router.

Current `ExecutionContext` contains:

- `kan_store`
- `spikan_engine`
- regime functions
- current regime features and label

Target `ExecutionContext` should contain:

- canonical current state snapshot
- optional current option snapshot
- backend registry
- routing policy

This is a cleaner contract and supports benchmark comparisons directly.

## 12. Acceptance Criteria

The backend interface is ready when:

1. one capability registry exists
2. one DSL query can run against two backends with the same output schema
3. outputs include backend identity and provenance
4. the executor no longer needs to know backend-specific implementation details for every verb

Until then, the repo should describe the routing layer as direct integration, not a fully modular backend architecture.
