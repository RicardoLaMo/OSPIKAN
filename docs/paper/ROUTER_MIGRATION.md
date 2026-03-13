# Router Migration

## 1. Purpose

This document explains how to move the repo from the current direct-executor
pattern to a backend-registry router design.

## 2. Current Executor Pattern

The current design in `src/options/dsl/executor.py` has several strengths:

- straightforward
- readable
- easy to test
- compatible with the current REPL

But it also has four structural limits.

### Limit A. Concrete dependencies in `ExecutionContext`

Current context holds:

- `kan_store`
- `spikan_engine`
- regime functions
- feature dicts and labels

This makes the executor aware of backend implementation details instead of only
capabilities and contracts.

### Limit B. Per-verb hardcoded routing

Each query type is manually wired in the executor.

That is fine for a single backend path, but it makes comparison backends more
expensive to add.

### Limit C. Output standardization is implicit, not formal

The result shapes are stable enough for formatter tests, but there is no formal
backend result contract with provenance and status.

### Limit D. State and routing are mixed

The executor currently mixes:

- current-state defaults
- backend invocation
- output shaping

This is the main reason the backend-swappability claim is not yet proven.

## 3. Migration Goal

Move to:

- AST parsing
- request normalization
- backend routing
- standardized backend results
- formatting

as separate concerns.

## 4. Target Architecture

Recommended high-level path:

1. Parser produces AST
2. Request normalizer converts AST to `BackendRequest`
3. Router resolves backend by capability and policy
4. Backend executes using canonical state and optional option snapshot
5. Backend returns standardized `BackendResult`
6. Formatter renders the result

This is the architecture that supports both real product use and paper-grade comparisons.

## 5. Recommended Migration Sequence

### Step 1. Introduce backend result schema

Branch after planning:
- `feat/router-result-schema`

Goal:
- standardize payload plus provenance plus status

Touch points:

- `src/options/dsl/executor.py`
- formatter paths
- executor tests

Gate:
- current outputs remain formatter-compatible

### Step 2. Introduce backend registry

Branch after planning:
- `feat/backend-registry`

Goal:
- replace concrete backend fields with a capability-aware registry

Touch points:

- `ExecutionContext`
- SPIKAN adapter
- pricing adapters
- KAN-store adapters

Gate:
- current REPL still works with adapters around the existing components

### Step 3. Normalize AST into backend requests

Branch after planning:
- `feat/request-normalization`

Goal:
- isolate parser-level terms from backend-level execution

Examples:

- `PRICE` AST becomes `price_option` request
- `SURFACE` AST becomes `price_surface` request
- `OUTLOOK` AST becomes `market_outlook` request

Gate:
- routing logic no longer depends on AST subclasses directly

### Step 4. Add one comparison backend

Branch after planning:
- `feat/router-baseline-backend`

Goal:
- prove backend swappability with one real alternative

Best first candidate:

- a simple pricing or Markov baseline

Gate:
- one query can run through both backends with a stable output schema

### Step 5. Add routing policy and fallback rules

Branch after planning:
- `feat/router-policy`

Goal:
- make fallback and preferred backend behavior explicit

Gate:
- failure and abstain behavior is reproducible and testable

## 6. Recommended Adapter Map

The easiest path is adapter-based, not rewrite-based.

### Adapter A. SPIKAN outlook adapter

Wrap:
- `src/options/spikan_outlook.py`

Capability:
- `market_outlook`

### Adapter B. Regime-adjusted pricing adapter

Wrap:
- `src/options/pricing/regime_adjusted.py`

Capabilities:
- `price_option`
- `price_surface`
- `scenario_shift`

### Adapter C. KAN-store transition/covariance adapter

Wrap:
- `src/options/kan_store/store.py`

Capabilities:
- `regime_transition_prob`
- `covariance_estimate`

### Adapter D. Regime baseline adapter

Wrap:
- `src/analysis/regimes.py`

Capabilities:
- `regime_current`
- possibly `regime_transition_prob`

This preserves existing code while exposing a cleaner architecture.

## 7. Tests Needed When Implementation Starts

The first implementation wave should add or update:

- backend registry resolution tests
- request normalization tests
- output schema tests
- dual-backend execution tests
- fallback and abstain tests

The current `tests/test_executor.py` is the right starting point, but it should
evolve from direct mocks toward capability-aware adapter tests.

## 8. Success Conditions

The router migration succeeds if:

1. `ExecutionContext` becomes contract-driven rather than component-driven
2. the executor delegates through a registry rather than hardcoding every backend
3. SPIKAN and at least one baseline backend share a query path
4. outputs are comparable and provenance-aware

## 9. Failure Signs

The migration is drifting if:

- the executor still owns business logic for each backend family
- backend-specific dict defaults remain scattered across the codebase
- comparison backends require custom output shapes
- routing policy is hidden in implementation details rather than declared
