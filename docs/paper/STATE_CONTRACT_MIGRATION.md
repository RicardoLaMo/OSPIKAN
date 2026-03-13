# State Contract Migration

## 1. Purpose

This document explains how to migrate the current repo from heuristic state
handling to the canonical state contract defined in `STATE_CONTRACT_SPEC.md`.

## 2. Current Gaps

### Gap A. Outlook path uses a handcrafted state vector

Current source:
- `src/options/spikan_outlook.py`

Issue:
- the adapter builds tensors from a small dict with implicit defaults
- this is useful for demos, but it is not the same as a canonical state object

Migration target:
- outlook engine accepts a `MarketStateSnapshot`
- projection to SPIKAN tensors happens in a dedicated adapter layer

### Gap B. Training path uses substring-based feature selection

Current source:
- `scripts/silver/train_spikan.py`

Issue:
- feature selection by substring is fragile and can drift as columns evolve

Migration target:
- training config points to explicit state-field groups
- one projection function builds asset, macro, and geometry tensors

### Gap C. Regime path still depends on fallback column resolution

Current source:
- `src/analysis/regimes.py`

Issue:
- the function is robust, but the canonical system should not depend on implicit
  column fallback rules in the main execution path

Migration target:
- canonical state generation resolves preferred columns once
- downstream regime logic consumes explicit canonical names

### Gap D. Tension diagnostics are not first-class in the shared state

Current source:
- `src/geometry/optimal_transport.py`

Issue:
- tension metrics exist, but they are not yet part of the main state schema

Migration target:
- tension diagnostics become optional canonical fields
- narrative and outlook experiments can consume them systematically

## 3. Recommended Migration Sequence

### Step 1. Introduce shared contract types

Branch:
- `feat/contracts-state-schema`

Add:
- `src/contracts/` or `src/core/`

Artifacts:
- market state type
- option snapshot type
- backend output type

Gate:
- no business logic changed yet
- types and schema docs only

### Step 2. Add state builder from current feature tables

Input:
- existing silver features DataFrame

Output:
- canonical market-state records

Gate:
- round-trip examples created from current processed data

### Step 3. Refactor regime logic to canonical inputs

Touch points:
- `src/analysis/regimes.py`

Goal:
- preserve existing behavior while removing ambiguity from field selection in
  the main path

Gate:
- geometric regime outputs remain unchanged or intentionally documented

### Step 4. Refactor SPIKAN outlook to canonical inputs

Touch points:
- `src/options/spikan_outlook.py`

Goal:
- move from free dict input to canonical state plus projection adapter

Gate:
- current `OUTLOOK` behavior remains functionally intact
- adapter is explicit and testable

### Step 5. Refactor SPIKAN training to canonical projections

Touch points:
- `scripts/silver/train_spikan.py`

Goal:
- remove substring heuristics from training data preparation

Gate:
- train, validation, and test splits still reproduce
- tensor field choices are explicit and logged

### Step 6. Pass canonical state into DSL execution context

Touch points:
- `src/options/dsl/executor.py`

Goal:
- `ExecutionContext` should receive canonical state and backend registry rather
  than a loose mix of dicts and functions

Gate:
- executor remains backwards-compatible during transition, or compatibility is
  documented clearly

## 4. Files Most Likely To Change In The Actual Implementation Phase

- `src/contracts/` or `src/core/`
- `src/options/spikan_outlook.py`
- `scripts/silver/train_spikan.py`
- `src/options/dsl/executor.py`
- `src/analysis/regimes.py`
- selected tests under `tests/`

## 5. Tests Needed Once Implementation Starts

When coding begins, these are the first tests to add or update:

- state-schema construction tests
- projection tests:
  - canonical state to SPIKAN outlook tensors
  - canonical state to training tensors
- executor tests using canonical state in `ExecutionContext`
- compatibility tests so current DSL queries still run

## 6. Success Conditions

The migration succeeds if:

1. state fields are explicit
2. SPIKAN training and inference consume the same source state
3. regime analysis, outlook generation, and evaluation refer to the same state vocabulary
4. benchmark code can compare multiple backends without redefining state per backend

## 7. Failure Signs

The migration is drifting if:

- multiple parallel state formats still exist after refactor
- tensor-building logic stays duplicated in training and inference
- the executor keeps absorbing backend-specific state hacks
- benchmarks rely on undocumented field conventions
