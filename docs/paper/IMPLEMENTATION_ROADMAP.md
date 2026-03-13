# Implementation Roadmap

## 1. Purpose

This roadmap converts the abstract and reconstruction plan into a concrete
execution sequence tied to the modules that already exist in the repo.

It is designed to avoid a broad rewrite.

## 2. Current Stable Touch Points

The present repo already exposes the main seams we should build around:

- protocol and query layer
  - `src/options/dsl/ast_nodes.py`
  - `src/options/dsl/validator.py`
  - `src/options/dsl/executor.py`
  - `src/options/llm/nl_to_dsl.py`
  - `scripts/options/option_dsl_repl.py`
- specialist-model layer
  - `src/physics/spikan.py`
  - `scripts/silver/train_spikan.py`
  - `src/options/spikan_outlook.py`
- state and regime layer
  - `src/analysis/regimes.py`
  - `src/geometry/optimal_transport.py`
  - `src/geometry/`
- data layer
  - `src/pipeline/`
  - `configs/`
- validation layer
  - `tests/test_executor.py`
  - `tests/test_nl_to_dsl.py`
  - `tests/test_kan_layers.py`
  - `tests/test_silver_pipeline.py`

These are the modules that should be preserved and clarified, not replaced.

## 3. Recommended Branch Sequence

### Phase 0. Planning Freeze

Branch:
- `paper/llm-spikan-architecture-plan`

Output:
- paper claim
- architecture plan
- experiment design

Gate:
- team agreement on scope

### Phase 1. State Contract

Branch:
- `feat/contracts-state-schema`

Goal:
- define one canonical market-state schema shared by geometry, regimes, DSL,
  SPIKAN, and evaluation

Expected touch points:

- new shared contract module
  - `src/contracts/` or `src/core/`
- adapters from:
  - `src/analysis/regimes.py`
  - `src/geometry/optimal_transport.py`
  - `scripts/silver/train_spikan.py`
  - `src/options/spikan_outlook.py`

Required fields:

- asset identifier
- timestamp
- regime label and probabilities
- volatility and momentum features
- geometry features
- manifold tension features
- provenance metadata

Gate:
- one schema document
- one typed record example
- one loader path from existing silver features into the schema

### Phase 2. Option Data Contract

Branch:
- `feat/options-data-contract`

Goal:
- make option-market inputs explicit and reproducible

Expected touch points:

- `src/pipeline/` or a dedicated import module
- configs for option data source and schema
- evaluation join logic with state records

Required fields:

- timestamp
- asset
- option type
- strike
- tenor
- mid or last price
- implied vol if available
- bid/ask or quality flag

Gate:
- one option snapshot schema
- one merged state-plus-options table
- one documented train/validation/test split

### Phase 3. Backend Interface And Router

Branch:
- `feat/router-backend-interface`

Goal:
- separate query execution from backend-specific logic

Expected touch points:

- `src/options/dsl/executor.py`
- `src/options/spikan_outlook.py`
- `src/options/pricing/`
- shared backend interface under `src/contracts/` or `src/core/`

Target design:

- `ExecutionContext` should become cleaner and more interface-driven
- SPIKAN should implement a specialist backend contract
- at least one baseline backend should implement the same contract

Gate:
- one DSL query can route to two backends with the same output schema
- executor logic does not hardcode backend-specific paths more than necessary

### Phase 4. Protocol Benchmark

Branch:
- `feat/protocol-eval-benchmark`

Goal:
- evaluate the LLM as a protocol compiler, not just a demo translator

Expected touch points:

- `src/options/llm/nl_to_dsl.py`
- prompt utilities under `src/options/llm/`
- tests or benchmark harness under `tests/` or `scripts/experiments/`

Required benchmark set:

- regime queries
- pricing queries
- outlook queries
- what-if queries
- malformed or ambiguous user requests

Primary metrics:

- valid DSL rate
- execution success
- retries per successful task
- abstain/fallback rate
- error taxonomy

Gate:
- benchmark report with at least 50 to 100 queries

### Phase 5. Quantitative Benchmarks

Branch:
- `feat/spikan-vs-baselines`

Goal:
- compare SPIKAN against simpler and cheaper alternatives on real tasks

Expected touch points:

- `scripts/silver/train_spikan.py`
- baseline experiment runners
- shared evaluation logic
- reports and tables

Required baseline set:

- SPIKAN
- regime-adjusted Black-Scholes baseline
- Markov/HMM baseline
- tree or boosting baseline on the shared state schema

Primary tasks:

- regime transition or stress prediction
- option price or implied-vol prediction
- scenario or outlook ranking

Gate:
- one benchmark table
- one calibration figure
- one stress-period slice

### Phase 6. Integration-Cost Benchmark

Branch:
- `feat/integration-cost-study`

Goal:
- prove the systems claim, not only the quant claim

Expected touch points:

- benchmark docs
- router/backends
- optionally branch-level engineering logs

Metrics:

- files touched per backend integration
- adapter LOC
- time to first executable benchmark
- number of backend-specific code paths

Gate:
- side-by-side comparison for:
  - LLM + DSL + router
  - direct backend-specific glue
  - template-only interface

### Phase 7. Paper Figures And Tables

Branch:
- `feat/paper-figures-and-tables`

Goal:
- freeze the paper evidence package

Required artifacts:

- architecture figure
- protocol trace figure
- benchmark table
- calibration figure
- event-study figure
- cost comparison table

Gate:
- all core claims map to at least one figure or table

## 4. Module Impact Map

The reconstruction should mostly affect these boundaries:

### Shared contract boundary

Needs to be introduced because it does not yet exist explicitly.

Most affected:

- `src/options/dsl/executor.py`
- `src/options/spikan_outlook.py`
- `scripts/silver/train_spikan.py`
- `src/analysis/regimes.py`

### Protocol boundary

Already exists, but needs formal evaluation.

Most affected:

- `src/options/llm/nl_to_dsl.py`
- `src/options/dsl/validator.py`
- `tests/test_nl_to_dsl.py`
- `tests/test_executor.py`

### Specialist-backend boundary

Exists implicitly, but needs abstraction so baselines can plug in.

Most affected:

- `src/options/spikan_outlook.py`
- `src/options/pricing/`
- `src/physics/spikan.py`

### Data boundary

Exists for silver features, but not yet for option datasets aligned to the same
evaluation clock.

Most affected:

- `src/pipeline/`
- `configs/`
- benchmark runners

## 5. First Credible Benchmark Slice

The first full proof should stay narrow:

- asset: silver
- horizons: one short horizon for outlook and one medium tenor for options
- state source: current silver/macroeconomic/geometry pipeline
- specialist backend: SPIKAN
- baseline backends:
  - regime-adjusted Black-Scholes
  - Markov/HMM
  - boosting baseline
- protocol test set:
  - at least 50 natural-language queries

This is enough to produce the first defendable benchmark package without
opening too many variables at once.

## 6. Non-Goals During Reconstruction

To avoid drift, these should not be part of the first implementation cycle:

- expanding to many assets before the single-asset workflow is proven
- introducing many new LLM providers before the protocol benchmark exists
- large visual or UX redesign
- broad physics-method expansion before baseline comparisons are complete

## 7. Immediate Execution Order

1. land the state contract
2. land the option data contract
3. refactor the router and backend interface
4. benchmark the protocol layer
5. benchmark SPIKAN vs baselines
6. benchmark integration cost
7. package figures and tables for the paper
