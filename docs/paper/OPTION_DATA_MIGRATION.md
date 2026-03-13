# Option Data Migration

## 1. Purpose

This document explains how to move the current options stack from synthetic-only
training assumptions to a real-data benchmark path.

## 2. Current Repo Reality

The current options architecture is strong at the interface level:

- DSL exists
- parser and validator exist
- pricing stack exists
- KAN store exists
- REPL exists

But the option-data path is not yet aligned to the paper claim.

The most important current limitation is:

- `scripts/options/train_kan_store.py` is synthetic-only by design

That is acceptable for engineering bootstrapping, but not enough for proving
the paper’s real-data interoperability claim.

## 3. Current Implicit Contracts

### Contract A. Query contract

From the DSL and parser:

- `PRICE` assumes spot, strike, tenor, vol, and type
- `WHAT_IF` assumes regime-conditioned repricing
- `SURFACE` assumes strike grid and tenor slice

### Contract B. Regime-feature contract

From `src/options/kan_store/feature_bridge.py`:

- the KAN store expects an 8-dimensional regime vector:
  - `ricci_mean_core_60d`
  - `ricci_min_core_60d`
  - `mst_stress_core_60d`
  - `ga_rotor_magnitude_60d`
  - `realized_vol_20d`
  - `momentum_10d`
  - `p_regime_0`
  - `p_regime_1`

### Contract C. Missing market-data contract

There is no canonical raw option snapshot schema yet.

That is the specific gap this phase addresses.

## 4. Recommended Migration Sequence

### Step 1. Freeze the option snapshot schema

Branch:
- `feat/options-data-contract`

Artifacts:
- option snapshot spec
- joined dataset spec
- benchmark target definitions

Gate:
- no code change required yet

### Step 2. Add option import contract

Expected implementation branch later:
- `feat/options-import-pipeline`

Goal:
- create one import path that lands raw option data into the canonical snapshot schema

Touch points likely:

- `src/pipeline/`
- `configs/`
- a dedicated options import module

Gate:
- one raw sample and one cleaned sample can be loaded reproducibly

### Step 3. Build state-option joiner

Expected implementation branch later:
- `feat/state-option-join`

Goal:
- join `MarketStateSnapshot` with `OptionSnapshot`

Touch points likely:

- shared contracts layer
- evaluation or benchmark module

Gate:
- one reproducible joined dataset with time splits

### Step 4. Refactor KAN store training to accept real targets

Expected implementation branch later:
- `feat/kan-store-real-data-training`

Current problem:
- vol surface KAN is trained on synthetic vol surfaces

Migration target:
- the same KAN architecture can be trained on observed option targets from the
  joined dataset

Gate:
- one benchmark run using observed `implied_vol` or `mid`

### Step 5. Add baseline pricing data path

Expected implementation branch later:
- `feat/options-baseline-benchmarks`

Goal:
- compare:
  - KAN-conditioned pricing
  - simpler volatility baselines
  - regime-unaware pricing baselines

Gate:
- all models consume the same joined option dataset

## 5. Files Most Likely To Change When Implementation Starts

- new options import module under `src/pipeline/` or `src/options/`
- `configs/options_dsl.yaml`
- `scripts/options/train_kan_store.py`
- `src/options/kan_store/store.py`
- evaluation scripts and tests

## 6. Recommended First Real-Data Benchmark

Do not start with the full option surface across many assets.

Start with:

- one underlying or proxy
- daily snapshots
- one target:
  - observed implied vol, preferred
  - or option mid if IV is unavailable
- one liquid tenor band
- one near-ATM bucket plus one OTM bucket

Why this is the right first slice:

- it is sufficient to validate the architecture
- it constrains noise
- it lets the paper make a real-data claim without overscoping the pipeline

## 7. Benchmark Outputs Required

The first options benchmark should produce:

- option data coverage table
- joined state-option coverage table
- static fit metrics by model
- fit metrics by regime
- error by tenor and moneyness bucket

If possible, also add:

- one surface comparison figure
- one stress-period error slice

## 8. Success Conditions

The option-data migration succeeds if:

1. the options stack has one canonical real-data input schema
2. the KAN store can train or evaluate against observed option targets
3. baseline models use the same joined dataset
4. the paper can report real-data option results instead of only synthetic demonstrations

## 9. Failure Signs

The migration is drifting if:

- synthetic and real option targets are mixed without clear labeling
- each backend needs its own custom option data loader
- option quality filtering is undocumented
- the benchmark target changes across models
