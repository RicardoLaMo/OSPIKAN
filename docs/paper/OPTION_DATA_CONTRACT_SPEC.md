# Option Data Contract Specification

## 1. Purpose

This document defines the canonical option-data schema needed to make the paper
claim real on market data.

It bridges the current gap between:

- the DSL pricing interface in `src/options/dsl/`
- the regime-adjusted pricer in `src/options/pricing/regime_adjusted.py`
- the KAN store in `src/options/kan_store/`
- the current synthetic-only KAN training flow in `scripts/options/train_kan_store.py`

The central issue is simple:

- the options stack already has a query and pricing contract
- but it does not yet have a canonical real-data option snapshot contract

Without that, the paper can show architecture and demos, but not a defensible
real-data benchmark.

## 2. Design Principles

The option data contract should be:

- executable
  - directly usable by pricing and benchmark code
- market-native
  - represent actual tradable contracts and quotes
- aligned
  - join cleanly with the market-state contract at time `t`
- quality-aware
  - encode quote quality, liquidity, and missingness explicitly
- backend-neutral
  - usable for SPIKAN, KAN-store baselines, and simpler pricing baselines

## 3. Canonical Unit

The proposed unit is an `OptionSnapshot`.

Each record represents one option contract quote for one underlying at one
timestamp.

## 4. Top-Level Fields

### Identity

- `timestamp`
  - quote timestamp or end-of-day timestamp
- `underlying`
  - canonical underlying key, e.g. `silver`
- `venue`
  - source venue or data source identifier
- `contract_id`
  - stable contract identifier if available
- `snapshot_version`
  - schema version
- `source_run_id`
  - data import run id

### Contract Terms

- `option_type`
  - `call` or `put`
- `strike`
  - strike price in underlying currency units
- `expiration`
  - contract expiry date
- `days_to_expiry`
- `time_to_expiry_years`

These map directly onto the current DSL and pricer expectations.

### Underlying Reference

- `spot_price`
  - underlying spot or proxy close used at the same timestamp
- `forward_price`
  - optional if available
- `risk_free_rate`
  - rate used for pricing reference
- `dividend_yield`
  - usually zero or proxy-specific if using ETF options

### Quote Fields

- `bid`
- `ask`
- `mid`
- `last`
- `volume`
- `open_interest`
- `quote_age_seconds`
  - optional when intraday feeds exist

At minimum, one of `mid` or `bid+ask` should exist.

### Derived Market Fields

- `log_moneyness`
  - `log(S / K)` using the aligned spot price
- `moneyness_ratio`
  - `S / K`
- `implied_vol`
  - observed or cleaned implied vol if available
- `delta`
- `gamma`
- `vega`
- `theta`
- `rho`

These are optional in the raw import, but strongly preferred in the benchmark
table. If not provided, they can be derived later.

### Quality Flags

- `is_valid_quote`
- `is_crossed_market`
- `is_wide_spread`
- `is_stale_quote`
- `is_low_liquidity`
- `quality_score`

These matter because option data is much noisier than spot data, and the paper
should not quietly mix high-quality and low-quality quotes.

## 5. Canonical Views

The option snapshot should be grouped into views:

### `contract`

- underlying
- type
- strike
- expiration
- tenor

### `market`

- spot
- rate
- dividend
- forward

### `quote`

- bid
- ask
- mid
- last
- size and activity fields

### `derived`

- moneyness
- implied vol
- Greeks

### `quality`

- validation flags
- quality score

### `provenance`

- source
- run id
- timestamp validity

## 6. Minimum Required Fields For Existing Repo Components

### DSL `PRICE`

Current minimum:

- `spot_price`
- `strike`
- `time_to_expiry_years`
- `option_type`
- one volatility input

For real-data benchmarking, this should become:

- `spot_price`
- `strike`
- `time_to_expiry_years`
- `option_type`
- `mid` or `implied_vol`
- `risk_free_rate`

### DSL `SURFACE`

Current query shape:

- asset
- regime
- strikes
- time to expiry

For real-data benchmarking, the canonical option snapshot must support
surface slicing by:

- timestamp
- underlying
- tenor bucket
- strike or moneyness bucket

### KAN Vol Surface

Current model input:

- regime features [8]
- `log_moneyness`
- `time_to_expiry`

Current missing link:

- no canonical real-data target field for training

Required target:

- cleaned `implied_vol` or an explicitly defined alternative target

### Regime-Adjusted Pricer

The pricer can already price contracts once contract terms and regime features
exist. The missing part is a canonical record from which benchmark targets are
drawn consistently.

## 7. Canonical Joined Table

The key paper dataset should not be the raw option snapshot alone.

It should be a joined record:

`StateOptionSnapshot = MarketStateSnapshot + OptionSnapshot`

Required join keys:

- `timestamp`
- `underlying`

Recommended join tolerance:

- strict same-day join for end-of-day studies
- explicit tolerance window for intraday sources

## 8. Required Joined Fields For First Benchmark

For the first credible slice, the joined dataset should contain:

- timestamp
- underlying
- option_type
- strike
- time_to_expiry_years
- spot_price
- mid
- implied_vol
- log_moneyness
- risk_free_rate
- state fields from the canonical market-state contract:
  - realized vol
  - momentum
  - geometry fields
  - regime probabilities
  - optional tension metrics

This is enough to evaluate:

- price prediction
- implied-vol prediction
- regime-aware surface behavior

## 9. Real-Data Target Definitions

The contract should define targets explicitly.

### Target A. Static Quote Fit

Predict:

- `mid`
  or
- `implied_vol`

at the same timestamp.

This is the easiest starting point and tests whether the state-aware backend can
fit the observed surface.

### Target B. Horizon Shift

Predict:

- future `mid`
- future `implied_vol`
- or future ranking of stressed vs calm surface states

This is harder, but it better matches the regime-switching story.

### Target C. Scenario Ranking

Predict:

- relative repricing or risk ranking under regime shifts

This is closest to the business case, but it should come after static fit and
horizon tasks are stable.

## 10. Quality Policy

The paper should define a minimum quality bar for inclusion:

- exclude crossed or locked markets unless documented
- exclude quotes with missing core fields
- exclude stale or zero-liquidity contracts for the first benchmark
- bucket or filter by tenor and moneyness to avoid distorted comparisons

This should be visible in the benchmark pipeline, not only in notes.

## 11. First Recommended Benchmark Slice

Keep it narrow.

Recommended first slice:

- one underlying: `silver`
- one option source or liquid proxy
- one quote frequency: daily end-of-day is acceptable for the first pass
- one target:
  - static implied-vol fit, or
  - static mid-price fit
- one tenor band:
  - e.g. short-dated liquid contracts
- one moneyness band:
  - near-ATM plus one OTM band

This is enough to test the architecture without overfitting to data plumbing.

## 12. Acceptance Criteria

The option data contract is ready when:

1. one schema document exists
2. one example option snapshot exists
3. one joined state-plus-option record exists
4. one benchmark task can consume the joined dataset reproducibly
5. the KAN store and pricing baselines can target the same observed field

Until then, the option side of the repo should be described as partially
synthetic and not yet fully validated on market option data.
