# LLM + SPIKAN Reconstruction Plan

## 1. Paper Claim

Primary claim:

> A constrained language layer can act as a reusable interoperability contract
> between humans and specialist quantitative models. In this repo, the LLM
> speaks natural language outward and DSL inward, while SPIKAN remains the
> finance-native engine for regime-aware option reasoning.

What we should prove:

1. The LLM+DSL layer reduces integration complexity relative to hand-wired model orchestration.
2. The quantitative engine remains grounded in real financial structure rather than free-form prompting.
3. The combined system can answer useful option/regime questions on real market data with measurable quality.

What we should not over-claim:

- The LLM predicts markets by itself.
- SPIKAN is the only possible specialist model.
- Natural language alone is sufficient without a formal contract.

## 2. Current Repo State From Git Tree

The tracked repository already contains the right building blocks:

- `src/pipeline/`
  - silver and macro data ingest, alignment, feature generation
- `src/analysis/`
  - trend and regime analysis
- `src/geometry/`
  - Ricci, MST stress, GA, and optimal transport methods
- `src/options/dsl/`
  - parser, validator, AST, executor
- `src/options/llm/`
  - NL-to-DSL and manifold mapping
- `src/options/pricing/`
  - pricing logic and regime-adjusted pricer
- `src/physics/`
  - KAN layers, SPIKAN, PDE losses, shock tools
- `scripts/options/`, `scripts/silver/`, `scripts/research/`
  - operational, domain, and exploratory entry points
- `tests/`
  - broad unit coverage across DSL, pricing, geometry, pipeline, and physics

The problem is not missing modules. The problem is that the repo still behaves
like a collection of strong subsystems rather than a single paper-grade system.

## 3. Diagnosis

The architecture is currently strongest at the component level:

- DSL stack is real and tested.
- SPIKAN stack is real and trained through scripts.
- Geometry stack is rich and thesis-worthy.
- Research scripts show narrative-to-state ideas.

The weak points are:

1. There is no single canonical state contract shared across geometry, regimes, DSL, and SPIKAN.
2. Options workflow and silver/regime workflow are still adjacent more than unified.
3. LLM evaluation is mostly functional, not scientific.
4. The proof path for the main claim is incomplete:
   - real option market data path is not yet central
   - baseline comparisons are not locked
   - integration-cost evidence is not formalized

## 4. Reconstruction Goal

We should reconstruct around six layers, while changing as little as possible
about the current source tree.

### Layer A. Data Plane

Purpose:
- produce reproducible spot, macro, regime, and option inputs

Current anchors:
- `src/pipeline/`
- `configs/`
- silver feature outputs under `data/processed/`

Reconstruction target:
- formal data contracts for:
  - state features at time `t`
  - option market snapshot at time `t`
  - evaluation label windows at `t+h`

### Layer B. State Plane

Purpose:
- turn raw market observations into a regime/state vector

Current anchors:
- `src/analysis/`
- `src/geometry/`
- manifold tension work in `src/geometry/optimal_transport.py`

Reconstruction target:
- define one canonical state schema used everywhere:
  - geometric stress
  - volatility and momentum
  - regime probabilities
  - manifold tension diagnostics

### Layer C. Protocol Plane

Purpose:
- convert human intent into valid machine actions

Current anchors:
- `src/options/llm/nl_to_dsl.py`
- `src/options/dsl/`
- `scripts/options/option_dsl_repl.py`

Reconstruction target:
- treat the DSL as the protocol contract
- separate:
  - user intent parsing
  - DSL validation
  - execution routing
- keep LLM output measurable by validity, recoverability, and task success

### Layer D. Specialist Model Plane

Purpose:
- execute quantitative finance reasoning

Current anchors:
- `src/physics/spikan.py`
- `src/physics/market_pdes.py`
- `src/options/spikan_outlook.py`
- `src/options/pricing/`

Reconstruction target:
- present SPIKAN as one backend implementing a common specialist-model interface
- allow side-by-side comparison against other backends without rewriting the full workflow

### Layer E. Evaluation Plane

Purpose:
- prove the system works and the claim is real

Current anchors:
- `tests/`
- silver reports
- research scripts

Reconstruction target:
- add explicit experiment paths for:
  - protocol quality
  - predictive quality
  - option pricing quality
  - integration-cost comparison

### Layer F. Product Plane

Purpose:
- show the business case

Current anchors:
- REPL
- reports
- what-if pricing and outlook queries

Reconstruction target:
- one demonstrable end-to-end case:
  - human asks in natural language
  - LLM emits DSL
  - executor routes to SPIKAN and pricing stack
  - system returns regime-aware option outlook with evidence

## 5. Proposed Minimal-Risk Code Architecture

This should be a reconstruction, not a rewrite.

Keep existing domain folders. Add clearer contracts between them.

Recommended target shape:

- `src/pipeline/`
  - still owns data ingest and feature assembly
- `src/analysis/` and `src/geometry/`
  - still own state feature generation
- `src/options/dsl/`
  - remains the executable contract
- `src/options/llm/`
  - limited to protocol translation and prompt strategies
- `src/physics/`
  - remains specialist-model implementation
- new shared contract layer
  - either `src/contracts/` or `src/core/`
  - owns:
    - state schema
    - option query schema
    - evaluation record schema
    - specialist-model interface
- new experiment layer
  - either `src/eval/` or `scripts/experiments/`
  - owns reproducible benchmark entry points

Why this is the right reconstruction:

- it preserves the repo's current strengths
- it reduces hidden coupling
- it makes new backends pluggable
- it makes the paper claim testable rather than rhetorical

## 6. Design For The Paper System

The paper system should be described as:

Human -> LLM -> DSL -> Model Router -> Specialist Backend -> Evidence-Aware Output

With concrete semantics:

1. Human
   - asks a regime, pricing, or scenario question in natural language
2. LLM
   - compiles intent into a constrained DSL expression
3. DSL
   - acts as the narrow waist and machine contract
4. Router
   - dispatches to SPIKAN or a comparison backend
5. Specialist backend
   - returns structured forecasts, prices, probabilities, or tension diagnostics
6. Output layer
   - returns result plus provenance:
     - model used
     - state snapshot used
     - calibration/confidence
     - failure or fallback status

This lets us make a more defensible claim:

> The innovation is the protocol architecture, not merely attaching an LLM to a quant model.

## 7. Real-Data Proof Strategy

The business case should stay narrow:

- domain: silver and related macro state
- decision object: option pricing and regime-aware outlook
- event type: regime transitions and shock windows

Recommended data stack:

1. Existing repo data
   - silver spot and macro features from the current pipeline
2. Required new market layer
   - option chain snapshots or implied-volatility surface data
   - minimum viable choice should prioritize practical accessibility, not perfect breadth
3. Event calendar
   - major silver/macro stress windows for event studies

Required proof artifacts:

- state table
- option snapshot table
- merged evaluation table
- benchmark manifest describing each backend and run configuration

## 8. Comparison Framework

We need to compare both model quality and system cost.

### Model comparisons

- SPIKAN
- Regime-adjusted Black-Scholes with non-SPIKAN volatility input
- HMM or Markov regime baseline
- tree/boosting baseline on the same state features
- simple neural sequence baseline if time permits

### Protocol comparisons

- LLM + DSL + router
- fixed-form template interface
- hand-coded direct integration for one specialist backend

### What the comparison should show

- a competitive backend can be swapped in
- but the LLM+DSL contract lowers the cost of doing so
- the workflow stays stable even when the backend changes

## 9. Milestones

### Milestone 1. Freeze The Contract

Deliverables:
- final paper claim
- state schema
- option query schema
- backend interface

Acceptance:
- one architecture diagram
- one canonical example query
- one agreed experiment table

### Milestone 2. Reconstruct The Repo Around Explicit Interfaces

Deliverables:
- shared contracts
- cleaner routing path from DSL executor to backend
- clear separation between research scripts and benchmark scripts

Acceptance:
- one end-to-end run path with no ambiguous ownership

### Milestone 3. Build The Real-Data Evaluation Path

Deliverables:
- option data ingestion or import contract
- merged state-plus-options dataset
- benchmark runner for backends

Acceptance:
- reproducible train, validation, and test windows

### Milestone 4. Measure Protocol Quality

Deliverables:
- natural-language task set
- DSL validity benchmark
- recovery benchmark

Acceptance:
- valid DSL rate, execution success rate, and error taxonomy

### Milestone 5. Measure Quantitative Quality

Deliverables:
- forecasting and pricing benchmark results
- calibration plots
- event studies

Acceptance:
- paper-ready tables and figures

### Milestone 6. Measure Integration Cost

Deliverables:
- backend onboarding comparison
- adapter complexity comparison
- workflow stability comparison

Acceptance:
- defendable evidence that the protocol layer lowers orchestration burden

## 10. Git Branch Plan

Use the current paper branch as the umbrella planning branch:

- `paper/llm-spikan-architecture-plan`

Recommended implementation branches after planning:

- `feat/contracts-state-schema`
- `feat/options-data-contract`
- `feat/router-backend-interface`
- `feat/protocol-eval-benchmark`
- `feat/spikan-vs-baselines`
- `feat/paper-figures-and-tables`

Rules:

- `main` stays runnable
- each branch should produce one measurable artifact
- avoid mixing architecture refactor with experiment results in the same PR

## 11. Success Criteria

We should be able to demonstrate all of the following:

1. A human can ask an options/regime question in natural language.
2. The LLM converts it into valid DSL with high success rate.
3. The same DSL can be routed to SPIKAN or an alternative backend.
4. SPIKAN produces financially meaningful outputs tied to real data.
5. The system is evaluated against real baselines, not only demos.
6. The paper can argue that language is a reusable interoperability layer for quantitative models.

## 12. Immediate Next Actions

1. Freeze the exact claim and failure criteria.
2. Define the canonical state contract from existing silver, geometry, and tension features.
3. Choose the first real options dataset and lock train/validation/test periods.
4. Define the benchmark suite before adding more model complexity.
5. Keep SPIKAN as the flagship backend, but build the interface so alternative models can be dropped in.
