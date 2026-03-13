# Experiment Design

## 1. Goal

Prove or falsify this claim:

> An LLM-mediated DSL can serve as a stable protocol between humans and
> specialist financial models, reducing orchestration cost while preserving
> quantitative rigor in regime-aware option workflows.

This requires evidence on three axes:

1. Protocol quality
2. Quantitative quality
3. Integration cost

## 2. Core Hypotheses

### H1. Protocol Hypothesis

Natural-language requests can be translated into valid executable DSL with high
success and low recovery burden.

### H2. Quant Hypothesis

A regime-aware specialist backend such as SPIKAN can produce useful option and
outlook outputs on real data that outperform simpler baselines on selected tasks.

### H3. Systems Hypothesis

The DSL contract lowers backend integration cost relative to hand-built,
backend-specific orchestration paths.

## 3. Data Design

### State Data

Use the current silver pipeline as the base state source:

- silver returns and momentum
- macro signals such as DXY, rates, equities, VIX
- geometry features such as Ricci and MST stress
- GA features
- manifold tension diagnostics where available

### Options Data

Minimum viable requirement:

- option chain snapshots or implied-volatility surface data aligned to the same
  trading dates as the state table

Recommended design:

- start with one liquid silver-linked options venue or proxy
- keep the first benchmark narrow rather than broad
- focus on data quality and alignment before model breadth

### Labels

We should define labels at two levels:

1. Market-state labels
   - regime transition
   - shock/no-shock
   - direction over horizon `h`
2. Option-task labels
   - next-horizon implied-vol move
   - option mid-price error
   - realized move relevance for scenario ranking

## 4. Benchmark Tasks

### Task A. NL -> DSL

Input:
- natural-language task set

Output:
- DSL command

Metrics:
- valid DSL rate
- one-shot success rate
- recovery success rate
- execution success rate
- median retries

### Task B. Regime/Stress Forecasting

Input:
- state features at time `t`

Output:
- probability of stress or regime transition by `t+h`

Metrics:
- ROC-AUC
- PR-AUC
- Brier score
- calibration slope/intercept
- Precision@K for top-risk windows

### Task C. Option Pricing / Surface Quality

Input:
- state features plus option contract metadata

Output:
- price, implied vol, or surface point

Metrics:
- MAE and RMSE vs market mid or IV
- MAPE where appropriate
- calibration by regime
- error during stress vs non-stress windows

### Task D. Scenario / Outlook Ranking

Input:
- natural-language or DSL scenario query

Output:
- ranked outlook or what-if output

Metrics:
- directional hit rate
- regret under wrong-ranking cases
- event-study usefulness around major shocks

### Task E. Integration Cost

Input:
- requirement to onboard a new backend

Output:
- working route from DSL to backend result

Metrics:
- adapter LOC
- number of files touched
- time to first executable benchmark
- number of custom glue functions
- number of backend-specific branches in the executor/router

## 5. Comparison Matrix

### Specialist backends

- `SPIKAN`
  - flagship backend
  - tests the physics-informed claim
- `Black-Scholes + exogenous volatility`
  - simple finance baseline
- `Markov/HMM regime model`
  - interpretable regime baseline
- `Gradient boosting or random forest`
  - tabular ML baseline on the same state vector
- `Sequence model`
  - optional stretch baseline if time and data quality allow

### Protocol baselines

- `LLM + DSL`
- `template-driven form input`
- `direct backend-specific glue code`

This creates two independent questions:

1. Is SPIKAN better than simpler quantitative baselines for the chosen tasks?
2. Is the LLM+DSL protocol better than ad hoc orchestration for usability and extensibility?

## 6. Experimental Controls

To keep the claim defensible:

- use strict time splits
- freeze feature availability at time `t`
- avoid forward leakage in shock labeling and scenario construction
- report performance separately in calm and stress periods
- evaluate calibration, not only ranking
- predefine failure cases where the LLM should abstain or route to a fallback

## 7. Paper Figures

The paper should aim for these core figures:

1. Architecture figure
   - Human -> LLM -> DSL -> Router -> Specialist backend
2. State figure
   - geometry/regime/tension view over time
3. Protocol figure
   - NL request to DSL to executable trace
4. Benchmark figure
   - SPIKAN vs baselines on key metrics
5. Cost figure
   - backend onboarding complexity with and without DSL

## 8. Ablations

Required ablations:

1. No LLM, template-only interface
2. LLM without DSL constraint
3. SPIKAN without geometry features
4. SPIKAN without physics loss
5. Options baseline without regime features
6. Narrative priors without manifold tension gating

These ablations matter because they isolate the contribution:

- DSL vs free-form prompting
- geometry vs plain tabular signals
- physics-informed backend vs generic model
- narrative integration vs pure price-state inference

## 9. Decision Rules

The project should move forward only if:

- protocol metrics are strong enough to trust execution
- quantitative metrics beat simple baselines on at least one primary task
- calibration is acceptable in stress periods
- integration-cost evidence is materially favorable

If not, the paper claim should be narrowed:

- from "better end-to-end decision engine"
- to "better interoperability architecture for specialist-model workflows"

## 10. Recommended First Benchmark Slice

Do not start with the full universe.

Start with:

- one asset focus
- one options dataset
- one primary horizon
- one regime task
- one pricing task
- three quantitative baselines
- one protocol benchmark set

That is enough to produce the first credible result table without overbuilding.
