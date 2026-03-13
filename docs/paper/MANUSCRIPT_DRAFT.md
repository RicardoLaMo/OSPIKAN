# Manuscript Draft

## Title

Language as a Protocol Layer for Specialist Financial Models:
LLM-Mediated DSL Routing for Regime-Aware Options and SPIKAN Backends

## Abstract

Specialist quantitative models are often difficult to integrate into human-facing
decision workflows because they require custom interfaces, backend-specific data
plumbing, and repeated orchestration code. This problem is especially acute in
finance, where mathematically disciplined models must coexist with ambiguous
human requests, evolving market narratives, and auditable execution constraints.
We propose a language-mediated architecture in which a large language model
serves as a semantic coordination layer between human users and specialist
financial backends. Rather than allowing unconstrained natural-language control,
the system compiles user intent into a constrained domain-specific language
(DSL), which acts as the narrow waist between human reasoning and quantitative
execution.

We instantiate this architecture in a regime-aware options setting centered on
silver markets and a specialist backend called SPIKAN, a physics-informed
Kolmogorov-Arnold network designed for market shock and regime dynamics. The
repo already contains three major subsystems: a geometric market-state pipeline
for silver and macro data, an options DSL and KAN knowledge-store stack, and a
SPIKAN backend with PDE-informed training and outlook generation. Our
contribution is to reorganize these into a unified architecture with explicit
state contracts, option-data contracts, backend routing interfaces, and
reproducible evaluation paths.

The paper is designed to test three claims. First, the LLM can function as a
protocol compiler rather than a free-form assistant, translating natural
language into valid executable DSL with measurable reliability. Second, SPIKAN
can act as a specialist backend whose outputs remain grounded in financially
meaningful state variables, regime dynamics, and option-related decision tasks.
Third, a DSL-mediated router can reduce backend-integration complexity relative
to bespoke, backend-specific glue code. We define a benchmark program spanning
protocol quality, quantitative performance, calibration, stress-period
robustness, and integration cost. The resulting framework positions language not
as a replacement for quantitative modeling, but as an interoperability contract
between human intent and mathematically disciplined machine intelligence.

## 1. Introduction

Large language models are increasingly used as front ends for analytical and
decision-support systems. In many deployments, however, the LLM acts as an
unconstrained conversational wrapper around a collection of tools and models.
This pattern is flexible, but it is often brittle. The system may work for a
demo, yet remain difficult to validate, audit, extend, or compare across model
backends. These problems are amplified in financial applications, where user
requests are often ambiguous, model outputs must remain economically meaningful,
and the cost of silent failure is high.

This paper studies an alternative architecture. Instead of treating the LLM as
the decision engine, we treat it as a protocol layer between human language and
specialist quantitative models. The key mechanism is a constrained
domain-specific language that translates natural-language intent into typed,
executable commands. This design separates semantic coordination from numerical
inference. The LLM handles human-facing interpretation and query composition,
while the backend remains responsible for finance-native computation.

We instantiate this idea in a silver-market research platform that already
contains three substantial subsystems. The first is a multi-asset state
representation pipeline covering silver, macro, geometry, and regime features.
The second is an options DSL and KAN knowledge store for pricing, surfaces,
covariance, transitions, and what-if analysis. The third is SPIKAN, a
physics-informed Kolmogorov-Arnold network designed to model market shocks and
regime dynamics. The current repository demonstrates these components
individually, but the paper contribution is to reconstruct them around explicit
contracts and measurable evidence.

The business case is regime-aware options analysis. This domain is a useful test
bed because it forces interaction between ambiguous human questions, structured
contract parameters, regime-sensitive state variables, and specialist backend
reasoning. A human may ask for a stress-regime outlook, a repricing scenario, or
an options surface under changing macro conditions. A free-form LLM is
insufficient for such tasks because the outputs must remain executable and
auditable. A purely quantitative backend, on the other hand, is too rigid or
opaque for natural human interaction without substantial custom interface work.
The protocol-layer design aims to bridge that gap.

The paper advances three claims. First, the LLM can reliably compile natural
language into executable DSL under a controlled benchmark. Second, SPIKAN can
serve as a specialist backend whose outputs are quantitatively meaningful rather
than decorative. Third, the DSL-router architecture lowers integration cost when
comparing or onboarding backends. The emphasis is therefore not only predictive
performance, but also interoperability, auditability, and engineering economy.

## 2. Related Framing

The work sits at the intersection of financial modeling, programmatic interfaces
for LLM systems, and modular AI architecture. In finance, regime detection and
option pricing are traditionally addressed through statistical state-space
models, volatility models, and analytical pricing frameworks. More recently,
geometric and topology-inspired methods have been proposed for capturing market
structure, while physics-informed neural methods attempt to encode dynamic
constraints directly into learning systems.

In parallel, LLM systems research has explored tool use, code generation, and
agent-style orchestration. Many such systems rely on loosely structured model
calls or tool wrappers, often prioritizing flexibility over stability. Our
position is narrower. We do not argue that unconstrained natural language should
control specialist financial models directly. Instead, we argue that the LLM is
most useful when it compiles intent into a constrained intermediate language
that can be validated, logged, replayed, and routed across specialist backends.

This perspective makes the contribution fundamentally architectural. The key
question is not whether a language model can "do finance" in the abstract. The
question is whether language can act as the semantic boundary between humans and
specialist quantitative engines without collapsing mathematical discipline. The
repo’s DSL, state contracts, and backend interfaces are designed to answer that
question concretely.

## 3. System Overview

The system is organized around six layers.

First, the data plane produces market and macro features from silver-centered
time series. These include returns, volatility, momentum, cross-asset
relationships, and provenance metadata. Second, the state plane derives richer
representations from those inputs, including geometric features such as Ricci
curvature, minimum-spanning-tree stress, geometric algebra rotor magnitude, and
regime probabilities. Newer research additions also include dual manifold
tension metrics: structural tension measured by Gromov-Wasserstein distance and
level tension measured by Wasserstein-2 distance.

Third, the protocol plane converts human intent into machine-executable form.
Natural-language requests are translated into a constrained DSL with verbs for
pricing, regime queries, covariance, scenario analysis, surfaces, explanations,
and outlook. The parser, validator, and executor already exist in the repo.
Fourth, the backend plane executes those requests through specialist models.
These include a regime-adjusted pricing stack, a KAN knowledge store for
surfaces and transitions, and SPIKAN for dynamic outlook generation. Fifth, the
evaluation plane measures protocol quality, quantitative performance, and
backend-integration cost. Sixth, the product plane exposes results through a
user-facing REPL and report artifacts.

The key architectural idea is the narrow waist between the protocol plane and
the backend plane. Human language remains flexible above the waist. Backend
implementations remain domain-specific below it. The DSL and canonical contracts
are the stabilizing middle layer that allow these sides to evolve without
constant rewiring.

## 4. Canonical Contracts

Two contracts are central to the paper design. The first is the market-state
contract. The current codebase uses state information in several different
forms: DataFrame columns in regime code, handcrafted dictionaries in the SPIKAN
outlook adapter, and substring-selected tensors in SPIKAN training. To remove
that ambiguity, the paper introduces a canonical `MarketStateSnapshot` with
identity, base market features, macro features, geometry features, regime
features, fluid and shock features, optional manifold-tension features, and
provenance. This contract is intentionally backend-neutral. Models consume
projections from the canonical state rather than redefining state themselves.

The second contract is the option-data contract. The existing options stack has
a mature query interface, but it lacks a canonical real-data option snapshot.
The paper therefore defines an `OptionSnapshot` with contract terms, underlying
reference fields, quote fields, derived fields such as moneyness and implied
volatility, and explicit quality flags. The joined benchmark unit is a
`StateOptionSnapshot`, formed by aligning option records with canonical market
state at the same evaluation time.

These contracts do not merely organize data. They define the measurable inputs
for protocol benchmarks, quantitative backends, pricing baselines, and
integration-cost studies. They are the main architectural bridge from prototype
code to paper-grade evidence.

## 5. Protocol Layer

The protocol layer is implemented as natural-language-to-DSL compilation with
validation-backed retries. Prompt rules constrain output to one command, enforce
capitalization and parameter formats, and provide few-shot examples across the
supported verb families. A validator then checks whether the generated DSL is
syntactically and semantically executable. This design already exists in the
repo, but until now it has mostly been demonstrated through unit tests and
interactive usage rather than through a reproducible benchmark.

The paper reframes this layer as a benchmarked protocol compiler. Each natural
language input is associated with an expected capability, expected semantic
constraints, and acceptable DSL patterns. The protocol benchmark measures valid
DSL rate, one-shot success, final executable rate after retries, semantic match
rate, retry burden, and controlled-failure behavior. This is critical because
the language component should be judged by protocol reliability, not by generic
fluency.

This framing also makes the contribution more defensible. If the LLM performs
poorly on protocol metrics, the architecture claim weakens immediately. If the
protocol metrics are strong, however, then the LLM can be understood as a
specialized semantic compiler rather than a free-form advisor.

## 6. Specialist Backends

The repo contains multiple backend families even before formal modularization.
The simplest backend is a pricing path built around Black-Scholes with
regime-conditioned volatility from the KAN store. This supports option pricing,
surface queries, and scenario analysis. A second backend family comes from the
KAN knowledge store itself, which maps regime features to implied-volatility
surfaces, covariance estimates, and regime-transition probabilities. A third
backend family is SPIKAN, which consumes asset, macro, geometry, and time inputs
to produce dynamic signals and trader-facing outlook outputs. Finally, existing
regime and Markov-style functions form a natural baseline backend family.

The paper defines a common backend interface over these families. Each backend
declares capabilities, executes a canonical request using canonical state and
optional option snapshots, and returns a standardized result object with payload,
status, provenance, and backend identity. This makes backend substitution a
concrete engineering target rather than an informal aspiration.

SPIKAN is the flagship backend because it embodies the project’s strongest
domain-specific modeling claim. It is physics-informed, regime-aware, and tied
to state features derived from the broader silver geometry pipeline. But the
architecture is explicitly designed so that SPIKAN is not the only backend that
can live behind the DSL. This distinction matters for both scientific clarity
and commercial realism.

## 7. Experimental Design

The evaluation program contains three benchmark families.

The first is the protocol benchmark described above. The second is the
quantitative benchmark. This benchmark begins with shock or stress prediction,
where SPIKAN is already partially instrumented through AUC, Precision@K, MSE,
and MAE in the current training script. The quantitative package extends this by
freezing baseline parity, adding calibration metrics, reporting stress-period
performance explicitly, and requiring ablations such as no-geometry and
no-physics-loss variants. The third benchmark family is the integration-cost
study, which compares DSL-mediated backend onboarding against template-only and
backend-specific glue approaches. This study uses concrete engineering metrics
such as adapter lines of code, files touched, custom branches added, and
time-to-first-benchmark result.

Real-data evaluation is staged. The current silver pipeline already supports
state generation, and the existing results docs contain walk-forward validation
numbers for earlier regime-classification work. The paper architecture reuses
those strengths but introduces stricter evidence packaging. For the options
side, the first real-data benchmark is deliberately narrow: one underlying or
proxy, one quote frequency, one liquid tenor band, and one near-ATM plus one OTM
bucket. This narrow slice is enough to validate the full architecture without
overscoping the data problem.

The manuscript should therefore be filled from three benchmark artifact lanes:

- protocol artifacts under `output/paper/protocol/`
- quantitative artifacts under `output/paper/quant/`
- integration-cost artifacts under `output/paper/integration_cost/`

## 8. Current Repo Status

At the time of drafting, the repo is strongest at the component level rather
than the full evidence level. The DSL stack, KAN store, SPIKAN modules, regime
pipeline, and geometry methods already exist. The current codebase already
supports:

- geometric regime detection
- DSL parsing and validation
- regime-adjusted pricing
- KAN-based surfaces and transitions
- SPIKAN training and outlook generation
- research scripts for manifold tension and narrative-state mapping

The main gaps are architectural consistency and benchmark completeness. The
state path is still duplicated across several conventions. The options path still
relies partly on synthetic training assumptions. The executor still routes
through concrete component fields rather than a backend registry. And the paper
claims around protocol quality and integration cost are not yet backed by
benchmark artifacts. The planning package in `docs/paper/` is designed to close
those gaps in a controlled order.

## 9. Preliminary Repo Evidence

Although the full paper benchmark program is not complete yet, the repo already
contains two classes of evidence that are stable enough to cite as preliminary
results.

First, the silver and geometry pipeline has existing walk-forward results from
the thesis track. The current validated summary reports 48 walk-forward folds,
3,213 deduplicated out-of-sample predictions, and the following regime
classification accuracies: BASE 64.50%, MACRO 66.67%, and SECTIONAL 65.43%. The
same summary reports stable-regime Sharpe ratios of 2.354 for BASE, 2.764 for
MACRO, and 2.414 for SECTIONAL. These are not yet the main quantitative results
for the LLM+SPIKAN paper, but they show that the repo already has a credible
out-of-sample validation culture and a non-trivial state-representation baseline
on real historical data.

Second, the options and protocol subsystems already have measurable engineering
maturity. The implementation summary currently reports 205 passing tests across
DSL, KAN store, executor, LLM integration, training, and high-fidelity geometry
work. Within that package, the high-fidelity SPIKAN path reports a geometric
correlation of 0.9632 between SPIKAN latent activations and Clifford Algebra
rotor magnitudes, as well as a large reduction in PDE residual variance from
125,894 to 1.5882. These results should be presented as subsystem-level evidence
rather than as final paper benchmark outputs, but they do support the claim that
the backend and protocol stack are already more than a conceptual sketch.

The manuscript therefore distinguishes three evidence tiers:

- implemented subsystem evidence already present in the repo,
- benchmark evidence still required for the paper claim,
- future option-data and integration-cost evidence not yet bound.

## 10. Expected Results Structure

The final paper should report one table for protocol metrics, one leaderboard
table for quantitative results, one ablation table, one integration-cost table,
and a small figure set covering architecture, protocol trace, state timeline,
quantitative comparison, and integration cost. The current draft intentionally
leaves numerical placeholders open unless they are already anchored by existing
repo evidence. This is deliberate. The manuscript should only promote metrics
into the main claims once they are produced by the benchmark harnesses defined
in the planning documents.

### Protocol Metrics Table Stub

| Metric | Split | Current Value |
| --- | --- | --- |
| valid DSL rate | benchmark split | `[PROTOCOL_VALID_DSL_RATE]` |
| one-shot executable rate | benchmark split | `[PROTOCOL_ONE_SHOT_EXEC_RATE]` |
| final executable rate | benchmark split | `[PROTOCOL_FINAL_EXEC_RATE]` |
| semantic match rate | benchmark split | `[PROTOCOL_SEMANTIC_MATCH_RATE]` |
| retry burden | benchmark split | `[PROTOCOL_RETRY_BURDEN]` |

Source target:

- `output/paper/protocol/metrics.csv`

### Quantitative Leaderboard Stub

| Backend | Task | Primary Metric | Calibration | Stress Slice |
| --- | --- | --- | --- | --- |
| SPIKAN | shock or regime task | `[SPIKAN_PRIMARY_TASK_METRIC]` | `[SPIKAN_CALIBRATION_METRIC]` | `[SPIKAN_STRESS_SLICE_METRIC]` |
| Markov or HMM baseline | same task | `[BASELINE_PRIMARY_TASK_METRIC]` | `[BASELINE_CALIBRATION_METRIC]` | `[BASELINE_STRESS_SLICE_METRIC]` |
| tabular baseline | same task | `[TABULAR_PRIMARY_TASK_METRIC]` | `[TABULAR_CALIBRATION_METRIC]` | `[TABULAR_STRESS_SLICE_METRIC]` |

Source target:

- `output/paper/quant/leaderboard.csv`
- `output/paper/quant/stress_slices.csv`

### Integration-Cost Table Stub

| Integration Style | Adapter LOC | Files Touched | Custom Branches | Time To First Benchmark |
| --- | --- | --- | --- | --- |
| DSL-mediated shared contract | `[INTEGRATION_COST_ADAPTER_LOC]` | `[INTEGRATION_COST_FILES_TOUCHED]` | `[INTEGRATION_COST_BRANCH_COUNT]` | `[INTEGRATION_COST_TIME_TO_FIRST]` |
| backend-specific glue | `[GLUE_ADAPTER_LOC]` | `[GLUE_FILES_TOUCHED]` | `[GLUE_BRANCH_COUNT]` | `[GLUE_TIME_TO_FIRST]` |

Source target:

- `output/paper/integration_cost/summary.csv`

There are two possible outcome patterns. In the stronger outcome, SPIKAN
outperforms simpler baselines on at least one primary task, remains calibrated
through stress periods, and the router architecture reduces onboarding burden. In
the weaker but still publishable outcome, the protocol benchmark and
integration-cost study succeed clearly while SPIKAN performs only competitively
rather than decisively. In that case, the paper should position SPIKAN as a
plausible specialist backend within a stronger interoperability architecture,
rather than as the uniquely best model.

## 11. Artifact Binding And Reproducibility

The paper should treat benchmark artifacts the same way the system treats
backend execution: through explicit contracts. All evidence-backed claims should
resolve to versioned outputs under `output/paper/`, with manifests that record
the generating commit, split policy, backend ids, and source data references.

The binding process should follow three documents together:

- `BENCHMARK_ARTIFACT_MANIFEST.md`
- `MANUSCRIPT_BINDING_MAP.md`
- `PAPER_AUTHORING_PROCESS.md`

This gives the manuscript a stable fill path:

1. produce benchmark artifacts
2. register them in the artifact manifest
3. bind them to figures, tables, and placeholders through the binding map
4. upgrade the prose only after the evidence exists

## 12. Discussion

The main contribution of this work is architectural. Finance is a useful domain
because it makes the tradeoffs visible: free-form language is not safe enough,
but pure quantitative machinery is too rigid for broad interactive use without
substantial engineering. The DSL contract provides a narrow waist that absorbs
that tension. It gives the LLM a disciplined role and gives specialist models a
stable entry point into human-facing workflows.

This architecture also clarifies what should count as progress. Adding more
models or more prompts is not enough. Progress requires stronger contracts,
better benchmark parity, clearer provenance, and more explicit failure handling.
That is why the planning stack emphasizes state contracts, option-data contracts,
router interfaces, and integration studies rather than immediately expanding the
model zoo.

The main limitation is that the strongest full-system claim still depends on
real-data option integration and benchmark completion. Until those are finished,
the paper should be described as a structured draft and evidence program rather
than a completed empirical result. That limitation is acceptable as long as the
paper is honest about which parts are implemented, which parts are benchmarked,
and which parts are still placeholders awaiting execution.

## 13. Conclusion

We have outlined a paper architecture in which language functions as a protocol
layer between humans and specialist financial models. In the proposed system,
the LLM does not replace quantitative reasoning. Instead, it compiles natural
language into a constrained DSL that can be routed through auditable, swappable,
finance-native backends. SPIKAN serves as the flagship specialist backend, while
the surrounding contracts and benchmarks make the broader architecture testable.

The central research question is therefore not whether an LLM can "do finance"
on its own. It is whether language can serve as a reliable interoperability
contract for financially disciplined machine intelligence. The repo is now
planned around answering that question with explicit contracts, benchmarked
protocol quality, quantitative backend comparisons, and engineering-cost
evidence.

## Appendix A. Current Evidence Placeholders

These items should be replaced by benchmark outputs as implementation proceeds:

- `[PROTOCOL_VALID_DSL_RATE]`
- `[PROTOCOL_ONE_SHOT_EXEC_RATE]`
- `[PROTOCOL_FINAL_EXEC_RATE]`
- `[PROTOCOL_SEMANTIC_MATCH_RATE]`
- `[PROTOCOL_RETRY_BURDEN]`
- `[SPIKAN_PRIMARY_TASK_METRIC]`
- `[SPIKAN_CALIBRATION_METRIC]`
- `[SPIKAN_STRESS_SLICE_METRIC]`
- `[BASELINE_PRIMARY_TASK_METRIC]`
- `[BASELINE_CALIBRATION_METRIC]`
- `[BASELINE_STRESS_SLICE_METRIC]`
- `[TABULAR_PRIMARY_TASK_METRIC]`
- `[TABULAR_CALIBRATION_METRIC]`
- `[TABULAR_STRESS_SLICE_METRIC]`
- `[ABLATION_NO_GEOMETRY_DELTA]`
- `[INTEGRATION_COST_ADAPTER_LOC]`
- `[INTEGRATION_COST_FILES_TOUCHED]`
- `[INTEGRATION_COST_BRANCH_COUNT]`
- `[INTEGRATION_COST_TIME_TO_FIRST]`
- `[GLUE_ADAPTER_LOC]`
- `[GLUE_FILES_TOUCHED]`
- `[GLUE_BRANCH_COUNT]`
- `[GLUE_TIME_TO_FIRST]`

## Appendix B. Minimal Figure Checklist

The draft expects the following final figures:

1. architecture figure
2. state/regime/tension timeline
3. protocol trace figure
4. quantitative comparison figure
5. integration-cost figure

## Appendix C. Manuscript Binding Inputs

Use these sources when converting the draft into an evidence-bound manuscript:

- `BENCHMARK_ARTIFACT_MANIFEST.md`
- `MANUSCRIPT_BINDING_MAP.md`
- `PRELIMINARY_EVIDENCE_BINDING.md`
