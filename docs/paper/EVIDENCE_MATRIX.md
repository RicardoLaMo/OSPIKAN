# Evidence Matrix

## 1. Purpose

This matrix links the paper claim to specific evidence that must exist in the
repo if the claim is to be defendable.

## 2. Main Claim

> A constrained language layer can serve as a reusable interoperability
> contract between humans and specialist financial models, reducing integration
> complexity while preserving quantitative rigor.

This breaks into four subclaims.

## 3. Claim-To-Evidence Map

### Claim A. The LLM behaves like a protocol compiler rather than a free-form assistant.

Evidence required:

- benchmarked NL-to-DSL task set
- DSL validity rate
- execution success rate
- retry/recovery statistics
- examples of abstain or controlled failure

Repo artifacts expected:

- protocol benchmark report
- query corpus
- error taxonomy

Failure condition:

- the LLM frequently emits invalid or non-executable DSL
- success depends on manual repair rather than the defined retry/recovery path

### Claim B. The DSL is the stable narrow waist of the system.

Evidence required:

- same DSL query executable against more than one backend
- same output schema across backends
- limited backend-specific branching in the query path

Repo artifacts expected:

- backend interface spec
- router comparison examples
- integration tests for backend-swappable execution

Failure condition:

- each backend requires custom query semantics or custom user prompts
- the DSL contract is not actually shared

### Claim C. SPIKAN is a real specialist finance engine, not decorative math.

Evidence required:

- real-data benchmark against finance-relevant baselines
- calibration evidence
- stress-period evidence
- regime-aware or option-relevant metrics

Repo artifacts expected:

- benchmark table
- calibration plot
- event-study result
- ablation on geometry or physics loss

Failure condition:

- SPIKAN does not beat simple baselines on any primary task
- performance is unstable in stress periods
- outputs are hard to tie back to real option or regime decisions

### Claim D. The protocol architecture lowers orchestration cost.

Evidence required:

- at least one backend onboarding comparison
- file-touch and adapter complexity comparison
- time-to-first-benchmark comparison

Repo artifacts expected:

- cost comparison table
- backend adapter diff summaries
- engineering notes or reproducibility logs

Failure condition:

- the protocol layer adds more custom engineering than it removes
- the backend integration burden remains roughly unchanged

## 4. Required Tables

### Table 1. Protocol Metrics

Columns should include:

- task family
- one-shot valid DSL rate
- final executable rate
- retry count
- abstain rate

### Table 2. Quantitative Benchmarks

Columns should include:

- backend
- task
- split
- primary metric
- calibration metric
- stress-period metric

### Table 3. Integration Cost

Columns should include:

- backend
- adapter LOC
- files touched
- time to first run
- custom branches added

## 5. Required Figures

### Figure 1. Architecture

Must show:

- human input
- LLM protocol layer
- DSL contract
- router
- specialist backend(s)
- evidence-aware output

### Figure 2. State And Tension Timeline

Must show:

- regime or stress periods
- geometry features
- manifold tension
- option-relevant state changes

### Figure 3. Protocol Trace

Must show:

- natural-language query
- generated DSL
- backend execution
- structured output

### Figure 4. Quantitative Comparison

Must show:

- SPIKAN vs baselines on one primary task

### Figure 5. Integration Cost

Must show:

- how onboarding complexity changes with and without DSL mediation

## 6. Required Ablations

The evidence package should include:

1. LLM with DSL vs LLM without DSL
2. DSL router with SPIKAN vs DSL router with baseline backend
3. SPIKAN with geometry vs SPIKAN without geometry
4. SPIKAN with physics loss vs SPIKAN without physics loss
5. narrative prior path with tension gating vs without tension gating

These ablations prevent the paper from collapsing into a vague bundled story.

## 7. Minimum Acceptance Package

The claim is strong enough to draft into a paper only if all of the following
exist together:

- one valid protocol benchmark
- one backend-swappable execution path
- one real-data quantitative benchmark
- one stress-period evaluation slice
- one integration-cost comparison

If any of these are missing, the paper should be reframed as:

- architecture proposal
- partial prototype
- or proof-of-concept, not a full validated system
