# Integration Cost Study

## 1. Purpose

This document defines how to measure the systems claim of the paper:

> A constrained language and routing layer lowers the cost of connecting humans
> and specialist quantitative backends compared with backend-specific workflow glue.

This is the claim that makes the project more than:

- a quant model demo
- an LLM UI wrapper
- or a finance prompt-engineering exercise

## 2. Why This Study Matters

A competing backend may equal or beat SPIKAN on a narrow metric.

That does not invalidate the project if the repo can show:

- faster onboarding
- lower interface complexity
- cleaner backend substitution
- more stable human-facing workflows

This study is the evidence for that advantage.

## 3. What Should Be Compared

Three integration styles should be compared.

### Style A. DSL + Router + Backend Adapter

Characteristics:

- natural-language front end optional
- DSL is the machine contract
- backend plugged through capability adapter

This is the target architecture.

### Style B. Template-Only Front End

Characteristics:

- fixed forms or rigid parameter mapping
- limited flexibility
- lower ambiguity, but weaker expressiveness

This is a useful systems baseline.

### Style C. Backend-Specific Glue

Characteristics:

- custom handler for each model
- custom I/O shaping
- custom business logic path

This is what the paper argues against.

## 4. Measurement Units

The study should measure onboarding a new backend for a fixed task set.

Suggested fixed task set:

- one `OUTLOOK` path
- one `PRICE` path
- one `REGIME_PROB` or equivalent path

This is enough to compare integration burden without overbuilding.

## 5. Primary Metrics

### `adapter_loc`

Lines of adapter or glue code needed to onboard the backend.

### `files_touched`

Number of repo files that must be edited to make the backend executable.

### `custom_branches_added`

Number of backend-specific conditional branches introduced into the router or executor.

### `time_to_first_executable_run`

Elapsed engineering time until one benchmark query can be executed.

### `time_to_first_benchmark_result`

Elapsed engineering time until one reproducible benchmark output exists.

### `interface_surface_area`

Count of distinct interface objects or bespoke translation layers required.

## 6. Secondary Metrics

- documentation overhead
- test count added
- number of hidden assumptions discovered late
- number of custom state transformations

These are lower priority, but they help interpret the primary metrics.

## 7. Fair Comparison Rules

The integration study should be run under fixed conditions:

- same benchmark task set
- same state contract
- same option-data contract where relevant
- same expected output schema
- same backend target capability

Otherwise, one approach will look simpler only because it is doing less.

## 8. Evidence Sources

The study should use repo-native evidence wherever possible:

- git diff stats
- branch change summaries
- number of modules touched
- adapter file counts
- benchmark logs

This is stronger than subjective engineering impressions.

## 9. Suggested Backend Cases

The first study should compare onboarding:

### Case 1. SPIKAN

Why:

- flagship backend

### Case 2. Markov / HMM baseline

Why:

- structurally different, but financially plausible

### Case 3. Simple pricing baseline

Why:

- exposes the difference between a specialist backend and a lightweight fallback

This gives a good spread without requiring too many integrations at once.

## 10. Expected Output Tables

### Table A. Integration Cost Summary

Columns:

- backend
- integration style
- adapter LOC
- files touched
- custom branches
- time to executable run
- time to benchmark result

### Table B. Workflow Stability

Columns:

- task
- DSL-mediated success
- template-only success
- direct-glue success
- notes on failure mode

## 11. Qualitative Evidence

The paper should also capture:

- where backend-specific glue became brittle
- where the DSL contract reduced churn
- where the router abstraction absorbed complexity cleanly

This should appear as short case notes, not vague narrative.

## 12. Decision Rule

The systems claim is credible if the DSL-mediated architecture shows:

- lower integration burden than direct glue for at least one non-trivial backend
- stable output schema across at least two backends
- limited router churn when adding a new backend

If not, the paper should narrow the claim to conceptual architecture rather than measured engineering advantage.

## 13. Acceptance Criteria

The integration-cost study is ready when:

1. one fixed task set exists
2. one measurement rubric exists
3. one side-by-side backend onboarding comparison exists
4. results are grounded in diffs, files, and runtime-to-first-result rather than only opinion
