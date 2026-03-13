# Protocol Benchmark Specification

## 1. Purpose

This document defines how to evaluate the LLM layer as a protocol compiler.

The point is not to show that the LLM is fluent. The point is to show that it
can reliably translate human requests into executable DSL under controlled
constraints.

This benchmark is the direct evidence for the claim:

> The LLM is a semantic coordination layer that speaks natural language outward
> and DSL inward.

## 2. Current Repo Reality

The repo already has a strong base for this benchmark:

- prompts in `src/options/llm/prompts.py`
- translation and retry logic in `src/options/llm/nl_to_dsl.py`
- unit tests in `tests/test_nl_to_dsl.py`
- parser and validator downstream

But current validation is still mostly:

- prompt-shape checking
- mocked happy-path translation
- a few retry tests

That is not yet a protocol benchmark.

## 3. Benchmark Objective

Measure whether natural-language requests can be converted into valid,
executable, semantically faithful DSL with predictable recovery behavior.

## 4. Benchmark Unit

Each benchmark item should contain:

- `query_id`
- `task_family`
- `natural_language_input`
- `expected_capability`
- `expected_constraints`
- `acceptable_dsl_patterns`
- `should_abstain`
- optional `gold_dsl`

The benchmark should not require only one exact string when multiple DSL forms
are semantically equivalent.

## 5. Task Families

The benchmark corpus should include at least these families:

### A. Pricing

Examples:

- vanilla `PRICE`
- finance-native `QUOTE`
- explicit rate/dividend
- regime-conditioned volatility

### B. Regime Queries

Examples:

- current regime
- transition probability
- odds wording vs probability wording

### C. Surface And Risk

Examples:

- vol surface
- covariance or basket-risk phrasing

### D. Scenario Queries

Examples:

- what-if regime shift
- partial parameter overrides
- metric subset requests

### E. Outlook Queries

Examples:

- horizon outlook
- backdrop override
- directional and stress language

### F. Explanation Queries

Examples:

- explain regime features
- feature aliases and finance shorthand

### G. Adversarial / Ambiguous Queries

Examples:

- missing asset
- contradictory tenor wording
- unsupported asset
- multi-intent query that should be rejected or decomposed

This family matters because the protocol claim is weakest when the input is messy.

## 6. Primary Metrics

### `valid_dsl_rate`

Fraction of outputs that parse and validate.

### `one_shot_success_rate`

Fraction of prompts that succeed on the first generation.

### `final_executable_rate`

Fraction of prompts that succeed after the full retry budget.

### `semantic_match_rate`

Fraction of successful outputs that satisfy the expected capability and required constraints.

Examples of required constraints:

- correct asset
- correct option type
- correct horizon bucket
- correct strike/spot fields
- correct verb family

### `retry_burden`

- average retries on successful tasks
- distribution of retries

### `abstain_or_safe_fail_rate`

Fraction of unsupported or ambiguous prompts that fail in a controlled way.

## 7. Secondary Metrics

- normalized edit distance to one canonical DSL form
- unsupported-feature trigger rate
- ambiguity-classification quality
- average token usage if instrumented later

These are useful, but not primary evidence for the paper.

## 8. Error Taxonomy

Every failure should be assigned a category.

Recommended categories:

- syntax failure
- validation failure
- wrong capability
- wrong asset
- wrong regime
- wrong tenor
- wrong parameter normalization
- unsupported request not safely rejected
- multi-intent collapse

Without this taxonomy, the benchmark becomes a pass/fail blob and is much less useful.

## 9. Acceptance Gates

For the paper claim to be credible, the protocol benchmark should show:

- high `valid_dsl_rate`
- high `final_executable_rate`
- strong `semantic_match_rate`
- low retry burden
- safe behavior on unsupported prompts

Suggested internal bar for a first credible result:

- `valid_dsl_rate >= 0.90`
- `final_executable_rate >= 0.95`
- `semantic_match_rate >= 0.85`

These are not publication laws, but they are good internal discipline.

## 10. Corpus Design

The first benchmark set should be small but representative.

Recommended minimum:

- 60 to 100 prompts
- 8 to 15 prompts per major family
- at least 10 adversarial or ambiguous prompts

Corpus balance matters more than raw size in the first pass.

## 11. Gold Standard Design

The benchmark should use three levels of checking:

### Level 1. Parse + validate

Does the output become a valid DSL command?

### Level 2. Semantic constraints

Does the output preserve the required meaning of the input?

### Level 3. Execution compatibility

Can the output run through the executor or request normalizer without manual repair?

This three-layer design is much better than requiring exact-string match only.

## 12. Benchmark Artifacts

The benchmark should produce:

- one corpus file
- one result table
- one failure taxonomy report
- one examples file with:
  - success cases
  - retry-recovered cases
  - controlled failures
  - unsafe failures

## 13. Relationship To The Paper

This benchmark proves the LLM side of the story.

It does not prove the quantitative backend claim. That comes separately.

The paper should treat this as:

- protocol evidence
- not predictive evidence

## 14. Acceptance Criteria

The protocol benchmark is ready when:

1. one corpus exists
2. one reproducible runner exists
3. parse, validation, semantic, and execution metrics are all reported
4. failure taxonomy is visible
5. results can be compared across prompt versions or model variants
