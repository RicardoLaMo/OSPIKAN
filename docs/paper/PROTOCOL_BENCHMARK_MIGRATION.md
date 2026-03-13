# Protocol Benchmark Migration

## 1. Purpose

This document explains how to move from the current prompt-and-unit-test setup
to a reproducible benchmark for the protocol layer.

## 2. Current Starting Point

The repo already has:

- prompt templates
- few-shot examples
- validator-backed retry logic
- mocked translator tests

That is a strong engineering base.

What is missing is a benchmark harness that treats NL-to-DSL as a measurable
system component.

## 3. Migration Sequence

### Step 1. Freeze the benchmark schema

Branch:
- `feat/protocol-eval-benchmark`

Artifacts:

- corpus item format
- metric definitions
- error taxonomy

Gate:
- benchmark logic is defined before new prompt tuning begins

### Step 2. Build the first corpus

Expected implementation branch later:
- `feat/protocol-corpus`

Goal:
- create a balanced set of natural-language requests across task families

Touch points likely:

- `tests/`
- benchmark data directory
- protocol docs

Gate:
- corpus includes normal, paraphrased, and adversarial requests

### Step 3. Add benchmark runner

Expected implementation branch later:
- `feat/protocol-benchmark-runner`

Goal:
- run the translator against the corpus
- store raw outputs and scored outputs

Touch points likely:

- `src/options/llm/nl_to_dsl.py`
- benchmark scripts
- result artifacts

Gate:
- one command produces a benchmark report reproducibly

### Step 4. Add semantic scorer

Expected implementation branch later:
- `feat/protocol-semantic-scorer`

Goal:
- move beyond parse/validate into semantic correctness checks

Gate:
- each task family has constraint-aware checks

### Step 5. Add model and prompt comparison support

Expected implementation branch later:
- `feat/protocol-comparison`

Goal:
- compare prompt variants, LLM variants, and retry policies

Gate:
- results are tracked in one comparable format

## 4. Likely Implementation Touch Points

- `src/options/llm/prompts.py`
- `src/options/llm/nl_to_dsl.py`
- `tests/test_nl_to_dsl.py`
- new benchmark runner under `scripts/` or `src/eval/`

## 5. Important Guardrails

Do not let the protocol benchmark drift into free-form judging.

The scorer should rely primarily on:

- parser validity
- validator validity
- explicit semantic constraints
- executor compatibility

This keeps the benchmark auditable.

## 6. Success Conditions

The migration succeeds if:

1. prompt changes can be measured, not only felt
2. retry logic can be evaluated systematically
3. unsafe translations are visible in taxonomy reports
4. the protocol claim is backed by reproducible evidence

## 7. Failure Signs

The migration is drifting if:

- benchmark items are too close to prompt examples
- exact-string match is treated as the only measure
- semantic mistakes are hidden by permissive scoring
- unsupported queries are quietly dropped instead of scored
