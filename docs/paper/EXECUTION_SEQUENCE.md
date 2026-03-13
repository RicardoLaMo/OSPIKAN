# Execution Sequence

## 1. Purpose

This is the exact next-step order for implementation after the planning phase.

It is intentionally narrow. The goal is to avoid parallel speculative refactors.

## 2. Recommended Branch Order

### First Wave: Contracts

1. `feat/contracts-state-schema`
2. `feat/options-data-contract`
3. `feat/router-backend-interface`

Why first:

- these branches define the interfaces the rest of the system must use

### Second Wave: Evidence Harnesses

4. `feat/protocol-eval-benchmark`
5. `feat/spikan-vs-baselines`
6. `feat/integration-cost-study`

Why second:

- these branches turn architecture into measurable claims

### Third Wave: Paper Freeze

7. `feat/paper-final-synthesis`

Why last:

- the paper package should summarize stable evidence, not moving targets

## 3. Recommended Work Pattern

For each implementation branch:

1. add minimal types and tests first
2. adapt one live path end-to-end
3. run benchmark or validation artifact
4. merge only after artifact exists

This prevents the repo from drifting into architecture notes without execution.

## 4. Branch Scope Rules

Each branch should own one primary question:

- state branch:
  - what is the canonical market-state record?
- option branch:
  - what is the canonical option snapshot and join?
- router branch:
  - how do queries reach interchangeable backends?
- protocol branch:
  - how reliable is NL-to-DSL?
- quant branch:
  - does SPIKAN beat meaningful baselines?
- cost branch:
  - does the architecture reduce integration burden?

If a branch starts answering two of these questions at once, it is too broad.

## 5. Merge Discipline

Recommended merge discipline:

- merge only after acceptance gate is satisfied
- keep `main` runnable
- do not merge speculative benchmark branches without reproducible outputs
- keep generated large artifacts out of git unless intentionally versioned

## 6. End-State Definition

The planning program is complete only when:

- the contracts are implemented
- the benchmark harnesses run
- the evidence matrix is filled
- the final paper package has stable figures and tables

That is the end of the planning-to-proof path.
