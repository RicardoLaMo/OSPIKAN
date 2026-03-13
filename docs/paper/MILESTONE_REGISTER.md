# Milestone Register

## 1. Purpose

This register defines the end-to-end milestone sequence from planning to a
paper-ready evidence package.

## 2. Milestones

### M0. Planning Freeze

Status:
- complete in planning branches

Deliverables:

- architecture plan
- experiment design
- evidence matrix

Dependencies:
- none

### M1. State Contract Implementation

Branch family:
- `feat/contracts-state-schema`

Deliverables:

- canonical market-state types
- state builder from current features
- projection tests

Gate:

- state contract used in both training and inference paths

### M2. Option Data Contract Implementation

Branch family:
- `feat/options-data-contract`

Deliverables:

- canonical option snapshot types
- import and cleaning path
- joined state-option dataset

Gate:

- one reproducible real-data option benchmark slice

### M3. Router / Backend Interface Implementation

Branch family:
- `feat/router-backend-interface`

Deliverables:

- backend registry
- request normalization
- standardized backend result schema

Gate:

- one query can run against at least two backends

### M4. Protocol Benchmark Implementation

Branch family:
- `feat/protocol-eval-benchmark`

Deliverables:

- benchmark corpus
- runner
- scored report

Gate:

- protocol metrics are reproducible and comparable across prompt/model variants

### M5. Quantitative Benchmark Implementation

Branch family:
- `feat/spikan-vs-baselines`

Deliverables:

- shared evaluation harness
- baseline runners
- leaderboard and calibration outputs

Gate:

- SPIKAN and baselines are evaluated on equal footing

### M6. Integration Cost Study Implementation

Branch family:
- `feat/integration-cost-study`

Deliverables:

- onboarding measurement rubric
- backend onboarding case studies
- git-backed evidence

Gate:

- systems contribution is measurable, not rhetorical

### M7. Paper Package Freeze

Branch family:
- `feat/paper-final-synthesis`

Deliverables:

- final figure set
- final table set
- methods summary
- results summary

Gate:

- all core claims map to concrete evidence artifacts

## 3. Merge Order

Recommended merge order:

1. state contract
2. option data contract
3. router/backend interface
4. protocol benchmark
5. quantitative benchmark
6. integration-cost study
7. paper package freeze

This order minimizes rework because later branches depend on the earlier contracts.

## 4. Hard Dependencies

- protocol benchmark depends on stable request and DSL semantics
- quantitative benchmark depends on stable state contract
- option-aware benchmark depends on option contract
- integration-cost study depends on router/backend architecture
- final paper package depends on all evidence layers

## 5. Minimum Paper-Ready Condition

The project is ready for a serious paper draft only after:

- M4 complete
- M5 complete
- M6 complete

Without those, the paper remains a design proposal or partial prototype.
