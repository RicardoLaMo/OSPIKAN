# Benchmark Artifact Manifest

## 1. Purpose

This document defines the versioned artifact set that must exist before the
paper can upgrade placeholders into claims.

The manuscript should bind only to stable outputs under a paper-specific
artifact root. That keeps the writing process auditable and avoids pulling
numbers from ad hoc logs.

## 2. Artifact Root

Use one paper artifact root:

- `output/paper/`

Within that root, keep one subdirectory per benchmark family:

- `output/paper/protocol/`
- `output/paper/quant/`
- `output/paper/integration_cost/`
- `output/paper/figures/`
- `output/paper/tables/`

## 3. Required Metadata For Every Benchmark Family

Each benchmark family must publish a small manifest with at least:

- `git_commit`
- `git_branch`
- `generated_at_utc`
- `benchmark_id`
- `split_id`
- `seed_policy`
- `backend_ids`
- `source_data_refs`
- `owner_doc`

Recommended file:

- `manifest.json`

## 4. Protocol Benchmark Outputs

Minimum required files:

- `output/paper/protocol/manifest.json`
- `output/paper/protocol/metrics.csv`
- `output/paper/protocol/cases.jsonl`
- `output/paper/protocol/failures.csv`
- `output/paper/protocol/examples.jsonl`

Expected fields in `metrics.csv`:

- `metric`
- `value`
- `split`
- `n_cases`
- `notes`

Expected key metrics:

- valid DSL rate
- one-shot executable rate
- final executable rate
- semantic match rate
- retry burden
- controlled-failure rate

## 5. Quantitative Benchmark Outputs

Minimum required files:

- `output/paper/quant/manifest.json`
- `output/paper/quant/leaderboard.csv`
- `output/paper/quant/stress_slices.csv`
- `output/paper/quant/calibration.csv`
- `output/paper/quant/ablations.csv`
- `output/paper/quant/casebook.csv`

Expected key groupings:

- backend
- task
- split
- primary metric
- calibration metric
- stress-period slice

## 6. Integration-Cost Outputs

Minimum required files:

- `output/paper/integration_cost/manifest.json`
- `output/paper/integration_cost/summary.csv`
- `output/paper/integration_cost/adapter_inventory.csv`
- `output/paper/integration_cost/task_log.jsonl`

Expected key metrics:

- adapter LOC
- files touched
- custom modules added
- time to first passing benchmark run
- number of backend-specific conditionals

## 7. Figure And Table Artifacts

If a figure or table is generated programmatically, store the generated version
under:

- `output/paper/figures/`
- `output/paper/tables/`

Use manuscript-aligned stable ids:

- `fig_architecture`
- `fig_state_timeline`
- `fig_protocol_trace`
- `fig_quant_comparison`
- `fig_integration_cost`
- `tab_contracts`
- `tab_protocol`
- `tab_quant`
- `tab_ablations`
- `tab_integration_cost`

## 8. Promotion Rule

A manuscript placeholder may be upgraded only when all are true:

1. the source artifact exists under `output/paper/`
2. the artifact manifest includes commit and split metadata
3. the manuscript binding map points to the exact artifact path
4. the evidence matrix says the metric is sufficient for the claim

## 9. Non-Qualifying Evidence

These do not qualify on their own for manuscript promotion:

- terminal screenshots
- notebook cells without exported artifacts
- one-off console metrics
- values copied manually from exploratory runs

## 10. Practical Outcome

This manifest turns the paper process into the same kind of contract-driven
system as the architecture itself:

- contracts define interfaces
- benchmark artifacts define evidence
- the manuscript consumes both through stable bindings
