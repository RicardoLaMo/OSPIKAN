# Manuscript Binding Map

## 1. Purpose

This document maps manuscript claims, tables, figures, and placeholders to the
exact documents or benchmark artifacts that are allowed to feed them.

It is the paper-side equivalent of an interface contract.

## 2. Binding Rules

- use one stable id per manuscript item
- bind each item to one primary source of truth
- keep preliminary evidence separate from final benchmark evidence
- do not promote placeholders without a matching artifact path

## 3. Figures

| Item ID | Manuscript Role | Source Of Truth | Current Status |
| --- | --- | --- | --- |
| `fig_architecture` | narrow-waist system overview | `docs/paper/PAPER_PACKAGE_PLAN.md` then generated file under `output/paper/figures/fig_architecture.*` | placeholder |
| `fig_state_timeline` | state, regime, and tension evolution | `output/paper/figures/fig_state_timeline.*` from state benchmark artifacts | placeholder |
| `fig_protocol_trace` | NL to DSL to backend execution example | `output/paper/protocol/examples.jsonl` and generated `output/paper/figures/fig_protocol_trace.*` | placeholder |
| `fig_quant_comparison` | headline model comparison | `output/paper/quant/leaderboard.csv` and generated `output/paper/figures/fig_quant_comparison.*` | placeholder |
| `fig_integration_cost` | onboarding complexity comparison | `output/paper/integration_cost/summary.csv` and generated `output/paper/figures/fig_integration_cost.*` | placeholder |

## 4. Tables

| Item ID | Manuscript Role | Source Of Truth | Current Status |
| --- | --- | --- | --- |
| `tab_contracts` | canonical interfaces summary | `docs/paper/STATE_CONTRACT_SPEC.md`, `docs/paper/OPTION_DATA_CONTRACT_SPEC.md`, `docs/paper/BACKEND_INTERFACE_SPEC.md` | draftable now |
| `tab_protocol` | NL-to-DSL reliability metrics | `output/paper/protocol/metrics.csv` | placeholder |
| `tab_quant` | quantitative leaderboard | `output/paper/quant/leaderboard.csv` | placeholder |
| `tab_ablations` | no-geometry / no-physics / reduced-state comparisons | `output/paper/quant/ablations.csv` | placeholder |
| `tab_integration_cost` | onboarding and adapter complexity | `output/paper/integration_cost/summary.csv` | placeholder |
| `tab_preliminary_evidence` | repo evidence already safe to cite | `docs/thesis/RESULTS_SUMMARY_FOR_THESIS.md` and `docs/project_status/IMPLEMENTATION_PROGRESS.md` | draftable now |

## 5. Placeholder Metrics

| Placeholder | Intended Source | Promotion Condition |
| --- | --- | --- |
| `[PROTOCOL_VALID_DSL_RATE]` | `output/paper/protocol/metrics.csv` | protocol corpus frozen and executable-rate audit complete |
| `[PROTOCOL_ONE_SHOT_EXEC_RATE]` | `output/paper/protocol/metrics.csv` | one-shot execution policy frozen and logged |
| `[PROTOCOL_FINAL_EXEC_RATE]` | `output/paper/protocol/metrics.csv` | retry policy frozen and final execution audit complete |
| `[PROTOCOL_SEMANTIC_MATCH_RATE]` | `output/paper/protocol/metrics.csv` | semantic rubric frozen and annotated sample reviewed |
| `[PROTOCOL_RETRY_BURDEN]` | `output/paper/protocol/metrics.csv` | retry accounting exported per corpus split |
| `[SPIKAN_PRIMARY_TASK_METRIC]` | `output/paper/quant/leaderboard.csv` | baseline parity benchmark complete |
| `[SPIKAN_CALIBRATION_METRIC]` | `output/paper/quant/calibration.csv` | calibration metric and binning policy frozen |
| `[SPIKAN_STRESS_SLICE_METRIC]` | `output/paper/quant/stress_slices.csv` | stress-slice definition frozen and reported |
| `[BASELINE_PRIMARY_TASK_METRIC]` | `output/paper/quant/leaderboard.csv` | baseline leaderboard exported on identical split |
| `[BASELINE_CALIBRATION_METRIC]` | `output/paper/quant/calibration.csv` | baseline calibration exported on identical split |
| `[BASELINE_STRESS_SLICE_METRIC]` | `output/paper/quant/stress_slices.csv` | baseline stress-slice export complete |
| `[TABULAR_PRIMARY_TASK_METRIC]` | `output/paper/quant/leaderboard.csv` | tabular baseline leaderboard exported on identical split |
| `[TABULAR_CALIBRATION_METRIC]` | `output/paper/quant/calibration.csv` | tabular calibration export complete |
| `[TABULAR_STRESS_SLICE_METRIC]` | `output/paper/quant/stress_slices.csv` | tabular stress-slice export complete |
| `[ABLATION_NO_GEOMETRY_DELTA]` | `output/paper/quant/ablations.csv` | ablation family complete with identical split rules |
| `[INTEGRATION_COST_ADAPTER_LOC]` | `output/paper/integration_cost/summary.csv` | adapter diff inventory frozen |
| `[INTEGRATION_COST_FILES_TOUCHED]` | `output/paper/integration_cost/summary.csv` | onboarding task logs reviewed |
| `[INTEGRATION_COST_BRANCH_COUNT]` | `output/paper/integration_cost/summary.csv` | branch instrumentation and comparison policy frozen |
| `[INTEGRATION_COST_TIME_TO_FIRST]` | `output/paper/integration_cost/summary.csv` | time-to-first-benchmark measurement exported |
| `[GLUE_ADAPTER_LOC]` | `output/paper/integration_cost/summary.csv` | backend-specific glue comparator exported |
| `[GLUE_FILES_TOUCHED]` | `output/paper/integration_cost/summary.csv` | backend-specific glue comparator exported |
| `[GLUE_BRANCH_COUNT]` | `output/paper/integration_cost/summary.csv` | backend-specific glue comparator exported |
| `[GLUE_TIME_TO_FIRST]` | `output/paper/integration_cost/summary.csv` | backend-specific glue comparator exported |

## 6. Preliminary Evidence Lane

These manuscript statements are allowed to use current repo docs now:

- existing walk-forward silver validation values
- current passing-test count
- current high-fidelity SPIKAN geometry alignment
- current PDE residual stabilization

These must remain explicitly labeled as:

- preliminary evidence
- subsystem evidence
- prior-track evidence

## 7. Review Use

Before merging any evidence-backed manuscript edit:

1. confirm the changed sentence points to a binding-map item
2. confirm the source artifact exists
3. confirm the artifact also appears in the benchmark manifest
4. confirm the claim strength matches the evidence matrix
