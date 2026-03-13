# Repository Architecture

## Overview

This repository combines two connected systems:

1. A silver and macro regime-analysis pipeline that turns market histories into aligned panels, engineered features, regime labels, and report artifacts.
2. An options-pricing stack that uses KAN models, regime context, and natural-language tooling to expose pricing and diagnostics through a DSL and REPL.

The codebase is organized more like a research platform than a library package. The important split is between handwritten source, operational scripts, and generated artifacts.

## Source Map

### Data and analysis path

- `src/pipeline/`
  - CLI-style entrypoints for ingest, alignment, QA, and feature generation.
- `src/analysis/`
  - Trend, macro, regime, and visualization logic built on the processed feature sets.
- `src/geometry/`
  - Market-structure methods: GA embeddings, graph curvature, Ricci flow, and optimal transport.
- `src/validation/`
  - Data-quality checks and out-of-sample evaluation helpers.

### Options and model-serving path

- `src/options/dsl/`
  - Lexer, parser, AST nodes, validation, and execution.
- `src/options/pricing/`
  - Black-Scholes and regime-adjusted pricing.
- `src/options/kan_store/`
  - Learned KAN models and feature-bridging code.
- `src/options/llm/`
  - Natural-language to DSL translation and manifold-mapping utilities.
- `src/physics/`
  - Core KAN and SPIKAN implementations used by downstream systems.

## Operational Directories

- `configs/`
  - Universe definitions, regime settings, and options runtime configuration.
- `scripts/`
  - Research runners, reports, training utilities, demos, the REPL, and the Codex QA agent.
- `scripts/research/`
  - Experimental narrative, OT, and notebook-integration scripts separated from the main operational entry points.
- `tests/`
  - Unit and integration coverage across geometry, regimes, pipeline, DSL, pricing, and model layers.
- `docs/`
  - Long-form runbooks, manuals, design notes, and archived implementation records.

### Documentation layout rules

- `docs/guides/`
  - Quick starts, operator cheat sheets, and usage-oriented walkthroughs.
- `docs/project_status/`
  - Milestone summaries, phase status, and implementation snapshots.
- `docs/thesis/`
  - Thesis-facing result summaries and deliverable packaging.
- `docs/archive/`
  - Historical notes kept for traceability rather than day-to-day navigation.

Keep only core project-entry files at the repo root:

- `README.md`
- `AGENTS.md`
- `requirements.txt`
- environment/bootstrap helpers such as `usercustomize.py`

## Generated Artifacts

These directories are primarily outputs, not architecture sources:

- `data/raw/`, `data/interim/`, `data/processed/`
- `reports/`
- `output/`

Use them for provenance, validation, and result inspection. Do not treat them as the canonical explanation of how the system is structured.

## Recommended Reading Order

1. `README.md`
2. `docs/END_TO_END.md`
3. `docs/INDEX.md`
4. `configs/silver_universe.yaml`
5. `configs/options_dsl.yaml`

Then drill into `src/pipeline/`, `src/analysis/`, and `src/options/` depending on the task.

## Branching Recommendation

- Keep `main` as the stable integration branch.
- Use task-specific branches for cleanup or reorganization.
- Keep generated artifacts out of structure-only commits unless the task is explicitly about outputs or reproducibility.
