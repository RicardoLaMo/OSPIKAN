# Paper Planning

This directory turns the LLM+SPIKAN abstract into a concrete repo plan.

Core documents:

- `LLM_SPIKAN_RECONSTRUCTION_PLAN.md`
  - target thesis claim
  - current repo state from the git tree
  - proposed architecture reconstruction
  - git branch strategy and milestones
- `EXPERIMENT_DESIGN.md`
  - real-data plan
  - model comparison matrix
  - evidence required to prove or falsify the claim
- `IMPLEMENTATION_ROADMAP.md`
  - branch-by-branch execution sequence
  - module touch points in the current repo
  - acceptance gates for each phase
- `EVIDENCE_MATRIX.md`
  - exact claim-to-evidence mapping
  - required tables, figures, baselines, and failure criteria
- `STATE_CONTRACT_SPEC.md`
  - canonical market-state schema derived from the current repo
  - projection rules into SPIKAN, regime logic, and DSL execution
- `STATE_CONTRACT_MIGRATION.md`
  - how to move from current heuristics to the canonical state contract
  - affected modules and migration gates
- `OPTION_DATA_CONTRACT_SPEC.md`
  - canonical option snapshot schema for real-data benchmarking
  - contract between DSL pricing tasks, KAN store inputs, and evaluation
- `OPTION_DATA_MIGRATION.md`
  - move from synthetic-only option training assumptions to real-data option snapshots
  - benchmark join design with the market-state contract

Working thesis:

> The LLM is not the quant model. It is the protocol layer that translates
> human intent into a constrained DSL so specialist models such as SPIKAN can
> be used in a disciplined, auditable workflow without hand-built orchestration
> glue for every model integration.

Planning constraints taken from the current repo:

- The repo already has a strong tracked base in `src/pipeline/`, `src/analysis/`,
  `src/geometry/`, `src/options/`, `src/physics/`, and `tests/`.
- The main gap is end-to-end integration and proof, not raw component count.
- The immediate objective is to reconstruct the architecture around explicit
  interfaces and measurable experiments, not to start another broad refactor.
