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
