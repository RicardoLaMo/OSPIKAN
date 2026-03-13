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
- `BACKEND_INTERFACE_SPEC.md`
  - common specialist-backend interface for SPIKAN and comparison models
  - output schema and capability model for router-based execution
- `ROUTER_MIGRATION.md`
  - migration from the current `ExecutionContext` + hardcoded executor path
  - target backend registry, routing rules, and compatibility gates
- `PROTOCOL_BENCHMARK_SPEC.md`
  - evaluation design for NL-to-DSL as a protocol compiler
  - metrics, corpus design, failure modes, and acceptance gates
- `PROTOCOL_BENCHMARK_MIGRATION.md`
  - migration from prompt/unit-test coverage to reproducible protocol benchmarks
  - implementation path for benchmark harnesses and artifacts
- `QUANT_BENCHMARK_SPEC.md`
  - real-data quantitative proof design for SPIKAN and comparison models
  - tasks, metrics, time splits, ablations, and decision rules
- `SPIKAN_BASELINE_MIGRATION.md`
  - migration from standalone SPIKAN training to benchmark-parity evaluation
  - baseline set, fairness rules, and paper-ready outputs
- `INTEGRATION_COST_STUDY.md`
  - measurement design for backend onboarding and workflow complexity
  - direct comparison between DSL-mediated and backend-specific integration
- `INTEGRATION_COST_MIGRATION.md`
  - how to instrument repo changes so the systems claim becomes measurable
  - branch, diff, and adapter-level evidence plan
- `PAPER_PACKAGE_PLAN.md`
  - final paper structure, figure/table map, and claim packaging
- `MILESTONE_REGISTER.md`
  - end-to-end milestone schedule, dependencies, and merge order
- `EXECUTION_SEQUENCE.md`
  - exact branch execution order from planning into implementation and evidence collection
- `MANUSCRIPT_DRAFT.md`
  - first full paper draft in prose with evidence placeholders
  - aligned to the current repo and benchmark plan
- `PAPER_AUTHORING_PROCESS.md`
  - writing workflow for turning benchmark artifacts into a submission-ready paper
  - rules for replacing placeholders and tightening claims

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
- The final planning target is one coherent path from abstract -> contracts ->
  benchmarks -> paper package, all under versioned git branches.
