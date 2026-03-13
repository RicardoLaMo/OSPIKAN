# OSPIKAN

OSPIKAN is a research and engineering repository for market modeling with geometric methods, centered on silver-market regime analysis and an options-pricing stack that reuses those regime signals.

## What Is In This Repo

### Silver trend and regime pipeline

- Ingests multi-asset histories for silver, macro, rates, volatility, credit, and related proxies.
- Builds aligned panels and engineered features.
- Computes geometric diagnostics such as graph curvature, Ricci-style measures, optimal transport, and geometric algebra rotor features.
- Generates reports for baseline, Markov-style, and geometric regime analysis.

### Options DSL and KAN stack

- Provides a domain-specific language for option pricing and regime-aware queries.
- Includes pricing engines, KAN-backed knowledge-store models, and LLM-assisted NL-to-DSL translation.
- Ships with an interactive REPL and a Codex/Claude-based QA agent.

## Project Map

- `src/pipeline/`: ingest, align, QA, and feature-generation entrypoints
- `src/analysis/`: trend, macro, regime, and visualization logic
- `src/geometry/`: GA, graph curvature, Ricci flow, and optimal transport methods
- `src/physics/`: KAN and SPIKAN model components
- `src/options/`: DSL, pricing, KAN store, LLM translation, and outlook tooling
- `src/validation/`: data-quality and out-of-sample evaluation helpers
- `scripts/options/`: options REPL, QA agent, and KAN training
- `scripts/silver/`: stable silver pipeline runners, reports, and training
- `scripts/`: compatibility wrappers for established command paths
- `scripts/research/`: exploratory narrative, OT, and notebook-integration scripts
- `tests/`: unit and integration coverage
- `docs/guides/`: quick starts and operator-facing how-to material
- `docs/project_status/`: implementation snapshots and milestone summaries
- `docs/thesis/`: thesis-facing results and deliverable summaries

## Start Here

- Silver pipeline runbook: `docs/END_TO_END.md`
- Options DSL docs: `docs/INDEX.md`
- Repo structure map: `docs/REPO_ARCHITECTURE.md`
- Workflow and Git conventions: `docs/WORKFLOW.md`
- Silver quick reference: `docs/guides/SILVER_QUICK_REFERENCE.md`

## Common Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Inspect the silver pipeline CLI:

```bash
python src/pipeline/silver_pipeline.py --help
```

Start the options REPL:

```bash
python scripts/option_dsl_repl.py --dsl-mode
```

Run the test suite:

```bash
pytest
```
