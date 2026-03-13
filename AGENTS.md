# AGENTS.md

## Project Summary

This repository is a research and engineering workspace for financial market modeling with geometric methods, centered on silver-market regime analysis and an options-pricing stack that reuses those regime signals.

The repo currently has two major product lines:

1. Silver trend/regime pipeline
   - Ingests multi-asset market data for silver, precious metals, macro, rates, volatility, credit, and related proxies.
   - Builds aligned panels and engineered features.
   - Computes geometric diagnostics such as graph curvature, Ricci-style metrics, optimal transport, and geometric algebra rotor features.
   - Produces reports for baseline regimes, Markov-style regimes, geometric regimes, signal diagnostics, and shock propagation.

2. Options Pricing DSL + KAN stack
   - Provides a domain-specific language for option pricing and regime-aware queries.
   - Includes parsing, validation, execution, Black-Scholes pricing, regime-adjusted pricing, KAN-backed knowledge-store models, and LLM-assisted NL-to-DSL translation.
   - Ships with an interactive REPL and a Codex/Claude-based testing agent.

These systems are related rather than separate: the options tooling depends on market/regime concepts that come from the broader silver + macro analysis work.

## Where To Look First

- `README.md`
  - Current project overview and entry points.
- `docs/REPO_ARCHITECTURE.md`
  - High-level structure map and directory responsibilities.
- `docs/END_TO_END.md`
  - Best operational overview for the silver pipeline.
- `docs/INDEX.md`
  - Best doc hub for the options DSL / REPL / agent workflow.
- `docs/guides/`, `docs/project_status/`, `docs/thesis/`
  - Human-oriented summaries, milestone snapshots, and thesis materials that were moved out of the repo root.
- `configs/`
  - Fastest way to understand the active universes, regime settings, and options runtime assumptions.

## Current Architecture

- `src/pipeline/`
  - Data ingestion, alignment, and feature-building entrypoints.
- `src/analysis/`
  - Trend logic, macro features, regimes, impact studies, and visualization helpers.
- `src/geometry/`
  - Geometric algebra, graph curvature, Ricci flow, and optimal transport methods.
- `src/physics/`
  - KAN/SPIKAN layers and physics-inspired market models used by higher-level components.
- `src/options/`
  - Full options stack: DSL, pricing, KAN store, LLM translation, and outlook tooling.
- `src/validation/`
  - Data quality checks and out-of-sample validation helpers.

## Mental Model For Future Agents

- Treat this repo as a research platform with production-style tooling, not a single-purpose package.
- The silver pipeline is the main data-generation backbone.
- The options DSL is the most polished user-facing interface layer.
- Geometry and physics modules are shared method libraries that feed both analysis and pricing/regime work.
- Many files under `reports/`, `output/`, and parts of `data/` are generated artifacts; inspect them only when the task is about experiment outputs or reproducibility.
