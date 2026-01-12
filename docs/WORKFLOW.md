# Workflow (Multi-Agent + Git) — Silver Trend/Regime System

This repo is set up to support parallel “agents” (human or AI) working on distinct modules with clear contracts and objective acceptance criteria.

## Branching model

- `main`: always runnable, tested, and thesis-safe.
- Feature branches (short-lived):
  - Data engineering: `feat/data-pipeline-...`
  - Regimes/econometrics: `feat/regimes-...`
  - Geometry/curvature: `feat/curvature-...`
  - GA/rotors: `feat/ga-...`
  - Viz/reporting: `feat/report-...`
  - QA/GitOps: `feat/qa-...`

Suggested commands:
- Create branch: `git checkout -b feat/regimes-hmm-walkforward`
- History graph: `git log --graph --decorate --oneline --all -n 30`
- What changed: `git diff --stat main...HEAD`

## Multi-agent ownership map

Each agent owns code + tests in a scoped area:

- **Agent A (Data Engineer)**: `src/pipeline/`, `configs/`, data contracts, ingest/alignment runs.
- **Agent B (Quant Researcher)**: `src/analysis/` (regimes + trend), evaluation notebooks.
- **Agent C (Geometry Scientist)**: `src/geometry/graph_curvature.py` (to be added) + curvature features.
- **Agent D (GA Engineer)**: `src/geometry/tensor_ga.py` integration for regime-shift diagnostics.
- **Agent E (Reporting)**: `reports/silver/` generators + narrative.
- **Agent F (QA/GitOps)**: `tests/`, linting, reproducibility checks, CI (optional).

## Data policy (what goes in Git)

- Code + configs + docs: tracked.
- Market data + run artifacts: **not tracked** (see `.gitignore` for `data/raw/`, `data/interim/`, `data/processed/`).
- If you want to version large binaries (PDFs/figures): consider `git-lfs` and keep them under `reports/` or `references/`.

## Note on Git location (Google Drive mount)

This working tree lives under a Google Drive mount. On this filesystem, Git reflog writes can fail.

Current setup:
- The repo’s `.git` in the working tree is a pointer file.
- The actual Git directory is stored at `/home/wliu23/.gitdirs/investment.git` (local filesystem).

If you clone/move this project to another machine, clone from a remote instead of relying on the Drive folder to carry Git history.

## Pipeline entrypoints (current)

### 1) Ingest the configured universe
Fetches raw OHLCV per symbol and writes a run folder under `data/raw/_runs/<run_id>/` with `run_metadata.json`.

`python src/pipeline/silver_pipeline.py ingest --config configs/silver_universe.yaml`

### 2) Align to an interim close-price panel
Builds an aligned close-price matrix and writes to `data/interim/`.

`python src/pipeline/silver_pipeline.py align --config configs/silver_universe.yaml`

## Quality gates

- Run tests before merging: `pytest -q`
- Keep changes reviewable:
  - Prefer ≤ ~300 LOC per PR unless unavoidable.
  - Add/adjust tests for contract changes (schemas, features, regime outputs).
