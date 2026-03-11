# Silver Price Trend & Regime Impact Analysis — Architecture + TODO Plan

This document turns the background in `notebooks/silver.md` and `World_Silver_Survey-2025.pdf` into an end-to-end engineering + research execution plan, with measurable outputs and a multi-agent work breakdown.

## 0) Problem framing (what “done” means)

### Primary questions
- **Trend**: What are the dominant medium/long-horizon trends in silver (spot and investable proxies), and how stable are they across time?
- **Regimes**: What market regimes best explain silver behavior (returns/vol/correlation/sensitivity), and how frequently do regime transitions occur?
- **Regime impacts**: How do regimes change silver’s conditional distribution (mean/vol/tails), **Gold:Silver Ratio (GSR)** dynamics, and macro betas (USD, real yields, growth proxies)?
- **Geometry adds value**: Do curvature/GA-rotation diagnostics provide earlier or more robust regime transition signals than standard econometrics?

### “Definition of Done” (DoD)
- A reproducible pipeline produces **(a)** curated datasets, **(b)** regime labels/probabilities, **(c)** regime impact report figures/tables, and **(d)** validation artifacts, from a single command (or notebook parameterization).
- All key claims are backed by **out-of-sample** evaluation and documented assumptions (data sources, look-ahead protections, sensitivity checks).
- Results are versioned and reviewable through Git, with stable project structure and clear experiment provenance.

## 1) Context from World Silver Survey 2025 (priors to encode)

Use these as *hypothesis priors* and long-horizon “fundamental state variables”, not as trading signals by themselves.

### High-signal extracted facts (from PDF text extraction)
- **2024 market deficit**: ~**148.9Moz**, reported as ~**15% of global supply** (deficit persists multiple years).
- **Industrial demand**: **680.5Moz** in 2024 (record), with PV “thrifting/substitution” noted but offset by installation scale; electronics/AI-linked demand highlighted.
- **Gold:Silver ratio**: mostly **80:1–90:1** range through much of 2024; commentary notes silver lagging gold during gold’s new highs.
- **2025 forecast deficit**: ~**117.6Moz** (still deficit, smaller than 2024 per survey narrative).

### How this maps into the analysis design
- Treat **deficit magnitude**, **industrial demand share**, and **GSR level** as regime drivers/markers.
- Expect at least two super-regimes:
  1) **Monetary/haven-dominated** (gold leads; silver follows with higher beta),
  2) **Industrial-cycle-dominated** (growth/PMI/China/PV proxies matter more).
- Expect above-ground stocks/ETP flows to modulate how fundamentals transmit into price (potential lag structure).

## 2) Data universe & sources (pipeline contract)

### Assets (minimum viable universe)
- Silver: `XAGUSD` (spot), `SI=F` (futures), `SLV`/`SIVR` (ETP proxy), miners: `SIL`, `SILJ`, (optional) key miners.
- Cross-metals: `GC=F` (gold), copper proxy (`HG=F` or `COPX`), platinum/palladium (optional).
- Macro: DXY, nominal yields (2Y/10Y), inflation expectations / real yields if available, equities (SPX/QQQ), VIX, oil.

### Survey/fundamental datasets (annual/quarterly)
- From `World_Silver_Survey-2025.pdf`: supply/demand components, market balance, sector demand (PV/electronics/jewelry), investment (ETP holdings/flows), inventories if available.

### Data access strategy (practical, reproducible)
- **Market time series**: prefer **OpenBB** (already in repo) with provider configuration and consistent column schema.
- **Fallback**: `yfinance` for gaps and simple prototyping.
- **Survey tables**: start with a *manual extraction* into CSV/Parquet with page references; later automate via `camelot`/`tabula` if the PDF is table-friendly.

## 3) Repository architecture (target tree + contracts)

Keep “raw” immutable, make transforms explicit, and separate research notebooks from reusable modules.

### Proposed target tree (incremental to current repo)
```text
investment/
  configs/
    silver_universe.yaml
    regimes.yaml
  data/
    raw/                  # immutable pulls (already present)
    external/             # survey-extracted tables + metadata
    interim/              # standardized joins/alignment
    processed/            # model-ready matrices
  notebooks/
    silver.md
    silver_trend_regime_plan.md
    silver_trend_regime_analysis.ipynb
  reports/
    silver/
      figures/
      tables/
      silver_regime_report.md
  src/
    pipeline/
      fetch_data.py
      preprocess.py
      silver_pipeline.py
    analysis/
      regimes.py
      trend.py
      regime_impacts.py
    geometry/
      tensor_ga.py
      graph_curvature.py
    validation/
      ga_invariants.py
      data_quality.py
      backtest_checks.py
  tests/
    test_silver_pipeline.py
    test_regimes.py
```

### Core “contracts” (interfaces to stabilize)
- **Dataset contract**: a canonical `DataFrame` index (`date`) + normalized columns (OHLCV where applicable) + metadata (provider, currency, timezone).
- **Feature contract**: well-defined feature matrix `X[t, f]` with a documented lookback policy and no leakage.
- **Regime contract**: `regime_id[t]` and/or `P(regime=k | t)` with a frozen label mapping and transition matrix export.

## 4) End-to-end pipeline (build order)

### Stage A — Ingest (raw layer)
**Objective**: reproducible pulls + provenance.

TODOs
- [ ] Implement a silver “universe fetch” wrapper (OpenBB-first) that saves raw pulls to `data/raw/` with a `run_metadata.json` (symbols, provider, timestamps).
- [ ] Add schema normalization: always output `open/high/low/close/volume` (where available) and a `close_adj` column when supported.
- [ ] Add “data coverage” report: missing rates by symbol, date range, and forward-fill impacts.

Objective measurements (acceptance)
- Data pulled for ≥ 95% of the requested symbols and ≥ 99% of trading days in the analysis window.
- All raw artifacts are write-once (reruns create a new run-id folder, not overwriting prior runs).

### Stage B — Clean + align (interim layer)
**Objective**: aligned multi-asset panel, consistent calendar rules.

TODOs
- [ ] Implement calendar alignment policy (US trading calendar vs union of trading days), documented explicitly.
- [ ] Add currency normalization where relevant (e.g., XAGUSD already USD; ensure no mixed currencies).
- [ ] Persist aligned panel to Parquet in `data/interim/`.

Objective measurements
- Missingness after alignment ≤ 1% for “core” symbols (silver/gold/dxy/yields), with a logged list of imputed points.

### Stage C — Feature engineering (processed layer)
**Objective**: features for trend + regimes + geometry.

Baseline features (fast path)
- Returns: log returns, multi-horizon returns (1d/5d/20d/60d).
- Volatility: rolling realized vol, ATR-like range metrics, volatility-of-vol.
- Trend: moving-average slope, z-scored trend strength, drawdown, time-since-high.
- Cross features: GSR (`gold/silver`), silver vs DXY beta proxy, gold beta proxy.

Geometry/structure features (add after baseline is stable)
- Rolling correlation distance matrix: `d = sqrt(2(1 - corr))`.
- Graph metrics: MST length (stress proxy), clustering coefficient, centrality.
- Ricci-style curvature:
  - Quick proxy: MST length + bottleneck metrics.
  - Full: Ollivier-Ricci / Forman-Ricci via a dedicated library (see §5).
- GA regime-shift diagnostics: rotor magnitude / bivector energy from `src/geometry/tensor_ga.py`.

Objective measurements
- Feature computation is deterministic (fixed seeds where applicable), vectorized, and runs in ≤ 2 minutes for 10y daily data on a laptop baseline.
- A leakage test confirms no feature uses future data (unit test + spot-check notebook).

### Stage D — Regime modeling
**Objective**: produce regimes with uncertainty and interpretability.

Regime model candidates (implement in this order)
1) **Rule-based baseline** (interpretable):
   - Trend regime: MA crossover / trend strength thresholds.
   - Vol regime: low/medium/high realized volatility buckets.
2) **Change point detection** (structural breaks):
   - `ruptures` on returns/vol/GSR/curvature to localize breaks.
3) **Probabilistic regimes**:
   - HMM on feature vectors (returns/vol/GSR/betas/curvature).
   - Markov-switching regression of silver returns on macro factors.
4) **Geometry-driven regime score**:
   - Curvature/stress-based transition alerts (Δ-curvature spikes).
   - GA-rotor “rotation angle” spikes between rolling windows.

Objective measurements
- Out-of-sample log-likelihood improvement vs baseline (probabilistic models).
- Regime stability: median regime duration and label persistence are not pathological (no flip-flopping day-to-day without evidence).
- Transition detection lead time: curvature/GA alerts are evaluated for *lead/lag* vs realized drawdowns/vol spikes.

### Stage E — Regime impact analysis (the thesis-grade output)
**Objective**: quantify how regimes change silver behavior and drivers.

Core analyses
- Conditional return stats by regime (mean/vol/skew/kurt, tail risk, drawdowns).
- Conditional correlations/betas by regime (vs gold, DXY, yields, equities).
- GSR dynamics by regime (level, trend, mean reversion, breakpoints).
- Event study around regime transitions (average path of silver returns, vol, GSR).
- Fundamental overlay: align annual survey deficit/demand metrics with regime prevalence (macro narrative consistency check).

Objective measurements
- All key comparisons include uncertainty (bootstrap CI) and multiple-hypothesis controls where applicable.
- Results are robust across at least 2 time windows (e.g., 2010–2019 vs 2020–present) and at least 2 data proxies (spot vs ETP).

### Stage F — Reporting + visualization
**Objective**: produce an interpretable “regime atlas” for silver.

Required figures
- Silver price with regime shading + transition markers.
- Regime probability time series (for HMM/MS models).
- GSR chart with regime overlay.
- “Driver sensitivity” heatmap (betas/correlations by regime).
- Geometry view: correlation network/MST snapshots for key dates + curvature/stress timeline.

Objective measurements
- Every figure is reproducible via code, saved under `reports/silver/figures/` with consistent filenames and a short caption.

## 5) Python library selection (recommended stack)

### Data + compute
- `openbb` (primary data access), `pandas`, `numpy`
- `pyarrow` (Parquet), optional `polars` (speed) if needed

### Econometrics + regimes
- `statsmodels` (Markov switching / regression diagnostics)
- `scikit-learn` (baselines + clustering)
- `ruptures` (change points)
- `hmmlearn` or `pomegranate` (HMMs)
- `arch` (GARCH-family volatility, optional)

### Geometry / graphs
- `networkx` (baseline graphs + MST)
- Optional (for full curvature): `GraphRicciCurvature` (Ollivier/Forman), or alternative curvature implementations if dependency friction arises
- `torch` (already in repo) for GA embeddings; optional `torch-geometric` for graph learning experiments

### Visualization
- `matplotlib`, `seaborn` (static)
- `plotly` (interactive), optional `streamlit` for a lightweight dashboard

### Quality + reproducibility
- `pytest` (tests), `ruff` (lint/format), `pre-commit` (hooks)

## 6) Multi-agent work plan (roles, outputs, measurable goals)

This is a *coordination* tool: each agent owns artifacts and must satisfy objective acceptance criteria before “handoff”.

### Agent A — Data Engineer (pipeline + data QA)
Owns: `src/pipeline/silver_pipeline.py`, `src/validation/data_quality.py`, `data/raw/`, `data/interim/`

TODOs
- [ ] Implement multi-symbol ingest with metadata, caching, and deterministic filenames/run-ids.
- [ ] Implement alignment and schema normalization.
- [ ] Add data-quality tests (missingness, duplicates, monotonic index, outlier flags).

Acceptance criteria
- Data-quality report auto-generated; core symbols pass thresholds.
- Pipeline reruns are idempotent (no silent overwrites).

### Agent B — Quant Researcher (baseline trend + regime models)
Owns: `src/analysis/trend.py`, `src/analysis/regimes.py`

TODOs
- [ ] Define trend metrics + regime baseline (trend/vol buckets) with clear thresholds.
- [ ] Implement at least one probabilistic regime model (HMM or Markov switching) with walk-forward evaluation.
- [ ] Document regime definitions and label mapping in `configs/regimes.yaml`.

Acceptance criteria
- Out-of-sample evaluation notebook/report exists; models beat naive baselines on likelihood or regime-consistency metrics.
- Regime labels are stable and interpretable (documented “what it means” per regime).

### Agent C — Geometry Scientist (graph curvature + regime alerts)
Owns: `src/geometry/graph_curvature.py`, geometry features in processed layer

TODOs
- [ ] Implement correlation distance graph construction and MST stress index (as per `notebooks/silver.md`).
- [ ] Add curvature metrics (proxy first; full Ollivier/Forman if feasible).
- [ ] Build transition alert logic and evaluate lead/lag vs known stress episodes.

Acceptance criteria
- Curvature/stress metrics are computed over time with runtime bounds.
- Alert precision/recall is quantified against a chosen “market stress” ground truth proxy (e.g., vol spike or drawdown threshold).

### Agent D — GA/Manifold Engineer (rotor-based regime shift diagnostics)
Owns: GA embedding + regime-shift outputs tied to `src/geometry/tensor_ga.py`

TODOs
- [ ] Specify mapping from engineered features → GA multivector (document basis mapping).
- [ ] Compute rolling rotors/regime “rotation magnitude” between windows.
- [ ] Compare GA regime signal to curvature and to econometric regimes.

Acceptance criteria
- GA signal is numerically validated using `src/validation/ga_invariants.py`.
- GA regime shifts show incremental information (statistical association + out-of-sample utility vs baselines).

### Agent E — Visualization + Reporting (regime atlas)
Owns: `reports/silver/`, final report markdown, figure generation scripts

TODOs
- [ ] Implement canonical plots with consistent styling and captions.
- [ ] Produce `reports/silver/silver_regime_report.md` as the narrative deliverable.

Acceptance criteria
- Report can be rebuilt end-to-end; figures are versioned and referenced.
- Plots include regime overlays and uncertainty where relevant.

### Agent F — QA/GitOps (tests + repo hygiene + version control)
Owns: tests, tooling, Git conventions, CI (if added)

TODOs
- [ ] Add `pytest` coverage for data contract + leakage checks + regime contract.
- [ ] Add `ruff` + `pre-commit` configuration; enforce import/order and basic style.
- [ ] Define Git workflow (branching, tags) and document in `README.md` or `QUICK_START_GUIDE.md`.

Acceptance criteria
- Tests run locally in one command; failures are actionable.
- Repo stays clean (no large binary artifacts committed unintentionally; data stays out of Git unless explicitly versioned).

## 7) Validation checklist (research + engineering)

### Engineering validation (must-pass)
- [ ] Deterministic runs (fixed seeds, controlled randomness).
- [ ] Data leakage tests for every model feature set.
- [ ] Unit tests for schema/contract invariants and alignment policy.

### Statistical validation (must-document)
- [ ] Walk-forward or rolling out-of-sample evaluation (no random CV).
- [ ] Sensitivity analysis for lookback windows and regime count `K`.
- [ ] Robustness across proxies (spot vs ETF, futures vs spot).
- [ ] Multiple testing controls for regime comparisons when many metrics are evaluated.

## 8) Git management recommendations (version control as a research tool)

### Repo initialization sanity check
- Verify Git is active: `git rev-parse --show-toplevel`
- If this directory is not a Git repo yet, initialize it once:
  - `git init`
  - `git add .`
  - `git commit -m "chore: initialize thesis investment repo"`

### Branching + commits
- Use short-lived feature branches: `feat/silver-pipeline`, `feat/regime-hmm`, `feat/curvature-metrics`.
- Commit in small, reviewable units with messages like: `pipeline: add multi-symbol ingest` / `analysis: add HMM walk-forward eval`.

### Data + artifact versioning
- Keep `data/raw/`, `data/interim/`, `data/processed/` out of Git (already in `.gitignore`).
- Consider `git-lfs` for large binaries if you plan to version PDFs/figures heavily.
- For serious experiment/data lineage: consider DVC *only if* the overhead is justified for the thesis workflow.

### “Git tree” commands to standardize in the workflow
- Repo structure snapshot: `find . -maxdepth 3 -type f | sort`
- History graph: `git log --graph --decorate --oneline --all -n 20`
- What changed since last milestone tag: `git diff <tag>..HEAD --stat`

## 9) Milestones (objective gates)

1) **M1: Data foundation**
   - Gate: ingest + alignment + QA report passes.
2) **M2: Baseline regimes**
   - Gate: rule-based + one probabilistic regime model with OOS evaluation.
3) **M3: Geometry regimes**
   - Gate: curvature/stress + GA rotation diagnostics implemented and evaluated.
4) **M4: Regime impact report**
   - Gate: final report + figures + robustness checks completed.

## 10) Open questions / decisions to make early
- What is the **primary prediction target**, if any (trend continuation, drawdown risk, volatility spike, regime transition probability)?
- What time horizon matters most for the thesis (daily vs weekly vs monthly)?
- Do we treat silver primarily as **monetary metal** (co-moving with gold) or **industrial metal** (growth cycle), or do we explicitly model the mixture via regimes?
