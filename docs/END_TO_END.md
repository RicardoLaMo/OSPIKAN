# End-to-End Process — Silver Trend/Regime System

This repo is set up to take a **config-defined silver + macro universe** and produce:

1) **Raw pulls** (per symbol, per run)
2) **Aligned close panel** (one matrix, many symbols)
3) **Feature table** (trend + macro + geometry + GA)
4) **Regimes + diagnostics** (baseline, Markov-switching, geometric)
5) **Reports** (figures + tables under `reports/`)

---

## 1) Folder map (what lives where)

- `configs/`
  - `configs/silver_universe.yaml`: symbols + data provider + date range
  - `configs/regimes.yaml`: baseline regime hyperparameters (trend/vol/HMM defaults)
- `data/`
  - `data/raw/_runs/<run_id>/`: one CSV per symbol + `run_metadata.json` (immutable raw pulls)
  - `data/interim/`: aligned close-price panel (wide matrix)
  - `data/processed/`: model-ready features (Parquet) + metadata JSON
  - `data/external/`: place for fundamentals/survey tables you extract (manual or automated)
- `src/`
  - `src/pipeline/`: ingestion + alignment + feature computation entrypoints
  - `src/analysis/`: regimes, trend, regime impacts, plotting helpers
  - `src/geometry/`: MST stress, Forman–Ricci curvature, Ricci flow, GA rotors
  - `src/validation/`: panel QA utilities
- `scripts/`: “one-shot” analysis/report generators that consume `data/processed/*.parquet`
- `reports/`: generated outputs (figures/tables)
- `tests/`: unit tests for pipeline + geometry + regimes

---

## 2) Data contracts (inputs/outputs you can rely on)

### Close panel (`data/interim/silver_panel_close_<run_id>.csv`)

- Index: date (first column), parsed as datetime, normalized to date
- Columns: symbol strings exactly as in `configs/silver_universe.yaml`
- Values: close prices (float), forward-filled after alignment

### Feature table (`data/processed/silver_features_<run_id>.parquet`)

Minimum required:
- `silver_close`
- `log_return_1d`

Common additional features (when symbols exist in the panel):
- Trend/vol: `realized_vol_20d`, `momentum_10d`, `drawdown` (rolling 252d), plus `drawdown_63d`, `drawdown_126d`, `drawdown_252d`, `drawdown_ath`
- Cross: `gsr`, `gsr_log`
- Macro: `dxy_log_return_1d`, `dxy_beta_60d`, `y10_change_1d`, `spx_log_return_1d`, `vix_log_return_1d`
- Geometry: `mst_stress_60d`, `ricci_mean_60d`, `ricci_min_60d`, `ricci_std_60d`
- Geometry (tail/core/pairwise): `ricci_p10_60d`, `mst_stress_core_60d`, `ricci_p10_core_60d`, `dist_silver_gold_60d`, `dist_silver_dxy_60d`
- GA: `ga_rotor_magnitude_60d`, `ga_bivector_energy_60d`

Provenance:
- `data/raw/_runs/<run_id>/run_metadata.json` logs provider/source status per symbol + git SHA.
- `data/processed/silver_features_<run_id>.metadata.json` captures the symbol mapping and windows.

---

## 3) End-to-end pipeline (fresh run)

### 0) Environment

`pip install -r requirements.txt`

### 1) Configure the universe

Edit `configs/silver_universe.yaml` (symbols, start date, provider).

### 2) Ingest raw histories (network required)

`python src/pipeline/silver_pipeline.py ingest --config configs/silver_universe.yaml`

Outputs:
- `data/raw/_runs/<run_id>/*.csv`
- `data/raw/_runs/<run_id>/run_metadata.json`
- `data/raw/_runs/LATEST` (contains `<run_id>`)

If ingest fails with `ModuleNotFoundError`, install dependencies in your active environment:
- `pip install -r requirements.txt`
- Or (minimal for ingest): `pip install yfinance`

### 3) Align into a close-price panel

`python src/pipeline/silver_pipeline.py align --config configs/silver_universe.yaml`

Output:
- `data/interim/silver_panel_close_<run_id>.csv`

### 4) QA the panel (missingness + sanity checks)

Example (core symbols only):

`python src/pipeline/silver_pipeline.py qa --panel data/interim/silver_panel_close_<run_id>.csv --required "SI=F,GC=F,DX-Y.NYB,^TNX,SPY"`

Output:
- `data/interim/silver_panel_close_<run_id>.csv.qa.json`

### 5) Compute features (incl. geometry + GA)

`python src/pipeline/silver_pipeline.py features --panel data/interim/silver_panel_close_<run_id>.csv`

Output:
- `data/processed/silver_features_<run_id>.parquet`
- `data/processed/silver_features_<run_id>.metadata.json`

Optional deep geometry (slower):
- `python src/pipeline/silver_pipeline.py features --panel ... --deep-geometry`

### Cached/offline run (skip ingest)

If you already have an existing panel/features file, you can regenerate reports without refetching data:

- Skip ingest/alignment (reuse an existing panel):
  - `PANEL_PATH=data/interim/silver_panel_close_<run_id>.csv sh scripts/run_silver_end_to_end.sh`
- Skip ingest/alignment and feature computation (reuse both):
  - `PANEL_PATH=data/interim/silver_panel_close_<run_id>.csv FEATURES_PATH=data/processed/silver_features_<run_id>.parquet sh scripts/run_silver_end_to_end.sh`

---

## 4) Reports + regime analysis (consume features)

All scripts accept `--features data/processed/silver_features_<run_id>.parquet` and write to `--out-dir`.

Baseline report (trend × vol):
- `python scripts/silver_report.py --features ... --out-dir reports/silver`

Markov-switching regression regimes (with macro + geometric exogenous vars if present):
- `python scripts/silver_markov_regimes.py --features ... --k 3 --out-dir reports/silver`

Pure geometric regimes (Ricci + MST + GA):
- `python scripts/silver_geometric_regimes.py --features ... --out-dir reports/silver`

Signal diagnostics (drawdown prediction, feature correlations, timeline plot):
- `python scripts/analyze_geometric_signals.py --features ... --out-dir reports/silver/signals`

Method comparison (baseline vs geometric):
- `python scripts/compare_regime_methods.py --features ... --out-dir reports/silver/analysis`

Ricci-flow shock scenarios (consume panel):
- `python scripts/ricci_flow_shock_scenarios.py --panel data/interim/silver_panel_close_<run_id>.csv --out-dir reports/silver/ricci_flow`

---

## 5) How to use outputs (decision-support loop)

This system is designed as a **risk + regime overlay** around any silver exposure (spot, futures, ETPs).

Suggested workflow:
- Start with **baseline trend × vol** regimes for interpretability.
- Use **Markov probabilities** to quantify uncertainty (don’t rely only on hard labels).
- Use **geometric regime labels** as a structural-stress indicator:
  - `STRESS`: risk-off overlay (de-lever, reduce risk, tighten stops)
  - `STABLE`: allow trend-following risk budget
  - `TRANSITION`: reduce risk until structure stabilizes
  - `RECOVERY`: watch for re-risking signals with confirmation

Use the generated tables as inputs:
- `conditional_return_stats.csv`: conditional drift/vol by regime
- `conditional_beta_dxy.csv`: how USD sensitivity changes by regime
- `geometric_regime_return_stats.csv`: performance by geometric regimes
- `markov_probs_k*.csv`: uncertainty, transition behavior

---

## 6) Known caveats (current)

- If `ga_rotor_magnitude_60d` is flat/zero, you are likely viewing an older features file; recompute features with the current code (older versions incorrectly anchored the reference vector and produced zeros).
- Miner ETFs (e.g., `SILJ`) have later inception and will introduce missingness; QA should focus on a “core” required set.
- Some Drive/FUSE mounts can produce 0-byte Parquet files when writing directly; the pipeline uses a temp-file write to reduce this risk.
