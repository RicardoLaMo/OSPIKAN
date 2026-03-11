# SPIKAN/PIKAN Enhancement Plan — Cross-Check vs Repo Implementation

Date: 2026-01-16  
Repo: `investment` (silver trend/regime + geometry framework)

This document cross-checks the **PIKAN/SPIKAN Enhancement Plan for Market Fluid Dynamics** (as provided) against the current codebase implementation. It is written to be actionable: **what exists, what’s missing, what’s risky, and what to do next** with objective measurements.

---

## 0) Executive Cross-Check (TL;DR)

### What is implemented (✅)
- ✅ **KAN core** (B-spline basis + KANLinear/KANLayer/KANNetwork): `src/physics/kan_layers.py`
- ✅ **SPIKAN architectures** (branch separability + multiscale): `src/physics/spikan.py`
- ✅ **Market PDEs + physics-informed loss** (Burgers + advection-diffusion + mean reversion): `src/physics/market_pdes.py`
- ✅ **Shock detection module** (multi-scale heuristics + labeling + neural head scaffold): `src/physics/shock_detector.py`
- ✅ **Non-neural “fluid dynamics” feature layer** (shock formation index, decay rate, divergence, macro transmission): `src/analysis/fluid_dynamics.py`
- ✅ **Unit tests (21 tests)** covering basis, gradients, SPIKAN forward/derivs, PDE residuals, shock detector, fluid features: `tests/test_kan_layers.py`

### What is *not* implemented yet (⚠️)
- ⚠️ **Training pipeline** for SPIKAN/PINN (datasets, batching, optimizer schedule, checkpoints): plan mentions `scripts/train_spikan.py` but it does not exist.
- ⚠️ **End-to-end integration** into the silver pipeline/regime workflow (feature join + regime logic + reporting). `src/analysis/fluid_dynamics.py` is not called by `src/analysis/features.py`, `src/analysis/regimes.py`, or existing report scripts.
- ⚠️ **1D Burgers “known-solution” validation** (Phase 1 checkpoint) beyond unit tests (no training demo).
- ⚠️ **Config plumbing** (SPIKAN hyperparams in YAML, CLI flags in `silver_pipeline.py`) is not present.

### Critical environment constraint discovered (⚠️)
This workspace is on a Drive/FUSE mount where **exclusive-create** file semantics can fail; Python’s `tempfile` probing and pytest’s default FD capture can break.
- Fix applied for `src.physics` imports: `src/physics/__init__.py` sets `tempfile.tempdir` to `.tmp/`.
- Fix applied for Parquet writes: `src/pipeline/silver_pipeline.py:_write_parquet_safe()` now writes Parquet via an **in-memory buffer** (no `tempfile.mkstemp`).
- Test note: full suite passes with `--capture=sys` (FD capture uses temp files).

---

## 1) Traceability Matrix (Plan → Implementation)

Legend: ✅ implemented, 🟡 partial, ❌ missing

| Plan Item | Expected Artifact | Current Evidence | Status | Notes |
|---|---|---:|:---:|---|
| New `src/physics/` module | `src/physics/*` | `src/physics/__init__.py`, `kan_layers.py`, `spikan.py`, `market_pdes.py`, `shock_detector.py` | ✅ | Module exists; exports provided in `__init__.py`. |
| KAN layer implementation | `KANLayer`, B-splines | `src/physics/kan_layers.py` | ✅ | Uses a stable cardinal B-spline kernel (approximation); not SciPy-based. |
| SPIKAN separable architecture | separable branches + combiner | `src/physics/spikan.py:MarketSPIKAN`, `MultiscaleSPIKAN` | ✅ | Uses **per-term elementwise product** across modalities, then sums terms. |
| Physics-informed loss | residual + data loss | `src/physics/market_pdes.py:physics_informed_loss()` | ✅ | Works for scalar-output PDEs; see multi-output derivative caveat below. |
| Burgers equation | residual + shock indicator | `src/physics/market_pdes.py:BurgersEquation` | ✅ | Includes `shock_formation_indicator()`. |
| Advection-diffusion | residual + param encoding | `src/physics/market_pdes.py:AdvectionDiffusion` | ✅ | Velocity encoding present (macro momentum proxies). |
| Modified Navier–Stokes | liquidity flow PDE | none | ❌ | Not implemented (likely too heavy without order-book data). |
| Shock/discontinuity detector | shock detection module | `src/physics/shock_detector.py` | ✅ | Heuristic + labeling + neural head scaffold. Not integrated into pipeline. |
| Feature integration layer | `analysis/fluid_dynamics.py` | `src/analysis/fluid_dynamics.py` | ✅ | Produces *non-neural* fluid proxies; not integrated into `compute_silver_features()`. |
| Integrate with Ricci features | use geometry as inputs | `MarketSPIKAN` has geometry branch; fluid features accept `geometric_features` for viscosity proxy | 🟡 | Plumbing exists, but pipeline doesn’t build tensors or call SPIKAN training/inference. |
| Modify `features.py` / `regimes.py` | add fluid/shock signals | none | ❌ | No calls to `compute_fluid_dynamics_features()` or new regime logic yet. |
| Add pipeline flag | `--fluid-dynamics` | none | ❌ | CLI not extended. |
| Add config hyperparams | YAML additions | none | ❌ | No SPIKAN config entries in `configs/*.yaml`. |
| Training script | `scripts/train_spikan.py` | none | ❌ | Missing. |
| Shock analysis script | `scripts/analyze_market_shocks.py` | none | ❌ | Missing. |
| Unit tests | `tests/test_kan_layers.py`, `tests/test_market_pdes.py` | `tests/test_kan_layers.py` only | 🟡 | Tests are consolidated into one file; acceptable, but plan text should reflect this. |

---

## 2) Technical Review: Correctness & Gaps

### 2.1 KAN Layer (`src/physics/kan_layers.py`)
- ✅ Shape math is correct: basis `(B, F, K)` and weights `(F, O, K)` → output `(B, O)`.
- ✅ Basis normalization enforces (approx) partition-of-unity, good for stability.
- 🟡 The implementation is a **cardinal kernel approximation**, not Cox–de Boor recursion over an adaptive knot vector. That’s OK if stated as “KAN-like spline features”; if the thesis claims “true KAN splines”, tighten wording.
- 🟡 Unused imports exist (`numpy`), minor.

### 2.2 SPIKAN Architecture (`src/physics/spikan.py`)
- ✅ Architecture reflects the plan’s separability idea more faithfully than simple concatenation:
  - `SeparableBranch` produces `(B, n_terms, hidden)` and multiplies across modalities.
- 🟡 “Outer product” is mentioned in docstring, but code uses **elementwise product** (better complexity; adjust wording in thesis/plan to match).
- ⚠️ `MarketSPIKAN.regularization_loss()` currently regularizes **only** each branch’s `input_proj`, not the `term_heads`. This is likely an oversight if sparsity/interpretability is a thesis claim.
- ⚠️ `forward_with_derivatives()` is effectively **scalar-output oriented**:
  - Unit tests only exercise `output_dim=1`.
  - If `output_dim=n_assets` (as in `create_market_spikan()`), derivatives are not well-defined as written (Jacobian/Hessian would be needed).

### 2.3 PDEs & Physics Loss (`src/physics/market_pdes.py`)
- ✅ Residual formulations are correct algebraically (Burgers, advection-diffusion, mean reversion).
- 🟡 Multi-asset handling is currently done by **reducing gradients across asset dimension** (mean/sum). This is a heuristic; it should be justified as a “1D effective coordinate” or replaced with a graph-domain Laplacian (recommended below).
- 🟡 Parameter encoders create constants with `torch.tensor(...)` without matching device/dtype; on GPU training this can cause device mismatch. Low effort to harden.

### 2.4 Shock Detector (`src/physics/shock_detector.py`)
- ✅ Implements multi-indicator shock detection and forward-looking labeling (`label_historical_shocks`).
- 🟡 “NeuralShockDetector” exists but isn’t trained/used; consider whether it belongs in scope now or later.
- ⚠️ Labeling uses **lookahead**, which is fine, but introduces leakage risk if the split is not strictly time-based and features aren’t lagged correctly.

### 2.5 Fluid Feature Layer (`src/analysis/fluid_dynamics.py`)
- ✅ Good “physics proxy” features for immediate use without training: SFI, viscosity proxy, divergence, transmission.
- ⚠️ Not yet integrated anywhere:
  - `src/analysis/features.py:compute_silver_features()` doesn’t call it.
  - `src/analysis/regimes.py` doesn’t incorporate shock-based regimes.
  - Existing report scripts don’t plot these signals.

---

## 3) Model-Risk / Investment Banking Review (What to Validate)

### 3.1 Core risks you must control
- **Narrative risk**: “Navier–Stokes in markets” can look like physics cosplay unless you clearly define: domain, boundary/initial conditions, and why residual constraints improve forecasts out-of-sample.
- **Leakage risk**: shock labels are forward-looking; ensure strict time splits and *feature availability constraints* (T+0 vs T+1).
- **Non-stationarity**: shocks differ by era (2008, 2011 silver spike, 2020 COVID, 2022 inflation); you need regime-aware validation.
- **Interpretability risk**: separability claims must match actual operations (elementwise vs outer product; modality-separable vs coordinate-separable).

### 3.2 Minimum validation stack (objective measurements)
For a thesis + IB-grade story, do **all** of the following:
- **Shock prediction classification**
  - Primary: AUC, Precision@K, Recall@K for lead times {1, 3, 5} days
  - Calibration: Brier score + reliability plot (probability calibration)
- **Event study on known shocks**
  - Mean/median cumulative returns around shock windows
  - Signal lead time distribution (days of advance warning)
- **Forecasting (if predicting returns/momentum)**
  - MAE/MSE on next-step momentum/returns
  - Diebold–Mariano test vs baseline models (optional but strong)
- **Trading/risk proxy**
  - Simple rules-based overlay (e.g., reduce exposure when `shock_prob` high)
  - Report: turnover, drawdown reduction, hit rate, net PnL with costs (even if toy)

Baselines to include (so SPIKAN is credible):
- Logistic regression / random forest on the same features
- Simple volatility-trigger + momentum reversal rule
- Markov regression/HMM with/without exogenous fluid features

---

## 4) Recommended Next Actions (Prioritized)

### A) Integrate fluid features into the existing silver feature set (high ROI, no training)
1. Add an optional flag to `src/analysis/features.py:compute_silver_features()` (e.g., `include_fluid_features: bool = False`).
2. Call `src/analysis/fluid_dynamics.py:compute_fluid_dynamics_features()` and join with existing features.
3. Add a plot/table in `scripts/silver_report.py` for:
   - `shock_formation_index_z`, `momentum_decay_rate`, `flow_divergence`
4. Add validation: show these spike around historical shock windows.

### B) Build the SPIKAN dataset + training script (minimum viable PINN)
Create:
- `scripts/train_spikan.py`:
  - build tensors from aligned `close_panel` + geometric + macro
  - strict time split (train/val/test by date)
  - training loop + checkpoint + metrics dump to `reports/silver/tables/`
- `scripts/eval_spikan.py`:
  - AUC + Precision@K + calibration, plus event study plots

### C) Make the “geometry ↔ physics” coupling defensible (thesis-grade)
Replace “derivative w.r.t. asset index” with a **graph-domain diffusion operator**:
- Use correlation/geometry graph `G(t)` and its Laplacian `L(t)`
- Replace `u_xx` with `L u` (discrete Laplacian) and justify it as diffusion on the market manifold
- Curvature/stress can modulate diffusion/viscosity (your Ricci ↔ PDE bridge)

This strengthens the math and fixes the conceptual mismatch of treating discrete assets as a continuous spatial coordinate.

---

## 5) “Multi-Agent” Work Breakdown (with objective measures)

Use these agents as parallel workstreams (human or LLM-assisted), each with measurable outputs.

1. **Data Engineer Agent**
   - Deliverable: reproducible dataset builder + schema (features available at time T)
   - Metrics: data completeness %, no-leakage checks, run time per build

2. **PINN/SPIKAN Engineer Agent**
   - Deliverable: training script + stable loss balancing + checkpointing
   - Metrics: convergence diagnostics (loss curves), out-of-sample AUC/MAE, runtime per epoch

3. **Geometry Agent**
   - Deliverable: graph operators for PDE coupling (Laplacian, curvature-driven coefficients)
   - Metrics: stability under window changes, sensitivity analysis, ablation vs no-geometry

4. **Quant Validation Agent**
   - Deliverable: baseline models + ablation study + event study notebook/report
   - Metrics: uplift over baseline (ΔAUC, ΔPrecision@K), robustness across eras

5. **Visualization/Thesis Agent**
   - Deliverable: 2–4 thesis-ready figures (regimes + shocks + probabilities + event windows)
   - Metrics: readability, clear captions, reproducible regeneration from scripts

---

## 6) Appendix: What was changed during cross-check

To make the repo runnable under Drive/FUSE constraints:
- `src/physics/__init__.py` now sets `tempfile.tempdir` to `.tmp/` early (needed for `torch` import via `dill`).
- `src/pipeline/silver_pipeline.py:_write_parquet_safe()` now writes Parquet via an in-memory buffer (avoids `tempfile.mkstemp` and `/tmp` writes).
- Note: running the full pytest suite in this environment requires `--capture=sys`.

