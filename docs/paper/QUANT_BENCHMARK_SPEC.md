# Quantitative Benchmark Specification

## 1. Purpose

This document defines the quantitative benchmark needed to prove that SPIKAN is
not only mathematically interesting, but materially useful on the selected
regime and options tasks.

This is the direct evidence for the claim:

> SPIKAN is a specialist finance-native backend, not decorative math attached
> to a language interface.

## 2. Current Repo Reality

The current repo already has important pieces:

- `scripts/silver/train_spikan.py`
  - time splits
  - shock labels
  - AUC, Precision@K, MSE, MAE
- `src/options/spikan_outlook.py`
  - trader-facing outlook output
- `src/physics/market_pdes.py`
  - PDE residual metrics
- geometric and regime features from the silver pipeline

But the current quantitative proof is still incomplete because:

- baseline parity is not formalized
- calibration is not central yet
- stress-period reporting is not yet standard
- option-market proof is still pending the option data contract

## 3. Benchmark Objectives

The benchmark should answer four questions:

1. Does SPIKAN beat simpler baselines on at least one primary task?
2. Is SPIKAN calibrated enough to be trusted in stress-sensitive decisions?
3. Does SPIKAN retain value during known stress regimes rather than only calm periods?
4. Do geometry and physics terms contribute meaningfully, or are they mostly narrative?

## 4. Primary Task Families

### Task A. Shock / Stress Prediction

Input:

- state features at time `t`

Output:

- probability or score for stress/shock over horizon `h`

Recommended metrics:

- ROC-AUC
- PR-AUC
- Precision@K
- Recall@K
- Brier score

This is the most natural first task because `train_spikan.py` already supports it.

### Task B. Next-Step Return Or Momentum Forecast

Input:

- state features at time `t`

Output:

- next return or momentum target

Recommended metrics:

- MSE
- MAE
- correlation with realized target
- optional Diebold–Mariano comparison

This task helps test whether the physics-informed dynamics add directional value.

### Task C. Outlook Quality

Input:

- canonical market-state snapshot

Output:

- outlook direction and confidence

Recommended metrics:

- directional hit rate
- confidence calibration
- stress detection usefulness

This links SPIKAN from training to product-facing output.

### Task D. Option-Aware Task

Input:

- joined `StateOptionSnapshot`

Output:

- static or horizon option target

Recommended metrics:

- IV MAE/RMSE, preferred
- or mid-price MAE/RMSE if IV is not available
- regime-sliced errors

This task should begin only once the option-data contract is implemented.

## 5. Baseline Set

The baseline set must be fixed before running large experiments.

### Baseline 1. Rule-Based

- volatility trigger
- momentum reversal
- simple shock heuristics

Purpose:

- sanity floor

### Baseline 2. Markov / HMM

- regime model with optional exogenous inputs

Purpose:

- interpretable regime baseline

### Baseline 3. Tabular ML

- logistic regression
- random forest
- gradient boosting

Purpose:

- strong classical baseline on the same state vector

### Baseline 4. Pricing Baseline

- Black-Scholes or regime-adjusted pricing without SPIKAN dynamics

Purpose:

- option-specific comparison

### Baseline 5. Optional Sequence Baseline

- small temporal model if data and time justify it

Purpose:

- checks whether gains are due only to generic sequence capacity

## 6. Fairness Rules

To keep the comparison defensible:

- all models use the same train, validation, and test windows
- all models use the same state contract fields for the shared task
- all models use the same target definition
- leakage rules are identical across models
- option-task baselines use the same joined dataset

Without these rules, the benchmark becomes storytelling instead of evidence.

## 7. Time Splits

The repo already uses time-based splits in `train_spikan.py`.

That should become policy for all models.

Recommended structure:

- train
- validation
- test

And separately report:

- calm periods
- transition periods
- stress periods

The test report should always include a stress slice.

## 8. Required Metrics

### Core classification metrics

- ROC-AUC
- PR-AUC
- Precision@K
- Recall@K
- Brier score

### Core regression metrics

- MSE
- RMSE
- MAE

### Calibration metrics

- Brier score
- reliability curve
- calibration slope/intercept if practical

### Stress robustness metrics

- primary metric on stress-only slice
- delta vs full-sample metric

### Efficiency metrics

- runtime per epoch
- inference cost per batch

These are not headline results, but they matter for deployment realism.

## 9. Required Ablations

At minimum, the benchmark must include:

### A. No-geometry ablation

Remove geometry fields.

Purpose:

- test whether Ricci, MST, and GA actually help

### B. No-physics-loss ablation

Set PDE penalty off.

Purpose:

- test whether the PINN part contributes

### C. No-fluid / no-shock-feature ablation

Remove shock-formation or decay diagnostics where available.

Purpose:

- measure incremental value of fluid features

### D. No-tension ablation

For narrative-aware or option-aware experiments, remove manifold tension features.

Purpose:

- test the bridge between narrative state and quantitative state

These ablations are necessary because the paper bundles several ideas together.

## 10. Event Studies

The quantitative package should include event studies around known stress windows.

Minimum outputs:

- average lead time before shock windows
- behavior of predicted stress scores around events
- comparison of SPIKAN and baselines near those windows

This matters because average metrics alone can hide regime-specific failure.

## 11. Decision Rules

SPIKAN should only be presented as a strong backend if:

- it beats at least one serious baseline on one primary task
- calibration is acceptable
- stress performance does not collapse
- ablations show non-trivial value from geometry or physics structure

If not, the paper should narrow the claim:

- from superior backend
- to plausible specialist backend within a better interoperability architecture

## 12. Benchmark Outputs

The quantitative benchmark should produce:

- leaderboard table
- regime-sliced table
- stress-period table
- calibration figure
- event-study figure
- ablation table

## 13. Acceptance Criteria

The quantitative benchmark is ready when:

1. one baseline set is frozen
2. one reproducible benchmark runner exists
3. one stress-period slice is always reported
4. calibration is included, not optional
5. ablation results are part of the main result package
