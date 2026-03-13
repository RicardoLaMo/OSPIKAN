# SPIKAN Baseline Migration

## 1. Purpose

This document explains how to move from the current standalone SPIKAN training
flow to a benchmark-parity evaluation stack.

## 2. Current Starting Point

The current `scripts/silver/train_spikan.py` already does several things right:

- strict time splits
- explicit shock labeling
- out-of-sample metrics
- checkpoint saving

That is a good engineering start.

The missing part is not "train SPIKAN harder". The missing part is:

- benchmark parity with baselines
- calibration reporting
- event-study packaging
- explicit ablations

## 3. Migration Sequence

### Step 1. Freeze benchmark tasks and targets

Branch:
- `feat/spikan-vs-baselines`

Artifacts:

- task list
- baseline list
- metric list
- fairness rules

Gate:
- no model comparison starts before these are frozen

### Step 2. Build shared evaluation harness

Expected implementation branch later:
- `feat/quant-eval-harness`

Goal:
- one evaluation pipeline used by SPIKAN and baselines alike

Touch points likely:

- evaluation module
- training scripts
- report generation

Gate:
- the same scorer runs all models

### Step 3. Add baseline runners

Expected implementation branch later:
- `feat/quant-baseline-runners`

Goal:
- add rule-based, Markov, and tabular baselines on the same state contract

Gate:
- each baseline emits comparable outputs

### Step 4. Add calibration and event studies

Expected implementation branch later:
- `feat/calibration-and-events`

Goal:
- make the benchmark decision-relevant, not just leaderboard-oriented

Gate:
- every primary model result includes calibration and stress-window reporting

### Step 5. Add ablation runners

Expected implementation branch later:
- `feat/spikan-ablations`

Goal:
- prove which components matter

Gate:
- no-geometry and no-physics-loss ablations are included by default

## 4. Important Fairness Rules

The benchmark becomes unreliable if any of the following happen:

- SPIKAN gets richer features than the baselines
- baselines get different time splits
- stress labels differ across models
- evaluation metrics differ across tasks without documentation

These must be prevented at the harness level.

## 5. Recommended Result Package

The first credible SPIKAN benchmark package should include:

- full-sample results
- stress-only results
- event-study figure
- calibration plot
- ablation table

This is the minimum needed for a thesis-quality claim.

## 6. Success Conditions

The migration succeeds if:

1. SPIKAN is compared fairly against frozen baselines
2. calibration and event-study outputs are standard
3. ablations are part of the main experiment path
4. the repo can support a defensible claim about backend value

## 7. Failure Signs

The migration is drifting if:

- SPIKAN remains the only model with a real runner
- baseline metrics are copied from different scripts or notebooks
- calibration is deferred indefinitely
- ablations are treated as optional cleanup instead of required evidence
