# Integration Cost Migration

## 1. Purpose

This document explains how to make the systems claim measurable inside the repo.

## 2. Current Starting Point

The repo already has useful structural clues:

- task-specific branches
- separable workstreams
- a growing planning stack
- existing git snapshot and tree tools

What it does not yet have is explicit instrumentation for onboarding-cost evidence.

## 3. Migration Sequence

### Step 1. Freeze the cost rubric

Branch:
- `feat/integration-cost-study`

Artifacts:

- fixed metrics
- fixed task set
- comparison styles

Gate:
- no backend cost claims before the rubric is frozen

### Step 2. Add integration log format

Expected implementation branch later:
- `feat/integration-log-format`

Goal:
- record:
  - files touched
  - adapter files
  - conditional branches added
  - time to executable path

Gate:
- every onboarding attempt emits the same artifact format

### Step 3. Add backend onboarding case studies

Expected implementation branch later:
- `feat/backend-onboarding-cases`

Goal:
- run the same onboarding measurement against at least two backends

Gate:
- results are comparable and reproducible

### Step 4. Connect git evidence

Expected implementation branch later:
- `feat/integration-git-evidence`

Goal:
- tie the study to:
  - diff stats
  - branch summaries
  - file touch sets

Gate:
- evidence comes from versioned repo state, not memory

## 4. Likely Implementation Touch Points

- router or backend registry module
- benchmark docs
- git snapshot artifacts under ignored output paths
- experiment or benchmark result directories

## 5. Important Guardrails

Avoid vanity metrics.

The study should not reward an architecture just for hiding complexity in one giant file.

That is why:

- `files_touched`
- `custom_branches_added`
- and `interface_surface_area`

must be considered together.

## 6. Success Conditions

The migration succeeds if:

1. backend onboarding can be measured consistently
2. git evidence supports the measurements
3. the paper can compare integration styles with concrete numbers
4. the systems contribution is no longer rhetorical

## 7. Failure Signs

The migration is drifting if:

- onboarding effort is estimated from memory
- different backends are judged with different tasks
- direct glue paths are underspecified
- the study ignores router complexity and counts only adapter LOC
