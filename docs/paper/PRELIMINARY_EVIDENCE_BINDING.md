# Preliminary Evidence Binding

## 1. Purpose

This document records which evidence from the current repo can already be bound
into the paper draft before the new benchmark program is implemented.

It exists to keep the manuscript honest:

- some numbers are already stable and citable
- some numbers must remain placeholders

## 2. Evidence That Can Be Bound Now

### A. Existing silver/thesis validation

Source:

- `docs/thesis/RESULTS_SUMMARY_FOR_THESIS.md`

Values currently safe to cite as prior or preliminary evidence:

- BASE accuracy: 64.50%
- MACRO accuracy: 66.67%
- SECTIONAL accuracy: 65.43%
- 48 walk-forward folds
- 3,213 deduplicated out-of-sample predictions
- stable-regime Sharpe:
  - BASE 2.354
  - MACRO 2.764
  - SECTIONAL 2.414

How to use:

- cite as existing real-data validation from the repo’s silver-regime track
- do not present as the final result of the LLM+SPIKAN paper

### B. Options / protocol / backend engineering maturity

Source:

- `docs/project_status/IMPLEMENTATION_PROGRESS.md`

Values currently safe to cite as subsystem evidence:

- 205 passing tests across the current options and LLM stack
- high-fidelity SPIKAN manifold alignment:
  - geometric correlation 0.9632
- PDE residual variance reduction:
  - 125,894 to 1.5882

How to use:

- cite as subsystem maturity or backend engineering evidence
- do not present as the final protocol benchmark or final quantitative benchmark

## 3. Evidence That Must Stay As Placeholders

These paper claims still need the new benchmark program:

- protocol valid DSL rate
- protocol semantic match rate
- final executable rate across a benchmark corpus
- SPIKAN vs baseline parity on the new shared benchmark harness
- integration-cost onboarding metrics
- option-data benchmark metrics tied to the new option contract

## 4. Manuscript Rule

When binding current evidence into the manuscript:

- label it as preliminary, prior-track, or subsystem evidence
- keep the final benchmark claims reserved for the new evaluation harnesses

This distinction should remain explicit in text, tables, and captions.
