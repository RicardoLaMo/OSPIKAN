# Paper Package Plan

## 1. Purpose

This document defines the final paper package that should emerge from the
planning branches.

The goal is to prevent the work from ending as a set of disconnected design
notes. The paper needs one coherent narrative, one evidence hierarchy, and one
figure/table package.

## 2. Core Thesis

Recommended thesis statement:

> We present a language-mediated architecture in which a constrained DSL serves
> as the interoperability contract between humans and specialist quantitative
> backends. In a regime-aware options setting, this architecture reduces
> orchestration burden while preserving financial structure, auditability, and
> quantitative discipline.

## 3. What The Paper Must Prove

The final paper package should prove four things:

1. the protocol layer works
2. the backend layer works
3. the architecture supports backend substitution cleanly
4. the full system operates on real data, not only synthetic or demo paths

## 4. Recommended Section Structure

### Section 1. Introduction

What to say:

- specialist models are hard to integrate into human workflows
- finance is an especially high-friction case because quantitative structure matters
- LLMs are useful as semantic coordinators only when constrained by an executable contract

### Section 2. System Architecture

What to show:

- human input
- LLM protocol layer
- DSL contract
- router
- backend adapters
- evidence-aware outputs

### Section 3. Market State And Option Contracts

What to show:

- canonical market-state snapshot
- canonical option snapshot
- joined benchmark table

### Section 4. Specialist Backends

What to show:

- SPIKAN backend
- pricing baseline backend
- regime baseline backend
- optional comparison backend

### Section 5. Protocol Benchmark

What to report:

- valid DSL rate
- semantic match rate
- retry burden
- controlled-failure behavior

### Section 6. Quantitative Benchmark

What to report:

- shock or regime prediction
- calibration
- stress slice
- option-aware benchmark where available
- ablations

### Section 7. Integration Cost Study

What to report:

- onboarding complexity
- files touched
- adapter complexity
- backend-swappability evidence

### Section 8. Discussion And Limits

What to say:

- limits of current option-data breadth
- limits of current asset coverage
- calibration and non-stationarity risks
- architecture contribution vs backend-performance contribution

## 5. Figure Package

The final paper should target five required figures.

### Figure 1. Architecture Figure

Content:

- human -> LLM -> DSL -> router -> backend -> result

Purpose:

- makes the systems contribution legible in one image

### Figure 2. State / Regime / Tension Timeline

Content:

- geometry features
- regime shifts
- optional manifold tension
- marked stress periods

Purpose:

- grounds the state representation

### Figure 3. Protocol Trace Figure

Content:

- natural-language query
- DSL output
- normalized request
- backend result

Purpose:

- proves the protocol story visually

### Figure 4. Quantitative Benchmark Figure

Content:

- SPIKAN vs baselines on one primary task

Purpose:

- proves the backend story

### Figure 5. Integration Cost Figure

Content:

- backend onboarding complexity across integration styles

Purpose:

- proves the systems advantage

## 6. Table Package

The final paper should target five required tables.

### Table 1. State And Option Contracts

Purpose:

- shows the narrow-waist schemas explicitly

### Table 2. Protocol Metrics

Purpose:

- quantifies NL-to-DSL reliability

### Table 3. Quantitative Leaderboard

Purpose:

- compares SPIKAN to baselines

### Table 4. Ablation Results

Purpose:

- shows which structural components matter

### Table 5. Integration Cost Summary

Purpose:

- turns the systems claim into measurable evidence

## 7. Claim Hierarchy

The paper should keep claims ordered from strongest to weakest.

### Claim Tier 1

- the architecture creates a reusable contract between humans and specialist models

### Claim Tier 2

- the protocol layer reliably compiles natural language into executable DSL

### Claim Tier 3

- SPIKAN is a plausible and competitive specialist backend for selected tasks

### Claim Tier 4

- the architecture reduces integration cost compared with backend-specific glue

If Tier 3 proves weaker than expected, Tiers 1, 2, and 4 can still support a strong paper.

## 8. Non-Negotiable Evidence

The paper should not be drafted as a full claim unless all of these exist:

- one protocol benchmark result set
- one real-data quantitative benchmark
- one stress-period evaluation slice
- one integration-cost comparison
- one backend-swappable execution path

## 9. Drafting Order

The writing order should be:

1. freeze figures and tables
2. draft methods and contracts
3. draft results
4. draft introduction and abstract last

That reduces the risk of overselling before evidence is stable.
