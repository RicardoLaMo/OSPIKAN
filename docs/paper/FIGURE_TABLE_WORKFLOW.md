# Figure And Table Workflow

## 1. Purpose

This document defines how figures and tables should be created for the paper so
that each one maps to a specific claim and benchmark artifact.

## 2. Rule

Every figure and every table must answer one question only.

If a visual tries to justify multiple unrelated claims, split it.

## 3. Required Figure Set

### Figure 1. Architecture

Question:

- What is the system architecture and where is the narrow waist?

### Figure 2. State / Regime / Tension Timeline

Question:

- How does the market-state representation evolve across stress and transition periods?

### Figure 3. Protocol Trace

Question:

- How does a natural-language request become a backend-executable result?

### Figure 4. Quantitative Comparison

Question:

- Does SPIKAN beat or match meaningful baselines on the chosen task?

### Figure 5. Integration Cost

Question:

- Does the architecture reduce onboarding complexity?

## 4. Required Table Set

### Table 1. Contract Summary

Question:

- What are the canonical interfaces?

### Table 2. Protocol Metrics

Question:

- How reliable is NL-to-DSL?

### Table 3. Quantitative Leaderboard

Question:

- How does SPIKAN compare with baselines?

### Table 4. Ablations

Question:

- Which components actually matter?

### Table 5. Integration Cost Summary

Question:

- What does backend onboarding cost under each integration style?

## 5. Workflow For Each Figure Or Table

1. write the question first
2. identify the exact benchmark artifact path
3. decide the minimal fields needed
4. draft the caption before polishing the visual
5. place the figure or table in the manuscript only after the caption and question align

## 6. Naming Convention

Use stable ids in text and source planning:

- `fig_architecture`
- `fig_state_timeline`
- `fig_protocol_trace`
- `fig_quant_comparison`
- `fig_integration_cost`
- `tab_contracts`
- `tab_protocol`
- `tab_quant`
- `tab_ablations`
- `tab_integration_cost`

## 7. Caption Rule

Captions should state:

- what is plotted
- what comparison or metric matters
- what the reader should infer

They should not restate the entire section.

## 8. Quality Check

Before freezing the paper package, verify:

- every figure is cited in the manuscript
- every table is cited in the manuscript
- every citation corresponds to one explicit claim in the evidence matrix
- no figure or table depends on unversioned manual editing
