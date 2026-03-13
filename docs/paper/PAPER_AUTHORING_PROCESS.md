# Paper Authoring Process

## 1. Purpose

This document defines how the manuscript draft should be turned into a
submission-quality paper without losing alignment with the repo.

The key rule is:

> the paper must be filled from versioned evidence artifacts, not from memory.

## 2. Source Of Truth

Use these sources in order:

1. benchmark result artifacts
2. canonical contract docs
3. benchmark specs and evidence matrix
4. manuscript draft

The manuscript is not the source of truth for metrics.

## 3. Writing Workflow

### Phase A. Structural Draft

Purpose:

- lock sections, claims, and placeholders

Artifacts already available:

- `MANUSCRIPT_DRAFT.md`
- `PAPER_PACKAGE_PLAN.md`
- `EVIDENCE_MATRIX.md`

### Phase B. Evidence Binding

Purpose:

- replace placeholders with metrics, figures, and tables from actual runs

Required actions:

- insert protocol benchmark numbers
- insert quantitative benchmark numbers
- insert integration-cost study numbers
- add figure/table references

### Phase C. Claim Tightening

Purpose:

- weaken or strengthen claims according to the observed evidence

Rule:

- if SPIKAN does not clearly dominate, keep the architecture claim primary
- do not let backend performance claims outrun the actual benchmark results

### Phase D. Submission Draft

Purpose:

- convert the evidence-bound manuscript into a consistent paper format

Potential output formats:

- Markdown manuscript for internal review
- LaTeX manuscript for final packaging

## 4. Section Ownership

Recommended writing ownership:

- introduction and framing
  - architecture lead
- contracts and methods
  - systems and data lead
- quantitative methods and benchmarks
  - quant lead
- integration-cost study
  - systems lead
- discussion and limitations
  - final synthesis pass

Even if one person writes everything, this ownership model helps structure the review.

## 5. Claim Discipline

Each section should distinguish:

- implemented now
- benchmarked now
- planned next

This avoids accidental inflation of the paper’s empirical claims.

## 6. Placeholder Replacement Rule

Every placeholder in the manuscript should be replaced only when:

1. the metric exists in a versioned artifact
2. the artifact path is known
3. the benchmark conditions are documented

If those conditions are not met, leave the placeholder or downgrade the sentence.

## 7. Figure Process

For every figure:

1. define the exact question the figure answers
2. define the source artifact(s)
3. define the caption before polishing the visual
4. ensure the text references the figure for one clear reason only

This prevents decorative figures that do not support a claim.

## 8. Table Process

For every table:

1. ensure one row/column structure is stable across models
2. report splits and units explicitly
3. include stress slices where relevant
4. avoid mixing incomparable tasks in one table

## 9. Review Passes

Recommended review order:

### Pass 1. Technical Truthfulness

Check:

- claims match code and benchmarks
- no planned work is written as completed work

### Pass 2. Architecture Coherence

Check:

- the narrow-waist DSL story is consistent end-to-end
- contracts and router logic are clearly explained

### Pass 3. Quantitative Rigor

Check:

- baselines are fair
- calibration and stress slices are reported
- ablations are included

### Pass 4. Writing Quality

Check:

- remove repetition
- shorten inflated framing
- keep results ahead of interpretation

## 10. Minimal Paper Completion Checklist

The draft can be promoted to a serious paper version only if all are true:

- manuscript sections are stable
- all primary placeholders are filled or explicitly deferred
- protocol benchmark is reported
- quantitative benchmark is reported
- integration-cost study is reported
- figures and tables map cleanly to the evidence matrix

## 11. Repo Process

Recommended git workflow:

- keep manuscript drafting on `feat/paper-draft`
- land benchmark implementations on their own feature branches
- only merge benchmark-backed wording into the main manuscript branch after evidence exists
- treat text-only claim upgrades as reviewable diffs, not silent edits

## 12. Final Output Targets

The authoring process should produce:

- one internal manuscript draft
- one evidence-bound manuscript
- one final formatted paper package
- one defense or presentation summary derived from the same evidence base
