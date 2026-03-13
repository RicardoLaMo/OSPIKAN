# Paper Build Process

## 1. Purpose

This document defines how to build the manuscript locally and how to manage
draft paper artifacts without polluting the repo.

## 2. Current Manuscript Sources

Primary draft sources:

- `docs/paper/MANUSCRIPT_DRAFT.md`
- `docs/paper/MANUSCRIPT_DRAFT.tex`

Supporting process docs:

- `docs/paper/PAPER_AUTHORING_PROCESS.md`
- `docs/paper/FIGURE_TABLE_WORKFLOW.md`

## 3. Build Commands

### Quick LaTeX build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/ospi_paper_build \
  docs/paper/MANUSCRIPT_DRAFT.tex
```

Expected output:

- `/tmp/ospi_paper_build/MANUSCRIPT_DRAFT.pdf`

### Clean temporary build outputs

```bash
latexmk -C -outdir=/tmp/ospi_paper_build docs/paper/MANUSCRIPT_DRAFT.tex
```

## 4. Artifact Policy

Do not commit generated PDFs or temporary LaTeX build files by default.

Allowed in git:

- manuscript sources
- figure/table source specs
- benchmark-bound captions and placeholders

Not allowed by default:

- generated `.pdf`
- generated `.aux`, `.fdb_latexmk`, `.fls`, `.log`, `.out`

## 5. Recommended Build Workflow

1. edit `MANUSCRIPT_DRAFT.md` or `MANUSCRIPT_DRAFT.tex`
2. bind or update benchmark placeholders from versioned artifacts
3. compile the LaTeX draft locally
4. review for broken references, overlong tables, and wording drift
5. only then promote text changes into review

## 6. Failure Triage

If LaTeX build fails:

1. fix syntax errors first
2. remove fragile package usage before debugging layout polish
3. keep tables simple until benchmark outputs stabilize
4. avoid adding bibliography dependencies until the structure is stable

## 7. Promotion Path

Recommended manuscript promotion path:

1. Markdown draft
2. LaTeX draft with placeholders
3. evidence-bound LaTeX draft
4. review PDF
5. submission package
