# TBME Submission Release v1

This repository is a cleaned, reviewer-facing and author-friendly release package for the project on tactile sensing and deep learning for intraoperative pulmonary nodule localization.

## What This Repository Is
This is not a full historical development dump.
It is a structured release package optimized for:
1. manuscript writing,
2. figure and table finalization,
3. reviewer inspection,
4. minimal reproduction of the final pipeline.

## Current Status
- Target journal: `IEEE TBME`
- Current Chinese manuscript: `01_manuscript/MANUSCRIPT_DRAFT_V12_CN.md`
- Current project phase: `submission consolidation`
- Current principle: keep only active deliverables in active folders, move replaced material to archive folders.

## Fast Start Paths

### If you are the author
Start here:
1. `00_project_management/PROJECT_STATUS_BOARD.md`
2. `01_manuscript/README_CN_CURRENT.md`
3. `06_ai_figure_prompts/README_ACTIVE_PROMPTS_CN.md`

### If you are a reviewer
Start here:
1. `REVIEWER_QUICKSTART.md`
2. `03_results_core/`
3. `04_core_models/`
4. `05_core_code/README_REPRODUCTION.md`

## Active Deliverables
The active project now centers on the following deliverables:
- Manuscript: `01_manuscript/MANUSCRIPT_DRAFT_V12_CN.md`
- Figure/table plan: `01_manuscript/FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md`
- Figure blueprint: `01_manuscript/FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md`
- Table package: `01_manuscript/TABLES_TBME_READY_V3_CN.md`
- Core results: `03_results_core/`
- Minimal checkpoints: `04_core_models/`
- Minimal reproducible code: `05_core_code/`
- Active AI figure prompts: `06_ai_figure_prompts/`
- Overleaf draft: `08_overleaf_draft_v1/`

## Simplified Folder Structure
- `00_project_management/`
  Project status board, deliverable register, and active/archive policy.
- `01_manuscript/`
  Active paper-facing materials only.
  Use `01_manuscript/README_CN_CURRENT.md` as the manuscript entry.
- `02_figures_final/`
  Final or near-final figure assets.
- `03_results_core/`
  Stable summaries and reports that support the paper's claims.
- `04_core_models/`
  Final released checkpoints.
- `05_core_code/`
  Minimal code needed to inspect or reproduce the main pipeline.
- `06_ai_figure_prompts/`
  Active prompt pack for AI-assisted concept and layout generation.
- `07_overleaf_figure_pack/`
  Overleaf-ready asset pack.
- `08_overleaf_draft_v1/`
  English LaTeX draft skeleton.
- `90_archive_unused/`
  Replaced, weak, deprecated, or purely historical materials kept for traceability.

## Active vs Archived Materials
We explicitly separate three layers:
- `active core`
  files used in the current manuscript, figure, and reproduction workflow.
- `supporting notes`
  useful background and drafting notes that are still relevant but not daily entry points.
- `archived materials`
  replaced manuscript versions, old prompt packs, weak visualizations, deprecated docs, and admin leftovers.

## Where Unused or Replaced Files Go
Unused or replaced files are not deleted blindly.
They are moved into dedicated archive folders:
- `90_archive_unused/manuscript_history/`
- `90_archive_unused/deprecated_docs/`
- `90_archive_unused/legacy_figure_prompts/`
- `90_archive_unused/rejected_visualization_gallery_*/`
- `90_archive_unused/repo_admin_misc/`

## Recommended Navigation
1. `00_project_management/PROJECT_STATUS_BOARD.md`
2. `01_manuscript/README_CN_CURRENT.md`
3. `01_manuscript/MANUSCRIPT_DRAFT_V12_CN.md`
4. `01_manuscript/FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md`
5. `03_results_core/`
6. `05_core_code/README_REPRODUCTION.md`
7. `06_ai_figure_prompts/README_ACTIVE_PROMPTS_CN.md`
8. `08_overleaf_draft_v1/`

## Notes
- The repository is intentionally slimmer than the full internal development workspace.
- Historical exploration is preserved, but no longer mixed with current deliverables.
- If something looks missing from the active path, check `90_archive_unused/` before assuming it was deleted.
