# Final vs Archive

## Recommended working root
Use this folder as the GitHub upload root:
- `tbme_submission_release_v1/`

This workspace is intentionally separated from the large historical archive so the paper project can be reviewed, versioned, and reproduced cleanly.

## Final folders
These folders are considered actively useful for the TBME paper and system release:

- `01_manuscript/`
  - Current manuscript drafts, results drafts, figure/table plan, protocol, and system blueprint.
- `02_figures_final/`
  - Final or near-final paper figures and GUI/result images worth keeping.
- `03_results_core/`
  - Core JSON/MD summaries that support the main claims in the manuscript.
- `04_core_models/`
  - Minimal checkpoint set for the final detector, size router, and hierarchical inverter.
- `05_core_code/`
  - Minimal code needed to reproduce training, explainability, and GUI inference.
- `06_ai_figure_prompts/`
  - Ready-to-use prompts for AI-assisted figure generation.
- `07_overleaf_figure_pack/`
  - Overleaf-ready figure package with standardized names, PNG/PDF exports, figure captions, and table CSV files.

## Archive folder
- `90_archive_unused/`
  - Materials not selected as the final narrative path.
  - Includes older drafts, deprecated model prototypes, and indexes to obsolete experiment outputs.
  - Keep for traceability, but do not use as the primary source when writing the TBME paper.

## How to use this release
1. Write from `01_manuscript/`.
2. Insert figures from `02_figures_final/`.
3. Cite metrics only from `03_results_core/`.
4. Reproduce with `05_core_code/` and `04_core_models/`.
5. Treat `90_archive_unused/` as reference-only history.

## Important caution
Do not mix this release workspace with the outer historical repository when preparing GitHub submission. This folder is the clean publication-ready package.
