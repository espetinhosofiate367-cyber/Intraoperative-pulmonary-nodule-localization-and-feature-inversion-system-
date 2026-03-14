# Reviewer Quickstart

This file is the fastest entry point for reviewers.

## 1. Read the Current Paper Draft

Open:
- `01_manuscript/MANUSCRIPT_DRAFT_V12_CN.md`

If you only want the figure/story structure, open:
- `01_manuscript/FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md`
- `01_manuscript/FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md`

## 2. Verify the Main Reported Results

Open these result summaries directly:
- `03_results_core/stage1_detection_summary.json`
- `03_results_core/raw_size_v2/paper_stage2_raw_positive_size_v2_summary.json`
- `03_results_core/raw_routeaware_depth_v2/paper_stage3_raw_size_routed_depth_v2_summary.json`
- `03_results_core/raw_routeaware_depth_v2/stage3_predicted_size_routing_summary.json`
- `03_results_core/latency_benchmark_v1/latency_benchmark_summary.json`

These files cover the main claims in the manuscript:
- raw detector > XGBoost on detection
- raw size-only v2 > XGBoost on size
- pure raw route-aware depth > XGBoost on predicted-route coarse depth
- unified hybrid model gives the strongest deployment result

## 3. Inspect the Released Checkpoints

Minimal checkpoints:
- `04_core_models/paper_stage1_dualstream_mstcn_best.pth`
- `04_core_models/paper_stage2_raw_positive_size_best.pth`
- `04_core_models/paper_hierarchical_positive_inverter_best.pth`

## 4. Reproduce the Core Pipeline

Open:
- `05_core_code/README_REPRODUCTION.md`

This file explains:
- which scripts are the main scripts
- which scripts are historical or secondary
- what can be reproduced directly from the release package
- what requires the original tactile dataset

## 5. Important Scope Note

This release package supports:
- metric verification from saved summaries
- code inspection
- checkpoint-based pipeline inspection

Full retraining requires the original tactile dataset, which is referenced by the training scripts but is not bundled inside this release package.
