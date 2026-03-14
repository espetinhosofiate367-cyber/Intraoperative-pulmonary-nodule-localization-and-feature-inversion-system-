# Reproduction Notes

This folder contains the minimal code used by the final paper story.

## The Main Scripts

The core training and evaluation path is:

1. Detection
- `models/train_stage1_dualstream_mstcn.py`

2. Raw-only size model
- `models/train_stage2_raw_positive_size_v2.py`

3. Raw-only route-aware depth model
- `models/train_stage3_raw_size_routed_depth_v2.py`

4. Predicted-size routing evaluation
- `experiments/evaluate_stage3_predicted_size_routing.py`

5. XGBoost structured baseline
- `experiments/train_xgboost_baselines.py`

All five scripts above expose a direct `main()` entry and can be executed with:
- `python <script_name>.py`

## What Reviewers Can Reproduce Quickly

### A. Inspect the architecture
Open:
- `models/dual_stream_mstcn_detection.py`
- `models/raw_positive_size_model_v2.py`
- `models/hierarchical_positive_inverter.py`

### B. Inspect the final protocol
Open:
- `models/task_protocol_v1.py`

### C. Re-run predicted-route depth evaluation
Use:
- `experiments/evaluate_stage3_predicted_size_routing.py`

This is one of the most direct scripts for checking the reported depth-routing behavior.

### D. Re-run the main training scripts if the dataset is available
From `05_core_code/`, the minimal commands are:

```powershell
python models/train_stage1_dualstream_mstcn.py
python models/train_stage2_raw_positive_size_v2.py
python models/train_stage3_raw_size_routed_depth_v2.py
python experiments/evaluate_stage3_predicted_size_routing.py
python experiments/train_xgboost_baselines.py
```

Typical outputs are written into the corresponding experiment output folders and summary JSON files.

## Dataset Note

The training scripts expect the original tactile dataset outside this release package.
By default, several scripts reference:
- `../整理好的数据集/建表数据`

Therefore:
- checkpoint inspection: supported
- saved-metric verification: supported
- full retraining from scratch: requires the original dataset

## Fastest Verification Route

If the goal is only to verify the paper claims without retraining:
1. read the JSON summaries in `03_results_core/`
2. inspect the released checkpoints in `04_core_models/`
3. inspect the model definitions in `models/`
4. optionally run `experiments/evaluate_stage3_predicted_size_routing.py`

## Recommended Reviewer Workflow

1. Read saved result summaries in `03_results_core/`
2. Inspect the model definitions in `models/`
3. Inspect the released checkpoints in `04_core_models/`
4. Optionally run the evaluation scripts if the original dataset is available
