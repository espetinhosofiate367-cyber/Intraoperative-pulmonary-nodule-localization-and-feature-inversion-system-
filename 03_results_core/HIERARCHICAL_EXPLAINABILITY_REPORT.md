# Hierarchical Positive Inverter Explainability Report

## Probe Summary
- Mean hybrid latent test R2: `0.5512`
- Mean raw trunk test R2: `0.4226`
- Mean size-only test R2: `0.2985`

## Branch Ablation
- `full`: size_top1=0.6180, size_mae=0.1566, depth_bAcc=0.5337
- `no_raw`: size_top1=0.5488, size_mae=0.2124, depth_bAcc=0.4165
- `no_shape`: size_top1=0.1689, size_mae=0.3587, depth_bAcc=0.3811
- `no_delta`: size_top1=0.6023, size_mae=0.1688, depth_bAcc=0.4705
- `no_tabular`: size_top1=0.3935, size_mae=0.3567, depth_bAcc=0.3742

## Hard-Pair Summary
- Clean pair filters: `raw_max_max < 170`, `spatial_entropy_max < 0.97`, and both sides correctly classified.
- Pair count: `11`
- Deeper sample higher p_deep rate: `1.0000`
- Mean raw_max gap: `13.3382`

## Phase Occlusion
- Largest true-class probability drop came from `peak_neighborhood` with mean drop `0.0802`.
