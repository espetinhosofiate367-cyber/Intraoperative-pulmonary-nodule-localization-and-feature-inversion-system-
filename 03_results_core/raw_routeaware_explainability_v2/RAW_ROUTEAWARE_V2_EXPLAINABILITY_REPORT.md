# Raw Route-Aware Depth V2 Explainability Report

- Checkpoint: `C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\outputs_stage3_raw_size_routed_depth_v2\paper_stage3_raw_size_routed_depth_v2_best.pth`
- Protocol: `1.CSV train / 2.CSV val / 3.CSV test`, positive windows only
- Mean latent probe test R2: `0.2712`
- Mean size-only probe test R2: `0.2356`
- Hard-pair success rate by p_deep: `0.7500`
- Dominant occlusion phase: `peak_neighborhood` with mean drop `0.2303`

## Probe top features

- `spatial_entropy_max` (distribution_complexity): latent test R2 `0.5626`, size-only test R2 `0.2185`
- `raw_max_max` (amplitude_response): latent test R2 `0.5340`, size-only test R2 `0.4760`
- `window_norm_global_std` (shape_contrast): latent test R2 `0.5236`, size-only test R2 `0.5958`
- `center_border_contrast_max` (shape_contrast): latent test R2 `0.4737`, size-only test R2 `0.5967`
- `window_rise_time_to_peak` (temporal_phase): latent test R2 `0.3713`, size-only test R2 `0.0123`

## Integrated Gradients subset sizes

- `shallow`: `40` correctly predicted windows
- `middle`: `40` correctly predicted windows
- `deep`: `40` correctly predicted windows

## Interpretation

- The current route-aware depth v2 model still shows a stronger latent recoverability than a size-only baseline, which supports true depth-related internal encoding.
- Hard-pair analysis indicates that the model can distinguish a portion of deeper-vs-shallower cases even when raw-max gaps are reduced, but the evidence remains partial rather than definitive.
- Phase occlusion confirms that the dominant temporal evidence remains concentrated near the peak neighborhood, so temporal strategy is improved but not yet fully physics-complete.