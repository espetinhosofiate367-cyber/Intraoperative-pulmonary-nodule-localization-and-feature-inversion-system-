# Depth Physics Report

## 1. Data Scope
- Positive-frame depth analysis pooled all three repetitions across 42 size-depth conditions.
- Positive frame count: `14546`.
- Positive window count: `7301`.
- Phase reconstruction used the normalized `raw_sum` trajectory of each record and split the pressing process into `loading_early, loading_late, peak_neighborhood, release`.

## 2. What Depth Changes
- Global pooled depth correlations should only be treated as auxiliary evidence.
- Amplitude features: `{"raw_max": -0.04, "raw_p95": -0.009, "raw_sum": 0.16, "raw_mean": 0.16}`
- Spread features: `{"hotspot_area": 0.12, "second_moment_spread": 0.194, "hotspot_radius": 0.206, "spatial_entropy": 0.225}`
- Shape features: `{"center_border_contrast": -0.034, "center_mean": 0.046, "anisotropy_ratio": -0.1, "peak_count": 0.027}`
- The pooled result confirms that depth is weaker than size and should not be written as a simple monotonic attenuation.

## 3. Stable vs Exploratory Findings
- Stable depth-sensitive features under the preset rule: `none`
- Features that remain mixed or non-monotonic should be described as condition-dependent rather than globally monotonic.
- This means depth mainly changes the surface response through a combination of amplitude attenuation, hotspot spreading, and timing shifts, with the exact trend depending on size.

## 4. Phase Sensitivity
- The phase with the strongest mean depth sensitivity is `loading_early`.
- Phase sensitivity scores: `{"loading_early": 0.45, "loading_late": 0.388, "peak_neighborhood": 0.413, "release": 0.442}`
- Depth conclusions in the manuscript should therefore prioritize phase-resolved analysis instead of full-sequence averaging.

## 5. Simplified Contact-Propagation Interpretation
- A deeper subsurface stiffness source is expected to project to the surface through a broader and weaker stress field.
- In the current dataset, this shows up more as a redistribution of hotspot width, entropy, and timing than as a clean linear drop in peak value.
- The non-monotonic cases are physically plausible because sensor compliance, local pressing trajectory, and nodule size jointly modulate how the deep stiffness contrast reaches the surface.

## 6. Paper-safe Interpretation Boundary
- We can state that depth influences the tactile stress pattern.
- We should not state that depth follows a universal linear decay law.
- We should explicitly say that depth effects are `phase-dependent` and often `size-dependent`.

## 7. Figures
- D1: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_physics_v1\D1_fixed_size_depth_amplitude_trends.png
- D2: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_physics_v1\D2_fixed_size_depth_spread_trends.png
- D3: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_physics_v1\D3_fixed_size_depth_temporal_trends.png
- D4: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_physics_v1\D4_phase_resolved_depth_heatmap_grid.png
- D5: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_physics_v1\D5_shallow_vs_deep_representative_heatmaps.png
- D6: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_physics_v1\D6_repeat_consistency.png
