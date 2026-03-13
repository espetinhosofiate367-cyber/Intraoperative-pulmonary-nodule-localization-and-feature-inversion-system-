# XGBoost Explainability Report

## 1. Baseline definition
- The XGBoost baseline uses only handcrafted window-level tactile physics features.
- Detection is trained on `train_det` windows; size and depth are trained on ground-truth positive windows from `train_all`.
- The split protocol remains unchanged: `1.CSV + 2.CSV` for development and `3.CSV` for final testing.

## 2. Why this baseline is interpretable
- The model input is not a raw image tensor but a structured feature table with explicit physical meaning.
- Tree models permit global interpretation through gain importance and additive contribution analysis (`pred_contribs`).
- Therefore we can analyze both what the model uses globally and why a specific sample is classified as shallow, middle, or deep.

## 2.1 What this explanation can and cannot claim
- It can explain which engineered features the fitted tree model relied on.
- It can explain which feature families pushed one specific test sample toward shallow, middle, or deep.
- It cannot by itself prove a causal tissue-mechanics law.
- It should therefore be interpreted together with the earlier non-deep-learning mechanism analysis.

### Detection
- Top features by mean |contribution|: raw_max_mean, center_border_contrast_last, raw_p95_mean, meanframe_norm_border_mean, meanframe_raw_raw_max, center_mean_last, center_mean_center, border_mean_first
- Dominant concept families: amplitude_response (43.3%), shape_contrast (41.7%), deformation_position (10.0%), spread_extent (5.0%)

### Size classification
- Top features by mean |contribution|: center_border_contrast_max, meanframe_norm_center_border_contrast, raw_max_max, anisotropy_ratio_min, deltaframe_centroid_col, maxframe_raw_center_border_contrast, spatial_entropy_max, window_norm_global_std
- Dominant concept families: shape_contrast (41.8%), deformation_position (20.6%), spread_extent (18.7%), amplitude_response (18.2%)

### Size regression
- Top features by mean |contribution|: window_raw_global_std, window_norm_global_std, raw_p95_max, spatial_entropy_max, meanframe_norm_raw_p95, maxframe_raw_hotspot_radius, maxframe_raw_raw_p95, deltaframe_hotspot_radius
- Dominant concept families: amplitude_response (39.9%), spread_extent (27.8%), shape_contrast (18.0%), deformation_position (12.2%)

### Depth coarse
- Top features by mean |contribution|: hotspot_radius_max, maxframe_raw_centroid_row, centroid_row_min, raw_max_max, deltaframe_centroid_row, anisotropy_ratio_min, peak_count_min, anisotropy_ratio_last
- Dominant concept families: deformation_position (32.3%), shape_contrast (27.9%), spread_extent (24.9%), amplitude_response (12.0%)

## 3. Depth-specific reading
- Depth coarse test accuracy: `0.5151`
- Depth coarse balanced accuracy: `0.5138`
- Top depth concepts: centroid_row, hotspot_radius, anisotropy_ratio, center_border_contrast, centroid_col, raw_max, second_moment_spread, peak_count
- In the current XGBoost model, depth is dominated by deformation/position, shape/contrast, and spread features, while pure amplitude contributes less.
- This is consistent with the earlier conclusion that depth is not just a peak-strength effect.
- Temporal/phase features are present but not dominant in the current handcrafted baseline, which suggests either that the current temporal features are still too weak or that depth information is encoded more strongly in spatial morphology than in our present temporal summary.
- If amplitude had dominated depth too strongly, the model would have been more vulnerable to confusing large-deep with small-shallow cases.

## 4. Local explanation usage
- `depth_local_explanations.png` shows feature contributions for representative correct and misclassified depth samples.
- These panels explain the model's decision toward the predicted class, not the causal ground truth itself.
- For manuscript use, the preferred summary panel is `Fig_XGB_depth_explainability_tbme.png`, which reorganizes the XGBoost results into a TBME-style multi-panel figure including feature-family contribution share, classwise concept reshape, representative tactile cases, and partial dependence trends.

## 5. Limits
- XGBoost explanations are model-usage explanations, not direct physical proof.
- Highly correlated handcrafted features may share or redistribute credit across one another.
- Therefore these explanations should be read together with the earlier mechanism analysis, not in isolation.

## 6. Implication for the next neural network
- The next neural model should not try to learn depth from a single shared black-box feature.
- It should explicitly preserve at least these concept families: amplitude, spread, shape contrast, and temporal/phase response.
- A concept-guided or size-conditioned depth branch is therefore justified by this baseline.
