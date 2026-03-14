# 神经网络 vs XGBoost 延迟与优势对比

## 关键优势
- Detection: AUC 提升 `0.0185`，AP 提升 `0.0294`，F1 提升 `0.0031`。
- Raw-input coarse depth: GT-route bAcc 比 XGBoost 高 `0.0100`。
- Deployment depth: unified predicted-route bAcc 比 XGBoost 高 `0.0200`。
- Raw-input size top-2 比 XGBoost 高 `0.0382`。

## 说明
- model-only: 输入已经准备成模型格式，仅测推理本体。
- end-to-end: 从原始 10 帧窗口开始，包含必要的归一化、特征整理与模型调用。
- 若 XGBoost 在纯 CPU 表格推理上更快，应如实保留；神经网络的优势不一定来自单点 CPU 延迟，而主要来自 raw-input learning、层级建模和系统级性能。

## Model-only latency (median ms)
- `xgb_detection_cpu`: 0.5740 ms (p95 0.7914 ms)
- `nn_detection_cpu`: 6.9379 ms (p95 8.3095 ms)
- `nn_detection_gpu`: 6.9878 ms (p95 10.6281 ms)
- `xgb_size_cpu`: 1.2774 ms (p95 1.7077 ms)
- `nn_size_cpu`: 8.9822 ms (p95 10.7267 ms)
- `nn_size_gpu`: 10.9482 ms (p95 15.7062 ms)
- `xgb_depth_cpu`: 0.6884 ms (p95 1.1845 ms)
- `nn_depth_cpu`: 9.3417 ms (p95 11.3034 ms)
- `nn_depth_gpu`: 12.9426 ms (p95 21.0741 ms)
- `unified_cpu`: 10.1477 ms (p95 12.6149 ms)
- `unified_gpu`: 13.1302 ms (p95 19.0976 ms)

## End-to-end latency (median ms)
- `xgb_detection_cpu`: 27.3447 ms (p95 37.7715 ms)
- `nn_detection_cpu`: 7.1143 ms (p95 11.8408 ms)
- `nn_detection_gpu`: 2.8131 ms (p95 8.2247 ms)
- `xgb_size_depth_cpu`: 8.8638 ms (p95 9.6990 ms)
- `xgb_cascade_cpu`: 9.4641 ms (p95 10.1033 ms)
- `nn_two_stage_cpu`: 14.6890 ms (p95 17.4744 ms)
- `nn_two_stage_gpu`: 14.5371 ms (p95 17.1661 ms)