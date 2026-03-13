# 结节应力机制探究报告

## 1. 数据与标签定义
- 数据范围：42 个 `size-depth` 条件，3 次重复实验，共 126 条时序记录。
- 帧级正样本：人工标注正段内的帧；帧级负样本：同一 CSV 中未标注帧。
- 窗口级样本：`seq_len=10`、`stride=2`；正窗定义为窗口中心帧落在正段内。
- 覆盖检查：公共条件组数 `42`，重复数 `3`，窗口中心帧标签可追溯 `True`。

## 2. 正负帧差异
- 在平衡主统计中，正帧与负帧数量分别为 `14546` 和 `14067`。
- `raw_max`：正帧均值 `93.04`，负帧均值 `40.11`，Hedges g=`1.31`。
- `raw_sum`：正帧均值 `2825.41`，负帧均值 `1125.85`，Hedges g=`1.05`。
- `center_border_contrast`：正帧均值 `34.95`，负帧均值 `11.79`，Hedges g=`0.97`。
- `norm_hotspot_area`：正帧均值 `0.1380`，负帧均值 `0.0806`，Hedges g=`0.68`。
- 这说明结节帧的应力热点更强、更集中，且中心区域相对边缘的增强更加明显。

## 3. Size 效应
- 全局相关显示，`size` 与 `raw_max`、`raw_sum`、`center_border_contrast`、`norm_hotspot_area` 的相关系数分别为 `0.890`、`0.743`、`0.897`、`0.798`。
- 整体上，结节越大，峰值越高、整体能量越强、中心集中性越明显，热点面积也更大。
- 这种趋势在固定深度的小图阵中也保持较好一致性，说明 `size` 是当前数据里更稳定、更强的主导因素。

## 4. Depth 效应
- 全局相关显示，`depth` 与 `raw_max`、`raw_sum`、`center_border_contrast`、`norm_hotspot_area` 的相关系数分别为 `-0.040`、`0.160`、`-0.034`、`0.120`。
- 深度效应明显弱于尺寸效应，而且在固定尺寸下常常表现为非单调变化，而不是简单的“越深越弱”。
- 因此论文正文应避免把深度影响写成强线性衰减，更合适的表述是“深度会改变热点形态与峰值分布，但其效应弱于尺寸，并存在条件依赖性”。

## 5. Size-Depth 交互
- 条件级线性模型 `feature ~ size + depth + size*depth` 已针对四个核心特征拟合并写入 `mechanism_summary.json`。
- 从当前结果看，`size` 主效应通常强于 `depth` 主效应；交互项存在，但不应在没有额外物理验证时做过强机制解释。

## 6. 重复一致性
- 三次重复实验已单独做条件级均值比较，并计算重复间相关与条件内变异系数。
- 这一步用于判断结论是不是只来自单次偶然实验；如果某一特征重复间相关稳定，则更适合作为后续模型的主特征线索。

## 7. 对模型设计的启发
- 检测任务建议优先采用 `center-frame label`，因为当前正负差异主要体现在热点峰值与中心集中度上。
- `size` 预测可作为第二主任务，因为它在统计上更稳定。
- `depth` 更适合作为条件化或 masked 分支，而不是直接与检测头同权重硬绑定。
- 更合理的网络路线是“两阶段”：先检测结节存在，再对正窗做 `size/depth` 反演。

## 8. 论文表述边界
- 当前结论支持“柔性阵列应力传感器能够从按压时序中捕获与结节存在相关的空间集中信号”。
- 当前结论也支持“结节大小对信号强度和集中性影响更稳定”。
- 但对于深层极小结节，不能仅凭这套统计就宣称具有强鲁棒检出能力；这部分需要在最终测试结果中谨慎表述。

## 9. 交付文件
- Fig M1: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M1_positive_negative_frame_comparison.png
- Fig M2: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M2_average_heatmaps.png
- Fig M3: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M3_condition_grid_positive_heatmaps.png
- Fig M4: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M4_condition_trends.png
- Fig M5: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M5_fixed_depth_size_trends.png
- Fig M6: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M6_fixed_size_depth_trends.png
- Fig M7: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M7_window_temporal_evolution.png
- Fig M8: C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\mechanism_exploration_v1\Fig_M8_repeat_consistency.png
