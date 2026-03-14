# 关键数据表格（TBME 严谨版）

## 使用原则
这版表格不再只是“把数值摆出来”，而是要做到：

1. 任务边界清楚  
2. scientific main model 与 deployment model 不混写  
3. **关键数字加粗**  
4. 结论能从表格本身读出来  

---

## Table 1 数据协议与任务定义

| 项目 | 内容 |
|---|---|
| 传感帧尺寸 | **12 × 8** |
| 单帧通道数 | **96** |
| 结节大小水平 | **7 个（0.25-1.75 cm）** |
| 埋藏深度水平 | **6 个（0.5-3.0 cm）** |
| 重复次数 | **3** |
| 总物理条件 | **42** |
| 总实验记录 | **126** |
| 窗口长度 | **10 帧** |
| 滑动步长 | **2** |
| Detection 划分 | **1.CSV + 2.CSV 开发，3.CSV 测试** |
| Size/Depth 划分 | **1.CSV 训练，2.CSV 验证，3.CSV 测试** |
| Size/Depth 训练样本 | **仅真实阳性窗口** |

**建议正文要点**：主任务和次任务的划分协议不同，体现了 detection-first 的问题组织方式。

---

## Table 2 Detection 结果

| 模型 | 角色 | AUC ↑ | AP ↑ | F1 ↑ |
|---|---|---:|---:|---:|
| XGBoost structured baseline | 结构化机制基线 | 0.8199 | 0.5063 | 0.6185 |
| Stage I raw+delta detector | scientific main model | **0.8383** | **0.5357** | **0.6216** |

**建议正文要点**：在主任务 detection 上，raw-input 时空神经网络已经稳定优于 structured baseline。

---

## Table 3 Size 结果（真实阳性窗口）

| 模型 | 角色 | Top-1 ↑ | Top-2 ↑ | MAE (cm) ↓ |
|---|---|---:|---:|---:|
| XGBoost structured baseline | 结构化机制基线 | **0.6701** | 0.8023 | **0.1472** |
| Raw size-only router | raw scientific line | 0.6600 | **0.8405** | 0.2907 |
| Unified hierarchical inverter | deployment enhancement | 0.6180 | 0.7859 | 0.1565 |

**建议正文要点**：raw-input 神经网络已能学习强大小信息，但 standalone size regression 最优仍由 structured baseline 保持。

---

## Table 4A Coarse depth：scientific main line

| 模型 / 路径 | 评价口径 | Balanced Accuracy ↑ |
|---|---|---:|
| Majority baseline | GT-positive coarse depth | 0.3333 |
| XGBoost structured baseline | GT-positive coarse depth | 0.5138 |
| Raw size-routed depth model | GT route | **0.5238** |

**建议正文要点**：raw-input 神经网络在显式 `size-aware` 组织下，已可在 scientific main line 上略超 XGBoost。

---

## Table 4B Coarse depth：deployment line

| 模型 / 路径 | 评价口径 | Balanced Accuracy ↑ |
|---|---|---:|
| Raw size-routed depth model | Predicted route (old chain) | 0.4822 |
| XGBoost structured baseline | GT-positive coarse depth | 0.5138 |
| Unified hierarchical inverter | Predicted hard route | **0.5337** |

**建议正文要点**：deployment enhancement 的价值体现在真实 predicted-route 条件下，而不是替代 raw scientific line。

---

## Table 5 延迟 benchmark

### Table 5A Detection 路径

| 路径 | 口径 | 中位延迟 (ms) ↓ |
|---|---|---:|
| XGBoost detection | Model-only CPU | **0.5740** |
| NN detection | Model-only CPU | 6.9379 |
| XGBoost detection | End-to-end CPU | 27.3447 |
| NN detection | End-to-end CPU | **7.1143** |
| NN detection | End-to-end GPU | **2.8131** |

### Table 5B 完整链路

| 路径 | 口径 | 中位延迟 (ms) ↓ |
|---|---|---:|
| XGBoost cascade | End-to-end CPU | **9.4641** |
| NN two-stage | End-to-end CPU | 14.6890 |
| NN two-stage | End-to-end GPU | 14.5371 |

**建议正文要点**：不要把延迟写成“神经网络全面更快”。更准确的写法是：structured baseline 更适合作为离线机制参照，而 raw-input 神经网络更适合作为 detection-first 在线主线。

---

## Table 6 核心模型参数与训练超参数

| 项目 | 数值 |
|---|---|
| 输入窗口 | **10 × 1 × 12 × 8** |
| 窗口步长 | **2** |
| size 类别数 | **7** |
| coarse depth 类别数 | **3** |
| frame feature dim | **24** |
| temporal channels | **48** |
| temporal blocks | **3** |
| tabular hidden dim | **64** |
| dropout | **0.28** |
| optimizer | **AdamW** |
| learning rate | **2e-4** |
| weight decay | **1e-3** |
| batch size | **48** |
| epochs | **120** |
| patience | **24** |
| grad clip | **1.0** |

**建议正文要点**：这张表的作用不是追求超参数堆砌，而是让 Methods 看起来更完整、更可复现。
