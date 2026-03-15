# 关键数据表格（TBME 强化版）

## 使用原则
这版表格的目标是让审稿人不依赖正文长段落，也能快速抓住三件事：

1. 数据协议是什么  
2. 纯 raw-input scientific line 到底赢在哪里  
3. deployment enhancement 又多带来了什么  

因此，这一版表格分成两层：
- `Main tables`：建议主文保留
- `Supplementary tables`：建议补充材料保留

---

## Main Table 1 数据协议与样本规模

| 项目 | 数值 / 说明 |
|---|---|
| 传感帧尺寸 | **12 × 8** |
| 单帧通道数 | **96** |
| 结节大小水平 | **7 个（0.25–1.75 cm）** |
| 埋藏深度水平 | **6 个（0.5–3.0 cm）** |
| 重复次数 | **3** |
| 总物理条件 | **42** |
| 总实验记录 | **126** |
| 窗口长度 | **10 帧** |
| 滑动步长 | **2** |
| Detection 划分 | **1.CSV + 2.CSV 开发，3.CSV 测试** |
| Detection train / val / test 窗口数 | **4648 / 1107 / 3747** |
| Detection train / val / test 阳性数 | **1328 / 223 / 961** |
| Size / Depth 划分 | **1.CSV 训练，2.CSV 验证，3.CSV 测试** |
| Raw size v2 val / test 阳性窗口数 | **593 / 953** |
| Raw route-aware depth v2 train / val / test 阳性窗口数 | **958 / 593 / 953** |

**建议正文要点**：把 detection 和 inversion 的划分协议分开写清楚，体现 detection-first 的问题组织方式。

---

## Main Table 2 Detection 结果

| 模型 | 角色 | AUC ↑ | AP ↑ | F1 ↑ |
|---|---|---:|---:|---:|
| XGBoost structured baseline | 结构化机制基线 | 0.8199 | 0.5063 | 0.6185 |
| Stage I raw+delta detector | raw scientific line | **0.8383** | **0.5357** | **0.6216** |

**建议正文要点**：在主任务 detection 上，raw-input 时空神经网络已经稳定优于 structured baseline。

---

## Main Table 3 Size 结果（真实阳性窗口）

| 模型 | 角色 | Top-1 ↑ | Top-2 ↑ | MAE (cm) ↓ | Count |
|---|---|---:|---:|---:|---:|
| XGBoost structured baseline | 结构化机制基线 | 0.6701 | 0.8023 | 0.1472 | 961 |
| Raw size-only router v2 | raw scientific line | **0.7177** | **0.8195** | **0.1242** | 953 |
| Unified hierarchical inverter | deployment enhancement | 0.6180 | 0.7859 | 0.1565 | 953 |

**建议正文要点**：优化后的 pure raw-input size router 已在 `Top-1 / Top-2 / MAE` 三个指标上全面超过 XGBoost。

---

## Main Table 4 Coarse depth：scientific line 与 deployment line

| 模型 / 路径 | 角色 | 评价口径 | Balanced Accuracy ↑ |
|---|---|---|---:|
| Majority baseline | 参考下限 | GT-positive coarse depth | 0.3333 |
| XGBoost structured baseline | 结构化机制基线 | GT-positive coarse depth | 0.5138 |
| Raw size-routed depth model v1 | raw scientific line | GT route | 0.5238 |
| Raw route-aware depth v2 | raw scientific line | GT route | **0.6066** |
| Raw size-routed depth model v1 | old predicted-route chain | Predicted route | 0.4822 |
| Raw route-aware depth v2 | raw scientific line | Predicted hard route | **0.5240** |
| Unified hierarchical inverter | deployment enhancement | Predicted hard route | **0.5337** |

**建议正文要点**：深度不是不可学，而是必须在 `size-aware + route-aware` 组织下建模；pure raw scientific line 已经在 predicted-route 上超过 XGBoost，而 unified model 继续提升部署鲁棒性。

---

## Main Table 5 延迟 benchmark

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

**建议正文要点**：不要写成“神经网络全面更快”。更准确的结论是：structured baseline 更适合作为离线机制参照，而 raw-input 神经网络更适合作为 detection-first 在线主线。

---

## Supplementary Table S1 Routing breakdown（raw scientific line）

| 路由模式 | Raw route-aware depth v2 BAcc ↑ |
|---|---:|
| GT size route | **0.6066** |
| Hard predicted-size route | **0.5240** |
| Soft predicted-size route | 0.5168 |
| Top2-soft predicted-size route | 0.5197 |
| Temperature-0.5 soft route | 0.5189 |
| Temperature-0.7 soft route | 0.5189 |
| Temperature-0.3 soft route | 0.5209 |
| Hard route when size is correct | 0.6380 |
| Hard route when size is wrong | 0.2383 |

**建议正文要点**：这张表特别适合支撑“真正阻碍部署的是 route mismatch，而不是 depth 信息不存在”。

---

## Supplementary Table S2 Explainability summary

### S2A Raw scientific line

| 指标 | 数值 |
|---|---:|
| Latent probe mean test `R^2` | **0.2712** |
| Size-only probe mean test `R^2` | 0.2356 |
| Hard-pair success rate | **0.7500** |
| Peak-neighborhood mean drop | **0.2303** |
| IG subset size (`shallow / middle / deep`) | **40 / 40 / 40** |
| Phase with largest occlusion drop | **peak neighborhood** |

### S2B Unified deployment line

| 指标 | 数值 |
|---|---:|
| Hybrid latent mean test `R^2` | **0.5512** |
| Trunk latent mean test `R^2` | 0.4226 |
| Size-only probe mean test `R^2` | 0.2985 |
| Hard-pair success rate | **1.0000** |
| Peak-neighborhood mean drop | **0.0802** |

**建议正文要点**：raw scientific line 负责证明“自动编码了部分物理结构”，并建议在正文中明确说明这里使用的是 current route-aware depth v2 的 latent probe、Integrated Gradients、hard-pair 与 phase occlusion 结果；unified line 则负责证明“部署模型也确实在利用这些结构”。

---

## Supplementary Table S3 Unified branch ablation

| 变体 | Size Top-1 ↑ | Size MAE (cm) ↓ | Depth hard-route BAcc ↑ |
|---|---:|---:|---:|
| Full | **0.6180** | **0.1566** | **0.5337** |
| No raw | 0.5488 | 0.2124 | 0.4165 |
| No shape | 0.1689 | 0.3587 | 0.3811 |
| No delta | 0.6023 | 0.1688 | 0.4705 |
| No tabular | 0.3935 | 0.3567 | 0.3742 |

**建议正文要点**：这张表更适合补充材料，因为它支持 deployment enhancement 的结构论证，但不是主文最核心的 scientific claim。
