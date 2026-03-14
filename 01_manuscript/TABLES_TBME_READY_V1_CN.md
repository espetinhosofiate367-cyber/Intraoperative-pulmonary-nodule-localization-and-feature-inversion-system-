# 关键数据表格（TBME 版）

## Table 1 数据协议与任务定义
| 项目 | 内容 |
|---|---|
| 传感帧尺寸 | 12 × 8 |
| 单帧通道数 | 96 |
| 结节大小水平 | 7 个（0.25-1.75 cm） |
| 埋藏深度水平 | 6 个（0.5-3.0 cm） |
| 重复次数 | 3 |
| 总物理条件 | 42 |
| 总实验记录 | 126 |
| 窗口长度 | 10 帧 |
| 滑动步长 | 2 |
| Detection 划分 | 1.CSV + 2.CSV 开发，3.CSV 测试 |
| Size/Depth 划分 | 1.CSV 训练，2.CSV 验证，3.CSV 测试 |
| Size/Depth 训练样本 | 仅真实阳性窗口 |

## Table 2 Detection 结果
| 模型 | AUC | AP | F1 |
|---|---:|---:|---:|
| XGBoost structured baseline | 0.8199 | 0.5063 | 0.6185 |
| Stage1 raw+delta detector | 0.8383 | 0.5357 | 0.6216 |

**建议正文结论**：raw-input 时空神经网络在主任务 detection 上优于结构化基线。

## Table 3 Size 结果（真实阳性窗口）
| 模型 | Top-1 | Top-2 | MAE (cm) |
|---|---:|---:|---:|
| XGBoost structured baseline | 0.6701 | 0.8023 | 0.1472 |
| Raw size-only router | 0.6600 | 0.8405 | 0.2907 |
| Unified hierarchical inverter | 0.6180 | 0.7859 | 0.1565 |

**建议正文结论**：原始张量本身已包含强大小信息，但 standalone size regression 最优仍由结构化基线保持。

## Table 4 Depth 结果
| 模型 / 路径 | 评价口径 | Balanced Accuracy |
|---|---|---:|
| Majority baseline | GT-positive coarse depth | 0.3333 |
| XGBoost structured baseline | GT-positive coarse depth | 0.5138 |
| Raw size-routed depth model | GT route | 0.5238 |
| Raw size-routed depth model | Predicted route (old chain) | 0.4822 |
| Unified hierarchical inverter | GT route | 0.5407 |
| Unified hierarchical inverter | Predicted hard route | 0.5337 |

**建议正文结论**：只有在 size-aware 组织下，raw-input 神经网络才可稳定学习 coarse depth；统一层级反演器进一步提升了真实 predicted-route 条件下的系统级性能。

## Table 5 延迟 benchmark
| 路径 | 口径 | 中位延迟 (ms) |
|---|---|---:|
| XGBoost detection | Model-only CPU | 0.5740 |
| NN detection | Model-only CPU | 6.9379 |
| XGBoost detection | End-to-end CPU | 27.3447 |
| NN detection | End-to-end CPU | 7.1143 |
| NN detection | End-to-end GPU | 2.8131 |
| XGBoost cascade | End-to-end CPU | 9.4641 |
| NN two-stage | End-to-end CPU | 14.6890 |
| NN two-stage | End-to-end GPU | 14.5371 |

**建议正文结论**：XGBoost 的表格模型本体在 CPU 上更快，但其显式特征提取增加了 detection 主线的端到端工程负担；raw-input 神经网络更适合作为 detection-first 在线主线。对于完整 size/depth 链路，不应把“神经网络全面更快”写成主结论。
