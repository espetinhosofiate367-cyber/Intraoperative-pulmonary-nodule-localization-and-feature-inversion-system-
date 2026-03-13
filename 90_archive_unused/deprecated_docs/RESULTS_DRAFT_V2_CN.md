# Results Draft V2 (CN)

## 1. 检测结果
在独立测试集（3.CSV）上，阶段一检测模型（raw+delta）达到：
- AUC = 0.8383
- AP = 0.5357
- F1@val-threshold = 0.6216

同口径 XGBoost 检测基线为：
- AUC = 0.8199
- AP = 0.5063
- F1 = 0.6185

结论：检测任务上，神经网络优于 XGBoost。

## 2. 大小反演结果
测试集真实阳性窗口（GT-positive）对比：
- XGBoost: top1=0.6701, top2=0.8023, MAE=0.1472 cm
- Stage2 size-only NN: top1=0.6600, top2=0.8405, MAE=0.2907 cm
- Unified Inverter: top1=0.6180, top2=0.7859, MAE=0.1565 cm

结论：
- size-only NN 在 top2 上有优势；
- Unified Inverter 在综合链路优化下牺牲了部分 size 指标；
- XGBoost 仍保持最优 MAE。

## 3. 深度粗分类结果（关键）
平衡准确率（Balanced Accuracy）对比：
- Majority baseline: 0.3333
- XGBoost (GT-positive): 0.5138
- Stage3 GT-route: 0.5238
- Stage3 predicted-route（旧链路）: 0.4822
- Unified Inverter hard predicted-route: 0.5337

结论：
1. 原始输入神经网络在正确路由（GT route）下已可超过 XGBoost。
2. 统一层级反演器进一步把“预测路由”深度性能提升到 0.5337，首次超过 XGBoost 0.5138。
3. 这说明本项目的核心增益来自“路由鲁棒性优化”，而不仅是单一 depth head 改进。

## 4. 误差来源与链路分析
由“GT route > predicted route”的差距可见，系统瓶颈主要来自 size 路由误差传导，而非 depth trunk 本身。
统一反演器通过联合优化 GT/Hard/Soft/Top2 路由目标与路由一致性约束，显著缓解了这一问题。

## 5. 可解释性证据
1. 机制分析显示深度更影响扩散、形态与阶段响应，而非简单峰值衰减。
2. XGBoost 可解释性显示 depth 主要依赖 deformation/shape/spread family。
3. 神经网络解释（latent probe、hard-pair、phase occlusion）表明模型已编码部分深度相关特征，但对 phase 的利用仍偏向 peak neighborhood。

结论：系统已形成“机制分析 -> baseline解释 -> 神经网络可解释验证”的闭环。

## 6. 图表索引（论文插图）
- Fig_R1_detection_compare.png
- Fig_R2_size_compare.png
- Fig_R3_depth_compare.png
- Fig_R4_depth_confusion_hard.png
- Fig_R5_depth_confusion_gt.png
- testset_best_grid_hierarchical.png
