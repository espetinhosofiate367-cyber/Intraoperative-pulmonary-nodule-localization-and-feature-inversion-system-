# 一种基于深度学习的多模态融合术中肺结节定位系统（初稿V2）

## 摘要
本研究面向术中肺结节定位，构建了“检测-大小反演-深度粗分类”的分层神经网络系统，并完成主程序集成与可解释性验证。系统输入为柔性阵列应力传感器采集的时序应力图（T=10，12×8），输出结节存在概率、结节大小和粗深度层级。为解决深度效应弱且与大小耦合导致的路由误差问题，提出统一层级反演器并显式优化预测路由下的深度性能。结果显示：检测AUC 0.8383（优于XGBoost 0.8199）；统一反演器在预测路由条件下深度平衡准确率达到0.5337（优于XGBoost 0.5138）；并通过latent probe、hard-pair与phase occlusion形成可解释性闭环。研究表明，深度信息可由原始时序应力数据学习获得，但其有效利用依赖大小感知路由与路由鲁棒性优化。

## 1 引言
术中肺结节定位长期依赖术前影像与术者经验。柔性阵列应力传感器可提供实时表面力学信息，但在深部和小结节场景下，信号受形变、接触工况和大小-深度耦合影响，导致误判风险增加。前期实验与统计分析已证实：结节大小是强主效应，而深度更表现为扩散、形态与时序阶段差异。本文目标是构建可部署、可解释的分层神经网络，实现从“检测”到“大小反演”再到“深度粗分类”的稳定闭环。

受 HMIL（Hierarchical Multi-Instance Learning）层级建模思想的启发，本文不再将检测、大小和深度视为彼此独立的平级输出，而是将任务组织为逐级细化的层级推理过程：先完成结节检测，再反演结节大小，最后在大小条件下进行深度粗分类。与病理 WSI 中的 bag-instance 层级对齐不同，本文将这种层级关系具体落实为 `size-routed depth experts` 与 `route-consistency optimization`，以处理本任务中特有的“大小-深度耦合”和路由误差传播问题。

## 2 方法
### 2.1 数据与协议
- 传感器阵列：每帧96通道，重排为12×8应力图。
- 样本：长度10帧滑动窗口（stride=2）。
- 标签：窗口中心帧规则用于检测，大小与深度使用实验条件标签。
- 划分：检测任务采用1+2开发/3测试；反演任务采用1训练、2验证、3测试，并仅在阳性窗口训练。

### 2.2 系统架构
#### 2.2.1 Stage1 检测
Dual-Stream（raw+delta）Spatial CNN + MS-TCN，输出结节概率 p_det。

#### 2.2.2 Stage2 大小反演
阳性窗口输入raw序列，输出size class（7类）与size regression（cm）。

#### 2.2.3 Stage3 深度粗分类
采用size-routed depth experts：在大小条件下选择深度专家头进行shallow/middle/deep分类。

#### 2.2.4 统一层级反演器
为提升真实部署性能，引入统一层级反演器（Hierarchical Positive Inverter），输入raw amplitude + normalized sequence + tabular physics features，联合优化：
- size cls + size ord + size reg
- depth_gt + depth_hard + depth_soft + depth_top2
- route consistency
该设计直接针对“预测路由深度性能”优化，而非仅优化GT路由。

该结构在思想上对应于“粗粒度到细粒度”的层级建模：`size` 分支提供更稳定的上层语义，`depth` 分支作为细粒度子任务在该条件下进一步判别。与 HMIL 采用 coarse/fine 双分支并在实例层和 bag 层进行一致性约束类似，本文在时序触觉任务中采用 `hard route / soft route / top2 route` 和 `route consistency` 来实现更适配部署场景的层级一致性优化。

### 2.3 可解释性设计
采用三层证据链：
1. 非深度学习机制分析（depth对扩散/形态/阶段响应影响）；
2. XGBoost可解释性（特征贡献）；
3. 神经网络解释（latent probe、hard-pair、phase occlusion）。

## 3 结果
### 3.1 检测结果
- Stage1 NN：AUC 0.8383，AP 0.5357，F1 0.6216。
- XGBoost：AUC 0.8199，AP 0.5063，F1 0.6185。
结论：检测任务上神经网络优于XGBoost。

### 3.2 大小反演结果（GT-positive）
- XGBoost：top1 0.6701，top2 0.8023，MAE 0.1472 cm。
- Stage2 size-only NN：top1 0.6600，top2 0.8405，MAE 0.2907 cm。
- Unified Inverter：top1 0.6180，top2 0.7859，MAE 0.1565 cm。
结论：统一反演器牺牲了部分独立size指标，以换取更强路由深度性能。

### 3.3 深度粗分类结果（核心）
平衡准确率（Balanced Accuracy）：
- Majority：0.3333
- XGBoost：0.5138
- Stage3 GT-route：0.5238
- Stage3 Predicted-route（旧链路）：0.4822
- Unified Inverter Hard Predicted-route：0.5337
结论：统一反演器首次把“预测路由深度性能”稳定推过XGBoost。

### 3.4 误差来源分析
GT-route到predicted-route性能下降表明链路瓶颈主要来自size路由误差传导。统一反演器通过多路由目标联合训练与一致性约束显著减小该差距。

### 3.5 可解释性结果
- latent probe：网络潜在表示可恢复多项深度相关特征，且优于size-only probe。
- hard-pair：对“大而深/小而浅”难例具备中等强度区分能力。
- phase occlusion：模型当前仍偏依赖峰值邻域，提示后续应强化loading/release阶段建模。

这些结果与前期机制分析及 XGBoost 可解释性共同构成了“结构先验 -> 原始输入学习 -> 事后解释验证”的证据链，也说明本文借鉴层级学习思想并非停留在任务拆分层面，而是进一步在网络内部形成了可检验的深度相关表征。

## 4 讨论
1. 仅做端到端深度分类易塌缩到多数类，size-aware路由是必要条件。
2. 统一反演器不是单点指标最优，但在真实部署链路上更优。
3. 当前不足：
- size MAE仍略逊于XGBoost；
- phase利用不足；
- 需进一步提高路由鲁棒性与难例分离能力。

## 5 结论
本文完成了可部署、可解释的术中肺结节定位分层系统。检测性能优于XGBoost，深度粗分类在预测路由条件下超过XGBoost基线，证明了“原始输入神经网络 + 路由鲁棒优化”在真实链路上的有效性。系统已形成“机制分析-基线解释-神经网络解释”闭环，可支持后续临床验证和系统迭代。

## 图表索引
- Fig_R1_detection_compare.png
- Fig_R2_size_compare.png
- Fig_R3_depth_compare.png
- Fig_R4_depth_confusion_hard.png
- Fig_R5_depth_confusion_gt.png
- testset_best_grid_hierarchical.png
