# Results Draft v1

## 1. 研究目标与结果口径

本研究将任务划分为三个层次：

1. 结节存在检测
2. 结节大小反演
3. 结节粗深度分类与可解释性分析

其中，深度任务采用更严格的独立重复验证口径：

- `1.CSV -> train`
- `2.CSV -> val`
- `3.CSV -> test`

仅使用真实正窗进行 size/depth 训练与评估。这样做的原因是，深度任务对 `size` 和重复条件高度敏感，不能继续沿用早期检测任务的条件组切分方式。

---

## 2. 非深度学习分析与基线结果

在引入神经网络之前，我们首先利用非深度学习方法分析了触觉时空应力数据中的可分信息。

### 2.1 机制分析

前期统计结果表明：

- `size` 是更强的主效应，主要影响峰值强度、总能量和中心集中度。
- `depth` 也会影响表面应力分布，但其影响主要体现在扩散、分布复杂度、形变位置和时序阶段响应，而不是简单的峰值衰减。

这意味着深度任务不是一个“越深越弱”的单调问题，而是一个与尺寸耦合、且具有阶段依赖性的弱效应识别问题。

### 2.2 XGBoost 基线

在相同协议下，基于手工物理统计特征的 `XGBoost` 模型取得了以下结果：

- Detection: `AUC = 0.8199`, `AP = 0.5063`, `F1 = 0.6185`
- Size: `top1 = 0.6701`, `top2 = 0.8023`, `MAE = 0.1472 cm`
- Depth coarse: `accuracy = 0.5151`, `balanced accuracy = 0.5138`

这些结果说明，`size/depth` 信息确实存在于结构化物理特征中；因此，如果神经网络在 `size/depth` 上表现较弱，问题不在于数据中没有信息，而在于网络尚未有效提取这些深度敏感线索。

---

## 3. 原始输入神经网络的深度学习结果

### 3.1 不进行 size-aware 解耦时，原始输入网络无法稳定学习深度

最初的 raw-input end-to-end depth 模型在正确验证口径下基本退化到多数类水平：

- test balanced accuracy `≈ 0.33`

这一结果表明，若不显式处理 `size-depth` 耦合关系，网络会被更强的尺寸/强度信息主导，无法稳定提取深度相关表示。

### 3.2 引入 size-routed depth experts 后，原始输入网络可以学习 coarse depth

随后，我们将深度分类器改为 `size-routed` 架构：模型仍然直接吃原始时序应力窗口，但根据 `size` 路由到不同的深度专家头。

在 `GT size route` 条件下，该模型达到：

- `accuracy = 0.5257`
- `balanced accuracy = 0.5238`

该结果已经接近 `XGBoost depth baseline = 0.5138`，说明**原始输入神经网络确实能够学习 coarse depth 信息，但前提是显式建模其 size-dependent 特性。**

这一结果也是全文深度章节的关键转折点：问题不在于“原始输入是否可学”，而在于“是否采用了正确的结构去表达 size 与 depth 的耦合关系”。

---

## 4. 网络是否学到了深度相关特征

为了检验该 raw-input size-routed 模型内部是否编码了具有物理意义的深度相关信息，我们冻结模型并在其潜在表示上训练线性 probe，去恢复前期分析中发现的深度敏感描述量。

### 4.1 Latent probe 结果

整体上：

- mean test `R^2` of latent probe: `0.3657`
- mean test `R^2` of size-only probe: `0.2356`

这说明模型潜在表征中包含超出 `size` 本身的附加信息。

进一步地，latent representation 对以下特征的恢复能力明显优于 `size-only`：

- `spatial_entropy`
- `temporal_phase`
- `deformation_position`

因此可以认为，raw-input size-routed 模型已经编码了一部分与深度相关的空间复杂度、时序阶段性和形变位置信息，而不只是简单地记住了结节大小。

### 4.2 Hard-pair 分析

我们进一步构造了 `大而深` 与 `小而浅` 的 hard pairs，并尽量匹配其峰值幅值。

结果显示：

- pair count: `8`
- deeper sample 获得更高 `p_deep` 的比例：`75.0%`
- matched pair 的平均 `raw_max` 差值仍有 `13.05`

这说明模型并非完全只依赖峰值幅值进行深度判断，但该证据仍属于中等强度，尚不足以支持“模型已经稳健地解决了大而深/小而浅混淆”的强结论。

### 4.3 Phase occlusion 分析

我们通过遮挡不同按压阶段来观察模型对深度真类概率的下降幅度。结果表明：

- `peak_neighborhood`: mean drop `0.1619`
- `loading_early`: mean drop `0.0573`
- `loading_late`: mean drop `0.0407`
- `release`: mean drop `-0.0167`

该结果说明，当前模型最依赖的是峰值邻域，而不是前期非深度学习分析中提示的 `loading/release` 线索。这意味着模型已经学习到部分深度相关结构，但其时间策略仍然偏向峰值阶段，尚未完整恢复前述机制分析中的全部 phase pattern。

---

## 5. 深度性能受 size 路由精度显著制约

虽然 `GT size route` 下深度模型能够达到 `balanced accuracy = 0.5238`，但在真实部署条件下必须使用预测的 `size` 进行路由。

我们首先评估了此前 `Stage 2 raw multitask` 模型的 size 路由能力。在 `Stage 3` 测试正窗上：

- size top1 `= 0.5226`
- size top2 `= 0.7471`

在此路由器下，depth 性能明显下降：

- hard predicted-size route: `balanced accuracy = 0.4408`
- soft predicted-size route: `balanced accuracy = 0.4389`

进一步分析发现：

- 当 size route 正确时，depth balanced accuracy `= 0.5557`
- 当 size route 错误时，depth balanced accuracy `= 0.3204`

这说明，**当前部署链路的主要瓶颈不是 depth trunk 本身，而是上游 size router 的误差传播。**

---

## 6. 专用 size router 可以显著改善整条链路

为了验证这一点，我们训练了一个仅在真实正窗上学习 `size` 的 raw-input 专用模型，不再让 detection 或 depth 任务参与干扰。

该专用 size router 取得：

- val top1 `= 0.7083`
- val top2 `= 0.8432`
- test top1 `= 0.6600`
- test top2 `= 0.8405`
- test MAE `= 0.2907 cm`

该结果明显优于此前共享多任务模型在相同 Stage 3 正窗上的 `size top1 ≈ 0.52`，并已接近 `XGBoost size baseline top1 = 0.6701`。

当用该专用 size router 为 Stage 3 提供预测路由时，depth 性能进一步提升至：

- hard predicted-size route: `balanced accuracy = 0.4822`
- soft predicted-size route: `balanced accuracy = 0.4758`

对应地：

- size route 正确率提高到 `0.6422`
- hard-route 条件下 depth balanced accuracy 与 `GT route` 的差距缩小到约 `0.0416`

这一结果表明，当前深度链路已经基本清楚：

1. 原始输入网络本身可以学习 coarse depth
2. 其关键前提是采用 `size-aware` 结构
3. 实际部署时的主要误差来源是 `size route`，而非 depth trunk 本身

---

## 7. 当前最稳妥的论文结论

基于以上结果，当前论文 `Results` 部分可以得出以下结论：

1. 原始时空应力数据中存在可学习的粗深度信息。
2. 若不进行 `size-aware` 解耦，raw end-to-end 网络难以稳定学习深度。
3. `size-routed` 原始输入神经网络可将 coarse depth 性能提升至与 `XGBoost` 相当的水平。
4. latent probe 证明模型内部编码了部分深度相关概念，尤其是分布复杂度、时序阶段与形变位置，而不只是结节大小。
5. hard-pair 与 phase occlusion 表明模型已经部分摆脱单纯的峰值依赖，但其时间策略仍偏向峰值邻域，尚未完全对齐机制分析中的完整 phase pattern。
6. 在真实部署链路中，depth 性能的主要瓶颈来自 `size router` 的误差传播。
7. 使用专用 raw-input size router 后，整条 `size -> depth` 链路显著改善，说明“先稳住 size，再进行 depth routing”是合理且必要的系统设计。

---

## 8. 当前结果的边界

目前仍然不能写成以下强结论：

- “模型已经完整恢复了深度的物理机制”
- “模型已经稳健区分所有大而深与小而浅结节”
- “深度 fine-grained 反演已经解决”

当前最合适的表述是：

- 已证明粗深度信息在原始输入中可学习
- 已证明 `size-aware` 结构是必要条件
- 已证明模型内部编码了部分深度相关表征
- 但模型的 phase 利用模式与非深度学习机制分析仍存在偏差，因此深度结果应被写为**机制支持下的粗粒度、可解释分类结果**，而不是完全解决的精细深度反演任务
