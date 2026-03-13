# Results Draft V3 (CN)

## 1. 检测任务结果
在独立测试集（3.CSV）上，阶段一 raw+delta 检测器取得：
- AUC = 0.8383
- AP = 0.5357
- F1@val-threshold = 0.6216

相同协议下，XGBoost 检测基线为：
- AUC = 0.8199
- AP = 0.5063
- F1 = 0.6185

**结果解释**：检测任务是一个以局部时空模式为主的问题。相较于基于结构化统计特征的 XGBoost，原始输入时空神经网络可以更有效地利用动态应力分布中的局部模式，因此在排序能力和平均精度上更优。

## 2. 大小反演结果
测试集真实阳性窗口（GT-positive）上的大小反演结果如下：

- XGBoost：top1 = 0.6701，top2 = 0.8023，MAE = 0.1472 cm
- raw-input size-only router：top1 = 0.6600，top2 = 0.8405，MAE = 0.2907 cm
- 统一层级反演器：top1 = 0.6180，top2 = 0.7859，MAE = 0.1565 cm

**结果解释**：
1. XGBoost 仍保持最优 MAE，说明结构化物理特征对于大小回归十分有效。
2. size-only 神经网络在 top2 指标上最好，说明其对相邻尺寸类别的区分具有一定优势。
3. 统一层级反演器并未追求“独立 size 最优”，而是在一定程度上牺牲 standalone size 指标，以换取更强的路由深度鲁棒性。

## 3. 深度粗分类结果
深度粗分类的关键指标为 balanced accuracy，结果如下：

- Majority baseline：0.3333
- XGBoost：0.5138
- Stage3 GT-route：0.5238
- Stage3 predicted-route（旧链路）：0.4822
- Unified Inverter hard predicted-route：0.5337

**结果解释**：
1. 原始输入神经网络在正确 size 路由（GT-route）下已经可以达到甚至略超过 XGBoost 的水平。
2. 当 size 路由误差引入真实链路后，旧 predicted-route depth 性能明显下降，说明深度任务对上游 route 错误高度敏感。
3. 统一层级反演器通过 route-aware 训练，将真实 predicted-route depth 性能提升到 0.5337，首次稳定超过 XGBoost 0.5138。

## 4. 为什么深度任务必须 size-aware
最初的 raw-input end-to-end 深度分类模型在严格重复划分协议下的测试 balanced accuracy 约为 0.33，接近多数类基线。这表明：

1. 深度信息并不是不存在；
2. 问题出在模型结构没有显式处理 size-depth 耦合；
3. 当大小是强主效应、深度是弱主效应时，普通共享 depth head 会优先吸收大小和强度线索，从而导致深度分类塌缩。

因此，depth 任务必须在架构上被组织为大小条件下的细粒度判别问题，而不是一个脱离 size 的独立平级输出。

## 5. 可解释性结果
### 5.1 机制分析与 XGBoost 解释
前期机制分析表明，深度主要影响扩散、形态、分布复杂度和时序阶段响应，而不是简单峰值衰减。XGBoost 特征贡献分析进一步验证，depth 主要依赖 deformation-position、shape-contrast 和 spread-related feature families，而非纯 amplitude family。

### 5.2 latent probe
冻结 raw-input size-routed depth 网络后，在其 latent representation 上训练线性 ridge probes，得到：
- mean test R2 (latent) = 0.3657
- mean test R2 (size-only) = 0.2356

**解释**：网络内部表征能够恢复多项超出 size identity 的深度相关结构，尤其体现在 distribution complexity、temporal phase 和 deformation position 等特征家族上。

### 5.3 hard-pair analysis
在“大而深 / 小而浅”的 hard-pair 对中，更深样本在约 75% 的样本对上获得更高的 `p_deep`。  
**解释**：模型并非单纯依赖峰值强度进行深度判别。  
**边界**：当前 hard-pair 证据强度仍属中等，尚不足以宣称模型已完全解决该类混淆。

### 5.4 phase occlusion
遮挡 `peak neighborhood` 后，模型真实深度类别概率的下降最大，而遮挡 `loading early` 和 `release` 的影响较小。  
**解释**：当前网络已经学习到部分深度相关时间结构，但其时间注意力仍偏向峰值邻域，尚未充分利用机制分析提示的 loading/release 线索。

### 5.5 统一层级反演器的可解释性结果
为了保证最终主文模型与可解释性分析口径一致，本文进一步对统一层级反演器执行 latent probe、分支消融、hard-pair 与 phase occlusion 分析。结果显示：
- mean test `R^2`（hybrid latent）=`0.5512`
- mean test `R^2`（raw trunk）=`0.4226`
- mean test `R^2`（size-only）=`0.2985`

这说明统一架构的中间表征不仅保留了原始时空分支学到的深度相关结构，而且在融合结构化物理特征后，相关概念的可恢复性进一步增强。

进一步的分支消融表明：
- 去除 `raw` 分支后，depth hard-route bAcc 从 `0.5337` 降至 `0.4165`
- 去除 `shape` 分支后，depth hard-route bAcc 降至 `0.3811`
- 去除 `delta` 分支后，depth hard-route bAcc 降至 `0.4705`
- 去除 `tabular` 分支后，depth hard-route bAcc 降至 `0.3742`

**解释**：最终统一模型并非单纯依赖某一类强度特征，而是同时使用了原始幅值、形态、时序变化和结构化物理特征，其中 `shape` 与 `tabular` 分支对系统性能尤其关键。

在统一层级反演器的 hard-pair 分析中，`10` 对“大而深 / 小而浅”样本中，较深样本在 `80%` 的配对中获得更高的 `p_deep`。  
**解释**：最终模型已经具备比早期 Stage3 更强的困难样本分离能力，但由于这些样本对仍存在较大的 `raw_max` 差异（均值约 `50.55`），因此目前仍更适合写成“中等到较强支持”，而非“已完全解决混淆”。

统一模型的 phase occlusion 结果同样显示，遮挡 `peak neighborhood` 造成的真实类别概率下降最大（mean drop `0.0802`），而 `loading early` 和 `release` 的影响较弱。  
**解释**：最终模型已经实现更高的部署性能，但时间上仍主要依赖峰值邻域，这提示后续若要进一步提升深度物理一致性，应继续加强对 `loading/release` 阶段的建模。

## 6. 链路级结论
本文的关键结果并不只是“某个单头模型在 depth 上有多高”，而是：

1. 深度信息可以由原始输入神经网络学习得到；
2. 这种学习必须显式依赖 size-aware 路由；
3. 真实部署中的主要误差来自 size routing，而不是 depth trunk 本身；
4. 统一层级反演器通过 route-aware 优化，将整条 predicted-route depth 链路稳定提升到超过 XGBoost 基线的水平。

这意味着，本文最终系统的优势不在于某一个单点指标“全面最好”，而在于它第一次在真实部署约束下，把检测、大小和深度组织成了一条更完整、更鲁棒、也更可解释的术中推理链路。
