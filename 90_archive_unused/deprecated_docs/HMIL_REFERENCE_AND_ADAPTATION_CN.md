# HMIL 参考与迁移设计说明

## 1. 参考文献
- 论文：`HMIL: Hierarchical Multi-Instance Learning for Fine-Grained Whole Slide Image Classification`
- 本地文件：`C:\Users\SWH\Desktop\HMIL_Hierarchical_Multi-Instance_Learning_for_Fine-Grained_Whole_Slide_Image_Classification.pdf`

## 2. 这篇文章的核心思想
HMIL 的核心不是单纯“用一个更大的网络做分类”，而是把一个细粒度分类问题显式拆成有层级关系的两个分支：

1. 粗粒度分支
- 先学习上层类别。

2. 细粒度分支
- 再学习更难、更细的下层类别。

3. 层级对齐
- 在实例层和 bag 层同时约束粗分支与细分支之间的一致性。

4. 动态权重
- 训练过程中动态平衡粗粒度与细粒度分支的影响。

5. 对比学习增强
- 用监督对比学习提升细粒度判别能力。

## 3. 哪些思想适合迁移到本项目
本项目不是 WSI，也不是传统 MIL，但 HMIL 中有 4 类思想非常值得借鉴。

### 3.1 层级任务拆分
HMIL 先粗后细的思想和我们当前任务高度一致。

在本项目中，对应关系可以写成：
- `Stage 1`：结节存在检测
- `Stage 2`：结节大小反演
- `Stage 3`：在 size 条件下进行 coarse depth classification

这和 HMIL 中“coarse branch -> fine branch”的基本逻辑是相通的。

### 3.2 层级一致性而不是平级多头
HMIL 强调粗粒度与细粒度之间不能完全割裂，而要通过层级关系进行对齐。

对我们来说，这一点非常重要，因为：
- `size` 是强主效应
- `depth` 是弱主效应
- `depth` 明显具有 `size-dependent` 特性

因此，`depth` 不能被设计成与 `size` 平级、彼此独立的普通 head，而应当被视作：
- 在 `size` 条件下的进一步细化判断

也就是说，我们不应该采用：
- `shared feature -> size head`
- `shared feature -> depth head`

而应该采用：
- `shared feature -> size prediction`
- `size-routed / size-conditioned depth prediction`

### 3.3 动态权重
HMIL 中的动态权重策略对我们也有启发。

我们当前系统中，检测、大小、深度三者难度不同，且深度最弱。如果在联合训练时不给予合理约束，就容易出现：
- 检测很好
- 大小一般
- 深度退化到多数类

因此我们可以借鉴 HMIL 的思想，但做针对性改造：
- 训练早期更重视 `size`
- 训练后期逐步提高 `depth routed loss` 的权重
- 或对 `GT route / hard route / soft route / top2 route` 采用分阶段加权

### 3.4 细粒度分支的判别增强
HMIL 用监督对比学习来强化细粒度分支。

对我们而言，最值得借鉴的不是原封不动的 contrastive loss，而是其“专门强化细粒度困难边界”的思想。

在本项目里，最典型的困难边界是：
- `大而深` vs `小而浅`

因此后续若继续提升 depth，可以引入：
- hard-pair contrastive learning
- route-aware metric loss
- size-conditioned margin loss

## 4. 哪些部分不能照搬
HMIL 不能直接照搬，至少有 4 个原因。

### 4.1 数据结构不同
HMIL 的基本单位是：
- bag = whole slide
- instance = image patch

而我们的基本单位是：
- sequence window = 一个时序应力窗口
- frame = 窗口内部的时间帧

所以本项目不适合机械套用 WSI 的 MIL 聚合形式。

### 4.2 我们的层级关系不是类别树，而是任务依赖
HMIL 的层级来自病理标签树。

我们的层级不是：
- coarse pathology label
- fine pathology subtype

而是：
- 是否存在结节
- 结节多大
- 在该大小条件下结节多深

因此本项目更准确的表述不是 “hierarchical classification tree”，而是：
- `task hierarchy`
- `size-conditioned depth inference`

### 4.3 我们的困难点是路由误差传播
HMIL 重点处理的是 fine-grained category confusion。

我们的关键困难是：
- `size route` 错误会传导到 `depth`
- 最终影响真实部署链路

所以我们最终采用了：
- `GT route`
- `hard predicted route`
- `soft predicted route`
- `top2 route`
- `route consistency`

这比 HMIL 的 bag-instance 对齐更贴合本任务。

### 4.4 我们更强调可解释性闭环
HMIL 的解释性主要体现在层级建模带来的判别增强。

本项目需要额外回答：
- 深度到底影响了什么
- 网络是否真的学习到了这些深度相关结构

因此我们必须补充：
- 机制分析
- XGBoost 特征解释
- latent probe
- hard-pair analysis
- phase occlusion

这部分是 HMIL 之外，我们本项目特有的解释闭环。

## 5. 本项目可借鉴的 4 个明确点
如果在论文里需要简洁说明“借鉴了 HMIL 哪些思想”，建议固定写成这 4 点：

1. 层级任务拆分
- 将困难的细粒度问题拆解为更稳定的上游任务和更困难的下游任务。

2. 层级一致性约束
- 不把各输出头视为独立任务，而强调上游大小预测与下游深度预测之间的结构联系。

3. 分阶段/动态加权训练
- 根据任务难度与稳定性差异，动态平衡不同分支损失。

4. 细粒度判别增强
- 对最困难的深度边界进行额外强化，而不只依赖普通交叉熵。

## 6. 对本项目最终架构的直接启发
基于 HMIL，我们当前项目最合理的最终表达应是：

### 6.1 检测与反演分层
- `Stage 1`: 检测
- `Stage 2`: size
- `Stage 3`: size-routed depth

### 6.2 路由一致性训练
统一反演器中同时优化：
- `size cls`
- `size ord`
- `size reg`
- `depth_gt`
- `depth_hard`
- `depth_soft`
- `depth_top2`
- `route_consistency`

### 6.3 可解释性与层级结构结合
后续讨论中，应强调：
- 深度不是与大小平级的独立标签
- 深度是在大小条件下进行的更细粒度判别

这正是 HMIL 对本项目最有价值的启发。

## 7. 建议写入论文的方法表述
可以在方法部分这样写：

“受 HMIL 中层级建模思想的启发，我们没有将结节检测、大小预测和深度预测视为彼此独立的平级多任务，而是将其组织为逐级细化的层级推理过程：先完成结节存在检测，再进行大小反演，最后在大小条件下执行深度粗分类。与 HMIL 在病理 WSI 中采用的 bag-instance 层级对齐不同，我们在触觉时序任务中进一步将层级关系落实为 size-routed depth experts 及 route-consistency optimization，从而更直接地处理大小-深度耦合带来的误差传播问题。” 

## 8. 当前最稳妥的结论
HMIL 不能被直接照搬，但它给了我们非常清晰的结构启发：
- 先粗后细
- 上下游之间做一致性约束
- 对细粒度任务单独强化
- 不把困难子任务当成普通平级分类

这与我们最终形成的：
- `detector -> size router -> routed depth inverter`
- 以及统一层级反演器

在逻辑上是一致的。
