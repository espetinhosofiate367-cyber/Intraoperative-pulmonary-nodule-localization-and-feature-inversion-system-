# 主文图注草稿（TBME 版）

## Fig. 1 引言总图
微创肺结节手术中的临床痛点、现有定位方法局限、病态逆问题以及本文研究路径示意。该图强调：结节触觉定位首先是一个 detection problem，其次才扩展到 size inversion 和 coarse depth discrimination。

## Fig. 2 方法总图
本研究的方法框架，包括柔性触觉阵列、离体猪肺实验设计、窗口级数据处理以及 `Detection -> Size -> Depth` 层级模型组织。该图同时区分 scientific raw-input line 与 deployment enhancement line。

## Fig. 3 机制分析与 XGBoost 解释总图
结构化物理特征分析和 XGBoost 基线共同表明，柔性触觉数据中同时存在 detection、size 和 coarse depth 信息。深度相关判别主要依赖 spread extent、shape contrast、deformation position 和 temporal phase，而非单一峰值幅值。

## Fig. 4 Detection 结果图
raw-input 时空神经网络在结节探测任务上的 ROC、PR 或综合指标比较。该图用于证明，在主任务 detection 上，原始时空张量建模优于结构化 XGBoost 基线。

## Fig. 5 Raw-input size 结果图
不同模型在大小分类与回归上的比较。该图强调：原始时空张量本身已包含强大小信息，raw-input 模型在分类上接近或部分超过 XGBoost，但 standalone size regression 的最优结果仍由结构化基线保持。

## Fig. 6 Raw-input depth 主结果图
粗深度分类结果比较，包括 majority baseline、XGBoost、普通 raw-input depth 头和 size-routed raw-input depth experts。该图用于说明：深度不是不可学，而是必须在 size-aware 架构下建模。

## Fig. 7 Raw-input explainability 主图
raw-input size-routed depth 模型的 explainability 结果，包括 latent probe、hard-pair analysis 和 phase occlusion。该图用于证明神经网络内部自动编码了部分与大小和深度相关的物理结构，同时也显示其时间策略仍偏向 peak neighborhood。

## Fig. 8 Deployment enhancement 图
统一层级反演器在真实 predicted-route 条件下的结果比较。该图强调 unified model 的角色是系统增强和 route-aware robustness 提升，而不是再次证明 raw-input 自动学习能力。

## Fig. 9 实时界面图
检测优先、反演增强的术中触觉导航界面示意。界面持续显示检测概率，而大小与深度结果仅在高置信条件下输出，用于体现本文已具备实时系统原型形态。

## Fig. 10 延迟与部署口径图
不同路线在 model-only 与 end-to-end 口径下的延迟比较。该图用于说明：XGBoost 更适合作为结构化机制基线，而 raw-input 神经网络更符合 detection-first 在线部署需求；本文并不把“神经网络全面更快”作为主结论。
