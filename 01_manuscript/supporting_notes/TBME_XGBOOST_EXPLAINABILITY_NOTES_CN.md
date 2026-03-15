# 面向 TBME 的 XGBoost 可解释性图表设计说明

## 1. 目的
这份说明用于回答两个问题：

1. 为什么当前论文中的 `XGBoost + 结构化物理特征` 解释图要这样组织。
2. 如果目标期刊是 `IEEE Transactions on Biomedical Engineering (TBME)`，我们应当借鉴哪些更稳妥的图表和写作习惯。

本说明不是单纯的审美记录，而是服务于论文论证链：

- 机制分析先提示候选深度敏感因素；
- `XGBoost` 证明这些结构化因素足以支持 `depth coarse classification`；
- 神经网络再证明原始输入也能学习这些信息；
- 最终通过神经网络解释性与结构化基线相互印证，形成闭环。

## 2. TBME/IEEE 图表规范上的直接借鉴

### 2.1 来自 IEEE Author Center 的稳定原则
结合 IEEE Author Center 的图形规范，TBME 投稿图应尽量满足以下要求：

- 单图结构清晰，尽量避免不必要的装饰背景；
- 字号在缩版后仍可读，常用 `Arial/Helvetica/Times/Cambria` 等标准字体；
- 坐标轴、图例、色条、子图标题都应自解释；
- 多子图应尽量通过 `A/B/C/...` 明确阅读顺序；
- 图注应解释“比较对象”和“核心结论”，而不是重复图中每个数字。

对我们这个项目来说，这意味着：

- 触觉热图要统一色图与归一化策略；
- 结构化解释图要按“全局 -> 类别 -> 个例”组织；
- 不要把一张图堆成纯信息墙，要突出主结论。

## 3. 相关论文中值得借鉴的做法

### 3.1 触觉/包埋体检测类论文的借鉴点
来自触觉成像和包埋体识别相关工作，可以稳定借鉴以下表达方式：

- 保留代表性的原始/伪彩触觉图，而不是只给数值表；
- 对比不同深度或不同条件下的代表性形态差异；
- 在“物理现象图”和“模型解释图”之间建立对应关系；
- 把深度解释写成 `spread / shape / deformation / phase` 的组合，而不是简化成单一峰值衰减。

这和我们当前数据分析结果是一致的：

- `depth` 不主要依赖单一 `raw_max`；
- 更相关的是 `deformation_position`、`shape_contrast`、`spread_extent`；
- 时间阶段因素存在，但在当前 XGBoost 基线中相对次要。

### 3.2 XAI / SHAP 类论文的借鉴点
从 SHAP / additive feature attribution 相关工作可以借鉴的不是“复杂公式”，而是图表组织方式：

- 一张图先给全局特征重要性；
- 一张图给类别级别的差异模式；
- 一张图给少量具有代表性的局部样本解释；
- 若有必要，再用偏依赖图说明“特征值变化如何影响模型输出”。

对于我们的 `XGBoost`，这正好对应：

- `depth_family_share_tbme.png`：家族级全局贡献；
- `depth_classwise_concept_heatmap_tbme.png`：类别级概念模式；
- `depth_representative_cases_tbme.png`：代表性局部样本；
- `depth_partial_dependence_top4.png`：主要特征的效应方向。

## 4. 为什么当前 XGBoost 主图这样组织

### 4.1 当前主图结构
当前推荐作为论文主图的 XGBoost explainability 图是：

- [Fig_XGB_depth_explainability_tbme.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/outputs_xgboost_explainability_v1/Fig_XGB_depth_explainability_tbme.png)

这张图采用了四块结构：

1. `Feature-family contribution`
2. `Classwise concept heatmap`
3. `Representative tactile cases`
4. `Partial dependence`

### 4.2 这样画的理由
这是因为深度任务不是一个“单数字解释问题”，而是一个“从物理现象到模型判别”的多层问题。

如果只画一张 feature importance 条形图，会有两个问题：

- 不能说明 `shallow / middle / deep` 是否依赖不同特征组合；
- 不能把结构化解释和真实触觉形态对应起来。

而当前四块结构正好回答四个问题：

1. 模型整体最依赖哪些特征家族？
2. 不同深度类别的判别模式是否不同？
3. 在真实样本上，这些模式长什么样？
4. 当关键特征变化时，模型输出趋势如何变化？

这更符合 TBME 读者的阅读习惯：先看生物医学问题，再看模型，再看可解释性。

## 5. 当前结果支持什么结论

基于当前 `XGBoost` 解释结果，我们可以较稳妥地写出：

1. `depth` 并非主要由单点峰值幅值决定。
2. `deformation_position`、`shape_contrast` 和 `spread_extent` 是更主要的结构化线索。
3. 这与前面的非深度学习机制分析是一致的，因此 `XGBoost` 解释并不是孤立证据。
4. 后续神经网络若要比 `XGBoost` 更有说服力，就必须证明自己也编码并利用了这些结构，而不是只学到 `size` 或峰值强度。

## 6. 写进 TBME 稿件时的推荐表述

推荐在正文里写成：

> To provide an interpretable structured baseline, we trained XGBoost models on handcrafted tactile physics descriptors and analyzed the fitted model using additive feature-contribution decomposition. The depth baseline was driven primarily by deformation-position, shape-contrast, and spread-related feature families rather than by peak amplitude alone, which is consistent with the independent mechanism analysis.

中文可写成：

> 为了构建可解释的结构化基线，我们基于手工提取的触觉物理描述量训练了 XGBoost 模型，并使用加性特征贡献分解对其进行解释。结果表明，深度粗分类主要依赖形变位置、形态对比和扩散相关特征家族，而非单一峰值幅值。这一结果与独立的机制分析结论保持一致。

## 7. 目前建议保留的图表组合

如果按 TBME 稿件压缩图数，建议优先保留：

1. 机制分析主图：
   - [Fig_R12_mechanism_xgboost_summary.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R12_mechanism_xgboost_summary.png)
2. XGBoost 深度解释主图：
   - [Fig_XGB_depth_explainability_tbme.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/outputs_xgboost_explainability_v1/Fig_XGB_depth_explainability_tbme.png)
3. 统一层级反演器解释主图：
   - [Fig_R13_hierarchical_explainability_summary.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R13_hierarchical_explainability_summary.png)

## 8. 当前边界

仍需如实说明：

- `XGBoost` 解释的是模型使用了什么特征，不是直接证明组织力学因果律；
- 时间阶段特征在当前结构化基线中的贡献较弱，不代表 phase 信息不存在，只说明当前摘要方式可能仍偏弱；
- 因此，`XGBoost` 更适合作为结构化参照和机制支撑，而不是最终系统的唯一解释来源。

## 9. 参考来源

1. IEEE Author Center: figure and graphics preparation guidance  
   https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/preparing-figures-tables-and-biographies/

2. IEEE template and editorial guidance  
   https://template-selector.ieee.org/

3. Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.  
   https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf

4. Nguyen T, et al. Tactile Imaging of an Imbedded Palpable Structure for Breast Cancer Screening.  
   https://pmc.ncbi.nlm.nih.gov/articles/PMC4173743/

5. Bewley A, et al. Optical-Tactile Sensor for Lump Detection Using Pneumatic Control.  
   https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.672315/full
