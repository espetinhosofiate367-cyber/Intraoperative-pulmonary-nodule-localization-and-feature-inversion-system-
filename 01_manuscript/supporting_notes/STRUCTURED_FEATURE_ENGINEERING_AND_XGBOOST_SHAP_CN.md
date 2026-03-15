# 结构化特征工程与 XGBoost + TreeSHAP 说明

## 1. 这一步在整篇论文里回答什么问题

在本文中，结构化特征工程不是为了和深度学习“抢主角”，而是为了先回答两个更基础的问题：

1. 原始触觉时空信号中是否真的包含与结节探测、大小和粗深度相关的可学习信息。
2. 如果这些信息存在，模型最先依赖的是哪些具有物理意义的线索。

因此，这一层的任务定位是：

- `XGBoost = structured baseline`
- `TreeSHAP / pred_contribs = interpretable attribution tool`

它们共同构成了后续 raw-input 神经网络之前的第一层解释性验证。

## 2. 结构化特征是怎么从原始触觉信号里构造出来的

### 2.0 关于“19 个特征”的口径说明

你记得的 `19` 个特征是对的，但它对应的是**早期核心物理特征口径**，不是当前最终输入 XGBoost 的全部列数。

目前项目里其实有三层不同的“特征数量”：

1. **19 个早期核心特征**
- 对应早期机制分析和物理解释最常用的紧凑版本；
- 可以理解为：
  - `15` 个单帧物理特征
  - 加上 `4` 个早期窗口时序特征
- 这套口径更适合讲“我们最初关注哪些物理量”。

2. **22 个当前基础描述量**
- 当前代码里的 `frame_physics_features` 给出 `15` 个单帧特征；
- `window_temporal_features` 给出 `7` 个窗口时序特征；
- 两者合起来是 `22` 个“基础描述量”。
- 这套口径更适合讲“当前物理特征体系本体包含哪些基本量”。

3. **226 个最终 XGBoost 输入列**
- 在真正训练 XGBoost 时，基础描述量会进一步展开为：
  - 窗口统计摘要
  - 代表性帧特征
  - 差分帧特征
  - 全局统计量
- 因此最终的结构化特征表不是 `19` 或 `22` 列，而是 `226` 列。
- 这套口径更适合讲“最终喂给 XGBoost 的完整结构化特征空间”。

所以，最稳妥的写法不是简单说“我们用了 19 个特征”，而是明确区分：

- **早期核心物理特征：19 个**
- **当前基础描述量：22 个**
- **最终 XGBoost 输入维度：226 个**

如果后面论文想压得更清楚，我建议主文写：

> 我们首先围绕约 19 个核心物理概念特征开展机制分析，随后在正式 XGBoost baseline 中将其扩展为窗口级结构化特征空间。

这样既保留你记得的旧主线，也和现在最终代码实现一致。

### 2.1 输入起点

每一帧原始输入为 `96` 个通道，重排后对应一个 `12 x 8` 的二维应力矩阵。正式样本单位不是单帧，而是长度为 `10`、步长为 `2` 的滑动窗口。

因此，特征工程并不是对单一静态热图做摘要，而是对一个短时动态按压窗口做多层描述。

### 2.2 单帧物理特征

首先，对窗口中的每一帧都提取一组物理描述量，主要来自：

- 幅值强度
- 扩散范围
- 形态对比
- 形变位置

具体包括：

- `raw_mean`, `raw_max`, `raw_sum`, `raw_p95`
- `center_mean`, `border_mean`, `center_border_contrast`
- `hotspot_area`, `hotspot_radius`, `second_moment_spread`, `spatial_entropy`
- `anisotropy_ratio`, `peak_count`
- `centroid_row`, `centroid_col`

其中：

- `center_border_contrast` 反映热点是否集中于中心区域；
- `hotspot_radius` 和 `second_moment_spread` 描述热点扩散尺度；
- `spatial_entropy` 反映分布复杂度；
- `anisotropy_ratio` 和 `peak_count` 对应形态不规则性；
- `centroid_row / centroid_col` 用于刻画主要受力中心的位置偏移。

### 2.3 窗口时序特征

在单帧特征之外，本文还对整个 `10` 帧窗口的时序演化进行摘要。核心时序量包括：

- `rise_time_to_peak`
- `peak_persistence_ratio`
- `decay_after_peak`
- `centroid_drift`
- `temporal_raw_sum_slope`
- `temporal_raw_max_slope`
- `window_raw_sum_gain`

这部分特征用于回答一个关键问题：同样出现局部高响应时，不同结节条件的响应是“快速冲高后衰减”，还是“维持更久、更稳定、更分散”。

### 2.4 从帧到窗口的统计摘要

为了避免只依赖某一时刻，本文对每个单帧特征在整个窗口内进一步做统计摘要：

- `mean`
- `std`
- `min`
- `max`
- `first`
- `center`
- `last`
- `delta`
- `slope`

例如，一个原始特征 `raw_max` 会被展开为：

- `raw_max_mean`
- `raw_max_std`
- `raw_max_first`
- `raw_max_last`
- `raw_max_delta`
- `raw_max_slope`

这种设计使得结构化模型既能看到“值有多大”，也能看到“值怎么变”。

### 2.5 代表性帧与差分帧特征

除了逐帧统计，本文还对四类代表性二维帧重新计算物理特征：

- `meanframe_raw`
- `maxframe_raw`
- `centerframe_raw`
- `meanframe_norm`

同时，对窗口内相邻帧差分的绝对值进行摘要，并构造：

- `delta_abs_mean`
- `delta_abs_std`
- `delta_abs_max`
- `deltaframe_*`

这一步的意义在于把“动态变化最明显时出现了什么空间结构”显式写进结构化特征空间。

### 2.6 最终的五类物理特征家族

为了后续解释性分析，所有结构化特征最终被归到五类物理概念家族：

1. `amplitude response`
2. `spread extent`
3. `shape contrast`
4. `deformation position`
5. `temporal phase`

这五类家族与本文的机制分析、XGBoost 解释以及神经网络 explainability 是同一套概念坐标系。

## 3. 为什么结构化 baseline 选 XGBoost

选择 `XGBoost` 的原因并不是“它流行”，而是它和当前问题的结构比较匹配：

1. 特征数量较多，且不同特征量纲不同；
2. 特征之间存在明显非线性与交互；
3. 样本规模适中，不适合一开始就只押注单一黑箱；
4. 树模型更容易和后续解释性分析对接。

在本文里，XGBoost 分别承担了三种结构化 baseline：

- detection：二分类
- size：七分类 + 连续回归
- coarse depth：三分类

其中：

- detection 在全部窗口上训练；
- size 和 depth 仅在真实阳性窗口上训练。

这使得 structured baseline 的作用非常明确：

- 先证明任务可学；
- 再看模型究竟在用什么物理线索。

## 4. 这里的 “SHAP” 具体指什么

本文实际使用的是 `XGBoost booster.predict(pred_contribs=True)` 得到的逐特征加性贡献分解。其原理与树模型的 `TreeSHAP` 一致，因此在论文写作中可以将其归入 `TreeSHAP-style additive attribution`。

这一层分析不是在问“因果律”，而是在问：

- 对当前训练好的结构化模型来说，
- 哪些特征在实际预测中贡献最大，
- 这些贡献是否能够整理成稳定的物理概念模式。

因此，这里的 SHAP/TreeSHAP 更适合被理解为：

- **模型依赖证据**
- 而不是**直接物理因果证明**

## 5. TreeSHAP 在本文中具体怎么用

在 explainability 脚本中，本文对 detection、size 和 depth 模型分别计算逐特征贡献，并进一步做三层聚合：

1. **feature level**
- 哪些具体特征贡献最大

2. **concept level**
- 例如 `centroid_row`、`hotspot_radius`、`center_border_contrast`

3. **family level**
- 即 `deformation_position`、`spread_extent`、`shape_contrast` 等更高层物理家族

对多分类深度任务，还进一步按 `shallow / middle / deep` 输出类别级贡献模式。

这也是为什么当前主图会被组织成：

- family contribution
- classwise concept heatmap
- representative tactile cases
- partial dependence

因为仅有单一 importance bar 并不足以支撑“深度依赖多因素耦合”这一结论。

## 6. 当前结果最能说明什么

根据当前 `xgboost_explainability_summary.json`，depth baseline 的家族级贡献占比大致为：

- `deformation_position`: `32.3%`
- `shape_contrast`: `27.9%`
- `spread_extent`: `24.9%`
- `amplitude_response`: `12.0%`
- `temporal_phase`: `2.5%`

这说明至少在当前结构化 baseline 中：

1. 深度不是主要由单点峰值幅值决定；
2. 形变位置、扩散范围和形态对比才是更重要的结构化线索；
3. 时间阶段信息是存在的，但在当前摘要方式下相对较弱。

具体到概念层，贡献靠前的深度特征包括：

- `centroid_row`
- `hotspot_radius`
- `anisotropy_ratio`
- `center_border_contrast`
- `centroid_col`
- `raw_max`
- `second_moment_spread`

这组结果对全文非常重要，因为它建立了一个后续神经网络解释必须回扣的参照系：

> 如果 raw-input 神经网络真的学到了深度相关结构，它就不应只围绕峰值强度转动，而应至少部分编码这些与位置、扩散和形态相关的线索。

## 7. 这一步和后面的神经网络 explainability 怎么衔接

本文的解释性工作分为两层：

### 第一层：结构化解释

- 手工特征
- XGBoost
- TreeSHAP

回答的是：

- 哪些物理概念首先让任务变得可学。

### 第二层：raw-input 神经网络解释

- latent probe
- Integrated Gradients
- hard-pair analysis
- phase occlusion

回答的是：

- 原始输入神经网络是否真的学到了与这些结构化概念一致的内部表示。

因此，前者不是后者的附录，而是后者的解释锚点。

## 8. 适合写进主稿的方法表述

可直接写成：

> 为了在进入 raw-input 深度学习之前先验证任务是否可学，本文首先从原始触觉时空窗口中构造结构化物理特征。特征提取分为三层：其一，对每帧 `12 × 8` 应力图提取幅值、扩散、形态和位置相关描述量；其二，对窗口内这些单帧特征的时间演化计算统计摘要和阶段特征；其三，对代表性帧与相邻帧差分重新提取二维物理特征。最终所有特征被归纳为 `amplitude response`、`spread extent`、`shape contrast`、`deformation position` 和 `temporal phase` 五类物理概念家族，并输入 XGBoost 构成结构化 baseline。随后，采用基于 `pred_contribs` 的树模型加性贡献分解，对 detection、size 和 coarse depth 模型进行解释，以识别不同任务最依赖的特征家族与概念。

## 9. 适合写进主稿的结果表述

可直接写成：

> XGBoost 结构化 baseline 不仅证明了 detection、size 和 coarse depth 在当前触觉数据上具备可学习性，也为后续 raw-input 神经网络提供了解释锚点。以深度任务为例，模型贡献主要集中在 `deformation_position`、`shape_contrast` 和 `spread_extent` 三类物理家族，而单纯的峰值幅值贡献相对有限。这表明深度效应更接近一个由位置、扩散和形态共同决定的耦合结构，而非简单强度衰减问题。

## 10. 相关代码与结果文件

- 特征工程与 XGBoost baseline：
  [train_xgboost_baselines.py](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/05_core_code/experiments/train_xgboost_baselines.py)
- 单帧/时序物理特征定义：
  `C:\Users\SWH\Desktop\GitHub_Docs_Package\Code_Archive\paper_flexible_sensor_lung_nodule_localization\experiments\depth_analysis_utils.py`
- XGBoost explainability：
  [explain_xgboost_baselines.py](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/05_core_code/experiments/explain_xgboost_baselines.py)
- explainability 汇总结果：
  [xgboost_explainability_summary.json](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/03_results_core/xgboost_explainability_summary.json)
