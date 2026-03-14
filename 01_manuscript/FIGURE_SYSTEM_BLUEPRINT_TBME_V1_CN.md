# TBME 主图蓝图（正式版）

## 这份文件的定位
这份文件取代“先看现有丑图再修”的做法。  
从现在开始，主文图片一律按**论文施工图**来组织：

1. 先定义每张图要回答什么问题  
2. 再定义每张图的 panel 结构  
3. 再决定是手工重画、数据重绘，还是保留真实截图  

请把当前自动生成的旧图当成**数据源或版式参考**，不要直接把它们当最终投稿图。

---

## HMIL 风格里最值得借的 4 点
结合 `HMIL` 的 `Fig.1 / Fig.2 / Fig.4 / Fig.5`，我们真正应该借的是：

1. **比较关系先于装饰**
- 图先讲清楚“谁和谁在比较”
- 不是先追求视觉冲击

2. **容器式组织**
- 每个大模块放在一个浅色分组框里
- panel 内部留白充足

3. **数据图与概念图分开**
- 概念图负责讲系统与逻辑
- 数据图负责给证据

4. **嵌入图只做补充，不抢主结果**
- t-SNE/PCA 图适合说明表征空间
- 不适合代替主结果图

---

## 主文图总表

| 图号 | 图名 | 类型 | 作用 | 推荐处理方式 |
|---|---|---|---|---|
| Fig.1 | 引言总图 | 概念图 | 临床痛点 + 病态逆问题 + 研究路径 | 手工重画 |
| Fig.2 | 方法总图 | 概念图 | 传感器、实验、预处理、层级模型 | 手工重画 |
| Fig.3 | 机制分析 + XGBoost 解释 | 数据图 | 证明信息存在并指出关键物理特征 | 数据重绘 |
| Fig.4 | Detection 结果 | 数据图 | 证明主任务 detection 上 NN 优于 baseline | 数据重绘 |
| Fig.5 | Size 结果 | 数据图 | 证明大小是强主效应且 raw-input 已可学习 | 数据重绘 |
| Fig.6 | Raw-input depth 主结果 | 数据图 | 证明普通 depth head 失败、size-aware 成立 | 数据重绘 |
| Fig.7 | Raw-input explainability 主图 | 数据图+案例图 | 证明网络自动编码了部分物理结构 | 版式重组 + 真实数据子图 |
| Fig.8 | Deployment enhancement | 数据图 | 证明最终部署增强发生在 predicted-route 条件 | 数据重绘 |
| Fig.9 | GUI/系统截图 | 工程截图 | 证明系统原型已跑通 | 真实截图 + callout 美化 |
| Fig.10 | 延迟 benchmark | 数据图 | 说明 online path 的工程取舍 | 数据重绘 |

---

## Fig.1 引言总图

### 核心问题
为什么术中肺结节定位难，为什么触觉恢复值得做。

### 版式
- 横向四联图
- `A Clinical need`
- `B Current methods and limitations`
- `C Ill-posed inverse problem`
- `D Study roadmap`

### 必须出现的内容
- 开放手术与微创手术对照
- 小而深结节
- 术前定位 / 超声 / 数字触诊 / 柔性触觉阵列四种方案
- 浅小 vs 深大 单帧模糊
- `mechanism analysis -> XGBoost -> raw-input NN -> interpretability -> deployment enhancement`

### 图里必须写得很少
- 每个模块不超过 2 行字
- 只保留关键词，不写大段解释

### 画法建议
- 白底
- 浅灰圆角容器
- 蓝绿灰主色
- 橙色只标病灶和关键路径

---

## Fig.2 方法总图

### 核心问题
从原始触觉实验到最终推理链，系统到底怎么走。

### 版式
- 横向四联图或五联图
- `A Sensor`
- `B Experiment matrix`
- `C Preprocessing`
- `D Scientific main model`
- `E Deployment enhancement`

### 必须出现的内容
- `12 x 8` 柔性阵列
- `7 size x 6 depth x 3 repeats`
- `10-frame window, stride = 2`
- `Detection -> Size -> Depth`
- raw scientific line 和 deployment line 的角色区别

### 不要犯的错误
- 不要把所有网络层都画上去
- 不要画得像系统框图海报

### 建议
- 主路径用粗箭头
- 侧线用细箭头
- scientific line 和 deployment line 用不同浅色底框区分

---

## Fig.3 机制分析 + XGBoost 解释

### 核心问题
数据里到底有没有 detection / size / depth 信息，以及 depth 主要依赖什么。

### 推荐 panel
- `A` feature-family contribution share
- `B` class-wise concept heatmap
- `C` representative tactile maps
- `D` one or two partial-dependence plots

### 必须标出的结论
- size 是强主效应
- depth 是弱效应但可学
- depth 主要依赖 spread / shape / deformation，而不是单一 peak amplitude

### 画法
- 条形图和热图用统一色板
- tactile maps 统一色条、统一裁切、统一边框

---

## Fig.4 Detection 结果

### 核心问题
主任务 detection 上，raw-input detector 是否优于 structured baseline。

### 推荐 panel
- `A` ROC
- `B` PR
- `C` AUC/AP/F1 柱状对比

### 必须标出的数值
- `AUC 0.8383 vs 0.8199`
- `AP 0.5357 vs 0.5063`

### 画法
- 曲线线宽加粗
- 图例外置
- 柱图直接标数值
- best 值加粗

---

## Fig.5 Size 结果

### 核心问题
大小能否从 raw-input 学出来，以及不同模型之间的权衡是什么。

### 推荐 panel
- `A` Top-1 / Top-2 grouped bars
- `B` MAE bars
- `C` optional parity inset or calibration inset

### 必须标出的数值
- XGBoost: `Top-1 0.6701`, `Top-2 0.8023`, `MAE 0.1472`
- Raw size-only: `Top-1 0.6600`, `Top-2 0.8405`, `MAE 0.2907`

### 图中要传达的结论
- size 是强主效应
- raw-input classification 已接近 structured baseline
- standalone regression 最优仍由 structured baseline 保持

---

## Fig.6 Raw-input depth 主结果

### 核心问题
为什么 depth 不能用普通共享头，以及为什么必须 size-aware。

### 推荐 panel
- `A` balanced accuracy comparison
- `B` GT-route confusion matrix

### 必须比较的对象
- majority baseline
- XGBoost structured baseline
- raw shared-head failure
- raw size-routed success

### 必须标出的数值
- majority `0.3333`
- XGBoost `0.5138`
- raw size-routed GT-route `0.5238`

### 图中要传达的结论
- 不是 raw-input 不可学
- 而是错误结构会把 depth 淹没

---

## Fig.7 Raw-input explainability 主图

### 核心问题
raw-input scientific main model 是否自动编码了与大小和深度相关的物理结构。

### 推荐 panel
- `A` latent probe family comparison
- `B` hard-pair examples
- `C` phase occlusion summary

### 必须传达的三句话
- latent representation 超出 size identity
- 模型并非只看单点峰值
- 时间策略仍偏向 peak neighborhood

### 画法
- `A` 用干净的 bar 或 lollipop
- `B` 图片两列对照，标题压缩到极简
- `C` 用短条形图，数值直接标注

---

## Fig.8 Deployment enhancement

### 核心问题
最终系统为什么还需要 deployment enhancement，而不是停在 raw scientific model。

### 推荐 panel
- `A` predicted-route balanced accuracy comparison
- `B` predicted-route confusion matrix
- `C` optional route-robustness comparison

### 必须标出的数值
- raw predicted-route `0.4822`
- unified predicted-route `0.5337`
- XGBoost `0.5138`

### 图中要传达的结论
- 这张图讲的是系统级 predicted-route robustness
- 不是讲“纯 raw 自动学习”

---

## Fig.9 GUI / 系统截图

### 核心问题
系统是否真的能以工程原型形式跑起来。

### 画法
- 保留真实截图
- 外围加 4 个 callout：
  - tactile map
  - detection probability
  - size distribution
  - depth distribution

### 注意
- 不要大面积重画
- 这是可信度来源

---

## Fig.10 延迟 benchmark

### 核心问题
不同路线的延迟差异到底说明了什么。

### 推荐 panel
- `A` detection-only latency
- `B` full-chain latency

### 必须标出的结论
- model-only CPU：XGBoost 更快
- detection end-to-end：raw-input NN 更快
- full chain：不要写成 NN 全面更快

### 画法
- 分组条形图
- scope 分区明确
- 关键数值直接写在条上

---

## 补充材料图建议

### S1 嵌入空间图
- t-SNE / PCA 四联图
- 参考 HMIL Fig.5 的角色
- 只做 supplementary，不抢主结果

### S2 物理特征分布图
- `raw_max` / `spatial_entropy` / `hotspot_radius`
- box + strip 组合

### S3 size-depth 热图
- correctness / confidence landscape

### S4 `p_deep` 分布与 overlap 结构图
- violin + scatter

### S5 Probe + phase summary 小图
- 紧凑解释性补图

### S6 更多代表性 hard-pair 案例
- 只放 supplementary

---

## 最终执行顺序
1. 先手工重画 `Fig.1`、`Fig.2`
2. 再按真实数据重绘 `Fig.3-8, Fig.10`
3. `Fig.9` 保留真实截图，只做排版美化
4. t-SNE/PCA 等嵌入图放 Supplement
