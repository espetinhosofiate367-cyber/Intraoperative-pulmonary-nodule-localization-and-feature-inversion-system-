# TBME 主图蓝图（重置版）

## 这份文件的定位
这份版本用于正式替代前一轮“先试各种可视化再挑图”的做法。
从现在开始，论文图表只保留两类：

1. 直接支撑主文结论的主结果图
2. 对主文结论有明确补充价值的少量辅助图

当前明确不采用：
- t-SNE
- UMAP
- PCA 嵌入图
- 任何“看起来有趣但不能稳定支撑主论点”的探索性拼图

---

## 当前图表总原则
1. 每张图必须回答一个明确问题。
2. 每张图的结论必须能用一句话写进正文。
3. 不让图片承担超出其证据能力的论证任务。
4. 图的作用优先级必须服务于 `Detection -> Size -> Depth -> Deployment` 主线。
5. 概念图与数据图严格分开。

---

## HMIL 风格里真正值得借的 4 点
1. **结构先于装饰**
- 先把比较关系和信息层级讲清楚，再谈美观。

2. **分组容器清晰**
- 大模块放在浅色圆角容器中，主路径突出，辅路径收敛。

3. **主结果图克制**
- 主文只保留最有说服力的数据图，不展示“为了好看而画”的统计装饰。

4. **图文分工明确**
- 图负责给证据，正文和图注负责解释，不在图里塞过多文字。

---

## 主文图总表

| 图号 | 图名 | 类型 | 核心问题 | 推荐处理方式 |
|---|---|---|---|---|
| Fig.1 | 引言总图 | 概念图 | 为什么做术中触觉定位，现有方案为什么不够 | 手工重画 |
| Fig.2 | 方法总图 | 概念图 | 从实验、预处理到层级模型的完整链路是什么 | 手工重画 |
| Fig.3 | 机制分析 + XGBoost 解释 | 数据图 | 数据里到底有什么可学信息，depth 主要依赖什么 | 数据重绘 |
| Fig.4 | Detection 结果 | 数据图 | detection 主任务上 raw-input NN 是否优于 baseline | 数据重绘 |
| Fig.5 | Size 结果 | 数据图 | raw-input NN 是否学到了强大小信息，和 baseline 的权衡是什么 | 数据重绘 |
| Fig.6 | Raw-input depth 主结果 | 数据图 | 为什么普通 depth head 失败，size-aware 为什么成立 | 数据重绘 |
| Fig.7 | Raw-input explainability 主图 | 数据图 + 案例图 | 网络是否自动编码了部分物理结构 | 版式重组 + 真实数据子图 |
| Fig.8 | Deployment enhancement | 数据图 | 为什么最终系统还需要 predicted-route 部署增强 | 数据重绘 |
| Fig.9 | GUI / 系统原型图 | 工程截图 | 系统是否真的跑起来了 | 真实截图 + 版式美化 |
| Fig.10 | 延迟 benchmark | 数据图 | 各路线的工程代价与在线取舍是什么 | 数据重绘 |

---

## Fig.1 引言总图
### 核心问题
为什么术中肺结节定位难，为什么触觉恢复值得做。

### 版式
- 四联图：`Clinical challenge / Current methods / Ill-posed inverse problem / Study roadmap`
- 一页横向大图

### 必须出现的内容
- 开放手术与微创手术对照
- 小而深结节
- 术前定位 / 超声 / 数字触诊 / 柔性触觉阵列方案
- 浅小 vs 深大的单帧模糊
- `mechanism analysis -> XGBoost -> raw-input NN -> interpretability -> deployment enhancement`

### 图里不要出现
- 大段文字
- 复杂人物与背景
- 花哨 3D 渲染

---

## Fig.2 方法总图
### 核心问题
整个系统从数据采集到最终推理到底怎么走。

### 版式
- 四到五联图
- `Sensor / Experiment matrix / Preprocessing / Scientific main model / Deployment enhancement`

### 必须出现的内容
- `12 x 8` 柔性阵列
- `7 size x 6 depth x 3 repeats`
- `10-frame window, stride = 2`
- `Detection -> Size -> Depth`
- raw scientific line 和 deployment enhancement 的角色区别

### 画法建议
- 主路径粗箭头
- 模块用浅色圆角框
- 不画过细的网络层细节

---

## Fig.3 机制分析 + XGBoost 解释
### 核心问题
数据里到底有没有 detection / size / depth 信息，以及 depth 主要依赖什么。

### 推荐 panel
- `A` feature-family contribution share
- `B` class-wise concept heatmap
- `C` representative tactile maps
- `D` partial-dependence or monotonicity view

### 必须讲出的结论
- size 是强主效应
- depth 是弱效应但不是不可学
- depth 主要依赖 spread / shape / deformation，而不是单一 peak amplitude

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

---

## Fig.5 Size 结果
### 核心问题
大小是否能从 raw-input 学出来，以及不同模型之间的权衡是什么。

### 推荐 panel
- `A` Top-1 / Top-2 grouped bars
- `B` MAE bars
- `C` optional compact parity inset

### 图要传达的结论
- size 是强主效应
- raw-input classification 已接近 structured baseline
- standalone regression 最优仍由 structured baseline 保持

---

## Fig.6 Raw-input depth 主结果
### 核心问题
为什么 depth 不能靠普通共享 head 学出来，以及为什么必须 size-aware。

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

---

## Fig.7 Raw-input explainability 主图
### 核心问题
raw-input scientific main model 是否自动编码了与大小和深度相关的物理结构。

### 推荐 panel
- `A` latent probe family comparison
- `B` hard-pair examples
- `C` phase occlusion summary

### 当前明确不使用的 panel
- t-SNE / UMAP / PCA
- 全局 embedding 空间图

### 图要传达的结论
- latent representation 超出 size identity
- 模型并非只看单点峰值
- 时间策略仍偏向 peak neighborhood

---

## Fig.8 Deployment enhancement
### 核心问题
最终系统为什么还需要 deployment enhancement，而不是停在 raw scientific model。

### 推荐 panel
- `A` predicted-route balanced accuracy comparison
- `B` predicted-route confusion matrix
- `C` optional route-robustness comparison

### 图要传达的结论
- 这张图讲的是系统级 predicted-route robustness
- 不是讲“纯 raw 自动学习”

---

## Fig.9 GUI / 系统原型图
### 核心问题
系统是否真正以工程原型形式跑通。

### 画法
- 保留真实截图
- 外围加 4 个 callout：tactile map / detection probability / size distribution / depth distribution

---

## Fig.10 延迟 benchmark
### 核心问题
不同路线的延迟差异说明了什么。

### 推荐 panel
- `A` detection-only latency
- `B` full-chain latency

### 必须讲出的结论
- model-only CPU：XGBoost 更快
- detection end-to-end：raw-input NN 更适合在线主线
- full chain：不能写成 NN 全面更快

---

## 当前补充材料图建议
### S1 更多 XGBoost explainability 细图
- feature family + class-wise heatmap 详细版

### S2 更多 raw-input hard-pair 案例
- 仅保留真正有说服力的 4-6 对

### S3 更多 phase occlusion 细图
- loading / peak / release 的更细粒度拆分

### S4 hybrid branch ablation
- 作为 deployment enhancement 的补充证据

### S5 条件矩阵大图或代表性 tactile map gallery
- 仅在确实排版整洁、信息密度合理时考虑

### 当前不纳入补充材料的图
- t-SNE
- UMAP
- PCA
- 各类 embedding 对照图

---

## 最终执行顺序
1. 先手工重画 `Fig.1`、`Fig.2`
2. 再按真实数据重绘 `Fig.3-8, Fig.10`
3. `Fig.9` 保留真实截图，只做排版美化
4. 不再为 embedding 图投入主线精力
