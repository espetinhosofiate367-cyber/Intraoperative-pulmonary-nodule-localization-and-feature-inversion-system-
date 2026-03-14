# 深度学习架构与数据图 Prompt（TBME版）

## 使用说明
这份文件专门服务当前论文里最关键的几类图：

1. 方法总图
2. raw scientific line 架构图
3. XGBoost 机制/解释总图
4. detection / size / depth / deployment 的数据对比图

统一风格要求：
- `IEEE TBME / biomedical engineering journal style`
- 白底
- 干净矢量信息图
- 蓝绿灰主色
- 橙色只高亮关键路径、结节、提升结果
- 参考 HMIL 的 grouped containers + rounded rectangles + concise labels
- 不要海报风、不要科幻 HUD、不要商业感排版

---

## Prompt N-1：方法总图（Fig.2）

### 目标
让读者一张图看懂：
- 触觉阵列
- 实验矩阵
- 数据处理
- 神经网络主线
- 部署增强线

### 中文 Prompt
请绘制一张适用于 IEEE TBME 论文 Methods 部分的总览方法图，主题为“基于柔性触觉感知与层级深度学习的术中肺结节定位流程”。采用从左到右的五模块布局，白色背景，医学工程矢量图风格，使用浅蓝、浅青、浅灰作为分组容器背景，橙色只用于高亮结节、关键箭头和最终输出。模块一显示 12×8 柔性触觉阵列贴附在肺表面并采集动态按压信号；模块二显示离体猪肺实验矩阵，标注 7 个 size、6 个 depth、3 次重复；模块三显示数据处理流程，包括 10-frame window、stride 2、raw amplitude、normalized shape、delta；模块四显示 scientific raw-input line：Stage I detection -> Stage II size v2 -> Stage III route-aware size-routed depth；模块五显示 deployment enhancement line：unified hybrid hierarchical inverter，融合 structured tactile features。请用圆角框、少量精确箭头、简短英文标签，整体像高质量 TMI/TBME 方法图，不要画成宣传海报。

### English Prompt
Create a publication-ready IEEE TBME-style methods overview figure for an intraoperative pulmonary nodule localization study based on flexible tactile sensing and hierarchical deep learning. Use a left-to-right five-module layout on a white background with clean biomedical vector graphics. Use pale blue, pale teal, and light gray grouped containers, and use orange only to highlight nodules, key arrows, and final outputs. Module 1 shows a 12×8 flexible tactile array conformally attached to the lung surface during dynamic pressing. Module 2 shows the ex vivo porcine lung experimental matrix with 7 size levels, 6 depth levels, and 3 repeats. Module 3 shows preprocessing: 10-frame windows, stride 2, raw amplitude, normalized shape, and delta streams. Module 4 shows the scientific raw-input line: Stage I detection -> Stage II size v2 -> Stage III route-aware size-routed depth. Module 5 shows the deployment enhancement line: a unified hybrid hierarchical inverter that fuses structured tactile features. Use rounded rectangles, minimal precise arrows, concise English labels, and a formal journal layout similar to high-quality TMI/TBME methods figures.

---

## Prompt N-2：Raw Scientific Line 架构图

### 目标
单独画出现在最重要的神经网络主线。

### 中文 Prompt
请绘制一张 IEEE TBME 风格的深度学习架构图，主题为“raw scientific line for tactile nodule localization”。图中重点展示三个阶段，但以统一结构语言呈现。左侧输入为 10×1×12×8 tactile tensor window，并拆成三路：raw amplitude、normalized shape、delta。Stage I 为双流 detection network，包含 frame encoder、MS-TCN、attention pooling 和 detection probability 输出。Stage II 为 raw size-only router v2，包含共享时序主干以及 size classification、ordinal size、expectation-based continuous size estimation、residual correction 四个组成部分。Stage III 为 raw route-aware size-routed depth model v2，包含深度 trunk 和 7 个 size-routed depth experts，并明确画出 GT route、hard predicted route、soft route、top2-soft route 四种训练路径。右侧输出为 nodule probability、size class、continuous size、coarse depth。使用分组容器、少量清晰箭头、模块间留白充足、英文标签简短。不要过度堆叠层数，不要画成神经网络教科书式细枝末节图，要更像 TBME 期刊里的结构总图。

### English Prompt
Create an IEEE TBME-style neural architecture figure focused on the scientific raw-input line for tactile pulmonary nodule localization. The left side begins with a 10×1×12×8 tactile tensor window, split into three streams: raw amplitude, normalized shape, and delta. Stage I is a dual-stream detection network with frame encoders, MS-TCN, attention pooling, and a detection probability output. Stage II is a raw size-only router v2 with a shared temporal trunk and four size-related components: size classification, ordinal size prediction, expectation-based continuous size estimation, and residual correction. Stage III is a raw route-aware size-routed depth model v2 with a shared depth trunk and seven size-routed depth experts. Explicitly visualize the four training paths: GT route, hard predicted route, soft route, and top2-soft route. The right side outputs nodule probability, size class, continuous size, and coarse depth. Use grouped containers, minimal clean arrows, strong whitespace, and concise English labels. Do not overdraw layer-level details; make it a journal-quality systems diagram rather than a textbook layer schematic.

---

## Prompt N-3：Unified Hybrid 部署增强图

### 目标
强调 unified model 的角色是 deployment enhancement，不是 scientific main model。

### 中文 Prompt
请绘制一张 IEEE TBME 风格的部署增强架构图，主题为“unified hierarchical inverter for deployment enhancement”。左侧输入包括 raw amplitude sequence、normalized shape sequence、delta sequence 和 structured tactile features。图中需要明确区分 raw branches 与 tabular feature branch，并在融合模块处展示 trunk feature、tabular feature 和 interaction fusion。随后画出 size classification、ordinal size、continuous size regression 和 routed coarse depth prediction 四个输出头。请在图中用简洁文字突出：this model is used for deployment robustness and predicted-route enhancement, rather than for proving raw-input automatic learning. 使用白底、浅色容器、橙色只高亮 fusion 和 predicted-route output，不要画成黑箱感很强的复杂网络图。

### English Prompt
Create an IEEE TBME-style deployment enhancement architecture figure for a unified hierarchical inverter. Inputs include raw amplitude sequences, normalized shape sequences, delta sequences, and structured tactile features. Clearly separate the raw-input branches from the tabular feature branch, and visualize the fusion stage as a combination of trunk features, tabular features, and feature interaction. Then show four outputs: size classification, ordinal size prediction, continuous size regression, and routed coarse depth prediction. Add a concise note that this model is used for deployment robustness and predicted-route enhancement rather than for proving raw-input automatic learning. White background, light grouped containers, orange only for fusion and predicted-route output emphasis, no cluttered black-box aesthetic.

---

## Prompt X-1：机制分析 + XGBoost 主图（Fig.3）

### 目标
让 XGBoost 图既像机制图，又像解释图。

### 中文 Prompt
请绘制一张适用于 IEEE TBME 主文的机制分析与 XGBoost 解释总图。整图采用 2×2 panel 布局。Panel A 为 feature-family contribution share，显示 amplitude、spread、shape、deformation-position、temporal-phase 等特征家族的相对贡献；Panel B 为 class-wise concept heatmap，展示 shallow、middle、deep 三类对这些概念特征的不同依赖模式；Panel C 为 representative tactile maps，对比浅层/中层/深层代表性伪彩应力图；Panel D 为 partial dependence 或 monotonic trend 图，展示一个关键特征变化时 shallow-middle-deep 预测倾向如何变化。整体应强调：depth depends more on spread, shape, deformation position, and phase than on peak amplitude alone。风格要克制、白底、统计图规整、子图标题短。

### English Prompt
Create a 2×2 IEEE TBME-style main figure for mechanism analysis and XGBoost interpretability. Panel A shows feature-family contribution share for amplitude, spread, shape, deformation-position, and temporal-phase families. Panel B shows a class-wise concept heatmap comparing shallow, middle, and deep dependence patterns. Panel C shows representative pseudo-color tactile maps for shallow, middle, and deep examples. Panel D shows a partial dependence or monotonic trend view illustrating how a key feature shifts shallow/middle/deep prediction tendency. The overall message should be explicit: depth depends more on spread, shape, deformation position, and temporal phase than on peak amplitude alone. White background, restrained statistical style, compact titles, and clean panel organization.

---

## Prompt D-1：Detection 对比数据图（Fig.4）

### 中文 Prompt
请绘制一张 IEEE TBME 风格的数据比较图，主题为“Detection performance comparison between XGBoost and raw-input detector”。推荐三联 panel：Panel A 为 ROC 曲线，Panel B 为 PR 曲线，Panel C 为 grouped bar chart 显示 AUC、AP、F1。数值需突出 raw-input detector 优于 XGBoost，尤其 AUC 0.8383 vs 0.8199、AP 0.5357 vs 0.5063。风格要求：白底、线条简洁、图例小而清楚、颜色统一，raw-input detector 用更醒目的蓝绿色，XGBoost 用灰蓝色。不要加入多余装饰。

### English Prompt
Create an IEEE TBME-style quantitative comparison figure for detection performance between XGBoost and the raw-input detector. Use a three-panel layout: Panel A ROC curves, Panel B PR curves, and Panel C grouped bars for AUC, AP, and F1. Highlight that the raw-input detector outperforms XGBoost, especially AUC 0.8383 vs 0.8199 and AP 0.5357 vs 0.5063. White background, clean lines, compact legends, unified palette, with the raw-input detector shown in a more prominent teal/blue-green and XGBoost in muted gray-blue.

---

## Prompt D-2：Size 对比数据图（Fig.5）

### 中文 Prompt
请绘制一张 IEEE TBME 风格的数据图，主题为“Size inversion comparison”。推荐 panel 布局：Panel A 为 Top-1 / Top-2 grouped bars，比较 XGBoost、raw size-only router v2、unified hierarchical inverter；Panel B 为 MAE 柱状图；Panel C 可选一个小型 parity plot inset 或 error distribution inset。图中必须清晰传达：raw size-only router v2 在 Top-1、Top-2 和 MAE 上均超过 XGBoost，而 unified model 主要承担部署增强角色。请把关键数字标注在柱子上，最优值用粗体或深色强调，整体风格规整、白底、期刊数据图风格。

### English Prompt
Create an IEEE TBME-style quantitative figure for size inversion comparison. Recommended layout: Panel A grouped bars for Top-1 and Top-2 comparing XGBoost, raw size-only router v2, and the unified hierarchical inverter; Panel B a bar chart for MAE; Panel C an optional small parity-plot inset or error-distribution inset. The figure must clearly communicate that the raw size-only router v2 surpasses XGBoost on Top-1, Top-2, and MAE, while the unified model mainly serves a deployment enhancement role. Annotate key values directly on the bars, use bold or darker emphasis for the best values, and keep the overall style clean, white-background, and journal-ready.

---

## Prompt D-3：Depth 主结果图（Fig.6）

### 中文 Prompt
请绘制一张 IEEE TBME 风格的数据图，主题为“Coarse depth results under hierarchical organization”。推荐 2×2 layout。Panel A 为 balanced accuracy comparison，比较 majority baseline、XGBoost、raw shared-head failure、raw size-routed depth v1、raw route-aware depth v2、unified hybrid model；Panel B 为 GT-route confusion matrix；Panel C 为 predicted-route hard confusion matrix；Panel D 为 route breakdown，显示 old raw predicted-route、raw route-aware predicted-route、unified hybrid predicted-route。图中结论必须非常清楚：depth is not impossible, but it becomes learnable only under size-aware and route-aware organization. 配色应把 scientific raw line 和 deployment line 区分开。

### English Prompt
Create an IEEE TBME-style quantitative figure for coarse depth results under hierarchical organization. Use a 2×2 layout. Panel A is a balanced-accuracy comparison across majority baseline, XGBoost, raw shared-head failure, raw size-routed depth v1, raw route-aware depth v2, and the unified hybrid model. Panel B is a GT-route confusion matrix. Panel C is a predicted-route hard confusion matrix. Panel D is a route-breakdown comparison showing old raw predicted-route, raw route-aware predicted-route, and unified hybrid predicted-route. The main message must be explicit: depth is not impossible, but it becomes learnable only under size-aware and route-aware organization. Use visually distinct colors for the scientific raw line versus the deployment line.

---

## Prompt D-4：Explainability 主图（Fig.7）

### 中文 Prompt
请绘制一张 IEEE TBME 风格的 explainability 总图，主题为“raw-input explainability summary”。推荐三联图。Panel A 为 latent probe family comparison，比较 probe 对 distribution complexity、temporal phase、deformation position 等特征家族的恢复能力；Panel B 为 hard-pair examples，展示几对 deep-vs-shallow 但 peak amplitude 相近的触觉案例；Panel C 为 phase occlusion summary，显示 loading early、peak neighborhood、release 不同阶段被遮挡后概率下降幅度。图中不要使用 t-SNE、UMAP、PCA。整体结论要聚焦：the network automatically encodes part of the physically meaningful structure, but still over-relies on peak neighborhood.

### English Prompt
Create an IEEE TBME-style explainability summary figure for the raw-input neural pipeline. Use a three-panel layout. Panel A compares latent probe recovery performance across concept families such as distribution complexity, temporal phase, and deformation position. Panel B shows hard-pair examples in which deep and shallow cases have similar peak amplitude but different broader structure. Panel C summarizes phase occlusion effects across loading early, peak neighborhood, and release phases. Do not include t-SNE, UMAP, or PCA. The central message should be: the network automatically encodes part of the physically meaningful structure, but still over-relies on the peak neighborhood.

---

## Prompt D-5：Deployment Enhancement 图（Fig.8）

### 中文 Prompt
请绘制一张 IEEE TBME 风格的数据比较图，主题为“Deployment enhancement under predicted-size routing”。推荐三联图。Panel A 为 grouped bars 比较 XGBoost、old raw predicted-route chain、raw route-aware v2、unified hierarchical inverter 在 predicted-route depth balanced accuracy 上的表现；Panel B 为 unified model 的 predicted-route confusion matrix；Panel C 为 route robustness breakdown，可展示 hard route、soft route、top2-soft route、temperature-soft route 的对比。图中要清楚表达：pure raw route-aware model has already crossed the XGBoost baseline, while the unified hybrid model further improves deployment robustness. 风格要克制、清楚、工程期刊感强。

### English Prompt
Create an IEEE TBME-style quantitative figure for deployment enhancement under predicted-size routing. Use a three-panel layout. Panel A shows grouped bars comparing XGBoost, the old raw predicted-route chain, raw route-aware v2, and the unified hierarchical inverter on predicted-route coarse-depth balanced accuracy. Panel B shows the predicted-route confusion matrix for the unified model. Panel C shows route robustness breakdown across hard route, soft route, top2-soft route, and temperature-soft route. The figure should make one message clear: the pure raw route-aware model already crosses the XGBoost baseline, while the unified hybrid model further improves deployment robustness. Clean, restrained, engineering-journal style.

---

## Prompt D-6：延迟与部署图（Fig.10）

### 中文 Prompt
请绘制一张 IEEE TBME 风格的延迟 benchmark 图，主题为“Latency under model-only and end-to-end settings”。推荐两联图。Panel A 为 detection-only latency，比较 XGBoost model-only CPU、XGBoost end-to-end CPU、NN model-only CPU、NN end-to-end CPU、NN end-to-end GPU；Panel B 为 full-chain latency，比较 XGBoost cascade end-to-end CPU、NN two-stage CPU、NN two-stage GPU。必须明确区分 model-only 与 end-to-end，不要写成神经网络全面更快。图注式的视觉表达应传达：explicit feature extraction makes the structured route less suitable for a detection-first online mainline.

### English Prompt
Create an IEEE TBME-style latency benchmark figure titled “Latency under model-only and end-to-end settings.” Use two panels. Panel A compares detection-only latency for XGBoost model-only CPU, XGBoost end-to-end CPU, NN model-only CPU, NN end-to-end CPU, and NN end-to-end GPU. Panel B compares full-chain latency for the XGBoost cascade end-to-end CPU, the NN two-stage CPU pipeline, and the NN two-stage GPU pipeline. Clearly distinguish model-only from end-to-end; do not imply that neural networks are universally faster. The visual takeaway should be that explicit feature extraction makes the structured route less suitable for a detection-first online mainline.

