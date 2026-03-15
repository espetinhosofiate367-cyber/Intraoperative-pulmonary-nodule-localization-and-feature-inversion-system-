# Gemini 生图 Prompt（TBME 定稿版）

## 这份文件怎么用
这套 prompt 专门给 `Gemini` 用，目标不是“随便生成好看图片”，而是：

1. 概念图和架构图可以直接生成  
2. 数据图用来生成**期刊风格重绘版式**  
3. 所有图统一成 `TBME / TMI` 论文风格  

## 重要原则

### A. 可以直接让 Gemini 生的
- 引言总图
- 方法总图
- raw scientific line 架构图
- unified deployment 架构图
- GUI 排版美化概念图

### B. 不建议让 Gemini“凭空捏数字”的
- ROC / PR
- 柱状图
- confusion matrix
- latency 数值图

这些图更合理的用法是：
- 让 Gemini 生成 **重绘风格图 / 版式稿**
- 再把你自己的真实数值和图层替换进去

也就是说：
**Gemini 负责“像什么样”，真实数据负责“画什么数”。**

---

## 统一总前缀

下面所有 prompt 前面都建议先加这一段：

```text
Create a publication-ready figure for an IEEE TBME-style biomedical engineering paper. White background, clean vector-like infographic style, no poster aesthetics, no black background, no neon sci-fi HUD, no glossy commercial 3D rendering. Use grouped containers with rounded rectangles, balanced whitespace, concise English labels, consistent panel spacing, and restrained colors. Main palette: teal, blue-gray, light cyan, light gray; orange only for lesions, key arrows, and highlighted best results. Make the figure look like a high-quality journal figure rather than a marketing graphic.
```

---

## Fig.1 引言总图

### 用途
说明：
- 临床痛点
- 现有方法为什么不够
- 病态逆问题
- 本文研究路径

### Gemini Prompt
```text
Create a large IEEE TBME-style introductory overview figure for a paper on tactile sensing and deep learning for intraoperative pulmonary nodule localization. Use a wide four-panel layout from left to right with clean biomedical vector graphics and clear panel letters A-D.

Panel A: Clinical challenge. Compare open surgery versus minimally invasive thoracoscopic or robot-assisted surgery. In open surgery, the surgeon can directly palpate the lung. In minimally invasive surgery, long rigid instruments remove direct tactile feedback. Show a small deep pulmonary nodule inside the lung and emphasize uncertain localization.

Panel B: Current methods and limitations. Show four compact method modules: preoperative CT-guided hook-wire or dye localization, intraoperative ultrasound, digital palpation, and flexible tactile-array sensing. Under each, place short labels such as pneumothorax risk, air interference, subjective, and quantitative tactile feedback.

Panel C: Ill-posed inverse problem. Show a small shallow nodule and a large deep nodule producing similar single-frame surface tactile hotspots. Add a short temporal arrow and a small sequence to indicate that dynamic pressing later reveals different spread, shape, and temporal response.

Panel D: Study roadmap. Show the full logic: mechanism analysis, XGBoost structured baseline, raw-input hierarchical neural network, interpretability validation, and deployment enhancement. Show the task chain Detection -> Size -> Depth.

Keep labels short, preserve large whitespace, and make the whole figure read as one coherent narrative for a journal introduction figure.
```

---

## Fig.2 方法总图

### 用途
说明：
- 传感器
- 实验矩阵
- 数据处理
- 神经网络主线
- 部署增强线

### Gemini Prompt
```text
Create a publication-ready IEEE TBME-style methods overview figure for an intraoperative pulmonary nodule localization system based on flexible tactile sensing and hierarchical deep learning. Use five grouped modules arranged from left to right.

Module 1: A 12×8 flexible tactile array attached to the lung surface during dynamic pressing.
Module 2: Ex vivo porcine lung experimental matrix with 7 size levels, 6 depth levels, and 3 repeats.
Module 3: Data processing showing 10-frame windows, stride 2, raw amplitude, normalized shape, and delta streams.
Module 4: Scientific raw-input line, clearly labeled as Stage I raw detector, Stage II raw size-only router v2, and Stage III raw route-aware size-routed depth model v2.
Module 5: Deployment enhancement line using a unified hybrid hierarchical inverter with structured tactile feature fusion.

Use grouped rounded containers, minimal arrows, concise English labels, balanced spacing, and a clean journal methods-figure layout.
```

---

## Fig.2B Raw Scientific Line 架构重点图

### 用途
单独强调最重要的深度学习主线。

### Gemini Prompt
```text
Create an IEEE TBME-style neural architecture figure focused on the scientific raw-input line for tactile pulmonary nodule localization. Start from a 10×1×12×8 tactile tensor window and split it into three streams: raw amplitude, normalized shape, and delta.

Stage I: dual-stream detection network with frame encoders, MS-TCN, attention pooling, and detection probability output.
Stage II: raw size-only router v2 with a shared temporal trunk and four size-related components: size classification, ordinal size prediction, expectation-based continuous size estimation, and residual correction.
Stage III: raw route-aware size-routed depth model v2 with a depth trunk and seven size-routed depth experts. Explicitly show GT route, hard predicted route, soft route, and top2 route as separate training paths.

Outputs on the right: nodule probability, size class, continuous size, and coarse depth.
Avoid over-detailed layer diagrams. Use grouped containers and a concise system-level journal layout.
```

---

## Fig.2C Unified Deployment Enhancement 架构图

### 用途
强调 unified model 只是部署增强，不是 scientific main model。

### Gemini Prompt
```text
Create an IEEE TBME-style deployment architecture figure for a unified hierarchical inverter used for deployment enhancement. Inputs include raw amplitude sequences, normalized shape sequences, delta sequences, and structured tactile features. Clearly separate the raw branches and the tabular feature branch. Show fusion of trunk features, tabular features, and interaction features. Then show outputs for size classification, ordinal size prediction, continuous size regression, and routed coarse depth prediction.

Add a concise annotation stating that this model is used for predicted-route robustness and deployment enhancement rather than for proving raw-input automatic learning. White background, light grouped containers, orange only for fusion emphasis and final routed output.
```

---

## Fig.3 机制分析 + XGBoost 主图

### 用途
把“前面提取的各种特征”正式挂入主文。

### Gemini 用法
这张图更适合让 Gemini 生成 **重绘版式**，然后你再填入真实图和数值。

### Gemini Prompt
```text
Create a 2×2 IEEE TBME-style journal figure layout for mechanism analysis and XGBoost interpretability in tactile pulmonary nodule localization. This should be a precise scientific figure layout, not a decorative infographic.

Panel A: feature-family contribution share, with grouped bars for amplitude, spread, shape, deformation-position, and temporal-phase.
Panel B: class-wise concept heatmap comparing shallow, middle, and deep dependence patterns.
Panel C: representative tactile maps for shallow, middle, and deep examples, shown as pseudo-color stress maps.
Panel D: a compact partial-dependence or monotonic trend plot for one key feature.

The visual message must be that depth depends more on spread, shape, deformation position, and temporal phase than on peak amplitude alone. Use restrained typography, clean axes, compact legends, and formal journal-style panel spacing. Keep all numeric/chart elements visually editable and do not hallucinate extra labels.
```

---

## Fig.4 Detection 数据图

### 用途
Detection 主结果。

### Gemini Prompt
```text
Create a clean IEEE TBME-style quantitative figure layout for detection performance comparison between XGBoost and a raw-input detector. Use a three-panel arrangement:

Panel A: ROC curve panel
Panel B: PR curve panel
Panel C: grouped bar panel for AUC, AP, and F1

Use teal for the raw-input detector and muted gray-blue for XGBoost. Leave axes, legends, and bar labels clean and editable. The visual emphasis should make the raw-input detector clearly appear as the stronger method. White background, publication-quality scientific layout, no decorative elements.
```

### 你后续要填的真实数值
- AUC `0.8383 vs 0.8199`
- AP `0.5357 vs 0.5063`
- F1 `0.6216 vs 0.6185`

---

## Fig.5 Size 数据图

### 用途
展示 raw size-only v2 已全面超过 XGBoost。

### Gemini Prompt
```text
Create a publication-ready IEEE TBME-style figure layout for size inversion comparison. Use a three-panel arrangement:

Panel A: grouped bar chart for Top-1 and Top-2 across XGBoost, raw size-only router v2, and unified hierarchical inverter.
Panel B: bar chart for MAE in centimeters.
Panel C: compact inset reserved for a parity plot or absolute error distribution.

Make the raw size-only router v2 the visual focal method. The unified model should appear as a deployment-reference method, not as the scientific main model. Use bold visual emphasis for the best values, clean editable axes, and short journal-style titles.
```

### 你后续要填的真实数值
- XGBoost: `0.6701 / 0.8023 / 0.1472`
- raw size-only v2: `0.7177 / 0.8195 / 0.1242`
- unified: `0.6180 / 0.7859 / 0.1565`

---

## Fig.6 Depth 主结果图

### 用途
展示：
- shared head 为什么失败
- size-aware 为什么成立
- route-aware 为什么进一步成立

### Gemini Prompt
```text
Create an IEEE TBME-style quantitative figure layout for coarse depth results under hierarchical organization. Use a 2×2 panel layout.

Panel A: balanced-accuracy comparison across majority baseline, XGBoost structured baseline, raw shared-head failure, raw size-routed depth v1, raw route-aware depth v2, and unified hybrid model.
Panel B: GT-route confusion matrix.
Panel C: predicted-route confusion matrix.
Panel D: route-breakdown comparison showing old raw predicted-route, raw route-aware predicted-route, and unified hybrid predicted-route.

Use separate visual color families for the raw scientific line and the deployment line. Make the main takeaway explicit through layout hierarchy: depth is not impossible, but becomes learnable only under size-aware and route-aware organization. Keep the chart editable and journal-formal.
```

### 你后续要填的真实数值
- majority `0.3333`
- XGBoost `0.5138`
- raw size-routed v1 GT `0.5238`
- raw route-aware v2 GT `0.6066`
- old predicted-route `0.4822`
- raw route-aware v2 predicted `0.5240`
- unified predicted `0.5337`

---

## Fig.7 Explainability 主图

### 用途
展示：
- latent probe
- phase occlusion
- Integrated Gradients
- hard-pair

### Gemini Prompt
```text
Create an IEEE TBME-style explainability summary figure for the raw-input neural pipeline. Use a 2×2 panel layout.

Panel A: latent probe family comparison across concept families such as distribution complexity, temporal phase, and deformation position.
Panel B: phase occlusion summary across loading early, peak neighborhood, and release.
Panel C: class-wise mean Integrated Gradients maps for shallow, middle, and deep depth groups, shown as three compact pseudo-color attribution maps within the panel.
Panel D: hard-pair examples in which deep and shallow cases have similar peak amplitude but different broader spatial structure.

Do not include t-SNE, UMAP, or PCA. The visual message should be that the network automatically encodes part of the physically meaningful structure but still over-relies on the peak neighborhood. Keep the figure restrained, scientific, and panel-balanced.
```

### 你后续要填的真实数值
- raw scientific line latent probe `R^2 = 0.2712`
- size-only probe `R^2 = 0.2356`
- hard-pair success `0.7500`
- peak-neighborhood mean drop `0.2303`
- IG subset size `40 / 40 / 40`

---

## Fig.8 Deployment Enhancement 图

### 用途
展示：
- pure raw scientific line 已经超过 XGBoost
- unified 继续提升部署鲁棒性

### Gemini Prompt
```text
Create an IEEE TBME-style quantitative figure layout for deployment enhancement under predicted-size routing. Use three panels.

Panel A: grouped bars comparing XGBoost, the old raw predicted-route chain, raw route-aware v2, and the unified hierarchical inverter on predicted-route coarse-depth balanced accuracy.
Panel B: predicted-route confusion matrix for the unified model.
Panel C: route robustness breakdown including hard route, soft route, top2-soft route, and temperature-soft route.

The layout should first show that the pure raw route-aware model already crosses the XGBoost baseline, and then show that the unified hybrid model further improves deployment robustness. Use a clean engineering-journal style and keep all numeric layers editable.
```

### 你后续要填的真实数值
- XGBoost `0.5138`
- old raw predicted-route `0.4822`
- raw route-aware v2 `0.5240`
- unified `0.5337`

---

## Fig.9 GUI / 系统原型图

### 用途
真实截图的论文版式美化。

### Gemini Prompt
```text
Create a clean IEEE TBME-style figure layout for a real-time surgical tactile-navigation interface. Use a real screenshot as the central visual region and add four compact callouts around it with concise English labels: tactile map, detection probability, size distribution, and depth distribution. White background, light annotation boxes, minimal arrows, no futuristic HUD style, no heavy glow effects. The figure should look like an engineering prototype figure in a journal paper.
```

---

## Fig.10 延迟图

### 用途
强调：
- model-only 和 end-to-end 要分开
- structured route 的特征提取有工程负担

### Gemini Prompt
```text
Create an IEEE TBME-style latency benchmark figure layout titled “Latency under model-only and end-to-end settings.” Use two panels.

Panel A: detection-path latency comparing XGBoost model-only CPU, XGBoost end-to-end CPU, NN model-only CPU, NN end-to-end CPU, and NN end-to-end GPU.
Panel B: full-chain latency comparing XGBoost cascade end-to-end CPU, NN two-stage CPU, and NN two-stage GPU.

Clearly separate model-only from end-to-end measurement scope. Do not visually imply that neural networks are universally faster. The intended engineering takeaway is that explicit feature extraction makes the structured route less suitable as a detection-first online mainline.
```

### 你后续要填的真实数值
- detection model-only CPU: `0.5740 / 6.9379`
- detection end-to-end CPU: `27.3447 / 7.1143`
- detection end-to-end GPU: `2.8131`
- full chain end-to-end CPU: `9.4641 / 14.6890`
- full chain end-to-end GPU: `14.5371`

---

## 补充材料图 Prompt

### S1 特征分布补充图
```text
Create an IEEE TBME-style supplementary figure layout for tactile feature distributions. Arrange 4 to 6 compact violin or box plots comparing representative handcrafted tactile features across size or depth groups. Focus on spread, shape contrast, deformation position, and temporal phase related features. White background, journal-style axes, short panel titles, no decorative elements.
```

### S2 Routing breakdown 补充图
```text
Create an IEEE TBME-style supplementary figure layout showing route-aware breakdown for the raw scientific line. Include compact bars for GT route, hard route, soft route, top2-soft route, and temperature-soft route, plus a small side comparison of hard-route accuracy when size routing is correct versus incorrect. Clean, publication-ready, data-driven layout.
```

### S3 Explainability补充图
```text
Create an IEEE TBME-style supplementary figure layout for explainability details. Include additional hard-pair examples, finer-grained phase occlusion bars, and a compact summary of latent probe concept families. Avoid embedding maps. The figure should look like a real supplement figure for a biomedical engineering paper.
```

---

## 最快完稿建议

如果你想最快推进，建议 Gemini 先生成这 4 张：
1. `Fig.1` 引言总图
2. `Fig.2` 方法总图
3. `Fig.2B` raw scientific line 架构图
4. `Fig.2C` unified deployment 架构图

然后数据图部分：
- 用 Gemini 先出期刊版式
- 你再把真实数值、曲线、混淆矩阵替换进去

这会比让 Gemini 直接捏全部数据图更快也更稳。

