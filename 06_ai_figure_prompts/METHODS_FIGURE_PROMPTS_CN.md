# Materials and Methods 配图 AI 提示词包

本文件专门用于生成论文 `Materials and Methods` 部分的配图，目标不是展示数值结果，而是帮助读者快速理解：

- 传感器系统长什么样
- 实验准备和按压采集是如何进行的
- 原始数据如何整理成网络输入
- 最终模型架构是如何组织的

建议与引言配图保持同一视觉体系：
- `IEEE TBME / biomedical engineering journal style`
- 白色背景
- 简洁的医学工程矢量图风格
- 蓝绿灰为主，橙色仅强调结节、触觉热点、关键路径
- 适合论文图，不要海报风，不要花哨 3D

---

## Prompt M-0：方法总图（推荐优先生成）

### 用途
作为 `Materials and Methods` 的一张总图，把传感器系统、实验准备、数据处理和模型架构整合在同一张图中。

### 推荐版式
- 横向大图，建议 `16:9` 或 `1.8:1`
- 采用 `A-B-C-D` 四分区布局
- 从左到右阅读，形成清晰的方法链

### 中文提示词
请绘制一张适用于 IEEE TBME 论文 `Materials and Methods` 部分的**大型总览方法图**，主题为“柔性触觉阵列肺结节定位与特征反演系统的方法框架”。整张图采用从左到右的四分区叙事结构，白色背景，简洁医学工程矢量图风格，蓝绿灰为主色，橙色只用于高亮结节、伪彩触觉热点和关键流程。

第一区（A，Sensor System）：展示柔性触觉阵列、肺表面贴附状态、12×8 通道阵列布局、信号采集电路或数据读取模块。强调传感器是共形贴附在肺表面的柔性阵列，可在动态按压过程中输出二维应力分布。

第二区（B，Experimental Preparation）：展示离体肺组织实验平台、埋藏在不同深度的结节、机械按压探头或手术器械按压动作，以及 7 个大小 × 6 个深度 × 3 次重复的实验设计矩阵。画面中可加入简化剖面示意，显示浅层/深层结节与柔性阵列之间的空间关系。

第三区（C，Data Processing）：展示原始逐帧 12×8 触觉矩阵流如何变成时序窗口输入，包括：原始幅值序列、归一化形态序列、差分序列、滑动窗口（10 帧、stride 2）、以及从窗口中提取的结构化物理特征。建议用小型伪彩热图序列和箭头表示处理流程。

第四区（D，Model Architecture）：展示最终层级模型流程：Stage 1 结节检测 -> Stage 2 大小反演 -> Stage 3 大小感知深度粗分类 / 统一层级反演器。输入包括 raw amplitude、normalized shape、delta 和 structured tactile features；共享编码后输出 size classification、size regression 和 routed coarse depth experts。请明确标出“Detection -> Size -> Depth”的层级关系。

请在整张图中预留清晰的 A/B/C/D 标签和英文小标题空间。整图必须体现完整方法链：`sensor system -> experiment design -> data processing -> hierarchical model`。不要做成海报风，不要黑底，不要霓虹，不要复杂背景纹理，不要夸张金属 3D 质感，不要过多小字。

### English prompt
Create a **large Materials and Methods overview figure** for an IEEE TBME paper, themed “Methodological framework of a flexible tactile-array system for pulmonary nodule localization and feature inversion.” Use a left-to-right four-panel narrative layout on a white background in a clean biomedical engineering vector infographic style. Use restrained blue/teal/gray colors, with orange only for lesions, tactile hotspots, and key flow arrows.

Panel A (Sensor System): show a conformal flexible tactile array attached to the lung surface, a 12×8 channel layout, and a compact signal readout or acquisition module. Emphasize that the array captures two-dimensional tactile stress maps during dynamic pressing.

Panel B (Experimental Preparation): show the ex vivo lung platform, embedded nodules at different depths, a pressing probe or surgical instrument, and the experimental matrix with 7 sizes × 6 depths × 3 repetitions. Include a simplified cross-sectional schematic showing shallow and deep nodules relative to the tactile array.

Panel C (Data Processing): show how raw framewise 12×8 tactile matrices are converted into model input, including raw amplitude sequences, normalized shape sequences, delta sequences, sliding windows (10 frames, stride 2), and structured tactile physics features. Use mini pseudo-color tactile maps and arrows to depict the data pipeline.

Panel D (Model Architecture): show the final hierarchical model flow: Stage 1 detection -> Stage 2 size inversion -> Stage 3 size-aware coarse depth classification / unified hierarchical inverter. Inputs include raw amplitude, normalized shape, delta, and structured tactile features. After shared encoding, the outputs include size classification, size regression, and routed coarse-depth experts. Clearly label the hierarchy as “Detection -> Size -> Depth.”

Reserve clear space for A/B/C/D panel letters and short English subtitles. The figure should communicate one coherent methodological chain: `sensor system -> experiment design -> data processing -> hierarchical model`. No dark background, no neon glow, no glossy 3D commercial style, no busy textures, and no excessive tiny labels.

---

## Prompt M-1：传感器系统图

### 用途
单独展示柔性阵列传感系统和肺表面贴附方式。

### 中文提示词
请绘制一张适用于 IEEE TBME 论文方法部分的系统图，主题为“肺表面柔性触觉阵列传感系统”。展示柔性 12×8 触觉阵列贴附在肺表面、连接到信号采集模块、并输出二维伪彩应力图的过程。请同时表现阵列的共形贴附特征、通道布局和动态按压下的触觉热点。白色背景，学术矢量风格，适合论文方法图。

### English prompt
Create an IEEE TBME-style methods figure showing a flexible 12×8 tactile sensor array conformally attached to the lung surface, connected to a data acquisition module, and producing two-dimensional pseudo-color tactile stress maps. Highlight conformal attachment, channel layout, and tactile hotspots under dynamic pressing. White background, clean scientific vector style.

---

## Prompt M-2：实验准备与参数矩阵图

### 用途
展示离体肺模型、埋藏结节和实验设计矩阵。

### 中文提示词
请绘制一张适用于医学工程论文的方法图，主题为“离体肺实验平台与结节参数设计矩阵”。图中包含离体肺组织、埋藏在不同深度的模拟结节、按压探头，以及一个清晰的参数矩阵示意：7 个结节大小、6 个埋藏深度、每种条件 3 次重复。要求同时表现空间剖面关系和实验设计结构，风格简洁、规范、适合 TBME 论文。

### English prompt
Create a clean biomedical engineering methods figure showing the ex vivo lung experimental platform and the nodule parameter design matrix. Include ex vivo lung tissue, embedded simulated nodules at different depths, a pressing probe, and a clear experimental design matrix with 7 nodule sizes, 6 embedding depths, and 3 repetitions per condition. Show both the spatial cross-sectional relationship and the structured study design. White background, publication-ready vector style.

---

## Prompt M-3：数据处理流程图

### 用途
展示逐帧触觉数据如何整理成模型输入。

### 中文提示词
请绘制一张适用于 IEEE TBME 论文的方法流程图，主题为“触觉时序数据预处理与窗口化流程”。从左到右依次展示：逐帧 12×8 触觉矩阵流、伪彩热图重排、raw amplitude 序列、normalized shape 序列、delta 序列、滑动窗口切片（长度 10，步长 2）、以及结构化物理特征提取。请突出不同输入分支最终都送入层级模型。要求箭头清楚、模块整齐、风格学术。

### English prompt
Create an IEEE TBME-style data-processing flowchart showing how raw tactile time-series data are prepared for the model. From left to right, include framewise 12×8 tactile matrices, pseudo-color tactile map reshaping, raw amplitude sequences, normalized shape sequences, delta sequences, sliding-window segmentation (length 10, stride 2), and structured tactile physics feature extraction. Emphasize that these branches are integrated into the final hierarchical model. Clean vector flowchart, white background, publication-ready.

---

## Prompt M-4：层级模型架构图

### 用途
单独展示最终核心模型结构。

### 中文提示词
请绘制一张论文方法图，主题为“层级时空神经网络架构”。图中展示输入分支：raw amplitude、normalized shape、delta、structured tactile features；中间展示共享编码器，包括 frame encoding、temporal modeling 和 phase-aware pooling；输出端展示 Stage 1 detection、Stage 2 size classification/regression、以及基于大小路由的 Stage 3 coarse depth experts。强调这是一个层级推理系统，而不是平级多头。整体风格适合 TBME 投稿，结构清晰，标注简洁。

### English prompt
Draw a publication-ready TBME-style neural network architecture figure for a hierarchical spatiotemporal model. Show the input branches: raw amplitude, normalized shape, delta, and structured tactile features. The shared encoder should include frame encoding, temporal modeling, and phase-aware pooling. The output side should show Stage 1 detection, Stage 2 size classification/regression, and Stage 3 size-routed coarse-depth experts. Emphasize that this is a hierarchical inference system rather than a flat multi-head network. White background, clean vector scientific style.

---

## Prompt M-5：统一层级反演器训练目标图

### 用途
说明 unified hierarchical inverter 是怎么训练的。

### 中文提示词
请绘制一张适用于 IEEE TBME 论文的方法图，主题为“统一层级反演器的训练目标与路由策略”。展示 unified model 同时输出 size classification、size ordinal prediction、size regression 和 routed depth。请用清晰的分支图表示 hard route、soft route、top2 route 和 route consistency 约束，并标出这些训练目标共同优化真实 predicted-route depth 性能。风格要干净、结构严谨、适合论文方法图。

### English prompt
Create an IEEE TBME-style methods figure illustrating the training objectives and routing strategy of the unified hierarchical inverter. Show that the unified model jointly outputs size classification, size ordinal prediction, size regression, and routed depth. Use a clean branch diagram to represent hard route, soft route, top-2 route, and route-consistency constraints, and emphasize that these objectives are designed to optimize real predicted-route depth performance. Clean scientific vector style, white background.

---

## Prompt M-6：材料与方法图文摘要版

### 用途
如果想在方法部分开头放一张高度浓缩的总图。

### 中文提示词
请绘制一张简洁的医学工程图文摘要风格方法图，主题为“从柔性触觉阵列采集到层级深度反演的整体方法”。图中从左到右包括：柔性阵列与肺表面接触、动态按压触觉热图序列、窗口化与特征处理、Detection -> Size -> Depth 层级模型，以及输出的结节概率、大小和深度。整体要求极简、结构清楚、适合 Methods 开头而不是结果展示。

### English prompt
Create a compact graphical-methods summary figure in a biomedical engineering journal style. From left to right, show the flexible tactile array contacting the lung surface, dynamic tactile heatmap sequences, windowing and feature processing, the hierarchical model Detection -> Size -> Depth, and final outputs for nodule probability, size, and depth. Keep the figure minimal, clean, and structured, suitable for the beginning of the Methods section rather than a results figure.

---

## 通用负面约束

### 中文
不要黑底海报风，不要霓虹发光，不要科幻 HUD，不要卡通风，不要商业广告排版，不要夸张三维玻璃质感，不要复杂背景纹理，不要人物面部特写，不要无意义装饰元素，不要太多小字。

### English
No dark poster style, no neon sci-fi HUD, no cartoon aesthetics, no glossy 3D commercial rendering, no busy textured background, no human face close-ups, no meaningless decorative elements, and no excessive tiny labels.
