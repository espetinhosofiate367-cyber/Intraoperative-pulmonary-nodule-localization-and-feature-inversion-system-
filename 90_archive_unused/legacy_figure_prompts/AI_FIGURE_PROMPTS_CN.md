# TBME 论文配图 AI 提示词包

本文件用于生成论文中的“概念图 / 方法图 / 系统图 / GUI 概念图”，不用于生成实验结果图。  
实验结果图仍以真实数据和脚本绘图为准。

## 总原则
- 风格：`IEEE TBME / biomedical engineering journal style`
- 画面：白色背景，简洁，矢量信息图风格，避免花哨渐变背景
- 配色：蓝绿灰为主，橙色仅用于强调病灶或关键路径
- 字体：仿照 `Arial/Helvetica/Cambria` 科研风格
- 画幅：优先横版，适合论文双栏或单栏排版
- 内容：强结构、少装饰、标注清楚、留有图注空间
- 严禁：科幻 UI、夸张 3D 发光、无意义装饰分子/粒子、过多阴影、医疗器械拟人化

## Prompt 1：系统总览图

### 用途
论文图1，展示从柔性阵列采集到层级神经网络再到术中可视化界面的全链路。

### 中文提示词
请绘制一张面向 IEEE TBME 论文的生物医学工程系统总览图，主题为“基于柔性阵列触觉传感与层级时空神经网络的术中肺结节定位与反演系统”。画面从左到右依次包括：离体肺组织或肺表面示意、贴附在肺表面的柔性 12x8 触觉阵列、按压过程中产生的伪彩应力热图序列、阶段一结节检测模块、阶段二大小反演模块、阶段三大小感知深度粗分类模块、以及右侧术中实时 GUI 界面。突出层级流程：Detection -> Size -> Depth。用整洁的箭头连接各模块。风格为简洁的医学工程矢量图，白色背景，蓝绿灰主色，橙色仅用于强调结节和关键结果。图中应预留英文标题和 A/B/C 模块标签空间，适合学术论文插图。

### English prompt
Create a clean IEEE TBME-style biomedical engineering system overview figure for a paper on an intraoperative lung nodule localization and inversion system based on flexible tactile sensor arrays and hierarchical spatiotemporal neural networks. Show a left-to-right pipeline: ex vivo lung surface, a conformal 12x8 flexible tactile array, pseudo-color tactile stress-map sequences during pressing, Stage 1 nodule detection, Stage 2 size inversion, Stage 3 size-aware coarse depth classification, and a real-time surgical GUI on the right. Emphasize the hierarchical flow Detection -> Size -> Depth. Use clean arrows, white background, restrained blue/teal/gray palette, with orange only for the lesion and key highlights. Scientific vector infographic style, publication-ready, readable labels, space for panel letters and caption.

## Prompt 2：最终方法流程图

### 用途
论文方法总流程图，强调训练和推理逻辑，而不是临床场景。

### 中文提示词
请绘制一张适用于 IEEE TBME 论文的方法流程图，主题为“层级式术中肺结节反演系统的训练与推理流程”。上半部分展示训练流程：机制分析与结构化 XGBoost 基线 -> Stage 1 检测器训练 -> Stage 2 大小路由器训练 -> 统一层级反演器训练。下半部分展示推理流程：输入 10 帧 12x8 时序触觉窗口 -> 结节检测概率 -> 若超过阈值则输出大小分类与回归 -> 基于大小路由到深度粗分类 -> 最后在 GUI 中显示概率、大小、深度。要求采用论文风格矢量图，模块框整齐，层级清楚，适合黑白打印和电子版显示。

### English prompt
Draw a publication-ready IEEE TBME-style method flowchart for a hierarchical intraoperative lung nodule inversion system. The training path should show: mechanism analysis and structured XGBoost baseline, Stage 1 detector training, Stage 2 size-router training, and unified hierarchical inverter training. The inference path should show: input 10-frame 12x8 tactile window, detection probability, threshold-gated size classification/regression, size-routed coarse depth classification, and final GUI display of probability, size, and depth. Clean vector flowchart, white background, organized boxes, readable scientific style, suitable for journal print.

## Prompt 3：深度与大小解耦的物理机制图

### 用途
解释为什么“大而深”和“小而浅”会混淆，以及动态按压如何帮助解耦。

### 中文提示词
请绘制一张医学工程论文风格的概念机制图，主题为“肺内结节大小与深度对表面触觉应力分布的不同影响”。左侧展示浅层小结节与深层大结节在某一时刻可能产生相似峰值应力，从而导致病态逆问题；右侧展示在动态按压过程中，两者在热点扩散范围、中心-边缘对比、形变位置和时间阶段响应上出现差异。图中需包含伪彩热图示意、简单剖面肺组织示意和时间箭头。整体风格学术、简洁、物理机制导向，不要夸张三维渲染。

### English prompt
Create a biomedical engineering concept figure explaining how nodule size and depth affect surface tactile stress distributions differently. On the left, show that a small shallow nodule and a large deep nodule may produce similar peak stress at a single time point, causing an ill-posed inverse problem. On the right, show that during dynamic pressing they differ in spread extent, center-to-border contrast, deformation position, and phase-dependent temporal response. Include pseudo-color tactile maps, a simplified lung cross-section, and time arrows. Publication-ready scientific infographic, clean and mechanism-driven.

## Prompt 4：XGBoost 解释性概念图

### 用途
解释为什么先做 XGBoost，再做神经网络。

### 中文提示词
请绘制一张适合 IEEE TBME 论文的解释性概念图，主题为“结构化物理特征基线如何为深度神经网络提供机制锚点”。左侧显示手工提取的触觉物理特征家族：幅值、扩散、形态对比、形变位置、时间阶段；中间显示 XGBoost 基线用于识别哪些特征与粗深度分类相关；右侧显示神经网络从原始时空触觉数据中重新学习这些信息，并通过层级结构实现实时部署。要求风格为学术信息图，白底，模块清晰，强调“XGBoost is not the final system but the interpretable bridge”.

### English prompt
Create an IEEE TBME-style explanatory concept figure showing how a structured tactile-physics XGBoost baseline serves as a mechanism anchor for a later neural network. On the left, illustrate engineered feature families: amplitude, spread, shape contrast, deformation position, and temporal phase. In the middle, show XGBoost identifying which feature families are relevant for coarse depth classification. On the right, show a neural network learning related information directly from raw spatiotemporal tactile input and integrating it into a hierarchical deployable system. Clean scientific infographic, white background, no decorative clutter.

## Prompt 5：统一层级反演器架构图

### 用途
画最终神经网络主架构。

### 中文提示词
请绘制一张论文方法图，主题为“统一层级正窗反演器（Hierarchical Positive Inverter）”。输入包括原始幅值序列、归一化形态序列、差分序列以及结构化物理特征；中间为共享编码模块（2D frame encoder + temporal modeling + phase pooling）；输出包括 size classification、size ordinal head、size regression、以及 routed coarse depth experts。强调 depth 头由 size 路由控制，并在训练中考虑 hard route、soft route、top2 route 和 route consistency。风格必须适合 TBME 投稿，结构严谨，尽量避免过多小字。

### English prompt
Draw a clean TBME-style neural network architecture diagram for a Hierarchical Positive Inverter. Inputs include raw amplitude tactile sequences, normalized shape sequences, delta sequences, and structured tactile physics features. The shared encoder contains 2D frame encoding, temporal modeling, and phase-aware pooling. Outputs include size classification, size ordinal prediction, size regression, and routed coarse-depth experts. Explicitly show that depth experts are conditioned or routed by size, and that training considers hard route, soft route, top-2 route, and route-consistency objectives. Publication-ready scientific vector diagram.

## Prompt 6：术中 GUI 概念图

### 用途
如果想单独做一个更美观的系统界面示意图。

### 中文提示词
请绘制一张医学工程论文风格的术中导航界面概念图，主题为“柔性触觉引导的肺结节实时定位界面”。界面包含：实时伪彩触觉热图、结节概率曲线、大小分布图、深度分布图、关键数值输出框。整体应像一个真实的科研原型界面，而不是商业广告海报。配色克制，白色和深青蓝为主，信息组织清晰，适合放在论文末尾或补充材料中。

### English prompt
Create a scientific GUI concept illustration for an intraoperative lung nodule localization interface guided by flexible tactile sensing. The interface should contain a real-time pseudo-color tactile heatmap, nodule probability trend, size distribution panel, depth distribution panel, and compact numeric output boxes. It should look like a credible biomedical engineering research prototype, not a commercial advertisement. Clean layout, restrained blue/teal palette, readable labels, publication-friendly.

## Prompt 7：图文摘要 / TOC Graphic

### 用途
投稿时如果需要图文摘要。

### 中文提示词
请绘制一张简洁的图文摘要，主题为“从柔性触觉应力图到肺结节检测、大小反演和深度粗分类”。左侧是一块贴附肺表面的柔性触觉阵列和伪彩热图，中间是层级网络流程 Detection -> Size -> Depth，右侧是概率、大小、深度的输出框。整体极简、学术、图标化，适合期刊图文摘要。

### English prompt
Create a compact graphical abstract showing the pathway from flexible tactile stress maps to lung nodule detection, size inversion, and coarse depth classification. Left: a conformal flexible tactile sensor array and pseudo-color tactile maps on a lung surface. Middle: a hierarchical pipeline Detection -> Size -> Depth. Right: compact outputs for probability, size, and depth. Minimal, academic, iconographic, suitable for a journal graphical abstract.

## 负面约束（可附加到所有提示词后）

### 中文
不要科幻风，不要霓虹光效，不要黑底海报风，不要人物特写，不要卡通，不要夸张 3D 玻璃质感，不要过量阴影，不要难以辨认的小字，不要无意义装饰元素。

### English
No sci-fi neon, no dark poster aesthetic, no cartoon style, no glossy glassmorphism, no dramatic cinematic lighting, no meaningless decorative particles, no tiny unreadable labels, no commercial advertisement style.
