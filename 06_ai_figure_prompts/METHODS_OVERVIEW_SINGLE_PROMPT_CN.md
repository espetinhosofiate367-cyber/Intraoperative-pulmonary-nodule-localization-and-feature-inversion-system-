# 方法总图单文件 Prompt

## 推荐用途
用于生成论文 `Materials and Methods` 的**一张总览大图**，把传感器系统、实验准备、数据处理和模型架构放在同一画面里。

## 建议参数
- 画幅：`16:9` 或 `1.8:1`
- 风格：`IEEE TBME / biomedical engineering journal style`
- 背景：纯白
- 配色：蓝绿灰为主，橙色仅强调结节、伪彩触觉热点、关键路径
- 画风：干净矢量信息图、学术论文风格、可发表

## 中文直接复制版
请绘制一张适用于 IEEE TBME 论文 `Materials and Methods` 部分的**大型总览方法图**，主题为“柔性触觉阵列肺结节定位与特征反演系统的方法框架”。整张图采用**从左到右的四面板叙事结构**，在一张横向大图中同时解释传感器系统、实验准备、数据处理和模型架构。整体要求：白色背景、干净的医学工程矢量图风格、蓝绿灰为主色、橙色仅用于高亮结节、伪彩触觉热点和关键路径、具有明确的信息层级、适合 TBME 期刊方法配图。

**Panel A: Sensor System.** 展示柔性 12×8 触觉阵列贴附在肺表面，连接到信号采集模块，并在动态按压过程中输出二维伪彩应力图。要表现阵列通道布局、共形贴附特征和肺表面接触关系。

**Panel B: Experimental Preparation.** 展示离体肺组织实验平台、埋藏在不同深度的模拟结节、机械按压探头或手术器械按压动作，以及实验设计矩阵：7 个结节大小 × 6 个埋藏深度 × 3 次重复。可加入简化剖面示意，表现浅层和深层结节在肺内的位置关系。

**Panel C: Data Processing.** 展示原始逐帧 12×8 触觉矩阵流如何转化为模型输入，包括：raw amplitude 序列、normalized shape 序列、delta 序列、10 帧滑动窗口切片（stride 2）以及结构化物理特征提取。用小型伪彩热图序列和流程箭头表示数据整理过程。

**Panel D: Model Architecture.** 展示最终层级模型流程：Stage 1 结节检测 -> Stage 2 大小分类/回归 -> Stage 3 大小感知深度粗分类 / 统一层级反演器。输入包括 raw amplitude、normalized shape、delta 和 structured tactile features，共享编码后输出 size classification、size regression 和 routed coarse depth experts。明确标出“Detection -> Size -> Depth”的层级关系，并强调这是一个 hierarchical inference system，而不是平级多头。

请为四个分区保留清晰的 `A / B / C / D` 标签和英文小标题位置。整图必须体现一条完整方法链：`sensor system -> experiment design -> data processing -> hierarchical model`。不要做成海报风，不要黑底，不要霓虹，不要科幻 HUD，不要复杂背景纹理，不要夸张金属 3D 质感，不要人物面部特写，不要堆满小字。

## English direct-copy version
Create a **large Materials and Methods overview figure** for an IEEE TBME paper, themed “Methodological framework of a flexible tactile-array system for pulmonary nodule localization and feature inversion.” Use a **single wide landscape figure** with a **left-to-right four-panel narrative layout**. The figure should simultaneously explain the sensor system, experimental preparation, data processing, and model architecture. Use a clean biomedical engineering vector infographic style on a white background, with restrained blue/teal/gray colors and orange only for lesions, tactile hotspots, and key flow arrows. The figure must be publication-ready and suitable for the Methods section of a TBME-style paper.

**Panel A: Sensor System.** Show a flexible 12×8 tactile array conformally attached to the lung surface, connected to a data acquisition module, and producing two-dimensional pseudo-color tactile stress maps during dynamic pressing. Emphasize channel layout, conformal contact, and the lung-sensor interface.

**Panel B: Experimental Preparation.** Show the ex vivo lung experimental platform, embedded simulated nodules at different depths, a pressing probe or surgical instrument, and the experimental design matrix: 7 nodule sizes × 6 embedding depths × 3 repetitions. Include a simplified cross-sectional diagram indicating shallow and deep nodules inside the lung.

**Panel C: Data Processing.** Show how raw framewise 12×8 tactile matrices are converted into model inputs, including raw amplitude sequences, normalized shape sequences, delta sequences, sliding windows of 10 frames with stride 2, and structured tactile physics feature extraction. Use mini tactile heatmaps and clean process arrows.

**Panel D: Model Architecture.** Show the final hierarchical model flow: Stage 1 detection -> Stage 2 size classification/regression -> Stage 3 size-aware coarse depth classification / unified hierarchical inverter. Inputs include raw amplitude, normalized shape, delta, and structured tactile features. After shared encoding, the outputs include size classification, size regression, and routed coarse-depth experts. Clearly label the hierarchy as “Detection -> Size -> Depth,” and emphasize that this is a hierarchical inference system rather than a flat multi-head design.

Reserve clear space for `A / B / C / D` panel letters and short English titles. The figure should communicate one coherent methodological chain: `sensor system -> experiment design -> data processing -> hierarchical model`. No dark poster style, no neon sci-fi HUD, no glossy 3D commercial rendering, no busy textures, no face close-ups, and no excessive tiny labels.
