# Introduction 配图 AI 提示词包

本文件专门用于生成论文 `Introduction` 部分的配图，目标不是展示实验结果，而是帮助读者在引言阶段快速理解：

- 临床痛点是什么
- 为什么术中触觉恢复很重要
- 现有方法为什么不够
- 为什么这个问题是病态逆问题
- 本文的研究路径是什么

建议风格统一为：
- `IEEE TBME / biomedical engineering journal style`
- 白色背景
- 简洁的医学工程矢量图风格
- 蓝绿灰为主，橙色只强调结节/风险点/关键路径
- 不要海报风，不要科幻感，不要黑底发光，不要商业广告感

---

## Prompt I-1：术中肺结节定位的临床痛点图

### 用途
作为引言第一张图，说明微创术中“看得见但摸不准”的核心矛盾。

### 中文提示词
请绘制一张适用于 IEEE TBME 论文引言部分的医学工程概念图，主题为“微创肺结节手术中的触觉缺失与定位困难”。左侧展示开放手术场景中医生可以直接触摸肺组织并定位结节；右侧展示胸腔镜或机器人辅助手术场景，医生仅能通过细长器械操作，缺乏直接触觉反馈。请在肺组织中标出一个小而深的结节，并强调在微创场景下该结节难以通过视觉直接确认。使用清晰的左右对照布局，白色背景，蓝绿灰为主色，橙色用于高亮结节和风险区域。风格必须为学术矢量信息图，适合 TBME 期刊引言配图。

### English prompt
Create a clean IEEE TBME-style biomedical engineering introductory figure illustrating the loss of tactile feedback in minimally invasive pulmonary nodule surgery. On the left, show open thoracic surgery where the surgeon can directly palpate the lung and localize a nodule. On the right, show thoracoscopic or robot-assisted surgery where the surgeon manipulates the lung with long rigid instruments and lacks direct tactile feedback. Include a small deep nodule inside the lung and emphasize that it is difficult to localize visually in the minimally invasive scenario. Use a left-right comparison layout, white background, restrained blue/teal/gray palette, with orange used only to highlight the lesion and risk area. Scientific vector infographic, publication-ready.

---

## Prompt I-2：现有术中定位方法及局限对比图

### 用途
说明为什么仅靠术前定位、超声和数字触诊都不够。

### 中文提示词
请绘制一张适用于 TBME 论文引言的对比信息图，主题为“现有术中肺结节定位方法及其局限”。图中横向排列四种策略：术前 CT 引导钩线/染料定位、术中超声、数字触诊、柔性触觉阵列方案。每种方法用简洁图标或小示意图表示，并在下方用短标签标注其主要问题，例如气胸/移位风险、空气干扰与经验依赖、主观性强、以及本文方案中的实时触觉恢复与定量优势。整体风格简洁、白底、学术矢量图，信息层级清楚，适合论文引言中的方法背景比较图。

### English prompt
Create an IEEE TBME-style comparison infographic showing existing intraoperative pulmonary nodule localization methods and their limitations. Arrange four strategies horizontally: preoperative CT-guided hook-wire or dye localization, intraoperative ultrasound, digital palpation, and the proposed flexible tactile-array approach. Represent each with a clean icon or mini schematic, and add short labels underneath highlighting the main limitation or advantage: pneumothorax or displacement risk, air interference and operator dependence, subjectivity, and real-time quantitative tactile restoration for the proposed system. White background, scientific vector style, clear information hierarchy, suitable for the Introduction section.

---

## Prompt I-3：病态逆问题概念图

### 用途
这是引言里最关键的一张，说明为什么深度和大小会耦合。

### 中文提示词
请绘制一张医学工程论文风格的概念机制图，主题为“肺内结节大小与深度反演中的病态逆问题”。左侧展示一个浅层但较小的结节，右侧展示一个深层但较大的结节。要求这两个结节在某一单帧按压时刻于肺表面产生幅值和形态相近的伪彩应力热点，从而造成单帧触觉图难以区分。图中需包含简化的肺组织剖面、埋藏结节、表面柔性触觉阵列和表面应力热图。请用箭头和文字简洁表达“single-frame ambiguity”或“ill-posed inverse problem”的概念。整体风格要学术、白底、矢量图风格，不要夸张 3D 渲染。

### English prompt
Create a clean biomedical engineering concept figure illustrating the ill-posed inverse problem in pulmonary nodule size-depth inversion. On the left, show a small shallow nodule; on the right, show a large deep nodule. Both should produce similar pseudo-color surface tactile stress hotspots at a single pressing instant, demonstrating that a single-frame tactile map may be ambiguous. Include simplified lung cross-sections, embedded nodules, a surface flexible tactile array, and surface stress maps. Use arrows and concise labels such as “single-frame ambiguity” and “ill-posed inverse problem.” White background, vector scientific style, no dramatic 3D rendering.

---

## Prompt I-4：动态按压帮助解耦大小与深度的概念图

### 用途
承接病态逆问题图，解释为什么时序是必要的。

### 中文提示词
请绘制一张 IEEE TBME 风格的概念图，主题为“动态按压时序如何帮助解耦结节大小与深度”。左侧展示单帧应力图无法区分浅小与深大的结节；右侧展示随时间推进的多帧伪彩应力图序列，强调两者在热点扩散范围、中心-边缘对比、时间滞后和按压阶段响应上的差异。图中请包含时间箭头、简洁的时序曲线小图以及浅层/深层结节剖面示意。要求画面干净、信息层级清晰、适合论文引言和方法过渡部分。

### English prompt
Create an IEEE TBME-style concept figure explaining how dynamic pressing helps disentangle nodule size and depth. On the left, show that a single-frame tactile map cannot distinguish a small shallow nodule from a large deep nodule. On the right, show a temporal sequence of pseudo-color tactile maps, emphasizing differences in spread extent, center-to-border contrast, temporal lag, and phase-dependent response. Include time arrows, small temporal response curves, and simplified shallow-versus-deep cross-sectional lung schematics. Clean publication-ready scientific infographic, suitable as a bridge from Introduction to Methods.

---

## Prompt I-5：本文研究路径图（引言末尾用）

### 用途
引言结尾，用于概括本文的整体研究路径，而不进入过细算法实现。

### 中文提示词
请绘制一张适用于医学工程论文引言结尾的研究路径图，主题为“本文的研究逻辑：机制分析 -> 结构化基线 -> 原始输入神经网络 -> 可解释性闭环”。图中从左到右依次展示：柔性阵列触觉数据采集、非深度学习机制分析、XGBoost 结构化基线、原始输入层级神经网络、以及解释性验证模块（latent probe / ablation / hard-pair / phase occlusion）。要求突出本文不是简单从传统方法切换到神经网络，而是先建立机制和结构化参照，再训练神经网络并反向验证其学习到的深度相关结构。风格为学术信息图，白底，蓝绿灰主色，橙色用于强调“depth”与“interpretability”。

### English prompt
Create a publication-ready IEEE TBME-style research roadmap figure for the end of the Introduction. Show the logic of the study from left to right: flexible tactile data acquisition, non-deep-learning mechanism analysis, structured XGBoost baseline, raw-input hierarchical neural networks, and interpretability validation modules such as latent probes, ablation, hard-pair analysis, and phase occlusion. Emphasize that the work does not simply replace a traditional model with a neural network; instead, it first establishes a mechanistic and structured reference, then trains neural networks on raw input, and finally validates what the network learned. White background, restrained blue/teal/gray palette, orange only for emphasis on depth and interpretability.

---

## Prompt I-6：引言图文摘要版（可放第一页或 Graphical Abstract 预览）

### 用途
如果你想在引言第一页放一个更浓缩的视觉总图。

### 中文提示词
请绘制一张简洁的生物医学工程图文摘要风格配图，主题为“柔性触觉阵列恢复术中肺结节定位中的缺失触觉”。画面从左到右包括：贴附肺表面的柔性触觉阵列、动态伪彩应力图、层级神经网络的 Detection -> Size -> Depth 流程，以及右侧输出的结节概率、大小和深度结果。图像要极简、结构清楚、论文风格强，不要加入复杂仪器细节或花哨背景，更适合作为引言配图而不是方法大图。

### English prompt
Create a compact graphical-introduction figure in a biomedical engineering journal style. The theme is restoring missing tactile perception for intraoperative pulmonary nodule localization using a flexible tactile array. From left to right, show a conformal tactile array on a lung surface, dynamic pseudo-color tactile maps, a hierarchical neural pipeline Detection -> Size -> Depth, and final outputs for nodule probability, size, and depth. Keep the image minimal, clean, structured, and publication-friendly, more suitable for the Introduction than for a detailed Methods figure.

---

## 通用负面约束

### 中文
不要黑底海报风，不要霓虹发光，不要科幻 HUD，不要卡通风，不要商业广告排版，不要夸张三维玻璃质感，不要复杂背景纹理，不要人物面部特写，不要无意义装饰元素，不要太多小字。

### English
No dark poster style, no neon sci-fi HUD, no cartoon aesthetics, no glossy 3D commercial rendering, no busy textured background, no human face close-ups, no meaningless decorative elements, and no excessive tiny labels.
