# 引言总图单文件 Prompt

## 推荐用途
用于生成论文 `Introduction` 的**一张总览大图**，把临床痛点、现有方法局限、病态逆问题和本文研究路径放在同一画面里。

## 建议参数
- 画幅：`16:9` 或 `1.8:1`
- 风格：`IEEE TBME / biomedical engineering journal style`
- 背景：纯白
- 配色：蓝绿灰为主，橙色只强调结节、风险点、关键路径
- 画风：干净矢量信息图、学术论文风格、可发表

## 中文直接复制版
请绘制一张适用于 IEEE TBME 论文引言部分的**大型总览概念图**，主题为“基于柔性阵列触觉传感与层级时空神经网络的术中肺结节定位与特征反演系统”。整张图采用**从左到右的四面板叙事结构**，在一张横向大图中同时解释临床问题、现有方法不足、病态逆问题和本文研究路径。整体要求：白色背景、干净的医学工程矢量图风格、蓝绿灰为主色、橙色仅用于高亮结节、风险点和关键路径、具有明确的信息层级、适合 TBME 期刊引言配图。

**Panel A: Clinical Challenge.** 展示开放手术与胸腔镜/机器人微创手术的对照。开放手术中医生可以直接用手触摸肺组织定位结节；微创手术中医生仅通过细长刚性器械操作，丢失直接触觉反馈。请在肺组织中标出一个“小而深”的结节，并用简洁文字强调“看不见、摸不准”的术中定位困难。

**Panel B: Current Methods and Limitations.** 横向并列展示四种策略：术前 CT 引导钩线/染料定位、术中超声、数字触诊、柔性触觉阵列方案。每个模块下方用极短标签标出关键局限或优势，例如“pneumothorax / displacement risk”“air interference / operator dependent”“subjective / non-quantitative”“dynamic tactile sensing / quantitative feedback”。要求图标化、简洁、信息清楚。

**Panel C: Ill-posed Inverse Problem.** 展示一个浅层但较小的结节和一个深层但较大的结节，在某一单帧按压时刻产生相似的表面伪彩应力热点，表达“single-frame ambiguity”和“ill-posed inverse problem”。包含简化肺组织剖面、埋藏结节、贴附表面的柔性触觉阵列，以及表面应力热图。再加入一个时间箭头和一小段动态按压序列，说明两者随后在热点扩散范围、中心-边缘对比、时间滞后和阶段响应上出现差异。

**Panel D: Our Research Roadmap.** 展示本文完整研究链条：柔性阵列触觉数据采集 -> 非深度学习机制分析 -> XGBoost 结构化基线 -> 原始输入层级神经网络 -> 可解释性验证。明确标出最终层级任务流为“Detection -> Size -> Depth”，右下角放置简洁输出图标，表示结节概率、大小和深度。突出本文不是简单从传统方法切换到神经网络，而是先建立机制和结构化解释，再训练原始输入网络，并通过可解释性分析反向验证网络学到的深度相关结构。

请为四个分区保留清晰的 `A / B / C / D` 标签和英文小标题位置。整图必须体现**一条完整叙事链**：`clinical need -> limitations -> inverse problem -> our solution`。不要做成海报风，不要黑底，不要霓虹，不要科幻 HUD，不要夸张 3D 玻璃质感，不要复杂背景纹理，不要人物面部特写，不要商业广告排版，不要堆满小字。

## English direct-copy version
Create a **large introductory overview figure** for an IEEE TBME paper, themed “An intraoperative pulmonary nodule localization and feature inversion system based on flexible tactile sensing and hierarchical spatiotemporal neural networks.” Use a **single wide landscape figure** with a **left-to-right four-panel narrative layout**. The figure should simultaneously explain the clinical need, the limitations of current approaches, the ill-posed inverse problem, and the study roadmap. Use a clean biomedical engineering vector infographic style on a white background, with restrained blue/teal/gray colors and orange only for lesions, risk points, and key pathways. The figure must be publication-ready and suitable for the Introduction section of a TBME-style paper.

**Panel A: Clinical Challenge.** Compare open surgery versus minimally invasive thoracoscopic or robot-assisted surgery. In open surgery, the surgeon can directly palpate the lung to localize a nodule; in minimally invasive surgery, the surgeon uses long rigid instruments and loses direct tactile feedback. Show a small deep pulmonary nodule inside the lung and clearly emphasize the challenge: visible anatomy but uncertain localization.

**Panel B: Current Methods and Limitations.** Show four horizontally aligned mini-modules: preoperative CT-guided hook-wire or dye localization, intraoperative ultrasound, digital palpation, and the flexible tactile-array solution. Under each module, add very short labels such as “pneumothorax / displacement risk,” “air interference / operator dependent,” “subjective / non-quantitative,” and “dynamic tactile sensing / quantitative feedback.” Keep the style icon-based, clean, and compact.

**Panel C: Ill-posed Inverse Problem.** Show that a small shallow nodule and a large deep nodule may generate similar pseudo-color surface tactile stress hotspots at a single pressing frame, illustrating “single-frame ambiguity” and an “ill-posed inverse problem.” Include simplified lung cross-sections, embedded nodules, a conformal tactile array on the lung surface, and tactile heatmaps. Then add a temporal arrow and a short dynamic pressing sequence to show that the two cases diverge in spread extent, center-to-border contrast, temporal lag, and phase-dependent response over time.

**Panel D: Our Research Roadmap.** Show the full logic of the study: flexible tactile data acquisition -> non-deep-learning mechanism analysis -> structured XGBoost baseline -> raw-input hierarchical neural networks -> interpretability validation. Clearly label the final hierarchical task flow as “Detection -> Size -> Depth,” and place compact output icons for nodule probability, size, and depth in the lower right. Emphasize that the study does not simply replace a traditional model with a neural network; instead, it first builds mechanistic and structured interpretability, then learns from raw tactile sequences, and finally validates what the model has learned.

Reserve clear space for `A / B / C / D` panel letters and short English titles. The figure should read as one coherent narrative: `clinical need -> limitations -> inverse problem -> our solution`. No dark poster style, no neon sci-fi HUD, no cartoon aesthetics, no glossy 3D commercial rendering, no busy background textures, no face close-ups, and no excessive tiny labels.
