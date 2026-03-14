# Fig.1 引言概念图 Gemini Prompt（定稿版）

## 这份文件的目标
这份 prompt 专门用于生成论文 `Fig.1`：

- 不是方法图
- 不是结果图
- 而是 **Introduction 概念总图**

它要在一张图里回答 4 个问题：
1. 为什么术中肺结节定位难
2. 为什么现有方法还不够
3. 为什么这个问题是病态逆问题
4. 我们这篇论文的研究路径是什么

---

## 建议生成参数
- 画幅：`16:9` 或 `1.8:1`
- 风格：`IEEE TBME / biomedical engineering journal figure`
- 背景：纯白
- 色彩：蓝绿灰为主，橙色只用于高亮结节、风险点、关键路径
- 画风：干净矢量信息图、期刊图、留白充分

---

## 通用前缀

每次给 Gemini 之前，先加这一段：

```text
Create a publication-ready figure for an IEEE TBME-style biomedical engineering paper. White background, clean vector-like infographic style, no poster aesthetics, no black background, no neon sci-fi HUD, no glossy commercial 3D rendering. Use grouped containers with rounded rectangles, balanced whitespace, concise English labels, consistent panel spacing, and restrained colors. Main palette: teal, blue-gray, light cyan, light gray; orange only for lesions, key arrows, and highlighted pathways. Make the figure look like a high-quality journal figure rather than a marketing graphic.
```

---

## 推荐主版本：四联总图

### 中文 Prompt
```text
请绘制一张适用于 IEEE TBME 论文引言部分的概念总图，主题为“基于触觉感知与深度学习的术中肺结节定位系统”。整张图采用横向四联 panel 布局，白色背景，学术矢量信息图风格，保留清晰的 A、B、C、D 分区和英文短标题。整体视觉必须克制、结构清楚、像高质量 TBME 或 TMI 论文中的引言主图，而不是海报。

Panel A: Clinical challenge. 左侧展示开放手术中医生可以直接触摸肺组织定位结节；右侧展示胸腔镜或机器人辅助手术中，医生只能使用细长刚性器械，缺乏直接触觉反馈。请在肺组织内部标出一个 small deep pulmonary nodule，并强调在 minimally invasive setting 下该结节 visually difficult to localize and not directly palpable。

Panel B: Current methods and limitations. 横向排列四个小模块：preoperative CT-guided hook-wire or dye localization、intraoperative ultrasound、digital palpation、flexible tactile-array sensing。每个模块下方只放极简短标签，例如 pneumothorax risk、air interference、subjective、quantitative tactile feedback。不要画太多文字，不要复杂图标。

Panel C: Ill-posed inverse problem. 用两个简化肺组织剖面做对照：一个 small shallow nodule，一个 large deep nodule。在 single pressing frame 下，这两个情形在表面柔性触觉阵列上产生相似的 pseudo-color tactile stress hotspot。请明确表现 single-frame ambiguity，并加一个小时间箭头和 3-4 帧小序列，说明 dynamic pressing can later reveal different spread, shape, and temporal response.

Panel D: Study roadmap. 用从左到右的简洁流程图展示：mechanism analysis -> XGBoost structured baseline -> raw-input hierarchical neural network -> interpretability validation -> deployment enhancement。右侧明确标出任务链 Detection -> Size -> Depth。这个 panel 的重点是说明本文不是简单从传统方法切换到神经网络，而是先建立机制与结构化参照，再让 raw-input neural network 学习，最后用解释性和部署增强收口。

请确保：
1. 所有标题和标签使用简短英文
2. 保持足够留白
3. 主箭头少而清楚
4. 不要复杂背景纹理
5. 不要人物面部细节
6. 不要做成宣传海报
```

### English Prompt
```text
Create a publication-ready introductory concept figure for an IEEE TBME paper on tactile sensing and deep learning for intraoperative pulmonary nodule localization. Use a wide four-panel horizontal layout on a white background with a clean biomedical vector infographic style. Keep clear panel letters A-D and short English panel titles. The overall visual tone must be restrained, structured, and journal-like, similar to a high-quality TBME or TMI introduction figure rather than a poster.

Panel A: Clinical challenge. Show open surgery, where the surgeon can directly palpate the lung, versus minimally invasive thoracoscopic or robot-assisted surgery, where long rigid instruments remove direct tactile feedback. Include a small deep pulmonary nodule inside the lung and emphasize that it is visually difficult to localize and not directly palpable in the minimally invasive setting.

Panel B: Current methods and limitations. Arrange four compact method modules horizontally: preoperative CT-guided hook-wire or dye localization, intraoperative ultrasound, digital palpation, and flexible tactile-array sensing. Under each module, use only very short labels such as pneumothorax risk, air interference, subjective, and quantitative tactile feedback. Avoid too much text or overly complex icons.

Panel C: Ill-posed inverse problem. Show two simplified lung cross-sections: one with a small shallow nodule and one with a large deep nodule. At a single pressing frame, both should generate similar pseudo-color tactile stress hotspots on the surface tactile array. Explicitly convey single-frame ambiguity, and add a small temporal arrow plus a short 3-4 frame sequence indicating that dynamic pressing later reveals different spread, shape, and temporal response.

Panel D: Study roadmap. Show a concise left-to-right flow: mechanism analysis -> XGBoost structured baseline -> raw-input hierarchical neural network -> interpretability validation -> deployment enhancement. Explicitly label the task chain Detection -> Size -> Depth. The emphasis of this panel is that the study does not simply replace a traditional model with a neural network; it first establishes mechanistic and structured reference points, then learns from raw tactile input, and finally closes the loop with interpretability and deployment enhancement.

Requirements:
1. Use short English labels only
2. Preserve generous whitespace
3. Use few but precise arrows
4. No textured background
5. No human facial details
6. Avoid a poster-like composition
```

---

## 备选版本：更强“临床问题”导向

### 适合什么时候用
如果你发现第一版太像“研究路线图”，而不够像引言图，就改用这一版。

### Gemini Prompt
```text
Create a publication-ready IEEE TBME-style introductory figure with stronger clinical emphasis for intraoperative pulmonary nodule localization. Use a four-panel layout on a white background.

Panel A should dominate visually and focus on the loss of tactile feedback in minimally invasive thoracic surgery.
Panel B should compare current localization options and their limitations.
Panel C should explain the ill-posed inverse problem using shallow-small versus deep-large nodules.
Panel D should show the proposed tactile sensing and deep learning solution as a compact study roadmap.

Make Panel A larger than the others so the figure reads first as a clinical problem figure and only second as a study-roadmap figure. Keep the style formal, minimal, and journal-ready.
```

---

## 负面约束

每次都建议附上这段：

```text
No dark poster style, no neon sci-fi HUD, no glossy commercial 3D look, no busy textured background, no cartoon style, no dramatic human faces, no excessive small text, no decorative medical stock-illustration aesthetic.
```

---

## 生成建议

### 第一轮
先让 Gemini 出：
- 2 张 `四联总图`
- 1 张 `临床问题更强版本`

### 第二轮挑图时看什么
优先保留：
1. 模块边界清楚
2. 病态逆问题能一眼看懂
3. 路线图不抢临床问题主角
4. 文字少但信息足

淘汰：
1. 太像海报
2. 太像商业宣传图
3. 太多器械细节
4. Panel 字体混乱

---

## 这张图在论文里的职责

`Fig.1` 最终只需要做到一件事：

**让读者在 10 秒内明白：为什么术中肺结节定位难、为什么需要触觉、为什么这个问题不是简单分类、以及本文解决问题的整体逻辑。**
