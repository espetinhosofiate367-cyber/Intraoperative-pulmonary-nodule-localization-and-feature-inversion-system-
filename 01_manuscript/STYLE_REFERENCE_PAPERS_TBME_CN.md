# 可学习的论文图表风格参考（TBME 定稿向）

## 这份清单怎么用
这不是“推荐你引用所有这些论文”，而是推荐你 **学它们的图表组织方式**。

我只保留了对你当前论文最有帮助的几类：
- 引言概念图怎么画
- 方法总图和架构图怎么画
- 结果图和对比图怎么画
- explainability 图怎么画

---

## 一、最值得优先模仿的 6 篇

### 1. HMIL: Hierarchical Multi-Instance Learning for Fine-Grained Whole Slide Image Classification
- 链接（arXiv）：https://arxiv.org/abs/2411.07660
- 链接（IEEE Xplore）：https://ieeexplore.ieee.org/document/10810475/
- 本地参考 PDF：`C:\Users\SWH\Desktop\HMIL_Hierarchical_Multi-Instance_Learning_for_Fine-Grained_Whole_Slide_Image_Classification.pdf`

**最值得学什么**
- `Fig.1 / Fig.2` 这类 **分组清楚、容器清楚、箭头很少但主线很强** 的方法图组织方式。
- 一张图里既有总流程，也有关键模块局部展开，但不会乱。
- panel 字体、留白、线条克制，非常适合你现在的 `Fig.2` 和 `Fig.2B/2C`。

**适合你借鉴到哪里**
- `Fig.2` 方法总图
- `Fig.2B` raw scientific line 架构图
- `Fig.2C` unified deployment 架构图

**不建议照抄什么**
- 不要照抄 pathology 的对象画法。
- 学的是结构组织，不是医学对象本身。

---

### 2. Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning (HIPT)
- 链接（CVPR Open Access PDF）：https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Scaling_Vision_Transformers_to_Gigapixel_Images_via_Hierarchical_Self-Supervised_Learning_CVPR_2022_paper.pdf
- 链接（GitHub）：https://github.com/mahmoodlab/HIPT

**最值得学什么**
- 分层架构图如何画得“像系统工程”，而不是像网络层堆砌图。
- 多尺度/层级结构如何用少量模块表达清楚。
- 非常适合“输入 -> 表征 -> 聚合 -> 输出”这类流程的视觉组织。

**适合你借鉴到哪里**
- `Fig.2B` raw scientific line
- `Fig.2C` unified deployment enhancement

**不建议照抄什么**
- HIPT 偏视觉 transformer，别把你自己的 MS-TCN / route-aware 结构硬画成 transformer 风格。

---

### 3. Transformer-based biomarker prediction from colorectal cancer histology: A large-scale multicentric study
- 链接（PMC）：https://pmc.ncbi.nlm.nih.gov/articles/PMC10507381/

**最值得学什么**
- 这篇非常适合学 **方法图 + 主结果图**。
- `Figure 1`：把数据预处理、模型架构、队列概览放在同一张图里，但层级非常清楚。
- `Figure 2`：热图 + ROC/PR + 多队列对比放在一张主结果图里，信息很多但不乱。

来源依据：PMC 页面明确说明 `Figure 1` 包含 “pre-processing pipeline + model architecture + cohort overview”，`Figure 2` 包含 “AUROC heatmap + ROC + PRC”等结果组织。
链接同上。

**适合你借鉴到哪里**
- `Fig.2` 方法总图
- `Fig.3` 机制分析 + XGBoost 主图
- `Fig.4` detection 数据图
- `Fig.5` size 数据图
- `Fig.6` depth 主结果图

**不建议照抄什么**
- 这篇图的信息密度高，你如果全搬，会让你自己的图太挤。
- 学它的 panel 逻辑，不要学它的拥挤程度。

---

### 4. A Minimally Invasive Robotic Tissue Palpation Device
- 链接（PMC）：https://pmc.ncbi.nlm.nih.gov/articles/PMC11178256/
- 链接（IEEE Xplore）：https://ieeexplore.ieee.org/document/10412094/

**最值得学什么**
- 这是和你领域最接近、最应该看的工程系统论文之一。
- 适合学习“临床问题 -> 设备设计 -> 工作机制 -> 定量验证”这一整条图表叙事。
- PMC 正文明确写到：`Figs. 1A, 1B` 用来展示系统与测量思想，`Fig. 5` 强调关键设计创新点。

**适合你借鉴到哪里**
- `Fig.1` 引言概念图中的临床问题表达
- `Fig.2` 方法总图
- `Fig.9` GUI / 系统原型图

**不建议照抄什么**
- 它偏器械论文，图里会有更多硬件细节。
- 你这篇不能被画成纯硬件论文，要保留算法主线。

---

### 5. Intraoperative identification of pulmonary nodules during minimally invasive thoracic surgery: a narrative review
- 链接（PMC）：https://pmc.ncbi.nlm.nih.gov/articles/PMC9622445/
- PubMed：https://pubmed.ncbi.nlm.nih.gov/36330174/

**最值得学什么**
- 学它的“引言问题场景化”方式。
- 这篇不是让你学模型图，而是学它怎么把“为什么定位难、现有方法有哪些、各自缺点是什么”讲得一目了然。
- PMC 页面中 `Figure 1` 用一个具体成像/定位机制解释某一类技术路线，`Table 2` 则用临床试验总结方式把方法比较清楚展开。

**适合你借鉴到哪里**
- `Fig.1` 引言概念图
- 引言中的方法局限 panel
- 补充表格中“现有方法对比”排版

**不建议照抄什么**
- 这篇偏 review，图的“研究路线”成分不够。
- 你只能借它的临床问题表达和方法对比方式。

---

### 6. Towards automatic pulmonary nodule management in lung cancer screening with deep learning
- 链接（PMC）：https://pmc.ncbi.nlm.nih.gov/articles/PMC5395959/

**最值得学什么**
- 这篇适合学“肺结节任务的原始输入深度学习结果图怎么画”。
- 它用多视图、多尺度、原始输入，不依赖额外人工信息，这一点和你当前 raw-input scientific line 的叙事是相通的。
- PMC 页面显示它围绕 nodule type 和 workup 组织图与结果，风格比较朴素但很规范。

**适合你借鉴到哪里**
- `Fig.4` detection
- `Fig.5` size
- `Fig.6` depth

**不建议照抄什么**
- 这篇年代稍早，视觉上没有 HMIL 或近年的 TMI/TBME 那么精致。
- 更适合学“结果图标准化”，不适合学“概念图美感”。

---

## 二、explainability 最值得参考的 1 篇

### 7. Saliency-driven explainable deep learning in medical imaging: bridging visual explainability and statistical quantitative analysis
- 链接（PMC）：https://pmc.ncbi.nlm.nih.gov/articles/PMC11193223/

**最值得学什么**
- 这篇很适合学 explainability 图怎么“既有视觉图，又有量化图”。
- PMC 页面明确显示：
  - `Fig. 3` 是 explainability pipeline
  - `Fig. 4/5` 用 “top-panel metric + bottom-panel confusion matrix”
  - `Fig. 6/7` 用输入样本 + 多种解释图列阵对照
  - `Fig. 8` 以后开始做定量 explainability 曲线

**适合你借鉴到哪里**
- `Fig.7` explainability 主图
- Supplement explainability 细图

**不建议照抄什么**
- 这篇的 saliency map 网格很多，容易把图做得过满。
- 你只该学“视觉解释 + 定量总结”的组合逻辑，不该学满屏 heatmap。

---

## 三、如果只选 4 篇来照着画

### 最推荐组合
1. `HMIL`
- 学方法图和整体版式

2. `Transformer-based biomarker prediction ...`
- 学方法总图 + 结果主图

3. `A Minimally Invasive Robotic Tissue Palpation Device`
- 学工程系统图和临床设备表达

4. `Intraoperative identification of pulmonary nodules ...`
- 学引言临床痛点与现有方法比较

这 4 篇组合起来，基本就够支撑你整套主文图的风格决策了。

---

## 四、对应到你这篇论文，应该怎么学

### 你的 Fig.1
建议主要学：
- `Intraoperative identification of pulmonary nodules ...`
- `A Minimally Invasive Robotic Tissue Palpation Device`

因为你要的是：
- 临床问题能一眼看懂
- 现有方法局限能一眼对照
- 再自然过渡到你的触觉路线

### 你的 Fig.2 / Fig.2B / Fig.2C
建议主要学：
- `HMIL`
- `HIPT`
- `Transformer-based biomarker prediction ...`

因为你要的是：
- grouped containers
- 模块边界清楚
- 方法总图像系统工程图，而不是炫技网络图

### 你的 Fig.3–Fig.8
建议主要学：
- `Transformer-based biomarker prediction ...`
- `Towards automatic pulmonary nodule management ...`
- `Saliency-driven explainable deep learning ...`

因为你要的是：
- 数据图正规
- panel 有层级
- explainability 图不乱

---

## 五、最后一句最直接的建议

如果你现在就开始画：
- `Fig.1` 先照 **pulmonary nodule review + robotic palpation device** 的风格学
- `Fig.2` 先照 **HMIL + Transformer-based biomarker prediction** 的风格学
- `Fig.4–Fig.8` 先照 **Transformer-based biomarker prediction** 的结果组织学

这样最稳，也最像 `TBME` 会接受的图表气质。
