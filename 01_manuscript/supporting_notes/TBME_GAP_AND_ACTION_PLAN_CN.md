# TBME 冲刊缺口与补强方案

## 1. 当前为什么可以冲 TBME
根据 TBME 官方 scope，期刊关注 biomedical engineering 中的 instrumentation、biosensors、biomechanics、signal processing、diagnostic systems 与 computational methods。本文当前稿件同时覆盖：
- 柔性触觉传感与术中定位场景
- 生物力学启发的结构化机制分析
- 原始时空神经网络与可解释性验证
- route-aware 系统组织与实时界面原型

这使稿件具备明显的 TBME 气质：不是单纯算法论文，也不是单纯器件论文，而是一个由 sensing、mechanics、modeling 和 system integration 共同构成的 biomedical engineering 方案。

官方参考：
- TBME aims and scope: https://www.embs.org/tbme/aims-and-scope/
- IEEE figure preparation: https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-the-text-of-your-article/preparing-figures-tables-and-biographies/
- TBME cover letter guidance: https://www.embs.org/tbme/information-for-authors/

## 2. 当前最有竞争力的亮点
1. 问题定义清楚：以术中肺结节探测为核心，大小反演与粗深度辨别作为逐层扩展。
2. 数据与协议清楚：42 个物理条件、3 次重复、窗口级标签协议明确。
3. 证据链完整：机制分析 -> XGBoost -> raw-input NN -> explainability -> deployment enhancement。
4. 工程落地清楚：不仅有离线结果，还有 GUI 原型、route-aware 推理与延迟 benchmark。

## 3. 目前最可能被 TBME 审稿人追问的点
1. 深度是否写得过满：必须坚持只写 `coarse depth discrimination`。
2. Hybrid 模型与 XGBoost 是否角色混淆：必须强调 raw scientific line 与 deployment line 的分工。
3. 临床转化边界：必须写清这是离体猪肺条件下的探索性研究，而非最终临床系统。
4. 延迟结论是否夸大：不能写“NN 全面更快”，只能写 detection end-to-end 更适合 raw-input 在线主线。

## 4. 必补的主文层强化项
### A. 图表
- 固定 Fig.1-Fig.10 主文图顺序
- 统一 caption 口径：每张图只回答一个问题
- 深度相关负面结论放到 caption 和 discussion 中写清边界

### B. 文稿
- 摘要中突出 sensing + mechanics + modeling + explainability + deployment
- 引言末尾明确四个科学问题
- Discussion 中把“为什么这是一项 exploratory biomedical engineering study”写透

### C. 结果表达
- 对 detection：强调主任务价值与 raw-input 优势
- 对 size：强调 strong effect，但不过度宣称全面超越 baseline
- 对 depth：强调 size-aware 才成立，且当前仅 coarse depth
- 对 deployment：强调 predicted-route robustness 和 system organization

## 5. 推荐投稿策略
### 主投
- IEEE Transactions on Biomedical Engineering (TBME)

### 备选
- Biomedical Signal Processing and Control
- Computer Methods and Programs in Biomedicine

## 6. 我们现在最该做的三件事
1. 把主文图 caption 全部写成 TBME 风格。
2. 把主文英文结构提纲定下来，避免中文稿继续发散。
3. 准备一版 <=250 words 的 cover letter，突出 biomedical engineering novelty 和 exploratory but rigorous positioning。
