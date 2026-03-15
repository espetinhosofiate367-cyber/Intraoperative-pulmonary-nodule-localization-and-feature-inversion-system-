# 系统设计依据、科学问题、解决方案与验证方案

## 1. 为什么这个系统值得被设计出来
本文并不是先有一个算法，再去寻找应用场景；相反，系统设计是从临床手术中的真实困难出发逐步形成的。

### 1.1 临床定位困难是系统设计的直接起点
肺结节术中定位的困难，并不只是“结节小”或者“影像不清楚”，而是：

1. 术前 CT 虽然能够给出病灶位置，但在真实手术过程中，肺组织会因放气、萎陷、牵拉和操作而发生明显形态变化。
2. 因而，术者往往难以仅依据术前 CT 在术中精准对应结节位置。
3. 对于深部、小尺寸或表面不明显的结节，这种“术前影像明确、术中位置难寻”的落差尤其明显。

这一点和现有肺结节术中定位综述中的结论一致：微创条件下，术中识别肺结节仍然是一个持续存在的问题，现有方案往往依赖额外设备、术前标记或术中辅助成像。

参考文献：
1. Tang L, Zhang Y, Wang Y. Intraoperative identification of pulmonary nodules during minimally invasive thoracic surgery: a narrative review.  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9622445/
2. Azari F, Kennedy G, Singhal S. Intraoperative Detection and Assessment of Lung Nodules.  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10257859/

### 1.2 经验触诊说明“触觉仍然有临床价值”，但缺少标准化
在开放手术或经验丰富的术者操作中，局部硬度异常仍然是判断结节大致位置的重要线索。这说明：

1. 触觉并没有在临床上失去意义。
2. 触觉仍然是许多术者最自然、最符合手术直觉的定位方式。

但问题同样明显：

1. 这种经验性触诊高度依赖个体经验，难以成为统一的临床范式。
2. 触觉所感知到的“硬度异常”很难直接转化为结节的可量化信息。
3. 对于难以感知的病灶，临床实践往往不得不转向有创定位或成本较高的替代方案。

文献同样支持这一点：前人已经指出，许多术者依赖术中 palpation 和 feel，但这在 VATS/MIS 场景下明显受限，而且不同替代技术常伴随并发症、成本或工作流问题。

参考文献：
1. Azari F, Kennedy G, Singhal S. Intraoperative Detection and Assessment of Lung Nodules.  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10257859/
2. Tang L, Zhang Y, Wang Y. Intraoperative identification of pulmonary nodules during minimally invasive thoracic surgery: a narrative review.  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9622445/

### 1.3 为什么“触觉迁移”在物理上是成立的
从乳腺等浅表触诊迁移到肺，并不是因为器官本身相似，而是因为它们共享同一条更底层的物理机制：

1. 组织内部都可能存在局部机械异质性病灶。
2. 在有限按压或动态接触下，这种内部异质性会改变表面机械响应。
3. 这种改变不只表现为单点峰值，还会体现在扩散范围、空间形态和时间演化上。

这正是“柔性触觉阵列 + 动态按压 + 时空信号建模”能够成立的设计依据。

参考文献：
1. Tactile Imaging of an Imbedded Palpable Structure for Breast Cancer Screening.  
   PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC4173743/
2. Robot-assisted tactile sensing for minimally invasive tumor localization.  
   DOI: https://doi.org/10.1177/0278364909101136
3. Integration of force reflection with tactile sensing for minimally invasive robotics-assisted tumor localization.  
   DOI: https://doi.org/10.1109/TOH.2012.64
4. A resonant tactile stiffness sensor for lump localization in robot-assisted minimally invasive surgery.  
   DOI: https://doi.org/10.1177/0954411919856519

### 1.4 为什么不能直接照搬浅表触诊方案
迁移成立，不代表照搬成立。肺场景比乳腺等浅表场景更复杂，至少体现在：

1. 肺是含气、柔软、易塌陷、曲率变化明显的器官。
2. 术前 CT 与术中真实组织状态之间存在明显错位与失配。
3. 深度信息在肺中更弱，并且更容易和大小、扩散、形态、按压阶段耦合。

因此，系统不能只停留在“能不能摸到”，而必须进一步回答：

1. 能否先稳定判断当前位置是否存在结节；
2. 若能判断存在，是否还能定量反演结节大小；
3. 在更保守的条件下，是否还能获得粗深度信息。

---

## 2. 科学问题是如何逐层形成的

本文的科学问题不是一开始就预设成“同时反演 detection、size、depth 三个任务”，而是从临床和实验共同驱动下逐层形成的。

### 2.1 第一层科学问题：结节是否存在
这是最先出现的问题，因为：

1. 它最符合临床第一需求；
2. 它最接近前人工作已经证明可行的 detection/localization 起点；
3. 只有 detection 成立，后续大小和深度问题才有意义。

### 2.2 第二层科学问题：大小能否被反演
当实验设计中 7 个结节大小条件在触觉分布上呈现稳定差异时，size 才自然成为第二层问题。也就是说，size 不是平行追加的任务，而是 detection 之后进一步提出的问题。

### 2.3 第三层科学问题：深度是否具备可辨识性
深度变量虽然在实验设计中客观存在，但从机制上看，它比大小更弱、更耦合，因此不适合一开始就作为与 detection、size 对称的目标。本文最终只把它谨慎地定义为 `coarse depth discrimination`。

---

## 3. 解决方案为什么要设计成现在这样

### 3.1 先做结构化特征和 XGBoost baseline
这样设计的依据是：

1. 在新型触觉场景中，首先要证明“任务本身可学”。
2. 结构化特征可以直接回扣物理机制。
3. XGBoost + SHAP 可以先回答：
   - 哪些任务可学；
   - 哪些特征最关键；
   - depth 是否真的弱于 size。

因此，XGBoost 在本文里不是旧方法，而是证据链的第一环。

### 3.2 再做 raw-input 神经网络
只有在结构化基线已经证明任务可学之后，raw-input 神经网络才有明确意义。它要回答的问题不是“神经网络能不能用”，而是：

1. 不显式输入手工特征时，原始时空触觉张量本身能否支持这些任务；
2. 是否能够达到甚至超过结构化 baseline；
3. 网络内部是否会自动编码与重要物理特征一致的结构。

### 3.3 为什么任务要做成层级式
这是整套系统设计中最重要的一点：

1. detection 是第一层；
2. size 是 detection 之后的第二层；
3. depth 不是独立头，而是 size-aware / route-aware 问题。

这一组织不是为了“模型好看”，而是由问题本身决定的。

### 3.4 为什么最后还要有系统级整合方案
当 raw scientific line 已经证明任务成立以后，才有必要进入系统级问题：

1. 如何把 detection、size、depth 串成真实术中工作流；
2. 如何处理 predicted-route 条件下的误差传播；
3. 如何把结果呈现为实验人员和临床术者能够使用的界面。

因此，最终的 unified / deployment enhancement model 不是“科学问题的唯一主角”，而是系统级方案。

---

## 4. 验证方案为什么这样设计

### 4.1 数据验证
使用真实猪肺组织 + 仿体结节，而不是只用纯仿体，有两个目的：

1. 尽可能保留肺组织的真实力学环境；
2. 又能控制大小和深度两个变量，形成统一实验矩阵。

### 4.2 重复实验
每个条件做 3 次重复，不是为了凑样本，而是为了验证：

1. 同一物理条件下触觉时空响应是否可重复；
2. 模型能否在重复间保持稳定；
3. 任务可学性是否依赖偶然实验波动。

### 4.3 分层验证路径
验证不是一次完成，而是分四层推进：

1. **结构化验证**：XGBoost + SHAP，先确认 detection、size、coarse depth 是否可学；
2. **raw-input 验证**：神经网络直接学习原始张量，看能否达到甚至超过 baseline；
3. **解释性验证**：用 probe、Integrated Gradients、hard-pair、phase occlusion 检查网络学到了什么；
4. **系统级验证**：把模型放入统一工作流和 GUI，验证 deployment robustness 与实时可用性。

---

## 5. 可以直接写进论文的一段总括

> 本研究的系统设计并非先有算法、再找应用，而是从术中肺结节定位的真实临床困难出发逐层形成。肺放气萎陷后，术者往往难以仅依据术前 CT 在术中精准对应结节位置；经验丰富的术者虽然可以通过局部硬度异常感知病灶的大致位置，但这种经验性触诊难以成为统一、可量化的临床范式。前人在乳腺等浅表组织触诊和微创肿瘤触觉定位研究中已经证明，内部机械异质性可以通过动态接触转化为表面时空响应，因此为本文提供了可迁移的物理依据；但肺场景在含气性、塌陷性、形变、CT 失配以及 size-depth 耦合方面显著更复杂。基于此，本文按 detection、size、coarse depth 的顺序组织科学问题，先以结构化特征和 XGBoost baseline 证明任务可学，再以 raw-input 神经网络验证原始时空信号能否直接支持这些任务，并最终通过解释性分析与系统级整合完成闭环验证。
