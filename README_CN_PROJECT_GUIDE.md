# 项目中文总览与导航说明

> 注：本文件保留为“项目全貌与历史脉络”说明。  
> 如果你当前要继续写稿、画图、复现实验或整理投稿材料，请优先使用：
> - `00_project_management/PROJECT_STATUS_BOARD.md`
> - `01_manuscript/README_CN_CURRENT.md`
> - `REVIEWER_QUICKSTART.md`

## 1. 这个项目现在到底是什么
这个仓库不是一个普通的“只放代码”的项目，而是一个已经被整理成**论文投稿 + 复现实验 + 演示系统**三位一体的发布包。

它围绕一个核心问题展开：

- 如何利用柔性阵列触觉传感，在术中对肺结节进行
  - 结节存在检测
  - 结节大小反演
  - 结节深度粗分类
  - 实时可视化显示

当前仓库已经不是早期探索状态，而是一个**已经收口后的 TBME 投稿工作区**。

你现在应该把它理解成：

1. `01_manuscript/`：论文文字主线
2. `02_figures_final/`：论文图和演示图主线
3. `03_results_core/`：所有能支撑论文结论的核心结果
4. `04_core_models/`：最终最小模型权重集合
5. `05_core_code/`：最终最小复现代码
6. `07_overleaf_figure_pack/`：面向 Overleaf 的图表打包结果
7. `90_archive_unused/`：历史探索分支和淘汰方案


## 2. 项目的主线故事是什么
这篇工作的故事线不是“训练了一个模型”，而是一个逐层递进的闭环：

### 主线 A：机制分析
先回答：

- 深度信息到底是否存在
- 大小与深度分别影响哪些触觉统计量

结论是：

- `size` 是强主效应
- `depth` 不是简单“越深越弱”
- `depth` 更主要影响：
  - 扩散
  - 分布复杂度
  - 形变位置
  - 时间阶段响应

对应文件：

- `03_results_core/MECHANISM_EXPLORATION_REPORT.md`
- `03_results_core/DEPTH_PHYSICS_REPORT.md`


### 主线 B：结构化基线
然后回答：

- 如果不用神经网络，只用结构化物理特征，能不能学到大小和深度

这里我们使用了 `XGBoost`。

结论是：

- detection 上，XGBoost 不是最强
- 但在 `size/depth` 上，它证明了信息真实存在
- 同时它还提供了很重要的解释锚点

对应文件：

- `03_results_core/XGBOOST_BASELINE_REPORT.md`
- `03_results_core/XGBOOST_EXPLAINABILITY_REPORT.md`


### 主线 C：原始输入神经网络
接着回答：

- 不直接喂入手工物理特征，只吃原始时空触觉窗口，神经网络能不能自己把这些信息学出来

这条线最后得到的结论是：

- detection：神经网络优于 XGBoost
- size：神经网络可以接近或部分超过 XGBoost
- depth：如果不做 `size-aware` 设计，网络会失败
- 只有引入 `size-routed` 或统一层级设计后，粗深度分类才真正成立


### 主线 D：系统级统一模型
最后回答：

- 在真实部署链路里，怎样把 `size` 和 `depth` 的误差传播一起处理掉

这里对应的是最终主文模型：

- `hierarchical_positive_inverter`

它不是单纯追求某个单点指标最好，而是追求：

- 在真实 `predicted-route depth` 条件下
- 整条系统链路的最终表现最稳


### 主线 E：可解释性闭环
最后一层不是单看分数，而是回答：

- 神经网络为什么能判别深度
- 它到底关注到了什么

这一步由下面几种分析共同完成：

- XGBoost feature contribution
- latent probe
- branch ablation
- hard-pair analysis
- phase occlusion

因此项目最终形成的不是“黑箱分类器”，而是：

**机制分析 -> 结构化基线 -> 原始输入神经网络 -> 系统级统一架构 -> 可解释性验证**


## 3. 推荐你现在怎么阅读这个项目
如果你现在觉得项目复杂，最稳妥的阅读顺序是：

1. 先看 `01_manuscript/README_CN_CURRENT.md`
2. 再看当前主稿 `01_manuscript/MANUSCRIPT_DRAFT_V8_CN.md`
3. 然后看 `01_manuscript/MODEL_ROLE_MAP_CN.md`
4. 最后再回到图表和结果目录

### 第一步：先看整篇文章主稿
先看：

- `01_manuscript/MANUSCRIPT_DRAFT_V4_CN.md`

这是总入口。


### 第二步：看图表说明
再看：

- `01_manuscript/FIGURE_AND_TABLE_DRAFT_V1_CN.md`

这样你会知道论文每张图和每张表分别承载什么论证任务。


### 第三步：看核心结果
如果你想直接看证据，就看：

- `03_results_core/paper_metrics_summary_v2.json`
- `03_results_core/stage1_detection_summary.json`
- `03_results_core/stage2_size_router_summary.json`
- `03_results_core/hierarchical_inverter_summary.json`
- `03_results_core/hierarchical_explainability_summary.json`


### 第四步：看最终图
对应结果图在：

- `02_figures_final/`

这些图是已经筛过一轮、可以进入论文的主图候选。


### 第五步：看最终代码
只看：

- `05_core_code/`

不要先看外层历史仓库，否则会被大量探索脚本干扰。


## 4. 目录逐个解释

### 01_manuscript
这是**论文主线目录**，优先级最高。

最重要的文件：

- `MANUSCRIPT_DRAFT_V4_CN.md`
  - 当前中文主稿
  - 最重要的论文正文入口
- `RESULTS_DRAFT_V3_CN.md`
  - 结果章节浓缩版
- `FIGURE_AND_TABLE_DRAFT_V1_CN.md`
  - 图表规划与图注草稿
- `TASK_DEFINITION_AND_LABEL_PROTOCOL.md`
  - 输入、标签、协议、划分规则
- `MODEL_AND_SYSTEM_BLUEPRINT.md`
  - 方法结构蓝图
- `TBME_XGBOOST_EXPLAINABILITY_NOTES_CN.md`
  - XGBoost explainability 这条线如何写进 TBME 的说明

一句话：
**如果你要写论文，先看这个目录。**


### 02_figures_final
这是**论文图和演示图主线目录**。

主要内容：

- `Fig_R1` 到 `Fig_R13`
  - 论文主图
- `Fig_XGB_depth_explainability_tbme.png`
  - TBME 风格 XGBoost 主解释图
- `testset_best_grid_hierarchical.png`
  - 42 条件联合矩阵图
- `replay_snapshot_hierarchical.png`
  - 主程序演示图
- `best_prediction_overview_hierarchical.png`
  - 代表性预测大图

一句话：
**如果你要挑图、插图、做答辩，先看这里。**


### 03_results_core
这是**论文证据目录**，里面放的是能支撑主张的核心 summary 和报告。

按功能分：

- 机制分析
  - `MECHANISM_EXPLORATION_REPORT.md`
  - `mechanism_summary.json`
- 深度物理分析
  - `DEPTH_PHYSICS_REPORT.md`
  - `depth_physics_summary.json`
- XGBoost 基线
  - `XGBOOST_BASELINE_REPORT.md`
  - `xgboost_baseline_summary.json`
- XGBoost 可解释性
  - `XGBOOST_EXPLAINABILITY_REPORT.md`
  - `xgboost_explainability_summary.json`
- Stage1 检测
  - `stage1_detection_summary.json`
- Stage2 大小路由
  - `stage2_size_router_summary.json`
- 最终统一模型
  - `hierarchical_inverter_summary.json`
  - `HIERARCHICAL_EXPLAINABILITY_REPORT.md`
  - `hierarchical_explainability_summary.json`
- Stage3 路由评估
  - `STAGE3_PREDICTED_SIZE_ROUTING_REPORT.md`
  - `stage3_predicted_size_routing_summary.json`
- 汇总指标
  - `paper_metrics_summary_v2.json`

一句话：
**如果你要核对“论文里的数值是不是对的”，看这里。**


### 04_core_models
这是**最小可复现权重目录**。

包含：

- `paper_stage1_dualstream_mstcn_best.pth`
  - 最终检测器
- `paper_stage2_raw_positive_size_best.pth`
  - 专用大小路由器
- `paper_hierarchical_positive_inverter_best.pth`
  - 最终统一层级反演器

一句话：
**如果你要跑最终链路，至少要这三个权重。**


### 05_core_code
这是**最小可复现代码目录**。

子目录结构：

#### `models/`
- `task_protocol_v1.py`
  - 协议定义、标签与轴信息
- `dual_stream_mstcn_detection.py`
  - Stage1 检测器结构
- `raw_positive_size_model.py`
  - Stage2 大小路由器结构
- `hierarchical_positive_inverter.py`
  - 最终统一层级反演器结构
- `train_stage1_dualstream_mstcn.py`
  - Stage1 训练
- `train_stage2_raw_positive_size.py`
  - Stage2 训练
- `train_hierarchical_positive_inverter.py`
  - 最终统一模型训练

#### `experiments/`
- `train_xgboost_baselines.py`
  - XGBoost baseline 训练
- `explain_xgboost_baselines.py`
  - XGBoost explainability
- `explain_hierarchical_positive_inverter.py`
  - 最终统一模型 explainability
- `generate_composite_result_figures.py`
  - 汇总结果图生成
- `generate_system_overview_figures.py`
  - 系统总览图和流程图生成

#### `app/`
- `modern_detection_gui_optimized.py`
  - 主程序 GUI
- `two_stage_inference.py`
  - 推理桥接层
- `generate_replay_snapshots.py`
  - 回放截图生成
- `generate_testset_best_grid.py`
  - 42 条件联合矩阵图生成

一句话：
**如果你要复现实验或继续开发，就从这里开始。**


### 06_ai_figure_prompts
这是**AI 绘图提示词目录**。

主要文件：

- `README_ACTIVE_PROMPTS_CN.md`
- `GEMINI_TBME_FINAL_FIGURE_PROMPTS_CN.md`
- `FIG1_INTRO_CONCEPT_GEMINI_PROMPTS_V1_CN.md`

里面已经整理了：

- 当前活跃 prompt 的使用顺序
- 引言总图和方法总图 prompt
- 神经网络架构图 prompt
- XGBoost / 主结果数据图版式 prompt
- GUI / 系统原型图 prompt
- 已归档旧 prompt 的说明

一句话：
**如果你要让图像生成模型画方法图、概念图，就看这里。**


### 07_overleaf_figure_pack
这是**Overleaf 直接使用的图表打包目录**。

包含：

- `figures_png/`
  - PNG 版图
- `figures_pdf/`
  - PDF 版图，更适合投稿
- `tables/`
  - 表格 CSV
- `FIGURE_CAPTIONS_CN.md`
  - 中文图注清单
- `FIGURE_MANIFEST.csv`
  - 图名映射清单
- `OVERLEAF_INSERT_GUIDE.md`
  - Overleaf 使用说明

一句话：
**如果你已经准备去 Overleaf 写文章，这里最有用。**


### 90_archive_unused
这是**归档目录**，不是当前主线。

里面放的是：

- 旧版文稿
- 旧版结果稿
- 被淘汰的网络结构
- 已废弃的深度探索脚本
- 旧实验输出目录的索引

一句话：
**这里是历史记录，不是当前写论文的主要依据。**


## 5. 外层原始工程目录是什么关系
你当前看到的发布仓库只是从外层历史工程里“抽干净”后的版本。

外层完整历史目录：

- `docs/`
  - 历史稿件、说明、探索文档
- `experiments/`
  - 所有实验输出和绘图脚本
- `models/`
  - 所有模型和训练脚本，包括被淘汰版本
- `app/`
  - 主程序与推理桥接
- `doc_rewrite_work/`
  - 针对旧文档的替换稿工作区
- `pre_dl_baseline_summary/`
  - 机制分析和 baseline 的整理包
- `tbme_submission_release_v1/`
  - 当前这份干净发布版

换句话说：

- 外层目录 = 开发历史全景
- `tbme_submission_release_v1/` = 投稿与发布版


## 6. 如果你现在只关心“最终有效成果”
你只需要记住下面这些：

### 论文主稿
- `01_manuscript/MANUSCRIPT_DRAFT_V4_CN.md`

### 最终结果图
- `02_figures_final/`

### 最终结果数值
- `03_results_core/paper_metrics_summary_v2.json`

### 最终系统权重
- `04_core_models/`

### 最终系统代码
- `05_core_code/`

### Overleaf 投递图包
- `07_overleaf_figure_pack/`


## 7. 如果你现在只关心“结论”
目前项目最稳的结论是：

1. 原始时空触觉数据可以稳定完成结节检测
2. 结节大小是强主效应
3. 深度不是简单强度衰减问题，而是一个弱但可学习的 `size-dependent` 信号
4. 若不做 `size-aware` 设计，原始输入神经网络学不好深度
5. 统一层级反演器在真实 predicted-route depth 上已经超过 XGBoost
6. 可解释性分析表明最终模型确实利用了超出 size identity 本身的深度相关结构


## 8. 最后给你的最简导航
如果你以后忘了从哪开始，就按这个顺序：

1. 看 `README.md`
2. 看 `README_CN_PROJECT_GUIDE.md`
3. 看 `01_manuscript/MANUSCRIPT_DRAFT_V4_CN.md`
4. 看 `02_figures_final/`
5. 看 `03_results_core/paper_metrics_summary_v2.json`
6. 写作时直接用 `07_overleaf_figure_pack/`
