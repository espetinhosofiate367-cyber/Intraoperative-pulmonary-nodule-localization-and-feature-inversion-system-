# 文稿区当前使用说明

## 当前应优先使用的文件

### 1. 当前主稿
- [MANUSCRIPT_DRAFT_V11_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/MANUSCRIPT_DRAFT_V11_CN.md)

这是目前最新、最贴近 `TBME` 投稿口径的一版中文主稿。
当前定位是：
- 探测优先
- 大小反演为第二层
- 粗深度辨别为探索性扩展
- `raw-input` 神经网络负责科学主证据
- `hybrid` 统一模型负责部署增强
- 整体口径为“前沿性探索研究”，不过度写大
- 最新结果已经补入：
  - pure raw size v2 全面超过 XGBoost
  - pure raw route-aware depth predicted-route 超过 XGBoost

### 2. 当前图表组织方案（TBME 目标版）
- [FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md)
- [FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md)
- [FIGURE_STRATEGY_RESET_TBME_V1_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/FIGURE_STRATEGY_RESET_TBME_V1_CN.md)

这三份文件共同定义：
- 主文图放哪些
- 补充材料放哪些
- 哪些图服务 raw scientific line
- 哪些图服务 deployment enhancement line
- 当前明确不再采用哪些图

### 3. 模型角色说明
- [MODEL_ROLE_MAP_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/MODEL_ROLE_MAP_CN.md)
- [DEEP_MODEL_ARCHITECTURE_OVERVIEW_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/DEEP_MODEL_ARCHITECTURE_OVERVIEW_CN.md)

如果你担心论文叙事混乱，先看这份。它专门回答：
- XGBoost 在论文里是什么角色
- raw-input 神经网络是什么角色
- hybrid 模型是什么角色

如果你现在最关心的是“当前深度学习架构到底长什么样”，直接看：
- `Stage I raw detector`
- `Stage II raw size-only router v2`
- `Stage III raw route-aware size-routed depth model v2`
- `Unified hierarchical inverter`

### 4. TBME 冲刊文档
- [TBME_GAP_AND_ACTION_PLAN_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/TBME_GAP_AND_ACTION_PLAN_CN.md)
- [TBME_COVER_LETTER_DRAFT_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/TBME_COVER_LETTER_DRAFT_CN.md)
- [RELATED_WORK_DIRECT_CITATIONS_TBME_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/RELATED_WORK_DIRECT_CITATIONS_TBME_CN.md)
- [REFERENCE_VERIFICATION_TBME_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/REFERENCE_VERIFICATION_TBME_CN.md)
- [FIGURE_CAPTIONS_TBME_V1_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/FIGURE_CAPTIONS_TBME_V1_CN.md)
- [TABLES_TBME_READY_V2_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/TABLES_TBME_READY_V2_CN.md)
- [FORMULAS_AND_PARAMETERS_TBME_V1_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/FORMULAS_AND_PARAMETERS_TBME_V1_CN.md)

### 5. 画图 Prompt 与施工资料
- [NEURAL_AND_DATA_FIGURE_PROMPTS_TBME_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/06_ai_figure_prompts/NEURAL_AND_DATA_FIGURE_PROMPTS_TBME_CN.md)
- [INTRODUCTION_OVERVIEW_SINGLE_PROMPT_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/06_ai_figure_prompts/INTRODUCTION_OVERVIEW_SINGLE_PROMPT_CN.md)
- [METHODS_OVERVIEW_SINGLE_PROMPT_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/06_ai_figure_prompts/METHODS_OVERVIEW_SINGLE_PROMPT_CN.md)

这三份文件分别负责：
- 神经网络架构图、XGBoost 图、主结果数据图的 prompt
- 引言总图 prompt
- 方法总图 prompt

### 6. 支撑文稿写作的辅助文件
- [MODEL_AND_SYSTEM_BLUEPRINT.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/MODEL_AND_SYSTEM_BLUEPRINT.md)
- [TASK_DEFINITION_AND_LABEL_PROTOCOL.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/TASK_DEFINITION_AND_LABEL_PROTOCOL.md)
- [TBME_XGBOOST_EXPLAINABILITY_NOTES_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/TBME_XGBOOST_EXPLAINABILITY_NOTES_CN.md)
- [RAW_SIZE_MODEL_OPTIMIZATION_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/RAW_SIZE_MODEL_OPTIMIZATION_CN.md)
- [RAW_ROUTE_AWARE_DEPTH_OPTIMIZATION_CN.md](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/01_manuscript/RAW_ROUTE_AWARE_DEPTH_OPTIMIZATION_CN.md)

### 7. Overleaf 英文骨架
- [08_overleaf_draft_v1](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/08_overleaf_draft_v1)

这个目录包含：
- `main.tex`
- 英文章节草稿
- 参考文献占位
- 已复制进去的主文图 PDF
- 可直接上传到 Overleaf 的基础工程

## 已归档的历史草稿
旧版本已经移到：
- [90_archive_unused/manuscript_history](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/90_archive_unused/manuscript_history)

## 已归档的探索性可视化
以下图组已经明确退出当前论文主线：
- [rejected_visualization_gallery_v1_20260314_133856](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/90_archive_unused/rejected_visualization_gallery_v1_20260314_133856)

归档原因：
- 视觉说服力不稳定
- 对主文论证帮助有限
- 容易分散审稿人注意力

说明：
- 由于个别文件正在被本地程序占用，原目录下可能还残留少量同组图片。
- 这些残留文件同样视为已弃用，不再进入当前主线。

## 最推荐的阅读顺序
1. `MODEL_ROLE_MAP_CN.md`
2. `TBME_GAP_AND_ACTION_PLAN_CN.md`
3. `MANUSCRIPT_DRAFT_V11_CN.md`
4. `FIGURE_STRATEGY_RESET_TBME_V1_CN.md`
5. `FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md`
6. `FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md`
7. `DEEP_MODEL_ARCHITECTURE_OVERVIEW_CN.md`
8. `NEURAL_AND_DATA_FIGURE_PROMPTS_TBME_CN.md`
9. `RELATED_WORK_DIRECT_CITATIONS_TBME_CN.md`
10. `REFERENCE_VERIFICATION_TBME_CN.md`
11. `FIGURE_CAPTIONS_TBME_V1_CN.md`
12. `TABLES_TBME_READY_V2_CN.md`
13. `FORMULAS_AND_PARAMETERS_TBME_V1_CN.md`
14. `RAW_SIZE_MODEL_OPTIMIZATION_CN.md`
15. `RAW_ROUTE_AWARE_DEPTH_OPTIMIZATION_CN.md`
16. `03_results_core/raw_size_v2/`
17. `03_results_core/raw_routeaware_depth_v2/`
18. `03_results_core/latency_benchmark_v1/`
19. `08_overleaf_draft_v1/`
