# 项目管理入口

这个目录只负责三件事：
1. 定义当前活跃交付物
2. 记录当前工作状态与下一步
3. 明确哪些材料属于归档、哪些属于当前主线

## 建议先看
- `PROJECT_STATUS_BOARD.md`
- `DELIVERABLE_REGISTER.md`

## 当前活跃主线
- 论文主稿：`01_manuscript/MANUSCRIPT_DRAFT_V12_CN.md`
- 图表组织：`01_manuscript/FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md`
- 图表蓝图：`01_manuscript/FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md`
- 主结果：`03_results_core/`
- 最小复现代码：`05_core_code/`
- 审稿入口：`REVIEWER_QUICKSTART.md`

## 归档原则
- 已被新版本替代的文稿与表格：移入 `90_archive_unused/manuscript_history/` 或 `90_archive_unused/deprecated_docs/`
- 已被新的 Gemini/TBME prompt 体系替代的旧 prompt：移入 `90_archive_unused/legacy_figure_prompts/`
- 视觉说服力不稳定的探索性图：移入 `90_archive_unused/rejected_visualization_gallery_*`
