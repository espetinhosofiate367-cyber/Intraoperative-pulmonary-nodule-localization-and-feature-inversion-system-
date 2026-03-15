# 项目状态看板

## 当前目标
- 目标期刊：`IEEE TBME`
- 当前主稿：`MANUSCRIPT_DRAFT_V12_CN.md`
- 当前工程状态：`投稿收口阶段`

## 活跃工作流

### 1. 文稿主线
- 状态：`Active`
- 入口：`01_manuscript/README_CN_CURRENT.md`
- 当前基线：`MANUSCRIPT_DRAFT_V12_CN.md`
- 说明：文稿已按 `mechanism -> XGBoost -> raw scientific line -> deployment enhancement` 重组；支撑性写作资料已移入 `01_manuscript/supporting_notes/`。

### 2. 图表主线
- 状态：`Active`
- 入口：`01_manuscript/FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md`
- 说明：当前只保留能直接支撑主张的强证据图；embedding 等弱图已退出主线。

### 3. 结果证据
- 状态：`Active`
- 入口：`03_results_core/`
- 说明：Detection、Size、Depth、Latency 的核心 summary 已稳定，供论文与审稿复核使用。

### 4. 复现链路
- 状态：`Active`
- 入口：`05_core_code/README_REPRODUCTION.md`
- 说明：当前 release 以最小复现为目标，不再暴露完整历史开发脚本。

### 5. 历史归档
- 状态：`Archived`
- 入口：`90_archive_unused/`
- 说明：旧文稿、旧 prompt、探索性弱图、仓库管理杂项均已移出活跃区。

## 本轮整理后保留的核心活跃交付物
1. `01_manuscript/MANUSCRIPT_DRAFT_V12_CN.md`
2. `01_manuscript/FIGURE_AND_TABLE_DRAFT_V3_TBME_CN.md`
3. `01_manuscript/FIGURE_SYSTEM_BLUEPRINT_TBME_V1_CN.md`
4. `01_manuscript/TABLES_TBME_READY_V3_CN.md`
5. `03_results_core/`
6. `04_core_models/`
7. `05_core_code/`
8. `06_ai_figure_prompts/` 中的活跃 prompt 文件
9. `08_overleaf_draft_v1/`

## 本轮整理后已归档的材料
- `MANUSCRIPT_DRAFT_V10_CN.md`
- `MANUSCRIPT_DRAFT_V11_CN.md`
- `TABLES_TBME_READY_V1_CN.md`
- `FIGURE_REDRAW_BRIEFS_TBME_CN.md`
- 旧版引言/方法 prompt 文件 5 份

## 接下来最自然的三步
1. 完成 `Fig.1` 和 `Fig.2` 的最终概念图/方法图。
2. 统一 `Fig.3` 到 `Fig.10` 的数据图版式。
3. 将 `08_overleaf_draft_v1/` 与当前主稿、图表编号进一步对齐。
