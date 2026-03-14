# 活跃 Prompt 使用说明

## 当前应优先使用的 4 份文件
1. `FIG1_INTRO_CONCEPT_GEMINI_PROMPTS_V1_CN.md`
   - `Fig.1` 引言概念总图
2. `GEMINI_TBME_FINAL_FIGURE_PROMPTS_CN.md`
   - 主文全部图的 Gemini/TBME 定稿 prompt
3. `NEURAL_AND_DATA_FIGURE_PROMPTS_TBME_CN.md`
   - 神经网络、XGBoost 和数据图专用 prompt
4. `HMIL_FIGURE_STYLE_REFERENCE_CN.md`
   - 方法图与整体版式风格参考

## 不再建议继续使用的旧 prompt
以下文件已移至：`90_archive_unused/legacy_figure_prompts/`
- `AI_FIGURE_PROMPTS_CN.md`
- `INTRODUCTION_FIGURE_PROMPTS_CN.md`
- `INTRODUCTION_OVERVIEW_SINGLE_PROMPT_CN.md`
- `METHODS_FIGURE_PROMPTS_CN.md`
- `METHODS_OVERVIEW_SINGLE_PROMPT_CN.md`

## 当前建议的使用顺序
1. 先读 `GEMINI_TBME_FINAL_FIGURE_PROMPTS_CN.md`
2. 画 `Fig.1` 时直接用 `FIG1_INTRO_CONCEPT_GEMINI_PROMPTS_V1_CN.md`
3. 画 `Fig.2/2B/2C` 时结合 `HMIL_FIGURE_STYLE_REFERENCE_CN.md`
4. 数据图只用 Gemini 生成版式，不让 Gemini 凭空捏数字
