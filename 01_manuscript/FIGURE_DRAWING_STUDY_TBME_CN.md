# 图表绘制研究说明（从表到图）

## 目标
这份文件不再讨论“能画什么炫图”，而是只回答一件事：

**每张主文图，应该由哪张表出发，画成什么形式，才能最直接说明问题。**

---

## Fig. 3 机制分析 + XGBoost 解释总图

### 对应表
- `Main Table 4` 的 XGBoost 行
- `Supplementary Table S2` 中与结构解释相关的概念指标

### 最适合的画法
1. feature-family contribution share 条形图
2. class-wise concept heatmap
3. 代表性 tactile map 三联图
4. 一个 partial dependence / monotonic trend 小图

### 不建议
- 画太多 SHAP 小点云
- 画 embedding 图

---

## Fig. 4 Detection 结果图

### 对应表
- `Main Table 2`

### 最适合的画法
1. ROC 曲线
2. PR 曲线
3. AUC / AP / F1 三柱图

### 画图重点
- 强调 raw detector > XGBoost
- raw scientific line 用蓝绿
- XGBoost 用灰蓝

---

## Fig. 5 Size 结果图

### 对应表
- `Main Table 3`

### 最适合的画法
1. Top-1 / Top-2 grouped bars
2. MAE bar
3. 一个小型 parity inset 或 absolute error inset

### 画图重点
- raw size-only v2 是主角
- unified model 只是 deployment 参照
- 不要让图看起来像 unified 和 raw scientific line 在争主角

---

## Fig. 6 Depth 主结果图

### 对应表
- `Main Table 4`
- `Supplementary Table S1`

### 最适合的画法
1. BAcc comparison 主柱图
2. GT-route confusion matrix
3. Predicted-route confusion matrix
4. route-aware breakdown 小图

### 画图重点
- 一眼看出 shared-head failure
- 一眼看出 size-aware 成立
- 一眼看出 route-aware v2 > XGBoost

---

## Fig. 7 Explainability 主图

### 对应表
- `Supplementary Table S2`

### 最适合的画法
1. latent probe family bar chart
2. hard-pair representative examples
3. phase occlusion summary

### 不建议
- t-SNE
- UMAP
- PCA
- 任何“看上去很像结果、但实际上只是一种视觉 impression”的图

---

## Fig. 8 Deployment enhancement 图

### 对应表
- `Main Table 4`
- `Supplementary Table S1`
- `Supplementary Table S3`

### 最适合的画法
1. predicted-route BAcc 主柱图
2. unified predicted-route confusion
3. route robustness breakdown

### 画图重点
- 先让读者看到 pure raw already crossed XGBoost
- 再让读者看到 unified 继续往上推

---

## Fig. 10 Latency 图

### 对应表
- `Main Table 5`

### 最适合的画法
1. detection path latency bars
2. full-chain latency bars

### 画图重点
- 明确区分 `model-only` 与 `end-to-end`
- 不要把图画成“NN 全面更快”
- 要让审稿人看到：显式特征提取本身就是 structured route 的工程负担
