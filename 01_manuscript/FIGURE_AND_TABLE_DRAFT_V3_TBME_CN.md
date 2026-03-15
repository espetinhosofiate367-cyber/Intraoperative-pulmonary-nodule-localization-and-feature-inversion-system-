# 图表与表格草案（V4，TBME 重置版）

## 目标
这份版本只服务一个目的：把主文图表收缩成最有说服力的证据链，去掉所有视觉上不稳、证据力不足或容易分散主线的可视化。

## 当前明确不使用的图
1. t-SNE
2. UMAP
3. PCA
4. 各种全局 embedding 对照图
5. 任何不能直接支撑正文主论点的探索性拼图

## 主文图组织原则
1. 主文图优先回答 biomedical engineering 核心问题。
2. 每张图只承担一个结论。
3. `raw-input scientific line` 与 `deployment enhancement line` 严格分开。
4. explainability 只保留最强的直接证据，不再用 embedding 类图辅助。

## 建议主文图顺序
### Fig. 1 引言总图
用途：临床痛点、现有方法局限、病态逆问题、本文研究路径。

### Fig. 2 方法总图
用途：传感器系统、实验设计、数据处理、层级模型。

### Fig. 3 机制分析 + XGBoost 解释总图
用途：证明 detection / size / coarse depth 信息存在，并指出关键物理特征家族。
正文角色：结构化机制基线与解释锚点。

### Fig. 4 Detection 结果图
用途：证明 raw-input 神经网络在 detection 上优于 XGBoost。
正文角色：scientific line 的第一条强证据。

### Fig. 5 Raw-input size 结果图
用途：证明优化后的 pure raw size v2 已在 `Top-1 / Top-2 / MAE` 上全面超过 XGBoost。
正文角色：scientific line 的第二条强证据。

### Fig. 6 Raw-input depth 主结果图
建议内容：普通共享 depth head 失败、size-routed depth 成立、XGBoost depth、majority baseline、GT-route confusion。
正文角色：scientific line 的核心结果，证明 depth 不是不可学，而是必须 size-aware。

### Fig. 7 Raw-input explainability 主图
建议内容：latent probe、phase occlusion、Integrated Gradients 类均值空间归因图，以及 hard-pair inset。
正文角色：证明 raw-input 神经网络自动编码了部分与大小和深度相关的物理结构。

### Fig. 8 Deployment enhancement 图
建议内容：旧 predicted-route 链路 vs pure raw route-aware v2 vs unified hybrid model 的 predicted-route depth 提升，可附 hard confusion。
正文角色：先说明 pure raw 路线已经在真实链路中超过 XGBoost，再说明 unified hierarchical inverter 继续提供当前最强部署结果。

### Fig. 9 实时界面图
用途：说明方案已走向系统原型。
正文角色：工程落地与使用场景连接。

### Fig. 10 延迟与部署口径图
用途：区分 model-only 与 end-to-end 口径，解释为什么 XGBoost 更适合机制基线，而 raw-input NN 更适合作为 detection-first 在线主线。
正文角色：系统工程论证，不夸大为“NN 全面更快”。

## 建议补充材料图
- S1：XGBoost explainability 详细四联图
- S2：更多 raw-input hard-pair 对
- S3：更多 raw-input phase occlusion 细图
- S4：hybrid branch ablation
- S5：代表性 tactile map gallery 或条件矩阵图

## 当前不再进入补充材料的图
- embedding gallery
- clean UMAP triptych
- size-controlled UMAP
- t-SNE / PCA / UMAP 比较图

## 建议主文表格
### Table 1 数据协议与任务定义
### Table 2 Detection 对比结果
### Table 3 Size 对比结果
### Table 4 Depth 对比结果（含 GT-route / predicted-route）
### Table 5 延迟 benchmark

## 当前最稳的图表主线
主文图 = `问题提出 -> 方法组织 -> 结构化基线 -> raw scientific evidence -> deployment enhancement -> real-time prototype`

当前不再使用的视觉路径 = `embedding space -> representation impression`
