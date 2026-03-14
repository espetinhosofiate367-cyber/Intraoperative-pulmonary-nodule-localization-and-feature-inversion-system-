# 图表与表格草案（V3，TBME 目标版）

## 目标
这份版本只服务一个目的：把主文图表压缩成更适合 `TBME` 的主线证据链，避免主图过多、角色混乱或重复证明。

## TBME 主文图的组织原则
1. 主文图优先回答 biomedical engineering 问题，而不是堆所有实验细节。
2. 每张图只回答一个核心问题。
3. `raw-input` 科学主线与 `deployment enhancement` 工程主线必须分开。
4. 深度相关负面或边界信息可以在主文点到，但更细的展开下放补充材料。

## 建议主文图顺序

### Fig. 1 引言总图
用途：临床痛点、现有方法局限、病态逆问题、本文研究路径。
要求：四面板大图，白底，HMIL/TBME 风格，突出 `clinical need -> limitations -> inverse problem -> our solution`。

### Fig. 2 方法总图
用途：传感器系统、实验准备、数据处理、层级模型。
要求：必须强调 `Detection -> Size -> Depth` 的层级任务关系，以及 raw scientific line 与 deployment line 的区别。

### Fig. 3 机制分析 + XGBoost 解释总图
用途：证明 detection / size / coarse depth 信息确实存在，并指出哪些物理特征家族最关键。
正文角色：结构化机制基线与解释锚点。

### Fig. 4 Detection 结果图
用途：证明 raw-input 神经网络在主任务 detection 上优于 XGBoost。
正文角色：raw scientific line 的第一条强证据。

### Fig. 5 Raw-input size 结果图
用途：证明原始张量中存在强大小信息，同时诚实展示 size 回归尚未全面超过结构化基线。
正文角色：raw scientific line 的第二条证据。

### Fig. 6 Raw-input depth 主结果图
建议内容：普通共享 depth head 失败、size-routed depth 成立、XGBoost depth、majority baseline、GT-route confusion。
正文角色：raw scientific line 的核心结果，证明 depth 不是不可学，而是必须 size-aware。

### Fig. 7 Raw-input explainability 主图
建议内容：latent probe、hard-pair、phase occlusion 三联图。
正文角色：证明 raw-input 神经网络自动编码了部分与大小和深度相关的物理结构。

### Fig. 8 Deployment enhancement 图
建议内容：旧 predicted-route 链路 vs unified hybrid model 的 predicted-route depth 提升，可附 hard confusion。
正文角色：说明统一层级反演器的意义是系统增强，而不是再次证明 raw-input 自动学习。

### Fig. 9 实时界面图
用途：说明方案已经从离线比较走向了系统原型。
正文角色：工程落地与临床使用场景连接。

### Fig. 10 延迟与部署口径图
用途：区分 model-only 与 end-to-end 口径，解释为什么 XGBoost 更适合机制基线，而 raw-input NN 更适合作为 detection-first 在线主线。
正文角色：系统工程论证，不夸大为“NN 全面更快”。

## 建议补充材料图
- S1：XGBoost explainability 详细四联图
- S2：全条件矩阵大图
- S3：更多 raw-input hard-pair 对
- S4：更多 raw-input phase occlusion 细图
- S5：hybrid branch ablation
- S6：hybrid latent probe family 图

## 建议主文表格
### Table 1 数据协议与任务定义
### Table 2 Detection 对比结果
### Table 3 Size 对比结果
### Table 4 Depth 对比结果（含 GT-route / predicted-route）
### Table 5 延迟 benchmark

## 最稳的图表主线
主文图 = `问题提出 -> 方法组织 -> 结构化基线 -> raw scientific evidence -> deployment enhancement -> real-time prototype`
补充图 = `更多 explainability 细节 + 更多 robustness 细节`
