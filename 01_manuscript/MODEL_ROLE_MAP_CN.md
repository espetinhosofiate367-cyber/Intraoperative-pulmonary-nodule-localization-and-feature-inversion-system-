# 模型角色表（论文口径）

## 目的
这份表专门用来防止论文叙事混乱。  
核心原则是：

- **XGBoost**：证明结构化信息存在、可学、可解释  
- **raw-input 神经网络**：证明原始时空张量本身可学，并自动编码部分物理特征  
- **hybrid 统一模型**：提升最终部署链路在真实 `predicted-route` 条件下的系统级性能

---

## 1. XGBoost 结构化基线

**输入**
- 结构化物理特征表

**回答的问题**
- 数据里有没有可学习的信息？
- 哪些物理特征最重要？

**论文角色**
- 结构化机制基线
- 解释锚点
- baseline 参照
- 不是实时主线

**不应该承担的角色**
- 不作为最终术中部署模型
- 不作为“原始张量自动学习能力”的证据
- 不把“先提物理特征再预测”的链路写成最终在线方案优势

---

## 2. Stage1 raw detector

**输入**
- 原始时空触觉张量

**回答的问题**
- 不输入结构化物理特征时，原始张量本身能否支持稳定结节探测？

**论文角色**
- raw-input scientific model
- detection 主证据

**当前结论**
- detection 上优于 XGBoost
- 在 end-to-end 口径下，避免了显式物理特征提取带来的额外延迟

---

## 3. raw-input size-only router

**输入**
- 原始时空触觉张量

**回答的问题**
- 原始张量本身是否已经包含强大小信息？

**论文角色**
- raw-input scientific model
- size 学习能力证据

**当前结论**
- 优化后的 `raw size-only router v2` 已在 `Top-1 / Top-2 / MAE` 三项上全面超过 XGBoost
- 这条线现在不仅是“可学性证据”，也是 scientific line 中一条真正强的结果线

---

## 4. raw-input size-routed depth model

**输入**
- 原始时空触觉张量
- 不直接输入结构化物理特征

**回答的问题**
- 在显式考虑 `size-depth` 耦合后，原始张量本身能否支持 coarse depth？
- 网络内部是否会自动编码部分与深度相关的物理特征？

**论文角色**
- raw-input scientific main model
- explainability 主证据来源

**当前结论**
- 普通共享 depth head 不成立
- 第一版 `size-routed` 后，GT-route depth 已达到/略超 XGBoost
- 进一步加入 `route-aware` 训练后，pure raw predicted-route depth 也已超过 XGBoost
- latent probe / hard-pair / phase occlusion 支持其内部自动编码了部分物理特征

---

## 5. hybrid unified hierarchical inverter

**输入**
- 原始时空触觉张量
- 结构化物理特征

**回答的问题**
- 在真实部署链路里，如何让 predicted-route depth 更稳？
- 如何把 raw 学习与结构化先验融合成更好的系统模型？

**论文角色**
- deployment model
- system-enhancement model

**当前结论**
- predicted-route depth 超过 XGBoost
- 适合作为最终界面与系统集成模型
- 结构化分支在这里的定位是部署增强，而不是再次证明 raw 自动学习

**不应该单独承担的角色**
- 不作为“纯 raw-input 自动学习机制”的唯一证据

---

## 6. 一句话区分

### Scientific line
- `XGBoost -> raw detector -> raw size -> raw size-routed depth -> raw explainability`

### Deployment line
- `Stage1 detector + hybrid unified inverter + GUI`

---

## 7. 写作时最容易犯的错误

### 错误 1
把 hybrid 模型的结果直接写成：
- “神经网络从原始张量自动学到了所有物理特征”

### 错误 2
把 XGBoost 写成：
- “只是一个早期试验模型”

### 错误 3
把 raw scientific model 和 deployment model 混成同一个角色

---

## 8. 最稳的写法

### 对 XGBoost
- 结构化机制基线
- 由于依赖显式特征提取，更适合作为离线分析与机制参照

### 对 raw-input 神经网络
- 原始张量学习能力主证据
- 更适合作为检测优先的在线主线

### 对 hybrid 模型
- 最终部署增强模型
