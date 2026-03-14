# 深度学习模型架构总览（当前论文版本）

## 这份文件回答什么
这份说明专门回答：

1. 现在论文里真正使用的深度学习模型有哪些
2. 它们之间是什么关系
3. 每个模型的输入、主干、输出头分别是什么
4. 哪些适合画进主文，哪些只适合在补充材料中点到为止

---

## 一、整体模型地图

当前论文里的深度学习路线已经明确分成两条：

### 1. Scientific raw-input line
只输入原始时空张量，不直接输入结构化物理特征。

包含：
- `Stage I raw detector`
- `Stage II raw size-only router v2`
- `Stage III raw route-aware size-routed depth model v2`

这条线回答的是：
- 原始触觉张量本身能不能学
- 学到了什么
- 能不能在 detection、size 和 predicted-route depth 上超过 XGBoost

### 2. Deployment enhancement line
在 raw 输入之外，再融合结构化物理特征。

包含：
- `Unified hierarchical inverter`

这条线回答的是：
- 最终系统怎么做得更稳
- predicted-route depth 怎么再往上提

---

## 二、Stage I：Raw Detector

### 输入
- `10 × 1 × 12 × 8` 的归一化触觉窗口
- 一条 `raw/norm` 主序列
- 一条 `delta` 分支

### 结构
1. `FrameEncoder2D` 对每一帧做空间编码
2. `delta branch` 建模相邻帧变化
3. `1×1 temporal input projection`
4. 多层 `MS-TCN`
5. `TemporalAttentionPooling`
6. 二分类输出头

### 一句话理解
这是一个**双流时空检测器**：
- 一条看静态空间分布
- 一条看动态变化
- 中间用 `MS-TCN` 做短时序建模

### 适合怎么画
画成：
- `Input window`
- `Raw/Norm branch`
- `Delta branch`
- `MS-TCN`
- `Attention pooling`
- `Detection probability`

不需要把每一层卷积细节画满。

---

## 三、Stage II：Raw Size-Only Router v2

### 输入
- `raw amplitude sequence`
- `normalized shape sequence`
- `delta sequence`

### 主干
三条输入支路都用：
- `FrameEncoder2D`

然后拼接后进入：
- `temporal input projection`
- 多层 `MultiScaleTemporalBlock`
- `TemporalAttentionPooling`
- `PhaseAwarePooling`
- `trunk`

### 输出头
现在的 `v2` 不是旧版简单双头，而是：

1. `size_cls_head`
2. `size_ord_head`
3. `expected size from class probabilities`
4. `size_residual_head`

最终：
\[
\hat{s}_{norm} = \mathrm{clip}\left(\sum_c p(c|x)v_c + \Delta s,\ 0,\ 1\right)
\]

### 为什么这版更强
因为它把 size 问题拆成：
- 先判断属于哪一档
- 再做档内连续修正

而不是直接暴力回归一个绝对值。

### 适合怎么画
画成：
- 三路输入
- 共享时序主干
- `classification + ordinal + expectation + residual`
- 输出：
  - `size class probabilities`
  - `continuous size estimate`

---

## 四、Stage III：Raw Route-Aware Size-Routed Depth Model v2

### 输入
- 同样是纯 raw 输入：
  - `raw amplitude`
  - `normalized shape`
  - `delta`

### 深度模型本体
主干仍然是：
- 三路 `FrameEncoder2D`
- `MS-TCN`
- `TemporalAttentionPooling`
- `PhaseAwarePooling`
- `trunk`

不同点在于输出头：
- 不是一个共享 depth head
- 而是 `7` 个 `size-routed depth experts`

也就是：
- 每个 size 类对应一个 depth expert

### 训练时为什么叫 route-aware
因为它不是只按 GT size route 训练，而是同时优化：

1. `GT route`
2. `Hard predicted route`
3. `Soft route`
4. `Top2-soft route`
5. `Consistency KL`

这一步的目的不是再证明“depth 存在”，而是让 pure raw 路线在真实 `size -> depth` 链路中也稳住。

### 适合怎么画
最推荐画成：

- 左侧：`Frozen raw size router v2`
- 右侧：`Raw depth trunk + size-routed experts`
- 中间用 4 条不同颜色箭头表示：
  - `GT route`
  - `Hard route`
  - `Soft route`
  - `Top2 route`
- 底部画联合损失

这张图会非常有说服力，因为它直接解释了为什么 predicted-route depth 能提升。

---

## 五、Unified Hierarchical Inverter

### 输入
- `raw amplitude sequence`
- `normalized shape sequence`
- `delta`
- `structured tactile features`

### 主干
1. raw / shape / delta 三路编码
2. 时序主干
3. tabular branch
4. `trunk feat + tabular feat + interaction`
5. fusion

### 输出
- `size classification`
- `size ordinal`
- `size regression`
- `routed coarse depth`

### 它在论文里的角色
不是 scientific main model，  
而是 deployment model。

所以图里一定要强调：
- “structured feature branch added for deployment enhancement”

而不是画成：
- “神经网络全都自动学出来了”

---

## 六、主文最应该画哪两张架构图

### 最推荐主文保留
1. **Fig.2 方法总图**
   - 把整个链路讲清楚

2. **单独一张 raw scientific line 架构图**
   - `Detector -> Size v2 -> Route-aware Depth v2`

如果版面吃紧，可以把第 2 张并进 Fig.2 的右半部分。

### 不建议主文把所有网络细节分成 4 张
那会太碎。

更好的方法是：
- 主文：一个总图 + 一个 scientific line 重点图
- Supplement：再放 unified / route-aware 细节图

---

## 七、最适合你现在画图时用的文字骨架

### 一句话版
当前论文的深度学习体系，是一个以 `Detection -> Size -> Depth` 为主线的层级时空网络：
- `Stage I` 负责 detection
- `Stage II` 用纯 raw 输入完成 size v2 反演
- `Stage III` 用 size-routed experts 和 route-aware 训练完成 coarse depth
- 最后再用 unified hybrid model 做部署增强

### 图中建议固定出现的关键词
- `raw amplitude`
- `normalized shape`
- `delta`
- `MS-TCN`
- `attention pooling`
- `phase-aware pooling`
- `size classification`
- `ordinal size`
- `continuous size`
- `size-routed depth experts`
- `GT / hard / soft / top2 route`
- `deployment enhancement`

