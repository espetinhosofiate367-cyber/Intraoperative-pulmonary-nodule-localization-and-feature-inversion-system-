# 纯原始输入 Size 模型优化说明

## 当前结论先说清楚
**能优化，而且还有比较明确的优化空间。**

但这里要分清：

1. 当前 `raw-input size-only` 模型并不是“完全不行”
2. 它的问题主要集中在 **size regression / MAE**
3. 这个问题更像是 **输出头设计和训练目标不够贴合 size 的有序结构**，而不是 backbone 完全学不到大小信息

当前结果：

| 模型 | Top-1 | Top-2 | MAE (cm) |
|---|---:|---:|---:|
| XGBoost | **0.6701** | 0.8023 | **0.1472** |
| Raw size-only router | 0.6600 | **0.8405** | 0.2907 |

这说明：
- 原始时空张量里，**大小分类信息已经学出来了**
- 但 **连续大小回归** 仍然明显落后于 XGBoost

所以接下来的目标，不是推翻 raw-input 这条线，而是：

> **保留纯原始输入前提下，把 size 输出头和训练目标改得更像“有序尺寸估计”，而不是普通分类 + 生硬回归。**

---

## 当前 raw size-only 模型到底差在哪

当前 `RawPositiveSizeModel` 的 backbone 并不算弱：
- amplitude branch
- normalized shape branch
- delta branch
- temporal blocks
- attention pooling
- phase-aware pooling

所以它的问题不太像是“输入信息不够”或者“时空主干完全不行”。

更大的问题在于 **size head 设计偏朴素**：

1. 只有一个普通 `size_cls_head`
2. 只有一个普通 `size_reg_head`
3. 回归头直接从 trunk 输出一个连续 `cm` 值
4. 模型选择时更偏 `Top-1 / Top-2`，没有把 `MAE` 放在更核心的位置

这会带来两个后果：

### 1. 没有利用 size 的有序结构
7 个 size 类之间不是彼此独立类别，而是天然有序：

`0.25 < 0.5 < 0.75 < 1.0 < 1.25 < 1.5 < 1.75`

普通交叉熵会把：
- `1.0 -> 1.25`
- `1.0 -> 1.75`

都看成一样错。

但实际上第二种错误更严重。

### 2. 直接回归头太难
让 trunk 直接输出绝对大小，会把：
- 类别决策
- 有序关系
- 连续偏移

全部压到一个头里完成。

这对小样本、强离散标签的任务并不友好。

XGBoost 之所以 MAE 更强，一个重要原因是：
- 它已经先看到了对 size 很敏感的人工结构化特征
- 回归空间更稳定

---

## 为什么我认为纯 raw-input 还值得继续优化

因为现在的结果其实已经给了很强信号：

1. `Top-1 = 0.6600` 已经非常接近 XGBoost 的 `0.6701`
2. `Top-2 = 0.8405` 已经超过 XGBoost 的 `0.8023`
3. 这说明 raw-input backbone 对 size 的**排序感和近邻判断能力**已经不差

真正掉队的是：
- `MAE = 0.2907`

这说明模型更像是在：
- “大致知道属于哪一档”
- 但“档内连续位置估计”还不够稳

这类问题通常更适合通过 **输出头和损失函数重构** 去解决，
而不是第一时间无限堆大 backbone。

---

## 最值得先试的纯 raw-input 优化方向

### 方案 A：把 hybrid 模型里成熟的 size head 迁回 pure raw
这是我最推荐的第一步，也是**低风险高收益**的方案。

因为当前 unified / hybrid 路线里，其实已经有一套更成熟的 size 头：

1. `size_cls_head`
2. `size_ord_head`
3. `expected size from class probabilities`
4. `size_residual_head`

它的思想是：

先做分类概率：
\[
p(c \mid x)
\]

再算一个有序期望值：
\[
\hat{s}_{base} = \sum_{c=1}^{C} p(c \mid x)\, v_c
\]

其中 \(v_c\) 是 size 类对应的归一化中心。

然后再加一个小范围残差修正：
\[
\hat{s}_{norm} = \mathrm{clip}(\hat{s}_{base} + \Delta s, 0, 1)
\]

最终映射回物理大小：
\[
\hat{s}_{cm} = s_{min} + \hat{s}_{norm}(s_{max}-s_{min})
\]

这比“直接回归一个绝对值”更自然，因为它把问题拆成：
- 先决定大概属于哪一档
- 再决定档内偏移量

这特别适合我们这种：
- 类别有序
- 标签水平有限
- size 是强主效应

的任务。

### 为什么这一步重要
因为这一步仍然是 **纯 raw-input**

它没有引入结构化物理特征，因此不会污染我们“raw scientific model”的证据链。

---

### 方案 B：在 pure raw 中加入 ordinal loss
当前最值得补的第二步，是把 size 分类从“普通 7 类分类”改成“分类 + 有序约束”。

具体可以写成：
\[
\mathcal{L}_{size}
= \mathcal{L}_{cls}
 \lambda_{ord}\mathcal{L}_{ord}
 \lambda_{reg}\mathcal{L}_{reg}
\]

其中：

- \(\mathcal{L}_{cls}\)：普通交叉熵
- \(\mathcal{L}_{ord}\)：ordinal binary loss
- \(\mathcal{L}_{reg}\)：Smooth L1 regression loss

这样能显式告诉模型：
- `1.25 cm` 更接近 `1.0 cm`，而不是和 `0.25 cm` 一样远

这通常对 MAE 改善比单纯加深 backbone 更靠谱。

---

### 方案 C：模型选择标准改成 `MAE-first`
当前原始 size-only 路线还有一个很实际的问题：

> 最优 checkpoint 的选择逻辑过于偏向 `Top-1 / Top-2`

但如果我们现在的目标是补短板，就不该继续按分类优先选模型。

更合理的是：
- 验证集先看 `MAE`
- 再看 `Top-1`
- 再看 `Top-2`

否则模型会自然朝着“分档正确但连续值粗糙”的方向收敛。

这一步不是架构创新，但对结果很关键。

---

### 方案 D：加入邻域软标签
因为 size 类是有序的，我们可以允许模型对相邻类保持一定软概率。

例如对真实类别 \(y\)，构造一个高斯邻域软标签：
\[
t_c \propto \exp\left(-\frac{(c-y)^2}{2\sigma^2}\right)
\]

再加入一个小权重 KL loss：
\[
\mathcal{L}_{soft} = \mathrm{KL}(p(c\mid x)\,\|\,t)
\]

这一步的作用不是替代交叉熵，
而是减少“离谱错分”，让 class probability 更平滑、更有序。

对于 `Top-2` 已经不错但 `MAE` 仍然偏大的情况，这往往是有帮助的。

---

## 我不建议优先做的事

### 1. 不建议第一时间继续堆更大的 backbone
例如：
- 更深卷积
- 更多 temporal block
- 更大 hidden dim

原因是：
- 现在 `Top-1 / Top-2` 已经不差
- 如果 backbone 真不行，分类也不该这么接近 XGBoost

所以继续盲目堆主干，收益未必大，还会增加训练不稳和过拟合风险。

### 2. 不建议直接把结构化物理特征加回 pure raw 模型
因为那会模糊论文里的角色边界。

当前 pure raw 路线的任务是：
- 证明原始张量可学

而不是：
- 尽一切办法拿最优 size MAE

最优 MAE 可以留给 deployment enhancement 或 hybrid 模型去做。

---

## 我建议的 pure raw v2 架构

### Backbone
保持现有三流结构不变：
- amplitude branch
- normalized shape branch
- delta branch
- temporal blocks
- temporal attention pooling
- phase-aware pooling

### Size 输出头
从：
- `size_cls_head + direct size_reg_head`

改成：
- `size_cls_head`
- `size_ord_head`
- `expected_norm from size_probs`
- `size_residual_head`

### 损失函数
\[
\mathcal{L}
= \mathcal{L}_{cls}
 \lambda_{ord}\mathcal{L}_{ord}
 \lambda_{reg}\mathcal{L}_{reg}
 \lambda_{soft}\mathcal{L}_{soft}
\]

推荐初始权重：
- \(\lambda_{ord}=0.20 \sim 0.30\)
- \(\lambda_{reg}=0.60 \sim 0.80\)
- \(\lambda_{soft}=0.00 \sim 0.10\)

### 最优模型选择
验证集优先级：
1. `MAE`
2. `Top-1`
3. `Top-2`
4. `Loss`

---

## 这条优化路线如果成功，最可能改善什么

### 最可能改善
1. `MAE`
2. 近邻类之间的回归稳定性
3. 与 XGBoost 的 size gap

### 可能顺带改善
1. predicted-route depth
2. depth routing robustness

因为 depth 当前的一个主要瓶颈，本来就来自：
- 上游 size route 错误

所以 size 只要更稳，后面的 depth 链也会受益。

---

## 这条路线在论文里怎么写才合理

如果后面我们真的做了 pure raw v2，并且结果更好，
论文里最合理的说法不是：

> 我们把网络堆得更大了，所以更强。

而是：

> 结果表明，raw-input size learning 的主要瓶颈不在于 backbone 无法提取大小信息，而在于 size prediction head 没有充分利用类别的有序结构。通过把 size prediction 重构为“分类 + ordinal + expectation + residual”形式，模型在不引入结构化物理特征的前提下，进一步缩小了与 XGBoost 在 size regression 上的差距。

这条逻辑更严谨，也更有说服力。

---

## 最终建议

### 当前最值得做的，不是“换一个更大网络”
而是先做一个 **pure raw size v2**：

1. 保持现有三流 backbone
2. 把 hybrid 里的 size 头迁回 pure raw
3. 加 ordinal loss
4. 加 expectation + residual regression
5. 把 checkpoint selection 改成 `MAE-first`

如果这一步做完仍然明显不如 XGBoost，
那时再考虑：
- detector warm start
- curriculum training
- 更强 backbone

这样推进是最稳的。
