# 公式与参数说明（TBME 严格版）

## 1. 输入张量与窗口定义

设第 $t$ 帧的触觉矩阵为

$$
\mathbf{F}_t \in \mathbb{R}^{12 \times 8},
$$

每一帧由 96 个触觉通道重排得到。  
正式模型的基本样本单位不是单帧，而是长度为 $T=10$ 的滑动窗口：

$$
\mathbf{W}_i = \{\mathbf{F}_{i}, \mathbf{F}_{i+1}, \ldots, \mathbf{F}_{i+9}\},
$$

窗口步长固定为

$$
\Delta = 2.
$$

因此，正式协议的输入张量为

$$
\mathbf{W}_i \in \mathbb{R}^{10 \times 1 \times 12 \times 8}.
$$

---

## 2. Detection 标签定义

设人工标注的阳性时段集合为 $\mathcal{S}$。  
第 $i$ 个窗口中心帧索引记为 $c_i$。Detection 标签定义为

$$
y_i^{det} =
\begin{cases}
1, & c_i \in \mathcal{S}, \\
0, & c_i \notin \mathcal{S}.
\end{cases}
$$

这一定义保证了 detection 任务是一个**中心帧判定**问题，而不是“窗口内只要碰到阳性就算阳性”的宽松协议。

---

## 3. Size 与 coarse depth 标签

正式 size 轴为

$$
\mathcal{S}_{size} = \{0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75\}\,\text{cm},
$$

depth 轴为

$$
\mathcal{S}_{depth} = \{0.5, 1.0, 1.5, 2.0, 2.5, 3.0\}\,\text{cm}.
$$

为了避免过早做连续深度反演，本文定义 coarse depth 映射：

$$
\text{shallow} = \{0.5, 1.0\}, \quad
\text{middle} = \{1.5, 2.0\}, \quad
\text{deep} = \{2.5, 3.0\}.
$$

即：

$$
y_i^{depth} \in \{\text{shallow}, \text{middle}, \text{deep}\}.
$$

---

## 4. 结构化物理特征定义

设归一化帧为 $\widetilde{\mathbf{F}}_t$。本文中用于机制分析与 structured baseline 的关键物理特征包括：

### 4.1 峰值幅值

$$
f_{\text{raw-max}}(\mathbf{F}_t) = \max_{r,c} \mathbf{F}_t(r,c).
$$

### 4.2 空间熵

先定义归一化权重

$$
p_{r,c} = \frac{\max(\widetilde{\mathbf{F}}_t(r,c), 0)}{\sum_{u,v} \max(\widetilde{\mathbf{F}}_t(u,v), 0)},
$$

则归一化空间熵为

$$
f_{\text{entropy}}(\widetilde{\mathbf{F}}_t)
=
-\frac{\sum_{r,c} p_{r,c} \log p_{r,c}}{\log(12 \times 8)}.
$$

### 4.3 质心

$$
\mu_r = \frac{\sum_{r,c} r\,p_{r,c}}{\sum_{r,c} p_{r,c}}, \qquad
\mu_c = \frac{\sum_{r,c} c\,p_{r,c}}{\sum_{r,c} p_{r,c}}.
$$

### 4.4 热点半径

$$
f_{\text{radius}}(\widetilde{\mathbf{F}}_t)
=
\sqrt{\frac{\sum_{r,c} p_{r,c}\left[(r-\mu_r)^2 + (c-\mu_c)^2\right]}{\sum_{r,c} p_{r,c}} }.
$$

### 4.5 中心-边缘对比

设中心区域均值为 $\bar{F}_{center}$，边缘区域均值为 $\bar{F}_{border}$，则

$$
f_{\text{contrast}}(\mathbf{F}_t)
=
\bar{F}_{center} - \bar{F}_{border}.
$$

### 4.6 峰值到达时间

对窗口内序列 $\{f_{\text{raw-sum}}(\mathbf{F}_t)\}_{t=1}^{10}$，记峰值索引为 $t^\star$，则

$$
f_{\text{rise}} = \frac{t^\star - 1}{T - 1}.
$$

这些特征在代码中来自：
- `frame_physics_features`
- `window_temporal_features`

---

## 5. Raw-input scientific main model

scientific main model 只使用原始时空张量，不显式输入 hand-crafted physical features。  
其 detection 路径可写为

$$
p_i^{det} = \sigma\!\left(g_{\theta}(\mathbf{W}_i)\right),
$$

其中 $g_\theta$ 表示由 2D frame encoder、temporal modeling 与 attention pooling 组成的检测网络。

在 size-aware coarse depth 路径中，先由 raw-input size router 预测 size 分布

$$
\boldsymbol{\pi}_i = \text{softmax}\!\left(h_{\phi}(\mathbf{W}_i)\right),
$$

再将 depth 判别写成 size-conditioned experts：

$$
\mathbf{z}_i^{(k)} = e_k\!\left(q_{\psi}(\mathbf{W}_i)\right), \quad k = 1,\ldots,7,
$$

其中 $e_k(\cdot)$ 表示第 $k$ 个 size expert。  
若使用真实 route，则

$$
\mathbf{z}_i^{gt} = \mathbf{z}_i^{(s_i)}.
$$

这就是为什么本文强调 depth 不是普通共享头，而是 **size-aware conditional discrimination**。

---

## 6. Deployment enhancement model

最终部署增强模型在 raw tensor 外，额外引入结构化物理特征向量 $\mathbf{x}_i^{phys}$，用于提升真实 predicted-route 条件下的鲁棒性，而不是用于证明 raw-input 自动学习。

其编码过程可写为

$$
\mathbf{h}_i^{raw} = f_{raw}(\mathbf{W}_i^{raw}, \mathbf{W}_i^{norm}, \Delta \mathbf{W}_i),
$$

$$
\mathbf{h}_i^{phys} = f_{phys}(\mathbf{x}_i^{phys}),
$$

$$
\mathbf{h}_i^{hyb} = f_{fusion}\!\left([\mathbf{h}_i^{raw}, \mathbf{h}_i^{phys}, \mathbf{h}_i^{raw} \odot \mathbf{h}_i^{phys}] \right).
$$

其中 $\odot$ 表示逐元素乘积。

---

## 7. Route-aware depth 训练目标

统一层级反演器同时使用多种 route：

### 7.1 GT route

$$
\mathbf{z}_i^{gt} = e_{s_i}(\mathbf{h}_i^{depth}).
$$

### 7.2 Soft route

$$
\mathbf{z}_i^{soft} = \sum_{k=1}^{7} \pi_{i,k}\,\mathbf{z}_i^{(k)}.
$$

### 7.3 Hard predicted route

$$
\hat{s}_i = \arg\max_k \pi_{i,k}, \qquad
\mathbf{z}_i^{hard} = e_{\hat{s}_i}(\mathbf{h}_i^{depth}).
$$

### 7.4 Top-2 truncated route

$$
\mathbf{z}_i^{top2} = \sum_{k \in \text{Top2}(\boldsymbol{\pi}_i)} \tilde{\pi}_{i,k}\,\mathbf{z}_i^{(k)},
$$

其中 $\tilde{\pi}_{i,k}$ 为 top-2 截断后重新归一化的概率。

---

## 8. 总损失函数

统一层级反演器的训练目标可写为

$$
\mathcal{L}
=
\mathcal{L}_{size-ce}
\lambda_{ord}\mathcal{L}_{ord}
\lambda_{reg}\mathcal{L}_{reg}
\lambda_{nbr}\mathcal{L}_{nbr}
\lambda_{gt}\mathcal{L}_{depth}^{gt}
\lambda_{soft}\mathcal{L}_{depth}^{soft}
\lambda_{hard}\mathcal{L}_{depth}^{hard}
\lambda_{top2}\mathcal{L}_{depth}^{top2}
\lambda_{kl}\mathcal{L}_{kl}.
$$

对应代码中的默认权重为：

| 项目 | 默认值 |
|---|---:|
| $\lambda_{ord}$ | 0.15 |
| $\lambda_{reg}$ | 0.20 |
| $\lambda_{nbr}$ | 0.20 |
| $\lambda_{gt}$ | 0.60 |
| $\lambda_{soft}$ | 0.90 |
| $\lambda_{hard}$ | 0.25 |
| $\lambda_{top2}$ | 0.15 |
| $\lambda_{kl}$ | 0.10 |

其中：
- $\mathcal{L}_{size-ce}$：size classification cross-entropy
- $\mathcal{L}_{ord}$：size ordinal BCE
- $\mathcal{L}_{reg}$：size Smooth L1 regression
- $\mathcal{L}_{nbr}$：size neighbor soft-target KL
- $\mathcal{L}_{depth}^{gt/soft/hard/top2}$：不同路由条件下的 depth classification loss
- $\mathcal{L}_{kl}$：soft route 与 GT route 的 KL 一致性约束

---

## 9. 评价指标

### 9.1 Balanced accuracy

对 coarse depth 三分类，

$$
\text{BAcc} = \frac{1}{3}\sum_{c=1}^{3}\frac{TP_c}{TP_c + FN_c}.
$$

### 9.2 Mean absolute error

$$
\text{MAE} = \frac{1}{N}\sum_{i=1}^{N} \left|\hat{s}_i - s_i\right|.
$$

### 9.3 Top-$k$ accuracy

$$
\text{Top-}k = \frac{1}{N}\sum_{i=1}^{N}\mathbb{I}\left(y_i \in \text{Top-}k(\hat{\mathbf{p}}_i)\right).
$$

---

## 10. 关键参数表

| 类别 | 参数 | 数值 |
|---|---|---:|
| 输入协议 | 窗口长度 $T$ | 10 |
| 输入协议 | 窗口步长 $\Delta$ | 2 |
| 输入协议 | 帧尺寸 | $12 \times 8$ |
| 任务轴 | size levels | 7 |
| 任务轴 | depth levels | 6 |
| coarse depth | class count | 3 |
| 模型结构 | frame feature dim | 24 |
| 模型结构 | temporal channels | 48 |
| 模型结构 | temporal blocks | 3 |
| 模型结构 | tabular hidden dim | 64 |
| 训练 | dropout | 0.28 |
| 训练 | learning rate | $2\times10^{-4}$ |
| 训练 | weight decay | $1\times10^{-3}$ |
| 训练 | batch size | 48 |
| 训练 | epochs | 120 |
| 训练 | patience | 24 |

---

## 建议写法
公式部分不需要把所有神经网络层都写成逐层数学式。  
TBME 更需要的是：

1. 输入张量定义清楚  
2. 标签协议清楚  
3. route-aware 逻辑清楚  
4. loss 组合清楚  
5. 核心物理特征的定义清楚
