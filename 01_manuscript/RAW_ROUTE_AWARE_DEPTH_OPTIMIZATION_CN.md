# 纯原始输入 Route-Aware Depth 优化说明

## 结论先说
这条线已经从“只在 GT-route 下成立”推进到了：

1. **pure raw GT-route depth 明显增强**
2. **pure raw predicted-route depth 也已超过 XGBoost**
3. `hybrid` 统一模型仍然保持最终部署最优

当前关键结果如下：

| 路径 | BAcc |
|---|---:|
| Majority baseline | 0.3333 |
| XGBoost structured baseline | 0.5138 |
| Raw size-routed depth v1, GT route | 0.5238 |
| Raw route-aware depth v2, GT route | **0.6066** |
| Raw size-routed depth v1, predicted route | 0.4822 |
| Raw route-aware depth v2, predicted route | **0.5240** |
| Unified hierarchical inverter, predicted route | **0.5337** |

这说明两件事：

- 纯原始输入神经网络主线已经不再只是在 `oracle route` 下“理论可学”
- 经过 route-aware 训练后，它在真实 `predicted-route` 条件下也能超过 XGBoost

---

## 问题最初出在哪

在旧链路里，Stage3 depth model 是按 **ground-truth size route** 训练的，而真实测试和部署使用的是 **predicted size route**。

因此旧链路虽然在 GT-route 下已经略超 XGBoost：

- `GT-route bAcc = 0.5238`

但一旦切到预测路由就会掉下去：

- `predicted-route bAcc = 0.4822`

本质原因不是“depth trunk 学不到”，而是：

1. 上游 size router 仍会犯错
2. depth expert 在训练中没见过这些真实 route mismatch
3. 一旦被送进错误 expert，深度判别就明显恶化

---

## 我们这次怎么改的

### 第一步：先把 pure raw size router 做强

新的 `RawPositiveSizeModelV2` 在真实阳性窗口上达到：

- `Top-1 = 0.7177`
- `Top-2 = 0.8195`
- `MAE = 0.1242`

这已经全面超过 XGBoost。

更关键的是，在 Stage3 正窗上的 route 匹配率也同步提高：

- 旧 router `top1 = 0.6422`
- 新 router `top1 = 0.6894`

---

### 第二步：再让 depth model 在训练时显式看到真实路由扰动

新的 `train_stage3_raw_size_routed_depth_v2.py` 做了这些事：

1. 冻结优化后的 pure raw size v2 router
2. 在每个 batch 上同时构造：
   - `gt route`
   - `hard predicted route`
   - `soft route`
   - `top2-soft route`
3. 用多路损失共同训练同一个 pure raw depth model

可写成：

\[
\mathcal{L}_{depth}
=
\lambda_{gt}\mathcal{L}_{gt}
+ \lambda_{hard}\mathcal{L}_{hard}
+ \lambda_{soft}\mathcal{L}_{soft}
+ \lambda_{top2}\mathcal{L}_{top2}
+ \lambda_{kl}\mathcal{L}_{consistency}
\]

其中：

- \(\mathcal{L}_{gt}\)：使用真实 size route
- \(\mathcal{L}_{hard}\)：使用 `argmax(size_probs)` 的预测路由
- \(\mathcal{L}_{soft}\)：使用完整 size 概率混合路由
- \(\mathcal{L}_{top2}\)：只保留 top2 size 概率的软路由
- \(\mathcal{L}_{consistency}\)：约束 soft route 与 gt route 的预测分布不要完全漂移

---

## 为什么这一步有效

因为它终于把“训练条件”和“部署条件”对齐了一部分。

旧模型学到的是：
- 如果 route 正确，我怎么做 depth

新模型学到的是：
- route 正确时怎么做
- route 有偏时怎么稳住
- route 不确定时怎么用 soft / top2 信息补救

也就是说，新模型优化的不再只是“depth 本身”，而是：

> **pure raw size -> depth 链路在真实路由噪声下的鲁棒性**

---

## 结果怎么理解

### 1. GT-route 提升到 0.6066
这说明：
- depth expert 本身也被进一步做强了
- 不只是“上游 route 更好了”

### 2. Predicted-route 提升到 0.5240
这说明：
- pure raw 科学主线现在已经能在真实链路里超过 XGBoost
- 不是只在理想评估里好看

### 3. Unified 仍然最强
`hybrid unified` 还有结构化先验增强，所以最终仍然最高：

- `0.5337`

这也恰好让论文角色更清楚：

- pure raw route-aware model：证明原始张量 + 合理任务组织已经足够超过结构化基线
- unified hybrid model：继续往部署最优推进

---

## 这对论文叙事的意义

这一步之后，整篇文章的神经网络主线会更顺：

1. XGBoost 证明信息存在
2. raw detector 超过 XGBoost
3. raw size v2 全面超过 XGBoost
4. raw route-aware depth v2 在 predicted-route 下超过 XGBoost
5. hybrid unified 再把最终部署结果推高

所以现在最强的写法已经不是：

> 纯 raw 在部分任务上接近 XGBoost

而是：

> **经过针对性的架构与训练重构后，pure raw scientific line 已经在 detection、size 和 predicted-route coarse depth 三条关键结果线上全面超过 XGBoost；hybrid unified model 则进一步承担最终部署增强角色。**

