# 主文图重画说明（TBME 版）

## 总原则
不是所有图都适合“手绘重画”。  
当前主文图应分成两类：

1. **概念/方法图**
- 适合重新设计
- 重点是版式、结构、信息层级、审美

2. **结果/统计图**
- 不适合纯手绘替代
- 应该基于真实数据重新排版、重新配色、重新组合
- 不能把数据图画成示意图

## 最重要的判断
如果我们是冲 `TBME`：
- `Fig. 1 / Fig. 2 / Fig. 3 / Fig. 7` 最值得重画
- `Fig. 4 / Fig. 5 / Fig. 6 / Fig. 8 / Fig. 10` 更适合“数据重绘”
- `Fig. 9` 可以在保留真实截图的基础上做版式美化

---

## Fig. 1 引言总图
### 建议：必须重画

### 目标
让读者在 10 秒内理解：
- 为什么术中肺结节定位难
- 现有方案为什么不够
- 为什么触觉恢复是合理方向
- 为什么这是病态逆问题

### 版式
- 4 panel 横向大图
- `A Clinical challenge`
- `B Existing methods and limitations`
- `C Ill-posed inverse problem`
- `D Our study roadmap`

### 绝对不要
- 海报风
- 3D 发光肺
- 复杂人物
- 密密麻麻小字

### 风格关键词
- white background
- rounded containers
- thin gray separators
- teal/blue/gray base
- orange highlight only for lesion and key path
- HMIL/TBME layout logic

---

## Fig. 2 方法总图
### 建议：必须重画

### 目标
把 “sensor -> experiment -> preprocessing -> hierarchical model” 一眼讲清楚。

### 必须保留的模块
1. 柔性触觉阵列与肺表面接触
2. 离体猪肺实验与 `7 size x 6 depth x 3 repeats`
3. `10-frame window` 数据处理
4. `Detection -> Size -> Depth`
5. raw scientific line
6. deployment enhancement line

### 最好做成
- 左到右流程图
- 每个模块一个浅色圆角框
- 箭头只保留主路径
- 细节放图注，不堆在图里

---

## Fig. 3 机制分析 + XGBoost 解释
### 建议：重组 + 部分重画

### 这里不要纯手绘替换
因为这是“有数据含义”的图。

### 正确做法
把它重做成更整齐的多面板组合：
- `A` feature family contribution
- `B` class-wise concept heatmap
- `C` representative tactile maps
- `D` one or two partial dependence plots

### 视觉问题怎么改
- 去掉杂乱标题
- 统一字体、字号、线宽
- 热图与条形图配色统一
- 代表性 tactile map 用同一色条和同一裁切比例

### 目标
让读者一眼看到：
- depth 信息存在
- 而且主要不是靠 raw peak

---

## Fig. 4 Detection 结果图
### 建议：不要手绘，直接数据重绘

### 应保留
- ROC
- PR 或 AP/F1 柱图
- NN vs XGBoost

### 美化方向
- 统一为 TBME 风格色板
- 减少网格线
- 线宽加粗
- 图例外置或上移
- 直接标关键数值

### 目的
这是第一张“硬结果图”，必须干净、可信、专业。

---

## Fig. 5 Size 结果图
### 建议：不要手绘，直接数据重绘

### 应保留
- top-1
- top-2
- MAE
- 三个模型对比

### 最好形式
- 左：分类指标 grouped bar
- 右：MAE 单独小图

### 重点
要让图自己讲出这句话：
- raw-input size 学得不错
- 但 standalone regression 还不是最优

---

## Fig. 6 Raw-input depth 主结果图
### 建议：重组，不要手绘

### 应保留的对比
- majority baseline
- XGBoost
- raw shared-head failure
- raw size-routed success
- GT-route confusion

### 最好形式
- `A` balanced accuracy comparison
- `B` GT-route confusion matrix

### 图里要讲清楚
- 为什么普通 head 失败
- 为什么 size-aware 才成立

---

## Fig. 7 Raw-input explainability 主图
### 建议：重画排版 + 真实数据子图

### 适合做成三联图
- `A` latent probe family comparison
- `B` hard-pair examples
- `C` phase occlusion

### 重点
这张图不是为了“好看”，而是为了证明：
- raw-input 模型不是只在黑箱乱猜
- 它自动编码了部分和 XGBoost 一致的物理结构

### 视觉上最该改的
- hard-pair 的图片布局太乱
- 标题字太多
- 不同子图风格不统一

---

## Fig. 8 Deployment enhancement 图
### 建议：不要手绘，直接数据重绘

### 应保留
- old predicted-route chain
- unified predicted-route result
- XGBoost reference
- 可选 confusion

### 目标
明确说明：
- 这张图是讲 deployment robustness
- 不是讲 raw-input 自动学习

---

## Fig. 9 GUI 图
### 建议：真实截图 + 排版美化

### 不建议完全重画
因为这张图代表系统真的跑起来了。

### 更好的做法
- 保留真实界面截图
- 外围做整洁论文排版
- 用 callout 标出：
  - tactile map
  - detection probability
  - size distribution
  - depth distribution

---

## Fig. 10 延迟图
### 建议：不要手绘，直接数据重绘

### 应保留
- model-only vs end-to-end
- detection vs full chain
- CPU vs GPU

### 重点
这张图不是为了证明 “NN everywhere faster”
而是为了说明：
- XGBoost 更适合机制基线
- raw-input NN 更适合 detection-first 在线主线

---

## 最终建议的重画优先级
### 第一优先级
1. Fig. 1
2. Fig. 2
3. Fig. 7

### 第二优先级
4. Fig. 3
5. Fig. 9

### 第三优先级
6. Fig. 4
7. Fig. 5
8. Fig. 6
9. Fig. 8
10. Fig. 10

第三优先级不是“不重要”，而是它们更适合做成高质量数据重绘，而不是人工重画概念图。
