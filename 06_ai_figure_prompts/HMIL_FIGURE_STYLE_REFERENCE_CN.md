# HMIL 绘图风格参考与迁移说明

参考文献：
- `HMIL: Hierarchical Multi-Instance Learning for Fine-Grained Whole Slide Image Classification`
- 文件位置：`C:\Users\SWH\Desktop\HMIL_Hierarchical_Multi-Instance_Learning_for_Fine-Grained_Whole_Slide_Image_Classification.pdf`

本文件不讨论 HMIL 的算法内容，而是提炼其**图表表达风格**中适合迁移到我们 TBME 稿件的部分。

---

## 1. HMIL 这篇图风格最值得学的地方

### 1.1 结构先行，而不是装饰先行
HMIL 的图不是“好看优先”，而是“结构清楚优先”。

它的典型特征是：
- 白色背景
- 强结构分区
- 先有大框架，再放小元素
- 箭头和模块决定阅读顺序
- 颜色只承担“类别区分”和“重点强调”的功能

这很适合我们现在的论文，因为我们的故事本身就有明显层级：
- 临床问题
- 结构化基线
- 神经网络
- 可解释性闭环

---

### 1.2 喜欢用“分组容器”
HMIL 的总图和方法图里，经常把相关模块包在：
- 浅色填充的矩形框
- 虚线或圆角分组框
- 明确标题的小容器

这种做法很重要，因为它让读者一眼看出：
- 哪些模块属于同一分支
- 哪些是 coarse branch
- 哪些是 fine branch
- 哪些是比较对象

迁移到我们这里，最适合用在：
- `Detection / Size / Depth` 三层任务
- `raw / normalized / delta / tabular` 四类输入
- `mechanism / XGBoost / neural / interpretability` 四段研究路径

---

### 1.3 箭头少而准
HMIL 的流程图里，箭头不会满天飞。

它的箭头特点：
- 主流程箭头最粗最清楚
- 支线箭头较细
- 不做复杂交叉
- 尽量从左到右、从上到下

这对我们也很关键。  
我们的图如果把路由、分支、阶段、解释模块全画进去，很容易乱。

所以建议：
- 每张图只保留**一条主叙事箭头**
- 其余关系用容器和邻接关系表达

---

### 1.4 文本少，说明放在图注
HMIL 的图内部文字并不多。

通常只有：
- 模块名
- 分支名
- 很短的类别名
- 必要的坐标/标签

它不会把一大段解释塞进图里。  
真正详细的信息放在：
- caption
- 正文

这很值得学。

对我们来说：
- 图里写 `Detection -> Size -> Depth` 可以
- 图里写整段机制解释就不合适

---

### 1.5 配色克制
HMIL 的颜色不是“漂亮型”，而是“可读型”。

特点：
- 白底
- 浅蓝、浅灰、浅紫、浅绿作为模块区分
- 少量高饱和颜色只用于重点
- 同一张图里颜色含义稳定

我们可以借鉴成：
- 蓝绿灰做系统与流程
- 橙色只标结节、风险、关键路径
- depth 相关可以用略深一点的青蓝或紫蓝
- 不要红绿同时大面积对冲

---

### 1.6 多面板拼图非常规整
HMIL 的结果图和 t-SNE 图很规整，原因不是内容简单，而是：
- 子图尺寸统一
- 边距统一
- 组间有明显留白
- 图例放在边上，不挤主图

这点对我们之后做：
- hard-pair 图
- explainability 汇总图
- XGBoost 解释图

都非常重要。

---

## 2. 我们最适合模仿 HMIL 的不是“像素风格”，而是“组织方式”

### 不建议机械模仿的部分
- 病理切片的彩色 patch 风格
- 过多密集小方块
- 特定病理类别颜色体系

### 最值得模仿的部分
1. **大图分组方式**
2. **模块和箭头的秩序**
3. **图中文字密度控制**
4. **多面板留白**
5. **图例和主图分离**

---

## 3. 对我们论文的具体迁移建议

### 3.1 引言总图
应模仿 HMIL Fig.1 的思路：
- 不是画成“插画海报”
- 而是画成“对比 + 研究路径”的信息图

建议：
- `A/B/C/D` 四个 panel
- 每个 panel 放在清晰容器内
- 顶部可有简短 panel title
- 主路径从左到右

最像 HMIL 的地方应该是：
- 左侧对比传统方法和我们的体系
- 右侧突出我们的方法链

---

### 3.2 方法总图
应模仿 HMIL Fig.2 的思路：
- 先画大框架
- 再在框架内部放局部模块
- 每个分支颜色不同但不过饱和

对我们来说最适合的结构是：
- `Sensor System`
- `Experimental Preparation`
- `Data Processing`
- `Hierarchical Model`

每个区块再放 2 到 4 个关键元素，不要塞满。

---

### 3.3 模型架构图
HMIL Fig.2 对我们最有参考价值的是：
- 分支清楚
- 分层清楚
- 相同类型模块重复出现时样式一致

迁移到我们这里：
- 输入分支统一样式
- 编码器统一样式
- 输出头统一样式
- `Size-routed depth experts` 用虚线框或独立分组框强调

---

### 3.4 解释性图
HMIL 的 t-SNE 图和消融图给我们的启发是：
- 不要把图例塞进数据区域
- 同一类比较必须横向对齐
- 不同数据集/任务要分组呈现

所以我们之后的 explainability 图：
- `latent probe / ablation / phase occlusion / hard-pair`
最好做成 2×2 或 1×4 的规整拼图

---

## 4. 我们生成 AI 图时应该额外加的风格约束

如果你要让图更接近 HMIL 这种期刊风格，建议在 prompt 里额外加上下面这些词：

### 中文补充约束
- 使用清晰的模块分组框和圆角容器
- 面板之间留白充分
- 箭头简洁且主次分明
- 图中文字少而精炼
- 子模块大小尽量统一
- 图例放在侧边，不要遮挡主图
- 避免海报式夸张构图
- 更像期刊方法图而不是宣传海报

### English extra constraints
- use grouped containers with rounded rectangles
- maintain generous whitespace between panels
- keep arrows minimal and directional
- minimal in-figure text, concise labels only
- consistent module sizing across branches
- place legends outside the main plotting area
- avoid poster-like dramatic composition
- look like a journal methods figure, not a commercial infographic

---

## 5. 对我们当前图生成的直接建议

### 引言总图
在现有 prompt 后面追加：

`Use clear grouped containers, balanced whitespace, concise labels, and a journal-style structured layout similar to high-quality TBME or TMI methods figures. Avoid poster-like composition.`

### 方法总图
在现有 prompt 后面追加：

`Organize the four panels with clear rounded grouping boxes and a stable left-to-right reading order. Keep the figure modular, sparse, and publication-like rather than decorative.`

### 模型架构图
在现有 prompt 后面追加：

`The architecture should look like a formal journal network diagram with repeated module styles, consistent box shapes, restrained colors, and minimal explanatory text inside the figure.`

---

## 6. 一句话总结

我们要学 HMIL 的，不是“它画了什么对象”，而是：

**它如何把复杂方法和比较关系，组织成一张读者一眼能顺着读下去的论文图。**
