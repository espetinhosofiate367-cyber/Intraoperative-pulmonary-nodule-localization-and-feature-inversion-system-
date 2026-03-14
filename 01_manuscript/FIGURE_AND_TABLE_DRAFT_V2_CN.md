# 图表与表格草案（V2）

## 这版图表组织原则
本版图表按照新的论文逻辑分成两层：

1. **科学主线图**
- 用来证明：结构化信息存在、raw-input 神经网络可学、并且内部自动编码了部分物理特征。

2. **部署增强图**
- 用来证明：在真实 `predicted-route` 条件下，hybrid 统一层级反演器能把系统做得更稳。

换句话说：
- **主文图**优先服务“科学问题”
- **补充图/后置图**再服务“系统落地”

---

## 一、建议主文图

### Fig. 1 引言总图
**用途**：临床痛点、现有方法局限、病态逆问题、本文研究路径  
**建议状态**：主文必须保留  
**建议风格**：HMIL/TBME 风格的四面板大图

---

### Fig. 2 方法总图
**用途**：传感器系统、实验准备、数据处理、层级模型  
**建议状态**：主文必须保留  
**建议说明**：这里要强调 `Detection -> Size -> Depth` 的层级组织

---

### Fig. 3 机制分析 + XGBoost 解释总图
**对应文件**：
- [Fig_R12_mechanism_xgboost_summary.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R12_mechanism_xgboost_summary.png)
- [Fig_XGB_depth_explainability_tbme.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/outputs_xgboost_explainability_v1/Fig_XGB_depth_explainability_tbme.png)

**用途**：
- 先证明数据里有 detection / size / depth 信息
- 再说明深度主要依赖哪些结构化物理特征
- 给 raw-input 神经网络的解释性提供锚点

**建议标题**：
结构化机制分析与 XGBoost 基线共同揭示了可学习的检测、大小及粗深度信息

---

### Fig. 4 Detection 结果图
**对应文件**：
- [Fig_R1_detection_compare.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R1_detection_compare.png)

**用途**：
- 证明 raw-input 神经网络在主任务 detection 上优于 XGBoost

**建议标题**：
raw-input 时空神经网络在结节探测任务上优于结构化基线

---

### Fig. 5 Raw-input size 结果图
**对应文件**：
- [Fig_R2_size_compare.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R2_size_compare.png)

**用途**：
- 证明原始时空张量本身已经包含强大小信息
- 说明 raw-input 模型在分类上已接近/部分超过 XGBoost
- 同时诚实展示 size regression 仍未全面超过结构化基线

**建议标题**：
raw-input 神经网络可学习强大小信息，但 standalone size 最优仍由结构化基线保持

---

### Fig. 6 Raw-input depth 主结果图
**建议内容**：将下面内容整合成一个多面板主图
- raw-input 普通 depth head 失败
- raw-input size-routed depth 成功
- XGBoost depth baseline
- majority baseline

**建议可用文件**：
- [Fig_R3_depth_compare.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R3_depth_compare.png)
- [Fig_R5_depth_confusion_gt.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R5_depth_confusion_gt.png)

**用途**：
- 证明深度不是不可学
- 证明必须显式处理 `size-depth` 耦合
- 证明 raw-input size-routed depth 已可达到/略超 XGBoost

**建议标题**：
只有在 size-aware 架构下，raw-input 神经网络才能稳定恢复 coarse depth 信息

---

### Fig. 7 Raw-input 神经网络可解释性主图
**建议重新组合为一张总图**

建议包含：
1. latent probe family comparison  
2. hard-pair examples  
3. phase occlusion  

**对应素材来源**：
- raw-input size-routed 模型 explainability 结果目录  
- 如需辅助，可引用现有层级图中对应子图风格

**用途**：
- 这是“raw-input 神经网络是否自动编码了物理特征”的主证据图
- 必须服务科学主线，而不是服务 hybrid 部署模型

**建议标题**：
raw-input 神经网络内部自动编码了部分与大小和深度相关的物理结构

---

### Fig. 8 部署增强结果图
**建议内容**：
- predicted-route depth 从旧链路到 unified hybrid model 的提升
- 可选附一个 predicted-route confusion matrix

**对应文件**：
- [Fig_R4_depth_confusion_hard.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R4_depth_confusion_hard.png)
- 以及 depth compare 图中的 unified 指标

**用途**：
- 说明 hybrid unified model 的角色是“部署增强”
- 不是机制证明，而是系统性能增强

**建议标题**：
部署导向的统一层级反演器在真实 predicted-route 条件下进一步提升系统级 depth 性能

---

### Fig. 9 实时界面图
**对应文件**：
- [replay_snapshot_hierarchical.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/app/replay_snapshot_output/replay_snapshot_hierarchical.png)

**用途**：
- 收束全文
- 说明这不是停留在离线结果，而是能集成到实时系统

**建议标题**：
检测优先、反演增强的实时术中触觉导航界面

---

### Fig. 10 延迟与部署口径对比图
**对应文件**：
- [latency_compare_tbme.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/03_results_core/latency_benchmark_v1/latency_compare_tbme.png)

**用途**：
- 说明为什么 XGBoost 更适合作为结构化机制基线，而不是最终实时主线
- 强调 raw-input 神经网络在 detection 的 end-to-end 口径下更自然、更快
- 同时诚实保留 XGBoost 在纯 CPU 表格推理本体上的速度优势

**建议标题**：
结构化特征提取增加了实时链路负担，而 raw-input 神经网络更适合检测优先的在线部署路径

---

## 二、建议补充材料图

### Supplementary Fig. S1
XGBoost explainability 详细四联图  
**用途**：展开主文 Fig. 3 的细节

### Supplementary Fig. S2
原始全条件矩阵图  
**对应文件**：
- [testset_best_grid_hierarchical.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/app/replay_snapshot_output/testset_best_grid_hierarchical.png)

### Supplementary Fig. S3
raw-input 模型的更多 hard-pair 对

### Supplementary Fig. S4
raw-input 模型的更多 phase occlusion 细图

### Supplementary Fig. S5
hybrid unified model 的 branch ablation

### Supplementary Fig. S6
hybrid unified model 的 latent probe family 图

说明：
- `hybrid` 的解释性细节更适合放补充材料
- 主文里只保留它的系统增强角色和一张部署增强结果图

---

## 三、神经网络图的角色划分

### A. raw-input 神经网络图
这些图回答：
- 原始张量能不能学 detection？
- 原始张量能不能学 size？
- 在 size-aware 条件下能不能学 coarse depth？
- 它内部有没有自动编码物理特征？

这些图应该构成**论文主科学证据**。

### B. hybrid unified 模型图
这些图回答：
- 在真实 predicted-route 条件下，系统能不能更稳？
- 如何把 raw 学习与结构化先验融合成部署模型？

这些图应该构成**论文的系统落地和部署增强证据**。

---

## 四、建议表格

### Table 1 数据协议与任务定义
建议内容：
- 42 个 size-depth 条件
- 3 次重复
- 12×8 触觉帧
- 窗口长度 10
- stride 2
- detection / size / depth 的数据划分协议

### Table 2 结构化基线结果
建议列：
- Task
- XGBoost metric
- Majority/Dummy baseline

### Table 3 Raw-input 神经网络结果
建议列：
- Detection AUC / AP / F1
- Size top1 / top2 / MAE
- Depth bAcc

### Table 4 部署增强结果
建议列：
- Model
- Route mode
- Depth bAcc
- Relative gain over XGBoost

### Table 5 可解释性结果总结
建议列：
- Analysis
- Raw-input model finding
- Hybrid model finding
- Interpretation
- Limitation

---

## 五、一句话总原则
如果一张图是在回答：
- “原始时空张量自己能不能学出来？”  
它就归到 **raw-input scientific line**。

如果一张图是在回答：
- “最终系统在真实部署里能不能更稳？”  
它就归到 **hybrid deployment line**。
