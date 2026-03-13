# 图表与表格初稿（V1）

## 图1 系统总体框架图
**对应文件**：[Fig_R10_system_overview.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R10_system_overview.png)  
**建议标题**：基于柔性阵列触觉传感与层级时空神经网络的术中肺结节定位与特征反演系统总体框架  
**图注建议**：系统由柔性阵列应力采集、时序窗口构建、阶段一结节检测、阶段二大小反演、阶段三大小感知深度粗分类及实时可视化界面构成。检测结果始终显示，大小与深度结果仅在概率超过阈值后输出。

## 图2 机制分析图
**对应文件**：[Fig_R11_final_pipeline.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R11_final_pipeline.png)  
**建议标题**：最终层级系统的训练与推理流程图  
**图注建议**：展示从数据协议、机制分析和 XGBoost 结构化基线，到阶段一检测器、大小学习、统一层级反演器以及实际推理界面的完整流程，强调 route-aware depth 优化与系统级性能目标。

## 图3 XGBoost 基线与特征贡献图
**对应文件**：  
- [Fig_R12_mechanism_xgboost_summary.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R12_mechanism_xgboost_summary.png)  
- [Fig_XGB_depth_explainability_tbme.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/outputs_xgboost_explainability_v1/Fig_XGB_depth_explainability_tbme.png)  
**建议标题**：结构化物理特征基线及其对深度判别的贡献  
**图注建议**：建议优先采用 TBME 风格的四联解释图。XGBoost 结果表明深度判别主要依赖 deformation-position、shape-contrast 与 spread-related 特征家族，而非单一峰值幅值，说明深度信息确实存在于结构化触觉特征中。对应的家族贡献、类别级概念热图、代表性触觉样本和 partial dependence 可以组合成一张主图，也可与机制分析汇总图配合使用。

## 图4 检测任务结果图
**对应文件**：[Fig_R1_detection_compare.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R1_detection_compare.png)  
**建议标题**：检测任务上神经网络与 XGBoost 的性能比较  
**图注建议**：阶段一 raw+delta 检测器在独立测试集上的 AUC 和 AP 均优于 XGBoost，说明原始时空窗口在结节存在判别上更适合由神经网络建模。

## 图5 大小反演结果图
**对应文件**：[Fig_R2_size_compare.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R2_size_compare.png)  
**建议标题**：不同模型在结节大小反演任务上的性能比较  
**图注建议**：比较 XGBoost、size-only 神经网络和统一层级反演器在 top-1、top-2 和 MAE 指标上的差异，说明 standalone size 最优与整条链路最优并不完全一致。

## 图6 深度粗分类结果图
**对应文件**：[Fig_R3_depth_compare.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R3_depth_compare.png)  
**建议标题**：深度粗分类从多数类基线到统一层级反演器的性能演化  
**图注建议**：展示 majority baseline、XGBoost、Stage3 GT-route、旧 predicted-route 以及统一层级反演器在深度粗分类平衡准确率上的差异，突出 route-aware 优化的重要性。

## 图7 统一层级反演器深度混淆矩阵
**对应文件**：  
- [Fig_R4_depth_confusion_hard.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R4_depth_confusion_hard.png)  
- [Fig_R5_depth_confusion_gt.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R5_depth_confusion_gt.png)  
**建议标题**：统一层级反演器在 ground-truth route 与 predicted route 条件下的深度混淆矩阵  
**图注建议**：比较两种路由条件下的深度混淆模式，以揭示真实部署中 size routing 误差对 depth experts 的影响。

## 图8 神经网络解释性图
**建议标题**：原始输入深度网络的内部表征与时间阶段解释结果  
**对应文件**：  
- [Fig_R6_hierarchical_probe_family.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R6_hierarchical_probe_family.png)  
- [Fig_R7_hierarchical_branch_ablation.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R7_hierarchical_branch_ablation.png)  
- [Fig_R8_hierarchical_phase_occlusion.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R8_hierarchical_phase_occlusion.png)  
- [Fig_R9_hierarchical_hard_pairs.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R9_hierarchical_hard_pairs.png)  
**图注建议**：包含统一层级反演器的 latent probe、branch ablation、hard-pair analysis 与 phase occlusion。probe 说明最终模型内部编码了超出 size identity 的深度相关结构；branch ablation 说明 `shape` 与 `tabular` 分支对最终性能最关键；hard-pair 与 phase occlusion 则进一步说明模型并非仅依赖峰值幅值，但其时序注意力仍偏向 peak neighborhood。

## 图9 测试集条件矩阵大图
**对应文件**：[testset_best_grid_hierarchical.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/app/replay_snapshot_output/testset_best_grid_hierarchical.png)  
**建议标题**：测试集各 size-depth 条件下的最佳预测帧联合矩阵图  
**图注建议**：按深度为行、大小为列排列 42 个条件，每个子图显示对应条件下模型最高置信窗口的代表性应力分布及预测结果，用于展示系统在完整实验设计矩阵上的整体表现。

## 图10 主程序实时界面图
**对应文件**：[replay_snapshot_hierarchical.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/app/replay_snapshot_output/replay_snapshot_hierarchical.png)  
**建议标题**：统一层级反演器在主程序中的实时可视化界面  
**图注建议**：界面同步显示原始触觉图、特征图、AI检测结果、概率曲线、大小分布和深度分布，为术中交互提供实时量化决策支持。

## 图11 机制分析与 XGBoost 解释汇总图
**对应文件**：[Fig_R12_mechanism_xgboost_summary.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R12_mechanism_xgboost_summary.png)  
**建议标题**：机制分析与 XGBoost 深度解释的汇总图  
**图注建议**：当前版本已将 TBME 风格的 XGBoost explainability 主图并入右侧主面板，并与正负平均触觉图、条件趋势图共同展示，用于说明深度主要与扩散、形态和分布复杂度相关，而不是简单峰值衰减。

## 图12 统一层级反演器可解释性汇总图
**对应文件**：[Fig_R13_hierarchical_explainability_summary.png](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/experiments/paper_figures_v2/Fig_R13_hierarchical_explainability_summary.png)  
**建议标题**：统一层级反演器的可解释性汇总图  
**图注建议**：整合 latent probe、branch ablation、phase occlusion 与 hard-pair 四类结果，用于说明最终主文模型不仅能够实现路由深度判别，还确实编码并利用了超出大小身份本身的深度相关结构。

## 表1 数据集与任务协议
**建议内容**：
- 42 个 size-depth 条件
- 3 次重复
- 帧分辨率 12×8
- 窗口长度 10
- stride 2
- detection 与 inversion 使用的划分协议

## 表2 检测任务结果
**建议列**：
- Model
- Test AUC
- Test AP
- Test F1

## 表3 大小反演任务结果
**建议列**：
- Model
- Top-1
- Top-2
- MAE (cm)
- Median AE (cm)

## 表4 深度粗分类结果
**建议列**：
- Model
- Route Mode
- Accuracy
- Balanced Accuracy

## 表5 可解释性分析总结
**建议列**：
- Analysis
- Main Finding
- Interpretation
- Limitation
