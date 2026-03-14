# 可视化画廊（TBME 候选版）

这套画廊不是主文最终图，而是帮助选择 `TBME` 风格数据图的候选视图。
组织思路参考了 HMIL 的数据图习惯：
- 先给结构清楚的多面板图
- 再给嵌入空间图
- 再给分布图和热图
- 每张图只回答一个明确问题

## 当前生成的候选图

1. `G1_embedding_gallery_tsne_umap_pca`
   - 内容：raw scientific model 的 t-SNE / UMAP / PCA 嵌入对照图
   - 用途：适合补充材料，服务“网络确实学到结构”这一句

2. `G2_feature_distributions_by_depth`
   - 内容：峰值、空间熵、热点半径在不同 coarse depth 下的分布
   - 用途：适合主文或补充材料，作为机制与解释的桥梁

3. `G3_route_heatmaps`
   - 内容：按 `size x depth` 组织的正确率和深度置信度热图
   - 用途：很适合主文 Fig.8 或补充材料，用于展示部署增强后的系统景观

4. `G4_probability_structure_views`
   - 内容：`p_deep` 的分布以及幅值-扩散二维结构视图
   - 用途：适合作为 Supplement 的漂亮数据图

5. `G5_probe_phase_summary`
   - 内容：probe 与 phase occlusion 的紧凑汇总图
   - 用途：适合替换现有 explainability 图中过于拥挤的子图

## 使用建议

- `G1` 更像 HMIL Fig.5 的角色：展示表征空间，而不是主结果。若视觉上必须三选一，优先比较 `UMAP` 与 `PCA`，`t-SNE` 可作为参考而非唯一版本。
- `G2` 和 `G3` 更适合主文，因为它们兼具物理可解释性和工程说服力。
- `G4` 和 `G5` 更适合作为补充材料或 explainability 备选图。