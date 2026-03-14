# 已核验参考文献清单（TBME 版）

## 结论
当前 `Overleaf` 草稿里的核心参考文献**没有发现假文献**。  
但我已经修正了几条 `references.bib` 中不够准确的元数据，主要涉及：

- 作者名单写错
- 作者顺序不准
- 只有 DOI / PMID，未给出直接链接

对应已修正文件：
- [references.bib](/C:/Users/SWH/Desktop/GitHub_Docs_Package/Code_Archive/paper_flexible_sensor_lung_nodule_localization/tbme_submission_release_v1/08_overleaf_draft_v1/references.bib)

## 已核验条目与链接

### 1. 胸外科定位背景

#### `khereba2018preoperative`
- 标题：`Preoperative versus intraoperative localization of pulmonary nodules`
- 期刊：`The Journal of Thoracic and Cardiovascular Surgery`
- DOI：https://doi.org/10.1016/j.jtcvs.2018.07.090
- 说明：真实条目；原 `bib` 中作者写法不准，已改为 Crossref 对应信息。

#### `tang2022intraoperative`
- 标题：`Intraoperative identification of pulmonary nodules during minimally invasive thoracic surgery: a narrative review`
- PubMed：https://pubmed.ncbi.nlm.nih.gov/36330174/
- PMC：https://pmc.ncbi.nlm.nih.gov/articles/PMC9622445/
- 说明：真实条目；当前可作为术中定位方法综述的重要背景文献。

#### `wang2025anchored`
- 标题：`CT-Guided Anchored Needle Versus Hook-Wire Localization for Pulmonary Nodules: A Meta-Analysis`
- PubMed：https://pubmed.ncbi.nlm.nih.gov/41247803/
- 说明：真实条目；可用于支撑钩线/锚定针术前定位存在并发症和流程负担的论述。

### 2. 微创触觉 / 触觉定位背景

#### `perri2009robot`
- 标题：`Robot-assisted tactile sensing for minimally invasive tumor localization`
- DOI：https://doi.org/10.1177/0278364909101136
- 说明：真实条目；原 `bib` 中作者不完整，已按 Crossref 修正。

#### `trejos2013integration`
- 标题：`Integration of force reflection with tactile sensing for minimally invasive robotics-assisted tumor localization`
- DOI：https://doi.org/10.1109/TOH.2012.64
- 说明：真实条目；原 `bib` 作者列表与 DOI 元数据不一致，已按 Crossref 修正。

#### `ly2025acoustic`
- 标题：`Tactile force sensor based on a modified acoustic reflection principle for intraoperative tumor localization in minimally invasive surgery`
- DOI：https://doi.org/10.1007/s11548-025-03511-0
- 说明：真实条目；作者名单已按 Crossref 补全。

#### `huang2020tactile`
- 标题：`Tactile Perception Technologies and Their Applications in Minimally Invasive Surgery: A Review`
- DOI：https://doi.org/10.3389/fphys.2020.611596
- PubMed：https://pubmed.ncbi.nlm.nih.gov/33424634/
- PMC：https://pmc.ncbi.nlm.nih.gov/articles/PMC7785975/
- 说明：真实条目；原 `bib` 作者写法错误，已修正。

#### `bewley2022tactile`
- 标题：`Tactile Sensing for Minimally Invasive Surgery: Conventional Methods and Potential Emerging Tactile Technologies`
- DOI：https://doi.org/10.3389/frobt.2021.705662
- 说明：真实条目；原 `bib` 作者写法错误，已修正。

### 3. 方法学基线

#### `chen2016xgboost`
- 标题：`XGBoost: A Scalable Tree Boosting System`
- DOI：https://doi.org/10.1145/2939672.2939785
- 说明：真实条目；作为方法学基线引用没有问题。

## 当前建议
如果我们现在要继续推进图表，参考文献层面已经足够安全，可以放心往下走。  
后续真正要继续补的不是“真假”，而是：

1. 是否还要补 2-4 篇更贴近肺结节术中定位的临床文献  
2. 是否要补 1-2 篇更贴近柔性触觉阵列或肺部触觉感知的硬件文献  
3. 英文稿中哪些地方应该精简引用，避免引言过重
