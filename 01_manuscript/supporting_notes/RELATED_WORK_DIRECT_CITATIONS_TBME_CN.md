# 相关工作与直接引用建议（TBME 版）

## 用法说明
- 本文件用于给 `MANUSCRIPT_DRAFT_V12_CN.md` 提供可直接替换到 Overleaf 的参考文献候选。
- 当前正文中使用临时占位符 `R1-R8`。
- 后续写英文稿时，可把这些条目整理成 BibTeX，并将 `R1-R8` 替换为正式编号。

## 建议正文中的角色分配
- `R1-R3`：说明现有肺结节术中/术前定位方法及局限
- `R4-R8`：说明微创触觉定位、触觉传感与 AI 相关研究基础

## 候选参考文献

### R1
Khereba M, Ferraro P, Duranceau A, et al. Preoperative versus intraoperative localization of pulmonary nodules. *The Journal of Thoracic and Cardiovascular Surgery*, 2018.  
DOI: `10.1016/j.jtcvs.2018.07.090`

**建议用途**
- 引用“现有非触诊定位技术已较系统总结”的句子
- 用于支撑临床工作流里对定位方法优缺点的综述性描述

### R2
Tang L, Zhang Y, Wang Y. Intraoperative identification of pulmonary nodules during minimally invasive thoracic surgery: a narrative review. *Quantitative Imaging in Medicine and Surgery*, 2022.  
PubMed PMID: `36330174`

**建议用途**
- 引用“术中超声、视觉增强等方法依赖操作者经验和场景条件”的句子
- 用于引言和 discussion 中说明当前定位手段的边界

### R3
Wang G, Ren C-J, Shi Y-B, Miao H-M. CT-Guided Anchored Needle Versus Hook-Wire Localization for Pulmonary Nodules: A Meta-Analysis. *Journal of Computer Assisted Tomography*, 2025.  
PubMed PMID: `41247803`

**建议用途**
- 引用“钩线/锚定针等术前定位方案伴有并发症和位移风险”的句子
- 用于强化“临床上已有方案，但仍不理想”的背景论证

### R4
Perri MT, Trejos AL, Patel RV, Naish MD. Robot-assisted Tactile Sensing for Minimally Invasive Tumor Localization. *The International Journal of Robotics Research*, 2009.  
DOI: `10.1177/0278364909101136`

**建议用途**
- 引用“微创肿瘤触觉定位已有机器人辅助触觉感知研究”的句子
- 用于说明本工作并非凭空提出触觉定位想法，而是承接既有触觉研究方向

### R5
Trejos AL, Jayender J, Perri MT, Naish MD, Patel RV. Integration of Force Reflection with Tactile Sensing for Minimally Invasive Robotics-Assisted Tumor Localization. *IEEE Transactions on Haptics*, 2013.  
DOI: `10.1109/TOH.2012.64`

**建议用途**
- 引用“力反馈与触觉感知结合已被用于微创肿瘤定位”的句子
- 用于支撑本文中“触觉恢复不仅是 sensing，也和术者交互有关”的观点

### R6
Ly HH, Hoang T, Kim G, et al. Tactile force sensor based on a modified acoustic reflection principle for intraoperative tumor localization in minimally invasive surgery. *International Journal of Computer Assisted Radiology and Surgery*, 2025.  
DOI: `10.1007/s11548-025-03511-0`

**建议用途**
- 引用“新型声反射/力学触觉传感器已用于术中肿瘤定位探索”的句子
- 用于说明该方向仍在快速发展，但多集中于器件和单任务演示

### R7
Huang X, Kumar S, Van de Velde S, et al. Tactile Perception Technologies and Their Applications in Minimally Invasive Surgery: A Review. *Frontiers in Physiology*, 2020.  
DOI: `10.3389/fphys.2020.611596`

**建议用途**
- 引用“触觉恢复是 MIS 中持续关注的工程问题”的句子
- 用于讨论中说明本文在 broader tactile MIS literature 里的位置

### R8
Bewley T, Hamed A, Markides H, et al. Tactile Sensing for Minimally Invasive Surgery: Conventional Methods and Potential Emerging Tactile Technologies. *Frontiers in Robotics and AI*, 2022.  
DOI: `10.3389/frobt.2021.705662`

**建议用途**
- 引用“AI/新型触觉技术正成为 MIS 触觉研究的增长方向”的句子
- 用于讨论中把本文放回 emerging tactile technology 的背景下

## 可直接粘贴到正文的示例句

### 示例 1：临床方法背景
现有胸外科定位文献已系统总结了术前和术中非触诊定位方案，包括钩线、染料、支气管镜导航及术中超声等，并反复指出并发症风险、位移风险和操作者依赖性仍是其进入常规工作流的主要障碍 [R1-R3]。

### 示例 2：触觉相关工作背景
在微创肿瘤定位与触觉恢复方向，机器人辅助触觉感知、力反馈融合以及新型声反射或力学触觉传感器等研究已证明，深部包块可以通过机械响应差异被感知和定位 [R4-R8]。

### 示例 3：与我们工作的区别
与既有工作相比 [R4-R8]，本文更强调在肺部特异场景下把触觉探测、大小反演、粗深度辨别、raw-input 学习与可解释性验证组织为同一条层级证据链，而不是停留在单一器件或单一定位任务的可行性展示。
