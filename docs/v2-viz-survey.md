# hypviz v2 需求调研:双曲机器学习论文中的高频可视化类型(2021–2026)

> 调研日期:2026-07-05。
> 方法:通过 WebSearch / WebFetch 检索 arXiv 及 NeurIPS / ICML / ICLR / CVPR / ICCV / ACL / TMLR / ACM MM 等 venue 的双曲学习论文;每条引用均已在写入前通过抓取 arXiv abstract 页验证真实存在;图编号(Fig. N)仅在成功抓取论文 HTML(ar5iv / arxiv.org/html)或 PDF 文本并核对图注后才标注,未核对到图注的论文只引用、不标图号。venue 信息以 arXiv 页 comments 字段、官方会议页或出版社页面为准;仅凭记忆、未获得页面佐证的条目一律移入第 6 节并标注 [unverified]。
> 共验证 **29 篇**论文(2021–2026 主体 25 篇 + 奠基性前史 4 篇),归纳出 **14 种**反复出现的可视化类型:Tier 1 共 3 种、Tier 2 共 6 种、Tier 3 共 5 种。

---

## 1. 论文清单(全部已验证)

### 1.1 主体:2021–2026

| # | 论文(标题保留英文) | Venue + 年份 | arXiv |
|---|---|---|---|
| P1 | HoroPCA: Hyperbolic Dimensionality Reduction via Horospherical Projections | ICML 2021 | https://arxiv.org/abs/2106.03306 |
| P2 | Hyperbolic Busemann Learning with Ideal Prototypes | NeurIPS 2021 | https://arxiv.org/abs/2106.14472 |
| P3 | Fully Hyperbolic Neural Networks | ACL 2022 | https://arxiv.org/abs/2105.14686 |
| P4 | CO-SNE: Dimensionality Reduction and Visualization for Hyperbolic Data | CVPR 2022 | https://arxiv.org/abs/2111.15037 |
| P5 | Hyperbolic Vision Transformers: Combining Improvements in Metric Learning | CVPR 2022 | https://arxiv.org/abs/2203.10833 |
| P6 | Hyperbolic Image Segmentation | CVPR 2022 | https://arxiv.org/abs/2203.05898 |
| P7 | Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers | CVPR 2022 | https://arxiv.org/abs/2107.11472 |
| P8 | Hyperbolic Deep Reinforcement Learning | ICLR 2023 | https://arxiv.org/abs/2210.01542 |
| P9 | Hyperbolic Contrastive Learning for Visual Representations beyond Objects | CVPR 2023 | https://arxiv.org/abs/2212.00653 |
| P10 | Hyperbolic Image-Text Representations (MERU) | ICML 2023 | https://arxiv.org/abs/2304.09172 |
| P11 | Poincaré ResNet | ICCV 2023 | https://arxiv.org/abs/2303.14027 |
| P12 | The Numerical Stability of Hyperbolic Representation Learning | ICML 2023 | https://arxiv.org/abs/2211.00181 |
| P13 | HypLL: The Hyperbolic Learning Library | ACM MM 2023 | https://arxiv.org/abs/2306.06154 |
| P14 | Hyperbolic Deep Learning in Computer Vision: A Survey | IJCV 2024 | https://arxiv.org/abs/2305.06611 |
| P15 | Shadow Cones: A Generalized Framework for Partial Order Embeddings | ICLR 2024 | https://arxiv.org/abs/2305.15215 |
| P16 | Fully Hyperbolic Convolutional Neural Networks for Computer Vision (HCNN) | ICLR 2024 | https://arxiv.org/abs/2303.15919 |
| P17 | Hyperbolic Random Forests (HoroRF) | TMLR 2024 | https://arxiv.org/abs/2308.13279 |
| P18 | Learning Structured Representations with Hyperbolic Embeddings (HypStructure) | NeurIPS 2024 | https://arxiv.org/abs/2412.01023 |
| P19 | Compositional Entailment Learning for Hyperbolic Vision-Language Models (HyCoCLIP) | ICLR 2025 (oral) | https://arxiv.org/abs/2410.06912 |
| P20 | Hyperbolic Safety-Aware Vision-Language Models (HySAC) | CVPR 2025 | https://arxiv.org/abs/2503.12127 |
| P21 | Hyperbolic Genome Embeddings | ICLR 2025 | https://arxiv.org/abs/2507.21648 |
| P22 | Hyperbolic Fine-Tuning for Large Language Models (HypLoRA) | NeurIPS 2025 | https://arxiv.org/abs/2410.04010 |
| P23 | HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts | arXiv 2025(预印本) | https://arxiv.org/abs/2505.24722 |
| P24 | Hyperbolic Graph Neural Networks: A Review of Methods and Applications | arXiv 2022(2025 大修) | https://arxiv.org/abs/2202.13852 |
| P25 | Why are hyperbolic neural networks effective? A study on hierarchical representation capability | arXiv 2024(预印本) | https://arxiv.org/abs/2402.02478 |

### 1.2 前史奠基(2021 前,可视化范式的源头,大量被 2021–2026 论文复刻)

| # | 论文 | Venue + 年份 | arXiv |
|---|---|---|---|
| P26 | Poincaré Embeddings for Learning Hierarchical Representations | NeurIPS 2017 | https://arxiv.org/abs/1705.08039 |
| P27 | Hyperbolic Entailment Cones for Learning Hierarchical Embeddings | ICML 2018 | https://arxiv.org/abs/1804.01882 |
| P28 | Hyperbolic Graph Convolutional Neural Networks (HGCN) | NeurIPS 2019 | https://arxiv.org/abs/1910.12933 |
| P29 | Hyperbolic Image Embeddings | CVPR 2020 | https://arxiv.org/abs/1904.02239 |

---

## 2. 可视化类型总览

| 编号 | 类型 | Tier | 已确认使用的论文数(本清单内) |
|---|---|---|---|
| V1 | Poincaré 盘 2D embedding 散点图 | **Tier 1** | ≥10 |
| V2 | 层级/树嵌入 + 测地线边 | **Tier 1** | ≥7 |
| V3 | 范数/到原点距离分布图(直方图、散点、norm 排序样例、逐像素热图) | **Tier 1** | ≥8 |
| V4 | 双曲降维投影(HoroPCA / CO-SNE / hyperbolic UMAP) | Tier 2 | ≥5 |
| V5 | Entailment cones / shadow cones 锥体可视化 | Tier 2 | ≥5 |
| V6 | 双曲决策边界(gyroplane / horosphere 分类面) | Tier 2 | ≥5 |
| V7 | 测地线遍历 / 插值(embedding→root 的语义 traversal) | Tier 2 | 3 |
| V8 | 流形操作概念示意图(hyperboloid + 切空间 / exp map / Möbius 运算) | Tier 2 | ≥6 |
| V9 | δ-hyperbolicity / 距离失真分析图 | Tier 2 | ≥6 |
| V10 | 维度–性能曲线(低维优势分析) | Tier 3 | ≥4 |
| V11 | 训练动态 / 数值稳定性诊断图(梯度范数、崩溃分析) | Tier 3 | ≥4 |
| V12 | 理想点 / 边界原型可视化(ideal prototypes、Busemann 水平集) | Tier 3 | ≥4 |
| V13 | 多模型 / 多坐标卡对比图(Poincaré vs Lorentz vs Klein vs 半平面) | Tier 3 | ≥3 |
| V14 | 曲率分析图(曲率分布、曲率消融) | Tier 3 | 3 |

---

## 3. 各类型详细条目

### V1 · Poincaré 盘 2D embedding 散点图 —— Tier 1

**(a) 画的是什么**:把(2 维或降维后的)双曲 embedding 画在单位圆盘内,点按类别/层级深度着色,圆盘边界画出;"generic 靠圆心、specific 靠边界"是标准读图方式。

**(b) 回答的 qualitative 问题**:层级结构是否自发涌现?类是否沿边界分离成扇区?哪些样本被推到边界(模型"自信")、哪些留在圆心附近(模糊/OOD/高层概念)?

**(c) 使用它的论文**(≥3,全部已验证):
- Poincaré Embeddings for Learning Hierarchical Representations — NeurIPS 2017 — https://arxiv.org/abs/1705.08039 (Fig. 2:WordNet mammals 子树的 2D Poincaré 嵌入,训练中期 vs 收敛后)
- Hyperbolic Image Embeddings — CVPR 2020 — https://arxiv.org/abs/1904.02239 (Fig. 1:MNIST/Omniglot 嵌入,模糊样本聚在圆心;Fig. 6:few-shot 嵌入的 UMAP 盘上投影)
- Hyperbolic Vision Transformers — CVPR 2022 — https://arxiv.org/abs/2203.10833 (Fig. 2:Cars-196 的 Hyp-DINO 嵌入在 Poincaré 盘上,按类着色)
- The Numerical Stability of Hyperbolic Representation Learning — ICML 2023 — https://arxiv.org/abs/2211.00181 (Fig. 2、Fig. 4:同一数据在 Poincaré / Lorentz / Euclidean 参数化下的盘上散点)
- Learning Structured Representations with Hyperbolic Embeddings — NeurIPS 2024 — https://arxiv.org/abs/2412.01023 (Fig. 5–6:CIFAR-10/100 hyperbolic UMAP 盘上散点,含 OOD 分离)
- Hyperbolic Deep Reinforcement Learning — ICLR 2023 — https://arxiv.org/abs/2210.01542 (Fig. 10、Fig. 12:RL 状态表示沿轨迹在盘上的演化)
- 另见:P2 (Fig. 4)、P6 (Fig. 3, 7)、P17 (Fig. 6)、P21 (Fig. 3)、P14 (Fig. 5,综述转载)。

**(d) 复现所需数据形态**:`(n, d)` 双曲 embedding(Lorentz 或 Poincaré 坐标;d>2 时需先过 V4 降维)+ 每点的类别/层级标签;可选:每点标量(置信度、深度)映射到颜色/大小;圆盘边界与坐标卡由库负责。

---

### V2 · 层级/树嵌入 + 测地线边 —— Tier 1

**(a)**:在 V1 的散点之上,把树/DAG 的父子边画成**测地线弧**(Poincaré 盘中为正交于边界的圆弧),根在圆心、叶在边缘。

**(b)**:嵌入是否忠实还原了给定的层级(is-a / 系统发育 / 标签树)?边是否短而不交叉?哪些子树被挤压或错位?

**(c)**:
- Poincaré Embeddings for Learning Hierarchical Representations — NeurIPS 2017 — https://arxiv.org/abs/1705.08039 (Fig. 1(b) 树嵌入示意;Fig. 2 蓝色边为 WordNet is-a 关系)
- HoroPCA — ICML 2021 — https://arxiv.org/abs/2106.03306 (Fig. 4:WordNet mammal 层级的 2D 可视化,PGA vs HoroPCA,标注失真值)
- CO-SNE — CVPR 2022 — https://arxiv.org/abs/2111.15037 (Fig. 1、Fig. 6:mammal 子树,根居中、叶近边界)
- Shadow Cones — ICLR 2024 — https://arxiv.org/abs/2305.15215 (Fig. 5:mammal 子图在**半平面模型**中的嵌入)
- Hyperbolic Entailment Cones — ICML 2018 — https://arxiv.org/abs/1804.01882 (Fig. 3:均匀树与 WordNet 子集的对比嵌入)
- 另见:P12 (Fig. 1–2 合成树)、P17 (Fig. 6 WordNet 名词层级)。

**(d)**:`(n, d)` 双曲 embedding + **边表** `(E, 2)`(树或 DAG)+ 可选节点深度;渲染需按坐标卡把每条边离散化为测地线段(Lorentz 中枢内插即可)。

---

### V3 · 范数 / 到原点(root)距离分布图 —— Tier 1

**(a)**:对每个 embedding 计算 d(O, x)(或 Poincaré 范数),画分组直方图/密度图;变体:范数 vs 某标量(词频、正确率)的散点或曲线、按范数排序的样例网格、逐像素范数热图(分割任务)。

**(b)**:范数是否编码 **generality / 不确定性 / 层级深度**?文本是否比图像更靠 root?高频 token 是否更靠圆心?低范数像素是否对应语义边界?in-domain 与 OOD 的径向分布是否分离?

**(c)**:
- Hyperbolic Image-Text Representations (MERU) — ICML 2023 — https://arxiv.org/abs/2304.09172 (Fig. 4:image/text 嵌入到 [ROOT] 距离的分布,CLIP 重叠 vs MERU 分离)
- Hyperbolic Safety-Aware Vision-Language Models — CVPR 2025 — https://arxiv.org/abs/2503.12127 (Fig. 2:safe/unsafe × image/text 四峰距离分布)
- Hyperbolic Fine-Tuning for Large Language Models — NeurIPS 2025 — https://arxiv.org/abs/2410.04010 (Fig. 1:token 频率呈幂律 + 频率与范数负相关)
- Hyperbolic Image Embeddings — CVPR 2020 — https://arxiv.org/abs/1904.02239 (Fig. 5:MNIST vs Omniglot 到原点距离分布,用作 OOD 分离)
- Hyperbolic Image Segmentation — CVPR 2022 — https://arxiv.org/abs/2203.05898 (Fig. 1、5:逐像素不确定性热图;Fig. 4:置信度与语义边界距离的相关)
- 另见:P2 (Fig. 5 距原点距离 vs 分类正确性)、P19 (Fig. 5a 范数直方图)、P8 (Fig. 11 按表示幅值排序的 CIFAR 样例)、P23 (Table 3 泛化词 vs 专指词范数)。

**(d)**:`(n, d)` 双曲 embedding + 分组标签(模态/安全性/类别)或标量(token 频率、正确性);逐像素热图需 `(H, W, d)` 像素级嵌入;库内在 Lorentz 中枢统一计算 d(O,·)。

---

### V4 · 双曲降维投影(HoroPCA / CO-SNE / hyperbolic UMAP)—— Tier 2

**(a)**:把高维 Lorentz/Poincaré embedding 降到 2D 双曲盘再做 V1/V2 式展示;降维本身保持双曲结构(horospherical 投影、双曲 SNE、双曲 UMAP)。

**(b)**:高维嵌入里是否真的存在层级/聚类结构?不同方法(或消融)学到的空间在"形状"上差在哪?(它是 V1 在 d>2 时的前置步骤。)

**(c)**:
- HoroPCA — ICML 2021 — https://arxiv.org/abs/2106.03306 (方法论文;Fig. 1–5)
- CO-SNE — CVPR 2022 — https://arxiv.org/abs/2111.15037 (方法论文;Fig. 4–8:合成簇、单细胞分化、WordNet、双曲 MNIST 特征、Poincaré-VAE 隐空间)
- Compositional Entailment Learning (HyCoCLIP) — ICLR 2025 — https://arxiv.org/abs/2410.06912 (Fig. 5b/5c:用 HoroPCA 与 CO-SNE 作分析工具;Fig. 9:对 MERU 空间做同样投影对比)
- Learning Structured Representations with Hyperbolic Embeddings — NeurIPS 2024 — https://arxiv.org/abs/2412.01023 (Fig. 5–6:hyperbolic UMAP)
- Hyperbolic Vision Transformers — CVPR 2022 — https://arxiv.org/abs/2203.10833 (Fig. 2:UMAP 投到盘上)

**(d)**:`(n, D)` 高维双曲 embedding(D 几十到几百)→ `(n, 2)` 盘坐标;需要可插拔的 HoroPCA / CO-SNE / hUMAP 实现或适配器,且输出直接进入 V1/V2 渲染管线。

---

### V5 · Entailment cones / shadow cones 锥体可视化 —— Tier 2

**(a)**:在盘/半平面中画出以某 embedding 为顶点、开角随半径变化的测地凸锥(或影锥),并展示子概念落入父概念锥内;方法示意与定性验证两用。

**(b)**:偏序(蕴含、包含、部分-整体)关系是否被几何化?哪些 pair 违反锥约束?锥开角(aperture)如何随离原点距离缩放?

**(c)**:
- Hyperbolic Entailment Cones — ICML 2018 — https://arxiv.org/abs/1804.01882 (Fig. 2:不同半径处的 Poincaré 角锥与传递性验证;Fig. 3)
- Hyperbolic Image-Text Representations (MERU) — ICML 2023 — https://arxiv.org/abs/2304.09172 (Fig. 3:entailment loss 的锥示意,外角 ext(x,y) 定义)
- Shadow Cones — ICLR 2024 — https://arxiv.org/abs/2305.15215 (Fig. 1–4:半平面/盘中 umbral 与 penumbral 锥的四种构型)
- Compositional Entailment Learning (HyCoCLIP) — ICLR 2025 — https://arxiv.org/abs/2410.06912 (Fig. 1:image/text box 的层级锥;Fig. 3:aperture 阈值 η 的缩放机制)
- Hyperbolic Safety-Aware Vision-Language Models — CVPR 2025 — https://arxiv.org/abs/2503.12127 (Fig. 1:safe/unsafe 内容的 entailment 层级)

**(d)**:顶点 embedding + 锥开角函数(如 Ganea 的 K/‖x‖ 形式或 shadow-cone 光源参数)+ 待检验的点对集合;渲染需锥边界曲线(测地线/等距线)与锥内区域填充,以及 `cone_contains(parent, child)` 谓词用于违例着色。

---

### V6 · 双曲决策边界(gyroplane / horosphere 分类面)—— Tier 2

**(a)**:在 2D 盘上画分类器的决策面——Poincaré gyroplane(过某点、以某方向为法向的测地子流形)或 horosphere 分裂面——叠加数据散点。

**(b)**:双曲分类面与欧氏直线面相比如何切分层级数据?margin/到边界的测地距离长什么样?为什么欧氏超平面在树状数据上失效?

**(c)**:
- Hyperbolic Image Segmentation — CVPR 2022 — https://arxiv.org/abs/2203.05898 (Fig. 2:gyroplane 的 offset/orientation 与像素到决策边界的测地距离)
- Hyperbolic Random Forests — TMLR 2024 — https://arxiv.org/abs/2308.13279 (Fig. 1:欧氏超平面 vs horosphere 分裂;Fig. 3–4:多级 horosphere 分裂过程)
- Hyperbolic Genome Embeddings — ICLR 2025 — https://arxiv.org/abs/2507.21648 (Fig. 3:双曲 vs 欧氏模型在 Poincaré 盘/平面上学到的 2D 决策边界)
- Hyperbolic Image Embeddings — CVPR 2020 — https://arxiv.org/abs/1904.02239 (Fig. 3:双曲超平面与 Möbius 运算示意)
- Hyperbolic Deep Learning in Computer Vision: A Survey — IJCV 2024 — https://arxiv.org/abs/2305.06611 (Fig. 2:gyroplane / prototype / sample 三种监督策略示意)

**(d)**:2D 双曲数据 + 分类器参数(gyroplane:偏置点 p 与切向法向 a;horosphere:理想点 ω 与半径/Busemann 值);通用做法是对盘内网格求分类器得分再画等值线/区域填充——需要"盘内网格采样 + Lorentz 批量评估"设施。

---

### V7 · 测地线遍历 / 插值(traversal toward root)—— Tier 2

**(a)**:从某个 embedding 沿测地线走向原点([ROOT])或另一 embedding,在中途各点做最近邻检索,展示"图像 → 越来越泛化的文本"的语义链条。

**(b)**:径向方向是否对应"具体 → 抽象"的语义轴?两个概念的测地中点是不是它们的共同上位概念?不安全内容沿测地线走向原点是否变安全?

**(c)**:
- Hyperbolic Image-Text Representations (MERU) — ICML 2023 — https://arxiv.org/abs/2304.09172 (Fig. 5、7–11:image traversal,MERU vs CLIP)
- Compositional Entailment Learning (HyCoCLIP) — ICLR 2025 — https://arxiv.org/abs/2410.06912 (Fig. 6、10–12:图像对之间及向原点的测地插值)
- Hyperbolic Safety-Aware Vision-Language Models — CVPR 2025 — https://arxiv.org/abs/2503.12127 (Fig. 3:unsafe→safe 的 traversal)

**(d)**:两端点 Lorentz embedding + 测地线等距采样(测地线参数化)+ 一个检索语料库(caption/image 嵌入集合)与最近邻查询回调;产出通常是"样例条带图"(每步的检索结果网格)而非几何图,几何图上则画出带步点标记的测地线。

---

### V8 · 流形操作概念示意图(hyperboloid + 切空间 / exp map / Möbius 运算)—— Tier 2

**(a)**:方法节的"Fig. 1 型"示意:3D hyperboloid 及其到 Poincaré 盘的投影、切平面与 exp/log map 箭头、Lorentz boost/rotation、Möbius 加法、平行移动、horosphere 几何等。

**(b)**:不直接回答实验问题,而是向读者解释模型的几何机制;是双曲论文与读者沟通的事实标准,库若能一键生成会显著降低论文作图成本。

**(c)**:
- Fully Hyperbolic Neural Networks — ACL 2022 — https://arxiv.org/abs/2105.14686 (Fig. 1:切空间变换、Lorentz boost、Lorentz rotation 在 hyperboloid 上的示意)
- Hyperbolic Image Embeddings — CVPR 2020 — https://arxiv.org/abs/1904.02239 (Fig. 3:Möbius 和、测地线、双曲平均、双曲超平面)
- HoroPCA — ICML 2021 — https://arxiv.org/abs/2106.03306 (Fig. 2:汇聚于理想点的测地线束与同心 horosphere;Fig. 5:ℍ³ 中 horospherical 投影的"open book"图)
- Learning Structured Representations with Hyperbolic Embeddings — NeurIPS 2024 — https://arxiv.org/abs/2412.01023 (Fig. 3:hyperboloid / Klein / Poincaré 三模型关系图)
- Hyperbolic Genome Embeddings — ICLR 2025 — https://arxiv.org/abs/2507.21648 (Fig. 1:序列投到 hyperboloid + 双曲决策边界的总览图)
- HELM — arXiv 2025 — https://arxiv.org/abs/2505.24722 (Fig. 2:流形间投影的 MiCE / 注意力模块示意)

**(d)**:无需真实数据;需要可编程的"几何场景"原语:3D hyperboloid 面、投影连线(hyperboloid→盘/Klein/半平面)、切平面矩形、exp/log map 箭头、平行移动箭头、horosphere/horocycle,可加文字标注。

---

### V9 · δ-hyperbolicity / 距离失真分析图 —— Tier 2

**(a)**:量化"数据/嵌入有多像树":Gromov δ-hyperbolicity 的分布或数值表、图距离 vs 嵌入距离的散点(失真)、失真随方法/维度的对比。

**(b)**:该用双曲空间吗(数据本身 δ 小吗)?嵌入保距吗?双曲带来的增益是否与数据双曲性相关?

**(c)**:
- Hyperbolic Image Embeddings — CVPR 2020 — https://arxiv.org/abs/1904.02239 (Fig. 4:测地三角形"slim"性示意 + 各数据集 δ 值表)
- Hyperbolic Deep Reinforcement Learning — ICLR 2023 — https://arxiv.org/abs/2210.01542 (Fig. 3:树/双曲/欧氏空间的 δ 示意;Fig. 4:训练中相对 δ 下降与性能提升的相关)
- Hyperbolic Genome Embeddings — ICLR 2025 — https://arxiv.org/abs/2507.21648 (Fig. 10–12:δ 分布、δ_worst 与性能增益的相关、"emergent hyperbolicity")
- Hyperbolic Fine-Tuning for Large Language Models — NeurIPS 2025 — https://arxiv.org/abs/2410.04010 (Table 2:token embedding 的 δ-hyperbolicity 量表)
- HoroPCA — ICML 2021 — https://arxiv.org/abs/2106.03306 (Fig. 4:两方法失真值对比 0.534 vs 0.078)
- Learning Structured Representations with Hyperbolic Embeddings — NeurIPS 2024 — https://arxiv.org/abs/2412.01023 (Fig. 2:真值树度量 vs 学到距离的散点;Fig. 4:失真 δ_rel 随维度变化)

**(d)**:成对测地距离矩阵 `(n, n)`(Lorentz 内积即可)± 参考图距离矩阵;δ 计算需 Gromov 积的采样近似;绘图为直方图/散点/折线,属于常规 2D 图但依赖库内距离与 δ 例程。

---

### V10 · 维度–性能曲线(低维优势分析)—— Tier 3(常见但形式常规)

**(a)**:x 轴为 embedding 维度(常低至 2),y 轴为任务指标,双曲 vs 欧氏两条曲线。

**(b)**:"双曲空间在低维更省参数"这一核心卖点是否成立?几维以下差距拉开?

**(c)**:
- Hyperbolic Image Segmentation — CVPR 2022 — https://arxiv.org/abs/2203.05898 (Fig. 6:256 维→2 维)
- Hyperbolic Random Forests — TMLR 2024 — https://arxiv.org/abs/2308.13279 (Fig. 5:2D→10D 的 WordNet 分类)
- Learning Structured Representations with Hyperbolic Embeddings — NeurIPS 2024 — https://arxiv.org/abs/2412.01023 (Fig. 4:失真随维度)
- Hyperbolic Genome Embeddings — ICLR 2025 — https://arxiv.org/abs/2507.21648 (Fig. 5:不同通道维度下的性能)

**(d)**:多组 `(维度, 指标)` 训练结果;纯 2D 折线图,库只需提供风格化模板(非几何渲染)。

---

### V11 · 训练动态 / 数值稳定性诊断图 —— Tier 3

**(a)**:训练过程中梯度范数、embedding 范数、NaN/溢出、性能崩溃的曲线与对比;双曲特有(边界附近浮点耗尽、Riemannian 梯度消失/爆炸)。

**(b)**:哪种参数化(Poincaré vs Lorentz vs 欧氏参数化)更好训?feature clipping / 正则是否消除了梯度爆炸?嵌入是否被"卡"在离圆心太近处?

**(c)**:
- The Numerical Stability of Hyperbolic Representation Learning — ICML 2023 — https://arxiv.org/abs/2211.00181 (Fig. 3:各参数化的 Riemannian 梯度范数比随 epoch 变化;Fig. 2:Poincaré 嵌入聚在中心 vs Lorentz 铺开)
- Hyperbolic Deep Reinforcement Learning — ICLR 2023 — https://arxiv.org/abs/2210.01542 (Fig. 6:朴素双曲实现的梯度幅值/方差爆炸;Fig. 7:S-RYM 正则后的恢复)
- Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers — CVPR 2022 — https://arxiv.org/abs/2107.11472 (核心贡献即幅值裁剪治理消失梯度;图编号未逐一核对)
- CO-SNE — CVPR 2022 — https://arxiv.org/abs/2111.15037 (Fig. 3:双曲 t 分布 vs Cauchy 分布的梯度/斥力分析)

**(d)**:训练日志(step, grad_norm, emb_norm, metric)+ 可选 dtype 信息;双曲特有的增值点是把"float32/float64 可表示半径"画成盘上的参考圆(P12 的理论分析)。

---

### V12 · 理想点 / 边界原型可视化(ideal prototypes、Busemann 水平集)—— Tier 3

**(a)**:把类原型放在盘的**边界**(理想点)上,画出样本向理想原型汇聚的过程,或以颜色场展示 Busemann 函数/惩罚损失的水平集(即 horocycle 族)。

**(b)**:边界原型是否带来类间均匀分离?样本离圆心的距离是否表达置信度?Busemann 损失的梯度场把点往哪里推?

**(c)**:
- Hyperbolic Busemann Learning with Ideal Prototypes — NeurIPS 2021 — https://arxiv.org/abs/2106.14472 (Fig. 1:样本向理想原型优化;Fig. 2:单个理想原型的惩罚 Busemann 损失颜色场与径向梯度;Fig. 4:二维输出空间的类分布)
- HoroPCA — ICML 2021 — https://arxiv.org/abs/2106.03306 (Fig. 2–3:理想点、horosphere 与保距投影)
- Hyperbolic Random Forests — TMLR 2024 — https://arxiv.org/abs/2308.13279 (Fig. 1、3:以理想点为心的 horosphere 分裂)
- Hyperbolic Deep Learning in Computer Vision: A Survey — IJCV 2024 — https://arxiv.org/abs/2305.06611 (Fig. 2:boundary prototype 策略示意)

**(d)**:边界方向向量(理想点,单位向量)集合 + 样本 embedding;标量场渲染需要盘内网格 + Busemann 函数 B_ω(x) 求值(Lorentz 下有闭式)。

---

### V13 · 多模型 / 多坐标卡对比图 —— Tier 3

**(a)**:同一份数据/几何对象并排画在 Poincaré 盘、Lorentz hyperboloid、Klein 盘、半平面等不同坐标卡中,或画出模型间的投影对应。

**(b)**:不同坐标卡各自失真什么(角度 vs 直线)?数值上哪个参数化能表示更远的点?半平面为什么更适合展示某些锥/层级?

**(c)**:
- Learning Structured Representations with Hyperbolic Embeddings — NeurIPS 2024 — https://arxiv.org/abs/2412.01023 (Fig. 3:hyperboloid / Klein / Poincaré 中"直线"的不同形态)
- The Numerical Stability of Hyperbolic Representation Learning — ICML 2023 — https://arxiv.org/abs/2211.00181 (Fig. 2:同一树在三种参数化下的最终嵌入并排)
- Shadow Cones — ICLR 2024 — https://arxiv.org/abs/2305.15215 (Fig. 1/4 半平面构型 vs Fig. 2/3 盘构型;Fig. 5 半平面 mammal 嵌入)

**(d)**:任意场景(点、边、锥)以 Lorentz 坐标存储 + 每坐标卡的投影;这正是"Lorentz 为计算中枢、多坐标卡渲染"架构的直接卖点,需要"同场景多视图"的并排布局 API。

---

### V14 · 曲率分析图 —— Tier 3

**(a)**:嵌入的局部曲率分布(如 Ricci 曲率直方图)、任务指标随曲率 c 的消融曲线、混合曲率专家的曲率配置。

**(b)**:数据"需要"多少负曲率?性能对曲率敏感吗?不同层/专家学到的曲率是否不同?

**(c)**:
- HELM — arXiv 2025 — https://arxiv.org/abs/2505.24722 (Fig. 1:token embedding 的 Ricci 曲率分布,论证局部双曲性)
- Hyperbolic Image Segmentation — CVPR 2022 — https://arxiv.org/abs/2203.05898 (Fig. 8:不同维度下的曲率消融)
- Hyperbolic Graph Convolutional Neural Networks — NeurIPS 2019 — https://arxiv.org/abs/1910.12933 (引入可训练曲率并分析其作用;图编号未逐一核对)

**(d)**:多次训练的 `(curvature, metric)` 结果或图上局部曲率估计值;绘图常规,但库需在所有几何原语中把曲率 c 作为一等参数,才能支撑"曲率扫描"小倍数图。

---

## 4. 频率分级依据

- **Tier 1(几乎每篇带分析节的论文都有)**:V1 盘散点、V2 树+测地线边、V3 范数分布。29 篇中超过三分之二至少含其一;P14(IJCV 综述)明确将"Poincaré 盘可视化"与"范数=不确定性"总结为社区惯例(其 Fig. 3、Fig. 5 直接转载 P6、P5 的图)。
- **Tier 2(常见,特定子方向内近乎必备)**:V4 双曲降维(分析工具链的标准前置,HyCoCLIP 同时用 HoroPCA+CO-SNE);V5 锥体(蕴含/偏序方向必备);V6 决策边界(分类器方向必备);V7 traversal(多模态方向 2023 年后的标配定性实验);V8 流形示意图(方法节事实标准);V9 δ/失真(动机论证的标准证据)。
- **Tier 3(小众但有代表性)**:V10 维度曲线、V11 稳定性诊断、V12 理想点/Busemann 场、V13 多坐标卡对比、V14 曲率分析。

---

## 5. 映射到 hypviz v2 功能需求

前提:hypviz 现有原语 `Point / Geodesic / ExpMap / MobiusAdd / PointCloud`,以 Lorentz 模型为计算中枢,支持 Poincaré / Klein / 半平面坐标卡。以下按"新增原语 → 新增模块 → 渲染/布局设施"组织,并标注支撑的可视化类型与优先级(P0=Tier 1 支撑,P1=Tier 2,P2=Tier 3)。

### 5.1 新增几何原语

| 原语 | 说明 | 支撑 | 优先级 |
|---|---|---|---|
| `EdgeBundle(points, edges)` | 边表批量测地线渲染(现有 `Geodesic` 的向量化版),支持按深度/权重着色、透明度、箭头 | V2, V1 | **P0** |
| `IdealPoint(direction)` | 边界理想点,一等公民:可作原型标记、horosphere/锥的锚点 | V12, V5, V6 | **P0** |
| `Horocycle / Horosphere(ideal_point, busemann_value)` | 以理想点为心的极限圆/球;含 `Busemann(x)` 求值 | V6, V12, V8 | P1 |
| `Cone(apex, aperture_fn, kind='entailment'|'umbral'|'penumbral')` | 蕴含锥/影锥区域与边界渲染 + `contains(x)` 谓词(违例检测着色) | V5 | P1 |
| `Gyroplane(p, normal)` | Poincaré gyroplane / Lorentz 超平面(测地子流形),含到平面的测地距离 | V6 | P1 |
| `GeodesicPath(x, y, n_steps)` / `RadialPath(x)` | 测地线等距采样器,返回途经点(供检索回调),渲染为带步点的弧 | V7 | P1 |
| `TangentPlane(p)`、`LogMap`、`ParallelTransport` | 配合现有 `ExpMap` 组成示意图套件(箭头、切平面面片) | V8 | P1 |
| `Hyperboloid3D` 场景 | 3D hyperboloid 面 + 到 Poincaré/Klein/半平面的投影连线(matplotlib 3D / plotly 后端) | V8, V13 | P2 |
| `ScalarField(fn, chart, resolution)` | 盘/半平面内网格采样 + 等值线/填充渲染(决策边界、Busemann 场、损失景观的统一底座) | V6, V12 | **P0**(底座) |

### 5.2 新增模块

| 模块 | 内容 | 支撑 | 优先级 |
|---|---|---|---|
| `hypviz.plots`(高层图表 API) | `disk_scatter()`(V1:类着色、边界圆、root 标记、图例)、`tree_plot()`(V2:散点+EdgeBundle+深度色带)、`norm_hist()` / `radial_profile()`(V3:d(O,x) 分组直方图/KDE/小提琴、范数 vs 标量散点) | V1–V3 | **P0** |
| `hypviz.reduce` | `horopca()`、`cosne()`、`humap()`(自实现或适配 penalty:输出 2D Lorentz 坐标,直接进 `disk_scatter`);统一接口 `reduce(X, method=...)` | V4 | P1 |
| `hypviz.metrics` | `delta_hyperbolicity(X, n_samples)`(采样 Gromov 积)、`distortion(D_graph, D_emb)`、`pairwise_dist(X)`;配套 `delta_hist()`、`distance_scatter()`(真值 vs 嵌入距离密度散点)、`distortion_vs_dim()` 折线模板 | V9, V10 | P1 |
| `hypviz.diagnostics` | 训练日志适配(grad/emb 范数曲线模板);`float_horizon(dtype, chart)`:把浮点可表示半径画成参考圆叠加在任何盘图上(源自 P12 的分析) | V11 | P2 |
| `hypviz.traversal` | `traverse(x, y_or_origin, n_steps, retrieve_fn)` → 结构化结果 + "条带图"(每步最近邻缩略图/文本)排版器 | V7 | P1 |
| `hypviz.schematic` | 预制场景:exp/log map 图、Möbius 加法图、boost/rotation 图、horosphere 投影图(论文 Fig.1 生成器) | V8 | P2 |

### 5.3 渲染 / 架构设施

1. **同场景多坐标卡并排视图**(`scene.render(charts=['poincare','klein','halfplane'])`):所有原语已存 Lorentz 坐标,只需布局器;直接支撑 V13,也让 V2/V5 可选半平面(Shadow Cones 证明半平面对深层级更易读)。**P1**
2. **曲率 c 为场景级一等参数**:所有原语/距离/锥公式接受 c,支持 `curvature_sweep()` 小倍数图(V14)。**P2**
3. **大点云性能**:V1/V3 的真实用例是 10⁴–10⁶ 点(像素级、token 级),`PointCloud` 需 datashader/rasterization 路径。**P1**
4. **样式系统**:默认样式对齐社区惯例——边界圆细线、根/原点十字标记、按层级深度的顺序色带、violation 高亮红——降低论文作图摩擦。**P2**
5. **逐像素范数热图导出**:`norm_map(H×W×d) → H×W` 数组 + colormap(V3 分割子型),属 metrics 与 plots 的交集,注意它不需要盘渲染。**P2**

### 5.4 建议的 v2 里程碑

- **M1(覆盖 Tier 1)**:`EdgeBundle`、`IdealPoint`、`ScalarField` 底座 + `hypviz.plots` 三件套(disk_scatter / tree_plot / norm_hist)。
- **M2(覆盖 Tier 2)**:`hypviz.reduce`、`Cone`、`Gyroplane`、`Horocycle`、`GeodesicPath` + `hypviz.traversal`、`hypviz.metrics`。
- **M3(覆盖 Tier 3 与增值项)**:多坐标卡并排、`hypviz.schematic`、`hypviz.diagnostics`(含 float_horizon)、曲率扫描、大点云光栅化。

---

## 6. 低置信度 / 未完全验证条目 [unverified]

以下条目在检索中出现(arXiv 链接曾在搜索结果或引用中出现),但**未**逐一抓取其 arXiv 页核对题目/venue/图注,不计入正文统计,仅供后续排查:

- Klein Model for Hyperbolic Neural Networks — arXiv:2410.16813 [unverified venue](搜索结果中出现 PDF 链接;若属实,与 V13 相关)
- Hyperbolic Deep Learning for Foundation Models: A Survey — arXiv:2507.17787 [unverified venue](搜索结果中出现 HTML 链接)
- Hyperbolic Learning with Multimodal Large Language Models — arXiv:2408.05097 [unverified venue/内容](搜索结果引用中出现;疑与 V3 的"范数=不确定性"分析相关)
- Brain-Inspired AI with Hyperbolic Geometry / Hyperbolic Brain Representations — arXiv:2409.12990 [unverified:两个搜索结果对同一 ID 显示了不同标题,疑为版本改名]
- Lorentzian Graph Convolutional Networks — WWW 2021 [unverified arXiv ID](仅在他文引用文本中确认存在,未定位 arXiv 页)
- P7(Clipped Hyperbolic Classifiers)与 P28(HGCN)的**具体图编号**未核对,正文中仅作无图号引用;P9、P11、P13、P16、P24、P25 同样未核对图注,仅用于清单与频率佐证。

---

## 附:验证方法说明

1. **存在性**:P1–P29 每篇均通过 WebFetch 抓取 `arxiv.org/abs/<id>` 成功返回题目/作者/摘要(P13、P9、P16 的 venue 另经 dl.acm.org、openaccess.thecvf.com/dblp、iclr.cc/openreview 搜索结果佐证)。
2. **图编号**:P1, P2, P4, P5, P6, P8, P10, P12, P14, P15, P17, P18, P19, P20, P21, P22, P23, P26, P27, P29 共 20 篇成功抓取 ar5iv / arxiv HTML 或 PDF 文本并核对图注;其余论文不标图号。
3. **venue 归属**:优先采用 arXiv comments 字段;其次官方会议/期刊页(ICML PMLR、CVF Open Access、Springer IJCV、ACM DL、iclr.cc);两者皆无时标 "arXiv(预印本)"。
