# Literature Notes — `docs/literature_notes.md`

前序工作笔记。**目的不是罗列引用，而是明确本项目的创新边界**：哪些事情已经被别人做过
（因此本项目绝不声称是自己提出的），以及本项目剩下的、真正需要验证的问题是什么。

## 0. 立场声明（重要）

- 本项目**不**声称以下任何一项为自己的贡献：candidate-based grounding、region–text
  matching、candidate interaction、hard-negative grounding、NONE / no-target / rejection
  grounding、temperature scaling、risk–coverage 评估、DeepSets / Set Transformer 等 set 模型。
- 本项目唯一试图回答的问题是：在 **candidate-set shift**（K、组成、hardness、target
  availability 变化）下，grounding 输出的 **reliability**（calibration / selective risk /
  abstention 质量）是否**以结构化方式**退化；以及这种退化是否**在不使用 candidate-level
  set information 的情况下**就能被解释与修正。
- 因此下文每条笔记都写出 "与本项目的关系"：它构成我们的 baseline、我们的动机、还是我们
  明确排除的主张来源。
- 标注 **需查证** 的条目表示：本笔记基于已有知识撰写，检索未能逐条二次确认（年份/venue/
  具体数字），在正式引用前必须核对原始文献；不允许把未核实的细节当作事实写进论文正文。

---

## 1. Candidate region ranking / region–text matching（candidate-based VG）

| 工作 | 年份 | 一句话贡献 | 与本项目的关系 |
|---|---|---|---|
| **TGVCN — Text-Guided Convolution Networks for Grounding Subsets of Unknown Size in Natural Images** (Hu et al.) | CVPR 2019 | 把 grounding 明确建模为 "在未知大小的候选子集上做 text-guided region ranking" | 直接说明 "candidate ranking" 不是我们的想法；我们的差异点是**同一 target、嵌套 K 下 reliability 的变化**，而不是 ranking 架构本身 |
| **RCCF — A Real-Time Cross-Modality Correlation Filtering Method for Referring Expression Comprehension** (Liao et al.) | CVPR 2020 | 把 REC 重构为 phrase-region 相关性过滤/排序，并给出 RefCOCO/RefCOCO+ 的统计口径（我们 §dataset_protocol 引用的数据规模即来自此） | 提供 candidate-based 公式化的合法性；同时是我们的数据规模引用来源 |
| **SSN — Self-Supervised Set-to-target Network for Visual Grounding by Subquery Matching** (Ding et al.) | CVPR 2021 | 显式使用 "set-to-target" 对齐与负样本来强化 candidate 判别 | 我们的 hardness regime 概念与之相邻；本项目不引入此类辅助任务（第一阶段的 loss 保持简单） |
| **KLSP — Knowledge Learned from Scenario Prompts in Transformer Helps Visual Grounding** (Chen et al.) | ICCV 2021 | 用 scenario prompts 注入场景先验，在多候选判别式 grounding 中提升表现 | 属于 "更复杂的 candidate-aware 模型" 路线；我们在 Gate Q1/Q2 通过之前**刻意不**走这条路，把它作为对照文献而非 baseline 实现 |
| **TransVG — Towards End-to-End Visual Grounding with Transformers** (Zhu et al.) | NeurIPS 2022（需查证：常见记为 NeurIPS 2021 / TPAMI 扩展） | 单流 end-to-end transformer 做整体图像 grounding（非显式 candidate 集合） | 说明 "非 candidate-set 接口" 也是主流；本项目的 candidate-set 接口是刻意选择的、面向 vision-agent 部署的设定 |
| **LAVT — Language-Assisted Vision Transformer** (Yang et al.) | CVPR 2022 | 跨模态分层融合（视觉主干内注入语言 token）做 REC | 我们不用其架构；只说明 "模态融合深度" 与我们的可靠性问题正交 |
| **VTL — Towards Visual Text Language Reasoning** (Sun et al.) | CVPR 2023 | 把 grounding 拆成对固定 object proposals 的 region-text 匹配与多种推理类型标注 | 与我们共享 "proposals 固定、决策在后" 的结构；我们的问题是该结构下**概率是否可信** |
| **ReCLIP — A Strong Zero-Shot Baseline for Referring Expression Comprehension** (Subramanian et al.) | ACL 2022 | CLIP + 定位细化（Grad-CAM）即可成为强 zero-shot REC baseline，并指出 CLIP 的 image-text 打分基本不考虑空间关系 | 直接支撑我们的 B1（frozen CLIP cosine）作为**合理且必要**的 baseline；同时提示 CLIP 空间盲区是需要写明的限制（RefCOCO+ 标注采集禁止绝对位置词、更侧重外观可在一定程度上缓解，但关系型与上下文型表达仍可出现：> RefCOCO+ annotation collection prohibits absolute location words, reducing reliance on simple absolute-position shortcuts, while relational and contextual expressions can still occur.） |
| **Set-of-Mark (SoM) Prompting** (Yang et al.) | arXiv 2023（需查证 venue） | 给图像区域打上显式标记编号，让 GPT-4V 类模型在 "被提供的候选集合" 中选择 | 说明 "把 candidate set 交给模型选择" 是当前 LVLM 的主流交互形式，因此 candidate-set shift 是现实问题；我们不用 LLM，但动机与之相关 |
| **Grounding DINO / 检测式 grounding 基础模型** | 2023（需查证） | 端到端 open-set grounding | **本项目不使用、不微调**（prompt 第二十九节明确禁止 Grounding DINO fine-tuning） |

## 2. Hard negatives / 对抗性候选

| 工作 | 年份 | 一句话贡献 | 与本项目的关系 |
|---|---|---|---|
| **Learning Message-Passing CNNs with Multi-modal Hard Negative Samples** (Zeng et al.) | ICPR 2017（需查证） | 早期即证明多模态 hard negatives 对 phrase grounding 关键 | 我们的 hard regime 不是新技巧，而是**自变量**：我们测的是 reliability 随 hardness 的轨迹 |
| **CAN — Correlation Attention Network (Adversarial Reasoning on Captioned Images)** (Zhang et al.) | ICCV 2019（需查证） | 用对抗性 relation-aware 推理解析空间/关系表达 | 提示 "关系型困难" 主要来自语言端；本项目第一阶段不人工生成 relation negatives，交给 FineCops-Ref |
| **HULA — Hard Negative Augmentation for Visual Grounding** (Shamsian et al.) | CVPR 2022（需查证） | 在特征空间生成 hard negatives 做数据增强 | 属于训练端技巧；本项目禁止 fancy loss 大规模搜索，仅把 hard negatives 用于 evaluation regime |
| **CLIP-hard 候选选择（本项目定义）** | — | 用 frozen CLIP query–crop similarity 挑出最易混淆的错误 proposal | **明确声明为 diagnostic / adversarial evaluation regime**，不是真实 detector 分布；且我们知道 B1 在该 regime 上会因 "用同一模型选负样本" 而系统性变差，这一点必须公开写明而非掩盖 |

## 3. Rejection / NONE / target-absent grounding

| 工作 | 年份 | 一句话贡献 | 与本项目的关系 |
|---|---|---|---|
| **OMG-LLVA / "Not All Regions Are Paved With Gold"** (Ma et al.) | CVPR 2022（需查证） | 在区域选择任务中显式加入 "no-answer / 背景区域" 的处理 | 支撑 "NONE 是可建模的对象" 这一前提；我们的差别是区分 **synthetic omission vs natural proposal miss** |
| **ViNoRe / "When and What: Image Region Selection for Grounded VQA"** (Schneider et al.) | EACL/NAACL 2021–2022（需查证） | 让模型同时决定 "选哪个区域" 与 "是否需要区域（no-region）" | 同上，属前序 "可弃权区域选择" 路线 |
| **GRES / gRefCOCO** (Liu et al.) | CVPR 2023 | 把 referring 任务推广到 zero-target（无匹配）与 multi-target，并提供 GREC/GRES 评测 | 是 "scene-level 目标不存在" 的权威出处；本项目主线更关心 **candidate omission**（目标在图里但候选没提供），把 gRefCOCO 记为后续扩展 |
| **FineCops-Ref** (Liu et al.) | EMNLP 2024 | 可控难度（object/attribute/multi-hop relation）+ 通过细粒度编辑/生成构造 negative text 与 negative images，专测 reject 能力（检索到的 test-set 规模：9,605 positive / 9,814 negative expressions / 8,507 negative images） | **本项目的 external stress test**；我们不参与其榜单，只借用其压力条件检验 reliability 结论是否迁移 |
| **Ref-L4** | 2024（**需查证**：作者、venue 与标注维度未二次确认） | 为 grounding 提供 reasoning length / 难度标签，用于评测 instruction-tuned LVLM | 仅记录为后续扩展候选（难度分层与我们 K/hardness 分层是正交的两种 "难度"） |
| **"Faithful Query–Region Binding for Frozen-Detector Visual Grounding"** 一类工作 | 2026（检索到，**需查证** 具体出处与是否同行评审） | 在冻结检测器 + 语言绑定的设定下讨论 absent-referent 处理 | 说明 "冻结 detector + 后期决策 + 目标缺失" 已有关注；我们必须把自己的设定与之明确区分（我们聚焦 calibration/selective risk 与 gate 式终止） |
| **Flat (K+1)-way NONE、max-confidence / margin threshold（本项目 N0/N1/N2）** | — | 简单弃权机制 | 作为 Phase 2 的 baseline 集合；不声称它们是新颖机制 |

## 4. Calibration

| 工作 | 年份 | 一句话贡献 | 与本项目的关系 |
|---|---|---|---|
| **Probabilistic Outputs for DNNs and Statistical Implications regarding Training / Temperature Scaling** (Guo, Pleiss, Sun, Weinberger) | ICML 2017 | 现代 DNN 普遍过自信；**单参数 temperature scaling** 在保持 accuracy 的同时显著降低 ECE | 我们的 B2/C1 就是它。我们额外测的是：在 K/hardness shift 下**单个全局 T 是否仍然足够**（这正是 Gate Q2 的核心） |
| **Platt scaling / Logistic regression model combination** (Platt 1999；Zadrozny & Elkan 2001–2002) | 1999–2002 | 标量校准与分而治之的校准基线 | 与 C3 同族（scalar → calibrated probability）；我们把它作为 stats-only calibrator 的近邻 |
| **MMCE — Loss Functions for the Calibration of Pixel-wise Object Segmentation and Unlabeled Object Discovery** (Naeini, Cooper, Shepherd, Goldsmith, Cohen) | AAAI 2015 | 提出可微的 Kernel-based calibration error，把校准作为训练目标 | 本项目**不**采用校准损失训练（属 fancy loss 范畴）；只作为 metric 家族参考（MMCE 与 equal-mass binning 的对比） |
| **Measuring Calibration in Deep Learning (ACE / BACE, adaptive binning)** (Nixon et al.) | JVCI 2019 | 系统比较固定宽 bin 与 **adaptive（等质量）bin**，指出 ECE 的分桶敏感性 | 我们主指标 **top-label ECE 使用 adaptive / equal-mass bins** 的直接依据；也要求报告 bin 数敏感性 |
| **Soft Calibration Objectives (SCE, S-AvUC; 指出 L2 + equal-mass binning)** (Karandikar et al.) | NeurIPS 2021 | 提出 soft 校准损失，并在评测中强调 **equal-mass binning**、AvUC 等 | 支撑我们对 bins 的选择；其 AvUC 与 risk–coverage/AUC 家族关系可作为 secondary 讨论 |
| **Calibration Errors and Uncertainty Bars for Grouped Robustness and Adaptation ("All Errors Are Local", ACE-based confidence intervals)** (Varma et al.) | 2022–2023（需查证） | 批评固定分桶 ECE，提出基于 local errors 的校准误差与置信区间 | 与我们的 image-level paired bootstrap 精神一致：**给 reliability 指标配上不确定性** |
| **Can You Trust Neural Network Under Distribution Shift? Investigating Model Calibration in Transfer Learning** (Ovadia et al.) | ICLR 2019 | 显示域移下校准会显著退化，且简单校准器不一定够 | 概念模板：我们把 "域移" 换成 **candidate-set shift**，并加上 gate 式的 "先证明存在退化、再证明简单方法是否足够" |
| **Correcting Confidence Calibration for Out-of-distribution Detection (post-hoc calibration under shift)** (Fang et al.) | NeurIPS 2020（需查证） | 用少量源域信息校正目标域置信度 | 与 C2（K-aware T）思路相近：把 "K" 当作条件变量做外推 |
| **Enabling Calibration in the Zero-shot Inference of Large Vision-Language Models Through Priors** (LeVine et al.) | ICLR 2024（openreview 检索到，卷期需查证） | 发现 CLIP zero-shot 推理存在系统性 miscalibration，并用类先验校正 | 支撑 "CLIP 概率不可信" 的动机；我们的条件变量是 **candidate set**，与之互补 |
| **"CLIP accuracy is related to the number and size of candidate objects"** (Rentschler et al.) | 2023（**需查证**：本轮检索未定位到权威条目） | 报告 CLIP 的分类性能随候选数量与目标尺寸变化 | 若核实，将成为我们 RQ1 最直接的先行证据（K 影响 accuracy）；我们的推进是把 **calibration / selective risk** 与 accuracy **分离** measurement |
| **Multi-class vs top-label calibration 的区分（本项目 §10.1）** | — | 以 \(\hat c, p=\max_i P(c_i), r=\mathbf 1[\hat c=c^*]\) 定义 top-label ECE / binary Brier / correctness NLL | 明确采用 top-label（"我选对了的概率"）而非 multiclass ECE 作为主指标；multiclass NLL/Brier 仅 secondary |

## 5. Selective prediction / learning to reject

| 工作 | 年份 | 一句话贡献 | 与本项目的关系 |
|---|---|---|---|
| **A Baseline for Detecting Misclassified and Out-of-Distribution Examples (reject option)** (Hendrycks & Gimpel) | ICLR 2017（需查证） | 用 max softmax probability 做拒绝/异常检测的 baseline | 我们的 N0（max-confidence threshold）就是它；作为强 baseline 而非创新 |
| **Selective Prediction for Deep Neural Networks / Learning to Reject** (Geifman & El-Yaniv) | NeurIPS 2017 | 形式化 selective prediction，给出与置信度阈值联合优化的拒绝策略 | 我们的 risk–coverage 评估与其框架一致；我们不引入新的 selective loss |
| **On Risk-Coverage for Regression and Classification** (El-Yousef & Vernick) | NeurIPS 2017（需查证） | 给出 risk–coverage 的有限样本刻画与排序统计检验 | 支撑我们把 **AURC / risk@coverage** 作为主指标之一并配 bootstrap CI |
| **Trained Classifiers May Not Achieve Optimal Selective Prediction Accuracy** (Wen, Eichson, Keriven, Pehovan) | ICLR 2020（需查证） | 指出仅靠单一置信度排序无法达到最优选择性精度，需要额外校准 | 与 RQ2 相关：说明 "scalar confidence 不够" 在文献中已有先例，我们把它具体化到 candidate-set shift |
| **On the Calibration of Solvers and Classifiers (conditional risk, AURC 变体)** (Geissinger, Pfister, Giegerich) | ICLR 2022（需查证作者/标题细节） | 系统讨论 AURC 及其条件化变体的性质与缺陷 | 我们报告 AURC 时必须同时给 risk@50/80/90/95 与曲线，避免单值误读 |
| **Population AURC 的形式化统计刻画** (Zhou, Landeghem et al.) | PMLR 2025（检索确认存在，卷号 v267；完整标题需查证） | 给出 population AURC 的等价表达式与解释 | 提醒我们：AURC 的点估计有其随机性 → 与 image-level bootstrap 的做法一致 |
| **False Selection Rate（本项目重点指标）** | — | target absent 时仍选择某 candidate 的比例 | 面向 vision-agent 部署的直接风险度量；不是学术创新，而是问题选择 |

## 6. Set functions / permutation-invariant models

| 工作 | 年份 | 一句话贡献 | 与本项目的关系 |
|---|---|---|---|
| **Deep Sets** (Zaheer et al.) | NeurIPS 2017 | 给出 permutation-invariant 函数的通用形式 \(f(\{x_i\})=\rho(\sum_i \phi(x_i))\) | Phase 1 第一版实现（**不是 Transformer**）；参数量 <0.5M |
| **PointNet** (Qi et al.) | CVPR 2017 | max-pool + point MLP 的集合编码范式（我们的 \(u_{max}\) 与其精神一致） | 说明 mean/max 聚合并非新想法 |
| **Set Transformer** (Lee et al.) | ICML 2019 | 用 attention（含 inducing points）建模集合，并提出 permutation-invariant/equivant head 对比 | 只有在 Gate Q3 需要更强 set 表达时才考虑；**不默认 attention 最优** |
| **Intriguing Properties of Attention Models / attention 的排序敏感性评测** (Zhang et al.) | ICLR 2020（需查证） | 系统评测 attention 模型的置换/顺序性质 | 提醒我们在报告 set 模型时必须验证置换不变性（已列入 sanity check 习惯） |
| **Equivariant / invariant two-layer MLP（本项目 RQ3 备选）** | — | 与 DeepSets 同容量级别的对照 | 我们优先 "简单 set 模型 vs stats-only"，而不是先做 attention 搜索 |

## 7. 与本项目问题最接近的、需要在论文中重点对立的几类工作

1. **"K 会影响 accuracy"**（例如 CLIP 候选数相关工作，需查证）：我们不复述该结论作为贡献，
   而是要求同时报告 \(\Delta\)Accuracy 与 \(\Delta\)Calibration，并禁止把 softmax 分母随 K
   变化解释为 calibration failure。
2. **"CLIP 概率不可信"**（zero-shot 校准文献）：我们不复述该结论，而是问 "不可信程度是否是
   candidate-set 的函数、且能否被 \(T(K)\) 或 stats-only 模型吸收"。
3. **"加入 candidate interaction 提升 grounding"**（KLSP/SSN/VTL/LAVT/TransVG 等）：我们
   **不**以此为目标；Gate Q3 要求 candidate-aware 模型必须超过 **强 stats baseline**，
   而不是只超过 max-confidence，否则判 NO-GO。
4. **"提出新的 NONE/abstention 架构"**（OMG-LLVA/ViNoRe/GRES/FineCops-Ref 一类）：我们把
   NONE 作为**评估条件**（synthetic vs natural、presence-shift 10/25/50/75%），并优先比较
   factorized vs flat 的**校准后果**，而不是提出新头。

## 8. 引用完整性待办（写作前必须核对）

```text
[ ] SSN / KLSP / VTL / LAVT / TransVG / RCCF 的准确 venue 与年份
[ ] TGVCN、Zeng hard negatives、CAN、HULA 的准确出处
[ ] OMG-LLVA / ViNoRe / GRES-gRefCOCO / Ref-L4 的准确出处与任务定义
[ ] Nixon ACE、Karandikar SCE、Varma "All Errors Are Local"、MMCE、Guo TS 的 bib 精确字段
[ ] El-Yousef & Vernick、Geifman & El-Yaniv、Wen et al.、Geissinger et al. 的 selective 文献准确性
[ ] Rentschler et al. "CLIP accuracy is related to the number and size of candidate objects" 是否可核实
[ ] FineCops-Ref 的下载方式、license、是否 test-only
[ ] COCO / RefCOCO 的官方 attribution 文本（README 引用块）
```
