# Paper Blueprint v1 — 基于已冻结证据的论文架构

> **本文档的性质**：写作规划（planning only）。它**不新增任何实验、不新增任何数字、不重算任何统计量**；
> 所有主张必须能回溯到已经提交（committed）的 artifact。论文正文尚未撰写。
>
> **纪律**：
> 1. 每一条 claim 必须带 artifact 出处（文件路径 + 字段/行），写作时按出处转录数字，不得从记忆写数字；
> 2. 「明确禁止」栏来自 `results/final_registry/claims.csv` 的 `prohibited_wording`、各 verdict 的
>    `interpretation_boundary.forbidden`、以及协议中已冻结的禁句；这些禁句在论文里只能出现在
>    显式的否定/边界句中；
> 3. `results/final_registry/` **不重新生成**：主线（C1–C6）取 registry，V2 轴与 RQ4-M1 取
>    per-axis artifact（材料库 `docs/final_result_summary.md` 已按此分栏）；
> 4. 实验程序已冻结（`V2-G/D/M/M3/P CLOSED`、`RQ4-M1 CLOSED`、`EXPERIMENTAL PROGRAM FROZEN`）；
>    任何需要新模型前向 pass、新数据集、新 proposal family、新 intervention 的补充分析都属 future work，
>    **不得**以「审稿人要求」的名义在本文完成。
>
> **论文定位**：*controlled reliability study*（受控可靠性研究），不是「提出新的 candidate-aware 架构」。
> 两条方法线的负结果（V2-M、V2-M3）与一次被否证的机制解释（V2-P2-M）是这一定位的正面支撑。

---

## 0. 全局证据骨架（写作时的一句话主线）

| 层 | 结论（可发表的措辞） | 出处 |
|---|---|---|
| 现象 | selective-reliability degradation grows with candidate count | `results/final_registry/claims.csv` C1 |
| 尺度解释被排除 | temperature / score-scale is not the mechanism | C2、N1 |
| 分数信息不足 | we did not find exploitable additional reliability information from the full score set under the tested lightweight models | C2、N2、N3 |
| 语义含额外信息（温和） | modest, consistent, statistically significant but below practical threshold | C3（A7 = INCONCLUSIVE） |
| 强竞争下被放大 | under GT same-category competition the semantic signal grows monotonically with competitor count | C4、N/A11 |
| 外部语义迁移 | cross-dataset external validation under a shared COCO visual domain | C5 |
| 外部放大不可评估 | hard amplification is NOT ASSESSABLE on RefCOCOg | C6、N7 |
| 跨 backbone | `STRONG_CROSS_BACKBONE_GENERALITY = YES`（Q1 3/3、Q2 3/3） | `results/v2_backbone_generalization/v2a1_result_record.json` |
| 跨标注质量 / 跨语言分布 | `CORE FINDINGS ROBUST TO REVIEWED ANNOTATIONS` / `CORE FINDINGS CROSS-DATASET ROBUST` | `results/v2_data_robustness/d1_reviewed_annotations/verdict.json`、`results/v2_d2_refcoco_lang/c1_c4_verdict.json` |
| 跨 proposal family | `CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST`；P2 扩展 `C1_REPLICATED_ON_THIRD_PROPOSAL_FAMILY`（C1 only; C4 not tested） | `results/v2_proposal_robustness/p1_f5_c4/f5_verdict.json`、`proposal_family_final_table.md` |
| 机制（否证） | 同类语义冗余解释被撤回（P2-A0）；`MECHANISM_PARTIAL` | `results/v2_proposal_robustness/p2_amendment_history.csv`、`p2_m_mechanism/p2_m_verdict.json` |
| 机制（精确分解） | `DECOMPOSITION_REPORTED`，三族均 `CONFIDENCE_CHANGE_HEAVIER` | `results/v2_rq4_mechanism/m1_transition_confidence/verdict.json` |

---

## 1. Introduction

**精确研究问题**
一个基于候选集（candidate-based）的视觉定位打分器，**何时还能可信地相信自己的选择**？
当候选数量与候选语义组成改变时，其 selective reliability 是否退化；退化能否由打分本身（含温度/尺度）
解释；候选语义是否携带额外可靠性信息；以及这些结论是否跨 backbone、数据、proposal family 成立。

**允许的主张**
- 研究对象是 reliability（selective risk / ranking），不是新的 accuracy SOTA；raw accuracy 在 nested
  candidate sets 下部分是结构性的（C1 的 limitation 栏要求如此声明）。
- 单一冻结主干 + 单一冻结 proposal family 的受控设定是**特性**而非缺陷：它把变量隔离到候选制度本身。
- 稳健性被系统复制到：2 个额外 backbone、2 种数据变体（人工复核标注 / 语言分布迁移）、
  2 个额外 proposal family。
- 机制层面本文只承诺一项 **exact statistical accounting decomposition**（§7），并明确它不是因果识别。
- 负结果是被测量的：两次方法尝试失败、一个机制候选解释被否证、一个外部复制不可评估。

**支撑 artifact**
`results/final_registry/claims.csv`（C1–C6）、`experiments.csv`（阶段/门控/修正案时序）、
`docs/research_protocol.md`（§8/§9 程序状态 + RQ4-A1）、`docs/experiment_log.md`、
上表 §0 中的 V2 与 RQ4-M1 verdict 文件。

**表/图候选**
- 主文：Table「证据骨架」（由 §0 缩写为 5–6 行）、Figure「K vs AUROC / E-AURC」总览
  （`results/final_registry/figures/fig1_cardinality_reliability.png`）。
- 附录：完整阶段-修正案-时序表（`results/final_registry/protocol_history.csv`）。

**明确禁止**
- 「accuracy drop alone proves cardinality harm」（C1 prohibited_wording）。
- 「cross-domain visual generalization」（C5 prohibited；三个数据集共享 COCO 视觉域）。
- 「we propose a new candidate-aware architecture that solves it」（两条方法线均为负结果）。
- 「we identified the mechanism / X causes the degradation」（RQ4-M1 forbidden 列表）。
- 「semantic features solve the cardinality problem」（C3 prohibited）。

**主文 vs 附录**：主文只给现象 + 定位 + 贡献清单；阶段协议细节、修正案全文、逐轴判定表进附录。

---

## 2. Problem Formulation

**精确研究问题**
如何形式化「在动态候选集合上的可靠定位」，使得「退化」是可度量、可比较、不被 base rate 混淆的量？

**允许的主张**
- 设定：图像 + 一句自然语言指称；候选集 C_K（|C_K| = K，nested），K ∈ {5, 10, 20, 50}；
  正确性标签由 GT IoU 决定；置信度为候选维 softmax 上的 MSP；打分栈全程冻结。
- 度量族明确区分 **ranking 类**（AUROC_correct、E-AURC、RER@coverage）与 **错误率类**（accuracy）；
  并声明 nested sets 下 accuracy 下降部分为结构性，因此以 ranking/selective 类为主指标（C1 要求）。
- 推断单位：image-cluster paired bootstrap（5000 次，95% CI），因为表达式在图像内相关。
- 该形式化足以支撑 §7 的两因子分解：correctness-state transition（标签）× confidence ranking（分数排序）。

**支撑 artifact**
`results/final_registry/metrics.csv`（T1–T5 每个数值的 source_file 与列名）、
`results/phase0a_cosine/figures/reliability_per_K.png`（reliability map 随 K）、
`src/ccg/reliability/`（指标实现）、RQ4-M1 冻结定义
`results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json`（groups/corners/AUROC 重构式）。

**表/图候选**
- 主文：Table「符号与指标定义」（手写，数字不入表）、Figure「reliability diagram per K」
  （`results/phase0a_cosine/figures/reliability_per_K.png`）。
- 附录：ECE/Brier/NLL 随 K（`results/phase0a_cosine/figures/ece_brier_acc_vs_K.png`）、
  置信度直方图（`confidence_hist_per_K.png`）、风险-覆盖曲线（`risk_coverage_per_K.png`）。

**明确禁止**
- 把 E-AURC/RER 的定义写成「标准」或首次提出（本项目只使用它们，不主张贡献度量学）。
- 主张 raw accuracy 下降即可靠性下降（C1 limitation）。
- 把 temperature scaling 当作可修复手段（N1 已排除）。
- 使用任何未在任何 artifact 中定义的中间量（例如 per-candidate tail pressure：RQ4-M1 明确 `NOT USED`，
  因为冻结的预测 artifact 不含 per-candidate raw scores，计算它将需要被禁止的新模型前向 pass）。

**主文 vs 附录**：主文给完整符号与指标定义（1 页内）；诊断性曲线（ECE/Brier/histogram）进附录。

---

## 3. Experimental Protocol

**精确研究问题**
一套 staged protocol 如何在「结果前冻结」与「必要的结果后纠正」之间保持诚实可审计？

**允许的主张**
- 逐阶段区分时序：A1/A2 为 pre-result 方法学约束；**A5 是结果后可见的 post-hoc 方法学纠正，
  不构成预注册**；A6–A8 在各自阶段结果可见前冻结；A9–A11 为 staged external validation。
  该时序在 `experiments.csv` 的 `amendment_timing` 列逐行登记。
- V2 每条轴各自冻结于其结果之前，且**不改动主线任何数字**；RQ4-M1 为 `DESCRIPTIVE_MECHANISM_DECOMPOSITION`，
  其 freeze 文档写明 `real_results_seen_before_freeze: false`。
- 输入可溯源：RQ4-M1 在读取任何真实 row 之前先建立 18 个预测 artifact 的 sha256 + size manifest，
  结果轮启动时逐一复哈希（`verify_input_manifest`）。
- append-only 治理：撤回解释以台账（P2-A0）记录，不改写原始 gate/threshold。

**支撑 artifact**
`results/final_registry/protocol_history.csv`、`experiments.csv`（commit 号列）、
`docs/research_protocol.md`（含 A1–A11 与 RQ4-A1）、`docs/experiment_log.md`、
`results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json` +
`input_artifact_manifest.csv`、`results/v2_proposal_robustness/p2_amendment_history.csv`、
各轴 config_freeze（`p1_f4_f5_config_freeze.json`、`p2_c1_gdino_config_freeze.json`、
`p2_m_mechanism_config_freeze.json`）。

**表/图候选**
- 主文：Table「阶段 × 冻结时序 × 门控 × 判定」（压缩版，8–10 行）。
- 附录：Table「修正案全表」（`protocol_history.csv` 直出）、
  Table「frozen input artifacts 与 sha256」（manifest 前缀列，可截断哈希）。

**明确禁止**
- 任何 retroactive preregistration 措辞（`This section is a retrospective result record` /
  `It does not retroactively preregister any completed` 必须照抄进相关小节）。
- 把 V2 轴的冻结说成「论文一开始就预注册」；每条轴只对自己的结果预注册。
- 用「所有分析都预注册了」概括 A5（A5 是结果后纠正）。

**主文 vs 附录**：主文一段概述 + 时序表；完整修正案文本、manifest、config freeze 进附录/补充材料。

---

## 4. Candidate-Cardinality Reliability Degradation

**精确研究问题**
RQ1：当候选数量从 5 增到 50，冻结打分栈的 selective reliability 是否系统性退化？退化幅度多大？

**允许的主张**
- 主线（B3, random regime）：K5→K50 的 E-AURC 相对恶化 ≈ +226%、RER@50 0.821→0.365、
  AUROC_correct 0.843→0.790；所有报告的 bootstrap CI 排除 0（C1，support level = Strong：2 scorers × 3 seeds）。
- 该退化跨 backbone（V2-G Q1 3/3 `CARDINALITY_REPLICATED`）、跨复核标注（V2-D1 C1 `ANNOTATION-ROBUST`）、
  跨语言分布（V2-D2 C1 `CROSS-DATASET REPLICATED`）、跨 proposal family（V2-P C1 三族全部通过：
  RPN / DETR-R50 / Grounding DINO）。
- 跨族幅度差异本身是结果的一部分：ΔAUROC(K5−K50) +0.0517 / +0.0852 / **+0.2366**，
  E-AURC 相对恶化 +225.7 % / +455.5 % / **+865.4 %**，RER@50 下降 0.466 / 0.294 / **0.668**。
- 候选数量解释被排除：三族 bank 均 64 框、同一 presented K（V2-P2-M 的算术论证）。

**支撑 artifact**
`results/final_registry/tables.md` Table 1 + `metrics.csv`、`results/phase0b_independent/aggregate.csv`、
`results/v2_backbone_generalization/final_summary.csv`、
`results/v2_data_robustness/d1_reviewed_annotations/verdict.json`、
`results/v2_d2_refcoco_lang/c1_c4_verdict.json`、
`results/v2_proposal_robustness/p2_c1_gdino/c1_point.csv` 与 `proposal_family_final_table.md`、
`results/v2_rq4_mechanism/m1_transition_confidence/point_decomposition.csv`（同一退化的分解）。

**表/图候选**
- 主文：Table「K × {Accuracy, AUROC_correct, E-AURC, RER@50}」（Table 1 直出）；
  Table「三 proposal family 的 C1 幅度对照」；
  Figure `results/v2_proposal_robustness/p2_c1_gdino/figures/fig1_k_vs_auroc_three_families.png`、
  `fig2_k_vs_eaurc_three_families.png`。
- 附录：`results/v2_proposal_robustness/p1_f4_c1/figures/fig1_k_vs_auroc.png` / `fig2_k_vs_eaurc.png`、
  V2-G 的 per-backbone ΔAUROC、V2-D1/D2 的 C1 图
  （`results/v2_data_robustness/d1_reviewed_annotations/figures/fig_d1_c1_cardinality_delta.png`）。

**明确禁止**
- 「accuracy drop alone proves cardinality harm」。
- 把三族幅度差异归因于「GDINO 候选更多/更少」或任何未测量的性质（V2-P2-M 已排除候选数量解释，
  且 RQ4-M1 只给 accounting，不给因果）。
- 主张 GDINO 的 hard-regime / same-category 结果（供给门 0.6892 < 0.85，C4 从未授权，
  `不存在任何 GDINO 强竞争主张`）。
- 「cross-visual-domain」generalisation。

**主文 vs 附录**：主文给主线 Table 1 + 三族对照 + 一张 K-vs-metric 图；backbone 复制、
标注/语言迁移的逐轴数字进附录。

---

## 5. When Candidate Semantics Matter

**精确研究问题**
RQ3：候选语义信息在**什么条件下**为可靠性判断提供打分之外的增量？

**允许的主张**
- 温和但一致：Phase 1 Stats→Stats+Semantic（E1b）K50 ΔAUROC +0.0179 [+0.0140, +0.0216]，
  E-AURC 降低 8.15%，3/3 seed 同向；但**低于预冻结实用阈值**，A7 判 INCONCLUSIVE
  （必须保留 gray-zone 表述：「modest, consistent, statistically significant but below practical threshold」）。
- 强竞争下放大：RefCOCO+ SameCategory-K5 vs matched-random，ΔAUROC 从 +0.0012 升到 +0.0318
  （diff-of-diffs +0.0306 [+0.0266, +0.0348]），E-AURC 降 15.37%，并随 m = 0/2/4/8 单调增强
  （−0.0007 / +0.0106 / +0.0182 / +0.0299）。
- 跨设定复制：V2-G Q2 3/3 `HARD_SEMANTIC_REPLICATED`（放大 A：B1 +0.0223、B2 +0.0311，dose ρ = 1.0）；
  V2-D1 C4 `ANNOTATION-ROBUST`；V2-D2 C4 `CROSS-DATASET REPLICATED`；V2-P C4 在 DETR 复现
  （Δhard +0.0213，CI 排除 0）。
- 外部语义迁移成立但受限：RefCOCOg strict（2909 expr / 1102 imgs，与 RefCOCO+ 零图像重叠）
  same-category ΔAUROC +0.0178、E-AURC 降 20.4%，Q1 = CONFIRMED；措辞必须限定
  「cross-dataset external validation under a shared COCO visual domain」。
- 外部放大**不可评估**：Q2 = NOT ASSESSABLE（manipulation 0/3 seed 显著，clip_margin12 方向不符），
  综合 Case D。
- same-category 是 GT 辅助的诊断性构造，不是自然分布。

**支撑 artifact**
`results/final_registry/claims.csv` C3/C4/C5/C6、`tables.md` Table 2/3、
`results/phase1_semantic_sufficiency/`、`results/phase1f_hard_semantic/`、
`results/phase1e_refcocog_external/`、`results/v2_backbone_generalization/v2a1_result_record.json`、
`results/v2_proposal_robustness/p1_f5_c4/f5_verdict.json`、
`results/v2_data_robustness/d1_reviewed_annotations/verdict.json`、
`results/v2_d2_refcoco_lang/c1_c4_verdict.json`、
`results/v2_proposal_robustness/proposal_family_final_table.md`（GDINO C4 = not tested）。

**表/图候选**
- 主文：Table「random vs SameCategory（+dose response）」；
  Figure `results/phase1f_hard_semantic/figures/fig2_delta_auroc_random_vs_hard.png`、
  `fig3_reduction_and_rer50.png`；外部迁移图
  `results/phase1e_refcocog_external/figures/a11_fig1_delta_auroc.png`、`a11_fig2_eaurc_rer50.png`。
- 附录：`fig1_grounding_accuracy_random_vs_hard.png`、`fig4_cand_vmax_distribution.png`、
  manipulation-check 全表、`results/phase1_semantic_sufficiency/figures/ambiguity_stratified_k50.png`、
  V2-P `p1_f5_c4/figures/fig3_random_vs_hard_increment.png`。

**明确禁止**
- 「deployable natural hard-negative benchmark」（C4 prohibited）。
- 「semantic features solve the cardinality problem」（C3 prohibited）。
- 把 RefCOCOg 的 hard−random 正点估计当作复制（C6 prohibited：FAILED / CONFIRMED / replicated externally）。
- 「unmatched boxes / H2a / semantic ambiguity is the cause」（RQ4-M1 与协议 §21 的禁令仍有效）。
- 任何 GDINO 强竞争/语义结论（未授权）。
- 「composition in general is ruled out」（V2-P2-M 只否证了同类冗余这一具体解释）。

**主文 vs 附录**：主文给 Phase 1F 放大 + dose 曲线 + 外部 Q1；Phase 1 gray-zone 全表、
manipulation check、可行性审计（A9/A10）进附录。

---

## 6. Robustness Across Backbones, Data and Proposal Families

**精确研究问题**
第 4–5 节的两项核心效应（数量退化、强竞争下的语义增量）在替换视觉编码器、替换标注/语言分布、
替换 proposal family 后是否仍然成立？

**允许的主张**
- V2-G：两效应均 3/3 复制，`STRONG_CROSS_BACKBONE_GENERALITY = YES`。
- V2-D1：`C1 ANNOTATION-ROBUST` + `C4 ANNOTATION-ROBUST` → `CORE FINDINGS ROBUST TO REVIEWED ANNOTATIONS`。
  D1 是 **annotation cleanup robustness**，不是新数据集。
- V2-D2：RefCOCO testA∪testB（1494 图 / 3707 指称 / 10 544 行）→ `CORE FINDINGS CROSS-DATASET ROBUST`；
  必须原样引用 artifact 自带边界句：cross-dataset transfer **under a shared COCO visual domain and a
  shared frozen proposal system; NOT cross-visual-domain generalisation**。
  并披露严格的 image-disjoint 前提在 RefCOCO/RefCOCO+ 上**不可能成立**（同批物理图像，
  `strict_image_disjoint_survivors = 0`），因此该轴是重定义后的语言分布迁移。
- V2-P：`CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST`；P2 扩展 `C1_REPLICATED_ON_THIRD_PROPOSAL_FAMILY`
  （括号内原文 C1 only; C4 not tested）。
- 所有 V2 轴均「不改主线数字、不新增训练参数」，且每条轴各自 pre-result 冻结。

**支撑 artifact**
`results/v2_backbone_generalization/{v2a1_result_record.json,final_summary.csv}`、
`results/v2_data_robustness/d1_reviewed_annotations/verdict.json`、
`results/v2_d2_refcoco_lang/{c1_c4_verdict.json,phase1/cohort_audit.json}`、
`results/v2_proposal_robustness/{v2_p_program_summary.md,proposal_family_final_table.md,p2_c1_gdino/p2_c1_verdict.json}`、
`results/v2_proposal_robustness/p2_gdino_probe/probe_report.json`（供给门 0.6892 vs 0.85）。

**表/图候选**
- 主文：Table「轴 × 判定 × 边界」（6 行：G / D1 / D2 / P1 / P2-C1 / P2-M），
  Table「三族 C1/C4 对照」；图沿用 §4/§5 的三族图。
- 附录：V2-G 逐 backbone 的 ΔAUROC 与 manipulation check、D1 的 mapping/same-category 直方图
  （`results/v2_data_robustness/d1_reviewed_annotations/figures/`）、D2 的 cohort audit。

**明确禁止**
- 任何 unqualified「cross-visual-domain」表述。
- 把 D1 写成「新数据集」、把 D2 写成「图像不相交迁移」。
- 对 GDINO 作 C4 表述（`GDINO C4` / `GDINO hard-semantic` 类句式只能在显式禁止/边界句中出现）。
- 把 V2-P 的 C1 幅度差异当作机制结论（机制在 §7）。

**主文 vs 附录**：主文只给判定表 + 一句边界；逐 seed / 逐 backbone 明细进附录。

---

## 7. Mechanism Analysis: Transition–Confidence Decomposition

**精确研究问题**
RQ4-M1：已发表的 K5→K50 AUROC 退化，能否在**不做任何拟合、不引入任何新变量**的前提下，被**精确地**
记入两个可测因子——correctness-state transition 与 confidence-ranking change？跨族差异又落在哪个因子上？

**允许的主张**
- 分解定义与恒等式（对称两因子 Shapley）：`D_label = −L`、`D_conf = −C`，
  `L = 0.5[(A10−A00)+(A11−A01)]`、`C = 0.5[(A01−A00)+(A11−A10)]`；恒等式
  `L + C = A11 − A00`、`D_total = D_label + D_conf`，实测 max residual = 0
  （group-reconstruction residual ≤ 1.11e-16）。
- Headline（三种子均值，`point_decomposition.csv` mean3）：
  RPN D_total 0.051698 = D_label 0.002605 + D_conf 0.049093；
  DETR 0.085157 = 0.006166 + 0.078991；
  GDINO 0.236599 = **−0.009145** + 0.245744。
  primary verdict `DECOMPOSITION_REPORTED`；三族 × 三种子全部 `CONFIDENCE_CHANGE_HEAVIER`。
- 允许的概括句：*Across all three proposal families, the K5→K50 AUROC degradation is accounted for
  predominantly by changes in the ranking of confidence with respect to correctness, rather than by the
  correctness-state transition component itself.*
  GDINO 句：*The unusually large degradation is almost entirely accounted for by the confidence-ranking
  component in the exact decomposition; the correctness-transition component is slightly negative.*
  紧随边界：*This is an exact statistical accounting decomposition, not a causal identification of why the
  confidence ranking changes.*
- 不确定性（own-cohort，image-cluster，5000，95% CI）：D_conf 三族区间全部排除 0；
  **D_label 三族区间全部包含 0**（GDINO [−0.022816, +0.004701] 虽点估计为负，也不得称显著）；
  D_total 三族区间排除 0。跨族 gap：GDINO−RPN +0.184902 = −0.011750 + 0.196651，
  GDINO−DETR +0.151442 = −0.015311 + 0.166753，DETR−RPN +0.033459 = +0.003561 + 0.029898。
  这些 own-cohort 区间必须命名为 **descriptive under independent-family resampling**，**不是** paired-expression CI。
- flip-rate 悖论（必须主动写，审稿人必问）：GDINO 的 F 组（K5 对而 K50 错）占该族行数
  40.0 % / 40.5 % / 40.0 %，但 AUROC 是 ranking statistic、不是 error-rate statistic；D_label 在固定分数与
  固定分组权重下衡量标签翻转的排序效应，而 F 组在 K50 的平均置信与一直错误的 E 组几乎同区段
  （0.8266 / 0.8275 / 0.8188 vs 0.8303 / 0.8289 / 0.8253），成对得失相互抵消。**high flip rate 与 small
  D_label 相容**。
- matched-expression 次要支持（shared image-cluster draws）：GDINO∩RPN gap_total +0.177824
  （label −0.011598 / conf +0.189421）、GDINO∩DETR +0.134950（−0.012170 / +0.147120）、
  DETR∩RPN +0.022697（CI 跨 0）。只可说：*the confidence-heavy ordering survives matched-expression
  restriction*；**不得**声称 DETR-vs-RPN 幅度差异显著。
- 前置的否证结果属于本节：V2-P2-M 判 `MECHANISM_PARTIAL`，同类冗余与逐表达式伤害只弱相关
  （ρ −0.014 / +0.089 / +0.053），按冗余五分位匹配后跨族差距完全不收缩（shrinkage −0.180 / −0.065，阈值 0.50）；
  因此旧解释以台账 P2-A0 撤回（prose only，gates/thresholds 全未变）。
- 零 delta 声明：本轮 `new_training_parameters: 0`、`new_model_forward: 0`、`new_features: 0`、
  `new_parameters: 0`、`gpu_required: false`；输入仅 18 个已哈希预测 artifact。

**支撑 artifact**
`results/v2_rq4_mechanism/m1_transition_confidence/`：`protocol_freeze.json`（定义/边界/停止条件/manifest sha）、
`input_artifact_manifest.csv`、`point_decomposition.csv`、`bootstrap_decomposition.csv`、
`transition_groups.csv`、`pairwise_auc_components.csv`、`selective_error_sources.csv`、
`cross_family_gap_decomposition.csv`、`matched_intersection_decomposition.csv`（family_mean 分量）、
`verdict.json`、`report.md`（§6 含 paired gap 表）、`metadata.json`。
前置：`results/v2_proposal_robustness/p2_m_mechanism/{p2_m_verdict.json,mechanism_report.md}`、
`results/v2_proposal_robustness/p2_amendment_history.csv`。

**表/图候选**
- 主文：Table「family × {A00, A11, D_total, D_label, D_conf, verdict}」；
  Figure `figures/fig1_signed_decomposition.png`（signed stacked bar）、
  `figures/fig3_cross_family_gap_components.png`（gap 分解）。
- 主文或附录：Figure `figures/fig2_transition_group_confidence.png`（S/F/E 的 p50 分布，flip-rate 讨论配图）。
- 附录：`pairwise_auc_components.csv`（角点重构与 residual）、
  `selective_error_sources.csv`（selective accepted-error 的 F/E 来源随 coverage）、
  `transition_groups.csv` 全表、`report.md` 全文可作补充材料。

**明确禁止**
- `X causes the degradation` / `We discovered the true mechanism` /
  `unmatched boxes / H2a / semantic ambiguity is the cause` / 任何无独立 intervention 协议的因果或干预表述
  （`verdict.json.interpretation_boundary.forbidden`，逐字引用）。
- 措辞规则：只用 *accounts for in the exact statistical decomposition*，不得用 *causes* / *is the causal mechanism*。
- 不得声称任何族的 D_label 显著（三族区间均含 0）。
- 不得把 cross-family gap 的 own-cohort CI 说成 paired CI。
- 不得引入新的 mechanism variable 或补算 tail-pressure（冻结为 `NOT USED`，理由：per-candidate raw scores
  不在冻结 artifact 中，计算它需要被禁止的新模型前向 pass）。
- 不得声称 DETR 与 RPN 的退化幅度差异显著（paired gap CI 跨 0）。
- 不得把「不可计算」写成「已有等价量」（`duplicate_of_existing_claim: NOT MADE`）。

**主文 vs 附录**：主文给 headline 表 + fig1/fig3 + 一段边界声明 + flip-rate 说明（约 1.25 页）；
角点重构、selective 支持、manifest 与 identity 校验进附录。

---

## 8. Attempts at Reliability Modeling and Negative Results

**精确研究问题**
能否用**打分侧**或**语义侧**的轻量模型把 K20/K50 的退化补回来？（答案：两条线都没有成功。）

**允许的主张**
- 分数侧（C2 + N2 + N3）：MSP 即最佳 score-only 模型；handcrafted stats、logK、entropy、
  乃至完整 score-set DeepSets 都不能稳定消除退化（DeepSets K50 AUROC 0.553 vs stats 0.792）。
  允许的上限句：*we did not find exploitable additional reliability information from the full score set
  under the tested lightweight models.*
- 尺度侧（N1）：修正后的 global 温度为 interior 最优，per-K oracle 温度仅约 10% 漂移。
- 语义侧学习尝试（N4 + N5）：E2-learned interaction 与 E3-full-set 均输给简单语义统计
  （E2 0.7878 < E1b 0.8069；E3 −0.034…−0.045，CI 全 <0）。
- 学习型局部竞争模块（V2-M）：M1 gate 未通过（ΔAUROC −0.0294，CI 下界 −0.0324，3/3 seed 负向，
  gate verdict = false）；M2 仍未通过（LCR − E1b −0.0019，CI [−0.0053, +0.0016]）；
  按冻结规则记为 **negative result** 并停止该路线，`V2-MG_NOT_AUTHORIZED`。
- 诊断与后续（M2.5 / M3）：`SPECIALIST TRADEOFF PRESENT`（m=8 +0.0258 [0.0224, 0.0293]，
  m=0 −0.0089）为 **diagnostic only**；据此立项的 M3 自适应混合判
  `ADAPTIVE MIXTURE NOT SUPPORTED`（0/2 backbone，ΔMacroAUROC +0.0001 / +0.0004，CI 跨 0）。
- 外部路线被主动停止（N6）：FineCops 上冻结 RPN 的 target recall@0.5 = 0.7579 < 0.80，
  外部路线 STOPPED（proposal-domain mismatch 会把 A8 效应与感知偏移混淆）。
- 综合：**没有端到端定位性能提升主张**；这些负结果正是「受控可靠性研究」定位的正面证据。

**支撑 artifact**
`results/final_registry/claims.csv` C2、`negative_results.csv` N1–N7、
`results/phase05_score_sufficiency/`、`results/phase1_semantic_sufficiency/`、
`results/v2_local_competition/{gate.json,m2_curriculum/gate.json,m25_specialist_audit/verdict.json,m3_mixture/conf/cross_backbone_verdict.json,m3_mixture/conf/m3_conf_audit.json}`、
`results/phase1e_finecops/`（STOPPED 路线）。

**表/图候选**
- 主文：Table「information-sufficiency ladder」（Table 2 直出）；
  Table「两次方法线尝试 × gate × 判定」；
  Figure `results/final_registry/figures/fig2_information_ladder.png`。
- 附录：`results/v2_local_competition/figures/m1_cohort_delta_auroc.png`、
  `m2_curriculum/figures/m2_severity_curve.png`、`m25_specialist_audit/figures/m25_specialist_audit.png`、
  `m3_mixture/conf/b1|b2/figures/m3_conf_*_macro_delta.png`、
  `results/phase05_score_sufficiency/figures/score_only_reliability_vs_K.png`、
  FineCops 可行性图 `results/phase1e_finecops/figures/fig1_finecops_feasibility.png`。

**明确禁止**
- 「the score set contains no information / softmax scale mismatch only」（C2 prohibited）。
- 「LCR learned nothing」这类无边界表述（治理测试把它列为必须带否定语境的句式）。
- 把 M2.5 的诊断当 confirmatory（`Diagnostic only` 前缀必须随引）。
- 把 M1 的失败描述成「CI 上界 < 0」（artifact 根本没有 ci_high 字段；记录必须如实说明）。
- 声称任何方法带来部署级增益。

**主文 vs 附录**：主文给 ladder 表 + 负结果汇总表（1 页内，诚实但简短）；
逐 backbone 的 M3 曲线与 gate.json 细节进附录。

---

## 9. Limitations

**精确研究问题**
本证据集合的边界在哪里，哪些结论是被制度而非被数据限制的？

**允许的主张（逐条都有 artifact 出处）**
- 单一冻结主干（OpenCLIP ViT-B/32）；V2-G 只把它复制到两个额外编码器，仍在同一 COCO 图像域。
- proposal family 三个，但共享同一冻结打分栈与同一图像域；GDINO 仅进入 C1
  （same-category 供给门 0.6892 < 0.85 失败，C4 从未授权）。
- RefCOCO+/RefCOCOg/RefCOCO 共享 COCO 视觉域；V2-D2 是语言分布迁移，**不是** image-disjoint 迁移
  （strict image-disjoint survivors = 0，不可行的前提被如实披露）。
- same-category 为 GT 辅助诊断构造，非自然分布。
- 外部 hard 放大不可评估（A11 Q2 = NOT ASSESSABLE / INVALID_MANIPULATION）。
- random regime 下语义增益温和（A7 = INCONCLUSIVE，gray zone 保留）。
- 两条方法线均为负结果 ⇒ 无端到端性能提升主张。
- 无跨视觉域复制；主线内无 target-absence 结果。
- 机制边界：候选数量解释被排除（三族 64 框、同一 presented K）；同类冗余解释被否证；
  RQ4-M1 只给 exact accounting；**what causally produces the confidence-ranking change 仍未回答**，
  本文标为 `OUT OF SCOPE FOR THIS PAPER`、`NO RQ4-M2 PLANNED`。
- 可计算性限制：per-candidate raw scores 不在冻结预测 artifact 中，因此 added-candidate tail-pressure
  类量在本轮 `NOT USED`（不可计算 ≠ 已有等价量）。
- 工程披露：结果轮曾因绘图 API 调用崩溃在所有计算完成后、artifact 写出前；修复严格限于渲染层
  并调整写出顺序（数值 artifact 先落盘），该披露属可复现性诚实项。

**支撑 artifact**
`results/final_registry/claims.csv`（limitations 列）、`docs/final_result_summary.md` §1/§2 局限段、
`results/v2_proposal_robustness/p2_gdino_probe/probe_report.json`、
`results/v2_d2_refcoco_lang/phase1/cohort_audit.json`、
`results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json`（excluded_from_primary）、
`docs/experiment_log.md`（RQ4-M1 result record + close-out record）。

**表/图候选**：主文无表；可选一张「限制 × 对应证据」清单（由上面条目直接排版）。

**明确禁止**
- 在 Limitations 里补做未授权的分析来「填补」限制。
- 把 NOT ASSESSABLE 写成 FAILED 或 CONFIRMED。
- 删除或改写任何历史 NOT STARTED / 撤回记录（append-only）。

**主文 vs 附录**：全部在主文（约 0.75 页，逐条带引用）。

---

## 10. Conclusion

**精确研究问题**
一句话：本工作对「候选集变化下的可靠视觉定位」给出的、可被 artifact 完全支撑的结论是什么？

**允许的主张**
- 现象：candidate-count growth 造成可测量的 selective-reliability degradation，跨 backbone、
  标注质量、语言分布与 proposal family 复制（判定串逐字沿用 §0）。
- 解释边界：不是温度/尺度、不是所测试的 score-only 表示能补回；语义增量存在但在随机制度下温和，
  在受控同类强竞争下放大且呈剂量反应。
- 机制：一项 **exact statistical accounting decomposition** 把退化主要记入 confidence-ranking 因子，
  跨族差异同样几乎全部落在该因子；这不是因果识别。
- 负结果：两次可靠性建模尝试与一个机制候选解释均被如实登记。
- 开放问题：what causally produces the confidence-ranking change（future work，需另立
  pre-result frozen intervention 协议）。

**支撑 artifact**：§0 表全部；结论不引入任何新的数字来源。

**表/图候选**：无新表；可用「证据骨架」表收尾。

**明确禁止**
- 任何因果 / 「we discovered the true mechanism」表述（RQ4-M1 forbidden 逐字）。
- 「first benchmark for …」、部署级或 SOTA 主张、跨视觉域泛化主张。
- 「GDINO 的 C4 / hard-regime」任何形式的存在性主张。
- 把 DETR-vs-RPN 的差异说成显著。

**主文 vs 附录**：主文一段（≤150 词）；无附录材料。

---

## 11. 全局排版建议（主文 8–9 页口径）

| 章节 | 预算 | 主文必放 | 移入附录 |
|---|---|---|---|
| 1 Introduction | 1.0 页 | 贡献清单（4 条：现象 / 稳健性 / 精确分解 / 负结果） | — |
| 2 Problem Formulation | 1.0 页 | 符号、指标族、nested-set 结构性声明 | ECE/Brier 诊断 |
| 3 Protocol | 0.75 页 | 时序表（pre/post-result 标注） | 修正案全表、manifest |
| 4 Cardinality | 1.25 页 | Table 1、三族对照、fig1/fig2 三族图 | 逐 seed 明细 |
| 5 Semantics | 1.25 页 | random vs hard + dose、外部 Q1、C6 不可评估 | manipulation check、可行性 |
| 6 Robustness | 0.75 页 | 轴 × 判定 × 边界表 | 逐 backbone/轴明细 |
| 7 Mechanism | 1.25 页 | headline 表、fig1/fig3、flip-rate 段、边界句 | 角点重构、selective 支持 |
| 8 Negative results | 0.75 页 | ladder 表 + 尝试汇总表 | M3 曲线、gate.json |
| 9 Limitations | 0.75 页 | 全部在主文 | — |
| 10 Conclusion | 0.25 页 | 一段 | — |

标题候选（沿用材料库，暂不定稿）：
*Reliable Visual Grounding under Dynamic Candidate Sets*；
*When Should a Visual Grounding Model Trust Its Choice?*；
*Candidate-Set Shift and Semantic Ambiguity in Reliable Visual Grounding*.

---

## 12. 写作时必须原样引用的边界句（清单）

1. V2-D2：cross-dataset transfer **under a shared COCO visual domain and a shared frozen proposal system;
   NOT cross-visual-domain generalisation**。
2. RQ4-M1：**This is an exact statistical accounting decomposition, not a causal identification of why the
   confidence ranking changes.**
3. V2-P2-M：该诊断 **does not explain the between-family amplification**（分类 DESCRIPTIVE_MECHANISM）。
4. A11 Q2：**NOT ASSESSABLE**（不是 FAILED，不是 CONFIRMED）。
5. C2 上限句：**we did not find exploitable additional reliability information from the full score set under
   the tested lightweight models.**
6. 结果登记节的 retrospective 声明：**This section is a retrospective result record / It does not
   retroactively preregister any completed**（描述治理时引用）。
7. V2-P：GDINO **C1 only; C4 not tested**，且不存在任何 GDINO 强竞争主张。

---

## 13. 已知证据缺口（审稿人可能问、但当前 artifact 无法回答）

- 置信度排序**为何**变化（无 intervention 授权，属 future work）。
- GDINO 强竞争/歧义行为（供给门失败，C4 从未授权）。
- 跨视觉域（非 COCO 图像域）复制（无 artifact）。
- target-absence（图中无目标）制度下的可靠性（主线无该结果）。
- added-candidate 尾部压力类量（冻结 artifact 缺 per-candidate raw scores；补算需要新模型前向 pass，被禁止）。
- 自然 hard-negative 基准（same-category 为 GT 辅助构造）。

若需回答其中任何一条：**先另立结果前冻结的协议，再谈实验**；不在本文的写作轮里补做。

---

## 14. STOP 条件

本蓝图完成后不撰写正文。下一步动作（章节草稿、图表定稿、投稿目标选择）需明确指示。
