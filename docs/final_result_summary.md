# Final Result Summary — Candidate-Set Shift and Semantic Ambiguity in Reliable Visual Grounding

> Material bank for the paper, **not** the final abstract. All mainline numbers in §1–§2 are the
> frozen values in `results/final_registry/` (derived from phase artifacts by
> `scripts/build_final_registry.py`, consistency-audited against `docs/experiment_log.md`).
> The candidate-cardinality / semantic-reliability mainline is **frozen** after A11:
> no new benchmark, backbone, architecture, reranking, or feature redesign.
>
> The **V2 axes** (the V2 subsections below) are separate protocols, each frozen before its own
> results, run *after* that mainline freeze. They test whether the mainline findings survive a
> change of backbone, of annotation quality, of language distribution and of proposal family, and
> they record the negative results of two attempted methods. **They change no mainline number.**
> Their values come from the per-axis artifacts cited inline; `results/final_registry/` has **not**
> been regenerated for V2, so do not expect to find V2 values there.

---

## 1. 中文技术摘要（约 1000 字）

### 问题（Problem）
视觉定位（visual grounding）模型在给定一张图与一句自然语言指称后，需从动态候选框集合中选出目标。
本研究问的不是「如何把定位精度刷得更高」，而是一个**可靠性（reliability）**问题：当候选数量与候选
语义组成变化时，模型「相信自己的选择」这一能力是否仍然可信；以及这种可信度的变化能否仅由打分本身
解释，还是必须借助候选的语义信息。所有实验基于**单一冻结**的 OpenCLIP ViT-B/32 backbone 与**单一冻结**
的 class-agnostic RPN proposal 家族，全程零训练参数改动（除早期一次性训练的 candidate-blind 打分器 B3）。

### 协议（Protocol）
采用 staged protocol，并如实区分 pre-result 与 post-result 修订：候选制度 taxonomy 与 proposal bank
定义（A1、A2）为结果前方法学约束；度量与温度有效性修正（A5）是**结果后可见的 post-hoc 方法学纠正，
不构成预注册**；Phase 0.5/1/1F 的模型 zoo 与 gate（A6、A7、A8）均在各自阶段结果可见前冻结；FineCops
与 RefCOCOg 外部路线（A9、A10、A11）为 staged external validation。统一使用 image-cluster paired
bootstrap（5000 次，95% CI）。

### 关键发现（Key findings）
- **A｜候选数量退化可靠性**：B3 上 K5→K50 时 E-AURC 相对恶化约 +226%、RER@50 从 0.821 降到 0.365、
  AUROC_correct 从 0.843 降到 0.790（CI 均不跨 0）。因 nested candidate sets，raw accuracy 下降部分是
  结构性结果，论文重点放在 E-AURC / AUROC_correct / RER@coverage。
- **B｜温度/打分尺度不是解释**：修正温度优化边界后 global 温度为 interior 最优，per-K oracle 温度仅约
  10% 漂移，temperature scaling 无法修复 selective reliability 退化。
- **C｜打分信息不足**：MSP、margin、entropy、手工统计、logK、乃至完整 score-set DeepSets 都不能稳定消除
  K20/K50 退化；仅措辞为「在所测试的轻量模型下未发现可从完整分数集中额外利用的可靠性信息」，而非
  「分数集不含信息」。
- **D｜候选语义含少量额外信息**：Phase 1 中 Stats→Stats+Semantic 在 random 分布下一致、显著但幅度低于
  预冻结实用阈值，A7 判定保持 INCONCLUSIVE（保留该 gray-zone 结果）。
- **E｜受控强竞争下语义价值被放大**：RefCOCO+ SameCategory-K5 相对 matched-random 把 ΔAUROC 从 +0.0012
  提升到 +0.0318，E-AURC 降低 15.37%，且随同类竞争者数 m=0/2/4/8 单调增强——这是最重要的内部结果。
  same-category 为 GT 辅助的诊断性构造，非自然分布。

### 外部验证（External validation）
在严格 image-disjoint 的 RefCOCOg 子集（2909 表达式 / 1102 图，与所有 RefCOCO+ 图像零重叠）上，冻结模型
zero-shot 外部确认：same-category ΔAUROC +0.0178、E-AURC 降低 20.4%，Q1 语义迁移 = **CONFIRMED**。措辞严格
限定为「在共享 COCO 视觉域下的跨数据集外部验证」，禁止写「跨视觉域泛化」。但 Q2 hard-放大复制为
**NOT ASSESSABLE**：RefCOCOg 的 same-category 操纵未显著提高 cand_vmax / cand_top12_sim，且 clip_margin12
方向不符，故不得用 hard−random 的正点估计声称复制。综合判定 Case D（放大轴 EXTERNAL INCONCLUSIVE）。

### V2 稳健性轴（A11 之后独立冻结的协议；不改动主线任何数字）
- **V2-G｜跨 backbone**：把冻结打分栈换到 OpenCLIP B/16 与 SigLIP B/16，两个核心效应均复制——
  Q1 candidate-cardinality 3/3 `CARDINALITY_REPLICATED`（B1 ΔAUROC(K5−K50) +0.0540、B2 +0.0751），
  Q2 hard-semantic 3/3 `HARD_SEMANTIC_REPLICATED`（放大 A：B1 +0.0223、B2 +0.0311；manipulation 3/3；
  dose ρ = 1.0）；`STRONG_CROSS_BACKBONE_GENERALITY = YES`。
  工件：`results/v2_backbone_generalization/v2a1_result_record.json`、`results/v2_backbone_generalization/final_summary.csv`。
- **V2-D1｜人工复核标注**：`C1 ANNOTATION-ROBUST` + `C4 ANNOTATION-ROBUST`，综合
  `CORE FINDINGS ROBUST TO REVIEWED ANNOTATIONS`。
  工件：`results/v2_data_robustness/d1_reviewed_annotations/verdict.json`。
- **V2-D2｜语言分布迁移（不是图像不相交迁移）**：RefCOCO testA∪testB（1494 图 / 3707 指称 / 10 544 行，
  与 RefCOCO+ 是同一批物理图像）上 `C1 CROSS-DATASET REPLICATED` + `C4 CROSS-DATASET REPLICATED`，
  综合 `CORE FINDINGS CROSS-DATASET ROBUST`；C1 ΔAUROC +0.0469。工件自带的边界声明须照抄：
  “cross-dataset transfer under a shared COCO visual domain and a shared frozen proposal system;
  NOT cross-visual-domain generalisation”。
  工件：`results/v2_d2_refcoco_lang/c1_c4_verdict.json`。
- **V2-M / V2-M3｜两条方法线，均为负结果**：M1（random-only 训练）`M1_GO` 未通过
  （LCR − E1b 的 ΔAUROC −0.0294，CI 下界 −0.0324，三个 seed 方向全为负，gate verdict = False）；
  M2（竞争课程训练）仍未通过（−0.0019，CI [−0.0053, +0.0016]），按冻结规则 LCR v1 记为
  **negative result** 并停止该路线。M2.5 诊断给出
  `SPECIALIST TRADEOFF PRESENT`（课程 E1b 在 m=8 强竞争下显著更好 +0.0258 [0.0224, 0.0293]，
  在 m=0 无竞争下显著更差 −0.0089）；据此立项的 M3 自适应混合在确认阶段判
  `ADAPTIVE MIXTURE NOT SUPPORTED`（0/2 backbone，ΔMacroAUROC +0.0001 / +0.0004，CI 均跨 0）。
  工件：`results/v2_local_competition/gate.json`、`results/v2_local_competition/m2_curriculum/gate.json`、
  `results/v2_local_competition/m25_specialist_audit/verdict.json`、
  `results/v2_local_competition/m3_mixture/conf/cross_backbone_verdict.json`。
- **V2-P｜proposal family 轴（最新，已收束）**：只改变 proposal family、其余不变量全部冻结。
  C1 在 RPN / DETR-R50 / Grounding DINO（class-prompt）三族全部通过——ΔAUROC(K5−K50)
  +0.0517 / +0.0852 / **+0.2366**，E-AURC 相对恶化 +225.7 % / +455.5 % / **+865.4 %**，
  RER@50 下降 0.466 / 0.294 / **0.668**；C4 在 DETR 复现（Δhard +0.0213，CI 排除 0）；GDINO 因
  same-category 供给门（probe 0.6892 < 0.85）判为 out-of-scope，**不存在任何 GDINO 强竞争主张**。
  P1 综合 `CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST`；P2 扩展在总表中的判定为
  `C1_REPLICATED_ON_THIRD_PROPOSAL_FAMILY`（括号内原文：C1 only; C4 not tested）。
  机制诊断 V2-P2-M（分类为 DESCRIPTIVE_MECHANISM，不得创造或削弱任何存在性主张）判
  **MECHANISM_PARTIAL**：族内同类冗余与逐表达式伤害只弱相关（ρ：RPN −0.014 / DETR +0.089 /
  GDINO +0.053），按冗余做五分位匹配后跨族差距**完全不收缩**
  （shrinkage −0.180 / −0.065，阈值 0.50）——GDINO 的同类干扰物**最少**（R1 6.23，对比 DETR 8.71 /
  RPN 11.51），伤害却**最大**（H1c 净伤害 0.4016）。因此此前「同类语义竞争解释了最强放大」的机制
  解读**已被撤回**（台账 P2-A0）；候选数量解释仍被排除（三族 bank 均 64 框、同一 presented K，属算术）。
  工件：`results/v2_proposal_robustness/v2_p_program_summary.md`（索引）、
  `results/v2_proposal_robustness/p2_amendment_history.csv`（台账）、
  `results/v2_proposal_robustness/proposal_family_final_table.md`、
  `results/v2_proposal_robustness/p2_m_mechanism/mechanism_report.md`。

### 局限（Limitations）
主线 backbone 仍是单一冻结的 OpenCLIP ViT-B/32（V2-G 只是把它复制到另两个视觉编码器，仍在同一 COCO
图像域内）；proposal 家族已复制到三个（RPN / DETR-R50 / Grounding DINO），但三者共享同一冻结打分栈与
同一图像域，且 GDINO 只进入 C1（same-category 供给门 0.6892 < 0.85 失败，C4 从未授权）；
RefCOCO+/RefCOCOg/RefCOCO 共享 COCO 视觉域，V2-D2 是语言分布迁移而非图像不相交迁移；same-category 为
GT 辅助诊断构造；外部 hard 放大不可评估（A11 Q2 = NOT ASSESSABLE）；random 分布外语义增益幅度温和；
LCR v1 与自适应混合两条方法线均为负结果，无端到端定位性能提升主张；无跨视觉域复制；主线内无
target-absence 结果；GDINO 放大现象的机制仍是**开放问题**（首个候选解释已被否证，未提出替代解释）。

---

## 2. English technical summary (≈ 350 words)

**Problem.** We study the *reliability* of a candidate-based visual grounding scorer under dynamic
candidate sets, not raw accuracy. Given a frozen image–text backbone and a frozen class-agnostic
proposal family, we ask whether the model's confidence in its own choice remains trustworthy as the
number and the semantic composition of candidates change, and whether such changes can be explained
by the grounding scores alone.

**Protocol.** A staged protocol distinguishes pre-result constraints (regime taxonomy, proposal-bank
definition) from a post-result methodological correction (metric/base-rate validity and the temperature
optimization boundary — explicitly *not* a preregistration) and from stage gates that were frozen before
their own results (score sufficiency, semantic sufficiency, hard competition, external confirmation).
All inference uses image-cluster paired bootstrap (5000 replicates, 95% CI) with zero new training in the
later stages.

**Key findings.** (A) As K grows 5→50, the candidate-independent scorer degrades on base-rate-aware
selective metrics (E-AURC relative worsening ≈ +226%; RER@50 0.821→0.365; AUROC_correct 0.843→0.790);
the raw accuracy decline is partly structural under nested sets. (B) Corrected global temperature is an
interior optimum and per-K oracle temperatures shift only ~10%, so score scaling is not the mechanism.
(C) No score-only representation — MSP, margin, entropy, handcrafted statistics, log K, or a full
score-set DeepSets — removes the degradation: *we did not find exploitable additional reliability
information from the full score set under the tested lightweight models.* (D) Candidate semantics add a
modest but consistent, significant signal (Phase 1, gray zone → A7 INCONCLUSIVE). (E) Under controlled GT
same-category competition the semantic increment grows from ΔAUROC +0.0012 (random) to +0.0318 and rises
monotonically with the number of same-category competitors (m = 0/2/4/8).

**External validation.** On a strict image-disjoint RefCOCOg subset (2909 expressions / 1102 images, zero
overlap with all RefCOCO+), the frozen models transfer semantically: same-category ΔAUROC +0.0178 with a
20.4% E-AURC reduction, Q1 semantic transfer = CONFIRMED. This is *cross-dataset external validation under a
shared COCO visual domain*, not cross-domain visual generalization. Q2 hard-amplification replication is
**NOT ASSESSABLE** because the RefCOCOg manipulation did not reliably increase semantic ambiguity; the
positive hard-minus-random point estimate is not claimed as replication. Overall verdict: Case D.

**V2 robustness axes (each frozen before its own results; no mainline number is touched).**
*V2-G (backbone).* With the scoring stack transferred to OpenCLIP B/16 and SigLIP B/16, both core
effects replicate: cardinality degradation 3/3 `CARDINALITY_REPLICATED` (ΔAUROC K5→K50 +0.0540 /
+0.0751) and hard-semantic amplification 3/3 `HARD_SEMANTIC_REPLICATED` (+0.0223 / +0.0311,
manipulation 3/3, dose ρ = 1.0); `STRONG_CROSS_BACKBONE_GENERALITY = YES`
(`results/v2_backbone_generalization/v2a1_result_record.json`).
*V2-D1 (annotation quality).* On manually reviewed RefCOCO+ both are `ANNOTATION-ROBUST`, overall
`CORE FINDINGS ROBUST TO REVIEWED ANNOTATIONS`.
*V2-D2 (language distribution).* On RefCOCO testA∪testB (1494 images / 3707 refs / 10 544 rows — the
*same physical images* as RefCOCO+, development images unseen) both replicate: ΔAUROC +0.0469, overall
`CORE FINDINGS CROSS-DATASET ROBUST`. The artifact's own boundary must be quoted: cross-dataset
transfer under a shared COCO visual domain and a shared frozen proposal system, **not**
cross-visual-domain generalisation (`results/v2_d2_refcoco_lang/c1_c4_verdict.json`).
*V2-M / V2-M3 (two method lines, both negative).* The learned local-competition module failed its
gate at M1 (ΔAUROC −0.0294, CI low −0.0324, negative in all three seeds, gate verdict = false) and again at
M2 (LCR − E1b −0.0019, CI [−0.0053, +0.0016]); per the frozen rule LCR v1 is recorded as a
**negative result** and the route was stopped. The M2.5 diagnostic found `SPECIALIST TRADEOFF
PRESENT` (curriculum E1b better at m = 8,
+0.0258 [0.0224, 0.0293]; worse at m = 0, −0.0089), which motivated M3; the confirmatory adaptive
mixture then returned `ADAPTIVE MIXTURE NOT SUPPORTED` (0/2 backbones, ΔMacroAUROC +0.0001 / +0.0004,
both CIs straddling 0). *Artifacts: `results/v2_local_competition/`.*
*V2-P (proposal family; axis now closed).* Changing only the proposal family, with every other
invariant frozen, C1 passes on all three banks — RPN, DETR-R50 and a class-prompted Grounding DINO —
with ΔAUROC(K5−K50) +0.0517 / +0.0852 / **+0.2366**, E-AURC relative worsening +225.7 % / +455.5 % /
**+865.4 %** and RER@50 drops of 0.466 / 0.294 / **0.668**; C4 replicates on DETR (Δhard +0.0213, CIs
excluding 0), while GDINO is out of scope because its same-category supply probe scored 0.6892 < 0.85,
so **no GDINO hard-competition claim exists**. The published P1 overall verdict is
`CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST`; the P2 extension is recorded in the final table as
`C1_REPLICATED_ON_THIRD_PROPOSAL_FAMILY` (C1 only; C4 not tested). The
pre-frozen descriptive diagnostic V2-P2-M returned **MECHANISM_PARTIAL**: within-family same-class
redundancy co-moves only weakly with per-expression harm (ρ −0.014 / +0.089 / +0.053) and matching on
it leaves the cross-family gap **unshrunk** (shrinkage −0.180 / −0.065 against a 0.50 bar), because
GDINO carries the *fewest* same-class distractors (R1 6.23 vs 8.71 / 11.51) yet suffers the *most*
harm. **The mechanistic sentence that previously attributed the strongest amplification to same-class
semantic competition is therefore withdrawn** (ledger item P2-A0 in
`results/v2_proposal_robustness/p2_amendment_history.csv`); the candidate-count exclusion stands,
since all three banks hold 64 proposals evaluated at the same presented K. Index:
`results/v2_proposal_robustness/v2_p_program_summary.md`.

**Limitations.** The mainline backbone is a single frozen OpenCLIP ViT-B/32 (V2-G replicates on two
further encoders, still inside the COCO image universe); the proposal family now spans three banks
(RPN, DETR-R50, Grounding DINO) that share one frozen scorer stack and one image domain, and GDINO
participates in C1 only — its same-category supply gate failed at 0.6892 < 0.85, so no GDINO
hard-regime claim exists; RefCOCO+, RefCOCOg and RefCOCO share the COCO visual domain and V2-D2 is a
language-distribution transfer, not an image-disjoint one; the same-category regime is a GT-assisted
diagnostic construction; external amplification is unassessable (A11 Q2 = NOT ASSESSABLE); semantic
gains are modest outside hard competition; both attempted methods (LCR v1, adaptive mixture) are
negative results, so there is no end-to-end grounding-improvement claim; there is no cross-visual-domain
replication and no target-absence result in this mainline; and the mechanism behind the Grounding DINO
amplification remains **an open question** — the first candidate account has been refuted and no
replacement is offered.

---

## 3. 暂定研究方向（不进入本轮实现）

RQ1 candidate-set growth effect on reliability；RQ2 score information sufficiency；
RQ3 *when* candidate semantics add signal — primarily under strong local competition among
semantically similar candidates；RQ4（V2-P2-M 之后仍开放）：在候选数量与打分栈都被冻结/控制之后，
是什么可测量性质解释跨 proposal family 的放大差异——本轴只否证了第一个候选解释，没有提供替代解释，
任何后续机制诊断都必须自带结果前冻结的判据。
论文叙事定位为 **controlled reliability study**，而非
「提出新的 candidate-aware 架构」；V2-M 与 V2-M3 的两条负结果为这一定位提供了正面支撑。标题候选（暂不决定）：
*Reliable Visual Grounding under Dynamic Candidate Sets*；
*When Should a Visual Grounding Model Trust Its Choice?*；
*Candidate-Set Shift and Semantic Ambiguity in Reliable Visual Grounding*.
