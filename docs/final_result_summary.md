# Final Result Summary — Candidate-Set Shift and Semantic Ambiguity in Reliable Visual Grounding

> Material bank for the paper, **not** the final abstract. All numbers here are the
> frozen values in `results/final_registry/` (derived from phase artifacts by
> `scripts/build_final_registry.py`, consistency-audited against `docs/experiment_log.md`).
> The candidate-cardinality / semantic-reliability mainline is **frozen** after A11:
> no new benchmark, backbone, architecture, reranking, or feature redesign.

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

### 局限（Limitations）
单一冻结 backbone；单一 proposal 家族；RefCOCO+/RefCOCOg 共享 COCO 视觉域；same-category 为 GT 辅助诊断构造；
外部 hard 放大不可评估；random 分布外语义增益幅度温和；无端到端定位性能提升主张；无跨视觉域复制；主线内无
target-absence 结果。

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

**Limitations.** Single frozen backbone and single proposal family; RefCOCO+ and RefCOCOg share the COCO
visual domain; the same-category regime is a GT-assisted diagnostic construction; external amplification is
unassessable; semantic gains are modest outside hard competition; and there is no end-to-end grounding-improvement
claim and no target-absence result in this mainline.

---

## 3. 暂定研究方向（不进入本轮实现）

RQ1 candidate-set growth effect on reliability；RQ2 score information sufficiency；
RQ3 *when* candidate semantics add signal — primarily under strong local competition among
semantically similar candidates. 论文叙事定位为 **controlled reliability study**，而非
「提出新的 candidate-aware 架构」。标题候选（暂不决定）：
*Reliable Visual Grounding under Dynamic Candidate Sets*；
*When Should a Visual Grounding Model Trust Its Choice?*；
*Candidate-Set Shift and Semantic Ambiguity in Reliable Visual Grounding*.
