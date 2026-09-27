STATUS: FROZEN BEFORE PHASE-0 RESULTS

# Research Protocol — Calibrated Candidate Grounding

本文件是 `calibrated-candidate-grounding` 的**预注册研究协议（pre-registered protocol）**。
它在 Phase 0 第一个正式结果产生之前完整写出并冻结，包含 RQ1–RQ4、数据集、backbone、
proposal bank、candidate 构造、target omission、数据划分、Phase 0 模型、指标体系、
统计检验、GO/NO-GO gates 与禁止的事后修改条款。

- 冻结日期：2026-09（Phase 0 任何正式结果产生之前）
- 冻结时的 git commit：由 `git rev-parse HEAD` 在本次冻结提交产生后填入，并按
  `docs/experiment_log.md` 的 schema 记录一条 `git_commit` 条目（协议冻结本身不入实验结果表）
- 本文件的任何后续变更必须走 §14 的 amendment 流程，且**原始标准必须原样保留**

研究定位（重要）：本项目**不**把 "candidate-based grounding"、region–text matching、
hard-negative grounding、NONE/rejection grounding 本身当作创新点。这些都是已有前序工作的
领域（见 `docs/literature_notes.md`）。本项目要验证的是一个尚未有共识的问题：在
candidate-set shift 之下 grounding 输出的**可靠性**是否系统性失效，以及这种失效是否
能被 scalar / score-level statistics 解释和修正。项目不预设正结果；任意阶段出现 NO-GO
都允许终止该路线。

---

## 1. 输入/输出形式与总体假设

基础输入形式为：

\[
(I,q,\mathcal C)\;\rightarrow\;P(c_i\mid I,q,\mathcal C)
\]

- \(I\)：图像；
- \(q\)：referring expression；
- \(\mathcal C=\{c_1,\dots,c_K\}\)：当前 candidate regions；
- candidate 由 bounding box + crop embedding + geometry + objectness 等表示；
- 输出是在**当前 candidate set 上**的选择；
- 当 target 不存在于当前 proposal/candidate set 时，系统应允许输出 NONE / abstain。

总假设：视觉与文本特征由**冻结的** backbone 离线产生并缓存，所有 decision module 只在
embedding 层训练（trainable decision module 原则上 ≤ 1M 参数）。backbone 比较不是研究问题。

---

## 2. RQ1–RQ4（精确表述）

### RQ1 — Reliability shift

> How do candidate-set cardinality and composition affect ranking accuracy and
> calibration differently?

特别研究 \(K=5,10\rightarrow20,50\) 时：

- ranking accuracy 是否下降；
- confidence 是否漂移；
- ECE / NLL / Brier 是否恶化；
- selective prediction 是否恶化。

必须区分 \(\Delta\)Accuracy 与 \(\Delta\)Calibration，二者分开报告、分开判定。

**明确概念约束**：不得把 "softmax denominator 因 K 增加导致 probability 数值改变" 本身
当作 calibration failure。真正研究的是

\[
P(\hat c=c^*\mid \hat p=p,\,K)
\]

是否随 \(K\) 或 candidate composition 改变（即 confidence–正确性映射本身是否漂移）。

### RQ2 — Sufficiency of simple uncertainty statistics

> Can simple score-level statistics explain and correct the reliability degradation?

重点比较：max score；max softmax probability；top1-top2 margin；entropy；score mean/std；
candidate count \(K\)；temperature scaling；\(T(K)\)；tiny stats-only calibrator。

**顺序约束**：在证明这些简单方法不够之前，不允许实现复杂 candidate-aware Transformer。

### RQ3 — Set information

> Can candidate-level set information predict grounding reliability better than scalar
> or score-distribution statistics?

只有当 RQ1 确认 failure 存在（Gate Q1 通过）**并且** RQ2 确认简单 calibration 不够
（Gate Q2 未判 NO-GO）时，才进入 candidate-aware 模型。

优先实现：DeepSets、permutation-invariant / equivariant MLP。
最后才考虑：small Set Transformer、1–2 层 self-attention。
**不默认 attention 最优。**

### RQ4 — Target omission

实际 proposal system 中可能发生 \(y\notin\mathcal C\)。重点区分：

- **Scene-negative**：目标本身不存在于图像中；
- **Candidate omission**：目标存在于图像 \(y\in I\)，但 proposal/candidate system 没提供
  正确 candidate：\(y\notin\mathcal C\)。后者是本项目更关心的问题。

比较：1) max-confidence threshold；2) margin threshold；3) flat \((K+1)\)-way NONE；
4) stats-only target-present model；5) candidate-aware target-present model；
6) factorized formulation：

\[
p_{\rm present}=P(y\in\mathcal C\mid I,q,\mathcal C),\qquad
P(c_i\mid y\in\mathcal C,I,q,\mathcal C)
\]

最终

\[
P(NONE)=1-p_{\rm present},\qquad P(c_i)=p_{\rm present}\,P(c_i\mid \text{present}).
\]

---

## 3. Primary dataset 与外部 stress test

| 角色 | 数据集 | split | 说明 |
|---|---|---|---|
| Primary | **RefCOCO+** | **UNC image-level split** | 全部 Phase 0 / 0.5 / 1 / 2 主线结论来源 |
| External stress test | **FineCops-Ref** | 官方 test 组织 | 仅作外部压力测试，不参与任何 model/calibration 选择 |
| 后续扩展（不进入 Phase 0 主线） | gRefCOCO、Ref-L4 | — | 仅记录 |

选择 UNC image-level split 而非 Google-style object split 的理由：本项目大量缓存
image/proposal feature，必须尽量避免同一图像跨 split 造成的 contamination。

第一阶段**不**同时铺 RefCOCO / RefCOCO+ / RefCOCOg / gRefCOCO / FineCops / Ref-L4 的
多数据集全矩阵。FineCops-Ref 被选为外部 stress test，因为它适合测试 object hard negatives、
attribute confusion、relation confusion、compositional difficulty、negative / None selection
（这类 relation/负样本压力测试由它负责，而不是由人工构造负责）。

## 4. Primary backbone

第一阶段只固定一个主 backbone：**OpenCLIP ViT-B/32**（frozen，离线缓存 image/region/text
embeddings，512-d）。

理由：image/text joint embedding；规模适合 8GB GPU；region crop inference 简单；可一次性
离线提取 feature；后续大量模型实验都在 embedding 层运行。

第一阶段**不**比较 CLIP ViT-B/16、SigLIP、DINOv2、MobileCLIP 或其他更大 backbone。
若未来主结论稳定，再选一个 lightweight 或不同 family backbone 做 robustness replication
（属于 extension，不属于本协议的主张范围）。

---

## 5. Proposal / Candidate Bank 定义

视觉候选生成与 decision model 彻底分离。使用**一个冻结的 COCO-pretrained detector / RPN**
（不参与训练，选型见 `docs/dataset_protocol.md`），定义固定、query-independent 的
proposal bank：

\[
I\rightarrow\mathcal P_I=\{p_1,\dots,p_N\},\qquad N\approx 64 .
\]

每张图必须保存：1) 固定数量 proposal；2) box；3) objectness；4) candidate crop；
5) candidate CLIP embedding；6) proposal 与 GT object 的 IoU metadata。

**Target 定义（冻结）**：

\[
\max_i IoU(p_i,b^*)\ge 0.5
\;\Rightarrow\;
c^*=\arg\max_i IoU(p_i,b^*)
\]

即 target 为 IoU 最大的**唯一** proposal；同时**移除**其他所有满足

\[
IoU(p_j,b^*)\ge 0.5
\]

的 proposals，防止 candidate set 中存在多个等价正确答案（避免 "多 correct proposal" 造成
的 label ambiguity 与 accuracy 不可解释）。

**Proposal audit（在开始任何 grounding decision 实验之前必须完成）**：必须统计并保存
proposal recall@64、proposal recall@32、natural miss rate、IoU distribution。

## 6. Candidate Set Construction

candidate set construction 是本项目最关键的部分之一。核心要求：\(K\) 与 hardness 必须是
两个尽量独立的变量。

**嵌套性与 target 恒定性（冻结）**：对同一个 \((I,q,c^*)\) 构造

\[
C_5\subset C_{10}\subset C_{20}\subset C_{50},
\]

且 target candidate 在所有 \(K\) 下**完全相同**，因此 \(K\) 增加只能来自新增 distractors。

禁止：K=5 时 target 是 proposal A、K=50 时 target 变成 proposal B。否则无法区分
cardinality effect 与 proposal effect。

**三种负样本定义**：

1. **Random candidates**：\(C_K^{rand}=\{c^*\}+\text{random distractors}\)，从同一 image
   proposal pool 中随机采样；固定随机种子并保存 candidate indices。
2. **Same-category hard negatives**：若 proposal 与其他同类 COCO GT object 高 IoU，则优先加入。
3. **CLIP-hard negatives**：使用 frozen CLIP query–crop similarity \(s_i^{clip}\)，从错误
   proposals 中选择高 similarity 候选。

**Regime 声明（冻结）**：CLIP-hard candidate construction 是
**diagnostic / adversarial evaluation regime**，而**不是**真实 detector distribution。
基于它得到的任何数字都不得被解释为 "部署条件下的可靠性"。

第一阶段不人工生成 relation hard negatives（这类 stress test 交给 FineCops-Ref）。

**目标不存在的构造**见 §8。

## 7. Phase 0：Candidate-Set Failure Audit

Phase 0 是当前最重要阶段，**禁止实现 candidate-aware network**。只允许以下四个 baseline：

| ID | 模型 | 定义 | 备注 |
|---|---|---|---|
| **B0** | Random | candidate 中随机选择 | sanity floor |
| **B1** | Frozen CLIP cosine | \(s_i=\cos(z_q,z_i)\)，经 logits/temperature 得到 candidate probability | 无可训练参数 |
| **B2** | CLIP + global temperature scaling | 在独立 calibration split 上拟合 \(T\) | 单参数 |
| **B3** | Independent MLP scorer | 输入 \([z_q,z_i,z_q\odot z_i,\cos(z_q,z_i),g_i]\)，输出 \(s_i\) | **严禁读取其他 candidate** |

\(g_i\) 包含 normalized box \(x,y,w,h,area\)，以及可选 objectness。

**Phase 0 Test Grid**：训练 \(K\in\{5,10\}\)；测试 \(K\in\{5,10,20,50\}\)；
candidate regime \(\in\{\)random, hard\(\}\)。Phase 0 暂时只做 target-present
ranking/calibration。形成 \(4\times2\) test cells。

若 proposal count 不支持 K=50，应**先报告 candidate availability distribution**，再决定最大
\(K\)。**不得静默过滤困难样本。**

## 8. Target Omission 协议

必须同时设计并**分别评估**：

- **Synthetic omission**：从正常 candidate set \(C=\{c^*,c_2,\dots,c_K\}\) 中删除 \(c^*\)，
  得到 \(C^-=C\setminus\{c^*\}\)，标签 \(y=\) NONE。
- **Natural proposal miss**：若 \(\max_i IoU(p_i,b^*)<0.5\)，candidate bank 天然没有正确
  proposal；单独建立 `natural_omission` evaluation split。

**禁止**把 synthetic omission 和 natural proposal miss 混在一个数字里报告。
如果模型仅在 synthetic omission 上有效、但在 natural omission 上失败，必须明确报告。

Target-presence shift 的评估比例（Phase 2）：训练时 target-absent 比例较低，测试时逐渐增加
\(10\%\rightarrow25\%\rightarrow50\%\rightarrow75\%\)，观察 threshold 与 calibration 是否失效。

## 9. 数据划分与用途边界（严格）

RefCOCO+ official training split 用于训练。validation 必须进一步**按 image** 划分成
`val_select` 与 `val_calib`。

| Split | 允许用途 | 禁止用途 |
|---|---|---|
| `train` | 模型参数训练；训练期 candidate set 生成 | 用 test 数据做任何选择 |
| `val_select` | architecture、optimizer、early stopping、hyperparameter | 拟合 temperature / threshold |
| `val_calib` | **只能**用于 temperature、thresholds、calibration model、abstention threshold | 用于选架构、选超参、early stopping |
| `testA` / `testB` | 完全冻结，仅最终评估 | 调温度、选 threshold、选 architecture、选 candidate protocol |

**禁止使用 test 数据**：调温度；选 threshold；选 architecture；选 candidate protocol。
calibration split 与 model-selection split 必须是不同的 image-level 集合（§9 的
`val_calib` vs `val_select`），以避免 calibration split 被隐式用于 model selection。

Phase 0.5 的额外约束：测试 \(K=20/50\) 时**不能**直接针对 test 拟合 \(T(K)\)；必须使用
可外推（例如 \(T=a+b\log K\)）或 validation-conditioned 的方法。

---

## 10. 指标体系

### 10.1 符号定义

\[
\hat c=\arg\max_i P(c_i),\qquad p=\max_i P(c_i),\qquad r=\mathbf 1[\hat c=c^*].
\]

主要 reliability 问题是 \(P(r=1\mid p)\approx p\) 是否成立。

### 10.2 Calibration（主指标）

不要只报告 multiclass ECE。

- **Top-label ECE**（主）：使用 **adaptive / equal-mass bins**。
- **Binary correctness Brier**：\((p-r)^2\)。
- **Top-label correctness NLL**：\(-r\log p-(1-r)\log(1-p)\)。
- **Confidence–accuracy gap**。
- **Reliability diagrams**：分别画 K=5 / K=10 / K=20 / K=50。
- Secondary：普通 multiclass NLL、multiclass Brier 可报告，但仅作 secondary metrics。

### 10.3 Ranking

Top-1 accuracy；Top-5 accuracy（若 K 允许）；accuracy by K；accuracy by candidate
hardness。MRR 可实现，但不是核心指标。

### 10.4 Selective prediction

给每个 prediction 一个 confidence，按 confidence 从高到低接受样本。报告：risk–coverage
curve；AURC；selective accuracy；**risk@50% coverage、risk@80%、risk@90%、risk@95%**。
所有风险指标必须明确定义 \(risk=1-accuracy\) 或对应定义。

### 10.5 Target absence / abstention

AUROC；AUPRC；FPR@95TPR；NONE precision；NONE recall；NONE F1；
**false selection rate**（重点指标：target absent 时模型仍选择某 candidate 的比例）。

本项目 OOD 只聚焦 candidate-set distribution（cardinality shift / hardness shift /
target-presence shift / external dataset shift）。**不做** ImageNet-C 等普通视觉
corruption（除非后续作为 extension）。

---

## 11. GO / NO-GO Gates（冻结，不可根据结果修改）

### Gate Q1 — candidate-set shift 是否形成稳定的可靠性失效

对比：从 ID `K = 5/10, random` 到 candidate-set OOD `K = 20/50 and/or hard`。

**至少出现以下一种**：

\[
|\Delta \text{Accuracy}|\ge 3\text{pp}
\qquad\text{或}\qquad
\Delta \text{ECE}\ge 3\text{pp}
\qquad\text{或}\qquad
\text{AURC 恶化 } \ge 20\% \text{ relative}.
\]

**同时要求（全部三条）**：

1. paired bootstrap 95% CI 不跨 0；
2. cosine 与 independent MLP **至少都观察到同方向现象**；
3. **至少两个 OOD cells** 满足。

否则：

\[
\boxed{\text{NO-GO}}
\]

结论：candidate-set shift 没有形成足够稳定的独立研究问题。停止 candidate-aware
architecture。

### Gate Q2 — 简单 calibration 是否已经把问题解决（Phase 0.5 之后）

仅在 Gate Q1 通过后进行。若 stats-only model 已经可以在**所有主要 OOD cells** 中达到

\[
\text{ECE} < 3\%
\]

并且相较 global temperature 的

\[
\Delta\text{AURC} < 5\%
\]

（即相对** global temperature 的剩余改善空间** 小于 5%），则

\[
\boxed{\text{NO-GO for candidate-aware model}}
\]

此时允许继续把项目整理成一个 "candidate-set calibration audit + simple calibration
solution" 的较小研究结果。**不得**为了做模型而继续增加 candidate-aware architecture。

Phase 0.5 允许的简单校准器（冻结）：
- **C1** Global temperature \(T\)；
- **C2** K-aware temperature，例如 \(T=a+b\log K\)，或为每个训练可见 K 单独拟合（测试
  \(K{=}20/50\) 不得直接针对 test 拟合）；
- **C3** Stats-only calibrator：输入 `K, max_score, top1_top2_margin, entropy, mean_score,
  std_score, max_softmax`（必要时加入 top3 score statistics），**禁止输入 candidate
  embeddings**；输出 correctness probability / temperature / target-present probability
  之一，第一阶段优先直接预测 \(P(\hat c=c^*)\)。

### Gate Q3 — candidate-aware 表示是否提供超出 stats 的信息

candidate-aware 模型必须**超过强 stats baseline**，而不是只超过 max-confidence。

**至少在两个 OOD regimes 中同时达到**：

\[
\Delta \text{AUPRC}_{\rm presence}\ge 2\text{pp}
\qquad\text{以及}\qquad
\text{AURC relative 改善}\ge 5\%,
\]

并且 paired bootstrap 95% CI 不跨 0。

如果研究 reranking，则至少满足

\[
\Delta\text{Accuracy}\ge 1.5\text{pp}
\]

或 "ranking 基本不变但 calibration / selective prediction 显著改善"。

否则：

\[
\boxed{\text{NO-GO for candidate-aware representation}}
\]

不得继续实现更大的 Transformer。

### Phase 1 架构冻结（仅当 Q1 通过且 Q2 未判 NO-GO）

第一版实现 **DeepSets**（不是 Transformer）。每个 candidate
\(h_i=\phi(z_q,z_i,z_q\odot z_i,g_i,s_i)\)，投影 \(512\rightarrow128\)；
set aggregation \(u_{mean}=\frac1K\sum_i\phi(h_i)\)、\(u_{max}=\max_i\phi(h_i)\)、
\(u=[u_{mean},u_{max}]\)；candidate 输出 \(s_i'=\psi(h_i,u)\)；或 reliability head
\(p_{\rm correct}=g(u,\text{score stats})\)。第一版参数量控制：**< 0.5M**。

**默认路线优先是 Independent ranker + Set-aware reliability model**（保留
\(s_i=f(q,c_i)\)，另学 \(p_{\text{correct}}=g(q,\mathcal C,\{s_i\})\) 或
\(p_{\text{present}}=g(q,\mathcal C,\{s_i\})\)），而不是直接修改 ranking。只有当分析显示
candidate-aware information 能改善 ranking，才再实现 set-aware reranker。

### Phase 2 候选模型集合（仅当前面路线成立后加入）

N0 max-confidence threshold（\(\max_iP(c_i)<\tau\Rightarrow\) NONE）；N1 margin threshold；
N2 flat NONE classifier \(\{c_1,\dots,c_K,\text{NONE}\}\)；N3 stats-only presence model
（输入 K、max、margin、entropy、mean/std、top-k scores，输出 \(P(y\in\mathcal C)\)）；
N4 set-aware presence model（加入 candidate embeddings / pooled set representation）；
N5 factorized model：\(P(\text{NONE})=1-p_{\rm present}\)，
\(P(c_i)=p_{\rm present}\frac{e^{s_i/T}}{\sum_j e^{s_j/T}}\)。

---

## 12. Statistical Testing 协议

- 所有 learned model：**3 seeds**；
- 对于同一个 test candidate set，所有模型使用**完全相同样本**；
- 主要 comparison 使用 **paired bootstrap**；
- **image-level resampling 优先于 expression-level resampling**（避免同图表达高度相关）；
- **95% CI**；**至少 5,000 bootstrap replicates**；
- 随机 candidate construction 必须**预生成并冻结**，不得每次 evaluation 重新 sample。

## 13. Feature Cache 与资源约束（协议级约束）

总体结构：\(\boxed{\text{extract visual/text features once}+\text{run many cheap decision experiments}}\)。
视觉 feature 不得按 referring expression 重复存储。cache schema（image 级 global feature /
proposal boxes / proposal features `[N,512]` / objectness / gt_iou / gt_assignment；
query 级 ref_id/image_id/text/query_feature/gt_box/gt_object_id/split；candidate sets 单独存
indices：ref_id/regime/K/hardness/target_present/candidate_indices/target_candidate_index）
见 `docs/dataset_protocol.md`。

**修改 candidate sampling protocol 时不允许重新提取 CLIP feature。**

设备：RTX 4060 Laptop GPU（8GB VRAM）。feature extraction 使用 FP16/autocast，batch size 从
64 开始 benchmark，OOM 时降到 32；不为 batch size 做复杂 gradient tricks（无训练）。
目标 proposal 数 \(N=64\)；约 20k images 时约 1.28M candidate crops；512-d FP16 embeddings
原始 feature storage 约 1.3GB；query embeddings 约百 MB 量级；整个 cache 目标 < 3GB。
优先使用 `.npy`/`.npz`、HDF5、LMDB、safetensors 中选择一种简单、支持 random access 的实现；
避免几百万个小文件。

## 14. Prohibited Post-Hoc Changes（冻结条款）

1. Phase 0 首个正式结果生成后，**不允许**修改 §11 中任何 gate 的阈值、判据或比较方向。
2. 不允许在看过 test 结果后修改：§9 的 split 用途边界、§5/§6 的 proposal 与 candidate
   定义、§10 的指标定义、§12 的统计检验设置。
3. 如果确实必须修改 protocol，必须：
   1. **新增 amendment**（append 到本文件 §16 的 amendment 章节，不覆盖原文）；
   2. **记录日期**；
   3. **解释原因**；
   4. **保留原始标准**（原始判据必须继续可见、可追溯）。
4. 不允许把 diagnostic regime（CLIP-hard）的结果表述为真实部署分布结果。
5. 不允许把 synthetic omission 与 natural proposal miss 合并报告以改善观感。
6. 不允许通过删除 "对结果不利的 cell"（例如 candidate 不足的 K=50 样本）来规避判据；
   必须先报告 candidate availability distribution 再决定最大 \(K\)。

## 15. 第一阶段禁止事项清单（原始提示词第二十九节，逐条冻结）

明确禁止（除非 Phase 0 之后有清晰证据支持）：

1. 大型 VLM；
2. QLoRA；
3. end-to-end CLIP fine-tuning；
4. RL；
5. Grounding DINO fine-tuning；
6. 复杂 multimodal transformer；
7. >5M decision module；
8. 多 backbone benchmark；
9. 多 dataset 全矩阵；
10. 手工标注；
11. relation negative 人工生成；
12. fancy loss；
13. focal loss / pairwise/listwise loss 大规模搜索；
14. architecture search。

## 16. Amendments

（本章节为追加区。当前无 amendment。）

| # | 日期 | 修改条款 | 原因 | 原始标准保留位置 |
|---|---|---|---|---|
| — | — | — | — | — |

---

## 附录 A：Repo 结构与协议对应关系

```text
docs/research_protocol.md      本文件（冻结协议）
docs/dataset_protocol.md       数据/proposal 设计审计（Task 3）
docs/phase0_plan.md            Phase 0 执行 checklist（Task 7）
docs/literature_notes.md       前序工作笔记与创新边界
docs/experiment_log.md         实验日志规范与 schema（第二十八节）
src/ccg/data/                  refcoco.py, proposals.py, candidate_sets.py, splits.py
src/ccg/features/              clip_encoder.py, extract_image.py, extract_regions.py, extract_text.py
src/ccg/models/                cosine.py, independent.py, stats_calibrator.py, deepsets.py, presence.py
src/ccg/calibration/           temperature.py, thresholds.py
src/ccg/metrics/               ranking.py, calibration.py, selective.py, bootstrap.py
scripts/                       prepare_refcoco.py, extract_proposals.py, extract_features.py,
                               build_candidate_sets.py, audit_proposals.py, run_phase0.py
```

Python package 名 `ccg` = calibrated candidate grounding。

## 附录 B：本协议有意排除的主张

- 不声称 candidate-based grounding 是本项目提出的想法；
- 不声称任何数据集、backbone、architecture 上的领先性能；
- 不在结果存在之前对结论做任何方向性预测；
- 本项目成功标准不是 positive result，而是：Phase 0 能够以可复现、统计严谨、低成本的
  方式判断 candidate-set shift 是否真的形成独立的 grounding reliability 问题。
