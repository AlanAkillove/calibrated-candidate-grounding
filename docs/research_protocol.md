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

（本章节为追加区。登记索引见下表；amendment 全文append 于文件末尾。）

| # | 日期 | 修改条款 | 原因 | 原始标准保留位置 |
|---|---|---|---|---|
| A1 | 2026-09-27 | §6 regime taxonomy / §11 Gate Q1 证据层级（新增约束） | GT 信息不对称 + CLIP-hard↔B1 构造器耦合 | §6/§11 原文完整保留于上方；全文见文件末 “Amendment A1” |
| A2 | 2026-09-27 | §5 proposal bank 语义 / N-selection 预注册（新增工标准） | proposal bank 定义模糊（detector vs RPN）+ N 选择需工程预注册 | §5 原文完整保留于上方；全文见文件末 “Amendment A2” |
| A3 | 2026-09-27 | §A2.4 N-selection 执行 + §A2.5 待决项关闭（**结果记录**：不修改任何条款与 gate 数字） | proposal-system audit 完成（1500 图），登记 outcome：N=64 选定、K=50 排除报告义务、RQ4 功效风险、train2014 澄清、冗余极低不需新阈值 | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A3” |
| A4 | 2026-09-27 | Phase 0A（B1 cosine）audit outcome 登记 + 解读约束（新增披露义务；**结果记录**：不修改任何条款与 gate 数字） | Phase 0A 全量审计完成：硬 sanity check 全过、嵌套集理论性质未违反（无 VALIDATION_FAILURE）；T* 与 oracle per-K 温度均在拟合搜索下界凝结（边界简并）；calibration 证据须与 ranking 并列报告 | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A4” |
| A5 | 2026-09-28 | 度量有效性修正 + temperature 优化重做 + reliability GO replication criterion（**post-hoc amendment**：在 Phase 0A 结果可见后加入，*不构成 preregistration*；不修改任何原始 gate 数字） | Phase 0A 暴露三个统计问题：nested candidate sets 下 raw accuracy degradation 是结构性预期；raw AURC 与 base error rate 强耦合；temperature 最优解落在优化边界（T*=0.05 贴界）。后续 reliability GO 改为依赖 accuracy-normalized / base-rate-aware 指标（E-AURC / AUROC_correct / RER@c / corrected global-T reliability map） | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A5” |
| A6 | 2026-09-28 | Phase 0.5 Score-Information Sufficiency Audit protocol 冻结（split seed / 模型 zoo / 选型指标 / sufficiency gate；**结果可见前冻结**；不修改任何既有条款） | 进入 score-only 可靠性信息充分性审计：需在结果前固定 reliability_train/tune 切分（image-level, seed=20260928, 70/30）、训练 K 约束（K∈{5,10}）、L0/L1/L2 模型 zoo 与 §24/25/26 sufficiency gate 判定语义 | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A6” |
| A7 | 2026-09-28 | Phase 1 Candidate Semantic Information Sufficiency Audit protocol 冻结（E1/E2/E3 模型 zoo / 统计与 gate §28-31 / P1-P4 比较；**GO/NO-GO 数字在任何 Phase 1 结果可见前冻结**；不修改任何既有条款） | Phase 0.5 已证 score-only 信息不足（GO_candidate_embeddings，4 OOD cells 双 Route）；检验唯一未使用信息源 candidate/query semantic representation 是否携带额外可靠性信息 | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A7” |
| A8 | 2026-09-28 | Phase 1F Hard-Competition Semantic Confirmation（confirmatory stress test；cohort 规则 / 冻结评分 / A8.4 复现校验 / A8.6 gate 数字在任何 hard-regime 结果前冻结；不修改任何既有条款） | A7 verdict = INCONCLUSIVE（random regime 下 semantic 增量不显著）；唯一待检验假设：semantic 信息在 GT same-category 竞争下是否实质性更有用 | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A8” |
| A9 | 2026-09-29 | FineCops-Ref External Semantic Confirmation（staged external validation；工程 gate §6/§7 + regime 规则 §10/§11 + external gate §17-20 数字在任何 FineCops 模型推断前冻结；不修改任何既有条款） | A8 verdict = CONFIRMED；架构升级路径关闭，只允许检验该效应能否迁移到独立数据集（不同 image source / annotation pipeline / 语言构造） | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A9” |
| A10 | 2026-09-29 | RefCOCOg Image-Disjoint External Confirmation **Feasibility**（staged protocol；size / 工程 / same-category / 功效 / branch 五组数字在任何 RefCOCOg proposal 结果产生前冻结；不修改任何既有条款，**不改写 A9 的 EXTERNAL STOP 原因**） | A9 = EXTERNAL STOP（GQA 感知域偏移）；需一个保持 COCO image domain / COCO ontology / 冻结 proposal pipeline 可比性的外部候选，同时仍提供语言-标注分布 shift | 原始条款与 gate 判据全文保留于上方；全文见文件末 “Amendment A10” |

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

---

## Amendment A1 — 2026-09-27 — Candidate regime taxonomy & evidence hierarchy

> 本 amendment 为**追加条款**，不修改、不删除上方任何原始条款。原始 §5/§6/§11 全部文本
> 仍在上文原样保留、继续可见、可追溯。本条款只**新增约束层级**。

### A1.1 修改原因

首轮协议审计发现两处不对称：

1. **GT 信息不对称**：`same-category hard negatives` 与 `CLIP-hard negatives` 一样，在构造时
   都使用了 evaluation 时不可得的信息（前者使用 COCO GT 类别 + IoU，后者使用 frozen CLIP
   相似度），但原始协议只把 **CLIP-hard** 明确标为 diagnostic regime（§6 Regime 声明），对
   same-category hard negatives 只在风险清单 R7 处附带提及，证据地位不对等。
2. **构造器耦合未标注**：CLIP-hard 用 frozen CLIP 相似度选负样本，而 **B1（frozen CLIP
   cosine baseline）** 恰好用同一 CLIP 打分——存在 constructor / evaluator coupling，原始 §6/§11
   未点明该耦合对 Gate Q1 解释力的影响。

因此需要为三类 candidate regime 建立明确的**证据层级（evidence hierarchy）**，并据此**收紧**
Gate Q1 的通过条件（只加约束，不改任何原始数字）。

### A1.2 三类 candidate regimes（正式定义）

> **Random** — 主 controlled candidate distribution。distractor 的选择不依赖 query 文本，也不依赖
> GT category（仅从同一 image 的 proposal bank 中随机采样，固定随机种子并保存 candidate
> indices）。这是唯一可作为 "部署可类比分布" 论证基础的 regime。
>
> **GT Same-Category Hard** — **GT-assisted diagnostic stress test**。candidate 的类别通过
> `proposal → 最高 IoU COCO GT object` 的 assignment 获得：仅当某 proposal 与某 COCO GT object 的
> IoU ≥ 0.5 才赋予该 GT object 的类别，否则记为 `unknown`（无类别）。**明确声明**：本 regime
> (a) **使用 GT metadata**（COCO 类别 + GT box），真实 deployment 期的 detector 无法获得；
> (b) **不代表 deployment-realistic distribution**；(c) **不用于证明真实 detector 会自然产生同等的
> composition shift**；(d) 其主要用途是 **controlled same-category competition analysis**（在受控
> 条件下研究同类外观竞争候选对 ranking / calibration 的影响）。
>
> **CLIP-Hard** — **model-assisted adversarial diagnostic stress test**。**明确声明**：(a) 使用
> frozen CLIP 的 query–crop similarity 选择最易混淆的错误 proposals；(b) 与 **CLIP cosine
> baseline（B1）** 存在 **constructor / evaluator coupling**（选负样本与打分用同一模型）；因此
> (c) **B1 在该 regime 上的性能下降不能单独作为 candidate-set reliability failure 的主要证据**，
> 只能作为对抗性压力下的佐证。

### A1.3 Evidence hierarchy（照抄结构，作为 Gate Q1 的证据分层依据）

```text
Primary: candidate cardinality shift
Secondary: GT same-category composition shift
Adversarial diagnostic: CLIP-hard shift
```

### A1.4 Gate Q1 修订条款（只增约束，不改原始判据）

1. Gate Q1 **不允许仅凭 CLIP-hard cells 通过**（因其与 B1 存在 §A1.1(2) 所述耦合）。
2. 原始 §11 Gate Q1 第 3 条 "**至少两个 OOD cells** 满足" 中：满足条件的 OOD cells **至少一个必须
   属于 cardinality shift**（即 **random** regime 下的 `K=20` 或 `K=50` cells）。
3. 上述仅**收紧** Gate Q1 的通过条件；不放松、不替换任何原始比较方向与原始阈值。

### A1.5 原始 Gate 数字不变声明

原始 Gate Q1 / Q2 / Q3 的全部数字判据（`3pp` Accuracy、`3pp` ECE、`20%` AURC relative、
`ECE < 3%`、`ΔAURC < 5%`、`ΔAUPRC_presence ≥ 2pp`、`ΔAccuracy ≥ 1.5pp` 等）**保持不变**；
本次 amendment **仅新增 regime 的证据层级约束**。相关原始条款全部保留在上方 §11 正文中，
不删除、不改写。

---

## Amendment A2 — 2026-09-27 — Class-agnostic RPN proposal bank & N-selection rule

> 本 amendment 为**追加条款**，不修改、不删除上方任何原始条款（含 §5 target 定义、§13 N≈64
> 预算）。原始 §5 文本仍在上文原样保留、继续有效。本条款**澄清并冻结** proposal bank 的确切
> 语义，并**新增**一条工程性 N-selection 预注册规则。

### A2.1 主 proposal bank 冻结为 class-agnostic RPN proposals

主 proposal bank **正式冻结为 class-agnostic RPN proposals**（即 RPN 输出的候选框），**不是**
Faster R-CNN 的最终 detection boxes（后者已经过 ROI head 分类 + 置信度过滤 + 按类 NMS）。
pipeline 定义（照抄，为唯一权威口径）：

```text
image → frozen Faster R-CNN backbone/FPN → RPN proposals → RPN NMS → objectness ranking → top-N class-agnostic proposal bank
```

### A2.2 proposal 内容、类别 metadata 与禁止项

- 每个 proposal 只携带 `box`（几何）+ `objectness`（RPN 前景分）。**不含** detector 预测类别。
- COCO category **只能作为 offline diagnostic metadata**，且必须经
  `proposal → highest-IoU COCO GT object → GT category` 的 assignment 获得（IoU ≥ 0.5 才赋类别，
  否则 `unknown`）。此 assignment 仅用于 GT same-category hard-negative 构造（见 A1.2）与离线
  audit，**不进入** 主 candidate space 的定义。
- **禁止**使用 detector predicted class（ROI-head 分类结果 / 按类 NMS 后的标签）定义主 candidate
  space。**理由**：detector 的分类头置信度过滤与按类 NMS 会把 "检测置信度" 与 "目标存在性" 混入
  candidate 组成，从而**混淆 cardinality shift 与 detection confidence**，破坏 §6 所要求的 "K 与
  hardness 尽量独立" 以及 "cardinality shift 为 Primary 证据"（A1.3）的可解释性。

### A2.3 版本依赖与实现方式（引用名字，不写实现）

- 版本依赖记录：**torchvision 0.20.1 / torch 2.5.1**。
- RPN 提取**不依赖脆弱的 forward hook**，而是直接调用 `GeneralizedRCNN` 的公开子模块
  `transform` / `backbone` / `rpn` 的方式获得 class-agnostic RPN proposals。
- 实现细节以代码 `src/ccg/data/rpn.py` 为准（该文件由代码智能体并行实现，本 protocol 仅引用其
  名字，不在本文重复其实现）。

### A2.4 N-selection 预注册规则（工程标准，非研究 gate）

> 预注册 N-selection 规则（照抄语义）：
>
> ```text
> if N=64 allows K=50 construction for >=90% of otherwise eligible target-present examples:
>     use N=64;
> else:
>     use N=128.
> ```
>
> 若 `N=128` 时仍 `<90%`：**不自行删除 K=50**，必须先汇报 availability curve，再决定是否发起新的
> amendment。

**明确声明**：该 `90%` 是一条 **proposal engineering criterion（工程可用性判据）**，**不是**研究结果
gate，**不改变** Gate Q1 的任何判据或 §A1.4 的证据层级。它只决定 proposal bank 的 N 取值。

### A2.5 Target 定义保持原文 + duplicate 概念澄清

- §5 的 target 定义**保持原文不变**：`max IoU ≥ 0.5` 的**唯一** max-IoU proposal 为 target，并同时
  移除其他所有 `IoU(p_j,b*) ≥ 0.5` 的等价 proposal。
- **追加澄清**：以下两个是**不同概念**，不得混用——
  * **GT-equivalent proposal**：与 **GT box** `IoU ≥ 0.5` 的 proposal（§5 移除规则针对的就是这一类，
    以消除多正确答案歧义）。
  * **proposal-near-duplicate**：与 **target proposal** `IoU > threshold` 的 proposal（proposal 之间的
    近重复，与 GT box 无关）。
- 是否针对 proposal-near-duplicate 引入**更严格的 duplicate suppression**：**待** proposal audit 的
  "remaining candidate count distribution"（见 `docs/dataset_protocol.md` audit design 小节）结果出来后
  再决定；**在审计结果出来之前不得引入任何新的 IoU threshold**。

### A2.6 同步修正原 dataset 描述中的模糊表述

原 §5 "使用**一个冻结的 COCO-pretrained detector / RPN**" 的措辞存在 "detector vs RPN" 模糊。现**正式
选择**为：**torchvision `fasterrcnn_resnet50_fpn`（`FasterRCNN_ResNet50_FPN_Weights.COCO_V1` 权重）
之 RPN 子模块**（class-agnostic proposals，见 §A2.1）。该选择与 `docs/dataset_protocol.md` §4 的
proposal generator 选型一致；`detector / RPN` 原文不删除，由本条款给出唯一确定解读。

---

### Amendment A3 — 2026-09-27 — Proposal audit outcome & N-selection decision (results record)

> 本条款为**结果记录（results record）**：登记 2026-09-27 完成的 proposal-system audit
> （audit subset 1500 图）的 outcome，并按 §A2.4 预注册规则**执行** N-selection 决定。
> **本条款为追加记录：不修改、不删除上方任何原始条款，不修改任何 Gate 数字**——Gate Q1/Q2/Q3
> 的全部数字判据与 §A1.4 的证据层级保持不变；§5/§6 的 target 定义、嵌套性与
> K∈{5,10,20,50} 网格保持不变。逐项完整数字（recall@0.5/@0.7、CI、IoU 分布、冗余度等）见
> `docs/experiment_log.md` 条目 `audit-proposal-001`，本条款只登记协议层结论。

### A3.1 N-selection 决定（按 A2.4 预注册规则执行）

- 实测（audit 仅使用 `train` 池 1000 + `val_select` 500 的 image 级子集，不涉及 testA/testB 与
  `val_calib`；3752 expressions；N=64 与 N=128 两档对比）：
  - **N=64**：能构造 K=50（valid distractors ≥ K−1）比例 = **3719/3752 = 0.9912046908315565**
    [0.9876741623023597, 0.9937303782938145]；
  - **N=128**：K=5/10/20/50 全部为 1.0。
- 按 §A2.4 规则（N=64 能支撑 ≥90% otherwise eligible target-present examples 的 K=50 构造则用
  N=64）：**0.9912046908315565 ≥ 0.90 → 正式选定 N=64 作为主 proposal bank 规模**。
- 该 90% 为工程判据（§A2.4 已声明：不是研究 gate）；本决定不改变任何 Gate 判据。

### A3.2 K=50 排除样本的报告义务

- 实测 **0.88%（33/3752）** 的 expression 在 N=64 下无法支持 K=50 构造：这些样本**必须在
  K=50 cells 中显式报告排除计数（禁止静默过滤）**——重申 §7、§14.6 与 R5 的既有纪律，非新增判据。
- 主网格最大 K 维持 **50**（不因候选不足而降为 20）。

### A3.3 Natural omission 稀缺 → RQ4 统计功效风险（不影响 Gate）

- 实测 expression 级 natural omission（P(max IoU<0.5)）：N=64 = **53/3752 = 0.014125799573560768**；
  N=128 = 22/3752 = 0.005863539445628998。
- 记录：**RQ4 的 natural omission split 统计功效有限**，需在 Phase 2 前做功效评估。
- **明确**：该项为统计功效风险记录，**不触发、不改变任何 Gate**，Phase 0 主路线不受影响。

### A3.4 分割澄清：全部图像来自 COCO train2014（不改变 Gate 与评估协议）

- 实测：全部 **19,992** 张 RefCOCO+ 图像位于 **COCO train2014**（1500 张 val-split 抽样图在
  `instances_val2014` 中 0 命中）。本项目只需 COCO train2014 图像；val2014 图像非必需
  （`instances_val2014.json` 仅用于交叉核验）。
- **明确**：该澄清**不改变** Gate、评估协议、§9 split 用途边界或任何数据划分；同步修正见
  `docs/dataset_protocol.md` §1.3/§1.4。

### A3.5 A2.5 待决项处置：不引入新的 duplicate-suppression 阈值

- 实测冗余极低：两两 IoU>0.7 比例 N=64 = 0.001429563492063492（N=128 = 0.0006508366141732283）；
  remaining candidate count（移除 target-equivalent 后）median：N=64 = 59，N=128 = 120。
- 处置：**不引入**新的 duplicate-suppression IoU threshold——不存在“用近重复 box 凑 K=50”的问题；
  §A2.5 中“待审计结果出来后决定”的待决项就此关闭（审计结果已出，结论为不需要引入）。

---

## Amendment A4 — 2026-09-27 — Phase 0A fixed-CLIP cosine audit outcome & calibration interpretation constraints (results record)

> 本条款为**结果记录（results record）**：登记 Phase 0A（B1 frozen CLIP cosine，random regime，全量
> 19992 图）的执行与合规 outcome，并新增三条**解读约束**（§A4.3）。**本条款为追加记录：不修改、不删除上方
> 任何原始条款，不修改任何 Gate 数字**——Gate Q1/Q2/Q3 的全部数字判据、§A1.4 的证据层级、§5/§6 的
> target / candidate 定义与 K∈{5,10,20,50} 网格保持不变。逐项完整数字见 `docs/experiment_log.md` §9 条目
> `p0-cosine-kcardinality-20260927-01`（B1 审计）与 `audit-naturalomission-full-001`（全数据 omission
> counting）；本条款只登记协议层结论与解读约束。

### A4.1 执行与合规摘要

- B1（frozen CLIP ViT-B-32 laion2b_s34b_b79k）全量 **19,992 图**、4 splits、random regime、K∈{5,10,20,50}；
  评估单元 = sentence；跨 K 共享 common cohort（n=20,799；pooled 全 sentence 21,373）。
- 概率变体语义未变（§11）：native = softmax(100·s) 为主报告变体；T1 / global-T 为诊断变体。
- Calibration isolation（§9）：global T* 仅在 val_calib 拟合（pool K∈{5,10}）；testA/testB 从不参与拟合；
  oracle per-K 诊断按 §23 声明为 ORACLE/DIAGNOSTIC（不构成合法模型结果）。
- 三项硬 sanity check（协议 §17-19 的 STOP 契约）**全部通过**：score invariance 0 violations / 1,148,625
  candidate pairs（atol=0）；rank monotonicity 0 / 20,799 sentences（6 个 K 对）；accuracy monotonicity 通过
  （0.5349 > 0.3900 > 0.2857 > 0.1879）。**嵌套集理论性质未被违反 → 未写 VALIDATION_FAILURE.json，退出码 0。**
- **Phase 0A 不构成任何 Gate Q1/Q2 判定**：本审计为 B1-only、random-only 的单格证据；Gate 判定仍须按
  §11 + §A1.4 在完整 4×2 grid 与相应模型（B2/B3，可含 C*）完成后进行。

### A4.2 观测摘要（数字细节引用 §9 条目；本条款不重复全部数字）

- **Ranking**：common top-1 从 K5=0.5349 单调降至 K50=0.1879（Δ vs K5：−14.5 / −24.9 / −34.7pp）。
- **Native calibration**：ECE 0.2483 / 0.2931 / 0.3001 / 0.2772（K5/K10/K20/K50；相对 K5 变化
  +2.9~+5.2pp，置信度系统性高估）；T1 ECE 方向相反（0.3240 → 0.1662，置信度转为低估）。
- **Global-T（B2/C1 形式）**：T*=0.0500001 **收敛于拟合搜索界 [0.05, 100] 的下界边缘**（0.05+ε）；
  拟合前后 val_calib NLL 1.9153 → 1.4809；加权后 ECE ≈0.099~0.116，**ECE 的 K-依赖被压至 ≤1.7pp**
  （pooled paired diff：5→10 = −0.0132 [−0.0192, −0.0071]、5→20 = −0.0169 [−0.0242, −0.0087]、
  5→50 = −0.0048 [−0.0126, +0.0039] **跨 0**）。注意：绝对 ECE 仍 ≈10%+，不得引用为“校准良好”。
- **Oracle per-K 温度**：4 个 T_K 全部 =0.0500001（与 global 同一值），20/20 cells ΔECE≡0，
  verdict=no_meaningful_change——**该结果由边界凝结导致（见 §A4.3(2) 的解读约束）**。
- **Selective**：AURC 0.272 → 0.661（K5→K50），与 accuracy 降幅同数量级；risk@95 0.451 → 0.804；
  paired diff 均排除 0（−0.145 / −0.263 / −0.390）。
- **Reliability-map shift**（native，pooled）：K50 vs K5 在同置信 bin 的 empirical accuracy 差约
  −11.9~−21.6pp（顶 bin 0.9–1.0：0.7464 vs 0.5635）——与 ranking 降幅同源，见 §A4.3(3)。
- **全数据 natural omission**（`audit-naturalomission-full-001`）：expression 级 IoU 0.5 miss rate
  1.03%~2.68%（testB 最高）；A3.3 的 RQ4 功效风险结论不变。

### A4.3 解读约束（新增；不改任何判据；后续引用 Phase 0A 结果时必须遵守）

1. **T* 边界凝结的披露义务**：任何引用 T*=0.0500001 的场合必须同时注明 (a) 拟合搜索界为 [0.05, 100]，
   (b) 该值位于下界边缘。它是“在给定界内的 NLL 最优解”，**不是**“最佳锐化程度的无约束估计”；不得将该
   数值本身引作“CLIP logit scale 过锐”的定量结论。
2. **Oracle 诊断的边界简并**：per-K oracle 温度诊断在本搜索界内为**简并**（4 个 T_K 全部凝结于下界，
   ΔECE≡0）。**不得**解读为“per-K 温度无法修复校准漂移”；只能解读为“在既定搜索界内，per-K 相对 global
   无额外收益”。若要获得非简并的 per-K 温度诊断，需先发起新的 amendment 扩展搜索界后重拟合（test 数据
   永不参与拟合，§9 不变）。
3. **Calibration / selective 证据与 ranking 并列的报告义务**：native ECE 的跨 K 变化（≤5.2pp）与 accuracy
   降幅（最高 34.7pp）**必须并列报告**；在未控制 ranking accuracy 之前，AURC 的跨 K 恶化（绝对
   +14.5~+39.0pp）与 reliability-map 的同 bin accuracy 差**不得**先行表述为“独立于 ranking accuracy 的
   calibration / selective-risk shift”。K-依赖证据的强度评估必须同时考虑：单标量 global-T 已把 ECE 的
   K-依赖压至 ≤1.7pp（5→50 区间跨 0）。
4. **Gate 地位**：Phase 0A 的结论仅登记为证据，不产生任何 GO/NO-GO；Gate Q1 判定必须等待完整 4×2 grid
   与 B2/B3（可含 C*）按 §11 + §A1.4 完成。

### A4.4 伴随 counting 登记

- `audit-naturalomission-full-001`（全数据 natural omission counting，bank-v1 / top-64，counting only）：
  expression 级 miss rate 1.03%~2.68%，与 A3.3 的 audit 子集（1.41%）同数量级；登记为 RQ4 功效评估的
  底数，**不触发、不改变任何 Gate**。

---

## Amendment A5 — 2026-09-28 — Post-Phase-0A Metric Validity Correction

> **地位声明（必须先读）**：本 amendment 在 Phase 0A cosine results 已经可见之后加入，因此
> **不能称为原始 preregistration**，任何场合不得声称它是“结果出来之前已预注册的 gate”。
> 本条款同样为**追加记录**：不修改、不删除上方任何原始条款与 Gate 数字（Gate Q1/Q2/Q3 判据、
> §A1.4 证据层级、§A4.3 的披露义务均保留）。修改原因**不是追求正结果**，而是 Phase 0A 暴露了三个
> 统计/实现问题：(1) nested candidate sets 下 raw accuracy degradation 是结构性预期；
> (2) raw AURC 与 base error rate 强耦合；(3) temperature optimum 落在 optimization boundary。

### A5.1 度量有效性规则（新增强制报告义务）

- **Raw accuracy** 仍然报告，但 `raw ΔAccuracy` **不能单独**满足 reliability-shift 的 GO 判据。
- **Raw AURC** 仍然报告，但 `raw ΔAURC` **不能单独**成立 “confidence discrimination degradation”。
- 后续 reliability GO 必须依赖 **accuracy-normalized / base-rate-aware** 指标：
  - `E-AURC = AURC − AURC_oracle(r)`，其中 `r = 1 − Acc`，`AURC_oracle = r + (1−r)·ln(1−r)`（越低越好）；
  - `AUROC_correct = AUROC(1[pred=target], max_i P(c_i))`（cross-K primary discrimination metric；
    AUPRC_correct 仅 secondary，因 correct rate 随 K 大幅变化）；
  - `RER@c = (R1 − Rc)/R1`（c ∈ {0.50, 0.80, 0.90, 0.95}，越高越好）；
  - **corrected global-T reliability map**（fixed bins 供同 nominal confidence 跨 K 比较；equal-mass bins 供统计稳定性；
    任一比较桶 n < 100 必须标记 low-support，不得用于强结论）。
- 三类 reliability map 必须并列生成：native CLIP scale / corrected global T / oracle per-K T。

### A5.2 温度优化修正（实现层；替换 Phase 0A 的边界凝结实现）

- 以 `u = ln T` 参数化，初始范围 `T ∈ [1e-3, 10]`，直接最小化 candidate-set **mean NLL**
  （`scipy.optimize.minimize_scalar(method="bounded")`；**禁用粗 grid**）。
- 若最优点距任一边界不足一个数量级：自动扩大范围一次并记录；最终必须 **interior**，
  否则标记 `TEMPERATURE_OPTIMIZATION_WARNING` 并在报告中披露侧别。
- **合法拟合集不变**（§9 不松动）：global T 只用 `val_calib ∩ K∈{5,10}`（common cohort）拟合；
  objective = mean NLL（不是 ECE）；**testA/testB 与 K20/K50 永不参与拟合**。
- oracle per-K 温度：各 K 单独在 val_calib 对应 K 上重拟合，仅作 diagnostic，
  必须标注 `ORACLE / DIAGNOSTIC — NOT A VALID OOD METHOD`。

### A5.3 Phase 0A.1 修正结果登记（同 `p0a1-corrected-metrics-20260928-01` 条目）

- **corrected global T\* = 0.0335596**，`interior=True`（未触边界，距下界 1.53 decade / 上界 2.47 decade），
  未触发范围扩张与 warning；拟合集 n_sets=10,462（val_calib，K∈{5,10}）；NLL 1.9151 → 1.4386。
- **oracle T5/T10/T20/T50 = 0.034317 / 0.033007 / 0.032226 / 0.031264** → 修复边界伪影后 per-K 温度
  **不再完全相同**（≈10% 单调漂移）：上一轮 “all T=0.05” 确认属 **optimization-boundary artifact**；
  但漂移量级仍小（属逐 K sharpness 效应），不改变 §A5.4 的判据结构。
- 修正后 pooled（n=20,799）global-T 指标（K5/K10/K20/K50）：ECE 0.0224 / 0.0299 / 0.0399 / 0.0523；
  E-AURC 0.1404 / 0.1754 / 0.1849 / 0.1774；AUROC_correct 0.7413 / 0.7328 / 0.7329 / 0.7366；
  RER@50 0.384 / 0.260 / 0.188 / 0.122。
- 依据 §A5.4 对 cosine 的判定：**Route A 触发**（K5→K20 E-AURC relative worsening +31.7%、K5→K50 +26.3%，
  95% CI 均不跨 0；RER@50 下降 19.6pp / 26.1pp）；**Route B 未触发**（ΔAUROC ≈ −0.008 / −0.005，CI 跨 0）。
- 对 §A4.3 的更新：约束 (1)(2) 因温度拟合修正而失效（T* 不再凝结于边界，oracle 不再简并）；
  约束 (3) 继续有效——raw accuracy / raw AURC 仍不得单独作为独立 shift 的证据，但其证据门槛自此由 §A5.4 定义。

### A5.4 Reliability GO（replication）Criterion（post-hoc；自 B3 起生效）

仅用于判断 **是否值得继续 stats-only / candidate-aware reliability model**；要求 **cosine 与 B3 Independent MLP
都观察到同方向现象**，且至少满足 Route A 或 Route B 之一：

- **Route A — Selective discrimination**：从 K=5 到 K=20 或 K=50，`E-AURC` relative worsening ≥ 20% 且
  95% CI 不跨 0；同时 `RER@50` 或 `RER@80` 下降 ≥ 10 个百分点。
- **Route B — Correctness discrimination**：`AUROC_correct` 下降 ≥ 0.03 且 95% CI 不跨 0；
  同时 corrected global-T reliability map 存在稳定 shift。

判定语义：

- `cosine passes but B3 does not replicate` → **不能**声称 general candidate-set reliability failure；
  优先解释为 CLIP-cosine representation-specific phenomenon。
- `both fail` → **NO-GO for learned candidate-aware reliability model**。

### A5.5 Gate 地位（不变）

- 本 criterion **不替换** Gate Q1/Q2/Q3；它只决定 “是否值得继续 reliability model 路线”。
- Gate Q1 判定仍须按 §11 + §A1.4 在完整 4×2 grid 与相应模型完成后进行（§A4.1 最后一条不变）。

---

## Amendment A6 — 2026-09-28 — Phase 0.5 Score-Information Sufficiency Audit（结果可见前冻结）

> **地位**：本 amendment 在 Phase 0.5 任何结果产生**之前**写入，按指令冻结以下全部选择（split、模型 zoo、选型指标、gate 阈值与判定语义）。不修改任何既有条款与数字。训练/选择严格隔离：reliability model 只用 val_calib 派生的 reliability_train/tune 的 K∈{5,10}；testA/testB/K20/K50 永不参与训练、early stopping、选型。

### A6.1 固定 scorer 与数据源

- 主 scorer：**B3 Independent MLP**（3 seeds；复用 `results/phase0b_independent/seed_{1,2,3}/raw_scores/K{5,10,20,50}.npz`，20,799 common rows；**不重训**）。
- 次 scorer（replication / secondary）：**B1 frozen cosine**（复用 `results/phase0a_cosine/raw_predictions/K{5,10,20,50}.npz`，过滤到与 B3 完全相同的 common cohort）。
- MSP 温度使用各 scorer 已冻结的 corrected global T（B3 每 seed 1.1153 / 1.1166 / 1.1005；cosine 0.0335596；不重新拟合）。

### A6.2 Split 冻结

- val_calib 的行按 **image-level** 划分：seed = **20260928**、train_frac = **0.70**（`split_manifest.json` 先写后跑）。image-disjoint；确定性置换，冻结。
- 训练集 = reliability_train ∩ K∈{5,10}；选型集 = reliability_tune ∩ K∈{5,10}（mean AUROC over K5/K10）。
- 评估集 = testA、testB 及二者合并（`__pooled_test__`）；val_select 仅作 secondary 诊断。

### A6.3 Feature normalization 冻结

- 所有跨行使用的 μ/σ 只用 reliability_train ∩ K∈{5,10} 全量行估计（score 标准化、stats 标准化、logK/top1 标准化共用同一 fit）；对 K5/10/20/50 一律套用同一 μ/σ，禁止 per-K normalization。
- 指令 §9 的 z_top1=(s_(1)−μ_s)/(σ_s+ε) 属 per-set 统计（集合内部），其后如用于跨行特征再经 train μ/σ 标准化——两层命名在文档中区分。

### A6.4 模型 zoo 冻结（禁止 grid 扩张）

- **L0**：msp、top1_score、margin、neg_entropy、norm_entropy（无训练，直接作 selective confidence）。
- **L1**：stats（≤17 维：log_k / top1-3 / margins / moments / msp / entropy / logsumexp / z_top1 / z_margin / quantiles）× {LogisticRegression(L2, C∈{0.1,1,10}), TinyMLP(32→16→1, lr∈{1e-4,3e-4,1e-3}, wd=1e-4)} × {without-K, logK}；Logistic 附加 without-entropy 与 top-scores（§14：top5 z + gaps + logK + mean/std）变体。
- **L2**：**ScoreDeepSets**（φ:1→16→16；mean/max/h_top1 pooling + z_top1；head →32→1；±logK；lr∈{1e-4,3e-4,1e-3}）；参数 <5k。
- 选型：reliability_tune 的 mean AUROC(K5, K10)；仅此用途，绝不看 K20/K50。

### A6.5 Primary metrics 与统计

- Primary：AUROC_correct、E-AURC、RER@50、RER@80；Secondary：RER@90/95、ECE、Brier、NLL（ECE/Brier/NLL 仅对提供概率的模型）。
- image-level clustered paired bootstrap（5000 reps，cluster=image_id）：
  - 跨 K：K5 vs K20（secondary）、K5 vs K50（core）；models = {MSP, Margin, StatsLogistic, StatsMLP, ScoreDeepSets}；在 testA / testB / `__pooled_test__`。
  - 跨模型（同 K=5、50）：MSP vs StatsLogistic、StatsLogistic vs StatsMLP、BestSummary vs ScoreDeepSets（primary pairs, §28）。
- 每 scorer seed 独立运行；头部结论报 mean±std across seeds；**不合并 seeds 充样本量**。
- E-AURC relative 统一以 worsening 报告 w=(K_b−K5)/K5（正值=恶化），CI 端点同映射。

### A6.6 Sufficiency Gate（§24/25/26 冻结语义）

- **Best Score-Only Model** = 全部合法 L0/L1/L2 中，reliability_tune K5/K10 mean AUROC 最高（跨 3 seeds 取均值选出 family；per-seed 各自再选并报告一致性）。
- **NO-GO for candidate embeddings**（= score-only 信息已足够）：该模型在 K20/K50（`__pooled_test__`）同时满足：|ΔAUROC_correct| < 0.02（vs K5，absolute）；E-AURC rel worsening < 20% **或** absolute E-AURC < 0.03；RER@50 drop < 10pp。
- **GO for candidate embedding information**：仅当该模型在 ≥2 个 OOD cells（split∈{testA,testB} × K∈{20,50}）满足 Route A（E-AURC worsen ≥20% 且 RER@50 drop ≥10pp）或 Route B（ΔAUROC_correct ≥0.03 下降），且对应 CI 不跨 0（seed-mean 判定 + ≥2/3 seeds 一致）。
- §26 override：若 K50 上 AUROC≥0.85 且 E-AURC≤0.03 且 RER@50≥0.70 → “practically solved by score-only information”。

### A6.7 禁止事项（与指令 §32 一致）

candidate embedding / Set Transformer / query·crop embedding / geometry / objectness / GT metadata / 重训 grounding scorer / loss engineering / feature-subset search / 多检验挑选，一律禁止；本轮只回答 “How much reliability information is already present in the score set?”。

---

## Amendment A7 — 2026-09-28 — Phase 1 Candidate Semantic Information Sufficiency Audit（GO/NO-GO 数字在任何 Phase 1 结果可见前冻结）

**staged 声明**：本 amendment 制定于 Phase 0.5 score-sufficiency 结果已知之后（其 GO 结论是进入本阶段的动机）；属 staged research protocol，**不是**初始 preregistration；不修改任何既有条款。

### A7.1 固定 grounding 与数据

- 主 scorer：**B3 Independent MLP（seed 1-3）**；Phase 0B raw scores 直接复用，**不重训**；ĉ = argmax s_i 全程冻结；reliability model 不得改分数 / rerank / 改变 top-1 / 参与 grounding 训练。
- 切分与 Phase 0.5 **完全一致**（不重划）：复用 `results/phase05_score_sufficiency/split_manifest.json` 的 train/tune images（523/224；seed=20260928, 70/30）；训练/选型只用 K∈{5,10} 行；OOD 评估 K∈{20,50}（testA/testB/pooled）；testA/testB 不参与训练/早停/选型。
- 本轮不用 cosine（secondary 不参与）；3 个 B3 seed 各自独立跑完整流程，不混合。
- 行集/顺序：与 Phase 0.5 canonical 行集（common cohort, n=20799, sentence_id 升序）逐 K 对齐；image_id/eval_split 逐行断言相等。

### A7.2 输入（允许 / 禁止）

- 允许：z_q、z_i（已缓存 L2-normalized OpenCLIP，512-d）、B3 raw scores、candidate rank、top1 indicator、corrected softmax probability。
- 禁止：global image embedding、GT category/IoU/identity、oracle metadata、FineCops、**geometry/objectness**（B3 已用过；Phase 1 保持语义纯净）。

### A7.3 模型 zoo 冻结（E0/E1/E2/E3；禁止 grid 扩张）

- **E0（reference）**：MSP 与 Phase 0.5 Stats Logistic——直接复用 `results/phase05_score_sufficiency/predictions/` 中已保存的逐行预测（同数据同切分；运行时断言与本地重算一致）。
- **E1（handcrafted semantic stats，<30 维）**，全部确定性、无阈值搜索；设 r(i) 为 B3 score 降序 stable rank，t=winner，a_i=cos(z_q,z_i)，v_{ij}=cos(z_i,z_j)：
  `clip_top1=a_r1, clip_top2=a_r2, clip_margin12=a_r1−a_r2, clip_entropy=H(softmax(a/T_clip)), clip_normH=1−H/logK, clip_rank_top1=B3 winner 在 CLIP cosine 降序下的平均秩/K, cand_vmax=max_{j≠t} v_tj, cand_vmean, cand_vstd, cand_top12_sim=v_t,r2, cand_top15_mean=mean_{r=2..5} v_tj, density_070/#{j≠t:v_tj>0.7}/(K−1), density_080（τ=0.8）, q_top3=a_r3, q_margin13=a_r1−a_r3, cand_top13_sim=v_t,r3`（共 16 维；τ∈{0.7,0.8} 固定）。
  **预结果修订（2026-09-28，在生成任何 Phase 1 结果前）**：`T_clip = 0.01`（= 1/logit_scale；冻结的 OpenCLIP ViT-B/32 `openai` 权重的 logit_scale 实测精确为 100）。原稿写入的 T=1 在 CLIP cosine 尺度下使该 softmax 饱和（clip_entropy 退化为恒定 ≈ log K），属特征退化；`clip_entropy/clip_normH` 因此改为 `softmax(a/0.01)`。其余定义不变。
- **E1 模型**：LogisticRegression(L2, C∈{0.1,1,10})；**E1a**=semantic 16-d only；**E1b**=[A6 的 17-d stats（stats_logK 变体）⊕ semantic 16-d]。标准化 μ/σ 只用 reliability_train∩K∈{5,10}（同 A6.3）。
- **E2 TopCompetitor**（torch，BCE）：P_q: 512→64，P_v: 512→64（top1/top2 共享）；交互 [q,h1,h2,q⊙h1,q⊙h2,h1⊙h2,|h1−h2|] (448-d)；变体：**E2-score**（仅 4 维 score stats: top1/top2/margin/MSP）、**E2-semantic**（仅 448-d）、**E2-combined**（448-d ⊕ 4-d）；head 128→1；参数 <150k；lr∈{1e-4,3e-4,1e-3}，wd=1e-4，patience 30，≤300 epochs，val=reliability_tune。主模型=E2-combined。
- **E3 SemanticDeepSets**（torch，BCE）：h_i=[r_i(64), q⊙r_i(64), s_i* (train-μ/σ 标准化 raw score), p_i (corrected softmax), 1/rank_i, top1_ind_i] (132-d)；φ:132→128→GELU→128；u=[mean, max, h_t]；head→128→1；**默认 without logK**；参数 <250k；同 E2 训练配置。
  - **E3-full**（全 K）为主模型；**E3-top5**（B3 score 前 5，所有 K 固定）为预定义 ablation；`E3+logK` 为单项 ablation。不得用 K20/K50 选择。

### A7.4 选型

- 只用 reliability_tune 的 mean AUROC_correct(K5,K10)；|Δ|<0.002 时取更简单模型（更少参数）；每个变体的 lr/C 各自在这套规则下选定；OOD 不参与任何选择。

### A7.5 统计与主比较

- 指标同 A6.5（AUROC_correct/E-AURC/RER@50/RER@80 + secondary）；worsening/减幅约定与 Phase 0.5 一致（正値=K_b 更差 / 正値=改善，后者统一以 relative reduction = (score−sem)/score 报告并给 CI）。
- image-cluster paired bootstrap 5000 reps；跨 K（K5 vs K20/K50，testA/testB/pooled）与跨模型；跨模型固定对：**P1** StatsLogistic vs E1b；**P2** StatsLogistic vs E2-combined；**P3** StatsLogistic vs E3-full；**P4** E2-combined vs E3-full；另加 MSP vs {E1b, E2-combined, E3-full}（因 Phase 0.5 seed-mean winner 是 MSP，gate 需与其比较）；均在 pooled、K=5 与 K=50（若需要 gate 在 K20 上同样计算）。
- 每 scorer seed 独立；头部报 mean±std；**不合并 seeds**。

### A7.6 Candidate Semantic GO Gate（§28/29/30/31 冻结语义）

- 比较对象（结果前就指定、运行时仅按 tune 确定具体名字）：**best score-only**（tune 最高的 {MSP, StatsLogistic}）vs **best semantic**（tune 最高的 {E1a,E1b,E2-semantic,E2-combined,E3-full,E3-top5}）；两者均由 3-seed mean tune 选定一次。**预运行澄清（2026-09-28，在任何正式 Phase 1 结果前）**：E2-score 不进入 best-semantic 候选集——它不消费任何 embedding 信息（仅 score stats 的 MLP 消融），否则 §28 “candidate semantic model” 的比较会退化为两个 score 模型的比较（E2-semantic/E2-combined/E3 均消费 embedding，保留）。
- 单元定义：**cell(K), K∈{20,50}**，在 `__pooled_test__` 上：
  `cell_ok(K) = [ΔAUROC ≥ 0.02 且 ΔAUROC CI_lo > 0] ∧ ([E-AURC relative reduction ≥ 0.10 且 CI_lo > 0] ∨ [ΔRER@50 ≥ 5pp 且 CI_lo > 0])`
- **PASS（Semantic Signal PASS）** = cell_ok(20) ∧ cell_ok(50) ∧ (至少 2/3 seeds 在 K50 上 ΔAUROC CI_lo > 0) ∧ (3-seed mean 方向一致为正)。
- **STRONG（§29）** = K50 上 ΔAUROC ≥ 0.03 ∧（E-AURC reduction ≥ 0.20 或 ΔRER@50 ≥ 10pp）∧ 同样 CI_lo > 0；标注但不作为后续必要条件。
- **NO-GO（§30）** = 在 K20 与 K50 两个 cell 上同时：ΔAUROC < 0.01 ∧ |E-AURC relative reduction| < 0.05 ∧ |ΔRER@50| < 3pp。
- 优先级：**STRONG > PASS > NO-GO > INCONCLUSIVE**（其余情况及 0.01≤ΔAUROC<0.02 的 gray zone 一律 INCONCLUSIVE，先 error analysis，不自动加 Transformer）。

### A7.7 附加分析（§32-35）

- **OOD stability**：逐模型报 ΔAUROC(K5,K50)、E-AURC(K50)/E-AURC(K5)、RER50(K5)−RER50(K50)（pooled seed-mean）→ 区分“全面提升绝对质量” vs “减弱 K 依赖退化”。
- **Error subsets**：K50 pooled；A=B3 correct / B=incorrect；在 B 内按 cand_top12_sim  quartile 分组；同法按 clip_margin12 分组；报各组 n、score-only vs semantic 的 AUROC 与 Δ。
- **Matched-score diagnostic**：K50 pooled；treatment = cand_top12_sim 上四分位，control = 其余，匹配变量 = pooled 内 z-score 的 [MSP, margin12, entropy] 欧氏 1-NN 贪心无放回匹配（按 sentence_id 排序确定性执行；**不使用 correctness**）；报 matched Δcorrectness + image-cluster bootstrap CI。
- **E1 coefficients**：E1b 标准化系数（逐 seed + mean±std）；只作 predictive association diagnostic，不作 causal claim。

### A7.8 禁止事项（与指令 §43 一致）

changing grounding scores / reranking / candidate-aware grounding training / Set Transformer / cross-attention / CLIP finetuning / global image feature / hard-negative training / target omission / FineCops / RefCOCOg / backbone comparison / RL / VLM；只回答 “Does candidate semantic information improve reliability prediction beyond score information?”。

### A7.9 Artifacts 与存储约定

- `results/phase1_semantic_sufficiency/`：protocol.json、semantic_features/（E1 stats npz + provenance）、semantic_stats.csv（定义表 + summary moments）、e1_logistic/、e2_top_competitor/、e3_semantic_deepsets/、aggregate.csv、pairwise_vs_score_only.csv、bootstrap.csv、error_subsets.csv、matched_score_analysis.csv、figures/、predictions/（§39 列：ref_id, image_id, K, grounding_correct, B3_score, score_only_reliability, semantic_reliability, scorer_seed）。
- 原始 512-d embeddings（~GB 级，可从 cache/features 确定性重建）存 `cache/semantic_phase1/`（gitignored），不入库；results 中存其 cohort hash 与重建脚本引用。

### A7.10 测试要求（§41 的 15 项，全部必须 0 failed）

frozen ranking 不变 / reliability model 不可改分数 / 无 GT metadata / 无 geometry·objectness / 候选投影共享 / E3 置换不变 / 变 K 支持 / top1 身份在置换下保持 / E3-top5 恰为 B3 分数前 5 / K20·K50 不入训练·选型 / 切分与 Phase 0.5 完全一致 / scorer seed 隔离 / bootstrap 同 image 簇 / semantic stats 确定性 / matched-score 不使用 correctness。

---

## Amendment A8 — 2026-09-28 — Phase 1F Hard-Competition Semantic Confirmation（confirmatory stress test；gate 数字在任何 hard-regime 结果前冻结）

**staged 声明**：本 amendment 制定于 Phase 1 A7 verdict = INCONCLUSIVE 已知之后；属 staged confirmatory follow-up，**不是**初始 preregistration；不修改任何既有条款。只验证一个假设：semantic reliability information 是否在 GT same-category hard candidate composition 下实质性更有用。

### A8.1 固定与披露

- 冻结：B3 seeds 1-3（仅用于对新候选集推断性打分，不重训）；Stats Logistic 与 E1b 的系数与标准化完全继承 random regime（reliability_train∩K∈{5,10}）；semantic 16-d 定义冻结；不得用 hard-regime labels 训练/校准任何东西。
- 披露：GT same-category 为 **GT-assisted diagnostic stress test**（proposal→最高 IoU GT object，IoU≥0.5；unknown 不匹配），非 deployment-realistic；CLIP-hard 因 constructor/evaluator coupling 不做 primary（本轮不运行）。

### A8.2 Cohorts（规则冻结；执行前实测附注 2026-09-28）

- base = Phase 0.5 canonical cohort ∩ {testA, testB}（pooled test；n=10286 行 / 1490 images）。
- **SameCat-K5（primary）** = base ∩ {n_same_category_available ≥ 4}；实测 9487 行 / 1424 images / 3369 refs（hard_fraction[K5]=1.0）。
- **SameCat-K10（secondary）** = base ∩ {n_same ≥ 9}；实测 6765 行 / 1085 images → 满足 §7 门槛（≥1000 expressions ∧ ≥300 images），执行。
- **Exp B（hard-fraction curve）** = base ∩ {n_same ≥ 8}，K=10，levels n_samecat ∈ {0,2,4,8}（levels 不因结果调整）；实测 7410 行 / 1189 images → 执行；若 cohort 过小则取消。
- 配对：每行 random 与 same-category 用同一 ref/image/target/K；random 版本在同一 cohort 上重新评价（matched control）。runtime 断言两 manifest 的 target_index / valid pool / n_same 逐 ref 一致。

### A8.3 候选构造（冻结）

- ``C_K^regime = [target_index] + order_regime[:K-1]``，order 来自冻结 manifests（manifests-v1，seed=20260927）；不重采样。
- same_category ordering = 全部 same-cat（ascending bank index）+ rest shuffle；n_same≥K−1 时前缀全 same-cat（实测 K5/K10 全 1.0，driver 断言）。
- Exp B level m（K=10）：``C = [target] + same_order[:m] + same_order[n_avail : n_avail + (9-m)]``。

### A8.4 冻结评分与模型恢复（含 STOP 校验）

- B3 打分：``load_b3_model``（model.npz + training.json）冻结权重前向；输入组装与训练一致（z_q, z_i, z_q⊙z_i, cos, geometry 6-d centered）。
- **复现校验（STOP 条件）**：Random-K5/K10 重打分 vs phase0b raw_scores：max|Δ| ≤ 1e-4（预期 ~1e-6）；失败则中止本阶段。
- Stats Logistic（17-d stats_logK）×3 seeds 与 E1b（17-d stats ⊕ semantic 16-d）×3 seeds：在冻结 train 行上**重拟合（同超参）** 以恢复冻结模型，并用三重校验确认与 phase05/phase1 完全一致：(a) stats_logistic 预测 vs phase05 ``predictions/*/stats_logistic.csv.gz``（行级多重集 / 每 (K,split) AUROC 差异 ≤1e-9）；(b) E1b 系数 vs phase1 ``e1_logistic/coefficients.csv``（max|Δ| ≤1e-9）；(c) E1b tune mean AUROC vs phase1 metadata selections（≤1e-9）。
- normalization：stats17 与 semantic 均 train-only 重拟合（确定性）；重拟合的 stats17 与 phase05 ``features/*_normalisation.json``（stats_logK）对照 max|Δ| ≤1e-9。

### A8.5 指标与统计

- Primary：AUROC_correct / E-AURC / RER@50；Secondary：RER@80 / ECE / Brier / NLL。grounding accuracy 只作为 stress 真实性证据，不作为 semantic 成功度量。
- 全部对比：image-cluster paired bootstrap 5000 reps（seed=0, ci=0.95）；3 seeds 独立 + mean±std；**不拼接 seeds**。
- 关键量：``ΔAUROC^regime = AUROC(E1b) − AUROC(Stats)``（同 regime 同 cohort）；主判量 ``Δ^hard − Δ^rand``（同一 cluster 重采样内配对计算 diff-of-diffs + CI）。
- ``E-AURC reduction = (E_stats − E_e1b) / E_stats``；``RER@50 gain = 100 × (RER_e1b − RER_stats)`` pp。

### A8.6 A8 Gate（冻结）

- **CONFIRMED**：SameCat-K5 cohort：ΔAUROC^hard ≥ 0.02 ∧ CI_lo>0 ∧ (E-AURC reduction ≥ 10% ∨ RER@50 gain ≥ 5pp，所选分支 CI_lo>0) ∧ ≥2/3 seeds ΔAUROC CI_lo>0 ∧ 3-seed mean 方向为正 ∧ (Δ^hard − Δ^rand) ≥ 0.005。
- **STRONG**：ΔAUROC^hard ≥ 0.03 ∧ (reduction ≥ 20% ∨ RER@50 ≥ 10pp) ∧ CI_lo>0。
- **NO-CONFIRMATION**：ΔAUROC^hard < 0.02 ∧ reduction < 10% ∧ RER@50 < 5pp → STOP candidate-semantic architecture exploration（负结论有效）。
- 优先级：**STRONG > CONFIRMED > NO-CONFIRMATION > INCONCLUSIVE-HARD**（gray zone 不触发任何 escalation，不自动加 Transformer）。

### A8.7 操作检查与辅助分析（§19-22）

- Manipulation check（paired，K5）：cand_vmax / cand_top12_sim / clip_margin12 的 random→hard shift + bootstrap CI；不成立 → INVALID STRESS TEST（不得解读 reliability 结果）。
- Grounding difficulty：B3 accuracy / score margin / MSP / entropy 的 random→hard shift。
- Error-concentration：SameCat-K5 按 cand_vmax quartile 分组的 ΔAUROC / E-AURC improvement（secondary）。
- 系数稳定性：不重拟合；仅报 hard regime 的 feature 分布 shift。

### A8.8 禁止事项（与指令 §31 一致）

training new reliability models / retraining E1b / feature selection / E2-E3 tuning / candidate-aware grounding / reranking / Transformer / attention / FineCops / CLIP-hard primary / backbone changes / target omission。FineCops external confirmation 仅在 CONFIRMED 之后由下一指令决定。

### A8.9 Artifacts

``results/phase1f_hard_semantic/``：protocol.json、cohort_summary.json、manifests/（各 regime 的逐行 C_K bank 索引表 + provenance）、manipulation_check.csv、grounding_difficulty.csv、reliability_metrics.csv、semantic_increment.csv、paired_random_vs_hard.csv、bootstrap.csv、subgroup_analysis.csv、gate.json、figures/（§28 的 4 张）、predictions/（逐行 raw predictions，含 regime 列）。

### A8.10 测试要求（§29 的 12 项，全部必须 0 failed）

same-category candidates 与 target 同 GT category / no target-equivalent / matched random-hard cohorts identical / K fixed / frozen E1b coefficients unchanged / frozen Stats Logistic unchanged / no hard-regime labels enter training / same normalization reused / candidate ranking 不被 reliability model 改变 / paired bootstrap 同 image 簇 / deterministic manifest generation / manipulation features correctly computed。

---

## Amendment A9 — 2026-09-29 — FineCops-Ref External Semantic Confirmation（staged external validation；gate 数字在任何 FineCops 模型推断前冻结）

**staged 声明**：本 amendment 制定于 Phase 1F **A8 verdict = CONFIRMED 已知之后**，属预注册的
external confirmation 阶段；不修改任何既有条款，也不构成新的探索性主张。
唯一被检验的假设：**A8 在 RefCOCO+ 上发现的 semantic incremental value 是否在一个
完全独立的数据集（不同 image source、不同 annotation pipeline、不同语言构造）上仍然成立**。
本阶段**不是**训练阶段：FineCops 上不存在任何可学习参数（0 个），A8 的架构升级路径在此关闭。

**结果可见性**：F0/F1 的 metadata 事实（文件清单、counts、level/tuple_type 分布）与 F2 的
proposal 工程审计数字按指令 §1/§5/§31 属 **feasibility audit 本身**，是本 amendment 的记录对象；
**任何 reliability 模型的 FineCops 输出（B3 / Stats / E1b 分数、AUROC、E-AURC、RER）在本文写作时尚未被查看**，
A9.5 的全部判据数字在 F5–F10 推断开始前冻结，且 `results/phase1e_finecops/feasibility_protocol.json`
的 `frozen_before_run` 块在 F2 运行前已落盘（artifact 顺序可核验）。

### A9.1 数据与许可（实测核实，非引用官方数字）

- 来源：**FineCops-Ref 官方 repo** `liujunzhuo/FineCops-Ref` → **figshare article 26048050**；
  license **CC BY 4.0**（论文正文声明，非推断）；底层图像来自 **GQA / Visual Genome**。
- 本地只下载 **test split 标注**（4 个 json，`data/raw/finecops/`，逐文件 sha256 记录于
  `data/raw/finecops/dataset_card.json`）+ **GQA `sceneGraphs.zip`**（42.7 MB，解压后只有
  `train_sceneGraphs.json` / `val_sceneGraphs.json`）。
- **实测 counts**（positive test）：**9,605 expressions / 4,313 unique images**，level 1/2/3 =
  **5,730 / 3,404 / 471**；negative：**9,814 negative text + 8,507 negative image**
  （`test_expression_all.json` 合计 27,926 行）。官方数字与实测一致，但协议只承认实测值。
- 图像：**不下载 GQA 20.3 GB 整包**。`images.zip` 支持 HTTP `Range`，
  `ccg/external/gqa_images.py` 用 zip 尾部索引按成员抽取，只取审计子集所需的 JPEG；
  该 host 在并发下返回 **503**，故并发固定为 3 并带指数退避（不并行暴力拉取）。
- `neg_images.tgz`（567 MB）**不下**；negative 只做 parse/count（A9.7）。

### A9.2 冻结 pipeline（与 RefCOCO+ 完全同源，禁止任何为 FineCops 的适配）

image → **frozen torchvision `fasterrcnn_resnet50_fpn`（COCO_V1）RPN stage** → class-agnostic
**N = 64** proposals（top-64 by objectness；IoU≥0.5 等价 proposal 移除；target = argmax IoU）
→ **frozen OpenCLIP ViT-B/32 `laion2b_s34b_b79k`** crop/text embeddings（checkpoint hash、preprocess、
tokenizer、512-d、normalization 必须与 RefCOCO+ 侧一致，任一不符 → STOP）
→ **frozen B3**（seeds 1/2/3，各自独立评估）→ **frozen Stats Logistic / E1b**（
`ccg.semantic.frozen.recover_frozen_models` 的 A8.4 四重校验，`max|Δ| ≤ 1e-9`，任一 mismatch → STOP）。
不使用官方 CRS（Qwen2-VL / InternVL），不使用其提供的任何分数。
被复用的 RefCOCO+ 产物（selection.json / coefficients.csv / *_normalisation.json /
predictions/stats_logistic.csv.gz / split_manifest.json）的 sha256 已在
`feasibility_protocol.json.frozen_refcoco_artifacts` 中固定。

### A9.3 工程 gate（指令 §6/§7；在 F2 前冻结）

审计对象：**500–1000 张 positive test 图像的 deterministic subset**（本轮 1000 图，seed=20260929，
`numpy.random.default_rng(seed)` over sorted unique image ids，`audit_subset.csv` 可字节级再生成）。

- **GO**：target proposal **recall@0.5 ≥ 0.90** ∧ **K=5 availability ≥ 0.90**。
- **STOP**：recall@0.5 **< 0.80**（此时测的是 COCO-RPN → GQA 的感知域偏移，不可解读，只汇报）。
- **ENGINEERING GRAY ZONE**：0.80 ≤ recall@0.5 < 0.90 → 先汇报，不得更换 detector / 加大模型 /
  fine-tune RPN / 换 Grounding DINO / 把 N 改成 128（N=64 不因 FineCops 表现而改）。
  若 K=5 根本无法构造 → STOP and report，禁止 post-hoc 改 K。

### A9.4 candidate regime 规则（指令 §10/§11；在 F2 前冻结）

- FineCops 来自 GQA，**不存在也不允许人为构造 COCO category mapping**。
  same-category 的外部等价量是 **GQA scene-graph object `name` 的精确同名**
  （FineCops 官方 level 定义本身就以此区分：level 1 = 图中无同名对象；level 2 = 有同名对象、
  需 1 个 attribute/relation 区分；level 3 = 需 ≥2 个 relation/attribute）。
- 判据（与随机 cohort 使用**同一道 0.90 门槛**，非按结果调整）：
  - same-name K=5 availability **≥ 0.90** → **primary = same-name hard cohort**；
  - **0.50 ≤ 值 < 0.90** → **primary = 官方 level（2/3 vs 1）+ random K=5**，same-name 子集只作 diagnostic；
  - **< 0.50** → **primary = 官方 level**，same-name 子集连 diagnostic 都不再扩张（禁止用假的类别标签补齐）。
- secondary：level × tuple_type 分层；CLIP-hard 仅 optional diagnostic（constructor/evaluator coupling），不作 primary。

### A9.5 external gate（指令 §17–§20；正式评估前冻结，结果后禁止微调）

主统计量 **ΔAUROC = AUROC(E1b) − AUROC(Stats)**（同一 regime、同一 cohort；RefCOCO+ 上 ≈ **+0.032**）。
primary regime 由 A9.4 决定；三 seed 独立报告，**不拼接 seeds**。

- **CONFIRMED**：ΔAUROC ≥ **0.015** ∧ image-cluster paired bootstrap **95% CI lower > 0**
  ∧ (**E-AURC reduction ≥ 5%** ∨ **RER@50 ≥ +3 pp**) ∧ **3 seeds 方向一致** ∧ **≥2/3 seeds individually positive**。
- **STRONG**：ΔAUROC ≥ **0.025** ∧ (E-AURC reduction ≥ **10%** ∨ RER@50 ≥ **+5 pp**)。
- **NOT CONFIRMED**（有效结论，非失败）：ΔAUROC < **0.005** ∧ reduction < **3%** ∧ RER@50 < **2 pp**
  → 写法：A8 的 semantic-increment 效应未能在独立数据集上复现，说明该效应可能依赖 RefCOCO+/
  COCO 的候选构成，而不是 candidate semantics 的普适性质。
- 其余 = **INCONCLUSIVE**；不得据此微调模型 / 改 feature / 调阈值 / 重训，也不得回滚去升级架构。

### A9.6 metadata 与分层报告（指令 §13/§21/§22）

- `results/phase1e_finecops/difficulty_distribution.csv` **必须存在**，按**官方原始 level** 分组，
  禁止按效果重新合并（unexpected level 单独成行，不并入 1/2/3、不退出 `all` 分母）。
- 按 level 报告 B3 Acc / Stats AUROC / E1b AUROC / ΔAUROC，观察 gain 是否随 difficulty 增大
  （**很重要但不作硬 gate**）。
- tuple_type 仅当组内 **n ≥ 300** 才报告（`tuple_type_distribution.csv` 的 `reportable` 列）；
  不得自造 taxonomy。

### A9.7 negative / abstention 边界（指令 §2/§23/§24）

negative text（9,814）与 negative image（8,507）**全部 deferred**：不进入任何 gate、不训练 NONE head、
不与 positive reliability 混合评估；本轮只 download/parse/count
（`negative_distribution.csv`：negative_type / negative_level / negative_cate），作为 Phase-2 abstention
feasibility 的前置证据。positive 与 negative 在代码路径上物理分离（`load_test_expressions` 只读
positive test 文件，遇到 `neg_`/非数字 image id 直接 raise）。

### A9.8 domain-shift sanity（指令 §15/§28）

- 先描述性报告 B3 accuracy / MSP / margin / entropy，与 RefCOCO+ Random-K5 / SameCat-K5 对照；
  **B3 accuracy < 30% → 标记 SEVERE DOMAIN SHIFT 并先汇报**，不得把绝对分数下降解读为 signal 未迁移。
- 必须区分 **cross-dataset degradation**（绝对水平下降）与 **semantic incremental value**
  （ΔAUROC 等相对量）；后者才是本阶段的检验对象。
- absolute ECE 只作 descriptive（指令 §27）：**禁止**任何 temperature / Platt / isotonic / threshold 拟合。

### A9.9 指标与统计（指令 §16/§26）

primary：AUROC_correct / E-AURC / RER@50；secondary：RER@80 / ECE / Brier / NLL；
**image-level clustered paired bootstrap，5000 reps**（cluster key = GQA image id）。
外部复现表（指令 §29）固定 4 行：RefCOCO+ random-K5 / RefCOCO+ same-cat-K5 / FineCops easy /
FineCops hard × {Stats AUROC, E1b AUROC, ΔAUROC, E-AURC reduction, RER@50 gain}；
核心图（指令 §30）横轴 candidate/compositional difficulty、纵轴 ΔAUROC。

### A9.10 禁止事项（与指令 §35 一致）

FineCops training / calibration / threshold fitting / retrain B3 / retrain Stats-E1b /
semantic feature redesign / Transformer / reranking / Grounding DINO 替代 frozen RPN /
MLLM CRS inference / target absence 与 positive reliability 混合 / 依据 FineCops test 结果改任何阈值 /
FineCops train+val 标注进入任何训练、调参或校准环节。

### A9.11 预注册顺序与产物

`F0 download/parse → F1 metadata → F2 500–1000 图 RPN audit → F3 candidate availability →`
**`F4 protocol branch 判定（本轮终点：停下汇报）`** `→ F5 full proposal extraction → F6 CLIP features →`
`F7 frozen B3 → F8 frozen Stats/E1b → F9 bootstrap → F10 gate`。
产物目录 `results/phase1e_finecops/`：`feasibility_protocol.json`、`metadata_audit.json`、
`difficulty_distribution.csv`、`tuple_type_distribution.csv`、`negative_distribution.csv`、
`audit_subset.csv`、`cohort_inventory.csv`、`difficulty_examples.csv`、`image_dims_check.csv`（逐图
标注尺寸 / 实际像素 / scene-graph 尺寸三方比对）、`image_fetch_report.json`、`rpn_audit.csv`、
`rpn_audit_summary.json`、`recall_by_target_size.csv`（仅诊断用，不参与 gate）、`candidate_availability.csv`、
`external_branch_decision.json`、`figures/`、`metadata.json`。

### A9.12 测试要求（指令 §33 的 12 项，全部必须 0 failed）

`tests/test_finecops_external.py`：xywh→xyxy / image ID mapping（含 `neg_` 与非数字 id 被拒）/
target assignment 与等价 proposal 移除 / 冻结 N=64（driver 必须传 `top_n=fe.N_PROPOSALS`，禁止 128 默认）/
audit manifest 确定性 / train+val 标注不可达 / RefCOCO+ normalization 复用且不重拟合 /
frozen coefficient 与 artifact checksum 落盘 / external 代码不含任何 fit 调用 /
difficulty metadata 原样保留（含 unexpected level）/ bootstrap 以 image 为 cluster /
positive 与 negative 路径分离；另加 IoU key 命名规则、engineering gate 边界、regime 分支规则与
aggregate 算术一致性检查。

### A9.13 F0–F4 执行结论（2026-09-29 实测；**未修改本 amendment 任何阈值**）

冻结 COCO-pretrained RPN 在 1,000 图 / 2,235 条 FineCops positive test 上：
`target proposal recall@0.5 = 0.7579`（Wilson 95% CI 0.7398–0.7752，上界仍低于 0.80）、
`recall@0.7 = 0.6376`、`K5 availability = 0.7579`、`same-name K5 availability = 0.1861`。
按 A9 工程 gate（§6）判为 **EXTERNAL STOP**；按 §7 例外条件（K5 同类根本无法构造）判为
**STOP and report**，因此 **不**调整 N、**不**换 detector、**不** fine-tune、**不**用 Grounding DINO 替换。
regime 分支 = `level_primary_only`（same-name 供给连 0.50 诊断线都未达）。
F5–F10 未获授权：需一次由用户作出的**新的协议决策**才能继续；本轮从未查看任何 FineCops 模型结果，
不存在结果后调参。详见 `docs/experiment_log.md` 条目
`p1e-finecops-external-feasibility-audit-20260929-01` 与 `results/phase1e_finecops/external_branch_decision.json`。

---

## Amendment A10 — 2026-09-29 — RefCOCOg Image-Disjoint External Confirmation Feasibility（staged protocol；全部阈值在 proposal 结果产生前冻结）

**staged 声明**：本 amendment 制定于 **A9 verdict = EXTERNAL STOP 已知之后**，属 staged protocol
而不是 initial preregistration。A9 的 stop 原因（冻结 COCO-RPN 在 GQA 图像上
target proposal recall@0.5 = 0.7579 < 0.80 停止线；same-name K5 availability = 0.1861）
**不得被本 amendment 改写、稀释或重新解释**：FineCops F5–F10 永久不获授权，除非另有一次由用户作出的
新的协议决策。本 amendment 不降低 FineCops gate、不换 detector、不改 N、不做“弱形式 external result”。

**本轮范围**：只做 G0–G5 可行性审计（parse / overlap / subsets / frozen-RPN audit /
same-category availability / branch decision）。**任何 reliability 模型的 RefCOCOg 输出
（B3 / Stats / E1b 分数、AUROC、E-AURC、RER）在本文写作时尚未被查看**，本轮甚至未前向模型。
A10.4–A10.8 的全部判据数字在首个 proposal 产物落盘前已写入
`results/phase1e_refcocog_feasibility/feasibility_protocol.json` 的 `frozen_before_run` 块
（artifact 时序可核验：protocol 13:26 早于首个 RPN 产物 13:40）。

### A10.1 边界措辞（不可越界）

RefCOCOg — UMD split 与 RefCOCO+ **共用 COCO 图像域与 COCO object ontology**。因此本外部检验的
合法描述只有一种：

> **cross-dataset external validation under a shared COCO visual domain**

禁止写 `cross-domain visual generalization`（或任何等价说法）。优势恰恰在于控制了视觉/proposal 域，
使外部检验更接近对 **language / candidate-semantic effect** 的 replication 而非感知域检验；
代价是它**不能**声称跨视觉域泛化。论文与日志必须保持这条边界。

### A10.2 数据完整性与 split 类型（G0，实测非引用）

- 来源：RefCOCOg **UMD split**（`refs(umd).p`），**不是** Google object-level split；
  逐文件 sha256 与实测 counts 记录于 `dataset_summary.json`（`image_level_split: true`、
  `shared_images_between_splits: {}` 为硬校验）。
- 目标框来源：UMD refs 文件本身不携带 box；target box 由 `ann_id` join COCO instances 得到。
  本 amendment 要求先把该 join 做成**可证伪的审计**：archive 侧 `instances.json` 与官方
  `instances_train2014.json` 逐 ref 比 IoU，实测 **n=5,023 / IoU min=mean=max=1.0 /
  n_identical=5,023** → verdict `ARCHIVE AND OFFICIAL BOXES IDENTICAL`，两源可互换。
  此证据落盘于 `dataset_summary.json.target_box_source`，因为目标框定义直接决定 recall 的含义。
- GT object 集合（same-category 的候选池）来自官方 `instances_train2014.json`，
  与 RefCOCO+ 侧 `run_proposal_audit.py` 完全同一先例、同一文件。

### A10.3 exposed images 与两个 image-disjoint 子集（G1/G2）

RefCOCOg UMD test **本身不是** external。必须显式减去研究过程中暴露过的图像：

| 子集 | 排除集 | 定义对象 |
|---|---|---|
| `rg_external_devdisjoint` | RefCOCO+ train + val_select + val_calib | 所有参与 B3 training / reliability training / tuning / normalization / calibration / 架构与超参选择的图像 |
| `rg_external_strict` | **ALL** RefCOCO+ images（train + val_select + val_calib + testA + testB） | testA/testB 虽未参与训练，但研究过程反复查看其结果；论文最干净的 external 应优先用 strict |

实现义务：`subset_membership` 返回的是**分区**（strict 优先），故 dev 侧数字是 cumulative，
dev-only extra 单独成组报告；val_select/val_calib 已属 val，不得重复计入；
断言 `strict ∩ dev_only = ∅` 与 `strict_without_development = 0`（strict ⊆ dev）。

**size gate（§9）**：strict ≥ 1500 expressions ∧ ≥ 500 images → `PRIMARY = strict-disjoint`；
strict 不足但 dev ≥ 2000 / ≥ 700 → `EXTERNAL GRAY ZONE`（先汇报，不自动跑）；两者都太小 →
`REFCOCOG EXTERNAL STOP`。

### A10.4 冻结 proposal 系统（§10/§11，逐字复用 RefCOCO+）

`torchvision fasterrcnn_resnet50_fpn`（COCO_V1）RPN stage → class-agnostic →
post-NMS **top-64**（Amendment A2 选定，不因 RefCOCOg 表现而改）→ target = argmax IoU，
IoU **≥ 0.5** 才视为覆盖，target-equivalent proposals 移除。**不为 RefCOCOg 重选 N、
不重训、不 fine-tune、不替换 detector。** 统计量：target recall@0.5 / @0.7、natural omission、
K5/K10 random availability。

**工程 gate（比 A9 更严，因为这是同 COCO 域）**：
- **GO**：recall@0.5 ≥ **0.90** ∧ K5 availability ≥ **0.95**；
- **GRAY**：0.85 ≤ recall@0.5 < 0.90 → 先汇报，不自动跑；
- **STOP**：recall@0.5 < 0.85。

同域 recall 若仍低，说明 annotation / target definition / split plumbing 有问题，**必须先 debug**，
不得靠改生成器绕过。对照侧 RefCOCO+ 数值一律**读盘**自 `results/proposal_audit/`，禁止硬编码。

### A10.5 same-category regime（§13–§16，逐字复用 A8 规则）

proposal → highest-IoU COCO GT object（**IoU ≥ 0.5 才赋类别**）→ same-category K5 =
target + 4 个同类 distractor。报告 ≥1 / ≥2 / ≥4 / ≥9 distractor 覆盖率（含 Wilson CI）。

- availability ≥ **0.60** → same-category 可作 primary external hard regime（≥ 0.75 更好）；
- availability < **0.50** → same-category external confirmation 不成立 → **STOP hard-regime branch**。

**不要求复制 RefCOCO+ 的 90%**：图像与 target 分布不同，外部基准只需 cohort 足够大且统计功效足够；
但必须同时报告 eligible expressions / images / refs 三个量。
**功效 gate**：sameCat-K5 cohort 需 n_expressions ≥ **1000** ∧ n_images ≥ **300**，
否则 `HARD EXTERNAL UNDERPOWERED`，不进入正式 evaluation。

### A10.6 matched random control（§17）

对每一个 hard-eligible expression 同时构建 random-K5 与 sameCat-K5，二者在
sentence/ref、image、target、scorer seed 上**逐字段相同**，只改 distractor composition
（沿用 Phase 1F matched design）。identity 校验必须显式通过，否则 matched comparison 作废。

### A10.7 分布审计（§18/§19：external value 的来源）

RefCOCOg 与 RefCOCO+ 的外部价值主要来自**语言/标注分布 shift**，因此必须证明这不是同一标注分布的重复：

- 语言：expression length 的 mean / median / quantiles、vocabulary size、spatial / relation token prevalence
  （annotation 提供时才报告；**禁止**设计复杂 NLP taxonomy）。比较对象固定为
  RefCOCO+ test（testA+testB，读盘）vs RefCOCOg external。
- 类别：target-category 频率比较 RefCOCO+ Phase 1F hard cohort vs RefCOCOg external hard cohort，
  至少输出 top categories / category entropy / person fraction。
  分布差异**记录为 external shift**，第一版**不做 reweighting**、不强行拉平。

### A10.8 零训练、零重校准（§20/§21）

RefCOCOg train split 即使存在也**完全不使用**；本 external experiment 有 **0 个新训练参数**。
所有 B3 / temperature / Stats Logistic / E1b / normalization / coefficients 在未来正式 evaluation 时
必须**直接恢复** RefCOCO+ 冻结 artifact（A8.4 四重校验，`max|Δ| ≤ 1e-9`，任一 mismatch → STOP）。
本轮不重校准、不前向模型。

### A10.9 G5 branch 判定规则（只能三选一）

- **Branch A — CLEAN EXTERNAL**：strict-disjoint 规模达标 ∧ RPN recall ≥ 0.90 ∧
  same-cat K5 availability ≥ 0.60 ∧ hard cohort ≥ 1000 expr / 300 images → 下一阶段允许
  frozen external confirmation。
- **Branch B — LIMITED EXTERNAL**：strict 不足但 dev-disjoint 达标，或 recall ∈ 0.85–0.90 →
  `EXTERNAL LIMITED`，先汇报，不自动运行。
- **Branch C — EXTERNAL STOP**：recall < 0.85 ∨ same-cat availability < 0.50 ∨ hard cohort 功效不足。

Branch A 通过后，**下一轮**才复现 A8 的 Random-K5 vs SameCat-K5 比较（frozen Stats Logistic vs E1b，
primary statistic ΔAUROC(E1b − Stats)，并检验 Δ_hard − Δ_random）；本轮不运行。

### A10.10 禁止事项（与指令 §29 一致）

CLIP / OpenCLIP feature extraction / B3 inference / Stats-E1b inference / bootstrap 任何 external
模型统计量 / FineCops（A9）continuation / 任何形式的 retrain 或 calibration 拟合 /
RefCOCOg train 标注进入任何环节 / 依据本轮 feasibility 数字修改 A9 或 A8 的既有 gate。

### A10.11 Artifacts

``results/phase1e_refcocog_feasibility/``：`feasibility_protocol.json`、`dataset_summary.json`、
`image_overlap.csv`（+ `image_overlap_summary.json`）、`external_subsets.json`、
`expression_distribution.csv`、`category_distribution.csv`、`audit_expressions.csv`、
`rpn_audit.csv`、`rpn_summary.json`、`candidate_availability.csv`、`hard_cohort.csv`、
`hard_cohort_summary.json`、`image_dims_check.csv`、`image_fetch_report.json`、
`branch_decision.json`、`metadata.json`、`figures/`。

### A10.12 测试要求（指令 §26 的 15 项，全部必须 0 failed）

`tests/test_refcocog_external.py`：UMD split parser / image ID canonicalization /
overlap exclusion / dev-disjoint construction / strict-disjoint construction /
zero-overlap assertion / 冻结 N=64 RPN（driver 必须传 `top_n=fe.N_PROPOSALS`，且不得出现 `Ks=` /
`iou_thresh=` 覆盖）/ COCO category assignment / same-category candidates /
no target-equivalent proposals / matched random-hard cohort identity /
RefCOCOg labels 不进入 training / deterministic manifest / sample-size gate / external branch logic；
另加 GQA-domain 模块不复用（以 **ast import graph** 断言，prose 提及不构成依赖）。

### A10.13 G0–G5 执行结论（2026-09-29 实测；**未修改本 amendment 任何阈值**）

- **G0**：UMD image-level split 确认；test **9,602 expressions / 5,023 refs / 5,023 objects / 2,600 images / 76 类**，
  train 80,512 / 42,226 / 42,224 / 21,899，val 4,896 / 2,573 / 2,573 / 1,300；
  target box 与 COCO instances 逐 ref IoU = 1.0（5,023/5,023 完全一致），unmatched rate 0.0。
- **G1**：RefCOCOg UMD test 与 RefCOCO+ 的 image 重叠 —— train **1,257**、val_select 65、val_calib 58、
  development **1,380**、testA 47、testB 71、ALL RefCOCO+ **1,498**（∴ 必须做 image-disjoint 才配称 external）。
- **G2**：`rg_external_strict` **2,909 expr / 1,102 images / 1,512 refs**，
  `rg_external_devdisjoint` **3,448 expr / 1,220 images / 1,796 refs**（cumulative，dev-only extra 539 行）；
  size gate = **STRICT PRIMARY**。
- **G3（§11/§12）**：recall@0.5 **0.972155**、@0.7 **0.884840**、natural omission **0.027845**、
  K5 = K10 random availability **0.972155**、gt_object_recall@0.5 0.913232 → **EXTERNAL GO**；
  RefCOCO+ 冻结参照（读盘）0.9858742 → Δ **−0.013719**（同域 gap 仅 1.37pp，与 FineCops 的 −22.8pp 形成量级对比）。
- **G4（§14–§19）**：same-cat ≥1 **0.913029** / ≥2 0.828463 / ≥4 **0.649708** [0.632184, 0.666836] /
  ≥9 0.270540 → **SAME-CATEGORY PRIMARY**（≥0.60，未达 0.75“更好”线，如实记为 limited headroom）；
  hard cohort **1,890 expr / 755 images / 978 refs / 69 类** → **HARD COHORT OK**，matched identity 校验通过；
  语言 shift：tokens mean **3.5348 → 8.3875**（strict 8.2499）、vocab 2,942 → 4,038、
  spatial rate 0.4268 → 0.7867、absolute-position rate 0.0330 → 0.1858；
  类别 shift：entropy 2.5329 → 2.9210、person fraction 0.5224 → 0.3852（hard cohort 对 hard cohort）。
- **G5**：**Branch A — CLEAN EXTERNAL**，`next_stage_allowed = true`。
- 本轮从未前向任何模型：不存在 RefCOCOg 的 B3 / Stats / E1b 输出，因此不可能发生结果后调参。
  F5–F10（FineCops）与 A10 之后的正式 external confirmation 是**两个独立的授权决定**，本结论只授权后者进入下一步。
- **可复现性登记（不放宽任何阈值）**：A8.4 校验中的两项 lbfgs 重拟合输出对 **BLAS 线程数**敏感
  （归约顺序非结合）：本机上 T=16 给出 `stats_logistic_pred_max_abs = 2.220e-16` / `e1b_coefficients_max_abs = 0.0`
  （与 A8 当时逐位一致），而线程数被限制到 8/4/1 时该值漂到 2e-08…3.6e-08。三项结构校验（tune AUROC /
  normalisation）恒为 0.0，即**冻结输入本身 bit-identical**。因此 `_TOL = 1e-9` **保持不变**，而是把环境归位：
  正式 external confirmation 与全量 pytest 必须在与产生冻结 artifact 相同的解释器（python 3.10.19 /
  torch 2.5.1+cu121）与**全核 BLAS 线程数**下运行，并在 metadata 中记录线程数。本轮全量 pytest
  **668 collected / 668 passed / 0 failed（113 s）** 即在此条件下得到。



