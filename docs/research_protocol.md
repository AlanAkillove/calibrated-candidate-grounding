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
