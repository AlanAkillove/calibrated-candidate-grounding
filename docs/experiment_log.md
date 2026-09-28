# Experiment Log — `docs/experiment_log.md`

本文件规定 `calibrated-candidate-grounding` 的**实验日志格式与最小可复现留存要求**
（对应原始需求第二十八节 Experiment Logging）。目的：任何一次正式实验之后，必须能在**不重新
训练、不重新提取 feature** 的前提下，仅凭留存文件重算全部 calibration / selective /
abstention 指标与 bootstrap 置信区间。

日志为 **append-only**：不允许覆盖或删除历史条目；纠错通过追加新条目 + `supersedes:` 字段完成。

---

## 1. 目录与命名约定

```text
experiments/
├── phase0/<experiment_id>/
│   ├── config.yaml              # 本次运行的完整解析后配置（含所有默认值）
│   ├── log.yaml                 # 一个或多个条目，schema 见 §2
│   ├── predictions/             # 每个 test cell × seed 一个文件
│   │   └── K{K}_{hardness}_{presence}_seed{S}.parquet
│   ├── bootstrap/               # CI 结果与 replicate 摘要
│   ├── figures/                 # reliability diagrams / risk-coverage curves
│   └── GATE_{Q1,Q2,Q3}.md       # 仅 gate 运行需要
├── phase1/…
└── phase2/…
```

`experiment_id` 格式：`p{phase}-{model}-{short-desc}-{yyyymmdd}-{seq}`，
例如 `p0-independentmlp-kshift-20260926-01`。同一 gate 判定的所有条目必须共享同一
`candidate_protocol_hash`。

## 2. YAML schema（必填字段，逐字对应提示词）

```yaml
experiment_id:          # 全局唯一 ID（§1 格式）
git_commit:             # 完整 40 位 SHA；dirty 时必须写 dirty: true 与 diff 摘要
timestamp:              # ISO-8601，含时区
dataset:                # 例如 refcoco+ / finecops-ref
split:                  # train / val_select / val_calib / testA / testB / natural_omission
candidate_protocol:     # 冻结的 candidate 构造版本 + hash（例如 nested-v1#abcdef）
K:                      # 本条记录对应的候选数（5/10/20/50），或 [5,10,20,50] 表示网格
hardness:               # random / same_category_hard / clip_hard_diagnostic
target_presence:        # present / synthetic_omission / natural_miss / presence_shift_{10,25,50,75}
backbone:               # openclip-vit-b-32 (frozen, offline cache) + 权重 hash
model:                  # B0_random / B1_cosine / B2_global_temp / B3_independent_mlp /
                        # C1_global_T / C2_K_aware_T / C3_stats_only / deepsets / presence_*
seed:                   # 单个整数；3 seeds 时每个 seed 一条记录（外加一条聚合记录）
training_config:        # optimizer/lr/schedule/epochs/early-stopping split/参数量/损失
calibration_config:     # 拟合 split（必须为 val_calib）、拟合目标、bin 策略、温度形式、阈值来源
metrics:                # 见 §3 的必含键
```

字段纪律：

- `split` 与 `calibration_config.fit_split` 必须同时出现，且后者只能是 `val_calib`；
  若 gate 判定涉及 test，只允许 `eval_split: testA/testB`，且 `used_for_selection: false`。
- `candidate_protocol` 必须是**预生成并冻结**的集合 hash；若 hash 与主 grid 不同，该条记录
  不得参与 gate 判定。
- `hardness: clip_hard_diagnostic` 的记录在导出表格时必须带 `diagnostic` 标记，禁止与
  random 混为 "真实分布" 结论。
- `seed` 不得只报最好的一次；聚合记录需含 seed 间 mean/std。

## 3. `metrics` 必含键

```yaml
metrics:
  ranking:      {top1_acc, top5_acc, mrr_optional, acc_by_K, acc_by_hardness}
  calibration:  {top_label_ece_adaptive, top_label_ece_equal_width,
                 binary_correctness_brier, top_label_correctness_nll,
                 confidence_accuracy_gap, multiclass_nll, multiclass_brier}
  selective:    {aurc, risk_at_50_coverage, risk_at_80_coverage,
                 risk_at_90_coverage, risk_at_95_coverage, selective_accuracy}
  abstention:   {auroc_presence, auprc_presence, fpr_at_95_tpr,
                 none_precision, none_recall, none_f1, false_selection_rate}
  counts:       {n_samples, n_unique_images, n_excluded_for_insufficient_candidates}
  uncertainty:  {bootstrap_replicates, resampling_unit, ci_level, deltas: {...}}
  risk_definition: "risk = 1 - accuracy"   # 必须显式写出
```

## 4. 必须同时保存的原始产物（不得只存最终 accuracy）

| 产物 | 内容 | 为什么必需 |
|---|---|---|
| `predictions` | 每样本：selected index、selected candidate id、confidence、correctness | 重算 accuracy/risk-coverage |
| **`raw logits`** | 每个 candidate 的**未归一化**分数（含温度前） | 任何后续校准（含新 T、新校准器）都必须可离线复现，不重跑模型 |
| `candidate IDs` | 每样本完整 candidate index 列表 + target index（或 null） | 验证嵌套性 / target 恒定性 / 复现 hardness |
| `confidence` | 与 logits 对应的 max probability（显式存，避免重算歧义） | ECE / Brier / NLL |
| `correctness` | \(r=\mathbf 1[\hat c=c^*]\)；presence 任务的真实标签 | top-label 校准与 abstention 指标 |
| `score statistics` | K、max_score、top1_top2_margin、entropy、mean/std（及可选 top3） | C3 stats-only 与 Phase 2 N3 可直接复算 |
| `bootstrap summary` | replicate 数、resampling 单位、随机种子、差值分布分位数、CI | gate 判据的可审计性 |
| per-image 聚合视图 | 每 image 的样本数与均值 | 敏感性分析（不改变主判据） |

格式建议：`parquet` 或 `npz`；禁止只留 markdown 表格。任何未留存 logits 的运行只能标记为
`provisional`，不得作为 gate 依据。

## 5. Gate 判定条目

gate 判定必须单独成文（`GATE_Qx.md` + `log.yaml` 中的 `gate:` 段），至少包含：

```yaml
gate:
  name: Q1
  decision: GO | NO_GO
  frozen_criteria_source: docs/research_protocol.md#11
  per_cell:            # 每个 OOD cell 一条
    - cell: {K: 20, hardness: random}
      delta_acc_pp: null        # 数值在真实运行后填入
      delta_ece_pp: null
      aurc_relative_worsening_pct: null
      ci_excludes_zero: null
      cosine_same_direction: null
      independent_mlp_same_direction: null
  criteria_met: {threshold_breached_cells: null, ci_ok: null, both_models_ok: null, two_cells_ok: null}
```

判定所用阈值一律**引用**协议文本，不得在日志中重写数字（避免复制粘贴漂移）。

---

## 6. 示例条目（**EXAMPLE — 非真实结果，数值为占位符**）

> ⚠️ 以下整段为格式示例。其中的数字、结论字段全部为 `null` / 占位字符串，
> **不代表任何已观测现象**，禁止被引用为实验结果。

```yaml
# ===== EXAMPLE (NOT A REAL RESULT) =====
experiment_id: p0-independentmlp-formatexample-00000000-00
git_commit: "0000000000000000000000000000000000000000"
dirty: true            # EXAMPLE 占位
timestamp: "0000-00-00T00:00:00+00:00"
dataset: refcoco+              # EXAMPLE
split: testA                   # EXAMPLE（评估用 split）
candidate_protocol: nested-v1#EXAMPLEHASH
K: [5, 10, 20, 50]             # EXAMPLE：4×2 网格
hardness: [random, same_category_hard]
target_presence: present
backbone: openclip-vit-b-32-frozen#EXAMPLEHASH
model: B3_independent_mlp
seed: 0                        # EXAMPLE；正式记录需 seed 0/1/2 各一条 + 一条聚合
training_config:
  fit_split: train
  selection_split: val_select          # 架构/optimizer/early stopping/超参
  used_for_selection: [val_select]
  used_for_calibration: []             # 严禁 test / val_calib
  optimizer: "EXAMPLE: adamw, lr=<>, wd=<>"
  epochs: null
  early_stopping_on: "val_select top1"
  param_count: null                    # 必须 ≤ 1M
  loss: "EXAMPLE: cross-entropy over candidates"
calibration_config:
  form: "global temperature T (B2/C1)"
  fit_split: val_calib
  fit_objective: "val_calib NLL"
  applied_to: [testA, testB]
  used_test_for_fitting: false         # 必须为 false
metrics:
  ranking: {top1_acc: null, top5_acc: null, mrr_optional: null, acc_by_K: null, acc_by_hardness: null}
  calibration:
    top_label_ece_adaptive: null
    top_label_ece_equal_width: null
    binary_correctness_brier: null
    top_label_correctness_nll: null
    confidence_accuracy_gap: null
    multiclass_nll: null
    multiclass_brier: null
  selective:
    aurc: null
    risk_at_50_coverage: null
    risk_at_80_coverage: null
    risk_at_90_coverage: null
    risk_at_95_coverage: null
    selective_accuracy: null
  abstention:
    auroc_presence: null
    auprc_presence: null
    fpr_at_95_tpr: null
    none_precision: null
    none_recall: null
    none_f1: null
    false_selection_rate: null
  counts:
    n_samples: null
    n_unique_images: null
    n_excluded_for_insufficient_candidates: null    # 必须报告，不得静默过滤
  uncertainty:
    bootstrap_replicates: null                      # 正式运行 ≥ 5000
    resampling_unit: image                          # image-level 优先于 expression-level
    ci_level: 0.95
    deltas: {delta_acc_pp: null, delta_ece_pp: null, aurc_relative_pct: null}
  risk_definition: "risk = 1 - accuracy"
artifacts:
  predictions: experiments/phase0/p0-independentmlp-formatexample-00000000-00/predictions/
  raw_logits:  experiments/phase0/p0-independentmlp-formatexample-00000000-00/predictions/
  candidate_ids: included_in_predictions
  bootstrap_summary: experiments/phase0/p0-independentmlp-formatexample-00000000-00/bootstrap/
  figures:     experiments/phase0/p0-independentmlp-formatexample-00000000-00/figures/
notes: >
  EXAMPLE ONLY. 该条目仅示范 schema 与留存要求；不得出现在任何结果表或图表中。
```

## 7. 运行前后检查清单（每次正式实验）

```text
[ ] 前  git commit 已记录；工作树 clean（或 dirty: true 且附 diff 摘要）
[ ] 前  config 已解析并落盘（含全部默认值，不允许靠终端参数记忆复现）
[ ] 前  candidate_protocol hash 与本次 gate 所用一致
[ ] 前  split 用途声明已写入（selection / calibration / eval 三者互不重叠）
[ ] 后  predictions + raw logits + candidate IDs + confidence + correctness + score stats 已保存
[ ] 后  bootstrap summary（replicates ≥ 5000、resampling_unit: image）已保存
[ ] 后  被排除样本数已记录并解释（K=50 候选不足、图像缺失等）
[ ] 后  reliability diagrams 与 risk–coverage curves 已出图，且标注样本量与 bin 策略
[ ] 后  3 seeds 全部记录 + 聚合记录（mean/std）
[ ] 后  若为 gate 运行：GATE_Qx.md 已写，包含日期、commit、逐条判据核对
```

---

## 8. Amendment 记录（append-only；非实验结果，不入结果表）

> 本节记录对 `docs/research_protocol.md` 的 protocol amendment，遵 §14 amendment 流程（新增
> 不覆盖、记录日期、解释原因、保留原始标准）。以下条目为**协议变更日志**，**不是**任何已观测
> 实验结果，禁止被引用为 Phase 0 数据。

```yaml
# ===== PROTOCOL AMENDMENT RECORD (NOT AN EXPERIMENT RESULT) =====
type: protocol_amendment
date: "2026-09-27"
target: docs/research_protocol.md
amendments:
  - id: A1
    title: Candidate regime taxonomy & evidence hierarchy
    summary: >
      正式定义三类 candidate regimes（Random = 主 controlled distribution；GT Same-Category
      Hard = GT-assisted diagnostic stress test；CLIP-Hard = model-assisted adversarial
      diagnostic stress test）；新增 evidence hierarchy（Primary=cardinality shift /
      Secondary=GT same-category composition / Adversarial diagnostic=CLIP-hard）；收紧
      Gate Q1：不得仅凭 CLIP-hard cells 通过，且至少一个 OOD cell 属于 cardinality shift。
      原始 Gate 数字（3pp/3pp/20% 等）不变。
  - id: A2
    title: Class-agnostic RPN proposal bank & N-selection rule
    summary: >
      主 proposal bank 冻结为 class-agnostic RPN proposals（非 Faster R-CNN 最终 detection
      boxes）；proposal 只含 box+objectness，COCO category 仅作 offline diagnostic metadata；
      禁用 detector predicted class 定义主 candidate space；记录 torchvision 0.20.1 / torch 2.5.1
      与 src/ccg/data/rpn.py；预注册 N-selection 规则（N=64 能支撑 >=90% K=50 则用 64，否则 128）；
      澄清 GT-equivalent vs proposal-near-duplicate；正式选定 fasterrcnn_resnet50_fpn(COCO_V1) 之 RPN。
  - id: A3
    title: Proposal audit outcome & N-selection decision (results record)
    summary: >
      登记 proposal-system audit（2026-09-27）outcome 并按 A2.4 预注册规则执行 N-selection：正式选定
      N=64 作为主 proposal bank 规模；无法支持 K=50 的 expression 必须在 K=50 cells 显式报告排除计数
      （禁止静默过滤）；natural omission 稀缺记为 RQ4 统计功效风险（Phase 2 前做功效评估，不影响 Gate）；
      澄清全部 RefCOCO+ 图像来自 COCO train2014（val2014 非必需）；A2.5 duplicate-suppression 待决项关闭
      （不引入新 IoU threshold）。全部 gate 数字不变；观测数字见 §9 条目 `audit-proposal-001`。
  - id: A4
    title: Phase 0A fixed-CLIP cosine audit outcome & calibration interpretation constraints
    summary: >
      登记 Phase 0A（B1 frozen CLIP cosine，random regime，全量 19992 图 / 4 splits，K∈{5,10,20,50}，
      bootstrap 5000 replicates）outcome：三项硬 sanity check 全过、嵌套集理论性质未被违反（无
      VALIDATION_FAILURE.json）；并新增三条解读约束（不改任何条款与 gate 数字）：(1) T* 收敛于拟合
      搜索界 [0.05, 100] 下界边缘（0.05+ε）时必须披露边界凝结；(2) oracle per-K 温度诊断在该界内为
      边界简并（4 个 T_K 同值、ΔECE≡0），不得解读为“per-K 温度无法修复校准漂移”；(3) native ECE 的
      跨 K 增幅（≤5.2pp）与 ranking accuracy 降幅（至 34.7pp）必须并列报告，AURC 恶化不得先行表述为
      独立于 ranking 的 selective-risk shift。Phase 0A 为 B1/random 单格证据，不构成 Gate Q1/Q2 判定。
      观测数字见 §9 条目 `p0-cosine-kcardinality-20260927-01` 与 `audit-naturalomission-full-001`。
reason: >
  首轮协议审计发现：（1）GT 信息不对称——same-category hard negatives 与 CLIP-hard 一样使用
  GT/模型信息，但原协议只将 CLIP-hard 标为 diagnostic（且 CLIP-hard↔B1 存在构造器耦合）；
  （2）proposal bank 定义模糊（detector vs RPN、是否含预测类别）；（3）N 选择需要工程预注册
  标准（避免事后调 N 规避 K=50 候选不足）。（A3 追加原因：proposal-system audit 完成，登记 outcome
  并按 A2.4 规则执行 N-selection；结果数字见 §9 条目 `audit-proposal-001`。）（A4 追加原因：Phase 0A
  （B1 cosine）全量审计完成；T* 与 oracle per-K 温度均在拟合搜索下界凝结，必须显式登记解读约束。）
original_criteria_preserved: true
related_docs:
  - docs/research_protocol.md#amendment-a1
  - docs/research_protocol.md#amendment-a2
  - docs/research_protocol.md#amendment-a3
  - docs/research_protocol.md#amendment-a4
  - docs/dataset_protocol.md#10-proposal-system-audit-design
related_entries:
  - experiment_id: audit-proposal-001    # A3 依据的观测数字见 §9；本节不复制数字
  - experiment_id: p0-cosine-kcardinality-20260927-01    # A4 依据的观测数字见 §9；本节不复制全部数字
  - experiment_id: audit-naturalomission-full-001        # A4.4 全数据 omission counting
results_claimed: false
```

---

## 9. 正式条目（append-only）

> 以下为**真实结果**条目（非示例）。适用与 §8 相同的 append-only 纪律：不得覆盖或删除；
> 纠错只能追加新条目并以 `supersedes:` 指向被修正条目。

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: audit-proposal-001
git_commit: "f9b79b2087796398151d2dd5743ca88872729a0c"   # 实验运行时的审计流水线提交
timestamp: "2026-09-27"        # 日期级（运行日）
dataset: refcoco+              # refs(unc).p（latin1 解析）+ instances.json；图像 = COCO train2014
split: [train, val_select]     # image 级 audit subset：train 池 1000 + val_select 500（共 1500 图）
candidate_protocol: >-
  class-agnostic RPN top-N proposal bank（RPN NMS 后按 objectness 排序取 top-64；query-independent）；
  本 audit 对同一批 RPN 输出同时报告 N=64 与 N=128 两档（A2 冻结语义，未使用 detector predicted class）
K: null                        # 本 audit 不涉及 candidate K 网格（{5,10,20,50} 未使用）；proposal bank 档位 N ∈ {64, 128}
hardness: null                 # 未涉及（未构造 candidate sets / 三种 hardness regime）
target_presence: "target-present 分析（recall / K-availability）+ natural omission 统计（P(max IoU<0.5)）"
backbone: "fasterrcnn_resnet50_fpn RPN (torchvision 0.20.1)"   # COCO_V1 权重，frozen，仅推理；仅 RPN 子模块
model: class-agnostic RPN
seed: 20260927
training_config: null          # 无训练（frozen detector 仅推理）
calibration_config: null       # 未拟合任何 calibration（本 audit 不产生校准结果）
metrics:
  recall:
    gt_object_level:           # 15349 个 iscrowd=0 GT objects
      N64:  {recall_at_0.5: 0.8085217277998566, ci95: [0.8022200834625711, 0.8146689750424029], recall_at_0.7: 0.6793276434946902}
      N128: {recall_at_0.5: 0.8801876343735748, recall_at_0.7: 0.752166264903251}
    refcoco_target_level:      # 3752 expressions
      N64:  {recall_at_0.5: 0.9858742004264393, ci95: [0.9815704983373593, 0.989183965855167], recall_at_0.7: 0.8997867803837953}
      N128: {recall_at_0.5: 0.994136460554371, recall_at_0.7: 0.9304371002132196}
  natural_omission:            # P(max IoU<0.5)，IoU 阈值 0.5
    expression_level:
      N64:  {count: "53/3752", rate: 0.014125799573560768, ci95: [0.010816034144833078, 0.018429501662640808]}
      N128: {count: "22/3752", rate: 0.005863539445628998}
    gt_object_level:
      N64:  {count: "2939/15349", rate: 0.19147827220014332}
      N128: {count: "1839/15349", rate: 0.11981236562642518}
  target_max_iou:              # expression 级 target max-IoU 分布
    N64:  {p25: 0.767509251832962, median: 0.8250076770782471, p75: 0.879587858915329}
    N128: {p25: 0.7778179496526718, median: 0.8297169506549835, p75: 0.8810313940048218}
  k_availability:              # 定义：valid distractors = N − |{target} ∪ to_remove| ≥ K−1
    N64:
      K5:  {count: "3752/3752", rate: 1.0}
      K10: {count: "3752/3752", rate: 1.0}
      K20: {count: "3752/3752", rate: 1.0}
      K50: {count: "3719/3752", rate: 0.9912046908315565, ci95: [0.9876741623023597, 0.9937303782938145]}
    N128: {K5: 1.0, K10: 1.0, K20: 1.0, K50: 1.0}   # 全部 1.0
  same_category_availability:  # target category unknown = 0
    N64:  {ge1: 0.9752132196162047, ge2: 0.9616204690831557, ge4: 0.900319829424307, ge9: 0.6660447761194029}
    N128: {ge1: 0.988272921108742, ge2: 0.9816098081023454, ge4: 0.9536247334754797, ge9: 0.8091684434968017}
  redundancy:
    pairwise_iou_gt_0.7: {N64: 0.001429563492063492, N128: 0.0006508366141732283}
    pairwise_iou_gt_0.9: {N64: 3.5052910052910054e-05, N128: 1.2057086614173228e-05}
    remaining_candidate_count_median: {N64: 59, N128: 120}   # 移除 target-equivalent 后
  counts:
    n_images: 1500
    n_analyzed: 1500
    n_cached: 5
    n_extracted: 1495
    n_failed: 0
    n_gt_objects_iscrowd0: 15349
    n_expressions: 3752
    n_excluded_for_insufficient_candidates:
      "N64, K=50": 33          # 33/3752 = 0.88%；必须在 K=50 cells 显式报告排除计数，禁止静默过滤
      "N128": 0
  run:
    command: "scripts/run_proposal_audit.py --device cuda --figures"
    device: cuda
    runtime_s: 156.0
    gpu_peak_mb: 664
    proposal_cache_mb: 4.7
    proposal_cache_files: 1500   # npz
decision:
  n_selection: "选定 N=64（research_protocol.md A2.4 预注册规则：0.9912046908315565 ≥ 0.90）"
  k50_reporting_obligation: "33/3752 = 0.88% 无法支持 K=50 的 expression 必须在 K=50 cells 显式报告排除计数（禁止静默过滤）"
  rq4_risk: "expression 级 natural omission 仅 0.014125799573560768（N=64）→ RQ4 的 natural omission split 统计功效有限，需在 Phase 2 前做功效评估（不影响 Gate）"
  redundancy_note: "冗余极低（两两 IoU>0.7 = 0.001429563492063492，N=64）→ 不存在“用近重复 box 凑 K=50”的问题；不引入新的 duplicate suppression 阈值"
artifacts:
  summary: results/proposal_audit/summary.json
  csv:
    - results/proposal_audit/recall_by_N.csv
    - results/proposal_audit/candidate_availability_by_N.csv
    - results/proposal_audit/same_category_availability.csv
    - results/proposal_audit/natural_omission.csv
    - results/proposal_audit/proposal_iou_statistics.csv
  figures: results/proposal_audit/figures/     # 5 张
  proposal_cache: cache/proposal_audit/        # 1500 npz
notes: >
  本条目为 proposal-system audit（S1；非 decision-model 实验，§3 的 ranking/calibration/selective/
  abstention 指标不适用，故 metrics 按 audit 口径记录）。不产生任何 Gate Q1/Q2/Q3 结论；唯一作用为
  执行 A2.4 N-selection（选定 N=64）与关闭 A2.5 duplicate-suppression 待决项（不引入新阈值）。
  audit 仅使用 train / val_select 的 image 级子集，不查看 testA / testB。candidate-sets 层 hash 不适用
  （本 audit 未构造 candidate sets）。协议层登记见 docs/research_protocol.md Amendment A3；
  dataset 侧实测更新见 docs/dataset_protocol.md §1.4。
```

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p0-cosine-kcardinality-20260927-01
git_commit: "7e24cdedbbf15d3e62a048637e6d042f5eb225f4"   # 运行时的 HEAD（夜跑链路）
dirty: true                    # Phase 0A 链路代码（src/ccg/experiment/phase0a.py、data/manifests.py、
                               # features/cache.py、scripts/run_phase0a.py、8 个新测试文件等）运行时尚未提交；
                               # 随本条目同批收尾 commit 入库（新增文件 + .gitignore/resume 修改）
timestamp: "2026-09-27T15:20:26Z"   # 产物 created_utc；夜跑链路完成 15:21:46Z（logs_night/STATUS.json）
dataset: refcoco+              # refs(unc).p；图像 = COCO train2014（全部 19992 图）；candidate = 冻结 manifest（random）
split: [val_select, val_calib, testA, testB]   # 评估用；仅 val_calib 参与拟合（T*）；testA/testB 从不参与
candidate_protocol: >-
  nested-v1#ebd2690e4b1c1d524fbee80e6ba48dbfadd9f2f5e58b08566228d409afd55747（manifests-v1；
  regime=random，seed=20260927，top_n=64，iou_thresh=0.5）；C_K = [target] + distractor_order[:K-1]；
  三份 manifest（testA 1975 / testB 1798 / val 3805 refs）已复制入 candidate_manifests/
K: [5, 10, 20, 50]
hardness: random
target_presence: present       # 全部候选集含 target；target-missing / K=50 候选不足的 sentence 显式排除并计数（禁止静默过滤）
backbone: >-
  openclip-vit-b-32（ViT-B-32 laion2b_s34b_b79k，frozen，离线缓存；checkpoint
  sha256=1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad；fp16；特征缓存 phase0a-v1）
model: B1_cosine               # 冻结 CLIP cosine 打分；native = softmax(100.0 * s)（alpha=100，primary 变体）
seed: 0                        # bootstrap seed（打分确定性；无训练）
training_config: null          # 无训练（frozen backbone）
calibration_config:
  form: "global temperature T（跨 K 单标量；B2/C1）"
  fit_split: val_calib         # CalibrationIsolationError 保护在位；testA/testB 从不参与拟合
  fit_objective: "candidate-set NLL（ccg.calibration.mean_nll），在 val_calib 上 pool K∈{5,10}"
  temperature_bounds: [0.05, 100.0]
  fitted_temperature: 0.050000102135425877   # ⚠ 收敛于搜索下界边缘（0.05+ε）——边界凝结；解读约束见 A4.3
  nll_before: 1.9153269614775112
  nll_after: 1.4809285520674216
  applied_to: [val_select, val_calib, testA, testB]
  used_test_for_fitting: false
metrics:
  ranking:                     # __pooled__ common cohort（n=20799；跨 K 共享分母）
    top1_acc: {K5: 0.5348814846867638, K10: 0.3899706716669071, K20: 0.28568681186595507, K50: 0.18789364873311218}
    top5_acc: {K5: 1.0, K10: 0.8243665560844271, K20: 0.6416654646858022, K50: 0.4355978652819847}
    mrr_optional: {K5: 0.7112112761831498, K10: 0.5730947840135782, K20: 0.45031833929579335, K50: 0.31616990790739957}
    acc_by_K: {delta_acc_vs_k5: {K10: -0.1449108130198567, K20: -0.2491946728208087, K50: -0.34698783595365157}}
    acc_by_hardness: null      # random regime only；hardness 网格不属本 audit
  calibration:                 # native = primary；__pooled__ common
    top_label_ece_adaptive: {K5: 0.2482983856014841, K10: 0.2930576841957816, K20: 0.3001323987721474, K50: 0.27722758217776294}
    top_label_ece_equal_width: {K5: 0.24843016615369135, K10: 0.2930576841957816, K20: 0.3001323987721474, K50: 0.27722758217776294}
    binary_correctness_brier: {K5: 0.27630705208943346, K10: 0.2926699080224997, K20: 0.2724260762285447, K50: 0.2224079476823505}
    top_label_correctness_nll: {K5: 0.934625176599185, K10: 0.9437111353057122, K20: 0.8522327878977813, K50: 0.6926265024575344}
    confidence_accuracy_gap: {K5: 0.2482983856014841, K10: 0.29305768419578154, K20: 0.3001323987721474, K50: 0.277227582177763}
    multiclass_nll: {K5: 1.7448319904705683, K10: 2.5661456543806516, K20: 3.371517977443947, K50: 4.418095654277915}
    multiclass_brier: {K5: 0.6812383741952217, K10: 0.8538363416663426, K20: 0.955693363329776, K50: 1.024016417554413}
    diagnostics:               # 非替换变体（协议 §11）；bootstrap 覆盖三变体
      T1_ece_adaptive: {K5: 0.3240476601179348, K10: 0.2833783502634388, K20: 0.23192738656111678, K50: 0.16619333687008553}
      global_T_ece_adaptive: {K5: 0.09897379577661387, K10: 0.1122205477150405, K20: 0.11592289700418534, K50: 0.10381579623944548}   # T*=0.0500001
  selective:                   # __pooled__ common；risk = 1 - accuracy
    aurc: {K5: 0.2716234194650477, K10: 0.4170030107913692, K20: 0.5342065403866352, K50: 0.6614556683692103}
    risk_at_50_coverage: {K5: 0.2892307692307692, K10: 0.4522115384615385, K20: 0.5792307692307692, K50: 0.7048076923076922}
    risk_at_80_coverage: {K5: 0.40691105769230773, K10: 0.5549278846153847, K20: 0.6694711538461539, K50: 0.7763221153846154}
    risk_at_90_coverage: {K5: 0.4361111111111111, K10: 0.583119658119658, K20: 0.6918269230769231, K50: 0.7952457264957264}
    risk_at_95_coverage: {K5: 0.4505566801619433, K10: 0.5964574898785425, K20: 0.7031376518218624, K50: 0.8038967611336032}
    selective_accuracy_at_50_coverage: {K5: 0.7107692307692308, K10: 0.5477884615384615, K20: 0.4207692307692308, K50: 0.2951923076923078}
  abstention: null             # Phase 0A 仅 target-present；presence/abstention 属 Phase 2
  counts:
    n_samples: 21373           # __pooled__ 全部 sentence
    n_common_samples: 20799    # 跨 K 共享分母（= K=50 可用数）
    n_unique_images: 2981      # common cohort 的 image 聚类数（bootstrap resampling unit）
    n_excluded_for_insufficient_candidates:
      target_missing_sentences: 340     # 全 split；K5/10/20 亦排除（val_select 73 / val_calib 77 / testA 59 / testB 131）
      insufficient_for_K50: 234         # 仅 K=50（val_select 43 / val_calib 52 / testA 21 / testB 118）——显式报告，非静默过滤
  uncertainty:
    bootstrap_replicates: 5000
    resampling_unit: image
    ci_level: 0.95
    deltas:                    # diff = metric(K_a) − metric(K_b)；__pooled__ common；[ci_low, ci_high]
      accuracy_native:
        5_10: {diff: 0.1449108130198567, ci: [0.13910388035127227, 0.1509778404872323]}
        5_20: {diff: 0.2491946728208087, ci: [0.24186372154578595, 0.2570120280256724]}
        5_50: {diff: 0.34698783595365157, ci: [0.3387827362912931, 0.3557113953011908]}
      ece_adaptive_native:
        5_10: {diff: -0.0447592985942975, ci: [-0.051342230670797184, -0.03849286400674796]}
        5_20: {diff: -0.05183401317066333, ci: [-0.06008373732538603, -0.04356577643715734]}
        5_50: {diff: -0.02892919657627885, ci: [-0.0382075533820454, -0.019648158118321934]}
      ece_adaptive_global_T:   # 诊断变体
        5_10: {diff: -0.013246751938426629, ci: [-0.01917875651966902, -0.007127518323151044]}
        5_20: {diff: -0.01694910122757147, ci: [-0.024240376328465922, -0.00871013315753057]}
        5_50: {diff: -0.0048420004628316055, ci: [-0.012637243279265959, 0.0038811307432282016]}   # CI 跨 0
      aurc_native:
        5_10: {diff: -0.1453795913263215, ci: [-0.15348281538403757, -0.137325052926201]}
        5_20: {diff: -0.2625831209215875, ci: [-0.27334474333196634, -0.2520291620290432]}
        5_50: {diff: -0.3898322489041626, ci: [-0.40179444137142634, -0.3780979501873425]}
  risk_definition: "risk = 1 - accuracy"
hard_checks:                   # 协议 §17-19 STOP 契约；三项全过 → 未写 VALIDATION_FAILURE.json（退出码 0）
  score_invariance: {status: passed, atol: 0.0, n_candidate_pairs: 1148625, n_violations: 0, max_abs_diff: 0.0}
  rank_monotonic: {status: passed, n_sentences: 20799, n_K_pairs: 6, n_violations: 0}   # rank(K') >= rank(K), K'>K
  accuracy_monotonic: {status: passed, acc_by_K: {K5: 0.5348814846867638, K10: 0.3899706716669071, K20: 0.28568681186595507, K50: 0.18789364873311218}, n_violations: 0}
  confidence_monotonic: not_checked   # 明示不查（§20：校准随 K 漂移是研究对象）
run:
  command: "tools/night_run.py：extract_features --resume → audit_cache --sample 100 → run_phase0a --bootstrap-replicates 5000 → pytest -q（全部一次通过，attempts=1）"
  device: "cuda（特征提取，RTX 4060 Laptop）/ cpu（打分 + bootstrap）"
  step_seconds: {extract_features: 6045.2, audit_cache: 80.1, run_phase0a: 530.4, pytest: 70.1}
  phase0a_internal_seconds: {total: 519.5055997000018, bootstrap: 481.6167573999992}
  feature_cache: {n_images: 19992, n_crops: 1279488, n_sentences: 141564, cache_bytes: 1507745318}
  extract_notes: "resumed=true（本次补齐 11416 图 / 730624 crops；214.9 crops/s；peak VRAM 833.0 MB；GPU wall 6042.0s）"
artifacts:
  root: results/phase0a_cosine/
  predictions: raw_predictions/K{5,10,20,50}.npz（含原始 logits）+ prediction_summary.csv
  metrics: [ranking_metrics.csv, calibration_metrics.csv, selective_metrics.csv]
  calibration: [reliability_bins.csv, calibration_map_shift.csv, global_temperature.json, oracle_temperature_diagnostic.json]
  bootstrap: [bootstrap_ci.csv（120 行）, selective_curves.npz]
  figures: figures/（reliability_per_K.png / risk_coverage_per_K.png / ece_brier_acc_vs_K.png / confidence_hist_per_K.png）
  manifests: candidate_manifests/（三份 manifest + meta + manifest_summary.json）
  provenance: [metadata.json, cohort_summary.json]
notes: >
  B1（frozen CLIP cosine）全量审计，random regime；**不构成任何 Gate Q1/Q2 判定**（§11 + A1.4 要求
  4×2 网格与 B2/B3 等证据；本条目为 B1/random 单格证据）。协议层登记见 research_protocol.md
  Amendment A4（含三条解读约束：T* 边界凝结披露、oracle 简并禁止误读、calibration 证据须与 ranking
  并列报告）。native ECE 跨 K 变化（+2.9~+5.2pp vs K5）远小于 accuracy 降幅（-14.5~-34.7pp）；
  global-T 后 ECE ≈9.9%~11.6%（残差 ≤1.7pp；pooled 5→50 diff CI 跨 0）；oracle per-K 温度在搜索界内
  与 global 完全同值（边界简并，ΔECE≡0），该简并不得解读为“per-K 温度无法修复漂移”。
  全数据 natural omission counting 见条目 audit-naturalomission-full-001。
```

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: audit-naturalomission-full-001
git_commit: "7e24cdedbbf15d3e62a048637e6d042f5eb225f4"   # 运行时的 HEAD
dirty: true                    # scripts/natural_omission_full.py 与测试当时未提交；随收尾 commit 入库
timestamp: "2026-09-27"        # 日期级（产物 mtime 2026-09-27 ≈ 12:19:55Z；先于夜跑链路）
dataset: refcoco+              # refs(unc).p 全量 49856 ref；bank = cache/proposals.h5（bank-v1，top-64 class-agnostic RPN）
split: [train, val, testA, testB]   # val = 原 refcoco+ val split 全集（本 counting 不细分 val_select/val_calib）
candidate_protocol: "bank-v1（class-agnostic RPN top-64，frozen，post-NMS；created 2026-09-27T08:33:48Z）"
K: null                        # 不构造 candidate sets；仅对 target box 与 bank 行做 max-IoU 计数
hardness: null
target_presence: "natural_miss counting（P(max IoU < 0.5)；counting only，无 presence 模型）"
backbone: "fasterrcnn_resnet50_fpn RPN（torchvision 0.20.1+cu121，COCO_V1，frozen）"
model: null                    # counting only（multi-agent brief section 27 冻结计数规则）
seed: null
training_config: null
calibration_config: null
metrics:
  natural_omission_expression_level:   # 主口径（一个 sentence 一单位）；IoU 0.5；Wilson 95% CI
    train: {count: "1699/120191", rate: 0.01413583379787172, ci95: [0.013483787900559803, 0.014818937530212762]}
    val:   {count: "150/10758", rate: 0.013943112102621304, ci95: [0.01189447686833437, 0.016338757859818472]}
    testA: {count: "59/5726", rate: 0.010303877052043312, ci95: [0.00799687921676721, 0.013267513153171956]}
    testB: {count: "131/4889", rate: 0.026794845571691553, ci95: [0.022626312382751628, 0.03170644999402345]}
  other_units:                 # summary.json by_split 全量齐备；示例（val）
    unique_object_val: {count: "56/3805", rate: 0.014717477003942181}
    image_val: {count: "50/1500", rate: 0.03333333333333333}
  threshold_sensitivity: "IoU 0.3 / 0.7 行 is_primary=False（threshold_sensitivity.csv）；主判界 0.5 不变"
  max_iou_quantiles: {n: 49856, mean: 0.8107071097452666, p10: 0.7047449052333832, p25: 0.7689094096422195, p50: 0.8245694935321808, p75: 0.8766386657953262, p90: 0.9151924848556519}
  counts:
    n_refs_total: 49856
    n_refs_scored: 49856
    n_refs_missing_bank_image: 0
    n_refs_unscorable: 0
  uncertainty: {bootstrap_replicates: null, ci: "Wilson 95%（wilson_ci）", resampling_unit: null, ci_level: 0.95}
  risk_definition: null        # 无 risk 指标（counting only）
run:
  command: "E:\\conda\\envs\\deepminer\\python.exe -u scripts/natural_omission_full.py（全部默认参数）"
  device: cpu
  runtime_s: 15.219660000002477
artifacts:
  summary: results/natural_omission_full/summary.json
  csv: [results/natural_omission_full/by_split.csv, results/natural_omission_full/threshold_sensitivity.csv]
notes: >
  全量 natural omission counting（不使用任何模型预测；不构造 candidate sets）。登记目的：为 RQ4 /
  A3.3 的“natural omission 稀缺 → 统计功效”评估提供全数据底数。观测：expression 级 miss rate 为
  1.03%~2.68%（testB 最高 131/4889=2.68%），与 audit 子集（N64 = 1.41%）同数量级；A3.3 的 RQ4 功效
  风险结论不变（omission 属稀有事件；Phase 2 前需功效评估）。max-IoU 分布整体高（median 0.82），
  低尾即 miss 来源。不产生 Gate 结论。
```

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p0a1-corrected-metrics-20260928-01
git_commit: "ddf612d40557b3739de4447685accdf1b8bb2301"   # 运行时的 HEAD
dirty: true                    # 本条目与修正代码随收尾 commit 入库
timestamp: "2026-09-28"
dataset: refcoco+              # 与 p0-cosine-kcardinality-20260927-01 完全相同的 common cohort 与 nested manifests
split: [val_select, val_calib, testA, testB]
candidate_protocol: "bank-v1 + manifests-v1（frozen，禁止重采样；C_K = target + distractor_order[:K-1]）"
K: [5, 10, 20, 50]
hardness: random only
target_presence: target-present only（primary common cohort）
backbone: "OpenCLIP ViT-B/32 laion2b_s34b_b79k（frozen；checkpoint sha256 1bd3c717…d080ad，native logit scale 100.0）"
model: "B1 frozen cosine（概率变体：native / global_T_corrected / oracle_T_K；不训练任何模型）"
seed: 0                        # bootstrap seed
training_config: null
calibration_config:
  method: "log-T 空间有界优化（scipy.optimize.minimize_scalar bounded；objective = candidate-set mean NLL；禁用粗 grid）"
  legal_fit_set: "val_calib ∩ K∈{5,10}（common cohort）；n_sets=10462；testA/testB/K20/K50 未参与拟合"
  corrected_T: 0.0335596
  interior: true               # 距下界 1.53 decade / 上界 2.47 decade；未触发范围扩张与 warning
  bounds: [0.001, 10.0]
  nll: {before: 1.9151, after: 1.4386}
  oracle_T_K: {K5: 0.034317, K10: 0.033007, K20: 0.032226, K50: 0.031264}   # ORACLE / DIAGNOSTIC，非 OOD 方法
  oracle_note: "修正后 per-K 温度不再完全相同（≈10% 单调漂移）——上一轮 all T=0.05 确认为 optimization-boundary artifact"
metrics:
  ranking:                     # 与 Phase 0A 一致（同一 cohort）
    top1_acc: {K5: 0.5348814846867638, K10: 0.3899706716669071, K20: 0.28568681186595507, K50: 0.18789364873311218}
  calibration_global_T_corrected:   # __pooled__ common（n=20799）
    ece_adaptive: {K5: 0.02243906459780723, K10: 0.0299108819812615, K20: 0.03985989872743931, K50: 0.05225356464219643}
    conf_acc_gap: {K5: -0.010275198033414368, K10: -0.0233104335052548, K20: -0.03970534389822364, K50: -0.05113461495631147}
    brier_binary: {K5: 0.20474066553857195, K10: 0.19918389863424776, K20: 0.17562316495745928, K50: 0.13785741818079983}
    nll_binary: {K5: 0.5936072070844051, K10: 0.5839732797945749, K20: 0.529331230222234, K50: 0.43797206691266094}
  selective_normalized:        # base-rate-aware（global_T_corrected，__pooled__）
    e_aurc: {K5: 0.14039452617460957, K10: 0.17542511685818335, K20: 0.18492676907767885, K50: 0.1773866757134711}
    e_aurc_relative_worsening_vs_K5:
      5_10: {value: 0.24951535959461846, ci: [0.20870852121999842, 0.29019455365359176]}
      5_20: {value: 0.3171935837988744, ci: [0.2617565076539002, 0.3720301066636913]}
      5_50: {value: 0.26348712123472795, ci: [0.1936094805374815, 0.330095739766148]}
    auroc_correct: {K5: 0.7413411832480435, K10: 0.7328162733559876, K20: 0.7329312870208611, K50: 0.7366082165576419}
    auroc_delta_vs_K5:
      5_10: {diff: -0.008524909892055899, ci: [-0.01586158723286626, -0.0009799430876813319]}
      5_20: {diff: -0.008409896227182467, ci: [-0.017867081900052355, 0.0013328995496568016]}
      5_50: {diff: -0.004732966690401685, ci: [-0.016414230580263194, 0.007563546800773381]}
    rer_at_50: {K5: 0.38353181008571746, K10: 0.26028146674022684, K20: 0.1876274004483771, K50: 0.12217825378103042}
    rer_at_50_delta_pp:
      5_10: {diff_pp: -12.325034334549063, ci: [-13.77902958349951, -10.791956668866261]}
      5_20: {diff_pp: -19.590440963734036, ci: [-21.223931633579493, -17.851037669804506]}
      5_50: {diff_pp: -26.135355630468705, ci: [-27.88963694255214, -24.196691724532232]}
    rer_at_80: {K5: 0.13483589041204816, K10: 0.09298575861352931, K20: 0.06319709030578699, K50: 0.04221347608307561}
  reliability_map_global_T_fixed_bins:   # __pooled__；同 nominal confidence 跨 K 比较
    K5_0.4_0.5: {conf: 0.4479231658017683, acc: 0.44261910349683176, n: 4261}
    K50_0.4_0.5: {conf: 0.44070170280600546, acc: 0.5555555555555556, n: 351}
    direction: "修正后 K=50 在同 nominal confidence 上经验准确率高于 K=5（保守/欠自信）；与 native-scale 下的下移方向相反"
    low_support: "K=50 的 ≥0.6 桶 n<100，已标记 low-support"
  counts: {n_common_pooled: 20799, n_unique_images: 2981}
  uncertainty: {bootstrap_replicates: 5000, resampling_unit: image, ci_level: 0.95, seed: 0}
  risk_definition: "risk = 1 - accuracy"
  amendment_gate:              # Amendment A5 §A5.4（post-hoc criterion；不替换 Gate Q1）
    route_A_selective: true    # E-AURC rel worsen ≥20% 且 CI 不跨 0（K20 +31.7% / K50 +26.3%）且 RER@50 下降 ≥10pp（-19.6 / -26.1pp）
    route_B_correctness: false # ΔAUROC 未达 0.03 且 CI 跨 0；reliability shift 未达稳定阈值
hard_checks:
  note: "继承 Phase 0A 的同一 cohort 与同一批 raw scores；score invariance / rank monotonic / accuracy monotonic 仍全部通过"
run:
  command: "E:\conda\envs\deepminer\python.exe -u scripts/run_phase0a_corrected.py --features cache/features --manifests cache/manifests --refs \"data/raw/refcoco+/refcoco+/refs(unc).p\" --out results/phase0a_corrected --bootstrap-replicates 5000（日志 logs_p0a1.txt）"
  device: cpu
  step_seconds: {load_inputs: 2.2, score_sets: 256.8, fit_temperatures: 3.16, metric_tables: 1.69, reliability: 0.72, bootstrap: 1373.4, predictions_npz: 0.41, write_artifacts: 0.09}
artifacts:
  root: results/phase0a_corrected/
  files: [temperature_fit.json, calibration_metrics_corrected.csv, normalized_selective_metrics.csv, correctness_auroc.csv, reliability_bins_globalT.csv, reliability_bins_all_variants.csv, eaurc_bootstrap.csv, rer_bootstrap.csv, auroc_bootstrap.csv, per_sentence_predictions.npz, metadata.json]
notes: >
  Post-Phase-0A 度量/温度有效性修正（Amendment A5）。目的：在控制 base error rate 与修复温度边界伪影之后，
  重新评估 cosine 的 reliability 证据。关键观测：(1) corrected global T*=0.0336 为 interior 最优（旧 T*=0.05 确为
  边界伪影）；oracle per-K 温度不再完全相同（≈10% 漂移），但量级小；(2) corrected global-T 下跨 K ECE drift
  仅 +3.0pp（K50 vs K5），但绝对 ECE 仍 2.2%~5.2%——不得写 “calibration solved”；(3) base-rate 校正后的
  selective 指标仍显著恶化：E-AURC 相对恶化 +25%~+32%（CI 不跨 0），RER@50 下降 12.3/19.6/26.1pp；
  (4) 整体判别力 AUROC_correct 基本稳定（Δ≤0.009，K20/K50 CI 跨 0）；(5) corrected global-T 的 reliability map
  漂移方向为保守（K=50 同置信下更准）——旧 native 下移主要由 native scale 过自信放大。结论：Route A 触发、
  Route B 未触发 → “confidence 作为 abstention/selection 信号的质量随 K 下降”成立，但“整体判别力崩塌”不成立。
  解释文档见 docs/phase0a_interpretation.md；本条目不产生 Gate Q1/Q2 判定。
```

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p0b-b3-independent-20260928-01
git_commit: "ddf612d40557b3739de4447685accdf1b8bb2301"   # 运行时的 HEAD
dirty: true                    # 本条目与评测代码随收尾 commit 入库
timestamp: "2026-09-28"
dataset: refcoco+              # 与 p0-cosine-kcardinality-20260927-01 / p0a1 完全相同的 common cohort（n=20799）与 nested manifests
split: [val_select, val_calib, testA, testB]
candidate_protocol: "bank-v1 + manifests-v1（frozen；与 cosine 完全相同的 nested candidate sets，未重采样；paired comparison 合法）"
K: [5, 10, 20, 50]
hardness: random only
target_presence: target-present only（primary common cohort）
backbone: "OpenCLIP ViT-B/32 laion2b_s34b_b79k（frozen cached features z_q, z_i；不重抽取）"
model: "B3 Independent MLP（candidate-blind）——输入 [z_q, z_i, z_q⊙z_i, cos(z_q,z_i), g_i(5), o_i(1)] = 1543-d；网络 1543→128→128→1（ReLU）；214,273 params（<300k budget ✓，<1M ✓）；raw score 不读取 K / 其他候选 / 分数分布 / entropy / margin"
seed: [1, 2, 3]                # model seeds；candidate manifests 不随 seed 变化
training_config:
  train_data: "RefCOCO+ train，random regime，K∈{5,10}（50:50 采样），target-present；candidate prefix 来自 frozen manifest"
  loss: "listwise cross-entropy（per-set softmax over candidates）；无 focal / pairwise / triplet / contrastive / calibration loss"
  optim: "AdamW；lr 由 grid {1e-4, 3e-4, 1e-3}（val_select K5/K10 mean NLL）选 1e-4；wd=1e-4；batch=64；max 15 epochs；patience 3"
  per_seed: {epochs_run: [12, 13, 11], best_epoch: [9, 10, 8], best_val_select_nll: [0.74284, 0.74851, 0.74600]}
  isolation: "hyperparameter / early stopping 只用 val_select K5/K10；val_calib/testA/testB/K20/K50 从未参与训练与选型"
calibration_config:
  method: "Phase 0A.1 同一实现（u=lnT bounded 最小化 mean NLL）"
  legal_fit_set: "val_calib ∩ K∈{5,10}（common cohort）"
  corrected_T_per_seed: [1.115344, 1.116630, 1.100525]     # 全部 interior=True，无 warning
  oracle_T_K_per_seed:                  # ORACLE / DIAGNOSTIC — NOT A VALID OOD METHOD
    seed1: {K5: 1.1431, K10: 1.0978, K20: 1.0734, K50: 1.0191}
    seed2: {K5: 1.1414, K10: 1.1013, K20: 1.0756, K50: 1.0293}
    seed3: {K5: 1.1287, K10: 1.0829, K20: 1.0600, K50: 1.0151}
hard_checks:                   # 每 seed 全部通过（无 VALIDATION_FAILURE，exit 0）
  score_invariance: "max_abs_diff = 0.0（atol=0；共 1,148,625 个共享候选对，0 violations）"
  rank_monotonic: "0 violations（5-10 / 5-20 / 5-50 / 10-20 / 10-50 / 20-50）"
  accuracy_monotonic: "0 violations"
metrics:                       # pooled common cohort n=20799；global_T_corrected；跨 3 seed 报 mean±std
  ranking_top1: {K5: 0.7907±0.0020, K10: 0.6763±0.0029, K20: 0.5643±0.0022, K50: 0.4320±0.0027}
  calibration_ece_adaptive: {K5: 0.0098±0.0010, K10: 0.0163±0.0018, K20: 0.0356±0.0011, K50: 0.0673±0.0015}
  selective:
    e_aurc: {K5: 0.0378±0.0003, K10: 0.0705±0.0003, K20: 0.0961±0.0001, K50: 0.1226±0.0009}
    auroc_correct: {K5: 0.8426±0.0006, K10: 0.8102±0.0008, K20: 0.7995±0.0001, K50: 0.7904±0.0009}
    rer_at_50: {K5: 0.8212±0.0039, K10: 0.6484±0.0001, K20: 0.5093±0.0050, K50: 0.3653±0.0033}
  bootstrap_worsening_vs_K5:    # image-cluster paired bootstrap 5000 reps；每 seed；正值 = 随 K 退化
    e_aurc_relative: {K10: "+85.6%~+86.8%", K20: "+151.8%~+155.3%", K50: "+222.5%~+225.8%"}   # 每 seed CI 均不跨 0
    auroc_correct_drop: {K10: "0.0321~0.0326", K20: "0.0425~0.0438", K50: "0.0519~0.0524"}      # CI 均不跨 0
    rer_at_50_drop_pp: {K10: "17.0~17.7", K20: "30.6~32.2", K50: "45.1~46.1"}
    rer_at_80_drop_pp: {K10: "13.6~13.9", K20: "20.7~21.1", K50: "27.1~27.7"}
  reliability_map_global_T:      # equal_width 15 bins（protocol RELIABILITY_BINS=15）；pooled
    direction: "K=5 全桶近对角（|gap|≤0.03）；K=50 在同 nominal confidence 上经验准确率系统性高于 nominal（保守/欠自信），中高桶 gap +0.06~+0.17（n≥500），三 seed 一致"
  amendment_gate:                # Amendment A5 §A5.4（post-hoc replication criterion；cosine 与 B3 同方向）
    route_A_selective: true      # E-AURC rel worsen ≥20%（+152%~+226%，CI 不跨 0）且 RER@50/80 下降 ≥10pp（RER@50 −30.6~−46.1pp）
    route_B_correctness: true    # ΔAUROC_correct ≥0.03（−0.0425~−0.0524，CI 不跨 0）且 corrected global-T reliability map 稳定 shift（保守方向，三 seed 一致）
  paired_vs_cosine:              # B3(mean±std, 3 seeds) vs cosine(global_T_corrected)；pooled n=62397；paired image-cluster bootstrap
    K5:  {accuracy: "+0.2559 [0.2456,0.2655]", ece_adaptive: "-0.0126 [-0.0214,-0.0074]", e_aurc: "-0.1026 [-0.1087,-0.0967]", auroc_correct: "+0.1013 [0.0920,0.1107]", rer_at_50: "+0.4377", rer_at_80: "+0.2608"}
    K20: {accuracy: "+0.2786", ece_adaptive: "-0.0043 [-0.0144,+0.0034]", e_aurc: "-0.0889", auroc_correct: "+0.0666", rer_at_50: "+0.3216", rer_at_80: "+0.1231"}
    K50: {accuracy: "+0.2441", ece_adaptive: "+0.0151 [+0.0067,+0.0230]", e_aurc: "-0.0548", auroc_correct: "+0.0538", rer_at_50: "+0.2431", rer_at_80: "+0.0797"}
    note: "B3 全面优于 raw cosine —— 按 §25 这本身只是 'learned matching > raw cosine'，不是 candidate-aware evidence"
run:
  command: "E:\conda\envs\deepminer\python.exe -u scripts/run_phase0b.py --features cache/features --manifests cache/manifests --bank cache/proposals.h5 --refs \"data/raw/refcoco+/refcoco+/refs(unc).p\" --seeds 1 2 3 --out results/phase0b_independent --bootstrap-replicates 5000 --device cuda --resume（日志 logs_phase0b_eval.txt）"
  device: "cuda（scoring）/ cpu（bootstrap）"
  step_seconds: "首跑 seeds 段 ~135 min（每 seed 270 次配对 bootstrap）；resume 补跑 paired+aggregate+metadata 884.5s"
  incident: "首跑 paired 阶段 KeyError 'sentence_ids'（cosine 存档用单数 sentence_id）→ reader 修复为接受两种拼写 + 参数化回归测试 + 真实数据冒烟后 resume 补跑（seeds 未重算）"
artifacts:
  root: results/phase0b_independent/
  files: [metadata.json, aggregate.csv, paired_vs_cosine.csv, grid_search.json, seed_{1,2,3}/model.npz, seed_{1,2,3}/training.json, seed_{1,2,3}/raw_scores/K{5,10,20,50}.npz, seed_{1,2,3}/per_sentence_predictions.npz, seed_{1,2,3}/eval_metadata.json, seed_{1,2,3}/ranking_metrics.csv, seed_{1,2,3}/calibration_metrics.csv, seed_{1,2,3}/normalized_selective_metrics.csv, seed_{1,2,3}/reliability_bins.csv, seed_{1,2,3}/diagnostics.csv, seed_{1,2,3}/bootstrap.csv]
notes: >
  B3 Independent（candidate-blind）MLP 复现实验（指令 §13–§29）。目的：验证 candidate-cardinality reliability
  degradation 是否在另一个 independent scorer 上复现。关键观测：(1) 三个 seed 硬检查（score invariance max_abs_diff=0.0 /
  rank monotonic / accuracy monotonic）全部通过，无 VALIDATION_FAILURE；(2) 与 cosine 同方向且更强的 reliability
  degradation：E-AURC 相对恶化 K20 +151.8%~+155.3%、K50 +222.5%~+225.8%（CI 不跨 0），AUROC_correct 下降
  0.0425~0.0524（≥0.03，CI 不跨 0 → Route B 触发），RER@50 下降 30.6~46.1pp；(3) 3 seed 方差极小（E-AURC std ≤0.001），
  跨 seed 高度一致；(4) corrected global-T reliability map 漂移为保守方向（K=50 同 nominal confidence 上更准），
  与修正后 cosine 的漂移方向一致（注：cosine 修正审计用 10 等宽桶，B3 用 protocol 的 15 等宽桶；两者方向一致）；
  (5) B3 整体远超 cosine（accuracy +24~28pp、E-AURC −0.05~−0.10），按 §25 不构成 candidate-aware evidence；
  (6) Amendment A5 §A5.4 判定：cosine Route A ✓ / Route B ✗；B3 Route A ✓ / Route B ✓ → 两 scorer 同方向复现
  → 指令 §26 Case A：GO（下一问题：K + score statistics 是否已足够——留在后续阶段决定；本轮到此停止，不自动实现
  stats calibrator）。不产生 Gate Q1/Q2/Q3 判定。
```

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p05-score-sufficiency-20260928-01
git_commit: "2adfc09b9c957535c6312d6db35b58da929912fa"   # 全量运行时的 HEAD；本条目与代码随收尾 commit 入库
dirty: true
timestamp: "2026-09-28"
dataset: refcoco+              # 与 p0/p0a1/p0b 完全相同的 common cohort（n=20799）；raw scores 直接复用，grounding scorer 不重训
split: [val_select, val_calib, testA, testB]
candidate_protocol: "bank-v1 + manifests-v1（frozen；完全相同的 nested candidate sets，未重采样）"
K: [5, 10, 20, 50]
hardness: random only
target_presence: target-present only（primary common cohort）
backbone: "OpenCLIP ViT-B/32 laion2b_s34b_b79k（frozen cached features；本轮不重新前向）"
model: >
  Phase 0.5 score-information sufficiency zoo（协议 Amendment A6，结果可见前冻结；只读候选 score 集合，
  绝不读 candidate/query embeddings、geometry、objectness、GT）：L0 标量（msp/top1_score/margin/-entropy/
  normalized-entropy，无训练）；L1 handcrafted stats（≤17-d）× {LogisticRegression(L2, C∈{0.1,1,10}),
  TinyMLP 32→16→1(lr∈{1e-4,3e-4,1e-3})} × {without-K, logK}（+ without-entropy 与 top-scores 变体）；
  L2 ScoreDeepSets（φ:1→16→16, masked mean/max/h_top1 + z_top1, head→32→1, ±logK；1,969 / 1,937 params）
seed: [1, 2, 3]                # B3 scorer seeds，各自独立跑完整 zoo（不合并、不充样本量）；cosine 为 secondary
training_config:
  reliability_data: "val_calib 的 image-level 70/30 切分（seed=20260928；split_manifest.json 先写后跑；523/224 images）：reliability_train 训练、reliability_tune 选型；仅用 K∈{5,10} 行；testA/testB/K20/K50 从不进入训练、早停或选型"
  normalization: "跨行 μ/σ（stats / score / logK / z_top1）只用 reliability_train ∩ K∈{5,10} 估计，全 K 共用；禁止 per-K normalization"
  selection_metric: "reliability_tune 的 mean AUROC_correct(K5,K10)（唯一用途，A6.4）"
  early_stopping: "MLP/SDS：val = reliability_tune，patience 30，max 300 epochs，wd=1e-4（只触 tune，不触 test）"
calibration_config:
  fit_split: val_calib        # 不重新拟合温度：直接复用 p0a1/p0b 冻结的 corrected global T（B3: 1.115344/1.116630/1.100525；cosine: 0.0335596）
  method: "MSP 等标量 = softmax(scores / T*)（复用既有实现）；L0 原始分数标量超出 [0,1] 时仅做单调平均秩变换 (rank−0.5)/n 以满足 ccg.metrics 的 [0,1] 契约（所有报告指标 rank-based，实测逐位不变；raw 值存 features npz scalars_raw_K*）"
metrics:                       # pooled test (testA+testB)，n=62397；B3 报 3-seed mean±std；degradation 正値=恶化
  selection_b3_seed_mean: {msp: 0.8201, stats_logistic_noent: 0.8197, stats_logistic: 0.8195, stats_logistic_noK: 0.8195, margin: 0.8192, stats_mlp: 0.8190, stats_mlp_noK: 0.8176, neg_entropy: 0.7991, norm_entropy: 0.7991, top_scores_logistic: 0.7950, score_deepsets_noK: 0.7615, score_deepsets: 0.6134, top1_score: 0.4758}
  best_score_only_model: "msp（L0 标量；tune 0.8201）；best summary family = stats_logistic_noent（0.8197）；per-scorer best: seed1=margin / seed2=msp / seed3=msp / cosine=stats_mlp_noK"
  pooled_auroc_correct:        # K5 / K20 / K50
    msp:                {K5: "0.8407±0.0025", K20: "0.7995±0.0007", K50: "0.7890±0.0031"}
    margin:             {K5: "0.8333±0.0021", K20: "0.7821±0.0030", K50: "0.7718±0.0066"}
    stats_logistic:     {K5: "0.8408±0.0025", K20: "0.8015±0.0014", K50: "0.7915±0.0046"}
    stats_mlp:          {K5: "0.8395±0.0026", K20: "0.7985±0.0016", K50: "0.7834±0.0048"}
    score_deepsets:     {K5: "0.7380±0.0109", K20: "0.5518±0.0897", K50: "0.5529±0.0916"}   # logK 版 OOD 崩溃 + seed std ~0.09
    score_deepsets_noK: {K5: "0.8035±0.0040", K20: "0.7432±0.0142", K50: "0.7252±0.0164"}
  pooled_selective:            # E-AURC / RER@50（K5 → K20 → K50）
    msp:            {e_aurc: "0.0381→0.0969→0.1241", rer_at_50: "0.8248→0.5058→0.3588"}
    stats_logistic: {e_aurc: "0.0381→0.0958→0.1216", rer_at_50: "0.8257→0.5146→0.3611"}
    stats_mlp:      {e_aurc: "0.0386→0.0973→0.1270", rer_at_50: "0.8152→0.5065→0.3480"}
    score_deepsets: {e_aurc: "0.0879→0.2635→0.3047", rer_at_50: "0.5477→0.1632→0.1301"}
  bootstrap_degradation_pooled_K50:   # image-cluster paired bootstrap 5000 reps（seed=0, ci=0.95；K5 vs K50；b3_mean）
    msp: "dAUROC +0.0517 [+0.0380,+0.0656]；w(E-AURC) +2.257 [+1.930,+2.635]；RER50drop +46.60pp [43.03,49.92]；RER80drop +26.54pp"
    margin: "+0.0615 [+0.0479,+0.0758]；+2.279 [+1.965,+2.653]；+47.76pp [44.15,51.13]；+25.68pp"
    stats_logistic: "+0.0494 [+0.0359,+0.0632]；+2.188 [+1.869,+2.561]；+46.46pp；+26.36pp"
    stats_mlp: "+0.0561 [+0.0423,+0.0703]；+2.293 [+1.964,+2.679]；+46.72pp；+26.75pp"
    stats_logistic_noent: "+0.0461 [+0.0308,+0.0585]；+2.093 [+1.786,+2.457]；+45.86pp；+26.42pp"
    score_deepsets: "+0.1851 [+0.1685,+0.2012]；+2.463 [+2.290,+3.124]；+41.76pp（该度量的百分位 CI 整体高于点估计 ~9pp，见 notes）；+26.43pp"
  model_vs_model_K50:          # pooled；per B3 seed 1/2/3
    msp_vs_stats_logistic: {auroc: ["−0.0040", "−0.0007", "−0.0026"], note: "2/3 seed CI 完全 <0；K50 上 stats_logistic ≥ msp"}
    stats_logistic_vs_stats_mlp: {auroc: ["+0.0068", "+0.0077", "+0.0096"], note: "CI 全 >0；小容量 logistic 的外推稳定性优于 MLP"}
    best_summary_vs_score_deepsets: {auroc: ["+0.2983", "+0.2861", "+0.1253"], note: "CI 全 >0；summary ≫ full score set"}
  ablations:
    withoutK_vs_logK: "tune 0.8195=0.8195；K50 AUROC 0.7920 vs 0.7915 → cardinality 本身不携额外可靠性信息"
    without_entropy: "K50 AUROC 0.7948 vs 0.7915 → entropy 无正贡献（略负）"
    summary_vs_full_set: "ScoreDeepSets ≪ stats（K50 AUROC 0.553 vs 0.792；配对 CI 全 >0）→ full unordered score set 不提供超出手工统计的可用信息"
    logK_instability: "SDS+logK 在 K20/50 AUROC≈0.55、seed std 0.09（不稳定 extrapolation，§22 报告项）；SDS−logK 稳定但显著弱"
  within_K_vs_cross_K: "msp：同 K 跨 split（val_select→testA/testB）K5 0.8439→0.8594→0.8138（漂移 ≤±4.6pp）；同 split 跨 K（K5→K50）−5.2~−5.8pp；E-AURC/RER 的 cardinality 效应远强于 split 漂移"
  ece_adaptive_pooled: {msp: "0.0104→0.0697", stats_logistic: "0.0147→0.0358", stats_mlp: "0.0116→0.0909", score_deepsets: "0.0403→0.4761"}   # K5→K50
  cosine_secondary:             # cosine best family = stats_mlp_noK；AUROC 基本稳定（K5→K50 dAUROC +0.0044 [−0.0122,+0.0221]），E-AURC/RER 恶化
    e_aurc_rationale: "w(E-AURC) K50 +0.232；RER50drop +25.93pp → Route A 满足（4 cells，CI 不跨 0）"
  sufficiency_gate:             # Amendment A6.6（§24/25/26）；headline = best score-only model（msp）
    verdict: GO_candidate_embeddings
    route_A_cells: [testA/K20, testA/K50, testB/K20, testB/K50]
    route_B_cells: [testA/K20, testA/K50, testB/K20, testB/K50]
    per_seed_consistent: "3/3 seeds 均为 GO（各 4 cells）；seed-mean 与 per-seed 一致"
    override: false            # K50 pooled：AUROC 0.789 < 0.85；E-AURC 0.1241 > 0.03；RER@50 0.3588 < 0.70
    no_go_sufficient: false
    cosine_verdict: "GO_candidate_embeddings（Route A ×4，Route B ×0）——与 0A/0B 的 E-AURC/RER 恶化但 AUROC 稳定一致"
run:
  command: "E:\conda\envs\deepminer\python.exe _launch_detached.py logs_phase05.txt scripts/run_phase05.py --out results/phase05_score_sufficiency --bootstrap-replicates 5000 --log-file logs_phase05_driver.txt（detached；pid 33356）"
  device: cpu
  step_seconds: "total 4840.65s（~80.7 min）：load 0.2 / split 0.0 / features 4.1 / train 541.6（4 scorers × 8 families ≤300 epochs）/ metrics 2.8 / bootstrap 4280.4（5000 reps；660 cross-K + 96 pairwise rows）/ figure 0.7 / write 10.8"
  incident: >
    运行前冒烟阶段修复：(a) L0 原始分数标量做的单调 rank 变换（契约满足用，指标不变）；(b) features.py neg_entropy
    符号修复（原实现返回 +H(P)，与 §7「−H(P)，higher=more confident」相反；修复前该基线 AUROC≈0.14，修复后 tune 0.7991）；
    (c) gate RER@50 单位修复（fraction → ×100 转百分点后进 cell；evaluate.py 契约为 pp，修复前 Route A 恒空）；
    (d) partial run 的 gate 兼容缺 seed；(e) paired_bootstrap.csv schema（剔除 pairwise 额外 K 列）。(c) 之后 sufficiency_gate.json
    用 scripts/rebuild_phase05_gate.py 从冻结产物重建（重建校验：390 行 degradation 与已发布 CSV 在 1e-9 内一致；verdict 修复前后不变）。
    全量运行一次成功，无中断。
artifacts:
  root: results/phase05_score_sufficiency/
  files: [protocol.json, split_manifest.json, features/{b3_seed1,b3_seed2,b3_seed3,cosine}.npz + *_normalisation.json, scalar_baselines.csv, reliability_metrics.csv, aggregate.csv, cross_k_degradation.csv, paired_bootstrap.csv, sufficiency_gate.json, metadata.json, {stats_logistic,stats_logistic_noK,stats_logistic_noent,top_scores_logistic,stats_mlp,stats_mlp_noK,score_deepsets,score_deepsets_noK}/{selection.json,metrics.csv}, figures/score_only_reliability_vs_K.png, predictions/{scorer}/{msp,margin,norm_entropy,stats_logistic,stats_mlp,score_deepsets}.csv.gz]
  raw_predictions_columns: [ref_id, image_id, K, eval_split, grounding_correct, reliability_score, model, scorer_seed]
notes: >
  Phase 0.5 score-information sufficiency 审计（指令全文 + Amendment A6）。目的：在不读取 candidate embeddings 的
  前提下，测试「当前候选集合的 score distribution」是否已足以预测 grounding 正确性并消除 candidate-cardinality
  reliability degradation。关键观测：(1) 最佳 score-only 模型是 L0 标量 msp（tune 0.8201）——比所有 L1/L2 学习模型都好，
  logK/noK/entropy 消融均无正增益（cardinality 本身不携带额外可靠性信息）；(2) 该最佳模型在 K20/K50 OOD cells 仍显著退化：
  pooled K50 dAUROC +0.0517（CI 不跨 0）、E-AURC w +226%（CI +193%~+264%）、RER@50 降 46.6pp（CI 43.0~49.9）——四个 OOD
  cells（testA/testB × K20/K50）双 Route（A+B）同时命中，3/3 seeds 一致 → A6.6 判定 GO_candidate_embeddings（score-only 信息
  不足，candidate embedding 信息仍有潜在价值；且 §26 override 不成立）；(3) 回答题 1：candidate-count 与 score-distribution statistics
  不能解释/消除该退化——加入 K/logK、entropy、全部手工统计、乃至完整 score set 均不消除，退化幅度与 0B 原始 msp 基线同量级；
  (4) 回答题 2：完整无序 score set 不提供超出手工统计的信息——ScoreDeepSets ≪ stats logistic（K50 AUROC 0.553 vs 0.792；
  BestSummary vs SDS 配对 CI 全 >0 且 +0.125~+0.298），且 SDS+logK 出现 §22 要求报告的不稳定 extrapolation（seed std 0.09）；
  (5) cosine secondary 复现同向：E-AURC/RER 明显恶化（Route A ×4）而 AUROC 基本稳定（Route B 不触发）；(6) 附注：ScoreDeepSets 行
  的 RER@50 百分位 CI 相对点估计整体上移 ~9pp（近随机排序下 rank 度量的 cluster-resampling 有限样本偏置，独立复现确认；
  不影响 gate 判定——gate 的 msp/stats 行 CI 行为正常）。本阶段不实现 candidate-embedding DeepSets（§32 禁令），到此停止。
```
