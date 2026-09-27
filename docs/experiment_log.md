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
