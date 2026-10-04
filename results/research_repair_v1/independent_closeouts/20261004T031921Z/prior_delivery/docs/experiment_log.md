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

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p1-semantic-sufficiency-20260928-01
git_commit: "62c1eff037d717ce324c97846fb6616c48569029"   # 全量运行时的 HEAD；本条目与代码随收尾 commit 入库
dirty: true                    # src/ccg/semantic/*、scripts/run_phase1.py、4 个 semantic 测试文件、协议 A7、
                               # results/phase1_semantic_sufficiency/ 运行时尚未提交；随本条目同批收尾 commit 入库
timestamp: "2026-09-28"
dataset: refcoco+              # 与 p0/p0a1/p0b/p05 完全相同的 common cohort；B3 raw scores 直接复用，grounding scorer 不重训
split: [val_select, val_calib, testA, testB]   # 复用 A6.2 切分（reliability_train 523 / reliability_tune 224 images）；未重划
candidate_protocol: "bank-v1 + manifests-v1（frozen；与 0A/0B/0.5 完全相同的 nested candidate sets，未重采样；paired comparison 合法）"
K: [5, 10, 20, 50]             # 训练/选型仅 K∈{5,10}；OOD = K∈{20,50}
hardness: random only
target_presence: target-present only（primary common cohort）
backbone: "OpenCLIP ViT-B/32 laion2b_s34b_b79k（frozen cached z_q/z_i，L2-normalized 512-d；本轮不重新前向）"
model: >
  Phase 1 candidate semantic information sufficiency zoo（协议 Amendment A7；所有 gate 数字与模型定义在结果可见前冻结）：
  E0 参考 = MSP + A6 Stats Logistic（Phase 0.5 逐行预测直接复用 + 本地重算断言，max|Δ|=0.0 / ≤2.8e-16）；
  E1a = 16-d 语义手工统计 logistic（17 params）；E1b = A6 17-d score stats ⊕ 16-d 语义（34 params）；
  E2 TopCompetitor（P_q/P_v 共享 512→64；交互块 448-d；E2-score 769 / E2-semantic 123,265 / E2-combined 123,777 params）；
  E3 SemanticDeepSets（h_i 132-d；mean/max/h_t；E3-full 与 E3-top5 149,121 / E3+logK 149,249 params）。
  约束：只读冻结 B3 分数与 embedding；绝不改分数 / rerank / 改变 top-1 / 参与 grounding 训练。
seed: [1, 2, 3]                # B3 scorer seeds，各自独立跑完整 zoo（不合并、不充样本量）
training_config:
  isolation: "训练/早停只读 reliability_train∩K∈{5,10}；选型只读 reliability_tune∩K∈{5,10}；testA/testB/K20/K50 从不进入任何阶段（driver 级断言）"
  normalization: "E1 标准化 μ/σ 与 E3 的 score 标准化只用 reliability_train∩K∈{5,10} 估计（同 A6.3）"
  selection_metric: "reliability_tune 的 mean AUROC_correct(K5,K10)；|Δ|<0.002 时取更简单（更小 C/lr）（A7.4）"
  e2_e3_optim: "BCE；AdamW；wd=1e-4；patience 30；≤300 epochs；lr∈{1e-4,3e-4,1e-3} 各自选定"
calibration_config:
  note: "不重新拟合温度：corrected softmax 复用 Phase 0A.1/0B 冻结的 per-seed corrected global T（1.115344 / 1.116630 / 1.100525）"
metrics:                       # pooled test = testA+testB canonical 行集（n=10,286 / 1,490 images；跨 K 共享分母）
  selection_tune_mean:         # A7.4 选型值 3-seed 均值；gate 比较 = best score-only（msp 0.8201）vs best semantic（e1b 0.8202）
    msp: 0.8201
    stats_logistic: 0.8195
    e1_semantic_stats: 0.6890
    e1b_stats_semantic: 0.8202
    e2_score: 0.8216           # 原始 per-seed tune 最高族；A7.6 预运行澄清将其排除出 best-semantic 候选（不消费任何 embedding）
    e2_semantic: 0.6287
    e2_combined: 0.8105
    e3_full: 0.7824
    e3_top5: 0.7766
    e3_logk: 0.7789
  pooled_auroc_correct:        # K5 / K10 / K20 / K50；3-seed mean±std
    msp:                {K5: "0.8407±0.0020", K10: "0.8103±0.0024", K20: "0.7995±0.0005", K50: "0.7890±0.0025"}
    stats_logistic:     {K5: "0.8408±0.0021", K10: "0.8105±0.0024", K20: "0.8015±0.0012", K50: "0.7915±0.0038"}
    e1_semantic_stats:  {K5: "0.7152±0.0026", K10: "0.7116±0.0027", K20: "0.7202±0.0017", K50: "0.7143±0.0023"}
    e1b_stats_semantic: {K5: "0.8423±0.0024", K10: "0.8171±0.0027", K20: "0.8138±0.0016", K50: "0.8069±0.0051"}
    e2_score:           {K5: "0.8407±0.0021", K10: "0.8104±0.0022", K20: "0.8013±0.0009", K50: "0.7930±0.0036"}
    e2_semantic:        {K5: "0.6575±0.0010", K10: "0.6501±0.0022", K20: "0.6559±0.0020", K50: "0.6611±0.0011"}
    e2_combined:        {K5: "0.8267±0.0018", K10: "0.7984±0.0026", K20: "0.7946±0.0006", K50: "0.7878±0.0021"}
    e3_full:            {K5: "0.7979±0.0066", K10: "0.7750±0.0049", K20: "0.7671±0.0031", K50: "0.7508±0.0013"}
    e3_top5:            {K5: "0.7899±0.0043", K10: "0.7681±0.0039", K20: "0.7619±0.0046", K50: "0.7450±0.0052"}
    e3_logk:            {K5: "0.7941±0.0072", K10: "0.7721±0.0053", K20: "0.7636±0.0054", K50: "0.7480±0.0038"}
  pooled_selective:            # E-AURC / RER@50 / RER@80（K5→K10→K20→K50）
    msp:                {e_aurc: "0.0381→0.0710→0.0969→0.1241", rer_at_50: "0.8248→0.6515→0.5058→0.3588", rer_at_80: "0.3859→0.2593→0.1870→0.1206"}
    stats_logistic:     {e_aurc: "0.0381→0.0711→0.0958→0.1216", rer_at_50: "0.8257→0.6492→0.5146→0.3611"}
    e1b_stats_semantic: {e_aurc: "0.0381→0.0687→0.0903→0.1140", rer_at_50: "0.8149→0.6549→0.5304→0.3836", rer_at_80: "0.3898→0.2713→0.2011→0.1313"}
  bootstrap_e1b_vs_score_only_pooled_K50:   # image-cluster paired bootstrap 5000 reps（seed=0, ci=0.95）；diff>0 = semantic 更好；per seed 1/2/3
    vs_stats_logistic: {auroc: ["+0.0190 [+0.0153,+0.0226]", "+0.0152 [+0.0123,+0.0181]", "+0.0121 [+0.0092,+0.0149]"],
                        e_aurc_reduction: ["+0.0782 [+0.0619,+0.0941]", "+0.0625 [+0.0493,+0.0756]", "+0.0475 [+0.0338,+0.0609]"]}
    vs_msp:            {auroc: ["+0.0230 [+0.0184,+0.0275]", "+0.0159 [+0.0126,+0.0192]", "+0.0146 [+0.0110,+0.0182]"],
                        e_aurc_reduction: ["+0.1070 [+0.0880,+0.1252]", "+0.0706 [+0.0560,+0.0848]", "+0.0670 [+0.0501,+0.0837]"]}
  ablations:
    e2_variants_K50: "E2-score 0.7930±0.0036 ≈ msp；E2-semantic 0.6611（纯语义交互远弱）；E2-combined 0.7878；E2-combined vs stats_logistic per seed −0.0067/−0.0011/−0.0033（无正增益，1/3 seed CI 完全 <0）"
    e3_full_vs_top5_K50: "0.7508 vs 0.7450（+0.0058；full-set 略优）；E3 全族 vs stats_logistic −0.0336~−0.0452（CI 全 <0）"
    e3_logk_K50: "0.7480（vs e3_full −0.0028）→ logK 无正贡献"
  e1b_coefficients_b3_mean:    # 标准化 logistic 系数（predictive association，非因果）
    negative_top: {cand_vmax: "-0.4852±0.0678", cand_vmean: "-0.1300±0.0580"}
    positive_top: {density_070: "+0.1692±0.0306", cand_top15_mean: "+0.1417±0.0461", cand_top12_sim: "+0.1090±0.0196", clip_margin12: "+0.0959±0.0182"}
  ood_stability_b3_mean:       # ΔAUROC(K5−K50) / E-AURC(K50)/E-AURC(K5) / RER50(K5)−RER50(K50)
    msp: "+0.0517 / 3.257 / +0.4660"
    stats_logistic: "+0.0494 / 3.188 / +0.4646"
    e1b_stats_semantic: "+0.0354 / 2.990 / +0.4313"
    e2_combined: "+0.0389 / 2.909 / +0.4303"
    e3_full: "+0.0471 / 2.861 / +0.4234"
  error_subsets_K50:           # pooled；quartile 内 AUROC（score=stats_logistic / e1b / e2_combined / e3_full）
    cand_top12_sim: {Q1: "0.8061/0.8270/0.7854/0.7614", Q2: "0.7756/0.7950/0.7660/0.7391",
                     Q3: "0.7740/0.7906/0.7699/0.7395", Q4: "0.7704/0.7842/0.7630/0.7205", all: "0.7915/0.8069/0.7878/0.7508"}
    clip_margin12:  {Q1: "0.7314/0.7565/0.7230/0.6916", Q2: "0.7720/0.7893/0.7721/0.7348",
                     Q3: "0.7837/0.7967/0.7819/0.7335", Q4: "0.8193/0.8285/0.8146/0.7908"}
    note: "e1b 在全部 8 个分组均 ≥ stats_logistic（Δ+0.012~+0.035）；高错误率/歧义组内语义增益仍然存在"
  matched_score_K50:           # treatment = cand_top12_sim 上四分位；control = 其余；匹配变量 [MSP,margin,entropy] z-score 1-NN（不用 correctness）
    delta_correctness_per_seed: ["−0.1108 [−0.1301,−0.0918]", "−0.1369 [−0.1568,−0.1166]", "−0.1198 [−0.1405,−0.0997]"]   # CI 全排除 0
    note: "控制 score statistics 后，高 top1-top2 语义相似度（top-2 歧义）组 B3 正确率低 ~11-14pp —— score 之外的可靠性信息直接证据"
  sufficiency_gate:            # A7.6；比较 = msp → e1b_stats_semantic（3-seed mean tune 选定）
    verdict: INCONCLUSIVE      # gray zone（§31）：全部 CI 排除 0 且方向一致为正，但三项阈值全未达（0.0179<0.02 / 8.15%<10% / 2.48pp<5pp）；NO-GO 不触发（ΔAUROC≥0.01）→ 先 error analysis，不自动加 Transformer
    cell_K20: {delta_auroc: "0.0143 [0.0112,+0.0175]", e_aurc_reduction: "6.84% [5.31%,8.42%]", rer50_gain_pp: "+2.46 [+1.25,+3.88]"}
    cell_K50: {delta_auroc: "0.0179 [0.0140,+0.0216]", e_aurc_reduction: "8.15% [6.47%,9.79%]", rer50_gain_pp: "+2.48 [+1.48,+3.69]"}
    per_seed_K50_delta_auroc_ci_low: [0.0184, 0.0126, 0.0110]   # 3/3 seeds CI_lo>0；seed-mean Δ=0.0179
  counts: {n_pooled_common_rows: 10286, n_pooled_images: 1490, predictions_rows_per_seed: 62272}
  uncertainty: {bootstrap_replicates: 5000, resampling_unit: image, ci_level: 0.95, seed: 0}
  risk_definition: "risk = 1 - accuracy"
run:
  command: "E:\conda\envs\deepminer\python.exe _launch_detached.py logs_phase1.txt scripts/run_phase1.py --out results/phase1_semantic_sufficiency --bootstrap-replicates 5000 --log-file logs_phase1_driver.txt（detached；同链路先用 --tiny 冒烟）"
  device: "cuda（torch 2.5.1+cu121；E2/E3 训练）/ cpu（bootstrap，numpy）"
  step_seconds: "total 5915.7s（~98.6 min）：load 24.4 / features 16.7 / models 395.2 / aggregate 4.5 / bootstrap 5327.2（5000 reps；bootstrap.csv 750 行=cross-K 540+pairwise 168+ratio 42）/ analyses 5.5 / figures+gate 140.4"
  incident: >
    (a) A7.3 预结果修订：生成任何结果前发现 T_clip=1 使 CLIP softmax 饱和（clip_entropy 退化恒定 ≈logK；
    核实 openai ViT-B/32 logit_scale 恰为 100）→ 修订为 T_clip=0.01 并记入协议后才继续。(b) A7.6 预运行澄清：
    E2-score（无 embedding 输入）排除出 best-semantic 候选，防止 gate 退化为两个 score 模型比较。(c) 首次全量
    运行在 seed3 阶段 OOM（K50 float32 打包 ~2GB ×多份）中止 → 修复（embedding float16 视图 + E3 分块预测
    chunk=4096 + 不保留 packed 缓存）后重启，一次跑完无中断。(d) detached 运行 stdout 重定向 → tqdm 进度条
    自动禁用（仅显示层，非错误）。
artifacts:
  root: results/phase1_semantic_sufficiency/     # 26 文件
  top_level: [protocol.json, metadata.json, sufficiency_gate.json, semantic_stats.csv, aggregate.csv, bootstrap.csv,
              pairwise_vs_score_only.csv, ood_stability.csv, error_subsets.csv, matched_score_analysis.csv]
  families: [e1_logistic/{selection.json,metrics.csv,coefficients.csv}, e2_top_competitor/{selection.json,metrics.csv},
             e3_semantic_deepsets/{selection.json,metrics.csv}]
  embeddings_features: semantic_features/e1_stats_b3_seed{1,2,3}.npz
  predictions: predictions/b3_seed{1,2,3}.csv.gz（val_select/testA/testB × K∈{5,10,20,50}；62,272 行/seed；§39 列）
  figures: [figures/semantic_reliability_vs_K.png, figures/selective_risk_vs_K.png, figures/ambiguity_stratified_k50.png]
  embedding_cache: cache/semantic_phase1/（gitignored，~GB 级；cohort hash 与重建方式见 protocol.json）
notes: >
  Phase 1 candidate semantic information sufficiency 审计（指令全文 + Amendment A7）。目的：在冻结 B3 ranking 与
  score-only 结论（Phase 0.5 GO_candidate_embeddings）之上，检验 candidate/query semantic representation 是否携带
  超出手工 score statistics 的额外可靠性信息；以及（若有）该信息是否集中于最强竞争者。关键观测：
  (1) 选型（3-seed mean tune）：best score-only = msp（0.8201），best semantic（A7.6 候选集）= e1b（0.8202）；
  e2_score 原始 tune 最高（0.8216）但按 A7.6 预运行澄清不属语义候选。
  (2) 主结果（e1b vs msp，pooled K50）：ΔAUROC +0.0179 [0.0140,+0.0216]，E-AURC reduction +8.15% [6.47,9.79]，
  RER@50 +2.48pp [1.48,3.69]；K20 cell 同向（+0.0143 / +6.84% / +2.46pp）；全部 CI 排除 0，3/3 seeds 显著，
  但三项阈值全部略低于 A7.6 PASS（0.02 / 10% / 5pp）→ verdict INCONCLUSIVE（gray zone；NO-GO 不触发）。
  (3) 语义信息的集中性：e1b（仅 34 params 手工统计）一致优于 E2-combined（full-set 交互 + 123k params）与
  E3（149k params；E3-full 虽略优于 E3-top5 +0.0058 但仍显著低于 score-only）→ 有用信息集中在「胜者 vs
  最强竞争者」的几何关系（cand_vmax 系数 −0.485 主导；density_070 / cand_top15_mean / cand_top12_sim 次之），
  全候选集建模不增加价值；E2-semantic（纯语义）远弱（K50 0.66），说明语义单独不足以替代 score 信息。
  (4) 语义信息的直接证据：matched-score diagnostic（不用 correctness 匹配）中高 top1-top2 相似度组
  Δcorrectness −11~−14pp（CI 全排除 0）——控制 score statistics 后语义相似度仍携带正确性信息。
  (5) OOD stability：e1b 同时提升绝对质量（K50 AUROC +0.0179 vs msp）并减弱 K 退化（ΔAUROC K5→K50
  0.0354 vs msp 0.0517；E-AURC ratio 2.990 vs 3.257）；但幅度不足以通过 gate。
  (6) error subsets：e1b 在 K50 全部分组（cand_top12_sim / clip_margin12 各 quartile）优于 stats_logistic（+0.012~+0.035）。
  (7) 测试与校验：全量 pytest 561 passed（含 §41 15 项 driver 级测试映射）；E0 复现校验 msp 0.0 / stats_logistic ≤2.8e-16；
  产物完整性校验通过（26 文件 + predictions 62272 行/seed + E0 checks）。
  (8) 结论（回答最终两问）：Q1 —— 控制 score 信息后，candidate/query semantic representation 仍携带额外的可靠性信息
  （方向一致、CI 全部排除 0），但幅度处于「证据存在、阈值未达」的 gray zone，需下一阶段先做 error analysis
  （本条目不自动推进，未实现任何 Transformer / reranking）。Q2 —— 有用信息集中在最强竞争者的几何关系中，
  full candidate set 不增加进一步价值。本阶段到此停止（§44）。

# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p1f-hard-competition-semantic-confirmation-20260929-01
git_commit: "0a654b04e47eebb1767800e2435f7f81b74a535c"   # 全量运行时的 HEAD；本条目与代码随收尾 commit 入库
dirty: true                    # src/ccg/semantic/{hard,hard_scores,frozen,hard_eval}.py、scripts/run_phase1f.py、
                               # tests/test_hard_competition.py、协议 A8、results/phase1f_hard_semantic/ 运行时尚未提交
timestamp: "2026-09-29"
dataset: refcoco+              # 与 p0/p0a1/p0b/p05/p1 完全相同的 common cohort（testA+testB 10,286 行 / 1,490 images / 3,647 refs）
split: [testA, testB]          # pooled test 行集；未重划；A6.2 切分仅用于冻结模型的恢复校验，不用于本轮任何拟合
candidate_protocol: "manifests-v1（frozen，seed 20260927）；本轮唯一改变的是候选组成（matched random → GT same-category），target/image/ref 逐行严格配对，未重采样"
K: [5, 10]                     # 冻结模型只在 K∈{5,10} 上训练/选型；K20/K50 不进入本轮（无 OOD 外推）
hardness: "random（matched control）vs same_category（primary）vs level m∈{0,2,4,8}（ExpB 剂量-响应）"
target_presence: target-present only（与 Phase 0/0.5/1 相同的 common cohort 规则）
backbone: "OpenCLIP ViT-B/32 laion2b_s34b_b79k（frozen cached z_q/z_i，L2-normalized 512-d；本轮不重新前向）"
model: >
  Phase 1F 冻结推断 zoo（协议 Amendment A8；无任何新训练）：grounding = B3 independent MLP
  （results/phase0b_independent/seed_{1,2,3}/model_best.pt 权重逐字加载，不重训）；
  reliability = MSP（R0 参考）、Stats Logistic（A6 17-d score statistics，3-seed 恢复 max|Δ|=2.2e-16）、
  E1b = 17-d stats ⊕ 16-d semantic（A7 33-d，系数 max|Δ|=0.0）。
  冻结量：B3 权重与 per-seed corrected temperature（1.115344/1.116630/1.100525）、
  Stats/E1b 系数、train-only 17-d 与 16-d 归一化（features/{scorer}_normalisation.json、Phase 0.5/1 产物）。
  禁止事项（A8.8）：训练新 reliability 模型 / 重训 E1b / feature selection / E2-E3 tuning /
  candidate-aware grounding / reranking / Transformer / attention / FineCops / CLIP-hard primary / backbone 改动。
seed: [1, 2, 3]                # B3 scorer seeds，各自独立报告；significance 从不把三 seed 预测拼接为独立样本
treatment_design:
  samecat_k5: "C_5 = [target] + 4 个 GT same-category distractors（same_category manifest 的同类前缀）"
  matched_random_k5: "C_5 = [target] + 同一 manifest sorted order 的前 4 个 random distractors（逐行同 target/同 image 簇）"
  samecat_k10: "C_10 = [target] + 9 个 same-category distractors（same9 cohort）"
  expb_levels: "K=10，same8 cohort，m∈{0,2,4,8} 个同类 + 其余来自 manifest 尾部 shuffle（确定性、可复现）"
  cohort_feasibility: "per-ref 可用同类 distractor 数：min 0 / median 11 / mean 13.7 / max 46 → same4 9,487 行、same8 7,410 行、same9 6,765 行"
integrity_checks:
  a8_4_raw_score_stop: "random regime 重新打分必须复现冻结 raw_scores/K{5,10}.npz；实测 max|Δ| ≤ 1.29e-5（阈值 1e-4，float16 存储精度量级）"
  a8_4_frozen_recovery: "5 项恢复校验 max|Δ|：stats_logistic 预测 2.22e-16、E1b 系数 0.0、stats/E1b tune AUROC 0.0、stats17 归一化 0.0（阈值 1e-9）"
  matched_pairing: "hard/rand 单元逐行同 sentence_id（_dod_rows 断言）；cohort 级 per-ref target/sorted-order/n_same 相等在 load_hard_cohort 内断言"
  manifest_dump: "manifests/ 8 个 *_candidates.npz（逐行 C_K bank 索引 + index.json provenance + sha256）"
uncertainty: {bootstrap_replicates: 5000, resampling_unit: image cluster, ci_level: 0.95, rng_seed: 0,
              pairing: "hard/rand 四块预测共享同一次 cluster 抽样（diff-of-diffs）"}
results:
  cohort_sizes: {pooled: 10286, samecat_k5_matched_pair: 9487, samecat_k10_matched_pair: 6765, expb_same8: 7410,
                 images: {pooled: 1490, same4: 1424, same8: 1189, same9: 1085}}
  manipulation_check:   # K5 paired image-cluster bootstrap，3/3 seeds 通过（A8.7 criterion：cand_vmax 与 cand_top12_sim 位移 CI_low>0）
    cand_vmax:      "hard 0.7678 vs rand 0.7138，shift +0.0540 [CI_low ≈ +0.049]（3/3 seeds）"
    cand_top12_sim: "hard 0.7185 vs rand 0.6660，shift +0.0525 [CI_low ≈ +0.047]（3/3 seeds）"
    clip_margin12:  "hard 0.0251 vs rand 0.0312，shift -0.0061 [CI_high < 0]（竞争更激烈 → CLIP margin 收窄，方向符合预期）"
    manipulation_ok: true
  grounding_difficulty: # 3-seed mean；accuracy shift 为 paired bootstrap CI
    b3_accuracy: "rand5 0.7850±0.0024 → hard5 0.6169±0.0038（shift -0.1681 [-0.1818,-0.1549]）；rand10 0.6597 → hard10 0.5425（shift -0.1171 [-0.1318,-0.1026]）"
    score_margin: "3.427 → 1.758（K5）/ 2.000 → 1.403（K10）"
    msp: "0.7832 → 0.6274（K5）/ 0.6443 → 0.5120（K10）"
    entropy: "0.5357 → 0.9228（K5）/ 0.9495 → 1.3597（K10）"
  stats_logistic_auroc: "rand5 0.8370±0.0017 / hard5 0.8128±0.0011 / rand10 0.8001±0.0052 / hard10 0.8202±0.0027"
  e1b_auroc:            "rand5 0.8383±0.0023 / hard5 0.8446±0.0030 / rand10 0.8067±0.0048 / hard10 0.8543±0.0058"
  msp_auroc:            "rand5 0.8368±0.0018 / hard5 0.8122±0.0013 / rand10 0.7999±0.0048 / hard10 0.8203±0.0034"
  semantic_increment:   # E1b vs Stats Logistic，per-cell paired bootstrap（3-seed mean of per-seed values/CI endpoints）
    rand5: "ΔAUROC +0.0012 [-0.0017,+0.0041]（n.s.）；E-AURC reduction -0.22% [-2.19,+1.71]；RER@50 -1.21pp [-2.33,+0.95]；RER@80 +0.49pp"
    hard5: "ΔAUROC +0.0318 [+0.0286,+0.0353]；E-AURC reduction +15.37% [+13.56,+17.24]；RER@50 +4.64pp [+3.25,+6.02]；RER@80 +5.25pp [+4.22,+6.28]"
    rand10: "ΔAUROC +0.0066 [+0.0034,+0.0098]；E-AURC reduction +3.52% [+1.73,+5.28]；RER@50 +0.77pp [-0.91,+2.30]；RER@80 +1.20pp"
    hard10: "ΔAUROC +0.0341 [+0.0302,+0.0381]；E-AURC reduction +17.84% [+15.69,+20.01]；RER@50 +5.96pp [+4.50,+7.48]；RER@80 +3.53pp"
    expb_dose_m0_m2_m4_m8: "ΔAUROC -0.0007 / +0.0106 / +0.0182 / +0.0299（单调，m0 n.s.）；E-AURC reduction -1.21% / +4.43% / +8.11% / +15.17%；RER@50 -1.47 / +2.05 / +3.55 / +5.40pp"
  diff_of_diffs:        # A8.5 primary：Δ^hard - Δ^rand，四块共享同一 cluster 抽样
    k5_delta_auroc: "+0.03062 [+0.02662,+0.03482]（per-seed 0.03280 / 0.02865 / 0.03042）"
    k5_e_aurc_reduction_relative: "+0.15587 [+0.13217,+0.18133]"
    k5_rer50_gain_pp: "+5.849pp [+3.270,+7.459]"
    k10_delta_auroc: "+0.02750 [+0.02307,+0.03212]"
  seed_consistency: "SameCat-K5 ΔAUROC CI_low：0.03153 / 0.02623 / 0.02801 → 3/3 seeds 显著（要求 ≥2/3）；3-seed mean ΔAUROC +0.03185 > 0；跨 seed std 0.00231"
  error_concentration:  # SameCat-K5 quartile（secondary diagnostic，3-seed mean ΔAUROC / E-AURC reduction）
    by_cand_vmax: "Q1 +0.0093/2.65% · Q2 +0.0116/5.22% · Q3 +0.0266/9.72% · Q4 +0.0158/4.35%（acc 0.905→0.227）"
    by_cand_top12_sim: "Q1 +0.0347/16.33% · Q2 +0.0253/11.59% · Q3 +0.0307/11.38% · Q4 +0.0101/3.55%（acc 0.809→0.305）"
    reading: "增益在语义上中等模糊（vmax Q3 / top12_sim Q1-Q3）处最大，在极端组（vmax Q4、top12_sim Q4：B3 几乎全错、conf 无区分力）回落 → 与 Phase 1「信息集中在胜者 vs 最强竞争者几何关系」一致"
  a8_6_gate:
    verdict: CONFIRMED
    label: "CONFIRMED HARD-REGIME SEMANTIC SIGNAL"
    conditions_passed: "ΔAUROC ≥0.02 且 CI_low>0；E-AURC reduction ≥10% 且 CI_low>0（branch）；seed 一致性 3/3；diff-of-diffs ≥0.005；mean>0"
    conditions_failed: "RER@50 +4.64pp < 5.0pp（branch 由 E-AURC 满足，故仍 CONFIRMED）；STRONG 未达（E-AURC 15.37%<20%、RER@50<10pp，虽 ΔAUROC 0.0318≥0.03）"
    thresholds_frozen_before_results: true
run:
  command: "E:\conda\envs\deepminer\python.exe scripts/run_phase1f.py --out results/phase1f_hard_semantic --bootstrap-replicates 5000 --log-file logs_phase1f_driver.txt（先 --smoke 冒烟：1 seed / 100 reps → results/phase1f_hard_semantic_smoke，不参与正式 gate 判读）"
  device: "cuda（B3 冻结权重打分，torch 2.5.1+cu121，RTX 4060 Laptop）/ cpu（特征、bootstrap、指标，numpy）"
  step_seconds: "total 1175.2s（19.6 min）：load 42.9（B3Corpus preload + 冻结恢复）/ stop_check 16.9 / protocol 0.2 / score 25.8（8 变体 × 3 seed 打分）/ metrics 0.5 / bootstrap 1073.0（5000 reps；bootstrap.csv 114 行 = pairwise 48 + expb_pairwise 24 + diff-of-diffs 18 + ratio 12 + expb_ratio 12）/ analyses 13.3 / gate 0.0 / figures+dumps 2.6"
  incident: >
    (a) driver 初稿 4 轮缺陷修复（base 变体未建 cell、rand 单元缺 bootstrap、expb 单元缺 rer_at_80/e_aurc 导致 KeyError、
    gate/figures 键名），全部为工程缺陷，不涉及冻结量或阈值。(b) 首次 3-seed 运行在 bootstrap 阶段（尚未产生任何
    gate/figures/增量结果，仅 point metrics 落盘）被主动中止并重启：补丁仅**新增**分析维度（subgroup 增加
    cand_top12_sim 分组、gate.json 增加跨 seed std），未改动 A8.6 阈值、cohort 规则或任何冻结模型。(c) 冻结恢复
    校验发现 Phase 1 e1_logistic/coefficients.csv 无 intercept 行 → 只比较 33-d 系数向量（记录于 protocol.json，
    max|Δ|=0.0）。(d) STOP 校验 max|Δ|≈1e-5 源于冻结 raw_scores 以 float16 存储，量级即存储精度，非打分路径差异。
artifacts:
  root: results/phase1f_hard_semantic/          # 14 顶层产物 + 3 子目录
  top_level: [protocol.json, cohort_summary.json, reliability_metrics.csv, semantic_increment.csv,
              paired_random_vs_hard.csv, bootstrap.csv, manipulation_check.csv, grounding_difficulty.csv,
              subgroup_analysis.csv, gate.json, metadata.json]
  manifests: "manifests/（8 个 *_candidates.npz + index.json，逐行 C_K bank 索引与 sha256 provenance）"
  predictions: "predictions/{cell}__{scorer}.npz（24 文件：8 cell × 3 seed，含 raw scores、correct、三模型 conf、16-d 语义、17-d stats、margin/msp/entropy）"
  figures: [figures/fig1_grounding_accuracy_random_vs_hard.png, figures/fig2_delta_auroc_random_vs_hard.png,
            figures/fig3_reduction_and_rer50.png, figures/fig4_cand_vmax_distribution.png]
code: "新增 src/ccg/semantic/hard.py（cohort/manifest 构造）、hard_scores.py（冻结 B3 权重打分）、frozen.py（冻结模型恢复+5 项校验）、hard_eval.py（diff-of-diffs / shift bootstrap / quartile / A8.6 gate）、scripts/run_phase1f.py；tests/test_hard_competition.py（§29 12 项）"
tests: "全量 pytest 573 passed / 0 failed（132.5s，deepminer 环境）；新增 12 项 hard-competition 测试全部通过"
notes: >
  Phase 1F hard-competition semantic confirmation（指令全文 + Amendment A8）。目的：在 Phase 1 A7 INCONCLUSIVE 之后，
  用**完全冻结**的 B3 / Stats Logistic / E1b 模型，只改变候选组成（matched random → GT same-category），检验语义可靠性
  信息在受控强竞争下是否变得**实质更有用**。关键观测：
  (1) 压力测试成立：B3 accuracy 0.785→0.617（K5，shift -16.8pp CI 排除 0），cand_vmax +0.054、cand_top12_sim +0.053
  （CI_low 均 >0，3/3 seeds），clip_margin12 收窄（CI_high<0）→ 语义模糊度确实被操纵成功。
  (2) 主结果（SameCat-K5，E1b vs Stats）：ΔAUROC +0.0318 [+0.0286,+0.0353]，E-AURC reduction +15.37% [+13.56,+17.24]，
  RER@50 +4.64pp、RER@80 +5.25pp；而**同一模型对**在 matched random 控制下 ΔAUROC 仅 +0.0012 [-0.0017,+0.0041]（不显著）。
  (3) 配对差（A8.5 primary）：diff-of-diffs ΔAUROC +0.0306 [+0.0266,+0.0348]，E-AURC reduction 差 +0.156 [+0.132,+0.181]，
  RER@50 gain 差 +5.85pp —— 全部 CI 排除 0，3/3 seeds 一致（std 0.0023）。
  (4) 剂量-响应（ExpB，m=0/2/4/8 同类竞争者数）：ΔAUROC -0.0007→+0.0106→+0.0182→+0.0299 单调上升，m0（纯 random）
  与 Phase 1 的 gray-zone 结论吻合 → 增益来自竞争者**类别组成**而非样本筛选。
  (5) K10 同向且更强（ΔAUROC +0.0341、E-AURC reduction 17.84%、RER@50 +5.96pp）。
  (6) A8.6 gate = **CONFIRMED**（非 STRONG：RER@50 4.64pp 略低于 5pp 门槛，branch 由 E-AURC 满足；STRONG 需 E-AURC ≥20%）。
  (7) 结论（回答 §32 最终问题）：**YES** —— 当模型必须在同 GT 类别的竞争者中选择时，candidate semantic 信息变得
  实质更有用（相对 random 控制放大约 26 倍，CI 全程排除 0，且随同类竞争者数量单调增强）。按 A8.8/§23，下一步只做
  external confirmation（FineCops-Ref）可行性评估，不直接进入更复杂模型；本轮未实现任何新模型、reranking 或 Transformer。
  本阶段到此停止。

# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p1e-finecops-external-feasibility-audit-20260929-01
git_commit: "a3ee9a07d3f58d1c1f49e33b6f1dc9424c956e0f"   # parse/images stage 运行时的 HEAD（feasibility_protocol.json 记录）
dirty: true                    # src/ccg/external/{finecops,gqa_images,feasibility}.py、scripts/run_phase1e_feasibility.py、
                               # tests/test_finecops_external.py、协议 A9、results/phase1e_finecops/ 运行时尚未提交
timestamp: "2026-09-29"
stage: "F0–F4 ONLY —— external feasibility audit + protocol branch 判定；F5–F10（full proposals / CLIP / B3 / Stats-E1b / bootstrap / gate）未启动，本轮不含任何 FineCops 模型推断"
verdict: "EXTERNAL STOP（A9/指令 §6 工程 gate：target proposal recall@0.5 = 0.7579 < 0.80 停止线；GO 需 ≥0.90）"
branch: "level_primary_only（same-name K5 availability 0.1861 < 诊断线 0.50，更低于 primary 线 0.90）"
dataset: >
  FineCops-Ref（EMNLP 2024, arXiv:2409.14750）test split；figshare article 26048050，license **CC BY 4.0**（API 实测字段）。
  image 域 = **GQA / Visual Genome**（非 COCO）；对象与 same-name 关系来自 GQA val_sceneGraphs.json（10,696 graphs，覆盖率 1.000）。
  官方同时分发 train/val 标注（expression_all_train_set.json 74,999,544 B / val 8,429,891 B）→ **该数据集并非 evaluation-only**，
  但本轮刻意不下载 train/val，A9 明令其不得进入任何训练、调参或校准环节。
inputs:
  annotations: "test_expression_all.json (12,127,597 B) / _coco_format (21,099,434 B) / test_expression_pos.json (3,075,530 B) / _coco_format (5,298,930 B) / dataset_card.json"
  scene_graph: "data/raw/gqa/val_sceneGraphs.json（44,830,665 B，sha256_16 3224baf1e87e56d0）"
  images: "GQA images.zip（21,817,965,542 B / 148,855 members）经 HTTP Range 抽取审计子集，未下载整库"
  hashes_recorded: "results/phase1e_finecops/feasibility_protocol.json（逐文件 bytes + sha256_16）"
code: "新增 src/ccg/external/finecops.py（annotation/scene-graph 解析、xywh→xyxy、target 解析、level/tuple_type/negative 分布、审计子集选取）、gqa_images.py（Range 抽取）、feasibility.py（冻结原语：name code table、graph boxes、per-expression audit、summarise、engineering_verdict、candidate_regime_decision）、scripts/run_phase1e_feasibility.py（parse/images/rpn/decision 四 stage）"
frozen_constants: "N=64（不因 FineCops 改 128）、primary K=5 / secondary K=10、recall IoU {0.5,0.7}、GO 0.90 / STOP 0.80、same-category primary 0.90 / diagnostic 0.50、tuple_type 可报最小群 300、invalid crop 最小边 4.0、audit 1000 图 seed 20260929、graph 匹配 IoU≥0.9"
f1_metadata_facts:
  positive_test: "9,605 expressions / 4,313 images（官方口径实测；实测重核，非引用论文数字）"
  level: "L1 5,730（59.66%）/ L2 3,404（35.44%）/ L3 471（4.90%）"
  tuple_type: "0_hop 2,333 / 2_hop 2,555 / 1_hop 2,146 / and 1,639 / same_attr 705 / same_attr_two_hop 227（<300 → 不单独报告）"
  negatives: "18,321 行 = negative_text 9,814 + negative_image 8,507；negative_type object 8,122 / attribute 3,569 / order 2,029 / relation 1,891 / flip 1,555 / swap_attr 1,155；negative_level L1 12,726 / L2 5,595"
  geometry: "目标边长 median 127.34 px（p10 57.58 / p90 285.55）、图像面积 median 187,500 px²、box 越界 0.0521%、边长 <4px 0 例"
  format_cross_check: "coco_format 与 vanilla 9,605 条 id 完全一致，text/level/tuple_type/objects_id 不一致数均为 0"
  target_name: "97.98% 由 scene-graph box（IoU≥0.9）解析、2.02% 回落 objects_id[0]、0 未解析（313 个不同目标名）"
  level_mechanism: "官方 level 与图内精确同名对象数高度吻合：L1 99.70% 无同名、L2 93.80% 有同名、L3 97.66% 有同名 → level 本身就是同类竞争强度分层"
  same_name_supply: "但 ≥5 个同名对象的表达仅占 2.10%（L2 5.17% / L3 5.52%）；GQA categories 只有 1 个占位条目 → 外部 same-category 只能等于 GQA 精确同名"
f0b_images_acquisition:
  result: "1000/1000 就绪：本次 fetch 642 + 已在盘 358，failed 0、missing_in_archive 0；1,000 张 JPEG 全部可解码，标注尺寸与实际像素 0 处不一致（verdict IMAGE GEOMETRY OK）"
  efficiency: "requests 642 == images 642（每图 1 个请求）；wire 85,047,571 B vs payload 84,513,451 B → 比值 1.006；1,441.16 s（≈27 图/分钟，workers=3）"
  incident: >
    (a) 初始 zipfile 路径实测 3.5 MB wire/图（≈23× 读放大，12 requests 只换回 6 张图），吞吐 ~7 图/分钟、ETA 110 分钟；
    (b) 追加并行流未提升吞吐并触发 host 按 IP 限流（HTTP 503，探针 206 成功后续请求即 503）→ 立即杀并行流，改为测量定位；
    (c) 基准实测：每图新建 TLS 连接 5.47 s、keep-alive 单请求 1.52 s、keep-alive 两请求 0.90 s → 真因是 TLS 握手 + 读放大，不是请求数；
    (d) 重写为「中心目录只读一次 + 每线程一条持久 HTTPS 连接 + 每图单个合并 Range + span 不足时增长重试」，
        解码校验局部头签名/method/长度/CRC32；两个踩坑：zip 局部文件头是 30 字节 `<IHHHHHIIIHH`（局部 extra 长度可与中心目录不同 28 vs 24），
        且 zip 内 DEFLATE 是 **raw deflate（RFC 1951）→ 必须 zlib.decompressobj(-15)**，用默认 zlib 包装会 incorrect header check；
    (e) 结论：吞吐 7 → 23–45 图/分钟，且路径由「盲写文件」升级为「CRC 验证 + 原子 replace + 幂等补漏」。
f2_frozen_rpn_audit:
  cohort: "1,000 图像 / 2,235 positive test expressions（冻结 COCO-pretrained fasterrcnn_resnet50_fpn RPN，top_n=64，与 RefCOCO+ 完全同一原语与常量）"
  runtime: "本次新提取 205 banks + 复用 795 cached banks = 10.2 s；首轮 795 banks 28.9 s；GPU 峰值显存 664 MB"
  bank: "mean/min proposals = 64/64；mean valid distractors 59.73；mean equivalent 4.27；invalid crop rate 0.015408；冗余（pair IoU>0.7 占比）0.001224"
  recall_at_05: "0.757942 [Wilson 0.739751, 0.775248]（1,694/2,235）"
  recall_at_07: "0.637584 [0.617434, 0.657261]"
  target_max_iou: "mean 0.6492 / median 0.7714 / p10 0.1303 / p25 0.5277 / p75 0.8475 / p90 0.8942"
  natural_omission_at_05: "0.242058（冻结口径：target 在 N=64 bank 中无可达 proposal）"
  gt_object_recall: "全 VG 对象口径 @0.5 0.431315 / @0.7 0.303736（含大量小/遮挡/紧密对象，仅作对照不作 gate）"
  geometry_verdict: "RELEASED BOXES IN JPEG PIXEL SPACE（n_dimension_violations = 0）"
  by_level_recall05: "L1 0.77594（1,272 行）/ L2 0.72879（837）/ L3 0.76984（126）→ 缺口与官方难度无关"
  by_tuple_type_recall05: "0_hop 0.75183（548）/ 1_hop 0.75534（515）/ 2_hop 0.73898（590）/ and 0.78272（382）/ same_attr 0.800（150）/ same_attr_two_hop 0.760（50）—— 后两项 < 300 行不单独报告"
  dims_check: "image_dims_check.csv 1,000 行：标注尺寸 vs 实际 JPEG vs scene-graph 尺寸，三方 0 处不一致（184 种尺寸，宽 281–1229 px）"
f3_candidate_availability:
  k5_random: "0.757942 —— 与 recall@0.5 逐位相等：bank 恒为 64，故 K5/K10 的结构瓶颈全部来自 target 是否可被 proposal 覆盖，而非候选数量"
  k10_random: "0.757942"
  k5_same_name: "0.186130 [0.170537, 0.202800]"
  k10_same_name: "0.059508 [0.050435, 0.070092]"
  same_name_distractors: "mean 1.628 / median 0 / p90 6 / max 42；share ≥1 0.2752、≥4（K5 所需）0.1861、≥9（K10 所需）0.0595"
  by_level_k5_same_name: "L1 0.00943 / L2 0.41577 / L3 0.44444 —— 即便只看官方 L2/L3，same-name K5 也构造不出 ≥90% 的 cohort"
  refcoco_plus_reference: "同一冻结 RPN、同一 N=64 在 RefCOCO+（results/proposal_audit/，读盘不硬编码）：ref_target_recall@0.5 0.985874、@0.7 0.899787、gt_object_recall@0.5 0.808520、COCO-GT same-category K5 availability 0.900320"
  delta_vs_refcoco: "recall@0.5 −0.2279pp；同类可用性 0.1861 vs 0.9003（口径差异已在 comparison_caveat 中披露：RefCOCO+ 用 COCO GT 类别，FineCops 用 GQA 精确同名）"
  size_stratification: "recall@0.5 随目标边长单调上升：<32px 0.2273（22 行, 0.98%）/ <64px 0.5606（289, 12.93%）/ <128px 0.6910（809, 36.20%）/ <256px 0.8539（794, 35.53%）/ ≥256px 0.9034（321, 14.36%）；49.13% 目标边长 <128px → 小目标解释相当一部分缺口，但最大桶仍只有 0.903（< RefCOCO+ 全体 0.986）→ 余下是全局感知/域差距，不能仅归因于尺寸"
b3_pipeline_compatibility: "接口兼容：所有 1,000 图产出恰好 64 proposals、crop 解码全部通过尺寸校验、RefCOCO+ 归一化与冻结系数可直接复用（10 个产物 sha256 + recovery_tolerance 1e-9 已入协议）；但『接口能跑』≠『统计可用』，recall 未达 gate 故 F5–F10 未获授权"
domain_shift_status: "PENDING_MODEL_INFERENCE —— B3 accuracy / MSP / margin / entropy 属 F7/F8，本轮从未查看任何 FineCops 模型结果，§15 的 30% SEVERE DOMAIN SHIFT 判读未触发"
gate_computation:
  criteria: "GO 需 recall@0.5 ≥ 0.90 且 K5 availability ≥ 0.90；recall < 0.80 → STOP；0.80–0.90 → GRAY ZONE（先汇报）"
  measured: "recall@0.5 0.757942、K5 availability 0.757942"
  verdict: "EXTERNAL STOP"
  regime_branch: "level_primary_only（same-name K5 0.1861 < 诊断线 0.50 → 连 diagnostic cohort 也不成立；不因 GQA 缺失而虚构 COCO 类别映射）"
  §7_handling: "指令 §7 的例外条件（feasibility 显示 K5 根本无法构造）确实成立 → 按该条要求 STOP and report，**不**把 N 由 64 改成 128、**不**换 detector、**不** fine-tune RPN、**不**用 Grounding DINO 替代冻结 pipeline"
uncertainty: {resampling_unit: "GQA image id（bootstrap cluster key，与审计一致）", note: "本轮未做 bootstrap，正式 external 阶段才需要"}
tests: >
  tests/test_finecops_external.py = 49 项通过；全量 pytest **622 collected / 622 passed / 0 failed**。
  指令 §33 十二项映射：xywh→xyxy / image ID mapping 含 neg_ 与非数字 id 被拒 /
  target assignment 与等价 proposal 移除 / 冻结 N=64 / candidate manifest 确定性 / train+val 标注不可达 /
  RefCOCO+ normalization 复用 / frozen coefficients + checksum / external 代码无 fit 调用 / difficulty metadata 保留 /
  bootstrap 以 image 为 cluster / positive 与 negative 路径分离。本轮新增覆盖 range reader 的：中心目录只读一次、
  fetch_member 单请求返回精确字节、span 不足时增长重试且有界失败、错 offset→bad local header signature、
  翻转字节→CRC mismatch、截断→ValueError、空 body→shorter than a local header、拒绝非 https、
  fetch_images 端到端幂等（含 missing_in_archive 不伪造）、_refcoco_reference 按 k{threshold+1} 取行（不得误取最易阈值）、
  size bucket 对审计行做无重叠全覆盖划分。
artifacts:
  root: results/phase1e_finecops/
  files: [feasibility_protocol.json, metadata_audit.json, audit_subset.csv, cohort_inventory.csv,
          difficulty_distribution.csv, difficulty_examples.csv, tuple_type_distribution.csv, negative_distribution.csv,
          image_dims_check.csv, image_fetch_report.json, rpn_audit.csv, rpn_audit_summary.json,
          recall_by_target_size.csv, candidate_availability.csv, external_branch_decision.json, metadata.json,
          figures/fig1_finecops_feasibility.png]
notes: >
  Phase 1E F0–F4（指令全文 + Amendment A9）：在**未做任何 FineCops 模型推断**的前提下完成 external 可行性审计，结论是负向的——
  (1) 数据集本身可用且元数据干净（license CC BY 4.0、9,605/4,313 positive test、几何零越界异常、scene graph 100% 覆盖、
  双格式逐条一致、level 与图内同名机制吻合），
  (2) 但冻结 COCO-RPN 在 GQA 图像上的 target recall@0.5 只有 0.7579（Wilson 上界 0.7752，仍 < 0.80），
  按 §6 该实验将主要在测「COCO-trained RPN → GQA domain shift」而非 candidate-semantic reliability → **EXTERNAL STOP**，
  (3) 且 same-category 强竞争 cohort 在外部数据集上不可构造（same-name K5 0.1861、K10 0.0595，均低于 0.50 诊断线；
  RefCOCO+ 同量纲基线为 0.9003），故 §36 第二问的答案是 **official FineCops difficulty levels（L2/L3 vs L1）+ matched random K5**，
  same-name 竞争连 diagnostic 维度都不应保留；
  (4) §36 第一问的答案是 **NO / 有条件**：FineCops-Ref 在技术上可解析、工程上可复现，但统计上不适合作为「与 RefCOCO+ 同等条件」的
  独立 external test——proposal 覆盖差 22.8pp 会把 A8 效应与感知域偏移混在一起。若要继续，需要一次**新的协议决策**
  （例如显式把 recall 缺口作为 pre-registered 分层协变量、或改以 COCO 域的外部数据集验证），
  该决策不在本轮自行作出：A9 阈值未改、N 未改、未查看任何 FineCops 结果，因此不存在结果后 tuning。
  本阶段到此停止（指令 §31「在 F4 前停止一次并汇报」+ §36「不要正式运行完整 FineCops external result」）。

# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: p1e-refcocog-external-feasibility-audit-20260929-01
git_commit: "cab89c365e8b0c9f07b16f13e8ec05befb38960d"   # 运行时 HEAD（= A9/FineCops EXTERNAL STOP 提交本身）
dirty: true                    # src/ccg/external/{refcocog,coco_images}.py、scripts/run_phase1e_refcocog_feasibility.py、
                               # tools/download_refcocog.py、tests/test_refcocog_external.py、协议 A10、results/phase1e_refcocog_feasibility/ 运行时尚未提交
timestamp: "2026-09-29"
stage: "G0–G5 ONLY —— RefCOCOg image-disjoint external feasibility audit + protocol branch 判定；未前向任何模型（无 CLIP extraction / B3 / Stats / E1b / bootstrap）"
verdict: "Branch A — CLEAN EXTERNAL（strict-disjoint 规模达标 + 冻结 RPN recall@0.5 0.9722 ≥ 0.90 + same-cat K5 0.6497 ≥ 0.60 + hard cohort 1890 expr / 755 img ≥ 1000/300）"
boundary_wording: "cross-dataset external validation under a shared COCO visual domain；**禁止**写 cross-domain visual generalization（指令 §24）。本节所有描述均不得暗示已完成跨视觉域验证"
previous_stage_constraint: "A9 FineCops verdict = EXTERNAL STOP（recall@0.5 0.7579 < 0.80、same-name K5 0.1861）**不得因本 amendment 而被改写或稀释**；FineCops F5–F10 仍不获授权。本轮未降低 FineCops gate、未换 detector、未改 N、未做弱形式 external 结果"
dataset: >
  RefCOCOg **UMD split**（image-level split，非 Google object-level split）；来源为 RefCOCOg 官方分发点，
  逐文件 bytes + sha256 记录于 dataset_summary.json / feasibility_protocol.json：
  refs(umd).p sha256_16 0331c7533537b67c、refs(google).p e4d8320dfd15fc21（仅用于证明未误用 Google split）、
  instances.json 96c89b426c657f2f。**不使用 RefCOCOg train**（即使存在），本轮新训练参数 = 0。
g0_dataset_integrity: >
  实测（非引用网络文献数字）：test **9,602 expressions / 5,023 refs / 5,023 objects / 2,600 images / 76 类**；
  train 80,512 / 42,226 / 42,224 / 21,899；val 4,896 / 2,573 / 2,573 / 1,300；合计 95,010 / 49,822 / 49,820 / 25,799。
  image_level_split = true、shared_images_between_splits = {}（三个 split 间 0 图重叠）。
  COCO instances 208,960 anns / 25,799 imgs / 80 类；refs→anns join unmatched_rate = 0.0。
  **目标框来源审计**：UMD refs 不携带 box，target box 由 ann_id join 得到；archive instances.json vs 官方
  instances_train2014.json 逐 ref IoU：n=5,023、min=mean=max=**1.0**、n_identical=5,023 → verdict
  **ARCHIVE AND OFFICIAL BOXES IDENTICAL**（因目标框定义直接决定 recall 含义，故必须落盘而非假定）。
g1_image_overlap: >
  RefCOCOg UMD test（2,600 图）对 RefCOCO+ 各集合的重叠图数：train **1,257**（非重叠 1,343，剩 3,979 expr / 2,075 refs）、
  val_select 65（2,535）、val_calib 58（2,542）、**development 1,380**（1,220，剩 3,448 expr / 1,796 refs）、
  testA 47（2,553）、testB 71（2,529）、**ALL RefCOCO+ 1,498**（1,102，剩 2,909 expr / 1,512 refs）。
  → RefCOCOg test 本身 **不能**直接称 external：57.6% 的 test 图像曾在 RefCOCO+ 开发或评估中出现过。
g2_external_subsets: >
  rg_external_strict = test − ALL RefCOCO+ → **2,909 expr / 1,102 imgs / 1,512 refs**；
  rg_external_devdisjoint = test − (train ∪ val_select ∪ val_calib) → **3,448 expr / 1,220 imgs / 1,796 refs**（cumulative superset，
  dev-only extra 539 行 / 118 图，其 proposals 直接复用冻结 RefCOCO+ bank）。membership 是**分区**（strict 优先），
  因此 strict_without_development = 0、两子集成员无交集、outside 两子集的 test 图 6,154 条 expr。
  size gate 实测：strict 2,909 ≥ 1500 ∧ 1,102 ≥ 500 → **STRICT PRIMARY**（主子集 = rg_external_strict）。
  图像完备性：strict 子集所需的 **1,102 张 COCO train2014 全部由本轮下载**（178,258,375 B / 126.6 s /
  failed 0 / missing 0），dev-only 的 118 张已在盘（它们同时是 RefCOCO+ testA/testB 图，因此其 proposals
  直接复用冻结 bank 而不重提）；image_dims_check.csv 逐图与实际像素/官方尺寸 0 不一致 → IMAGE GEOMETRY OK。
f_frozen_generator: "与 RefCOCO+ 逐字同源：torchvision fasterrcnn_resnet50_fpn（COCO_V1）RPN stage / class-agnostic / post-NMS top-64 / target = argmax IoU 且 IoU≥0.5 / 等价 proposal 移除。未重选 N、未 fine-tune、未换 detector；driver 不出现任何 Ks= / iou_thresh= / top_n 字面量覆盖（测试钉住）"
g3_proposal_feasibility: >
  strict 子集 2,909 行 / 1,102 图：target **recall@0.5 = 0.972155** [0.965525, 0.97754]、**recall@0.7 = 0.884840**、
  natural omission **0.027845**、K5 = K10 random availability **0.972155**（bank 恒 64，故 availability 与 recall 逐位相等）、
  gt_object_recall@0.5 0.913232。proposal 生成：G3 阶段对 1,220 图 / 3,448 expr 运行同一冻结模型，
  其中 **1,102 张 RefCOCOg 图本轮新抽取**（存 `cache/phase1e_refcocog/proposals_refcocog.h5`，
  97.66 s，gpu_peak 665 MB，mean/min bank size = 64.0/64，bank_size_violations 0），
  dev-only 118 图直接**复用 RefCOCO+ 冻结 bank**（19,992 图 / bank-v1）；后续 decision/candidates 重跑
  读取该缓存（n_banks_extracted_this_run = 0）。
  → 同域工程 gate：**EXTERNAL GO**（需 recall@0.5 ≥ 0.90 ∧ K5 availability ≥ 0.95，实测 0.9722 / 0.9722）。
  RefCOCO+ 对照（读盘 results/proposal_audit/，不硬编码）：recall@0.5 0.9858742 / recall@0.7 0.8997868 /
  gt_object_recall@0.5 0.8085217 / same-cat K5 0.9003198 → **Δ recall@0.5 = −0.013719**。
  对比 A9 FineCops 的 −0.2279（0.7579 vs 0.9859）：同 COCO 域使缺口从 22.8pp 降到 1.37pp，两个量级。
g4_same_category_availability: >
  A8 规则逐字复用（proposal → highest-IoU COCO GT，IoU≥0.5 才赋类别；K5 = target + 4 同类 distractor）。
  strict：≥**1** distractor **0.913029** [0.902236, 0.922732] / ≥**2** 0.828463 / ≥**4**（K5 所需）
  **0.649708** [0.632184, 0.666836] / ≥**9**（K10 所需）0.270540；mean 6.2118 / median 5 / p90 13 / max 28。
  → **SAME-CATEGORY PRIMARY**（≥ 0.60 启用；未达 0.75 “更好”线 → 如实记为 limited headroom，不粉饰）；
  dev-only extra 539 行 ≥4 为 0.851577（mean 9.7774），说明剔除 testA/testB 重叠图确实剔掉了部分难竞争样本。
  不要求复制 RefCOCO+ 的 0.9003（Δ = −0.250612 已作为 external shift 记录，第一版不 reweight）。
g4_hard_cohort: >
  sameCat-K5 cohort：**1,890 expressions / 755 images / 978 refs / 69 类** → **HARD COHORT OK**（≥ 1000 expr ∧ ≥ 300 img）。
  matched random control：每个 hard-eligible expr 同时存在同 sentence/ref/image/target 的 random-K5 视图，
  n_matched_random_available = 1,890（100%）、identity_ok = true。RefCOCO+ Phase 1F 参照（读盘）：
  same4 cohort 9,487 rows / 1,424 imgs / 3,369 refs；base 10,286 rows / 1,490 imgs / 3,647 refs。
g4_language_shift: >
  RefCOCO+ testA+testB（all sentences, 10,615 句）tokens mean **3.5348** / median 3 / p90 6 / vocab **2,942** /
  spatial rate **0.4268** / absolute-position rate **0.0330**；RefCOCOg UMD test（9,602 句）tokens mean **8.3875** /
  median 8 / p90 14 / vocab **4,038** / spatial rate **0.7867** / absolute-position rate **0.1858**；
  strict 子集 8.2499 / 40.34 字符 / vocab 2,144 / spatial 0.7669 / absolute 0.1585。
  → 平均句长 ≈ 2.4 倍、描述性绝对位置用法 ≈ 5.6 倍：外部价值来自**语言/标注协议 shift**，不是同一分布的重复。
  （未设计复杂 NLP taxonomy，仅用 marker-list 统计 spatial / absolute-position token）
g4_category_shift: >
  target-category：RefCOCO+ Phase 1F hard cohort（3,369 expr / 66 类）entropy **2.5329** bits、person **0.5224**、
  top：person 1760 / bowl 68 / donut 66 / giraffe 66 / chair 59；
  RefCOCOg external hard cohort（1,890 expr / 69 类）entropy **2.9210** bits、person **0.3852**、
  top：person 728 / chair 115 / giraffe 90 / car 79 / zebra 67。
  → 类别分布更均匀、person 占比下降 13.7pp（记为 external shift，不 reweight）。
g5_branch_decision: "**Branch A — CLEAN EXTERNAL**。四项输入：size_gate STRICT PRIMARY + engineering_gate EXTERNAL GO + same_category_gate SAME-CATEGORY PRIMARY + power_gate HARD COHORT OK → next_stage_allowed = true。下一轮才允许 frozen external confirmation（Random-K5 vs SameCat-K5，ΔAUROC(E1b−Stats) 与 Δ_hard − Δ_random）；本轮未运行。"
not_run_this_round: "CLIP extraction / B3 inference / Stats-E1b inference / bootstrap 任何 external 模型统计量 / FineCops continuation / retrain / 重新 calibration。因此不存在任何 RefCOCOg 模型结果，也不可能发生结果后调参"
infrastructure_notes:
  interpreter: "所有运行与测试均用项目 conda 环境 E:\\conda\\envs\\deepminer（python 3.10.19 / torch 2.5.1+cu121 / cuda_available true），与 metadata.json 记录的产物环境一致"
  a84_blas_incident: >
    全量 pytest 曾在工具沙箱下报 5 个 ERROR：`b3_seed1: A8.4 check stats_logistic_pred_max_abs = 2.009e-08
    exceeds tolerance 1e-09`。定位结论（逐步排除）：(a) A10 未触碰任何冻结输入（cache/semantic_phase1 与
    cache/proposals.h5 mtime 未变，A10 外部件只写 cache/phase1e_refcocog/）；(b) `git stash push -u` 移除全部
    A10 文件后仍复现 → 与 A10 无关；(c) 进程内 threadpool_limits 线程扫描（T=1/2/3/4/6/8/16）：3 项结构校验恒 0.0，
    仅 2 项 lbfgs 重拟合输出随 BLAS 线程数在 1e-9…6e-6 漂（归约顺序非结合）；**T=16 时 pred 2.220e-16 / coef 0.0**，
    与 A8 当时记录的 2.22e-16 / 0.0 逐位一致；本机 16 逻辑核，而沙箱亲和把进程限到 8 线程（2.010e-08）。
    → 这是 **BLAS 线程数依赖的预存在数值敏感性**，不是数据/代码回归。处置：在 `OPENBLAS_NUM_THREADS=16`
    （即与产生冻结 artifact 时相同的全核线程数）下跑全量 pytest → 668 collected / 0 failed。
    **_TOL = 1e-9 未放宽、A8/A9 协议数字未改**；该敏感性作为已知限制记录在案，论文重现时应固定线程数。
runtime: >
  各 stage 壁钟（驱动日志 + 产物 `seconds` 字段）：G0 parse **29.1 s**、G1 overlap **3.64 s**、
  G2 subsets **42.2 s**、图像下载 **126.6 s**（1,102 张 / 178 MB）、G3 RPN **97.66 s**
  （含模型加载共 103.7 s）、G4 hard cohort **4.5 s**、G5 decision **0.01 s**；
  最终一致性重跑（subsets,rpn,candidates,decision，读缓存 0 新抽取）**80.9 s**。
  全量 pytest **113 s**。本轮无 GPU 密集工作（不前向 B3 / CLIP）。
tests: >
  新增 tests/test_refcocog_external.py = **46 项通过**，覆盖指令 §26 十五项：UMD split parser / image ID
canonicalization（带 object-id 后缀的 file_name → 纯整数 id）/ overlap exclusion / dev-disjoint construction /
strict-disjoint construction / zero-overlap 断言 / 冻结 N=64（driver 必传 top_n=fe.N_PROPOSALS 且不得出现
Ks= / iou_thresh= / DEFAULT_TOP_N）/ COCO category assignment（IoU≥0.5、未知类永不匹配）/
same-category candidates / no target-equivalent proposals / matched random-hard cohort identity /
RefCOCOg train 标注不可达（代码路径含 0 个 fit 调用）/ deterministic manifest / sample-size gate /
external branch logic（A/B/C 边界与优先级）；另以 **ast import graph** 断言 GQA/FineCops 模块未被复用
（prose 提及不构成依赖）。全量 pytest：**668 collected / 668 passed / 0 failed / 113 s**。
artifacts:
  root: results/phase1e_refcocog_feasibility/
  files: [feasibility_protocol.json, dataset_summary.json, image_overlap.csv, image_overlap_summary.json,
          external_subsets.json, expression_distribution.csv, category_distribution.csv, audit_expressions.csv,
          rpn_audit.csv, rpn_summary.json, candidate_availability.csv, hard_cohort.csv, hard_cohort_summary.json,
          image_dims_check.csv, image_fetch_report.json, branch_decision.json, metadata.json,
          figures/fig1_refcocog_feasibility.png, figures/fig2_language_shift.png]
notes: >
  Amendment A10 / 指令 §1–§30：在 FineCops A9 EXTERNAL STOP 之后，改以 **RefCOCOg UMD split** 作为外部候选，
  目标是在**保持 COCO 图像域 / COCO object ontology / 冻结 proposal pipeline 可比性**的前提下检验 Phase 1F 的
  hard-competition semantic-reliability 效应能否**跨 referring-expression dataset 迁移**。它不是跨视觉域验证。
  关键发现：(1) RefCOCOg test 与 RefCOCO+ 共享 1,498/2,600 张图（含 development 1,380），所以“外部”必须靠
  **image-disjoint 构造**挣得：strict 后仍有 2,909 expr / 1,102 imgs / 1,512 refs，远高于 1500/500 下线；
  (2) 冻结 RPN 在同域上 recall@0.5 0.9722（与 RefCOCO+ 只差 1.37pp），把 FineCops 那 22.8pp 的感知域混淆
  彻底移除，因此本轮得到的是一个**干净的语义/标注分布检验**；(3) same-category K5 availability 0.6497 达 0.60
  但未达 0.75，hard cohort 1,890 expr / 755 img 统计上充分（约为 RefCOCO+ Phase 1F cohort 的 20% 行数）；
  (4) 语言分布确实不同：平均句长 3.53 → 8.39 tokens、spatial 用法 0.43 → 0.79、absolute-position 0.03 → 0.19。
  §30 两问的结论：第一问 **YES**（能构造足够大、image-disjoint、保留冻结 COCO proposal 系统与 same-category
  硬竞争操纵的 RefCOCOg 测试集）；第二问 **YES**（相对于 FineCops，它是 Phase 1F 语义可靠性效应的更干净外部检验，
  因为 proposal 域偏移被控制住了，剩余差异集中在语言/候选语义），但边界措辞必须限于
  cross-dataset external validation under a shared COCO visual domain。本阶段到 G5 停止，等待下一轮正式
  frozen external confirmation 的单独授权。FineCops 结果未删除、未重解读。
```

```yaml
# ===== FORMAL ENTRY — REAL RESULT（非示例）=====
experiment_id: v2g-cross-backbone-generalization-20260930-01
git_commit: "72d86d0ee16cbb116c4ebdf6dae76d0ce226c07e"   # 运行时 HEAD（= B1/B2 phase0b 产物 metadata 记录的 commit）
dirty: true                    # V2-A1 协议（research_protocol.md）、scripts/run_v2g_{reliability,hard}.py、
                               # scripts/{analyze_v2g_cardinality,audit_v2g_final}.py、tests/test_v2g_regression.py、
                               # cache/v2_backbones/*、results/v2_backbone_generalization/* 运行时尚未提交
timestamp: "2026-09-30"
stage: "G1–G5（V2-A1 cross-backbone generalization）：B1 OpenCLIP B/16 + B2 SigLIP B/16 全链（feature extraction → phase0b reliability → G3 cardinality → G4 Phase A/B hard-semantic → G5 overall gate）；B0 只读 V1 frozen artifacts"
boundary_wording: "允许：replicated across three tested backbones spanning two vision-language model families and two patch resolutions；**禁止** universal across all vision-language models。不含 Grounding DINO / BLIP-style cross-encoder / 不同 proposal 家族"
question_numbering: "Q1 = candidate-cardinality reliability degradation（G1/G3）；Q2 = hard-competition semantic amplification（G2/G4）；Q3 = whether local-competition method development is justified（G5）。G4/G5 为 Q2 的决定性证据；勿与 V1 Gate Q1/Q2/Q3 或 A11 外部确认混用"
verdict: "Q1 YES（3/3 CARDINALITY_REPLICATED）/ Q2 YES（3/3 HARD_SEMANTIC_REPLICATED，manipulation 全有效，3 seed 一致）/ Q3 YES（>=2/3 规则，实际 3/3）；STRONG_CROSS_BACKBONE_GENERALITY = YES；V2_METHOD_DEVELOPMENT_AUTHORIZED = YES"
backbones:
  B0_openclip_b32: "frozen reference（V1 读取）：K5 acc 0.790743 / K50 0.431992 / AUROC 0.842606→0.790445；E-AURC 恶化 2.2568x；RER50 −46.596 pp；Δrand +0.001227 / Δhard +0.031848 / A +0.030622。provenance caveat：hard/manipulation 沿用 V1 A8 定义（stats_logistic→e1b_stats_semantic；cand_vmax/cand_top12_sim），与 V2-G 的 R1→R2 + 三指标定义不同；dose 在 V1 未运行（NA）"
  B1_openclip_b16: "K5 acc 0.778066 / K50 0.421591 / AUROC 0.837350→0.783358；E-AURC 2.1288x；RER50 −45.929 pp；Δrand −0.001233 / Δhard +0.021047 / A +0.022280 / dose rho 1.000"
  B2_siglip_b16: "K5 acc 0.708047 / K50 0.287113 / AUROC 0.793050→0.717934；E-AURC 1.6853x；RER50 −46.770 pp；Δrand +0.004531 / Δhard +0.035617 / A +0.031086 / dose rho 1.000"
cardinality_gate: >
  G3：K5→K50 ΔAUROC B1 **0.053992**（per-seed 0.0598/0.0552/0.0471，CI 全排除 0）、B2 **0.075115**
  （0.0761/0.0803/0.0689，CI 全排除 0），均 ≥0.03 → Route A 通过；E-AURC 恶化 B1 2.1288x / B2 1.6853x（≥20%），
  RER50 drop B1 45.93 pp / B2 46.77 pp（≥10pp，CI 全排除 0）→ Route B 通过；两 backbone CARDINALITY_REPLICATED。
hard_semantic_gate: >
  G4 Phase B（3 scorer seeds，全长 484.0 s）：B1 Δhard **0.021047** / A **0.022280**；B2 Δhard **0.035617** /
  A **0.031086**（image-clustered paired bootstrap，共享 draw，5000 reps / seed 0 / ci 0.95，cluster=image_id）；
  manipulation（winner_competitor_max_cos↑ / winner_top2_cos↑ / q_margin12↓，≥2/3 方向正确且 ≥1 CI 排除 0）
  两 backbone 全有效；dose-response m∈{0,2,4,8} Spearman rho = 1.000（secondary，不入总 gate）→ 两 backbone HARD_SEMANTIC_REPLICATED。
overall_gate: "G5：n_backbones=3 / n_both_replicated=3 → authorized=true；B0 按 V1 frozen replication 计入"
protocol_history: >
  (1) thresholds 在任何 B1/B2 结果产生前冻结（V2-A1 修正案 + g4_protocol_freeze.json，frozen_before_phase_a=true）；
  (2) G1→G5 未修改任何 primary gate（回归测试断言 gate.thresholds == freeze item 9）；
  (3) semantic feature 定义在 G4 前冻结（14 个 backbone-neutral primary，列序冻结）；
  (4) random（Phase A）出数后未改动 hard config；(5) dose-response 仅 secondary；(6) **LCR 尚未实现**。
  manipulation 具体化说明：V2-A1.6（2 指标、≥1 显著）→ 冻结 item 7（3 指标、≥2/3 + ≥1 CI），属收紧而非放松。
runtime: >
  B1 特征提取 wall 11537.6 s（region 8755.6 s / text 114.7 s；1,279,488 region crops；peak VRAM 1099 MB）；
  B2 特征提取 wall 4497.4 s（region 4026.8 s；peak VRAM 1449 MB）；phase0b reliability B1 7986.8 s / B2 6384.0 s（各 3 seed × 21,373 records）；
  G4 Phase B 484.0 s；全量 pytest 107.6 s。环境：deepminer（python 3.10.19 / torch 2.5.1+cu121 / cuda 12.1 / RTX 4060 Laptop）。
tests: >
  新增 tests/test_v2g_regression.py = **50 collected（item31：18 协议项 + 8 项 G4/G5 专项）**，覆盖 feature banks 完整性
  （19992 img / 1,279,488 region / 141,564 text / 0 invalid）、512/768 维、manifest identity、B3 candidate independence
  （score_invariance 0 violations）、参数预算 214,273 / 312,577 < 500k、3 seed 隔离、K20/K50 不进 selection、global T 仅用
  val_calib K5/K10、T interior、温度不跨 backbone/seed、无 OpenCLIP-specific 假设、primary continuous/rank、R1/R2 不读 raw
  embeddings、cohort identity、shared bootstrap draws、manipulation 仅 backbone-neutral primary、3-seed independence、gate
  deterministic；专项：系数/normalization 跨 backbone 隔离、selected C 可复推、reliability train 无 hard 行（7,372 < 10,286）、
  Random/Hard 行 identity、amplification=Δhard−Δrand 共享 cluster draw、dose manifests 不可变（n=7,410）、Phase-A 不能改
  Phase-B config。全量 pytest：**774 collected / 774 passed / 0 failed / 0 error / 107.6 s**。
audit: "scripts/audit_v2g_final.py 从 raw G3/G4/G5 artifacts 重算全部 headline 并逐项核对 gate/metadata/final_summary.csv：82 checks / 0 failures（AUDIT_OK），B0 行由 V1 frozen artifacts 载入（非硬编码）"
artifacts: >
  results/v2_backbone_generalization/：protocol.json、g0_provenance.json、backbone_metadata.json、feasibility.csv、
  b1_phase0b/、b2_phase0b/、g3_cardinality_gate.json、g4_protocol_freeze.json、g4_phaseA/、g4_phaseB/（gate.json /
  amplification.csv / manipulation_check.csv / dose_response.csv / g5_overall_gate.json / metadata.json）、
  final_summary.csv、v2a1_result_record.json；cache/v2_backbones/{openclip_b16,siglip_b16}/（独立 namespace，未动 V1 cache）；
  协议：docs/research_protocol.md（V2-A1 + Result Record）；脚本：run_v2g_reliability.py / run_v2g_hard.py /
  analyze_v2g_cardinality.py / audit_v2g_final.py。
observation: >
  SigLIP（B2）absolute grounding quality 最弱（K5 acc 0.708 / K5 AUROC 0.793，为三 backbone 最低），但 cardinality
  degradation（ΔAUROC 0.0751，最强）与 hard-semantic amplification（A 0.0311，新 backbone 中最强）反而最大 →
  支持 "absolute grounding performance != reliability robustness under candidate-set shift"。仅作观察，**不解释为因果机制**。
notes: >
  V2-A1 的正式执行结果（含 V2-A1.10 的 G0 BLOCKED→后补实测历史，feasibility/dims 由真实 checkpoint introspection 落实）。
  B0 数字全部从 V1 frozen artifacts 载入；rer50_drop 在 g3 内为 fraction（0.4593=45.93 pp），主表统一为 pp。本轮完成后停止；
  下一轮（单独设计）：V2-M Local Competition Reliability Module——protocol 必须建立在当前 frozen V2-G 结果之上。
```

---

# V2 post-A11 result records

> **本节性质**：append-only **回顾性结果登记**（retrospective result record）。它把 V2-A11 之后各轴**已经执行完毕、
> 已经冻结、已经落在各自 artifact 里**的结果正式登记进本日志；它**不**追溯性地把任何已完成实验声明为预注册，
> 也不改动上方任何历史条目（含 V2-G 条目 `protocol_history` 中「**LCR 尚未实现**」这类当时状态的句子——那是
> 2026-09-30 的真实历史文本，保留，不覆写）。
>
> **字段纪律**：本节日采用与 V2-G 条目相同的 **axis-specific / legacy 简化字段**风格，而**不是** §2 schema 的
> 逐字段填满。原因：这些轴的真实产物里根本没有 §2 的若干字段（例如 D1/D2/M 的 metadata 不携带
> `experiment_id` / `git_commit` / `dirty`）。为统一格式而编造这些字段值即为本节禁止事项，因此凡 artifact
> 未记录者，本节写 `未在 artifact 中携带`，并注明 provenance 只能由 git 历史回读。
>
> **分类学**（每条必须且只能属于其一）：`pre-result freeze`（结果产生前冻结的协议/config）／`result record`
> （结果登记）／`post-result correction`（结果产生后的解释修正）／`descriptive diagnostic`（描述性诊断，
> 不得创造、加强或削弱任何存在性主张）。
>
> **数字来源**：本节所有数值均从下列真实 artifact 回读，无一项由今日总结倒推：
> `results/v2_data_robustness/d1_reviewed_annotations/*`、`results/v2_d2_refcoco_lang/*`、
> `results/v2_local_competition/*`、`results/v2_proposal_robustness/*`。

## V2-D1｜人工复核标注鲁棒性（annotation-cleanup robustness）

```yaml
# ===== V2 post-A11 RESULT RECORD — axis-specific / legacy（非 §2 schema 全字段）=====
record_class: result record（登记动作发生在结果之后；测量本身有 pre-result 冻结的门禁，见 gates_frozen_before_measurement）
experiment_id: v2d1-reviewed-annotation-robustness-20261001-01   # 登记用标识；D1 artifacts 内无 experiment_id 字段
branch: v2-data-robustness
commit: "7feaa68（2026-10-01，D1 全部结果工件的入库提交）"
frozen_config_commit: >
  **不存在**：`results/v2_data_robustness/d1_config_freeze.json` 与
  `results/v2_data_robustness/d1_reviewed_annotations/protocol.json` 在磁盘上均不存在（已核实）。
  D1 的结果前约束不是 config 文件，而是 metadata.json 里的 12 项 `frozen_inputs_sha256`
  （`cache/proposals.h5`、6 份 candidate manifests、3 份 phase0b per_sentence 产物、2 份 refcoco+ 原始标注）
  加 3 份 `reviewed_source_sha256`。因此本条**只**能声称「输入身份被哈希冻结」，不能声称存在过一份 D1 协议文件。
classification: legacy / axis-specific result record（artifact 不携带 CONFIRMATORY 等 classification label；
  判定以 `c*_verdict.label` 与 `gate_verdict` 为准）
dataset: "RefCOCO+ 人工复核子集（`d1_clean`）；cohort 15,762 sentences = val 7,875 / testA 4,176 / testB 3,711"
candidate_protocol: "冻结 N=64 RPN proposal bank + V1 冻结 random / same_category manifests（未重建、未重新采样）"
backbone: "B0 = V1 冻结 OpenCLIP ViT-B/32（未更换）"
proposal_family: RPN
training_status: >
  `new_training_parameters: 0`。stage 1 `feasibility + mapping + proposal compatibility (no model run)`
  runtime 19.25 s；stage 2 `C1 + C4 replay on D1_clean (frozen weights, no training)` runtime 1152.25 s。
split_usage: "V1 冻结 splits 原样复用（val / testA / testB）；`cohort_unmodified_after_inspection: true`"
gates_frozen_before_measurement: >
  sample-size gate（clean test expressions ≥ 3000 且 images ≥ 500 → FULL）实测 8,010 expressions / 1,474
  images → `FULL_ROBUSTNESS_AUDIT`；proposal gate（target recall@0.5 ≥ 0.95 → CONTINUE）实测
  **0.9850531762** → `CONTINUE`。两条判据文本均在 `d1_reviewed_annotations/metadata.json` 的
  `gates.*.criterion` 中，本条只引用不回写。
metrics_k50: >
  ΔAUROC(correct) **+0.055635** CI [0.044618, 0.066585]；E-AURC absolute **−0.084453**
  CI [−0.090576, −0.078311] / relative **−0.711826** CI [−0.736816, −0.685044]；
  RER@50 **+0.451685** CI [0.423481, 0.478656]；C4 侧 delta-AUROC CI lower > 0 且
  degree-of-difficulty **0.030071** CI [0.025625, 0.034615]，与 V1 同方向。
verdict: >
  `C1 ANNOTATION-ROBUST` + `C4 ANNOTATION-ROBUST`，综合
  `CORE FINDINGS ROBUST TO REVIEWED ANNOTATIONS`；C4 `gate_verdict = CONFIRMED`。
artifact_paths: >
  results/v2_data_robustness/d1_reviewed_annotations/{verdict.json, metadata.json, metadata_stage2.json,
  feasibility.json, proposal_audit.json, source_manifest.json, clean_manifest.csv, mapping_audit.csv,
  removed_ambiguous_audit.csv, removed_ambiguous_summary.json, bootstrap.csv, c1_cardinality.csv,
  c4_hard_semantic.csv, figures/}
boundary_and_forbidden_claims: >
  允许：annotation cleanup robustness（同一 cohort 上把人工复核判定为 ambiguous 的表达式移除后，两个核心效应
  仍在）。**禁止**：(a) 把它写成「新数据集」——它是标注清洗，`reviewed_annotation_namespace:
  data/reviewed_refcocoplus/ (originals untouched)`，原始 `data/raw/refcoco+` 从未被修改；
  (b) 把它写成 cross-visual-domain / 跨视觉域泛化——仍在同一 COCO 图像域内；
  (c) 把「移除歧义行」写成对主线数字的改写（V1 数字不动）。
```

## V2-D2｜RefCOCO 语言分布迁移（含一次必须登记的 protocol redefinition）

```yaml
# ===== V2 post-A11 RESULT RECORD — axis-specific / legacy（非 §2 schema 全字段）=====
record_class: result record + post-result protocol redefinition（见 phase1_feasibility_finding，该重定义不得被隐藏）
experiment_id: v2d2-refcoco-language-transfer-20261001-01   # 登记用标识；D2 artifacts 内无 experiment_id 字段
branch: v2-data-robustness
commit: "0f9c7ab（2026-10-01，D2 全部结果工件的入库提交）"
frozen_config_commit: >
  **不存在**：`results/v2_d2_refcoco_lang/metadata.json` 与 `d2_config_freeze.json` 均不存在（已核实）。
  D2 的 pre-result 证据是 `c1_c4_verdict.json.provenance` 里的 4 项 sha256（cohort.csv /
  inference_report / phase1_audit / phase2_feasibility）与 `phase1/cohort.csv.sha256`；
  以及 Phase-2 冻结声明：`V2-D2 Phase 2 (proposal feasibility, frozen N=64 RPN)`，其
  `gate.note = "proposal generator frozen; a STOP is reported, never answered by tuning the RPN"`。
classification: legacy / axis-specific result record
dataset: "RefCOCO（UNC refs(unc).p + instances.json）testA ∪ testB"
candidate_protocol: "冻结 N=64 RPN bank；nested K ∈ {5,10,20,50}；primary pair (K5, K50)；random + matched hard (same-COCO-category)"
cohort: "10,544 rows / 1,494 images / 3,707 refs；C1 common cohort 10,400；C4 matched hard 9,748"
backbone: "B0 冻结 OpenCLIP ViT-B/32"
proposal_family: RPN
training_status: >
  无训练：`models.scorer = frozen phase0b B3 seeds (global_T_corrected)`、
  `models.reliability = frozen R1 stats_logistic / E1b e1b_stats_semantic`、`models.new_parameters = 0`；
  Phase 3b 冻结推理 runtime 310.58 s（peak VRAM 100.1 MB，5 jobs × 3 seeds）。
statistics: "image-clustered paired bootstrap，5000 reps / seed 0 / CI 0.95，hard vs random 共享 cluster draws"
selection_split: "无选择环节（零新参数）；calibration 沿用冻结 per-seed global_T_corrected，不按数据集重拟合"
eval_split: "RefCOCO testA ∪ testB（development images 未进入）"
phase1_feasibility_finding: >
  **strict image-disjoint RefCOCO premise impossible**。Phase-1 审计实测
  `overlap.image_disjoint_premise_holds = false`：RefCOCO test 的 1,500 张图像与 RefCOCO+ 完全重叠
  （per_set：testA overlap 750/750、testB overlap 750/750、all_refcoco_plus overlap 1500；仅 train / development
  为 0 重叠），因为 RefCOCO 与 RefCOCO+ 是对同一批 COCO 区域的重新标注。因此 D2 被**重新定义**为
  表达式语言分布迁移（artifact 原文）：
  `D2 is expression-language-distribution transfer on shared COCO testA/testB images under the frozen
  N=64 RPN; a strict image-disjoint RefCOCO cohort does not exist because RefCOCO and RefCOCO+ re-annotate
  the same COCO regions.`
  ——这是一次 **post-result 之前的 protocol redefinition**（在跑主 gate 前由 Phase-1 实测得出并写进 artifact），
  本节显式登记它，不隐藏。
phase2_feasibility: "gate verdict = FULL（measured recall@0.5 0.980655）；K5 same-name 供给率 0.906622，K50 same-name 供给率 0.0"
metrics_c1: >
  K5→K50 ΔAUROC drop mean **0.046856**（CI lower mean 0.033537，3/3 seed CI 排除 0）；E-AURC worsening
  mean **1.806747**（3/3 seed CI 排除 0）；RER@50 drop mean **0.468106**（3/3 seed CI 排除 0）；
  route_a_passed / route_b_passed / replicated 均 true。阈值以 artifact `c1.gate.thresholds` 为准（本条不回写）。
metrics_c4: >
  沿用 V2-G G4 item 9（frozen）在 D2 上重放：Δhard mean **0.036260**（CI lower 0.032345）、Δrand mean
  0.000680、amplification mean **0.035580**（CI lower 0.031279）；manipulation 三指标规则（≥2/3 方向正确且 ≥1 CI 排除 0）下 3/3 seed 有效。
verdict: >
  `C1 CROSS-DATASET REPLICATED` + `C4 CROSS-DATASET REPLICATED`，综合
  `CORE FINDINGS CROSS-DATASET ROBUST`。
boundary: >
  artifact 自带的边界声明必须逐字携带：**cross-dataset transfer under a shared COCO visual domain and a
  shared frozen proposal system; NOT cross-visual-domain generalisation**。
  允许：cross-dataset（语言分布）迁移下的复制、CORE FINDINGS CROSS-DATASET ROBUST。
  **禁止**：cross-visual-domain generalisation / 跨视觉域泛化 / 图像不相交外部验证（三者均不被本轴支持）。
  也不得把 D2 与 A11 的 RefCOCOg 外部验证（Q2 = NOT ASSESSABLE）合并表述。
artifact_paths: >
  results/v2_d2_refcoco_lang/{c1_c4_verdict.json, c1_point.csv, c1_bootstrap.csv, c4_amplification.csv,
  c4_manipulation.csv, inference_report.json, feature_extraction_stats.json, phase1/cohort_audit.json,
  phase1/cohort.csv, phase1/cohort.csv.sha256, phase1/image_sizes.json, phase2/feasibility.json,
  phase2/feasibility_per_expression.csv, predictions/*.npz}
```

## V2-M｜Local Competition Reliability（LCR）——M1 / M2 / M2.5

> 命名提醒：本轴的 `V2-M` 是 **Local Competition Reliability Module**（`results/v2_local_competition/`），
> 与 V2-P 轴内部的机制诊断 `V2-P2-M` **不是同一事物**；引用时不得混用。

```yaml
# ===== V2 post-A11 RESULT RECORD — M1（random-only 训练）=====
record_class: result record（针对 pre-result 冻结的协议）
experiment_id: v2m-m1-random-only-20260930-01   # 登记用标识；gate.json 内无 experiment_id 字段
branch: v2-local-competition-reliability
commit: "运行 HEAD = 0aaa29d5d2b48f28a408ab556c123a9aa8fbaa3e（dirty=true，见 m1_metadata.json）；结果入库 = c92c86d"
frozen_config_commit: >
  协议文件 `results/v2_local_competition/protocol.json`（`artifact: v2m_protocol_freeze`，
  `base_commit = 0aaa29d`）的 sha256 = `90085940f75ed8cbf6ac3c2f72219b9ff34bde80c97aad89362580bf5fedd88b`，
  被 `m1_metadata.json` / `m0_architecture.json` / `m1_audit.json` / `m2_curriculum/m2_audit.json` **一致引用**。
  **provenance 注意（不得美化）**：该文件在 git 中首次出现于 `c92c86d`——即与 M1 结果同一提交。因此它的
  「结果前存在」证据链是运行时记录的 sha256 与 `created_utc = 2026-10-01T00:00:00Z` 自声明，
  **不是**提交顺序。本条只能写成如此，不能写成「协议先于 M1 入库」。
  M0 实现门禁另存：`m0_architecture.json`（stage `V2-M M0 (implementation gate before M1)`，all_checks_passed）。
classification: >
  **legacy / axis-specific negative-result record**：M1 的 artifact **不携带** classification label
  （`gate.json` 顶层只有 artifact / primary_cell / primary_question / aggregation / bootstrap / comparisons /
  gates / interpretation_boundary），判定以布尔量 `gates.M1_GO.verdict` 为准。
dataset: "RefCOCO+（V1 冻结 cohort 与 manifests）"
candidate_protocol: "primary cell = SameCategory-K5；次要 K20-random / K50-random / Random-K5 / Random-K5-matched"
backbone: "B0 冻结 OpenCLIP ViT-B/32"
proposal_family: RPN
training_status: >
  **训练了可靠性层，未训练任何定位层**。M1 = random-only（`no_hard_exposure: true`）：
  reliability_train 7,372 rows 全部来自 `val_calib`，`hard_overlap = 0`，K ∈ {5,10}（pass=true）；
  Adam + BCEWithLogits（wd 1e-4）、300 epochs、batch 256、patience 30、lr grid {1e-4,3e-4,1e-3}、
  C grid {0.1,1,10}；scorer 冻结 B3 seeds 1/2/3。LCR n_params = **1,333**（预算 ≤ 50,000），
  seed-1 tune-mean-AUROC：R1 0.812037 / E1b 0.814914 / Aggregate-MLP 0.813898 / LCR-noGate 0.817238 / LCR 0.812057。
selection_calibration_eval: "selection = tune block（Random K5/K10）；calibration = 冻结 global_T_corrected（seed1 T=1.115344，不重拟合）；eval = 冻结 testA ∪ testB"
metrics: >
  primary question `LCR > E1b?` on SameCategory-K5：**ΔAUROC = −0.029442**；
  `gates.M1_GO` 仅记录 `delta_auroc_ci_low = −0.032442`（**artifact 无 ci_high 字段**，本条不写 CI 上界）；
  per-seed delta：seed1 −0.032731 / seed2 −0.026878 / seed3 −0.028717（**三个 seed 方向全为负**）；
  `e_aurc_reduction = −0.168746`；`rer50_gain_pp = −4.4353`；secondary guard（Random-K5）
  delta −0.001267 / ci_low −0.003842 / threshold −0.005 / pass=true。bootstrap 5000 / seed 0 / 0.95，共享 draw。
verdict: "M1_GO = false"
zero_effect_guarantee: >
  grounding 不变性实测：`hard5_stop_max_delta = 7.62939453125e-06`（LCR 不得改变候选得分/排序/top-1）；
  R1 冻结禁重训；relation vector 7 维 `r_j=[ds_norm, dp, a_t, a_j, a_t−a_j, v_tj, rank_j_over_k]`，
  6 项 forbidden_additions（raw 512-d embedding / GT category / 新 meta 信息 / objectness / target IoU / hard-random 标签）。
artifact_paths: >
  results/v2_local_competition/{protocol.json, m0_architecture.json, m1_metadata.json, m1_audit.json,
  gate.json, ablations/lcr_vs_nogate.csv, bootstrap/pairs.csv, figures/m1_cohort_delta_auroc.png,
  m1_random_only/point_metrics.csv, m1_random_only/seed_{1,2,3}/{confidences.npz, model_manifest.json}}
boundary_and_forbidden_claims: >
  artifact `interpretation_boundary`（逐字）：成功时才允许说
  "A lightweight local-competition reliability module improves correctness estimation under candidate
  competition"；**forbidden**："improves visual grounding accuracy"、"solves candidate-set shift"。
  M1 为 NO-GO，因此以上主张在本轴一律不成立；也不得把 M1 写成「LCR 已被完整证伪」——它只否定了
  random-only 训练下的迁移。
```

```yaml
# ===== V2 post-A11 RESULT RECORD — M2（竞争课程训练）=====
record_class: result record
experiment_id: v2m-m2-curriculum-20260930-01   # 登记用标识
branch: v2-local-competition-reliability
commit: "运行 HEAD = c92c86d5f4ee74a74348e18129c9fc980ef0e548（dirty=true，m2_curriculum/metadata.json）；结果入库 = 8fc4c7f"
frozen_config_commit: >
  `protocol_sha256 = 90085940f75ed8cbf6ac3c2f72219b9ff34bde80c97aad89362580bf5fedd88b`（与 M1 同一文件，未改）；
  `manifest_freeze_sha256 = 32d753bea652a98edd50f3f549d99482600fefb08c84ef6d5ab273a289123297`；
  `m0_sha256 = 5a16aeceffbc3ad0db7322e07ffabf612eca507ee940fb375ff0400e5afe5112`；
  `curriculum_sha256 = cffc00ce21faf86612e1e4c96f81007407c13a300a66a0ae9f3c30257b4a5010`。
  M2 在协议文件中属 **PREREGISTERED**（`M2_preregistration` 字段；note 原文："This file freezes the M0/M1
  protocol and PREREGISTERS M2 before any M1 result exists. It must not be modified after M1 results are seen."）。
classification: legacy / axis-specific negative-result record（无 classification label；以 `gates.M2_GO.verdict` 为准）
dataset: "RefCOCO+（V1 冻结 cohort）"
candidate_protocol: "K=10 固定；竞争严重度 m ∈ {0,2,4,8}（m = winner 之外的同类竞争者数）；primary cell = m8"
backbone: "B0 冻结 OpenCLIP ViT-B/32"
proposal_family: RPN
training_status: >
  课程训练 4 个模型（m0/m2/m4 参与训练与选择，m8 仅测试）；runtime 1135.0 s；audit 713.5 s（passed）。
  训练协议与 M1 一致（Adam+BCEWithLogits / wd 1e-4 / 300 epochs / batch 256 / patience 30 / lr・C grid）。
  `r1_anchor`：存盘系数为权威（m1_confidence_repro_max_delta = 2.22e-16，rows 10286）。
selection_calibration_eval: >
  selection：`tune balanced mean AUROC on m in {0,2,4} only (m=8 never seen)`；训练行共享同一冻结集合
  （m0 3,686 / m2 3,574 / m4 3,370 train rows）；eval：testA ∪ testB（K=10）；calibration：冻结 global_T_corrected。
metrics: >
  `gates.M2_GO`：`delta1_LCR_vs_E1b` = **−0.0018578** CI [−0.0053043, +0.0016435]（3/3 seed 为负，n_positive 0）
  → `condition_delta1_ge_0.010_and_ci_low_gt_0 = false`；`delta2_LCR_vs_AggregateMLP` = **+0.0357435**
  CI [0.0302815, 0.0413613]（3/3 为正）→ `condition_delta2_gt_0_and_ci_low_gt_0 = true`；`verdict = false`。
  其他 m8 对比：LCR − Aggregate-MLP +0.0357435、LCR − LCR-noGate −0.002241（CI 全负）、LCR − R1 +0.051711（CI 全正）；
  低严重度侧 m0 LCR−E1b = +0.006518（CI 全正）、m2 +0.00007（跨 0）、m4 +0.003233（跨 0）。
verdict: >
  **M2_GO = false** → LCR v1 = **negative result**（冻结）；`authorization` 逐字：
  `{"V2-MG": false, "B1/B2": false, "v2mg_not_authorized": true}`
  → **V2-MG_NOT_AUTHORIZED**。
correct_reading: >
  必须按两个已回读的事实同时登记：`LCR > Aggregate-MLP`（+0.0357435，CI 全正）**但** `LCR ≤ E1b`
  （−0.0018578，CI 跨 0 且 3/3 seed 为负）。正确解读：
  **curriculum activated the competition branch, but the structured model did not outperform the simple
  semantic-statistics baseline.** **禁止**将其简化为 "LCR learned nothing"。
  `m8 LCR−LCR-noGate = −0.002241`（CI 全负）说明 gate 本身未带来增益，但不得写成「gate 反向有害」的因果主张。
forbidden_followups: >
  `interpretation_boundary.negative_result_note` 逐字："if M2 is NO-GO: LCR v1 is frozen as a negative result
  (M1 random-only failure + M2 curriculum result); no m=8 gate tuning, no Transformer, no hidden-size / M /
  raw-embedding changes within this protocol -- any new structure requires a new method amendment."
  forbidden claims 同 M1（improves visual grounding accuracy / solves candidate-set shift）。
artifact_paths: >
  results/v2_local_competition/m2_curriculum/{gate.json, metadata.json, training_manifest.json, m2_audit.json,
  point_metrics.csv, bootstrap_pairs.csv, severity_curve.csv, gate_diagnostics.csv,
  manifests/{manifest_freeze.json, val_level_m{0,2,4}.npz}, seed_{1,2,3}/{confidences.npz,
  model_manifest_seed*.json}, figures/m2_severity_curve.png}
```

```yaml
# ===== V2 post-A11 RESULT RECORD — M2.5（专家分工审计）=====
record_class: descriptive diagnostic
experiment_id: v2m-m25-specialist-audit-20261001-01   # 登记用标识
branch: v2-local-competition-reliability
commit: "运行 HEAD = 8fc4c7fef88df42d65407badb03d306efc12453b（dirty=true，m25_specialist_audit/metadata.json）；结果入库 = 673f56a"
classification: >
  **diagnostic**——该分类不是事后追加，而是 artifact 自带边界声明的逐字内容：
  `verdict.json.interpretation.boundary = "diagnostic only; no success gate, no mixture training here, no B1/B2."`
  因此本条**不是**新的 confirmatory method success，也不是任何 gate。
frozen_config_commit: >
  无独立 config freeze 文件；约束写进 artifact 本身：`constraints = ["read-only: no training, no fitting,
  no hyper-parameter search, no architecture change, no new candidate sampling",
  "both experts are frozen M1/M2 manifests; only inference on the frozen K=10 severity cells"]`；
  metadata 同时记录 `read_only: true` / `no_training: true` / `repro_tolerance: 1e-09`。
dataset: "RefCOCO+（与 M1/M2 同一冻结 cohort）"
candidate_protocol: "K=10 固定；severity m ∈ {0,2,4,8}；对比对象 = random-only E1b vs curriculum E1b"
backbone: "B0 冻结 OpenCLIP ViT-B/32"
proposal_family: RPN
training_status: "零训练（两个专家均为已冻结的 M1/M2 manifest，仅推断）；runtime 358.8 s；audit 399.3 s"
reproducibility: "repro_checks：random vs M1 confidences 2.22e-16；curriculum vs M2 confidences 0.0；point table 0.0"
metrics: >
  curriculum − random（seed mean ΔAUROC）：m0 **−0.008902** CI [−0.012809, −0.005096]；m2 +0.003210；
  m4 +0.014568；m8 **+0.025830** CI [0.022435, 0.029320]。专家分歧 mean_abs_diff 均在 0.041–0.050，
  `non_decreasing_in_m = false`（不假设单调）。
verdict: "SPECIALIST TRADEOFF PRESENT"
boundary_and_forbidden_claims: >
  只登记为：两个冻结专家在不同竞争严重度上发生了**分工**（课程模型在高竞争 m8 更好、在无竞争 m0 更差）。
  **禁止**：把它当作方法成功、当作 LCR 复活、或当作可发表的正结果；不得从中推出集成/混合已得到验证
  （它只是后续 V2-M3 立项的预注册发现）。
artifact_paths: >
  results/v2_local_competition/m25_specialist_audit/{verdict.json, metadata.json, m25_audit.json,
  point_metrics.csv, paired_bootstrap.csv, calibration.csv, expert_disagreement.csv,
  figures/m25_specialist_audit.png}
```

## V2-M3｜Competition-Adaptive Reliability Mixture（B0 developmental / B1・B2 confirmatory）

```yaml
# ===== V2 post-A11 RESULT RECORD — V2-M3（自适应混合）=====
record_class: result record（两段：developmental B0 + confirmatory B1/B2）
experiment_id: v2m3-competition-adaptive-mixture-20261001-01   # 登记用标识
branch: v2-competition-adaptive-mixture
commit: >
  B0：运行 HEAD = 673f56a797d0e98871eb7e3d5d6b9d5fe438078a（dirty=true），入库 = 3599369；
  B1/B2：运行 HEAD = acd83b03a4daa7d7434b9be52d3c82438e17f1de（dirty=true），入库 = 11efec7。
developmental_vs_confirmatory: >
  **两段必须区分，不得合并为一项实验**：
  (a) `m3_mixture/b0/metadata.json.label = "POST-HOC DEVELOPMENTAL"`（amendment V2-M3；runtime 1177.1 s；
  `read_only: true` / `no_training: true` / `experts_frozen: true`）——混合器拟合仅在 m0/m2/m4，m8 仅诊断；
  (b) `m3_mixture/conf/metadata.json.label = "CONFIRMATORY"`（amendment V2-M3.1；runtime 2654.3 s；
  `read_only_experts: true` / `no_training_except_frozen_curriculum_protocol: true`）——这才是判定段。
frozen_config_commit: >
  `protocol_m3.json`（amendment **V2-M3**，title "Competition-Adaptive Reliability Mixture"，
  `freeze_point = "after B0: no modification of the competition index, the CDF transform, the three input
  features, the mixing equation, the loss, the expert definitions, the static baseline, the metrics or the
  cross-backbone gates may be driven by B0 test mixture results"`，入库 3599369）；
  `amendment_v2m31.json`（amendment **V2-M3.1** "Cross-Backbone Confirmatory Evaluation"，
  `status = "FROZEN before the B1/B2 mixture run"`，frozen_utc 2026-10-01，入库 cedfa30）。
  `amendment_note` 逐字："approved by the preregistered M2.5 specialisation finding (SPECIALIST TRADEOFF
  PRESENT); new method amendment, not an LCR v1 patch."
classification: CONFIRMATORY（B1/B2）+ POST-HOC DEVELOPMENTAL（B0）——两个 label 均直接回读自 metadata.json
dataset: "RefCOCO+（V1/V2-M 冻结 cohort）"
candidate_protocol: "K=10 固定；severity m ∈ {0,2,4,8}；专家 = 冻结 random-only 与 curriculum E1b"
backbone: "B0（V1 冻结）+ B1 OpenCLIP B/16 + B2 SigLIP B/16（跨 backbone 确认）"
proposal_family: RPN
training_status: >
  零新定位训练；仅混合器参数（B0 runtime 1177.1 s，B1/B2 runtime 2654.3 s）。
  primary_comparison："AdaptiveMix vs StaticMix (does competition-dependent adaptation add value beyond a
  plain ensemble?)"
metrics_confirmatory: >
  `conf/m3_conf_audit.json`：`passed = true`、`n_pass = 0`、`stop = true`、label =
  "CONFIRMATORY RESULT — adaptive mixture not supported"。B1：`delta_macro_adaptive_minus_static` =
  **+0.00014802** CI [−5.55e-05, +3.56e-04]（跨 0）；extreme_safety `m0_pass = false` / `m8_pass = true`
  （threshold −0.003）；`selective_support.satisfied = false`（E-AURC relative improvement 0.000959、
  rer50_gain_pp 0.0283）。Frozen-hash 审计：`frozen_hashes_verified` 列出 5 项（mixer.py / bootstrap.py /
  protocol_m3.json / g4_protocol_freeze.json / run_v2m_m3_conf.py）；`max_deltas` 对 b1/b2 全部 14 份产物
  均为 **0.0**（含 `alpha_diagnostics.csv`）。
alpha_evidence: >
  自适应权重在确认段**确实随竞争变化**（因此不得写“机制塌缩”）：`alpha_m8_minus_m0` = B1 **+0.020568** /
  B2 **+0.054241**；`alpha_by_level` 均值：B1 从 m0 0.820827 升至 m4 0.838917，B2 从 0.816395 升至 0.853210；
  而在开发段 B0，alpha 几乎平坦（alpha_mean_by_m 0.568365 → 0.574854，`alpha_non_decreasing_in_m = true`）。
  静态混合器均值 `static_c_mean = 0.873009`，自适应 `adaptive_beta_mean = 2.780859`。
verdict: >
  **ADAPTIVE MIXTURE NOT SUPPORTED**；`n_pass = 0/2`（两个新 backbone 均未通过）。按 `amendment_v2m31.json.stop_rule`
  逐字："ADAPTIVE MIXTURE NOT SUPPORTED stops the entire M3 method line; the specialist trade-off (M2.5) and
  the LCR no-go results remain the frozen record."
correct_reading: >
  正确表述：自适应权重在 B1/B2 **did respond to competition**，但 **did not produce practically meaningful
  gains over StaticMix**（ΔMacroAUROC 量级 1e-4，CI 均跨 0）。**禁止**表述为
  "adaptive mechanism collapsed everywhere" / 「机制全面塌缩」——因为 B1/B2 的 alpha 确实变化。
  也禁止把 B0（developmental）的任何数字当作确认证据。
artifact_paths: >
  results/v2_local_competition/m3_mixture/{protocol_m3.json, amendment_v2m31.json,
  b0/metadata.json, b0/*, conf/metadata.json, conf/cross_backbone_verdict.json, conf/m3_conf_audit.json,
  conf/b1/, conf/b2/}
```

## V2-P｜proposal family 轴（按 lineage 分开登记，不得合并为一个模糊的 proposal robustness experiment）

> **全轴不变量**（适用于 P1-F4 / P1-F5 / P2-C1 / P2-M 每一段，逐字回读自
> `v2_p_program_summary.md` §1 与各 freeze artifact）：**0 new training parameters**（`new_training_parameters: 0`）、
> **同一冻结打分栈**（B3 `IndependentMLPScorer` seeds 1/2/3 + per-seed `global_T_corrected`，从不按 family 重拟合）、
> **同一 presented K**（nested K ∈ {5,10,20,50}，baseline K5，primary pair (5,50)）、
> **同一 candidate construction regime**（V1 seeded-random，seed 20260927，target slot 0；amendment **P1-A0**）、
> **同一候选供给量**（三族 bank 均 **N = 64** proposals/image）、同一 COCO 图像域。
> 只变 proposal family：RPN → DETR-R50 → Grounding DINO base。

```yaml
# ===== V2 post-A11 RESULT RECORD — V2-P1 F0–F3（bank 构建与工程可行性）=====
record_class: pre-result engineering / feasibility stage（不产生任何 confirmatory 结果）
experiment_id: v2p1-f0-f3-detr-bank-engineering-20261001-01
branch: v2-proposal-robustness
commit: >
  阶段提交链：`2249206` → `bcdc2cd` → `8a831f1`（amendment **P1-A0**：候选排序歧义澄清，
  `p1_amendment_history.csv` 逐字："pre-result clarification (candidate ordering)... no (disambiguates protocol
  wording only; candidate-construction regime unchanged, P1-F3 already ran --regime random)"）→ `1285e95`。
classification: engineering / feasibility（F0–F3 不是 confirmatory；`p1_f4_f5_config_freeze.json.preconditions`
  将 P1-F3 的输出作为**前置条件**引用，而非作为证据）
dataset: "RefCOCO+（与主线同一冻结 cohort）"
candidate_protocol: "DETR-R50 proposal bank，N=64，num_queries=100，box 转换与 sanitize 规则冻结于 p1_detr_r50/protocol.json"
backbone: "B0 冻结 OpenCLIP ViT-B/32（表示层不变；DETR 只作为 proposal generator）"
proposal_family: DETR-R50（新建 bank）
training_status: "零新定位训练参数；`p1_f3_new_training_parameters: 0`"
metrics: >
  `p1_detr_r50/engineering_probe.json.verdict = "OK"`（wall 39.2 s，peak VRAM 在预算内，scores_descending_ok）；
  `p1_detr_r50/proposal_summary.json.verdict = "FULL"`：ref-target recall@0.5 = **0.997335**
  CI [0.995100, 0.998552]、recall@0.7 = 0.984542、K50 candidate availability = **0.922441**（gate ≥ 0.90）；
  阈值分级 FULL ≥0.95 / GRAY [0.90,0.95) / LIMITED [0.80,0.90) / STOP <0.80。
  前置条件实测（`p1_final_summary.json.preconditions_from_frozen_config`）：`proposal_b_feasibility = FULL`、
  `route_f = ROUTE_F_FULLY_USABLE`、`p1_f3_detr_random_k5_accuracy = 0.8732`、`p1_f3_detr_r1_auroc = 0.8008`、
  `p1_f3_detr_e1b_auroc = 0.8092`、`p1_f3_confidence_collapse = false`、`p1_f3_frozen_identity = "PASS"`。
verdict: "F0–F3 = 工程/可行性判定（OK / FULL / ROUTE_F_FULLY_USABLE），无科学主张"
boundary_and_forbidden_claims: >
  禁止把 F0–F3 写成「DETR 上已复制」；它们只说明 DETR bank 足以进入后续 confirmatory gate。
  禁止用 proposal 质量差异倒推任何可靠性结论（`route_f` 只回答“能不能用”）。
artifact_paths: >
  results/v2_proposal_robustness/{p1_detr_r50/protocol.json, p1_detr_r50/engineering_probe.json,
  p1_detr_r50/proposal_summary.json, p1_detr_r50/recall_by_N.csv, p1_detr_r50/candidate_availability_by_N.csv,
  p1_detr_r50/same_category_availability.csv, p1_detr_r50/natural_omission.csv,
  p1_detr_r50/checkpoint_identity.json, p1_amendment_history.csv, inference_report.json}
```

```yaml
# ===== V2 post-A11 RESULT RECORD — V2-P1 P1-F4（C1 candidate-cardinality 复制）=====
record_class: pre-result freeze + result record
experiment_id: v2p1-f4-c1-proposal-family-cardinality-20261001-01
branch: v2-proposal-robustness
commit: "结果入库 = feb40d8 / 6496119；轴汇总入库 = 7441ab1"
frozen_config_commit: >
  `p1_f4_f5_config_freeze.json`，提交 **`1285e95`**（提交信息逐字："P1-F4/F5 config freeze + replication core +
  frozen inference (pre-result)"）；文件内 `classification = "CONFIRMATORY"`、`frozen_before_results = true`、
  `frozen_utc = 2026-10-01T00:00:00Z`，note 原文："Both F4 and F5 configurations are frozen here simultaneously,
  before any F4 number is produced (protocol section 1). F5 must not be edited after F4 results exist."
classification: CONFIRMATORY（直接回读自 freeze 与 f4_verdict.json）
dataset: "RefCOCO+ testA ∪ testB（与主线同一冻结 cohort）"
candidate_protocol: "nested seeded-random（P1-A0 适用）；primary cohort = common-K50（行间同一）；K ∈ {5,10,20,50}"
backbone: B0 冻结 OpenCLIP ViT-B/32
proposal_family: "RPN（frozen reference）+ DETR-R50"
training_status: "0 new training parameters（`f4_verdict.json.new_training_parameters = 0`）"
selection_calibration_eval: >
  无选择环节；calibration = per-seed 冻结 `global_T_corrected`（变体名 `variant = global_T_corrected`）；
  eval = common-K50 cohort：RPN 10,286 rows / DETR 9,665 rows（DETR 漏斗 10,601 → K20 10,570 → K50 9,665，
  k50_availability 0.9117；RPN 10,425 → 10,286，0.9867）
statistics: "image-cluster paired bootstrap，5000 reps / seed 0 / 0.95 CI"
metrics: >
  DETR：auc_drop_mean **0.08515703**（ci_low_mean 0.06059571，3/3 seed CI 排除 0）、eaurc worsening mean
  **4.554741**、rer50_drop_mean **0.294314**、rer80_drop_mean 0.318931；per-seed ΔAUROC 0.079892 / 0.089352 / 0.086227。
  RPN（同一栈重算，与 frozen artifacts 交叉校验）：auc_drop_mean 0.05169761（ci_low 0.03804034）、
  eaurc 2.256772、rer50_drop 0.465960；frozen 参考值 ΔAUROC 0.052161 / rer50 drop 0.455995。
  matched intersection（secondary，无 CI）：n=9,418，RPN 0.059251 vs DETR 0.081947。
verdict: >
  `c1_verdicts = {"RPN": "YES", "DETR": "YES"}`；`detr_C1_PROPOSAL_FAMILY_REPLICATED = "YES"`。
boundary_and_forbidden_claims: >
  不得读成「DETR 比 RPN 更差」（两族都退化，绝对精度反而 DETR 更高）；不得写任何因果主张；
  不得拿 matched-intersection 的 gap 当假设检验（freeze 定义为 secondary/descriptive，无 CI）。
artifact_paths: >
  results/v2_proposal_robustness/p1_f4_c1/{f4_verdict.json, c1_point.csv, c1_bootstrap.csv,
  c1_secondary_raw_auroc.csv, figures/}, p1_f4_f5_config_freeze.json, predictions/p1__{RPN,DETR}__*.npz
```

```yaml
# ===== V2 post-A11 RESULT RECORD — V2-P1 P1-F5（C4 hard-semantic 放大复制）=====
record_class: pre-result freeze + result record
experiment_id: v2p1-f5-c4-proposal-family-hard-semantic-20261001-01
branch: v2-proposal-robustness
commit: "结果入库 = 6496119；轴汇总入库 = 7441ab1"
frozen_config_commit: >
  与 P1-F4 **同一份** freeze（`p1_f4_f5_config_freeze.json` @ `1285e95`）：F4 与 F5 在任何 F4 数字产生前
  同时冻结，且 F5 不得在 F4 出数后修改（freeze note 原文）。因此 F5 属于真正的 pre-result freeze。
classification: CONFIRMATORY
dataset: "RefCOCO+ testA ∪ testB"
candidate_protocol: "matched Random-K5 / SameCategory-K5（仅 distractor composition 不同）"
backbone: B0 冻结 OpenCLIP ViT-B/32
proposal_family: "RPN + DETR-R50"
training_status: "0 new training parameters"
selection_calibration_eval: "同 P1-F4；C4 侧 cohort：RPN matched_rows 9,623（same_cat ≥4 availability 0.9231）/ DETR 8,376（0.7901）"
statistics: "image-clustered paired bootstrap 5000 / seed 0 / 0.95，hard vs random 共享 cluster draws"
metrics: >
  DETR（protocol："V2-G G4 item 9 (frozen), replayed on the proposal family"；comparison
  "stats_logistic -> e1b_stats_semantic on SameCat-K5 (matched random control)"）：
  Δhard mean **0.02129854**（ci_low 0.01764925）、Δrand mean 0.00661776、amplification mean
  **0.01468077**（ci_low 0.01077852）；pass_delta_hard / pass_amplification = true；n_seed_same_direction 3/3；
  manipulation（三指标规则）valid 3/3 seed。RPN 重算：Δhard 0.03173793 / amplification 0.03057344（ci_low 0.02649733）。
verdict: >
  两族 gate verdict 均为 `HARD_SEMANTIC_REPLICATED`；`detr_C4_PROPOSAL_FAMILY_REPLICATED = "YES"`；
  `rpn_C4_replication_reference = "YES"`；轴综合 `overall_p1_verdict.label = CORE_FINDINGS_PROPOSAL_FAMILY_ROBUST`
  （value YES；detail："candidate-cardinality degradation and hard-semantic amplification both replicate
  under DETR proposals."）
boundary_and_forbidden_claims: >
  same-category 为 **GT 辅助诊断构造**，不得写成自然场景；不得作因果解释（“semantic information causes…”）；
  不得把 P1 的 YES 外推到 GDINO（P1 只包含 RPN/DETR，GDINO 属 P2）。
artifact_paths: >
  results/v2_proposal_robustness/p1_f5_c4/{f5_verdict.json, c4_amplification.csv, c4_manipulation.csv,
  c4_selective.csv, figures/fig3_random_vs_hard_increment.png}, p1_final_summary.json,
  proposal_family_final_table.md
```

## V2-P2｜Grounding DINO（第三 proposal family，**C1-only** confirmatory）

```yaml
# ===== V2 post-A11 RESULT RECORD — V2-P2-C1（GDINO class-prompt 上的 C1 复制；C4 从未授权）=====
record_class: pre-result freeze + result record
experiment_id: v2p2-c1-gdino-third-family-cardinality-20261002-01
branch: v2-proposal-robustness
commit: >
  前置可行性审计入库 `9dd14e0`（`phase_b_feasibility_audit.json`，protocol "V2-P2-B0"，提交信息注明
  "no experiments run"）；工程 probe 入库 `75ce48d`（提交信息逐字："V2-P2-B1: GDINO E1 engineering probe -
  4/5 gates PASS, same-category supply gate FAILS"）；C1 verdict 入库 `7905aad`；轴汇总与台账入库 `9b010e6`。
frozen_config_commit: >
  `p2_c1_gdino_config_freeze.json`，提交 **`91f2758`**（提交信息逐字："V2-P2-C1: config freeze BEFORE any P2
  result - GDINO participates in C1 only (user-authorised protocol decision); C4/hard-regime for GDINO
  explicitly out of scope citing the frozen probe same-category supply failure"）；`frozen_utc =
  2026-10-02T00:00:00Z`；`protocol = "V2-P2-C1"`、`schema = "p2-config-freeze-v1"`；authorization 逐字：
  "User authorised 'GDINO participates in C1 only' (new protocol decision beyond the frozen audit on_failure
  rule) on 2026-10-02, BEFORE any P2 result exists. This file must be committed before any P2 numeric
  artifact is produced."
classification: >
  CONFIRMATORY（回读自 `p2_c1_verdict.json.classification`；同文件 protocol 字段逐字为
  "V2-P2-C1 (C1 replication on the GDINO class-prompt family; C4 out of scope)"）
dataset: "RefCOCO+ testA ∪ testB（与 V2-P1 同一冻结 cohort）"
candidate_protocol: "random（nested seeded-random，P1-A0 原文适用）；primary cohort = common-K50；K ∈ {5,10,20,50}"
backbone: B0 冻结 OpenCLIP ViT-B/32
proposal_family: "Grounding DINO base（class-prompt，第三族）；RPN / DETR 数值逐字复用 V2-P1 冻结产物"
training_status: >
  `new_training_parameters: 0`；freeze 的 `zero_fit` 逐字："0 new training parameters. B3 no retrain,
  temperature no refit, R1/E1b no refit, no new scorer, no NMS, no threshold change."；
  `immutable_reuse_from_v2_p1.note` 逐字："All analysis code below is IMPORTED VERBATIM from the frozen P1
  stack; P2 adds only plumbing (a FAMILIES entry + CLI choices), never a new metric or gate."
selection_calibration_eval: >
  无选择环节；calibration = per-seed 冻结 `global_T_corrected`（`variant = global_T_corrected`）；
  eval = GDINO common-K50 cohort **10,402 rows**（target-present 10,547 → K5/K10/K20 均 10,547、K50 10,402；
  `k50_availability` 0.9863；1,489 images / 3,695 refs）。**GDINO 不存在 same_category cohort**（冻结范围规定
  其不构建、不打分行、不出 verdict）。
statistics: "image-cluster paired bootstrap，5000 reps / seed 0 / 0.95 CI（paired = true）"
metrics: >
  GDINO `auc_drop_mean` **0.23659945**（`auc_drop_ci_low_mean` 0.21619267，`auc_all_seeds_ci_exclude_0` true）、
  `eaurc_worsening_mean` **8.65414442**、`rer50_drop_mean` **0.66781081**、`rer80_drop_mean` 0.38295881；
  per-seed ΔAUROC 0.23885029 / 0.24025433 / 0.23069372；冻结阈值 `route_a_auc_drop_min` 0.03 /
  `route_b_eaurc_worsen_min` 0.2 / `route_b_rer50_drop_min` 0.1，`route_a_passed` 与 `route_b_passed` 均 true，
  `replicated` true。复用完整性校验：`rpn_detr_reuse_verbatim_check = "PASSED (all frozen F4 point rows
  matched to 1e-12)"`。secondary intersections（**无 CI，descriptive，不得当假设检验**）：GDINO∪RPN
  n=10,125，RPN 0.05340577 vs GDINO 0.23122964（gap +0.1778）；GDINO∪DETR n=9,589，DETR 0.08625652 vs
  GDINO 0.22120686（gap +0.1350）。
verdict: >
  `c1_verdicts = {"RPN": "YES", "DETR": "YES", "GDINO": "YES"}`；
  `gdino_C1_PROPOSAL_FAMILY_REPLICATED = "YES"`；总表 label **`C1_REPLICATED_ON_THIRD_PROPOSAL_FAMILY`**
  （`proposal_family_final_table.md` 原文括号限定：**C1 only; C4 not tested**）。
boundary_and_forbidden_claims: >
  **本条只授权 C1。**禁止出现的措辞（experiment_log / protocol / 论文一律禁止）：`GDINO C4`、
  GDINO hard-semantic replication、GDINO same-category result、任何 GDINO 强竞争（hard-regime）主张。
  依据是可回读事实：`p2_gdino_probe/probe_report.json` 的 `same_category_ge4_frac` 实测
  **0.6892430278884463 < 阈值 0.85 → pass false**，整体 `verdict = "FAIL"`（其余四门通过：
  boxes_ge_64_share 1.0/0.98、ref_target_recall_05 0.9880478/0.95、candidate_availability_K50
  0.9920319/0.90、peak_vram_gb 2.631041/6.0），因此 C4 从未被授权、属 out of scope。
  freeze 原文："No GDINO same_category manifest is built, no hard_k5 job is scored, no C4 verdict may be
  computed or reported for GDINO. The probe number is final and never re-measured."
  verdict 原文 `out_of_scope_confirmed`："no GDINO same_category manifest, hard_k5 job, or C4 verdict
  exists or is reported"。同时禁止：任何「某个 detector 优于另一个」的主张；E2（query-conditioned GDINO
  top-K）或 E3（DINO-DETR / DETR-R101）作为 gate-bearing 实验；结果产生后修改 N、K levels、truncation、
  NMS、thresholds、checkpoints、scorer heads 或 metrics。约 2.8× 的放大**幅度**是确认性数字，
  但它的机制解读已被撤回（见下方 P2-A0）。
  stop_conditions 逐字："After the C1_GDINO verdict + secondary artifacts are committed, V2-P2 Phase B STOPS.
  Grounding DINO hard-regime, E3 families, and any fourth proposal family require a NEW authorized protocol."
artifact_paths: >
  results/v2_proposal_robustness/{phase_b_feasibility_audit.json, p2_gdino_probe/probe_report.json,
  p2_gdino_probe/checkpoint_identity.json, p2_gdino_bank/bank_extraction_stats.json,
  p2_gdino_bank/bank_extraction_report.md, p2_c1_gdino_config_freeze.json,
  p2_c1_gdino/p2_c1_verdict.json, p2_c1_gdino/c1_point.csv, p2_c1_gdino/c1_bootstrap.csv,
  p2_c1_gdino/c1_secondary_raw_auroc.csv, p2_c1_gdino/analysis_report.md,
  p2_c1_gdino/figures/fig1_k_vs_auroc_three_families.png,
  p2_c1_gdino/figures/fig2_k_vs_eaurc_three_families.png, proposal_family_final_table.md}
```

## V2-P2-M｜descriptive mechanism diagnostic（**不是** confirmatory mechanism proof）

```yaml
# ===== V2 post-A11 RESULT RECORD — V2-P2-M（composition vs count 机制诊断）=====
record_class: pre-result freeze + result record（descriptive diagnostic）
experiment_id: v2p2-m-composition-vs-count-mechanism-diagnostic-20261002-01
branch: v2-proposal-robustness
commit: >
  freeze 入库 **`87009ac`**（提交信息逐字："V2-P2-M: mechanism-diagnostic freeze BEFORE any P2-M number -
  composition vs count"）；verdict 与撤回文字入库 `0c60104`；台账与轴汇总入库 `9b010e6`。
frozen_config_commit: >
  `p2_m_mechanism_config_freeze.json` @ **`87009ac`**；`created_utc = 2026-10-02T08:20:00Z`、
  `authored_before_any_result = true`；四个 decision labels、R1 bin edges、0.50 shrinkage bar、
  2-of-3 families 要求与三条 stop conditions 全部写在任何 P2-M 数字存在之前。
classification: >
  **`DESCRIPTIVE_MECHANISM`**（回读自 freeze 与 `p2_m_verdict.json.classification`）。
  `classification_meaning` 逐字："This protocol may not create, strengthen or weaken any existence claim.
  C1's existence is already settled by V2-P1/V2-P2-C1 (confirmatory). P2-M only asks which measured property
  of the candidate pool co-varies with the measured harm." 因此本条**不是** confirmatory mechanism proof。
dataset: "RefCOCO+ testA ∪ testB；三族同一冻结 cohort（RPN 10,286 / DETR 9,665 / GDINO 10,402 expressions）"
candidate_protocol: "只读取已冻结的预测列与已有 primitive；不新建 manifest / proposal bank / seed"
backbone: B0 冻结 OpenCLIP ViT-B/32
proposal_family: "RPN + DETR-R50 + Grounding DINO（三族同比较；GDINO 侧仍只涉及 C1 regime）"
training_status: >
  `new_training_parameters: 0`、**`model_forward_passes: 0`**、`wall_seconds` 330.28。freeze `forbidden`
  逐字包含 "any model forward pass, any detector/CLIP/B3 inference, any GPU use" 与
  "any fit / train / calibration / threshold tuning / refit (temperature, R1, E1b stay frozen)" 与
  "any new scorer, any new feature function: every measured quantity must be an existing frozen primitive
  or an existing stored prediction column"。
selection_calibration_eval: >
  无选择/校准环节；`unknown_target_share` 三族均为 0.0，故 R1 规则未被阻塞（`blocked_by_unknown_share = []`）；
  R1 quintile 无任何 bin 低于冻结的 200 行下限（`bins_dropped = 0`）。
statistics: >
  沿用冻结 bootstrap convention（image-cluster sampler，5000 reps / seed 0 / 0.95）；estimator 逐字为
  "the replicate resamples images and re-runs the point functional (mean over B3 seeds of the within-family
  per-seed Spearman rho)"；shrinkage ratio **无 CI**（freeze 禁止把它提升为检验）。
metrics: >
  rule (a) 族内剂量反应 rho(H1c, R1)：RPN **-0.014419** [-0.038327, +0.009518]、DETR **+0.089424**
  [0.061096, 0.119124]、GDINO **+0.052527** [0.024218, 0.080763]；`families_passing = [DETR, GDINO]`、
  `required = 2` → `passed = true`。
  rule (b) R1 五分位匹配（`bin_edges_R1 = [3,5,9,14]`）：unmatched gap GDINO−RPN 0.041035 /
  GDINO−DETR 0.105886；matched gap 0.048427 / 0.112817；shrinkage **−0.180145** 与 **−0.065458**
  （`shrinkage_min = 0.50`）→ `passed = false`。
  族均值（`mechanism_report.md` §2）：R1 = 11.51 (RPN) / 8.71 (DETR) / **6.23 (GDINO，最低)**；
  R2 unmatched fraction = 0.564 / 0.611 / **0.718（最高）**；R4 pair-IoU>.7 = 0.0014 / 0.0297 / 0.0094；
  H1c net harm = 0.3605 / 0.2957 / **0.4016（最高）**；GDINO 内 rho(H1c, R2) = +0.0036 [-0.024, +0.032]
  （与 0 不可区分）。
verdict: >
  **`MECHANISM_PARTIAL`**（decision_labels 定义："only (a) holds"）。两条实际结论：
  (1) 被测量的同类冗余通道**不解释**跨 proposal family 的放大差异
  （same-class redundancy does not explain the between-family amplification **under the tested diagnostic**）；
  (2) candidate **count** 被**算术**排除而非被实验排除：三族 bank 均恰好 64 proposals/image，
  且全部在同一 presented K 下评估（N=64 与 presented K 固定）。
boundary_and_forbidden_claims: >
  禁止升级为 "composition in general is ruled out"；禁止写 "the mechanism is explained"；禁止任何因果措辞
  （freeze `out_of_scope` 逐字："any causal language: this diagnostic is associational within family and
  stratified across families"）；禁止「某检测器优于另一检测器」；禁止对 GDINO 的任何 C4 / hard-regime 措辞
  （逐字："any C4 / hard-regime statement about GDINO (the B0 probe failed the same-category supply gate at
  0.6892 vs 0.85; GDINO stays C1-only by user authorisation)"）。`honesty_clause` 逐字："if
  MECHANISM_NOT_SUPPORTED or MECHANISM_GAP_PERSISTS, the report must state that the composition account is NOT
  established by this diagnostic and must not be used to explain the P2-C1 amplification; the P2-C1 numbers
  stand either way, since they are confirmatory and independent of this file"。
  `note_on_binary_harm` 要求：H1a/H1c 按构造是离散指标，对它的秩相关应按 rank-biserial 关联解读，
  **不得**当作连续函数形式的证据。GDINO 的高 R2（unmatched 份额）只是**尚未被测量的候选观察**，
  report §6 明确本诊断不能称其为机制。
artifact_paths: >
  results/v2_proposal_robustness/{p2_m_mechanism_config_freeze.json, p2_m_mechanism/p2_m_verdict.json,
  p2_m_mechanism/mechanism_point.csv, p2_m_mechanism/mechanism_association.csv,
  p2_m_mechanism/mechanism_strata.csv, p2_m_mechanism/mechanism_report.md, v2_p_program_summary.md}
```

## P2-A0｜post-result interpretation correction（**WITHDRAWN**）

```yaml
# ===== V2 post-A11 AMENDMENT RECORD — P2-A0（解释层撤回；不是 result tampering，也不是预注册）=====
record_class: post-result correction（prose / mechanistic interpretation only）
amendment_id: P2-A0
branch: v2-proposal-robustness
commit: >
  撤回文字随诊断结果入库 **`0c60104`**；正式台账行（`results/v2_proposal_robustness/p2_amendment_history.csv`）
  与轴汇总索引入库 **`9b010e6`**；material bank 同步入库 **`92f09d0`**。执行更正所依据的测量（V2-P2-M）
  本身在任何 P2-M 数字之前冻结（`87009ac`）。
classification: >
  台账 `classification` 逐字："post-result interpretation correction (prose only; no metric, threshold,
  gate, cohort or verdict label touched)"。因此本条是 **prose/mechanistic interpretation correction**，
  **不是** result tampering。
target_scope: >
  台账逐字："V2-P2-C1 p2_c1_gdino/analysis_report.md section 5 mechanistic reading;
  proposal_family_final_table.md closing paragraph"。
withdrawn_claim: >
  旧解释（状态：**WITHDRAWN**）：「GDINO 最强的放大由同类语义竞争解释」。其原文为
  `p2_c1_gdino/analysis_report.md` §5："the class-prompt bank fills its 64 slots with semantically
  redundant, same-class boxes ..."，并称约 2.8× 放大是
  "the strongest evidence in the program so far for the semantic-competition mechanism"。
  `mechanism_report.md` §6 以 "**Withdrawn**" 记录该撤回，并在原 §5 追加 addendum、修正总表结尾段。
replacement_safe_wording: >
  新的安全表述：**the measured same-class redundancy channel does not explain the between-family
  amplification**（限定于 the tested diagnostic）。不得改写为「composition 被整体排除」，也不得宣布任何
  替代机制。
unchanged_facts: >
  **all confirmatory C1/C4 numbers unchanged；all gates unchanged；all thresholds unchanged。**
  台账 `modified_original_gates = no`；`gate_frozen_before_results` 逐字："n/a - this item corrects an
  interpretation, not a gate; the correcting measurement (V2-P2-M) was itself frozen before any P2-M number
  existed (commit 87009ac)"。任何 ΔAUROC / E-AURC / RER / cohort / verdict label 均未被重算或改动。
still_supported: >
  candidate **count** 的排除仍然成立，且性质是算术而非测量；台账逐字：
  "The candidate-COUNT exclusion is NOT withdrawn (all three banks hold exactly 64 proposals and are
  evaluated at the same presented K - arithmetic, not measurement)."
open_question_consequence: >
  GDINO 放大的机制因此仍是**开放问题**（RQ4）。旧候选解释被否证**不构成**新解释；任何后续机制实验必须
  自带新的结果前冻结协议与判据。
artifact_paths: >
  results/v2_proposal_robustness/{p2_amendment_history.csv, p2_m_mechanism/mechanism_report.md,
  p2_c1_gdino/analysis_report.md, proposal_family_final_table.md, v2_p_program_summary.md}
```

## 本节登记完毕（本轮不新增任何实验）

> 本节全部条目均为 **append-only 回顾性结果登记**：不重跑任何推理、不新增任何 bootstrap / 统计检验 /
> 机制诊断 / 结果数字，不追溯预注册任何已完成实验。`docs/final_result_summary.md` 与
> `results/v2_proposal_robustness/v2_p_program_summary.md` 的既有数字在本节中只被**引用**，未被改写。
> **RQ4（GDINO 放大的机制）此刻只登记为 open question：NOT STARTED、NOT AUTHORIZED**，
> 详见 `docs/research_protocol.md` 的「V2 Post-A11 Program Result Record」与「Program status」。

# RQ4-M1（Transition–Confidence Decomposition）—— pre-result freeze record

> 本节是**结果前登记**：到本节入库为止，**未计算任何真实 RPN / DETR / GDINO 的 RQ4-M1 数字**。
> 它不新增任何 V2 / V2-P confirmatory 结果，也不修改任何已发布数字、gate 或 threshold。

```yaml
# ===== RQ4-M1 pre-result freeze record =====
record_class: pre-result freeze（mechanism decomposition line, not an intervention test）
protocol_id: RQ4-M1
amendment_id: RQ4-A1（见 docs/research_protocol.md 第 11 节）
classification: DESCRIPTIVE_MECHANISM_DECOMPOSITION
real_results_seen_before_freeze: false
new_training_parameters: 0
new_model_forward: 0
new_features: 0
branch: v2-rq4-mechanism
parent_branch_and_commit: v2-proposal-robustness @ 42fe6ad（五轴 CLOSED 保持不动；不 merge main）
program_status_delta: >
  V2-G / V2-D / V2-M / V2-M3 / V2-P 仍为 CLOSED。RQ4 由 OPEN / NOT STARTED / NOT AUTHORIZED 转为
  ACTIVE，但**仅覆盖 RQ4-M1 这一条 exact decomposition**；RQ4-M2 与任何新机制搜索 / 新干预未获授权。
frozen_inputs: >
  只读已冻结的三族 common-K50 cohort 上的 K5 / K50 `correct` 与 `conf_msp`（= frozen
  global_T_corrected），3 个 B3 seed，以及 published C1 参考表。禁：new forward / new feature /
  new candidate generation / new training / temperature refit / recalibration / new family / new
  dataset / new cohort。0 new parameters，0 GPU。
input_artifact_manifest: >
  results/v2_rq4_mechanism/m1_transition_confidence/input_artifact_manifest.csv —— 18 个 prediction
  artifact + published C1 表的 sha256 / file_size 清单（freeze 轮只做文件 metadata 与 hash，不加载
  prediction array，因此未违反「结果前不读 real row」）；row_count 来自已发布的 `c1_point.csv` `n` 列，
  npz 自身行数 deferred。结果轮由 verify_input_manifest() 重新逐条 hash，任一 drift 即中止：
  all real analyses must consume exactly these hashed artifacts。
frozen_formulas: >
  A00=AUROC(r5,p5) / A10=AUROC(r50,p5) / A01=AUROC(r5,p50) / A11=AUROC(r50,p50)；
  D_total=A00-A11；L=0.5[(A10-A00)+(A11-A01)]；C=0.5[(A01-A00)+(A11-A10)]；
  恒等式 L+C=A11-A00 与 D_label+D_conf=D_total，residual 要求 <= 1e-12；A00 / A11 必须复现 published
  C1 K5 / K50 AUROC（否则 STOP，不放宽）。D_label / D_conf 保持**有符号**，不 clip、不归一化、不强凑 100%；
  |D_total| < 1e-3 时不报 share。
  D_conf 的解释被收紧为 confidence-**ranking**-change（labels 固定在 Shapley 角上时跨表达式置信排序
  变化 induced 的 AUROC 改变），**明确不是** temperature / calibration-scale change（AUROC 对单调分数
  变换不变）；公式本身未改。
groups_and_invariant: >
  S=(1,1) / F=(1,0) / E=(0,0) / G=(0,1)；分组穷尽且互斥；**G=0 作为 runtime invariant**，任一 family/seed
  出现 G>0 即 STRUCTURAL_INVARIANT_FAILURE 并停止分析（不做补救）。
audit_log_before_commit: >
  commit 前的 protocol audit 包四个修正：(A) tail-pressure 状态由 duplicate-of-existing 表述改为
  NOT USED + 正确原因（frozen artifacts 缺 per-candidate raw scores，计算需新的 model forward，被禁止；
  不可计算不等于已有等价量）；(B) 新增 frozen input-artifact manifest；(C) D_conf 解释收紧为 ranking
  change；(D) cross-family own-cohort CI 改名为 descriptive under independent-family resampling，
  明确不是 paired-expression CI（paired secondary 只能来自 matched intersection + shared cluster draws）。
  四项均只改文字 / 新增清单，不改公式、estimator、cohort、coverage、threshold 或 verdict 词汇。
planned_outputs: >
  results/v2_rq4_mechanism/m1_transition_confidence/：protocol_freeze.json,
  input_artifact_manifest.csv, point_decomposition.csv, bootstrap_decomposition.csv,
  transition_groups.csv, pairwise_auc_components.csv, selective_error_sources.csv,
  cross_family_gap_decomposition.csv, matched_intersection_decomposition.csv, verdict.json,
  metadata.json, report.md, figures/（恰好三张）。
statistics_frozen: image-cluster bootstrap，5000 reps，seed 0，95% percentile CI；family 内 shared draws；
  cross-family primary = 各 family 自己的 cohort 独立重采样 + replicate-index-aligned descriptive gap CI；
  secondary = 三个 matched-expression intersection 上共享 draws 的 paired 分解（SECONDARY_MATCHED_DIAGNOSTIC）。
forbidden_wording_now: >
  「X causes the degradation」/「we discovered the true mechanism」/「unmatched boxes、H2a、semantic
  ambiguity is the cause」。允许的表述上限：the observed AUROC degradation can be exactly decomposed …；
  the larger GDINO degradation is statistically accounted for more by X than by Y in this decomposition。
next_allowed_step: >
  full pytest（0 failed / 0 errors）-> commit "Freeze RQ4-M1 transition-confidence decomposition" -> L3 ->
  push v2-rq4-mechanism；然后才运行 scripts/rq4_m1_decomposition.py --allow-real-data。结果轮禁止修改
  公式 / bootstrap / family pair / share 定义 / group 定义 / verdict wording，禁止新增机制变量。
```

# RQ4-M1（Transition–Confidence Decomposition）—— result record

> 本节是**结果后登记**：RQ4-M1 真实分解已在 freeze commit **`c68817b`** 推送之后才计算，
> 数字全部来自 `results/v2_rq4_mechanism/m1_transition_confidence/`，未改写任何已冻结公式。

```yaml
# ===== RQ4-M1 result record（描述性机制分解；不是因果检验）=====
record_class: result record（descriptive mechanism decomposition, computed after the freeze push）
protocol_id: RQ4-M1
freeze_commit: c68817b（"Freeze RQ4-M1 transition-confidence decomposition"，已 push v2-rq4-mechanism）
classification: DESCRIPTIVE_MECHANISM_DECOMPOSITION
real_results_seen_before_freeze: false
new_training_parameters: 0
new_model_forward: 0
new_features: 0
gpu_used: false
temperature_refit: 0
runtime: >
  真实分解 wall = 791.89s，CPU only；bootstrap = image-cluster 5000 reps / seed 0 / 95% percentile；
  输入 18 个冻结 prediction artifact + published C1 表，结果轮开头由 verify_input_manifest()
  逐条重哈希（19/19 通过，含 sha256 与 file_size），任一 drift 即中止。
identity_status: >
  max |Shapley residual| = 0.0；max |group-weight reconstruction residual| = 1.11e-16（容差 1e-12）；
  A00 / A11 逐位复现 published C1 的 K5 / K50 AUROC；G == 0 对三族三种子全部成立；
  stop_conditions_triggered = []，forbidden_labels_used = []；verdict = DECOMPOSITION_REPORTED。
headline_numbers: >
  三族 ordering 全部为 CONFIDENCE_CHANGE_HEAVIER（3-seed mean，D_total / D_label / D_conf）：
  RPN +0.051698 / +0.002605 / +0.049093；DETR +0.085157 / +0.006166 / +0.078991；
  GDINO +0.236599 / **-0.009145** / +0.245744。GDINO 的 correctness-transition 分量为**负**
  （该因子在此角上 improving reliability），全部退化由 confidence **ranking** 变化承载。
cross_family_gap: >
  GDINO-RPN gap D_total = +0.184902，拆为 label 分量 -0.011750 + confidence 分量 +0.196651
  （residual 2.8e-17）；GDINO-DETR = +0.151442 = -0.015311 + +0.166753；
  DETR-RPN = +0.033459 = +0.003561 + +0.029898。三个比较的 heavier component 均为 confidence change。
  这些是各族自己 cohort 上独立重采样、按 replicate index 对齐的 **descriptive** CI（不是
  paired-expression CI）：GDINO-RPN gap CI [0.161441, 0.207721]，confidence 分量 CI
  [0.176015, 0.218443]，label 分量 CI [-0.029831, 0.005643] 含 0。
matched_secondary: >
  secondary = matched-expression intersection + **shared** image-cluster draws（真 paired）：
  GDINO∩RPN（10125 表达式）paired gap +0.177824 CI [0.155407, 0.200415]，label -0.011598
  CI [-0.028824, 0.005678]，conf +0.189421 CI [0.169174, 0.210470]；GDINO∩DETR（9589）
  +0.134950 CI [0.107595, 0.162878]；DETR∩RPN（9418）+0.022697 CI [-0.004215, 0.048774]。
  结论：cohort 匹配后同一 ordering 存活（不覆盖 primary）。
diagnostics: >
  S/F/E 组成（seed1）：RPN 4373/3727/2186、DETR 5647/2873/1145、GDINO 4635/4166/1601，G=0。
  AUC(S,F;p50) vs AUC(S,E;p50)：RPN 0.763 / 0.837；DETR 0.737 / 0.658；GDINO 0.593 / 0.581
  —— GDINO 上新引入错误在 K50 仍保留接近随机的置信排序。selective@50% 接受错误中来自 F 的比例：
  RPN≈0.73、DETR≈0.63、GDINO≈0.71。
implementation_fix_in_result_round: >
  第一次真实运行在完成全部计算、渲染 fig2 时因 matplotlib Line2D 无 get_xpos() 而崩溃（未写任何
  artifact）。修复**仅**涉及渲染层：fig2 中位线标记改用 get_xdata()；数值 artifact 改为先于 figures
  写出；补齐 freeze 中已列出但驱动未实现的 report.md 输出；新增 rendering-tier 测试（纯 synthetic）
  使该类缺陷在 pytest 中暴露。公式 / bootstrap / family pair / share 定义 / group 定义 /
  verdict wording **零改动**，未新增机制变量；修复后重跑的三族点估计与首次运行逐位一致。
answer_to_questions: >
  RQ4-M1-Q1：三族的 K5→K50 退化在此精确分解中均由 confidence-ranking-change 分量承载为主，
  correctness-transition 分量近零（GDINO 为负）。RQ4-M1-Q2：GDINO 相对 RPN / DETR 的额外退化
  在该分解中几乎全部由 confidence 分量 accounted for（gap 的 conf 分量 +0.1967 / +0.1668，
  label 分量为负且 CI 含 0）。措辞上限："accounts for in the exact statistical decomposition"。
forbidden_wording_still: >
  不得写 X causes the degradation / we discovered the true mechanism / unmatched boxes、H2a、
  semantic ambiguity is the cause；不得把上述描述性分解当作干预证据。
tests_and_suite: >
  结果轮后 full pytest = 1107 tests / 0 failed / 0 errors / 2 skipped（RQ4-M1 的 7 个
  artifact-tier 测试由 skip 转为实际执行并通过）。
artifact_paths: >
  results/v2_rq4_mechanism/m1_transition_confidence/{protocol_freeze.json,
  input_artifact_manifest.csv, point_decomposition.csv, bootstrap_decomposition.csv,
  transition_groups.csv, pairwise_auc_components.csv, selective_error_sources.csv,
  cross_family_gap_decomposition.csv, matched_intersection_decomposition.csv, verdict.json,
  metadata.json, report.md, figures/fig1..fig3.png}
program_status_after_this: >
  RQ4-M1 COMPLETE。RQ4 到此停止：RQ4-M2、新 candidate-property search、新 detector / dataset /
  model、任何 intervention / ablation / causal test 均**未获授权**。V2-G/D/M/M3/P 仍 CLOSED，
  其 confirmatory 数字、gate、threshold 未被触碰。
```

## CLOSE-OUT RECORD — RQ4-M1 材料库关账与实验程序冻结（documentation/provenance only）

```yaml
date_utc: 2026-10-02
status: DONE
kind: documentation / provenance only
classification: NO_NEW_RESULT（本轮不跑 GPU、不跑 bootstrap、不新增任何统计量、无 model forward）
what_changed: >
  docs/final_result_summary.md（材料库）新增 RQ4-M1 小节：headline 表（三族 D_total / D_label /
  D_conf 与 A00→A11）、允许的论文措辞（含非因果边界句）、flip-rate 悖论说明、
  matched-expression secondary 支持、bootstrap 区间转录、RQ4 状态由「open / not started」改为
  **RQ4-M1 COMPLETED**（仍开放：what causally produces the confidence-ranking change，标为
  OUT OF SCOPE FOR THIS PAPER / NO RQ4-M2 PLANNED）；新增第 4 节 program status 块。
  docs/research_protocol.md 只以 append-only 方式追加同一 program status 块与去向声明。
source_of_numbers: >
  全部 RQ4-M1 数字从 results/v2_rq4_mechanism/m1_transition_confidence/{verdict.json,
  point_decomposition.csv, bootstrap_decomposition.csv, cross_family_gap_decomposition.csv,
  matched_intersection_decomposition.csv, transition_groups.csv, report.md} 转录；材料库不做任何
  重算，也不把数字发明为记忆值。本轮校对发现并当场修正了两处从记忆写入的错误：
  GDINO 的 D_label bootstrap 区间实际为 [-0.022816, +0.004701]（**含 0**，因此不得称其显著为负），
  以及 GDINO 的 F 组占比实为 40.0% / 40.5% / 40.0%（非早期笔记中的 22.6%）。
provenance_notes: >
  (1) matched 的 paired gap 只存在于 report.md §6（matched_intersection_decomposition.csv 的 gap 列
  为空），材料库已标明取数出处。(2) 上一轮删除的 run_summary.txt 是早期原型留下的
  untracked 陈旧文件，**不**在 freeze 的 outputs 列表内，也不是本驱动的产物；freeze 列出的
  12 个 artifact + 3 张图与实际落盘逐名一致。(3) results/final_registry/ **不重新生成**，
  V2 与 RQ4-M1 的值继续从 per-axis artifact 取。
  (4) 更正：上一轮结果记录里写的「结果轮后 full pytest = 1107」是在**追加该记录之前**跑的，
  追加后未重跑就提交了；本轮重跑暴露出两处由该记录自身的文本引起的测试失败：
  ① 冻结文档无数字扫描以「RQ4-M1（」为锚点到文件末尾，因而覆到了允许写数字的结果记录
  （修法：扫描在结果记录的 append-only 边界处截止）；② 结果记录的 artifact_paths 用了
  `figures/fig1..fig3.png` 压缩写法，存在性检查把它当作字面路径（真实文件名带描述后缀：
  fig1_signed_decomposition / fig2_transition_group_confidence / fig3_cross_family_gap_components）。
  修法：校验器把同前缀数字区间展开为逐索引前缀，并要求每个索引至少命中一个已提交文件，
  所以区间内少一张图仍会报错。两项均只改测试，不改任何数字、公式、gate 或 threshold，
  也不改写已提交的结果记录正文（append-only）。
tests_and_suite: >
  本轮新增 tests/test_rq4_m1_material_bank.py（15 项：材料库数字/标签与 artifact 反向校验、
  非因果边界、program frozen、final_registry 未被重生）；修正
  tests/test_rq4_m1_decomposition.py 的冻结文档切片边界与 tests/test_v2_governance_closeout.py
  的 artifact 路径展开。本轮 close-out 后 full pytest = 1122 tests / 0 failed / 0 errors / 2 skipped。
tests_added: >
  tests/test_rq4_m1_material_bank.py：材料库的 headline 数字、verdict 与 ordering 标签、
  D_total = D_label + D_conf、bootstrap 区间的符号、flip-rate 注记的 F 组占比、matched gap 与
  matched artifact 分量的一致性，均从 verdict.json / point_decomposition.csv 等 artifact 读出后
  反向校验材料库（不做独立硬编码）；并校验非因果边界句、RQ4-M1 CLOSED、
  EXPERIMENTAL PROGRAM FROZEN / No active experiment.、NO RQ4-M2 PLANNED 与 final_registry 未被触碰。
historical_records_untouched: >
  RQ4-A1 之前写为 NOT STARTED / NOT AUTHORIZED 的历史协议记录与台账**不修改**；
  两份治理文档均为严格 append-only（由 test_governance_docs_are_strictly_append_only_at_head 强制）。
program_status_after_this: >
  V2-G CLOSED / V2-D CLOSED / V2-M CLOSED / V2-M3 CLOSED / V2-P CLOSED / RQ4-M1 CLOSED；
  EXPERIMENTAL PROGRAM FROZEN；No active experiment。后续均为论文写作（A 阶段）与 future work。
```



# Research Repair v1 — execution and interpretation correction (2026-10-03)

This section is a retrospective repair record. It does not retroactively preregister
completed experiments. The user authorized this result-driven supplementary analysis,
three gpt-6-luna/max execution agents, necessary corrections/retraining, and overnight
execution. Full protocol: reviews/repair_protocol.md; handoff:
reviews/repair_execution_handoff.md. Historical sections above retain their original
wording and dates. Current scientific interpretation follows repaired artifacts.

## Identity and execution

Baseline commit: 07e15a66056436fcecb880c5aa2d75354beefe26. Baseline manifest:
results/research_repair_v1/input_manifest.json. Initial verification passed for all
794 inputs (9400891451 bytes). Actual task identities: /root/a_statistics,
/root/b_information, /root/c_candidates; each was successfully spawned with
model=gpt-6-luna, reasoning_effort=max, fork_turns=none and acknowledged startup.
Preparation and dispatch do not imply experimental completion. Only STATUS.json
with formal outputs and acceptance evidence can establish repair completion.

## Scientific interpretation corrections

E-AURC remains accuracy-dependent: in the population random-ranking limit,
E-AURC_random=-a ln(a). RER's oracle ceiling at coverage c is 1 when a>=c,
otherwise a(1-c)/(c(1-a)). They are selective utility measures, not pure
accuracy-independent discrimination evidence. AUROC is separate evidence.
Logits temperature preserves argmax but need not preserve maximum-softmax
confidence ordering across expressions; the former scale-exclusion inference is
withdrawn. Direct native/global-T/per-K diagnostics supersede that inference.
Mean CI endpoints across seeds are not a CI for the mean effect. Shared image
resampling must compute each fixed seed's effect, then average effects inside each
replicate. Relative/DoD/severity-macro statistics follow the same rule.
Four-corner identities are statistical accounting, not causal identification.
Small net Shapley label components can cancel large opposing path contributions.
H1c is accuracy harm and cannot explain AUROC gaps by identity. The measured
redundancy explanation did not receive support; broader semantic explanations
were not falsified. Unmatched proposals are unmatched to the annotations used,
not necessarily empty background. Proposal count is not distinct-object count.

## Final question and evidence organization

RQ1: expansion effects on accuracy/discrimination/calibration/selective utility.
RQ2: score extrapolation and query–crop versus candidate–candidate sources.
RQ3: four-corner accounting, paths, interaction and interpretation limits.
Historical RQ/gate numbers remain source identifiers in the appendix. Target
absence is outside this paper and repair. V2-G freezes encoders but trains heads;
V2-P and historical accounting reuse scorers; V2-M/M3 train or fit models.
Backbone and proposal axes are separate, not a full factorial design. Strict
RefCOCOg is image-disjoint within COCO, not a new visual domain.
Full pre-result configuration freezing differs from input identity freezing;
D1/D2 have limited evidence for a complete pre-result configuration, and D2
includes a target redefinition. Staged later protocols do not erase adaptivity
from seeing earlier test results. This repair is not independent confirmation.
Historical gates remain operational rules with their original thresholds.

## Citation and current-document correction

TransVG is Deng et al., ICCV 2021; MMCE is Kumar/Sarawagi/Jain, ICML 2018;
Ovadia's uncertainty-under-dataset-shift paper is NeurIPS 2019; ReCLIP uses
cropping/blurring region scoring and a spatial relation component. Primary
sources and the selective-metric reference are linked in docs/literature_notes.md.
Pre-repair summary and blueprint are preserved under docs/appendix/. Current
README/summary/blueprint are reorganized by questions and evidence dependencies;
history, labels, gates, and hashes remain in the appendix and registry.

## Newly reproduced implementation issue (pending formal results)

B reproduced a log-K standardization bug in the original Phase05 ScoreDeepSets
feature builder: the fit-row prefix over concatenated K arrays was not the actual
training mask. The minimal regression fails before the correction (mean logK
1.6094, SD zero in the test) and should pass with actual training rows (mean
1.9560, SD 0.3466). This is a train/tune isolation error. The minimal source fix
and corrected-model experiments are distinct from frozen-baseline restoration.
Old predictions/models/results remain byte-identical; new outputs are versioned
in information/. Failure/pass logs and formal result status will be recorded
when available. No original recovery anchor tolerance is relaxed.


## Repair input-coverage correction

The initial preparation manifest enumerated
`data/raw/mscoco/annotations/instances_train2014.json`, which is absent in this
checkout. The actual annotation file is
`data/raw/annotations/instances_train2014.json`. The 794-file initial check remains
valid for its recorded inputs; it must not be represented as covering every later
recovery/audit input. Before formal use, missing source inputs receive immutable
`supplemental_input_manifest.json` snapshots under the owning repair axis. Existing
snapshots are verified, not replaced. Final acceptance verifies the original baseline
and all supplements via scripts/verify_repair_inputs.py. No old input is rewritten.


## Repair execution: constrained recovery and retained failures

B's old Logistic refit failed its unchanged 1e-9 prediction anchor under both
BLAS2 (reported max difference 6.066e-8) and an explicitly scoped BLAS1 check
(reported 3.642e-8 for seed1). The recovery is UNVERIFIABLE under the allowed
resource environment. The new CPU float32 experiments continue with BLAS/Torch2,
while existing historical prediction artifacts remain available for direct canonical
row/point checks. No tolerance, architecture, hyperparameter grid or precision was
changed to make an anchor pass. This refit failure is distinct from the corrected
log-K feature builder and from validity of existing stored historical predictions.

C retained two audit failures while opening existing output CSVs for writing.
Independent same-directory, same-size/attribute copy probes did not reproduce the
EINVAL failure; the underlying cause is unconfirmed. Atomic CSV writes now preserve
the prior output on replacement failure. The two partial runs do not constitute a
completed audit; a third audit is authorized after writer tests, in the heavy audit
slot while A investigates a separate point-anchor mismatch.

A completed and stored 5000-draw Phase0A temperature jobs before its next anchor
comparison failed (reported old 0.534881484687, new 0.532179661676, tolerance1e-6).
Those completed jobs remain usable; no remaining scope is marked complete based
on their presence. Point/row/definition alignment is investigated without widening
anchors. All failed-attempt evidence remains under the corresponding repair axis.


## Repair cohort-definition correction

A traced the failed pooled anchor to different row universes: legacy Phase0A
`__pooled__` includes the entire common cohort (20,799 rows), whereas the new
`__pooled_test__` uses only testA+testB (10,286 rows, 1,490 images). Their K5
accuracies, 0.534881484687 and 0.532179661676 respectively, must not be compared
as a recovery anchor. The test-only pool remains an explicitly new supplementary
cohort without that historical anchor. Any historical headline/gate using the
all-common pool must retain its original mask for audit, be labelled as including
train/validation, and remain separate from test evidence. Other scope adapters
must check their actual pooled definitions rather than infer them from the name.


## Repair pooled-cohort clarification (actual row counts)

The prior execution note incorrectly described Phase0A's all-common pool as
including training rows. The actual stored split counts are val_select=5,282,
val_calib=5,231, testA=5,646 and testB=4,640; these sum to 20,799, with no
training evaluation rows. Its correct provenance label is
ALL_COMMON_INCLUDES_VALIDATION_NO_TRAIN. The original pool still mixes validation
and test evaluation; testA+testB remains a distinct supplementary 10,286-row pool.
This clarification supersedes the earlier guessed training-membership description
without modifying the historical log prefix or any numerical anchor.

A previously exported frozen Stats/E1b bundle was located at
results/phase1e_refcocog_external/frozen_models/models.json. Direct coefficient
loading and strict verification now take precedence over unnecessary Logistic
refitting. The recorded refit failures remain failures of that restoration route;
they do not establish that direct frozen-model loading is impossible. Its actual
verification outcome is pending and will be recorded separately.


## Direct frozen-bundle verification and scope

Direct loading of the original A11 coefficient bundle succeeded at BLAS2, retaining
the original 1e-9 tolerance. Evidence: information/reference_frozen_bundle_verification.json
(three seeds; Stats prediction errors 2.220446049250313e-16; E1b coefficient, Stats
normalization and temperature differences zero; 42 source-artifact identities).
This is distinct from the preserved BLAS1/2 refit failures. E1b end-to-end prediction
with newly reconstructed cell features is a separate check: B's compatibility check
found a large discrepancy (reported 0.3817), so that path is not yet verified. The
feature/order/precision construction is being compared against the original Phase1F
and A11 anchor paths before any sensitivity or reference comparison is accepted.
The source-bundle PASS must not be substituted for end-to-end feature-path PASS.


## Phase1F end-to-end feature verification

The original Phase1F feature path was independently reused on rand5/hard5 for all
three fixed seeds. Evidence: information/reference_phase1f_feature_audit.json.
The row identities and correctness match; each Stats17/sem16 feature has zero
maximum difference against B's derived cell features. Direct bundle predictions
match the stored Phase1F Stats probabilities within 2.220446049250313e-16 and
E1b probabilities within 2.7755575615628914e-16, retaining the 1e-9 tolerance.
This PASS applies to those Phase1F cells, not to all Phase1 random CSV predictions.

Canonical Phase0B K5 scores and Phase1F saved rand5 logits differ by up to
1.2874603271484375e-5 across these seeds despite identical rankings/correctness.
The audit therefore uses the exact saved Phase1F scores for its reliability
anchor; scorer-forward STOP checks and source-specific probability anchors remain
separate. No original prediction or cache is modified.


## Phase1 prediction-column model identity correction

The original run_phase1._dump_predictions writes the per-seed `best_score_only`
and `best_semantic` predictions into generic reliability columns. Its source
metadata.json identifies these as MSP and e2_score for all three seeds; the latter
is a score-only control despite belonging to the semantic-family search. Those
CSV columns are not fixed Stats Logistic and E1b predictions. The separate global
sufficiency_gate.json selects MSP and e1b_stats_semantic, so it must not be silently
anchored to the per-seed e2_score column either.

Comparing the loaded S/Full bundle against these different-model columns caused
the reported 0.10–0.38 differences. These are MODEL_IDENTITY_INCOMPATIBLE
comparisons, not failures to restore the same model. Existing CSVs and their numbers
remain untouched. Fixed Stats/E1b comparisons use their actual per-model source
metrics, coefficients and reconstructed original features with unchanged anchors.
Generic `semantic_reliability` column names cannot establish added semantic input.

## 2026-10-03 — resumed execution, completed audit, pre-bootstrap RER correction

The user explicitly requested continuation with gpt-6-luna/max subagents. The same three execution agents resumed. The previous Phase0A/B process ended with all twelve cohorts complete: 1632 stored formal estimates. B confirmed all 21 primary model selections, 5290 curve rows, and all three-seed 128-row diagnostics complete; its training lane was released. Correct-identity B reference anchors comprise 72 random S/Full rows and 36 Phase1F rows, all within unchanged original tolerances. Generic Phase1 MSP/e2_score CSV differences remain model-identity-incompatible diagnostics.

C's fourth audit used a fresh run directory, completed successfully, and published a hash-pinned current_audit.json pointer. It found 138 mismatches among 21026 expressions (RPN 86/10425; DETR 52/10601). All historical hard/dose source cohorts were audited separately. This completes the category/geometry audit, not the required frozen paired sensitivity analysis. Failed prior outputs are retained.

Primary code review caught a B pre-formal sign error: RER is relative risk removed by abstention, so Full-minus-baseline RER gain must be Full RER minus baseline RER. The provisional runner and fixture used the opposite sign. No formal B bootstrap had run; B is correcting both and merging matched/dose cells into joint image draws before formal execution. E-AURC reduction retains baseline minus Full. Publication export now also requires actual B bootstrap method metadata (5000 draws, seed 0, 95%, image cluster) rather than only a training/summary completion flag.

## 2026-10-03 — mechanism complete, first full-suite audit, pre-formal sensitivity checks

C completed all three formal mechanism families (5000 shared image-cluster draws, seed 0, 95% percentile). Four-corner and C1 point anchors passed; publication renders stored per-seed mean effects, paths, Shapley terms and interaction rather than averaging interval endpoints. Label paths have opposite observed signs in all three families; this descriptive accounting is not causal identification. The explicit resample_unit metadata was added from the already-executed helper without recomputing estimates.

The first complete test run had 1175 tests: 1165 passed, 3 failures, 5 setup errors, 2 skips, 224.970 s. All GPU checks used existing local weights with offline settings. Five setup errors are the unchanged 1e-9 constrained-thread logistic refit check (6.066e-8), distinct from the original frozen bundle load path that passes its original anchors. Two developing C sensitivity fixtures and one historical-material-bank path assertion failed. Their fixes retain mathematical/model invariants and original tolerances; the full suite is not yet accepted.

B's first formal bootstrap finished computing its first call but failed atomic NPZ publication with transient Windows sharing violation WinError32. The 44,573,300-byte temporary replicate artifact and attempt logs remain. No formal completion was claimed. B is adopting the shared bounded-retry writer and assessing recoverability from source/method fingerprints before restart.

Primary pre-formal review found that the developing C sensitivity orchestration passed sentence_id values into the image bootstrap helper in both regular and dose branches, despite the helper-level fixture correctly using image IDs. No C formal sensitivity run had started. C must correct the orchestration to actual image IDs, add a repeated-expression same-image integration fixture, and restrict main effect groups to testA, testB, and their test-only pool. Canonical sentence IDs remain identity keys, not resampling units. Source availability/geometry continues to record its full historical source cohorts separately.

## 2026-10-03 — preserved draws resumed; strict candidate preflight remains separate

B attempt2 resumed the original first-call 1320 float64 arrays of length 5000 after verifying archive readability/hash, saved runner/protocol snapshots, input hashes, estimator signature, cohort identity and all raw keys. A single discarded schema/key diagnostic draw was explicitly recorded; it does not enter formal estimates or replace the preserved 5000 arrays. Atomic output retry is covered by a transient-sharing-lock regression. The remaining eight calls are newly executed under the unchanged formal protocol.

The hard-competition tests now load the original frozen bundle and verify its original 1e-9 anchors rather than refitting a model as a prerequisite to testing a frozen model. Train-image/row masks, selected C, normalization, coefficients, predictions and immutability assertions remain. The old refit implementation and its failed constrained-thread route are retained. The entire hard-competition file passed 12 tests; full-suite acceptance is still pending.

C's actual GPU preflight took two canonical rows per source. The bundle-only anchor passed (2.22e-16); the first source's raw score replay error was 3.81e-6, within original 1e-4, while fresh confidence errors around 2.3–3.3e-7 exceeded original 1e-9. This full numerical replay route failed, with separate preflight files preserved. No tolerance was relaxed and no formal sensitivity was started. Exact saved old logits plus corresponding frozen feature/model load are being evaluated as the protocol's preferred lossless-artifact route; scorer replay and reliability replay have distinct original anchor contracts.

### Research Repair v1 — formal information and candidate-forward milestones (2026-10-03)

B completed all nine formal shared image-cluster bootstrap calls (5000 draws, seed 0,
95% percentile), with the preserved first-call raw arrays recovered under the logged
strict fingerprint checks. The final method record is `information/information_bootstrap.json`;
1860 flat estimates retain per-seed uncertainty, valid/invalid counts and raw evidence.
The source-derived publication figures show cellwise Full−(S+Q) effects and small/large-K
training results. Incremental V effects are not uniformly resolved across cells; the
repair does not promote them to a general candidate-interaction or information-limit claim.

C completed frozen forward for all eight historical hard/dose sources using the
versioned old/corrected availability intersection. Original STOP and reliability tolerances
remain unchanged. Phase1F uses exact saved old logits for reliability anchors, with
fresh scorer replay separately checked; V2-P preserves original lossless confidences
because full logits are unavailable, and reports fresh numerical residuals separately.
The strict fresh-confidence preflight failure and the two candidate-control preflight
failures are retained. Dose controls now use original expb_m0 candidate construction,
not the distinct rand10 source. Paired 5000-draw candidate sensitivity remains pending.

B released the heavy statistics slot to A. C released the model lane for any required
original-parameter recovery. A's first next-batch launch failed before bootstrap because
of duplicate estimate names; no formal results from that attempt are accepted. The
remaining-scope intervals, actual old/new gate decisions, final full suite and immutable
input verification remain required before COMPLETE.

### Research Repair v1 — independent acceptance and stored-array checks (2026-10-03)

The primary agent added `scripts/check_repair_acceptance.py`, which reads saved
artifacts without resampling or declaring completion. The required-scope ledger
covers 16 scopes, including those not yet registered or awaiting original-model
recovery. Scope omissions and UNVERIFIABLE declarations without concrete structured
failure evidence remain incomplete. Operational decisions must retain historical
thresholds and link to actual eligible formal estimators; invalid draws cannot be
used for an operational gate. Candidate chain checks are being strengthened from
current audit through forward and prediction manifest to paired sensitivity groups.

`scripts/verify_repair_estimates.py` independently recomputes percentiles, valid/invalid
counts, per-draw fixed-seed means, observed mean effects and seed SD from the stored
raw arrays. It creates no bootstrap samples. Its 1e-12 comparison tolerance concerns
serialized arithmetic consistency and does not alter any original recovery anchor
or STOP tolerance. This final numerical verification is pending complete stable
outputs. Targeted integration tests also verify that reversing draw pairing can be
detected even when aggregate percentile endpoints remain identical.

The running A batch was launched before additional adapters were written. A did not
capture the launch-time runner fingerprint, and this omission is explicitly retained;
the later live file hash will not be described as the executed runner version. Future
formal launches must preserve the runner snapshot and estimator specification identity.


### 2026-10-03 — recovery provenance classification and mixed-scope acceptance

The G4 recovery attempt initially placed two newly written repair implementation
files in its supplemental immutable-input manifest. Later authorized edits exposed
the classification error. The original manifest bytes are preserved as attempt
source provenance; the replacement supplemental manifest retains the other 46
original data entries with their original sizes and hashes. No original prediction,
model, candidate or cache hash was re-pinned. Future recovery attempts copy their
implementation sources before execution and separate those snapshots from immutable
scientific inputs. The M25 three-seed archives that had already passed remain usable;
that attempt's source-snapshot conflict is a runner defect, not scientific failure.

The original M3-B0 path passed its M1/M2 confidence anchors but failed committed
point-table anchors at all three seeds: AUROC discrepancies were approximately
4.765e-7, 9.490e-8 and 4.766e-7 against the unchanged original 1e-9 tolerance.
The recovery attempt and separate seed failure records remain preserved. These
failures cannot stand in for the required B1/B2 work. Mixed M3 coverage must
independently account for the B0, B1 and B2 jobs, linking an unrecoverable job to
structured failure records and continuing all recoverable jobs through the formal
5000-draw procedure. New acceptance tests reject omitted backbones and whole-scope
UNVERIFIABLE status used to skip pending formal jobs.


### 2026-10-03 — complete candidate sensitivity and remaining acceptance

The true-target-category sensitivity completed all 15 shared image-cluster calls:
12 regular source/split groups and three joint-dose groups, with 24 source/split
mappings and 1,404 stored estimates. Each call used 5,000 draws, seed 0 and 95%
percentile intervals; fixed-seed effects were averaged within each shared draw.
The independent candidate metadata validator found all group files, canonical
cohort identities, archive hashes, aggregate keys and three seed keys consistent.
A validator omission for the declared `dose_macro` cell was corrected without
altering bootstrap outputs. Raw array lengths, percentiles and per-draw seed means
remain assigned to the final independent numerical verifier.

Publication tables now name each confidence head and report seed SD so repeated
AUROC/selective rows cannot be confused. Old-minus-corrected effects condition on
the source-specific paired availability intersection. Corrected and original dose
arms each use their own m0 baseline; the fixed original-m0 contrast remains
separate and descriptive. An interval containing zero does not establish candidate
version equivalence. The heavy CPU slot has returned to the remaining A scopes;
overall repair completion and final acceptance remain pending.


### 2026-10-03 — original M3 recovery exhausted; remaining formal launch

M3-B0 and M3.1-B1/B2 completed the original deterministic recovery routes for all
nine fixed seeds. Frozen scorer/Phase-A checks and applicable M1/M2 confidence or
STOP checks passed, but every backbone had committed point-table discrepancies
above its unchanged 1e-9 metric tolerance. Independent per-backbone parent failure
records bind the three seed records, final run logs, source hashes and checkpoint
inventories. The original M3 outputs contain tables/figures rather than the missing
per-seed curriculum/mixer checkpoints or confidence arrays needed for exact replay.
No M3 repair bootstrap was substituted and no original tolerance was relaxed.
These are historical-recovery limitations, not evidence that the tested models
lack reliability information or that the scientific hypotheses are false.

A launched 31 remaining formal jobs, comprising 2,807 estimates, after all source
and anchor preflights passed. Launch provenance is stored in
`statistics/formal_batches/20261003T133926Z/launch_manifest.json`, including exact
code snapshots, specifications, command/environment and 199 distinct source/input
hashes. The batch uses 5,000 shared image-cluster draws, seed 0, 95% percentile
intervals, and BLAS at most two threads. Existing 28 jobs are not resampled.

Independent gate review identified original source/document discrepancies that
must remain explicit in the repair: Phase05's actual headline uses the seed-mean
predicate although its document requires seed consistency; G3's E-AURC CI test is
nonzero without a directional assertion; D1 C1 uses seed-mean rather than per-seed
rows; several P provenance labels referenced nonexistent wrappers. A5.4's separate
post-hoc Reliability GO criterion also needs its own accounting and cannot be
hidden by the nonexistent standalone B0 temperature gate's NOT_APPLICABLE status.
Gate repair retains actual historical thresholds and decisions, discloses these
source differences, and rejects unresolved null placeholders as completed evidence.


### 2026-10-03 — passing full suite before required M2 auxiliary registration

The source-bound existing-weight suite passed: 1,218 tests, zero failures/errors,
and the two original opt-in live-extraction skips, in 283.8 seconds. All 259
tracked source snapshots remained unchanged during this offline, two-thread run.
Its exact log, JUnit and source record are preserved under `logs/*before_m2_aux*`.

A missing M2 gate auxiliary estimator was identified after the formal batch launch:
relative E-AURC reduction at m8. The registered baseline and LCR endpoint estimates
already use the same image draws for every fixed seed. The missing quantity must
be derived as (E-AURC(E1b)-E-AURC(LCR))/E-AURC(E1b) inside each seed/draw and then
averaged across fixed seeds. Ratios of aggregate means and subtracted endpoint CIs
are not substitutes. A reproducible utility and meaningful mathematical tests are
required before using this diagnostic. Zero/nonfinite denominators remain invalid;
no additional bootstrap draws are authorized or needed for this derivation.
The new source addition requires a final full-suite rerun; the preserved passing
run does not by itself establish final repair acceptance.


### Repair v1: M2 paired derived estimator (2026-10-03)

The formal launch snapshot registers 2807 estimates in 31 remaining jobs. Its M2 endpoint draws are reused to derive the missing relative E-AURC gate diagnostic without new sampling: compute (E1b E-AURC - LCR E-AURC) / E1b E-AURC at the same replicate index for each fixed seed, then average seed ratios. Denominators at or below the original 1e-12 cutoff, or nonfinite endpoints, produce invalid draws. The exact historical per-seed and mean points retain the original absolute 1e-9 anchors. This is an explicitly registered post-launch deterministic derivation, separate from the launch-time estimator count.

Independent review verified the formula and anchors and prompted explicit common-archive path/hash checks plus a preflight with no writes. Focused tests pass. The previous stable 1218-test suite (1216 pass, two original opt-in skips) and source snapshot are preserved under logs/*_before_m2_aux.*; the final suite was relaunched after these source changes. No completion claim is made before formal outputs, raw numerical QA, input hashes and acceptance pass.


M2 reproducibility routes: the active process uses the original launch snapshot with 281 M2 estimates; its missing-estimate utility adds one estimate from saved draws before metadata/coverage and gate refresh. A fresh run using the updated builder computes the ratio directly and does not repeat that utility. The utility requires both endpoints to name the same physical raw archive, compares predeclared hashes when present, and always records a derive-time SHA with a read-after hash check. Current endpoint summaries have no predeclared raw SHA; this is not an independent comparison to an upstream stored hash. Final raw QA fingerprints every archive.


### Repair v1: launch count reconciliation (2026-10-03)

The frozen 31-job launch manifest reports 2807 through n_estimates fields that count ordinary prediction specs only. Actual outputs include six already-defined auxiliary estimates in each of D2 C4, P1 C4 RPN and P1 C4 DETR (18 total). These must be reconciled against the frozen preflight/source snapshots in a separate record, without editing the launch manifest. They are distinct from the one post-launch M2 ratio. Final coverage and verification use exact saved names and actual counts rather than the incomplete launch count field.


### Research Repair v1 acceptance completed (2026-10-04, Asia/Shanghai)

The three gpt-6-luna/max execution agents completed the authorized repair. Exact coverage accounts for all16 required scopes:15 formal scopes and M3 with independent B0/B1/B2 strict recovery-failure evidence under unchanged tolerances. Statistics has59 formal jobs and6646 estimates; the frozen31-job batch has2825 actual estimates (2807 ordinary-spec count plus18 pre-existing auxiliaries), followed by one deterministic M2 ratio derived from its saved shared draws. All23 operational gate rows are terminal (20 evaluated,2 not applicable,1 M3 unverifiable), without threshold changes.

Four-axis stored-array verification passed all9955 estimates: statistics6646, information1860, candidates1404, mechanism45. Independent M2 same-index ratio, aggregate seed mean, historical point anchors, fingerprints and exact282-name ledger also passed. Final fullsuite has1226 tests:1224 passed,2 original opt-in live-extraction skips,zero failures/errors;262 Python source files remained unchanged during the run. Final baseline plus supplemental verification covers5 manifests and834 unique files with zero changes; both original Git document prefixes remain preserved. Overall acceptance_check.json is PASS with zero unresolved checks.

M3 recovery limits do not provide new effect CIs or establish model ineffectiveness. The early A launch-fingerprint omission and incomplete pre-derive raw-SHA collection remain explicitly documented; current raw arithmetic/fingerprints and the recorded M2 source/read-after hashes pass. Current README, result summary, completion audit, tables and registry are regenerated from the accepted artifacts. Conclusions retain GT-assisted target-present conditions, fixed-seed uncertainty, shared COCO visual domain, cellwise feature gains and post-result test-set reuse. No new backbone, dataset, architecture or expanded interface was introduced.
