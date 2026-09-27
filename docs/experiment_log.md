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
