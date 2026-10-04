# Phase 0A 修正后解释（Phase 0A.1 / Metric & Temperature Validity Correction）

> Research Repair v1 (2026-10-03): the values below are a historical Phase0A.1
> record. The claim that E-AURC/RER isolate discrimination beyond accuracy is
> withdrawn: both retain accuracy dependence. Cosine AUROC evidence must be
> assessed separately. Interior global-T and small per-K temperature drift cannot
> exclude scale effects; direct repaired diagnostics supersede that interpretation.
> See `results/research_repair_v1/statistics/` and `docs/final_result_summary.md`.

- 状态：**post-hoc 解释文档**（依据 `docs/research_protocol.md` Amendment A5）。它在 Phase 0A
  cosine results 已可见之后产生，**不构成 preregistration**，不得被引用为“事前预注册的判据”。
- 数据来源：`results/phase0a_corrected/`（修正温度 + base-rate-aware 指标，cohort 与 Phase 0A 完全一致，
  pooled common cohort n = 20,799 句；不重新抽取任何特征）。
- 目的：区分四类效应，并回答一个问题：
  > Is there evidence that confidence quality degrades beyond the mechanical increase in
  > classification difficulty?

---

## 1. Ranking degradation —— 结构性 / 预期（不是 reliability novelty）

嵌套候选集（C_K = target + frozen distractors 前缀）下，top-1 随 K 单调下降是数学预期：

| | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| top-1 (pooled common) | 0.5349 | 0.3900 | 0.2857 | 0.1879 |
| mean target rank | 1.91 | 3.05 | 5.31 | 12.10 |
| ΔAcc vs K5 | — | −14.5pp | −24.9pp | −34.7pp |

该层**不能**单独作为“reliability / calibration shift”的证据（Amendment A5 §A5.1）。

## 2. Aggregate calibration drift —— 修正 global T 后：小且方向温和

temperature 修复（log-T 空间有界优化，T ∈ [1e-3, 10]）：

- **corrected global T\* = 0.0335596，interior = True**（距下界 1.53 decade / 上界 2.47 decade）；
  Phase 0A 的 T\*=0.05 贴界为 optimization-boundary artifact，已消除。
- oracle per-K 温度（val_calib 各 K 单独拟合，ORACLE/DIAGNOSTIC）：
  **T5 = 0.034317、T10 = 0.033007、T20 = 0.032226、T50 = 0.031264**（≈10% 单调漂移）——
  修复边界伪影后 per-K 温度**不再完全相同**（上一轮的“全为 0.05”确为边界伪影）；
  但漂移量级仍小，per-K 相对 global 无实质增益。

pooled（n = 20,799）corrected global-T 指标：

| K | ECE(adaptive) | conf−acc gap | Brier | NLL |
|---|---|---|---|---|
| 5 | 0.0224 | −0.0103 | 0.2047 | 0.5936 |
| 10 | 0.0299 | −0.0233 | 0.1992 | 0.5840 |
| 20 | 0.0399 | −0.0397 | 0.1756 | 0.5293 |
| 50 | 0.0523 | −0.0511 | 0.1379 | 0.4380 |

- **cross-K ECE drift 小**（K50 vs K5：+3.0pp；各 K 绝对值 2.2%~5.2%），但**绝对校准误差仍 ~2–5%**，
  且 gap 随 K 变负（轻微欠自信并随 K 加深）——只能写 “cross-K ECE drift is small”，
  **不得写 “calibration is solved”**（Amendment A5 §A5.1）。
- Brier / NLL 随 K 下降是**基率效应**（错误率 0.81 时预测低 p 反而降低 Brier），不得读作“更校准”。

## 3. Confidence discrimination drift —— selective 效用显著恶化，整体判别力不显著

以 corrected global-T confidence 为选择信号（pooled，image-cluster paired bootstrap，5000 reps）：

| 指标 | K=5 | K=10 | K=20 | K=50 |
|---|---|---|---|---|
| **E-AURC**（越低越好） | 0.1404 | 0.1754 | 0.1849 | 0.1774 |
| E-AURC relative worsening vs K5 | — | **+24.9%** [20.9, 29.0] | **+31.7%** [26.2, 37.2] | **+26.3%** [19.4, 33.0] |
| **AUROC_correct** | 0.7413 | 0.7328 | 0.7329 | 0.7366 |
| ΔAUROC vs K5 | — | −0.0085 [−0.0159, −0.0010] | −0.0084 [−0.0179, +0.0013] | −0.0047 [−0.0164, +0.0076] |
| **RER@50** | 0.3835 | 0.2603 | 0.1876 | 0.1222 |
| ΔRER@50 (pp) | — | −12.3 [−13.8, −10.8] | −19.6 [−21.2, −17.9] | **−26.1 [−27.9, −24.2]** |
| RER@80 | 0.1348 | 0.0930 | 0.0632 | 0.0422 |

- **E-AURC（已减去 base-rate oracle 地板）相对恶化 +25%~+32%，所有 CI 不跨 0**；
  **RER@50 下降 12~26pp**（≥10pp 阈值），CI 不跨 0 → **Amendment A5 Route A 触发**。
- **AUROC_correct 变化 ≤0.009（阈值 0.03），K20/K50 的 CI 跨 0** → **Route B 不触发**：
  整体“把正确排在错误前面”的能力基本稳定；
  退化集中在 **selective / abstention 效用**（confidence 在低覆盖率区间的可用性）这一维度。
- 该结论不依赖 raw accuracy / raw AURC：E-AURC 与 RER 都是 base-rate-aware
  （E-AURC 减 oracle 地板；RER 除以当前基础错误率）。

## 4. Reliability-map drift —— 修正后方向反转（保守/欠自信）

corrected global-T，fixed-width 10 bins，pooled（同 nominal confidence 跨 K 比较）：

| bin | K=5 conf / acc (n) | K=50 conf / acc (n) |
|---|---|---|
| [0.2, 0.3) | 0.272 / 0.261 (2,070) | 0.241 / **0.345** (2,261) |
| [0.3, 0.4) | 0.351 / 0.340 (4,789) | 0.342 / **0.485** (796) |
| [0.4, 0.5) | 0.448 / 0.443 (4,261) | 0.441 / **0.556** (351) |
| [0.5, 0.6) | 0.547 / 0.578 (3,167) | 0.540 / **0.649** (148) |
| [0.9, 1.0) | 0.946 / 0.940 (1,148) | 0.907 / 0.000 (n=2, **low-support**) |

- K=50 的置信分布整体左移（47% 质量落入 [0, 0.1)），但在**同一 nominal confidence** 上，
  K=50 的经验准确率**高于** K=5（且高于 nominal）→ 修正后 reliability map 的漂移方向是
  **保守（under-confident）**，与此前 native-scale 下 “same confidence ⇒ lower accuracy” 的
  下移方向**相反**：旧观察主要由 native scale 的过自信放大 K 效应造成。
- K=50 在 ≥0.6 的桶 n < 100：全部标记 low-support，不得用于强结论（A5 §A5.1）。

## 5. 回答 §11 的问题

> **Is there evidence that confidence quality degrades beyond the mechanical increase in
> classification difficulty?**

**是——但必须限定在 selective 效用维度，而不是 aggregate calibration 或整体判别力：**

1. **有**：E-AURC（base-rate 校正后）相对恶化 +24.9%/+31.7%/+26.3%（K10/K20/K50），
   RER@50 下降 12.3/19.6/26.1pp——两者都是 accuracy-normalized 指标，CI 均不跨 0（Route A 触发）。
2. **没有（或不足）**：AUROC_correct 下降 <0.01 且 K20/K50 不显著（Route B 不触发）；
   corrected global-T 下跨 K 的 ECE drift ≤3.0pp；reliability map 的漂移方向反而保守。
3. 因此可写：**“confidence 作为 abstention/selection 信号的质量随 K 显著下降”**；
   不可写：“模型整体不再知道何时正确（AUROC 层面）”，也不可写“calibration 崩塌”。

## 6. 限制与注意事项

- corrected T\* 的拟合集仍是 val_calib ∩ K∈{5,10}（§9 不松动）；testA/testB 从未参与拟合。
- oracle per-K 温度仅 diagnostic，禁止作为 OOD 方法引用（A5 §A5.2）。
- fixed-bin 表在 K=50 高端桶 n 很小（low-support 标记）；equal-mass 桶在
  `results/phase0a_corrected/reliability_bins_all_variants.csv` 中并列提供。
- 本文件的所有数字可在 `results/phase0a_corrected/` 复算：
  `temperature_fit.json`、`calibration_metrics_corrected.csv`、`normalized_selective_metrics.csv`、
  `correctness_auroc.csv`、`reliability_bins_globalT.csv`、`eaurc_bootstrap.csv`、`rer_bootstrap.csv`、
  `auroc_bootstrap.csv`、`per_sentence_predictions.npz`。
