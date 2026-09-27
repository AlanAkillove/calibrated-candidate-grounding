# Phase 0 Execution Checklist — `docs/phase0_plan.md`

对应原始需求 **Task 7**。本文件把 Phase 0（Candidate-Set Failure Audit）拆成有序、可勾选的
执行步骤，顺序为：

```text
proposal audit
→ feature extraction
→ candidate construction
→ cosine baseline
→ independent MLP
→ calibration
→ shift audit
→ bootstrap
→ Gate Q1
→ Gate Q2
```

所有判据、split 用途、指标定义以 `docs/research_protocol.md`（FROZEN BEFORE PHASE-0 RESULTS）
为准；本文件只规定执行顺序、输入输出、sanity check 与 failure condition，**不修改协议**。

## 0. 全局约束（贯穿所有步骤）

| 约束 | 内容 |
|---|---|
| 禁止事项 | 禁止实现任何 candidate-aware network（Phase 0 只允许 B0/B1/B2/B3，见协议 §7） |
| 冻结项 | proposal/target 定义、嵌套 candidate 构造、split 用途边界、gate 阈值 |
| 随机性 | 所有 learned model 3 seeds；candidate set 预生成并冻结；seed 由 `ccg.utils` 统一设定 |
| 日志 | 每个步骤结束都写 `experiments/phase0/<step>/`；正式实验按 `docs/experiment_log.md` schema 记录 |
| 报告纪律 | 不得静默过滤样本；被排除样本必须报数与报特征 |
| GPU | 仅步骤 1、2 需要 GPU（且步骤 2 是一次性）；步骤 3–10 基本为 CPU / 轻 GPU |
| 停止规则 | 步骤 1 或 3 触发 failure condition 时，先出 audit 报告再决定是否 amendment，不得继续往后跑 |

## 1. 顺序总览

| # | 步骤 | 脚本（规划） | GPU | 阻塞下一步的条件 |
|---|---|---|---|---|
| S0 | 环境与确定性检查、metrics-first 单测 | `pytest`, `scripts/run_phase0.py --selfcheck` | 否 | 单测不通过 |
| S1 | Proposal audit | `scripts/extract_proposals.py` + `scripts/audit_proposals.py` | **是** | recall/可用数未出报告 |
| S2 | Feature extraction（region + text） | `scripts/extract_features.py` | **是** | cache 校验失败 |
| S3 | Candidate construction（嵌套、冻结） | `scripts/build_candidate_sets.py` | 否 | 嵌套性/target 恒定校验失败 |
| S4 | Cosine baseline（B1，含 B0） | `scripts/run_phase0.py --model cosine` | 否 | ID cell 结果缺失 |
| S5 | Independent MLP（B3，3 seeds） | `scripts/run_phase0.py --model independent` | 轻 | 未收敛 / 违反 candidate-blind |
| S6 | Calibration（B2 + C1/C2/C3） | `scripts/run_phase0.py --calibrate` | 否 | calib split 泄漏 |
| S7 | Shift audit（4×2 grid 指标 + diagrams） | `scripts/run_phase0.py --grid` | 否 | 任一 cell 缺指标 |
| S8 | Bootstrap（image-level paired, ≥5000） | `ccg/metrics/bootstrap.py` | 否 | CI 未覆盖主比较 |
| S9 | **Gate Q1 判定** | `scripts/run_phase0.py --gate Q1` | 否 | 判 NO-GO 则停止 Phase 1 |
| S10 | **Gate Q2 判定**（Phase 0.5 之后） | `scripts/run_phase0.py --gate Q2` | 否 | 判 NO-GO 则停止 candidate-aware |

> **执行状态（2026-09-27）**：**S1 proposal audit 已完成**——1500/1500 图（failed=0）；
> N=64：RefCOCO+ target 级 recall@IoU0.5 = 0.9858742004264393（3752 expressions）、
> natural omission（expression 级）= 53/3752 = 0.014125799573560768；
> N=64 满足 K=50 比例 = 3719/3752 = 0.9912046908315565 ≥ 90% → 按 A2.4 预注册规则选定 **N=64**。
> **S2 起（feature extraction / candidate construction）实施时一律使用 N=64。**
> 详见 `docs/experiment_log.md` 条目 `audit-proposal-001` 与 `docs/research_protocol.md` Amendment A3。

---

## S0 — 环境 / 确定性 / metrics-first

| 字段 | 内容 |
|---|---|
| 输入 | 干净 conda 环境（`pip install -e ".[dev]"`）、`pyproject.toml`、`tests/` |
| 输出 | pytest 报告；确定性双跑一致性记录 |
| Sanity check | ① Top-1 / top-label ECE（adaptive & equal-width）/ binary Brier / correctness NLL / confidence–accuracy gap / risk–coverage & AURC / AUROC / AUPRC / false selection rate / paired bootstrap 全部有 synthetic 单测：perfect calibration、perfect ranking、overconfidence、underconfidence、all-correct、all-wrong、constant confidence；② 同一 config 跑两次，逐位相同（logits 级 allclose） |
| Failure condition | 任一指标单测不通过；同 seed 双跑不一致；bootstrap 在已知解析解的小样本上 CI 明显偏离 |
| 需要 GPU | 否（CPU 即可） |
| Expected artifact | `experiments/phase0/s0_selfcheck.md`；`tests/test_metrics_*.py` 通过记录 |

## S1 — Proposal audit（必须在任何 decision-model 结果之前完成）

| 字段 | 内容 |
|---|---|
| 输入 | COCO2014 `train2014` 图像（**已实测：RefCOCO+ 全部 19,992 图像均位于 train2014；val2014 图像非必需**）；RefCOCO+ 解析结果（gt_box、image_id、split）；冻结 torchvision Faster R-CNN R50-FPN COCO_V1 权重（159.7 MB） |
| 输出 | `cache/proposal_bank.h5`（boxes `[N,4]`、objectness `[N]`、gt_iou、gt_assignment）；`results/phase0/proposal_audit.{json,md}` |
| 过程要点 | 每张图固定 N=64（RPN NMS 后按 objectness top-K）；保留 torchvision 官方预处理（短边 800 / 长边 ≤1333）；**query-independent**（不得使用 expression 信息）；记录权重 URL + sha256 |
| Sanity check | ① 图像数 = 期望的 ~20k 量级；② 每图 proposal 数分布（是否普遍能达到 64）；③ **recall@64 / recall@32**；④ **natural miss rate**（`max_i IoU < 0.5` 比例）；⑤ **IoU distribution**（直方图 + 分位数）；⑥ target assignment 唯一性：确认存在多个 IoU≥0.5 proposal 的图已被正确记录，且等价 proposal 已从候选池中移除；⑦ 坐标约定（xyxy vs xywh、绝对 vs 归一化）在可视化抽查中正确 |
| Failure condition | (a) 图像缺失/无法解码 → 必须先补齐再评估；(b) `recall@64 < 0.85` → 触发 `dataset_protocol.md` §5 的处置流程（报告 + 考虑 amendment，不得静默过滤）；(c) 大量图像候选数 < 50 → 记录 candidate availability 分布，并据此决定是否把最大 K 降为 20（需 amendment）；(d) natural miss 过少 → 标记 RQ4 的 natural split 为小样本描述性结果 |
| 需要 GPU | **是**（FP16/autocast，batch 64 起步，OOM 降 32） |
| Expected artifact | `proposal_bank.h5`；`proposal_audit.md`（含 4 项必报统计 + 分层表：按 COCO 类别 / 目标面积 / split） |
| 完成状态 | **已完成（2026-09-27）**：1500/1500 图（analyzed=1500，cached=5 + extracted=1495，failed=0）；N=64：RefCOCO+ target 级 recall@0.5 = 0.9858742004264393、natural omission（expression 级）= 53/3752 = 0.014125799573560768；N=64 满足 K=50 比例 = 3719/3752 = 0.9912046908315565 ≥ 90% → 按 A2.4 选定 **N=64**；artifacts：`results/proposal_audit/`（summary.json + 5 CSV + figures 5 张）、`cache/proposal_audit/`（1500 npz，4.7 MB）。详见 `docs/experiment_log.md` `audit-proposal-001` 与 `docs/research_protocol.md` Amendment A3 |

## S2 — Feature extraction（一次性、离线缓存）

| 字段 | 内容 |
|---|---|
| 输入 | `cache/proposal_bank.h5` 的 boxes；COCO 图像；RefCOCO+ 表达文本；冻结 **OpenCLIP ViT-B/32** |
| 输出 | `cache/region_features.h5`（`[N,512]` FP16，按 image_id 组织）；`cache/text_features.h5`；`cache/global_features.h5`（可选）；`cache/FEATURE_MANIFEST.json` |
| 过程要点 | crop → resize/normalize 与 OpenCLIP 官方 val transform 一致（**具体缩放与 padding 规则必须在 manifest 中写明并冻结**）；不做 backbone 微调；视觉 feature **不按 expression 重复存储** |
| Sanity check | ① 特征 shape/dtype 正确、无 NaN/Inf；② L2 范数分布（是否已归一化，若存未归一化需说明）；③ 随机抽 20 个 crop 人工查看是否为合理目标区域；④ 同一 crop 两次提取结果一致（determinism）；⑤ 文本 token 截断率（>77 token 的表达占比，需报告）；⑥ 体积符合预算（region ≈1.3 GB、总 <3 GB） |
| Failure condition | 随机访问读取延迟不可接受（说明存储布局错误）；出现 NaN；crop 全黑/越界比例 > 预期；OOM 后未记录实际使用的 batch size；发现需要重提特征才能支持 candidate 变更 ⇒ 说明 §7 缓存结构设计失败，必须先修结构 |
| 需要 GPU | **是**（FP16 batch 64 → 32；无训练，不做 gradient tricks） |
| Expected artifact | 三个 `.h5` + manifest（含 backbone 名、权重 hash、transform 参数、耗时、batch size、硬件） |
| 规模冻结 | S2 起（feature extraction / candidate construction / candidate sets）一律使用 **N=64** proposal bank（2026-09-27 按 A2.4 / Amendment A3 选定；top-64 = RPN NMS 后按 objectness 排序截断） |

## S3 — Candidate construction（项目最关键步骤）

| 字段 | 内容 |
|---|---|
| 输入 | proposal bank（含 gt_iou / gt_assignment / objectness / COCO 类别）、region features、固定 seeds |
| 输出 | `cache/candidate_sets.h5`（或 jsonl）：`ref_id, regime, K, hardness, target_present, candidate_indices, target_candidate_index` |
| 必须满足（协议 §6） | ① **嵌套**：`C5 ⊂ C10 ⊂ C20 ⊂ C50`；② **target 恒定**：同一 `(I,q,c*)` 在所有 K 下 target 是同一个 proposal index；③ K 与 hardness 解耦（同一 hardness 下只增 distractors）；④ random 负样本固定 seed 并保存 indices；⑤ same-category hard 与 CLIP-hard 两种 hard 定义分别生成并打 regime 标签；⑥ synthetic omission 版本 `C⁻ = C \ {c*}` 与 natural omission 集合分别落盘 |
| Sanity check | ① 嵌套性单测：对每个 ref 校验 `set(C_K) ⊂ set(C_K')`（K<K'）；② target index 在所有 K 完全一致；③ target presence 标记与 `max IoU ≥ 0.5` 一致；④ 候选池内不存在第二个 IoU ≥ 0.5 的 proposal；⑤ 各 K 下可用样本数（`|P_I| ≥ K`）与排除数统计；⑥ CLIP-hard 的 similarity 分布确实比 random 更集中于高值（否则构造无效）；⑦ 训练只见 K∈{5,10}，测试网格 K∈{5,10,20,50} × {random, hard} = **4×2 cells** 全部存在 |
| Failure condition | 嵌套性/target 恒定性被违反（**硬失败，必须修复后才能继续**）；K=50 可用样本过少（转 amendment 决策）；hard regime 与 K 纠缠（例如 hard 集合规模不同）；candidate sets 未在 evaluation 前冻结（每次重新 sample） |
| 需要 GPU | 否（CLIP-hard 排序用已缓存 embedding，CPU 可算；数据量大时可 GPU 加速矩阵乘，但结果必须与 CPU 一致） |
| Expected artifact | `candidate_sets` 文件 + `experiments/phase0/s3_candidate_audit.md`（每 cell 的样本数、平均候选数、排除数） |

## S4 — Cosine baseline（B1）+ Random（B0）

| 字段 | 内容 |
|---|---|
| 输入 | `region_features`、`text_features`、`candidate_sets` |
| 输出 | B0/B1 在每个 test cell 上的 raw scores：`s_i = cos(z_q, z_i)`；probabilities（含一个显式记录的 logit scale/temperature，初始 T=1 也要记录） |
| Sanity check | ① B0 的 Top-1 ≈ 1/K（每个 K 都验）；② B1 的 logits 无 NaN、score 范围在 [-1,1]；③ 同一 cell 内所有模型评估样本集合完全相同；④ cosine 与 `s_i^clip`（用于 CLIP-hard 构造的分数）一致性检查（注意：这会让 CLIP-hard cell 上 B1 的 Top-1 下降，属预期的 diagnostic 现象，必须写明而不是掩盖） |
| Failure condition | B0 明显偏离 1/K（说明 candidate/target 索引错位）；样本集合在不同模型间不一致 |
| 需要 GPU | 否 |
| Expected artifact | `results/phase0/s4_cosine/{cell}_predictions.parquet`（含 logits、candidate ids、confidence、correctness） |

## S5 — Independent MLP scorer（B3，3 seeds）

| 字段 | 内容 |
|---|---|
| 输入 | 训练集 candidate sets；特征 `[z_q, z_i, z_q ⊙ z_i, cos(z_q,z_i), g_i]`，`g_i` = normalized box (x,y,w,h,area) + 可选 objectness |
| 输出 | 3 seeds 的 MLP checkpoint（参数量 ≤1M，须打印实际参数量）+ 每 seed 每 cell 的 predictions/raw logits |
| 训练设置 | 只在 `train` 上训练；`val_select` 用于 architecture/optimizer/early stopping/hyperparameter；K_train ∈ {5,10}；random negatives（第一版可加 same-category hard 的 ablation，但主 B3 用与测试解耦的配置，须在日志中写明） |
| Sanity check | ① **candidate-blind 证明**：单元测试打乱输入 candidate 顺序后，逐 candidate 分数按同一置换重排（permutation equivariance of an independent scorer），且任一 candidate 的分数不随其他 candidate 集合变化（用两个不同 K 的集合前缀测试）；② 参数量 ≤ 1M；③ 3 seeds 收敛曲线合理、val_select Top-1 不异常波动；④ loss 下降但 val 不退化 |
| Failure condition | 违反 candidate-blind（实现里意外 pool 了集合信息）→ 该模型必须重命名并停用；不收敛；seeds 间方差大到任何比较都无意义（记录并报告，不减 seed 数）；用 `val_calib` 或 test 做 early stopping（硬失败） |
| 需要 GPU | 轻（embedding 上的小 MLP，8GB 足够；CPU 亦可） |
| Expected artifact | `experiments/phase0/s5_independent_mlp/seed{0,1,2}/…`，含 config、logits、checkpoint |

## S6 — Calibration（B2 global T；Phase 0.5 的 C1/C2/C3）

| 字段 | 内容 |
|---|---|
| 输入 | B1/B3 在 **`val_calib`** 上的 logits 与 correctness 标签 |
| 输出 | 拟合的 `T`（B2/C1）；`T=a+b\log K` 的系数（C2）；C3 stats-only calibrator（输入仅 `K, max_score, top1_top2_margin, entropy, mean_score, std_score, max_softmax`，必要时 top3 统计；**禁止 candidate embeddings**）；abstention thresholds（同样只在 `val_calib` 上定） |
| 关键约束 | 测试 K=20/50 时**不得**针对 test 拟合；C2 必须使用可外推形式或 validation-conditioned 形式；test 上只应用、不调参 |
| Sanity check | ① `T` 在 val_calib 上确实降低 ECE；② 拟合前后 Top-1 **完全不变**（temperature 是单调变换；若 Top-1 变了说明实现有 bug）；③ C3 的特征列表与允许集合逐项相符（自动断言，禁 candidate embeddings）；④ calib 与 select split 的 image_id 交集为空 |
| Failure condition | 发现任何用 test 调温度/threshold 的代码路径（硬失败）；`val_calib` 样本量不足以稳定拟合（报告 CI）；C2 在 K=20/50 外推出现非正/极端 T（记录为失败模式，不得回退到 test 拟合） |
| 需要 GPU | 否 |
| Expected artifact | `results/phase0/s6_calibration/calibration_params.json`（含拟合数据量、split 名、拟合目标 NLL/ECE） |

## S7 — Shift audit（4×2 grid 完整指标）

| 字段 | 内容 |
|---|---|
| 输入 | S4/S5/S6 的全部 predictions |
| 输出 | 每个 cell（K ∈ {5,10,20,50} × regime ∈ {random, hard}）× 每个模型（B0/B1/B2/B3/C1/C2/C3）× 每个 seed 的完整指标表 |
| 必报指标 | Top-1、Top-5（若 K 允许）；accuracy by K / by hardness；**top-label ECE（adaptive / equal-mass bins）**；binary correctness Brier；top-label correctness NLL；confidence–accuracy gap；multiclass NLL / Brier（secondary）；risk–coverage curve、**AURC**、selective accuracy、**risk@50%/80%/90%/95%**；reliability diagrams（K=5/10/20/50 分别画） |
| Sanity check | ① 同一 cell 各模型样本集合完全相同；② 指标与 `ccg.metrics` 单测口径一致（risk = 1 − accuracy 已显式声明）；③ ΔAccuracy 与 ΔCalibration **分开**呈现；④ 每张 reliability diagram 标注样本量与 bin 数；⑤ 明确区分 ID cells（K=5/10 random）与 OOD cells（K=20/50 and/or hard） |
| Failure condition | 任一 cell 缺指标（禁止部分报告）；出现 "把 softmax 的 K 依赖当作 calibration failure" 的解读（措辞审查）；hard regime 未标 diagnostic 标签 |
| 需要 GPU | 否 |
| Expected artifact | `results/phase0/s7_grid/metrics_long.csv`、`figures/reliability_K*.png`、`figures/risk_coverage_*.png` |

## S8 — Bootstrap 统计检验

| 字段 | 内容 |
|---|---|
| 输入 | S7 的 per-sample predictions（必须已保存，不能只有 accuracy） |
| 输出 | 主比较的 paired bootstrap 差值分布与 95% CI：ΔAccuracy、ΔECE、ΔAURC（relative）、ΔBrier、ΔNLL |
| 协议要求 | **image-level resampling 优先于 expression-level**；**≥ 5,000 replicates**；95% CI；同一 test candidate set 上所有模型用完全相同样本；3 seeds 的结果分别报 + 报告 seed 间变异（不得只挑最好 seed） |
| Sanity check | ① 自我一致性：同一模型自比时 Δ≈0 且 CI 关于 0 对称；② replicate 数、随机种子写入 artifact；③ 重抽样单位为 image（打印每 replicate 抽到的唯一 image 数）；④ 极端 cell（样本量小）CI 宽度必须一并展示 |
| Failure condition | CI 计算用了 expression-level（需重跑）；replicates < 5,000；对小样本 cell 用点估计下结论 |
| 需要 GPU | 否 |
| Expected artifact | `results/phase0/s8_bootstrap/ci_table.csv`、`bootstrap_summary.json` |

## S9 — Gate Q1 判定

| 字段 | 内容 |
|---|---|
| 输入 | S7 指标 + S8 CI |
| 判定内容（照抄协议 §11，不得改阈值） | 从 ID（K=5/10, random）到 OOD（K=20/50 and/or hard）：至少出现 `\|ΔAcc\| ≥ 3pp` 或 `ΔECE ≥ 3pp` 或 AURC 恶化 ≥ 20% relative；**且**同时满足：paired bootstrap 95% CI 不跨 0、cosine 与 independent MLP 至少都观察到同方向现象、至少两个 OOD cells 满足 |
| 输出 | 明确的 `GO` / `NO-GO` 决议 + 支撑表（每条子判据逐 cell 打勾） |
| Sanity check | 判定完全由脚本 + 表格驱动（人工只核对，不重新解释阈值）；若判据部分满足部分不满足，必须如实记为 NO-GO 而不是模糊表述 |
| Failure condition | **NO-GO** → 停止 candidate-aware architecture 路线；把项目收敛为 "candidate-set reliability/audit" 类结果并如实报告 candidate-set shift 未形成足够稳定的独立研究问题 |
| 需要 GPU | 否 |
| Expected artifact | `results/phase0/GATE_Q1.md`（含日期、commit、脚本版本） |

## S10 — Phase 0.5 + Gate Q2 判定

| 字段 | 内容 |
|---|---|
| 前置条件 | 仅在 Gate Q1 = GO 时进行 |
| 输入 | C1 global temperature、C2 K-aware temperature、C3 stats-only calibrator 在 OOD cells 上的表现 |
| 判定内容（照抄协议） | 若 stats-only model 在**所有主要 OOD cells** 达到 **ECE < 3%**，且相较 global temperature 的 **ΔAURC < 5%** 剩余相对改善空间 → **NO-GO for candidate-aware model** |
| 输出 | `GO to Phase 1` 或 `NO-GO（项目收敛为 audit + simple calibration solution）` |
| Sanity check | "所有主要 OOD cells" 的集合必须在跑 gate 之前列出（不得看到结果后再定义 "主要"）；C3 未使用 candidate embeddings（S6 已断言） |
| Failure condition | 判 NO-GO 后仍继续增加 candidate-aware architecture（**协议违规**） |
| 需要 GPU | 否 |
| Expected artifact | `results/phase0/GATE_Q2.md` |

---

## 11. 勾选清单（复制到 issue / PR 使用）

```text
[ ] S0  metrics 单测全通过；双跑确定性
[x] S1  proposal_bank.h5 + recall@32/@64 + natural miss rate + IoU 分布 + candidate availability（2026-09-27 完成：1500 图；实际报告 N=64/128 两档；N=64 选定，见 Amendment A3）
[ ] S1  等价 IoU≥0.5 proposals 已移除，target 唯一性已验证
[ ] S2  region/text features（FP16）+ manifest（transform、batch、sha256、耗时）；总 cache < 3GB
[ ] S3  嵌套性 + target 恒定性单测通过；candidate sets 已冻结（含 synthetic/natural omission）
[ ] S4  B0 Top-1 ≈ 1/K；B1 全 cell logits 已保存
[ ] S5  B3 candidate-blind 证明通过；参数量 ≤ 1M；3 seeds
[ ] S6  T / T(K) / C3 仅在 val_calib 拟合；调温前后 Top-1 不变
[ ] S7  4×2 grid 全指标 + reliability diagrams（K=5/10/20/50）+ risk@50/80/90/95
[ ] S8  image-level paired bootstrap ≥ 5,000 replicates，3 seeds 全报
[ ] S9  GATE_Q1.md（GO/NO-GO 及逐条判据）
[ ] S10 GATE_Q2.md（仅在 Q1=GO 时）
[ ] 全程  未实现任何 candidate-aware network；未使用 test 做任何选择
```
