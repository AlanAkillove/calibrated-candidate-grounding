# V4 特征依赖审计与 `V_pure` 规范

## 审计范围和判定规则

本审计只读取并逐式核对 `ccg.repairs.information`、`ccg.reliability.features`、`ccg.semantic.features` 的实际实现，并参考 `ccg.v2.semantic_features` 中的旧候选特征实现。没有改写旧模型、旧特征、旧缓存或历史结果。S/Q/V 的冻结列名取自 `repairs.information`；其计算依赖取自 `reliability.features` 与 `semantic.features`，不能只由列名推断。

下表的“query”表示特征公式是否依赖 query 本身，或 query-conditioned B3 分数/排名；括号标出依赖路径。对 S17，`log_k` 只取 K，其余列从 B3 分数向量计算。Q8 直接读取 query-crop cosine。legacy V8 虽不读取 query embedding，却通过 B3 winner/名次间接依赖 query。`candidate` 包括候选数、候选分数、crop embedding 或候选框。`winner` 表示显式取用 B3 top-1 分数/候选作为锚点；`query_rank` 表示公式使用 query-conditioned B3 位置/名次，或把 B3 winner 放进 query-cosine 排名。所有 flags 描述计算依赖，不描述统计相关性。所有旧块均不直接读取 GT 或正确性标签。

## S17：`reliability.features.stat_features`

`log_k = ln K`；`top1/top2/top3` 与两个 margin 是 B3 分数的次序统计量；均值/标准差、熵、归一化熵、logsumexp 是对完整 B3 分数向量的集合统计；`msp` 是 B3 softmax 概率的最大值；`z_top1/z_margin` 用完整向量均值与 `ddof=0` 标准差归一化；`q25/q50/q75` 是分数向量的 NumPy 线性分位数。标准差和熵按代码的原公式计算。

| 特征 | query | candidate | winner | query_rank | GT | label | 依赖说明 |
|---|---|---:|---:|---:|---:|---:|---|
| `log_k` | 否 | 是 | 否 | 否 | 否 | 否 | 仅候选数 K。 |
| `top1` | 是（B3 分数） | 是 | 是 | 是 | 否 | 否 | B3 第一名分数。 |
| `top2` | 是（B3 分数） | 是 | 否 | 是 | 否 | 否 | B3 第二名分数。 |
| `top3` | 是（B3 分数） | 是 | 否 | 是 | 否 | 否 | B3 第三名分数。 |
| `margin_12` | 是（B3 分数） | 是 | 是 | 是 | 否 | 否 | 第一名减第二名。 |
| `margin_23` | 是（B3 分数） | 是 | 否 | 是 | 否 | 否 | 第二名减第三名。 |
| `mean` | 是（B3 分数） | 是 | 否 | 否 | 否 | 否 | 全候选分数均值。 |
| `std` | 是（B3 分数） | 是 | 否 | 否 | 否 | 否 | 全候选分数总体标准差。 |
| `msp` | 是（B3 分数） | 是 | 是 | 是 | 否 | 否 | 最大 softmax 概率对应 top-1。 |
| `entropy` | 是（B3 分数） | 是 | 否 | 否 | 否 | 否 | 完整 softmax 分布熵。 |
| `entropy_over_logk` | 是（B3 分数） | 是 | 否 | 否 | 否 | 否 | 完整 softmax 熵除以 `ln K`。 |
| `logsumexp` | 是（B3 分数） | 是 | 否 | 否 | 否 | 否 | 完整分数向量的 log-sum-exp。 |
| `z_top1` | 是（B3 分数） | 是 | 是 | 是 | 否 | 否 | top-1 相对完整分数均值/标准差。 |
| `z_margin` | 是（B3 分数） | 是 | 是 | 是 | 否 | 否 | top-1/top-2 margin 相对完整分数标准差。 |
| `q25` | 是（B3 分数） | 是 | 否 | 是 | 否 | 否 | 全分数的线性 25% 分位数。 |
| `q50` | 是（B3 分数） | 是 | 否 | 是 | 否 | 否 | 全分数的线性 50% 分位数。 |
| `q75` | 是（B3 分数） | 是 | 否 | 是 | 否 | 否 | 全分数的线性 75% 分位数。 |

## Q8：`semantic.features.semantic_stats` 的 query 侧列

记 `a_j = cos(z_q, z_j)`，并令 `order` 为 B3 分数降序稳定排列。下表只包含原语义向量被 `repairs.information.Q_NAMES` 选入的 8 列。query-cosine entropy 对全体 `a_j` 求 softmax 熵；其余带名次的 cosine 量按 B3 排名取候选，`clip_rank_top1` 另计算 B3 winner 在 query-cosine 排序中的平均名次。

| 特征 | query | candidate | winner | query_rank | GT | label | 依赖说明 |
|---|---|---:|---:|---:|---:|---:|---|
| `clip_top1` | 是（query cosine） | 是 | 是 | 是 | 否 | 否 | query cosine 在 B3 top-1 候选处的值。 |
| `clip_top2` | 是（query cosine） | 是 | 否 | 是 | 否 | 否 | query cosine 在 B3 top-2 候选处的值。 |
| `clip_margin12` | 是（query cosine） | 是 | 是 | 是 | 否 | 否 | B3 top-1 与 top-2 的 query-cosine 差。 |
| `clip_entropy` | 是（query cosine） | 是 | 否 | 否 | 否 | 否 | 所有 query-candidate cosine 的 softmax 熵。 |
| `clip_normH` | 是（query cosine） | 是 | 否 | 否 | 否 | 否 | 上述熵按 `ln K` 归一化。 |
| `clip_rank_top1` | 是（query cosine） | 是 | 是 | 是 | 否 | 否 | B3 winner 在 query-cosine 降序平均排名除以 K。 |
| `q_top3` | 是（query cosine） | 是 | 否 | 是 | 否 | 否 | query cosine 在 B3 top-3 候选处的值。 |
| `q_margin13` | 是（query cosine） | 是 | 是 | 是 | 否 | 否 | B3 top-1 与 top-3 的 query-cosine 差。 |

## legacy V8：`semantic.features.semantic_stats` 的 candidate 侧列

legacy V8 由 `repairs.information.V_NAMES` 固定为以下 8 列。实际代码先以 B3 top-1 候选 `t` 作锚，再计算 winner-candidate cosine；`cand_top12_sim`、`cand_top15_mean` 和 `cand_top13_sim` 还使用 B3 rank 2/3/2–5 位置。阈值密度排除 winner 自身，阈值比较严格大于。因 `t` 来自 query-conditioned B3 排名，legacy V8 不是严格 query-independent。

| 特征 | query | candidate | winner | query_rank | GT | label | 依赖说明 |
|---|---|---:|---:|---:|---:|---:|---|
| `cand_vmax` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | B3 winner 与所有其他候选 cosine 的最大值。 |
| `cand_vmean` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | B3 winner 与其余候选 cosine 的均值。 |
| `cand_vstd` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | B3 winner 与其余候选 cosine 的总体标准差。 |
| `cand_top12_sim` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | B3 winner 对 B3 rank-2 候选的 cosine。 |
| `cand_top15_mean` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | winner 对 B3 rank 2–5 候选 cosine 的均值。 |
| `density_070` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | winner 对其余候选 cosine 严格 `> 0.7` 的比例。 |
| `density_080` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | winner 对其余候选 cosine 严格 `> 0.8` 的比例。 |
| `cand_top13_sim` | 是（B3 选择） | 是 | 是 | 是 | 否 | 否 | B3 winner 对 B3 rank-3 候选的 cosine。 |

## 严格 `V_pure`：固定 12 列

`ccg.v4.pure_features.pure_features(crop_embeddings, candidate_boxes)` 只接受一个无序候选集合的 `[K,D]` crop embeddings 和对应 `[K,4]` `xyxy` boxes。实现先转 float64 并逐行重新单位化 crop embedding；零向量、非有限输入、非正框宽/高、非正或不可表示面积均报错，不填零。要求 `K >= 5`；每个候选成员都参与，不按 query 排名排序或截 top-K。每个 unordered pair 仅取 `i < j`，无对角；nearest neighbor 对每个候选排除自身。

| 顺序 | 特征 | 精确公式 | query | candidate | winner | query_rank | GT | label |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `cos_mean` | `mean({z_i·z_j : i<j})` | 否 | 是 | 否 | 否 | 否 | 否 |
| 2 | `cos_std` | `std({z_i·z_j : i<j}, ddof=0)` | 否 | 是 | 否 | 否 | 否 | 否 |
| 3 | `cos_q50` | `np.quantile({z_i·z_j : i<j}, 0.50, method="linear")` | 否 | 是 | 否 | 否 | 否 | 否 |
| 4 | `cos_q90` | `np.quantile({z_i·z_j : i<j}, 0.90, method="linear")` | 否 | 是 | 否 | 否 | 否 | 否 |
| 5 | `nn_cos_mean` | `mean_i(max_{j≠i}(z_i·z_j))` | 否 | 是 | 否 | 否 | 否 | 否 |
| 6 | `nn_cos_max` | `max_i(max_{j≠i}(z_i·z_j))` | 否 | 是 | 否 | 否 | 否 | 否 |
| 7 | `spectral_effective_rank_normalized` | `exp(-Σ_l p_l log p_l) / min(K,D)`，其中 `G=ZZᵀ`，`λ=eigvalsh(G)`，`λ=max(λ,0)`，`p=λ/Σλ` | 否 | 是 | 否 | 否 | 否 | 否 |
| 8 | `pair_iou_mean` | `mean({IoU(box_i,box_j) : i<j})` | 否 | 是 | 否 | 否 | 否 | 否 |
| 9 | `pair_iou_q90` | `np.quantile({IoU(box_i,box_j) : i<j}, 0.90, method="linear")` | 否 | 是 | 否 | 否 | 否 | 否 |
| 10 | `pair_iou_fraction_gt_05` | `count({i<j : IoU(box_i,box_j)>0.5}) / (K(K−1)/2)` | 否 | 是 | 否 | 否 | 否 | 否 |
| 11 | `center_distance_mean_normalized` | `mean({||c_i−c_j||₂ : i<j}) / ||(xmax−xmin, ymax−ymin)||₂`，分母为包住全体候选框的 bbox 对角线 | 否 | 是 | 否 | 否 | 否 | 否 |
| 12 | `log_area_std` | `std({ln((x₂−x₁)(y₂−y₁))}_{i=1}^K, ddof=0)` | 否 | 是 | 否 | 否 | 否 | 否 |

这里 `z_i` 是 float64 重新归一化后的第 i 个 crop embedding，`G` 是单位向量 Gram 矩阵，`λ` 是其特征值。谱熵中 `p_l=0` 项贡献为 0；数值负特征值截为 0，不依据结果阈值化微小正值。`c_i` 为 `xyxy` 框中心。`nn_cos_mean` 已锁定为唯一主要 ambiguity measure；`spectral_effective_rank_normalized` 是 Gram 矩阵的有效秩，不是 query 排名。

该模块按输入签名没有 query、scorer、分数/排名、winner、target identity、GT、label 或 objectness 参数。因此若同一张图和相同 K 对应的 crop embeddings 与 boxes 相同，即使换 query，`V_pure` 也相同。这个承诺只针对给定候选集上的函数依赖：受控 set 的上游表达式/GT 构造仍可能影响 set membership，不能据此推断端到端统计独立或因果正交。标签以及 S/Q 的拼接应留在后续独立模型层。

## 合成验证范围

`tests/test_v4_pure_features.py` 只构造小型内存数组，覆盖签名限制、固定顺序/维数、候选置换不变、所有 pair 的 `i<j` / 无对角、线性分位数、重复向量与框、正交向量的有效秩、几何平移和等比例缩放不变、`IoU=0.5` 严格边界、`K>=5` 与非法/零/非有限输入拒绝。它不读取真实缓存、GT、标签或 test endpoint，不实现或回归镜像旧 V8，也不运行训练或评估。

执行命令：`E:/conda/envs/deepminer/python.exe -B -m pytest tests/test_v4_pure_features.py -q`。2026-10-05 合成测试通过：12 项。
