# 实验设计与统计审查

2026-10-03。由综合评审按主题整理；不改动任何实验结果。完整结论见 strict_research_review_2026-10-03.md。

### 2.1 E-AURC 与 RER 并未彻底排除 accuracy 的影响

依据：`src/ccg/metrics/selective.py` 的 `oracle_aurc` / `e_aurc` 说明，`docs/research_protocol.md:799–849`。

代码对 E-AURC 的数值定义可以使用；错误在解释。减掉 oracle AURC，只消除了最佳可达风险曲线的基线，并不使指标变成与准确率无关的纯排序量。

设准确率为 a，置信度完全没有信息且独立于 correctness，在总体极限下：

```
AURC_random = 1-a
AURC_oracle = (1-a) + a ln(a)
E-AURC_random = -a ln(a)
```

因此，即使置信度区分能力始终为随机水平，E-AURC 也会随着 a 改变。这一问题已有选择性分类文献明确讨论，不能把它当作完全解决了的度量问题。[Traub et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf) §2.4 指出 e-AURC 仍受分类性能影响。

RER@c 的理论上限也依赖 a。总体形式为：

```
RER_oracle(a,c) = 1                          if a >= c
                  a(1-c) / [c(1-a)]         if a < c
```

项目 corrected cosine 的 AUROC K5→K50 为 0.7413→0.7366，差异很小且区间跨零；与此同时 RER@50 从约 0.384 降至 0.122。其 accuracy 约 0.5349→0.1879，使 RER@50 的上限从 1 降至约 0.2314。把两端观测值除以各自上限，约为 0.384→0.527。这不是一个推荐的新主指标，只是反例：原始 RER 下降不等价于“置信度排序一定变差”。

影响：A5 Route A 的“Selective discrimination”命名与据此声称两个 scorer 都复制 discrimination failure 的论证过强。cosine 可以支持选择性效用随任务变难降低，不能据这一分支证明独立的 correctness-discrimination 退化。

B3 主结果并不因此全部失效：它还具有 AUROC 下降证据。应重新区分三个问题：

- accuracy：目标选择是否更难；
- discrimination：置信度能否区分正确与错误；
- calibration：置信值是否对应正确频率。

E-AURC、RER 保留为实际选择性效用指标，避免称为完全 accuracy-independent。AUROC 也不是对所有分布变化免疫：K 改变 correctness 后，正确/错误两类的组成变化，仍需明确其比较对象。

### 2.3 直接平均三个 seed 的区间端点，不能当成均值的 95% CI

依据：`scripts/run_phase1.py:1291–1299`、`scripts/run_phase1f.py:925–964`、`scripts/run_phase05.py:878–890`；Phase 1 `sufficiency_gate.json.notes` 明确披露端点平均。

每个 seed 的 image-cluster paired bootstrap 是合理设计。问题在后续聚合：mean(ci_low)、mean(ci_high) 一般不等于 seed-mean 统计量 bootstrap 分布的相应分位数，没有其 95% coverage 保证。

这是统计汇总方法错误，不是“所有显著性都作废”。逐 seed 的区间仍可成立，强正结果也可能保持方向。未经正确重算，不能承诺总区间与 gate 判定完全不变。

建议：针对三个固定 seed 的均值效应，每轮共享同一 image-cluster 抽样索引，分别计算每个 seed 的比较效应，再取三个效应的均值，最后对这些均值取分位数。明确这衡量的是固定三个模型条件下的测试样本不确定性。seed 间均值±标准差另列；只有三个 seed，不宜冒充充分刻画训练随机性的层级推断。

不要混淆 mean(metric_per_seed) 与 metric(mean_predictions)：它们是不同估计对象。新修正应另存、标注统计纠正，不覆盖旧记录。

## 4. 实验设计的合理性与边界

### 4.1 GT 保证目标在场与唯一化：合理诊断，但估计对象需要写出来

依据：`src/ccg/data/manifests.py:592–622`、`src/ccg/data/candidate_sets.py` 与协议 §5。

目标采用 max-IoU proposal，强制纳入候选集合，移除其他与 GT IoU≥0.5 的候选；主跨 K cohort 又条件于能够形成 K50。这样可以分离目标缺失，避免同一正确目标存在多个有效框，是可辩护的受控实验。

但测试分布由 GT 协助形成：条件于目标覆盖、可构造最大 K、唯一目标清洗后的候选选择。它不是未经干预的 detector top-K 管道。真实部署中可能目标缺失、多框同时合法、按 objectness top-K 取框，候选数量与召回同时变化。

论文方法第一段就应声明这个条件，而不是只在 Limitations 脚注披露。主结论应落在 controlled candidate-based grounding。

### 4.2 Nested sets 给出 accuracy 单调性，不给出 AUROC 单调性

对于候选独立 scorer，在固定 tie-breaking 下，小集合中已选错的样本不能因仅加入新候选而变对；正确样本可能转错。因此 accuracy 下降的存在性本身没有多少新意。AUROC、calibration 与选拒风险没有相同的简单必然性，才是需要数据回答的部分。

应在论文中先给出这一结构事实，再提出经验问题，避免把常识性下降包装成发现。K 的变化与新增候选的具体身份也不可完全分离；项目识别的是所定义扩张路径上的效应，不是任何构成下 K 的普遍因果效应。

### 4.3 Same-category 操纵还存在两项待审计风险

第一，`manifests.py:600–622` 采用 target proposal 在全 GT 中最近 IoU 对象的类别。它未必等于 referring target annotation 的类别，例如重叠物体。这是设计事实与潜在错配，不是已证实大量样本污染。应报告实际 mismatch 的数量/比例与排除后敏感性。

第二，m=0/2/4/8 数的是同类 proposal 条数，而非不同物体数量。同一 GT 物体可能被多个 proposal 表示；同类列表还按 bank index（关联 objectness 排序）取前缀。因此不宜写“独立同类竞争对象数量的纯因果效应”。应报告 distinct-GT-object 数量、objectness、面积与 IoU 分布；当前主张限定为所定义 same-category proposal 制度下的收益变化。

manipulation check 比没有检查强，但不同 backbone/family 的检查通过也不能自动消除所有上述混淆。

### 4.4 现有稳健性验证各回答不同问题

| 证据 | 能支持 | 不能支持 |
|---|---|---|
| V2-G | 所测几个冻结编码器/对应训练头下有相似方向 | 所有 grounding 架构普遍失效；独立视觉域复制 |
| D1 reviewed | 删除被复核否定表达后结论仍在 | 所有框、类别与指称歧义都被完全纠正 |
| D2 RefCOCO | 同批图像上的表达分布迁移 | 图像不相交外部验证 |
| RefCOCOg strict | 共享 COCO 域中，未与 RefCOCO+ 重叠图像的外部验证 | 跨视觉域；hard-amplification 已复制 |
| V2-P | 相同冻结打分栈面对不同 proposal family 时现象仍在 | 各族 end-to-end 训练后仍如此；差距已经因果识别 |
| M1/M2/M3 | 这些受限实现/训练/选择协议未带来稳定改善 | 复杂模型必然无用；研究不可能被修复 |

V2-G 的“三个 backbone”也不完全等于同构的三次复制：`v2a1_result_record.json:32–33` 记录 B0 比较及操纵指标与新协议不同，B0 dose=NA。宜写“两个新增 backbone 按一致协议复制，并与原始 B0 证据方向一致”。backbone 轴与 proposal 轴分别展开，并非完成了三乘三全因子验证。

冻结 B3 在别的 proposal family 上评估可以研究上游变化带来的部署分布变化，但也引入了 scorer 对 proposal 族的适配问题。需要说明你测的是共享冻结 scorer 的迁移行为，不能包装成 proposal generator 的固有可靠性排名。

## 5. 研究治理与公开材料需要纠正的事实

1. README 的 RQ4 仍是 target omission，当前材料的 RQ4-M1 则是 transition–confidence mechanism。应按最终科学问题重新编号，将原 omission 路线明确记为未进入本文；机制章节无需继承旧阶段编号。
2. README 的主线冻结声明、single backbone/proposal 和 final_registry 入口没有清楚反映 V2/RQ4 的最终证据，读者会得到过时印象。保留历史 V1 freeze，但当前首页必须有清晰的最终总索引。
3. 蓝图 `docs/paper_blueprint_v1.md:267` 声称所有 V2 不新增训练参数，不正确。V2-G 有新 backbone 对应的 scorer/reliability 模型训练；V2-M/M3 也有训练/拟合。只有特定 proposal 和机制复用轴 new_training_parameters=0。应区分“编码器冻结”“grounding 预测固定”“某轴不训练”。
4. `final_result_summary.md:9,216` 对所有 V2 一概说协议均预先冻结，与 `experiment_log.md:1395–1401,1445–1451` 的记录不符。D1/D2 对完整 pre-result config 的证据不足，日志限定为输入 hash 冻结；D2 还发生过目标重定义。哈希证明身份一致，不自动证明完整分析计划预先承诺。不能仅从文件缺失认定没有任何先验约束，但当前可展示证据不足以支持统一强措辞。
5. 各阶段结果前制定下一阶段计划，是 staged study，不能消除研究者已经看过同一测试分布结果所带来的自适应性。没有证据据此指控直接 test 数据泄漏；但同一测试集反复指导研究路线，仍须与一次性独立确认区分。新语义消融最好留出新的未观察确认子集或明确标记探索性。
6. Gate 是研究预算与继续路线的操作标准。阈值 0.02/10%/5pp 并非某个应用的普适实用性边界。PASS/INCONCLUSIVE/STOP 应保留在日志，但正文主要展示效应、区间、条件与局限，少用全大写判定替代解释。

