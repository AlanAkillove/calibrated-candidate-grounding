# 严格研究审查：实验设计、证据链与论文组织

审查日期：2026-10-03。对象：当前工作区的研究协议、结果材料、论文蓝图与影响实验解释的核心逻辑。

本审查未训练模型、未改候选集、未修改历史协议或结果；并非逐行代码审计。三名 subagents 分别审查设计、证据和论文叙事，主审交叉核对关键结论。文中数学反例及四角计算属于审查推导，不应伪装成项目新实验。现有 `audit/` 和 `docs/paper_blueprint_v1.md` 为审查前已有文件，本次未改。

## 1. 总评

项目具备形成优秀毕业论文的实验材料，但当前材料库还不能直接作为一篇科学论证完整的论文。最明显的问题不是实验少，而是结论层级被拉高：操作性 gate 被写成科学判定，模型外推失败被写成信息不足，指标归一化被写成彻底消除基础错误率影响，统计分解被写成接近机制解释。

严格评价：工程可追溯性强，受控实验有价值；部分数学解释与统计汇总必须修正；机制识别仍弱；当前论文蓝图依然有“把全部阶段包装成贡献”的倾向。不能因为记录详细、种子多、修正案多，就认为所有推理成立。

最值得保留的结果是：

1. B3 在相同表达式、嵌套候选集上的 correctness-discrimination 随 K 增长降低。原主表 AUROC_correct 从 0.8426 降至 0.7904；后续 backbone 和 proposal 实验表明，这不是只在一个配置上观察到的孤例。
2. 在固定 grounding 预测、固定同一评估制度下，E1b 相对 Stats 提供额外的可靠性预测信号；受控同类 proposal 条件下增量显著扩大，且有固定 K 的剂量曲线。
3. 图像不相交的 RefCOCOg 子集支持嵌入特征增量的外部验证，但不能支持 hard-amplification 的外部复制。
4. 对复杂模型失败、外部操纵失败和旧解释撤回有保留记录。它们可帮助限定结论，不能独立当成科学贡献。

## 2. 必须修正的已证实问题

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

### 2.2 “统一 temperature 对 D_conf 贡献为零”是数学错误

依据：`results/v2_rq4_mechanism/m1_transition_confidence/protocol_freeze.json:96`、`report.md:124`、`docs/paper_blueprint_v1.md:27,106`。

AUROC 对最终标量置信值的严格递增变换不变，这是正确的。但 logits 的 temperature scaling 不一定是最大 softmax 置信值跨样本的统一递增变换。temperature 保持单一样本内部的 argmax，不保证不同样本的 MSP 排序不变。

反例：

| 三候选 logits | T=1 的 MSP | T=10 的 MSP |
|---|---:|---:|
| A=[2,0,0] | 0.7870 | 0.3792 |
| B=[1,0,-100] | 0.7311 | 0.5250 |

两样本的 MSP 排序反转。若 A 正确、B 错误，对应 AUROC 就会变化。项目自己的 `results/phase0a_corrected/correctness_auroc.csv` 中 native/global-T 两列也有不同数值。

因此不能用单调不变性排除 temperature 影响，也不能从 per-K 最优温度只漂移约 10% 推导“尺度不是机制”。温度漂移比例与其对排序的影响没有一般性的等价关系。`temperature_fit.json` 的末尾实际还写着 K-drift partly a per-K sharpness effect，与材料中的全盘排除措辞并不一致。

更严谨的表述：“在所采用的温度拟合目标与协议下，全局温度校准没有消除观察到的跨 K 退化。”若想进一步排除尺度解释，应直接比较温度调整前后的 AUROC、风险曲线及 calibration；且 NLL 最优温度不是 AUROC 或选择性风险最优温度。

### 2.3 直接平均三个 seed 的区间端点，不能当成均值的 95% CI

依据：`scripts/run_phase1.py:1291–1299`、`scripts/run_phase1f.py:925–964`、`scripts/run_phase05.py:878–890`；Phase 1 `sufficiency_gate.json.notes` 明确披露端点平均。

每个 seed 的 image-cluster paired bootstrap 是合理设计。问题在后续聚合：mean(ci_low)、mean(ci_high) 一般不等于 seed-mean 统计量 bootstrap 分布的相应分位数，没有其 95% coverage 保证。

这是统计汇总方法错误，不是“所有显著性都作废”。逐 seed 的区间仍可成立，强正结果也可能保持方向。未经正确重算，不能承诺总区间与 gate 判定完全不变。

建议：针对三个固定 seed 的均值效应，每轮共享同一 image-cluster 抽样索引，分别计算每个 seed 的比较效应，再取三个效应的均值，最后对这些均值取分位数。明确这衡量的是固定三个模型条件下的测试样本不确定性。seed 间均值±标准差另列；只有三个 seed，不宜冒充充分刻画训练随机性的层级推断。

不要混淆 mean(metric_per_seed) 与 metric(mean_predictions)：它们是不同估计对象。新修正应另存、标注统计纠正，不覆盖旧记录。

### 2.4 精确 Shapley 分解成立，但当前主推解释掩盖了交互与抵消

依据：`results/v2_rq4_mechanism/m1_transition_confidence/point_decomposition.csv` 的 mean3 行、`docs/paper_blueprint_v1.md:299–327`。

恒等式 residual=0 说明计算与定义相符；不能验证所选因素就是正确机制，也不能使归因具有唯一性。DETR mean3 四角值为：

| | p5 | p50 |
|---|---:|---:|
| y5 | A00=0.799356 | A01=0.584978 |
| y50 | A10=0.657804 | A11=0.714199 |

保持 p5，只切换标签，AUROC 损失 0.141553；保持 p50，同样切换标签，损失为 -0.129221，即 AUROC 反而提高。二者平均才得到 D_label=0.006166。

交互项：

```
I = A11-A10-A01+A00 = 0.270774
```

它远大于总退化 0.085157。对称 Shapley 将交互平均分配到两因素，因此“label 分量很小”不能解释为“错误转移不重要”。GDINO label 分量负值也不等于错误增加有保护作用。

应该说：“在所选对称两因子 Shapley 统计记账下，净退化更多分配给 confidence 分量，同时存在显著的路径交互与正负抵消。”这里“显著”若指统计显著需另算交互 CI；现阶段只应说交互项数值很大。

主图应展示四角、两条路径和有符号分量。95% 或 103.9% 等贡献比例不应脱离符号与交互单独作为摘要卖点。比例可大于 100%，不属于互斥原因的占比。

蓝图用 F/E 平均置信值接近解释抵消也不充分：AUROC 是成对排序量，均值不足以决定它；切换标签会改变正负集合与比较对权重，“固定分组权重”描述不准确。

## 3. 必须收缩、或通过针对性实验补足的结论

### 3.1 “Score information insufficient”混淆了信息、学习与外推

依据：`README.md:84–85`、`results/final_registry/claims.csv` C2、Phase 0.5 `selection.json` 和 `sufficiency_gate.json`。

可靠性模型只在 K5/K10 学习与选型，再在 K20/K50 评估。失败可能来自：没有额外信息、模型表达能力不足、训练未学到、特征尺度/聚合随 K 改变、训练目标与测试指标不一致、缺少大 K 训练支持。当前实验不能区分这些解释。

特别是 score_deepsets 在 tune 阶段已只有约 0.603/0.596/0.641 的 AUROC，而 MSP 平均约 0.820；K50 为 0.553。它在 ID 都明显弱于一个能由完整分数向量算出的简单统计量，因而不能作为“完整分数集的信息已经充分挖掘”的可靠证据。

应写：“在限定的小 K 训练及轻量模型设置中，所测试的分数表示未改善大 K 外推表现。”不要写“分数本身不足以解释退化”或“必须引入 set semantics”。

如果保留 sufficiency 主题，优先补一个受控的 ID/OOD 对照：同一 score 特征、同一模型容量，分别采用小 K 训练、大 K 的独立训练/验证子集训练；最终 test 只作评估。这能区分信息限制和外推限制，不需要追加大量架构。

### 3.2 E1b 的增益尚未定位到候选间交互

依据：`src/ccg/semantic/features.py:49–66`、`docs/research_protocol.md:930–933`。

16 维“semantic”特征同时包括：

- CLIP query–crop 分数、margin、entropy、CLIP 与 B3 排序关系；
- winner–candidate 的视觉余弦、密度等候选间关系。

因此 E1b 的额外信号可以来自另一路 scorer 与 B3 的互补或不一致，不必来自候选间语义竞争。现有 E2/E3 消融、系数分析和 matched-score 分析无法替代 feature-block 对照。

最有价值的补实验是四组，同一划分、训练、选择规则、grounding 预测：

| 模型 | 输入 |
|---|---|
| S | 原 Stats |
| S+Q | Stats + query–crop / alternate-scoring 特征 |
| S+V | Stats + candidate–candidate 特征 |
| S+Q+V | 完整 E1b |

核心比较是完整 E1b 相对 S+Q 的增量，并在 matched random、hard 和固定 K 的 dose 中观察。Q 组/ V 组特征划分需结果前写清，例如 winner 的 CLIP rank 属于 Q；不以事后系数选择特征。

若交互组有独立增量，可强化“候选关系”；若没有，论文应改写为“不同冻结打分视角的互补可靠性信号”。两种结果都能形成可信论文。

### 3.3 机制诊断的被解释对象曾发生错位

依据：`results/v2_proposal_robustness/p2_m_mechanism/mechanism_report.md:19–48` 与 §5。

V2-P2-M 的 H1c 恒等于 accuracy drop，三族约为 RPN 0.3605、DETR 0.2957、GDINO 0.4016；论文试图解释的突出差异却是 AUROC drop，分别为 0.0517、0.0852、0.2366。RPN/DETR 的排序都不同。

以冗余分层匹配后 accuracy-harm gap 不缩小，不能直接证明 AUROC-gap 的语义解释已被否证。它可以说明“GDINO 的同类 proposal 数更多”这个事实前提不成立，以及所测数量冗余不足以解释所分析的 accuracy harm；无法排除语义混淆度、错误样本构成及其与 MSP 排序的关系。

撤回原先过强的解释是正确行动；改写成“未获得支持”，不要进一步声称语义竞争机制普遍被否证。COCO 同类别条数不等于 embedding confusability，也不等于表达式层面的歧义。

report 将未达到标注对象 IoU≥0.5 的框称作 proposals on nothing 也过强：阈值未匹配可能包括局部物体、尺度错误或非穷尽标注对象。只能说“未匹配到所使用的 COCO 标注对象”。Spearman 相关也不能不加条件直接转成“解释了几百分比方差”。

另外，`p2_c1_gdino/analysis_report.md:64–66` 有可直接核对的数值错误：称 GDINO K5 的 RER@50=0.8065 最高，但同表 RPN=0.8248 更高；称 GDINO accuracy 比 DETR 高 0.057，但 GDINO=0.8463、DETR=0.8806，实际低约 0.0343。0.057 是相对 RPN 的差。最低 E-AURC 也不能直接称作最佳 calibration。历史报告的自然语言解读应重新核对，不能只核验 CSV 和恒等式。

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

## 6. 引用材料中存在真实错误

本次核对了部分核心条目，不是完整文献综述：

- `literature_notes.md:30` 的 TransVG 作者/会场不正确。官方是 Deng et al., ICCV 2021。[官方论文页](https://openaccess.thecvf.com/content/ICCV2021/html/Deng_TransVG_End-to-End_Visual_Grounding_With_Transformers_ICCV_2021_paper.html)
- `literature_notes.md:64` 将 MMCE 写成 Naeini 等、AAAI 2015，不正确。MMCE 对应 Kumar、Sarawagi、Jain，ICML 2018。[PMLR 官方论文页](https://proceedings.mlr.press/v80/kumar18a.html)
- `literature_notes.md:68` 的 Ovadia 标题与会场不正确，原论文是 *Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift*，NeurIPS 2019。[官方论文页](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- ReCLIP 被简写为 CLIP+Grad-CAM 定位细化不准确。其核心区域打分包括 cropping/blurring，另有空间关系解析组件。[ACL 官方论文页](https://aclanthology.org/2022.acl-long.357/)

“需查证”标记作为内部笔记是合理的；它不是写作时使用错误引用的免责条款。现有材料含若干不应直接引用的条目，正式论文要逐一回到原文。

真正应对齐的文献轴是：候选式 grounding、CLIP REC、distribution-shift selective prediction、calibration/discrimination 区别、选择性指标缺陷、set 表示。应说明你增加了什么实验对象与控制，而不是罗列十几个模型名称。没有完成文献检索前不要声称 first/new benchmark 或一般性创新。

## 7. 建议的论文主线

### 7.1 当前证据足以支撑的一句话

**在受控、目标在场的候选式视觉定位中，候选集扩张会改变预测正确性及其与置信度的关系；冻结嵌入衍生特征提供有限而依赖候选组成的额外可靠性信号，现有轻量模型尚未稳定修复这种变化。**

这条主线承认：基础错误率与 confidence ranking 都可能变化；语义特征收益有条件；没有成功的修复方法；当前分解是记账而非因果。

若语义分组消融证实 candidate–candidate 特征有独立增量，再强化为“候选关系的价值随受控竞争增强”。若没有，不要强行维持该命题。

题目建议：**《动态候选集下视觉定位的置信度可靠性：受控评估与条件性语义增益》**。英文可用 *Confidence Reliability in Visual Grounding under Candidate-Set Expansion*。“Calibrated”容易暗示已经给出可靠修复方法，摘要需解释清楚。

### 7.2 用三个研究问题统领，而不是用实验阶段统领

1. 在同一图像与指称的候选扩张中，accuracy、correctness discrimination、calibration 各发生什么变化？
2. 小 K 学到的分数表示能否外推；冻结嵌入特征带来的增量来自哪部分，在哪种候选组成下更有价值？
3. 观察到的 AUROC 变化在四角统计分解中如何分配，该分配有怎样的路径交互与解释边界？

目标缺失若没有实质结果，应留在未来工作。不要因已有 RQ4 标签而保留一条没有回答的主问题。

### 7.3 章节组织

| 章节 | 需要解决的问题 | 主文保留 | 附录/仓库保存 |
|---|---|---|---|
| 1 引言 | 为什么 accuracy 不足以评价变化的候选接口 | 任务情境、具体缺口、2–3 条可验证贡献 | 不放实验时间线 |
| 2 相关工作与问题定义 | 与现有 REC / selective prediction 的区别是什么 | 三种可靠性概念、结构性 accuracy 单调性、最终 RQ | 模型/指标扩展笔记 |
| 3 受控评估设计 | 你实际在什么分布上估计什么 | GT 条件、nested/common cohort、训练与校准划分、bootstrap | hashes、所有 gates/amendments |
| 4 候选扩张下的可靠性变化 | 除了目标更难选，还有什么经验事实 | 主效应曲线、absolute 数值、修正统计、多配置精简复制 | backbone/数据/族的完整逐 seed 结果 |
| 5 额外信息何时有用 | 外推限制、alternate scorer 与交互、组成条件性 | score对照、E1b分组消融、matched hard、固定K dose、外部边界 | 全模型 zoo、M1/M2/M3训练细节 |
| 6 统计分解与讨论 | 变化可以如何记账，不能解释什么 | 四角与两路径、交互、有符号归因、撤回解释 | 完整pairwise重构与机制报告 |
| 7 结论与局限 | 哪些问题已回答、哪些仍开放 | 受控分布、共享视觉域、没有成功修复、未识别因果 | 可复现性长记录 |

毕设可以扩展背景、理论推导和实现说明；结果章节仍按问题组织。论文不能按 Phase0→0.5→1→1F→V2-G→D→P→M 的顺序堆目录。

当前 blueprint 的问题：它虽然按主题命名章节，但仍要求大量 gate 判定表、独立 robustness 章、独立 negative-results 章及全套机制图，造成主题重复。稳健性证据应嵌入它验证的具体命题；负结果应放在它否定的假设旁；完整工作历史属于仓库。

### 7.4 四张核心图与两张主表

1. **实验接口示意图**：同一图/表达、相同目标、nested候选扩张，明确GT条件与冻结组件。
2. **可靠性变化图**：accuracy / AUROC / risk或E-AURC分面展示，主配置带区间，其他backbone/family简洁叠加；三者不能混成“校准曲线”。
3. **条件性信息增益图**：Stats / S+Q / S+V / Full在matched random、hard、dose的增量，固定K时是最有解释力的比较。
4. **四角与路径图**：展示交互和有符号归因，取代仅画confidence占比的图。

主表一：关键绝对指标及有效样本数；主表二：核心命题、验证条件、效应大小和局限。不要在摘要和主文反复展示+226%/+865%等比例而缺少起点，低基数会夸张读者感受。

每个实验段落遵循：待检验命题→如何区分竞争解释→关键结果→支持范围→仍无法排除什么。没有对应论点的结果应进附录。程序崩溃的渲染修复不值得占用主要 Limitations 篇幅。

## 8. 补工作优先级与停止规则

### P0：写正文前完成

1. 纠正 temperature 不变性、E-AURC/RER 的解释与所有相关结论措辞。
2. 对作为论文主证据的 seed-mean 效应正确构造区间；只重算必要统计，不重新训练或覆盖历史值。
3. 展示四角、两路径和交互；撤掉“标签变化不重要”及仅用均值解释抵消的句子。
4. 统一 README、summary、claims、blueprint 的最终范围、训练状态、RQ编号与冻结证据等级；核实引用。

### P1：最值得补的少量实验

1. E1b 的 Q/V 分组消融：直接决定“候选语义交互”是否可作为论文中心。
2. score-only 的 ID/OOD 对照与拟合能力检查：直接决定“信息不足”是否应改为“外推受限”。
3. same-category 的 target-category mismatch、distinct-object 和候选分布敏感性审计：主要用于保障最强语义结果的解释。

这些新分析应另立短协议、单独保存。如果继续使用已观察测试集，明确属于探索性，不能靠提前冻结本次脚本就声称没有自适应风险。

### P2：仅在目标确实需要时

自然 detector top-K 或不做GT答案清洗的评估，有助于回答部署相关性；更强现代grounding模型可用于扩大架构范围。二者并非当前受控论文成立的前提，但若没有，就相应限制题目与结论。

不建议此时增加第四个backbone、第四个proposal family、更多attention模块、更多mixture或硬补target-absence。它们不会优先解决当前识别缺口。

停止条件：当核心数学解释正确、主要CI正确、语义增益来源有明确结论、主问题都能被现有图表回答时，停止扩展实验。不能以“所有gate都通过”作为完成标准。

## 9. GitHub 与复试展示

首页应能让读者在一分钟内知道问题、受控设定、三条关键发现和四条边界；当前README更多是阶段性实验入口。

建议设置：

- 一个最终证据索引，清楚区分V1 registry与V2/RQ4的来源；修正结果另有version与解释。
- 一条不需全部原始数据的轻量再生成路径：从可分发的汇总/匿名预测材料生成主表主图。原始数据与模型复现实验另外说明。
- 少量真实案例：K5对/K50错、置信度仍高、额外特征能/不能识别；所有案例按预设规则选择，并明确它们是说明，不是额外统计证据。
- 保留负结果与撤回记录，但主入口不要求访客先读两千行修正案。

复试中最有说服力的是你能解释：为什么accuracy下降并不新奇；为什么E-AURC仍受基础错误率影响；为什么温度保持argmax却不保证MSP跨样本排序；为什么Shapley小分量可能是抵消；为什么会主动收缩原结论。对此没有清楚回答，堆积实验数量和哈希不会提高研究可信度。

最终判断：保留这个项目，修正证据链，收缩没有被识别的主张。现有材料适合形成严谨的受控经验研究；目前不足以主张通用可靠性修复、真正因果机制或候选集合的信息论必要性。
