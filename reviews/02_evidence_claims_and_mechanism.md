# 证据、结论与机制审查

2026-10-03。区分数学错误、未识别的解释及合理限制；不能将尚未检查频率的风险写成已发生的污染。

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

