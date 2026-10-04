# 分析阶段文献调研：四角、路径、可靠性与解释边界

调研日期：2026-10-04。项目证据基线：`2c832cf0ec44bd9d18302ce4db7a222d752a99e5`。

本文件服务于修补后的分析与论文写作。它不新增模型、候选条件或实验，也不把文献观点冒充本项目的实测结论。数值引用现有 Research Repair v1 产物；公式推导与文献结论分别标明。

后续进展：本轮理论完善已形成[命题、证明与经验核验](../docs/theory_analysis_v1.md)。下文 §5.2 已统一为仓库的 F=由正确转为错误、E=两端错误；前版自定义 F/E 与仓库相反，虽不破坏其自定义下的恒等式，却不适合直接对应 CSV。另纠正前版遗漏历史配对结果文件的记录；新分析的增量是路径展开及正式区间，不声称首次构造该恒等式。

## 1. 调研方法与证据等级

这是围绕具体论证问题的定向叙述性调研，**不是系统综述，也不是证明本项目“首次提出”的新颖性检索**。本轮登记 30 项来源，其中 29 项至少读到原始摘要，另 1 项仅核对元数据；15 项进一步读取了相关正文段落或公式。没有把“检索到全文”记为“完整精读”。

检索围绕以下问题，先查原始文献，再核对会议、出版社或作者页面：

| 检索问题 | 代表性检索词 | 优先来源 |
|---|---|---|
| 路径平均的数学依据是什么？ | Shapley decomposition; baseline Shapley; order dependence | RAND、PMLR、原始论文 |
| 何时可以称为因果解释或中介效应？ | causal Shapley; direct indirect effects; sequential ignorability | NeurIPS、作者论文、期刊 |
| 交互、分组显著与机制之间有什么区别？ | statistical interaction mechanistic interaction; difference significant not significant | 原始方法论文 |
| AUROC 和选择性指标究竟测量什么？ | AUC probability ranking; E-AURC flaws; population AURC | PubMed 原始摘要、NeurIPS、PMLR |
| 校准和温度有哪些被误用的性质？ | confidence top-label calibration; calibration error bias; temperature scaling | ICML、ICLR 作者版本、AISTATS |
| 特征消融能证明信息必要性吗？ | algorithm-agnostic variable importance; oracle predictiveness | JASA 作者手稿 |
| 小 K 训练失败能否证明分数表示不足？ | Deep Sets representation limitations; uncertainty dataset shift | NeurIPS、ICML |
| 当前置信区间、事后分析的外推边界是什么？ | clustered bootstrap; benchmark variance; adaptive holdout; directional differentiability; equivalence | 原始统计与学习理论论文 |

纳入规则：来源能直接解释本项目一个论证环节；使用原始研究或原始方法教程；出处可核对。排除规则：博客、二手综述中的转述、未定位的引用、与当前目标在场范围无关的实验宣传。PubMed 仅用来读取原始摘要及元数据，不作为“已读全文”的替代。

阅读深度使用三个标签：**F**＝读取相关正文/公式，不代表全文逐页精读；**A**＝原始摘要、引言片段或页面元数据；**M**＝仅元数据。访问失败、付费墙、扫描件和验证码在文献卡片中单独记录。未将论文全文复制进仓库；保留链接、定位、独立转述及项目分析。

判断也分三类：**文献直接支持**、**据文献定义作出的项目推导**、**现有项目产物支持的经验判断**。第三类不能靠引用替代数据，第二类不能伪装为作者已经研究了我们的系统。

## 2. 本轮最重要的判断

| 判断 | 依据性质及文献 | 对本项目的约束 |
|---|---|---|
| 两因素四角的两路径平均是精确的 Shapley 分摊 | 定义依据 [R02][r02]；项目代数推导见 §3 | 可以说精确分解；不能从加和恒等式推出因果份额 |
| Shapley 公理不自动赋予因果语义 | [R03][r03]、[R04][r04] 直接区分价值函数和干预定义 | 必须说明我们的价值函数是标签/置信度数组替换 |
| 中介分析需要定义干预和识别假设 | [R05][r05]、[R06][r06] | 当前四角不是已识别的自然直接/间接效应 |
| 路径反号与平均接近零可以同时成立 | 项目公式、现有路径表；解释依据 [R02][r02] | 小标签分量不能证明错误转移不重要 |
| AUROC 比较的是正负样本对的排序 | [R09][r09] 直接支持；S/F/E 分解是本文件推导 | 改标签会改比较对象，不能仅归结为平均置信度或类别比例 |
| 指标交互不等于内部语义机制 | 交互区分原则 [R07][r07]；项目推导 | `I` 只说明所选四角在 AUROC 尺度上非加性 |
| 一个条件显著、另一个不显著不证明增益差异 | [R08][r08] | “hard 条件增强了增益”需要直接差值及配对区间 |
| 保持 argmax 不等于保持 MSP 跨样本排序 | [R11][r11] 的实际保证；本项目温度反例 | 不能以统一温度宣称排除了尺度解释 |
| 当前 pooled ECE 属于 confidence calibration | [R12][r12] 的定义与本项目函数接口对照 | “严格 top-label calibration”需要额外条件化，不能仅凭函数名称认定 |
| ECE 有分箱估计偏差，bootstrap 有适用条件 | [R13][r13]、[R23][r23] | 不能用文献为当前自适应 ECE 区间无条件背书，也不能据此断言所有区间失效 |
| 有限算法的消融增益不等于内在信息必要性 | [R16][r16] 区分 oracle 与具体模型 | Full−(S+Q) 是指定学习流程下的增量预测效用 |
| 置换不变性和表达定理不保证跨 K 泛化 | [R17][r17]、[R18][r18] | ScoreDeepSets 的失败需区分训练支持、优化、表示和选型 |
| 固定模型的测试 bootstrap 不覆盖完整训练随机性 | [R20][r20]、[R21][r21] | 当前区间条件于三个已训练 seed 模型及既定划分 |
| 测试集未用于选 epoch，不等于独立确认 | [R22][r22] | 根据已观察结果补设分析仍是结果驱动的补充证据 |
| 不显著不等于等效 | [R24][r24] | 类别敏感性可报告实测变化；没有事前等效界值就不宣称等效 |

**建议采用的总判断：解释审计已足以约束若干过强主张；对内部原因的因果机制识别仍不足。** “充分”只针对明确的审计问题：数学解释是否正确、路径是否稳定、某种统计解释是否得到支持。它不表示所有替代解释已经排除，也不表示审查过程形成了新的模型理论。

## 3. 四角到底分解了什么？

### 3.1 定义：数据集级指标的两块输入替换

令同一 image–sentence cohort 上的正确性向量为 `r5, r50`，置信度向量为 `p5, p50`。每个 seed 内先计算：

\[
A_{00}=\operatorname{AUC}(r_5,p_5),\quad
A_{10}=\operatorname{AUC}(r_{50},p_5),\quad
A_{01}=\operatorname{AUC}(r_5,p_{50}),\quad
A_{11}=\operatorname{AUC}(r_{50},p_{50}).
\]

第一下标表示替换正确性向量，第二下标表示替换置信度向量。这里的 label 指 **预测正确性标签**，不是 target 的语义类别。对跨 seed 汇总，先计算各 seed 的这些量，再取均值；不对预测向量做 ensemble。

共同 cohort 是每个 family 内的要求，不代表不同 family 的可行性筛选保留了完全相同的表达。跨 family 表格用于检查模式是否复现；若要正式比较 family 效应大小，需另行对齐共同 cohort，不能直接把不同样本集的点估计相减称为配对差值。

两条路径：

\[
00\xrightarrow{L_0}10\xrightarrow{C_1}11,\qquad
00\xrightarrow{C_0}01\xrightarrow{L_1}11,
\]

\[
L_0=A_{10}-A_{00},\quad L_1=A_{11}-A_{01},\qquad
C_0=A_{01}-A_{00},\quad C_1=A_{11}-A_{10}.
\]

两条路径分别满足 `L0+C1 = C0+L1 = A11−A00`。定义

\[
I=A_{11}-A_{10}-A_{01}+A_{00}=L_1-L_0=C_1-C_0.
\]

对称分摊为

\[
\phi_L=\tfrac12(L_0+L_1)=L_0+\tfrac12I,\qquad
\phi_C=\tfrac12(C_0+C_1)=C_0+\tfrac12I,
\]

\[
\phi_L+\phi_C=A_{11}-A_{00}.
\]

上述都是项目定义下的代数恒等式。若沿用“下降”方向的 `D_total=A00−A11`，则 `D_label=−φL`、`D_confidence=−φC`，不能在同一图表里混用正负方向。

### 3.2 与 Shapley 的确切关系

构造两玩家游戏，玩家是整个正确性块 L 与整个置信度块 C：

\[
v(\varnothing)=0,\quad v(\{L\})=A_{10}-A_{00},\quad
v(\{C\})=A_{01}-A_{00},\quad v(\{L,C\})=A_{11}-A_{00}.
\]

按两种排列平均玩家的边际变化，就得到上面的 `φL, φC`。这可视为对数据集级 AUROC 函数做**两块 baseline replacement Shapley 分解**。该识别依据 Baseline Shapley 的价值函数定义 [R02][r02]，是本文件对项目公式的对应推导。

它并非逐样本模型特征 SHAP，也不是学习得到的因果 Shapley。公理唯一性以既定玩家、基线和价值函数为前提；更换 K 对、指标或块划分，会改变所回答的问题。[R02][r02]、[R04][r04]

### 3.3 已有结果应强调路径反号，而非单个贡献百分比

下表由 `mechanism/four_corners_paths_interaction.csv` 的 `mean3` 行生成。数值为 AUROC 差值，未乘 100。

<!-- GENERATED_MECHANISM_TABLE_START -->

| Family | A00 | A10 | A01 | A11 |
|---|---:|---:|---:|---:|
| RPN | 0.840720 | 0.779588 | 0.733100 | 0.789022 |
| DETR | 0.799356 | 0.657804 | 0.584978 | 0.714199 |
| GDINO | 0.830757 | 0.793072 | 0.538183 | 0.594158 |

| Family | L0（p5） | L1（p50） | C0（r5） | C1（r50） | I | φL | φC |
|---|---:|---:|---:|---:|---:|---:|---:|
| RPN | -0.061132 | +0.055922 | -0.107620 | +0.009434 | +0.117054 | -0.002605 | -0.049093 |
| DETR | -0.141553 | +0.129221 | -0.214378 | +0.056396 | +0.270774 | -0.006166 | -0.078991 |
| GDINO | -0.037685 | +0.055975 | -0.292574 | -0.198915 | +0.093659 | +0.009145 | -0.245744 |

<!-- GENERATED_MECHANISM_TABLE_END -->

三个 family 的标签切换路径均反号：使用 `p5` 时替换标签降低 AUC，使用 `p50` 时替换标签提高 AUC。RPN 的对称标签分量接近零，并不意味着两条边际变化都很小；它来自明显的正负抵消。

置信度路径也不能一概而论：RPN/DETR 的 C0 为负、C1 点估计为正，而 GDINO 两条置信度路径均为负。RPN 的 C1 区间跨零，不能仅凭正点估计称为稳定改善。GDINO 的 φL 为正、φC 为负，因此按下降方向计算的置信度分摊可以超过总下降的 100%；这是抵消的算术结果，不是超过 100% 的因果解释。

相应区间来自 `mechanism/bootstrap_ci.csv`：共享 image-cluster 抽样、每次抽样内部计算各 seed 的路径及交互、最后平均效应，5,000 次 percentile bootstrap。路径、交互和对称量的协方差必须保留，不能用独立端点加减合成区间。

<!-- GENERATED_INTERACTION_CI_START -->

| Family | I 的 95% CI | L0 的 95% CI | L1 的 95% CI | φL 的 95% CI |
|---|---:|---:|---:|---:|
| RPN | [+0.103589, +0.131467] | [-0.075699, -0.046926] | [+0.044541, +0.067966] | [-0.013612, +0.008415] |
| DETR | [+0.241200, +0.302348] | [-0.163570, -0.119877] | [+0.108156, +0.150860] | [-0.021067, +0.009076] |
| GDINO | [+0.065463, +0.121808] | [-0.055938, -0.019861] | [+0.034775, +0.077688] | [-0.004540, +0.023026] |

<!-- GENERATED_INTERACTION_CI_END -->

可以写：“在所选端点和 AUROC 尺度上，正确性替换的边际变化显著依赖于所使用的置信度向量；对称标签分量掩盖了两条路径的抵消。”这里“显著”仅对应表中各路径/交互的条件 bootstrap 区间，不能改写为内部因果交互已识别。

不宜写：“95%/103.9% 的退化由置信度机制造成，因此正确性变化不重要。”带符号分量占总差的比值可以超过 100%，因为另一分量抵消了它；总差接近零时比值还会不稳定。先展示四角、路径与有符号绝对分量，百分比最多放附录。

`I` 是这个指标和端点上的离散非加性，不等于 Q/V 语义特征之间的交互，更不是物理机制的唯一签名。`I=0` 也不能证明底层过程独立。关于统计交互、尺度和机制解释的区分见 [R07][r07]；本项目对 `I` 的限制来自上述定义。

## 4. 为什么“受控设计有价值”与“中介机制未识别”可以同时成立？

### 4.1 不应把整个实验贬为纯相关性

对相同图像、query、target，按明确算法改变候选集合，冻结打分器和 tie-breaking，能够直接比较该**具体候选构造操作**下的输出变化。若推断对象仅为这组固定输入与操作，这比对不同图像的 K 作横截面相关分析更强。

但 K 只是集合大小。不同新增候选的分数、质量、类别、重复框与对象覆盖不同；“把 K 从 5 改到 50”若不说明构造规则，不是唯一干预。现有实验能支持既定 random/hard/dose 规则下的条件结果，不能直接推广到任意自然候选增长，更不能把 proposal 条数解释为独立物体数量。

这一区分是依据干预与效应定义 [R05][r05] 作出的项目判断，**不采用“没有随机分配就一律没有因果信息”的粗糙标准**。

### 4.2 四角的混合量不自动成为可部署系统或中介干预

计算依赖关系可以写为：

```text
候选集合与 query ─→ 固定 scorer 的分数向量 ─→ argmax ─→ 正确性 r
                                            └────────→ 置信度 p
target annotation ──────────────────────────────────→ 正确性 r
数据集上的 (r, p) ───────────────────────────────────→ AUROC
```

这只是实现中的计算依赖图，**不是通过数据识别的因果 DAG**。`r` 与 `p` 都受候选/分数/预测过程影响；AUROC 又是整个数据集上的函数。

`A10` 用大 K 的正确性评判小 K 的置信度。它是明确且有用的诊断量，却一般不是某个实际系统输出的可靠性：小 K 置信度原本描述的是小 K 所选框的正确性，大 K 可能已经选了另一框。

若要把这些混合量称为自然直接/间接效应，必须先定义结构模型、干预对象与潜在结果，解释如何在改变候选时保持某个中介的意义，讨论识别所需条件。Pearl 与 Imai 等人的中介定义涉及 `Y(t,M(t′))` 一类反事实量和明确假设；四个可计算指标相似的代数形式，并不能代替这些要求。[R05][r05]、[R06][r06]

Heskes 等的 causal Shapley 价值函数显式使用 `do` 分布和因果知识；普通分摊公理在其他价值函数下也可以成立。[R04][r04] 因此不能借“Shapley”一词绕过识别问题。

### 4.3 对论文机制章节的定位

建议章节名为“**候选扩张下的正确性–置信度关系：四角审计与解释边界**”。

正文分三层：观测到的变化 → 哪些统计解释足够/不足 → 仍未识别的内部原因。不要把最后一层写成“机制已经被否证”；当前证据通常只能说明某个被测代理解释没有获得支持，或不足以解释某个端点。

尤其是 H1c 的对象是 accuracy harm。它与 AUROC gap 的因变量不同；即使证实增加相似候选损害准确率，也不能直接说明正误判别排序为什么改变。未匹配标注对象的 proposal 也不能被称为“框里什么都没有”。

## 5. 比均值更有解释力的 AUROC 配对分析

### 5.1 指标的比较对象

对正确性标签固定、正负类都存在的情况：

\[
\operatorname{AUC}(r,p)=\frac1{n_+n_-}\sum_{i:r_i=1}\sum_{j:r_j=0}
\left[\mathbf1(p_i>p_j)+\tfrac12\mathbf1(p_i=p_j)\right].
\]

其概率排序解释有原始文献依据 [R09][r09]。用于 grounding 时，正类是正确预测、负类是错误预测；引用医学原文时应说明这种变量对应，而不是沿用疾病语义。

本项目的关键推论是：固定标签时，严格递增的标量置信度变换保留 AUROC；改变正确性标签时，正负样本对的集合随之改变。AUROC 对某些仅重加权类别先验的情形不变，不意味着它对“哪些样本变错”不敏感。平均 confidence 的变化幅度也不能刻画这些样本对的排序变化。

### 5.2 S/F/E 配对恒等式与后续分析

在独立候选打分、固定 tie-breaking、target 保留及严格嵌套的条件下，`r50 ≤ r5`。设：

- S：两端都正确；F：K5 正确而 K50 错误；E：两端都错误。
- G：K5 错误而 K50 正确。在上述条件下应为零；如果不为零，先检查条件，而非直接删样本。

对任意固定分数向量 `p`，定义 `U_AB(p)` 为 A 组样本相对 B 组样本的平均有并列修正的排序胜率。于是：

\[
\operatorname{AUC}(r_5,p)=
\frac{n_Sn_E U_{SE}(p)+n_Fn_E U_{FE}(p)}{(n_S+n_F)n_E},
\]

\[
\operatorname{AUC}(r_{50},p)=
\frac{n_Sn_F U_{SF}(p)+n_Sn_E U_{SE}(p)}{n_S(n_F+n_E)}.
\]

推导：小 K 的正类为 S∪F、负类为 E；大 K 的正类为 S、负类为 F∪E。按组展开 AUROC 双重求和即可。空组的配对项不定义但其权重为零，可省略；整个标签向量只有单类时 AUROC 不定义。

因此，`L0` 比较的是在 `p5` 上移除 FE、新增 SF 配对并改变共同 SE 权重后的净变化；`L1` 做同样的替换，但排序向量换为 `p50`。不同排序向量完全可能产生相反方向，无需先假设某个内部机制。

历史 `results/v2_rq4_mechanism/m1_transition_confidence/pairwise_auc_components.csv` 已保存这些组间胜率点估计，前版调研遗漏了该产物。本轮复用它，并用同一冻结预测补齐路径项与配对变化的正式共享抽样区间，见[配对排序分析](../results/research_repair_v1/theory_analysis/v1/analysis_report.md)。不能将重构旧结果记为新实验。

这些组按两个条件下的结果定义。可用于描述性诊断，不能直接当作预处理因果分层；“F 组是什么原因变错”仍需要额外机制证据。

### 5.3 图表建议

主图用四角方形和两条有符号路径，右侧放 `I, φL, φC` 及区间。下一张图按 S/F/E 展示配对数量权重与排序胜率，解释“谁和谁比较”发生了什么变化。分数均值、entropy、margin 的散点图只能作为补充，不能取代与 AUROC 定义一致的分析。

## 6. 可靠性指标的文献边界与需要更正的术语

### 6.1 三类问题分开回答

| 问题 | 主要对象 | 论文应报告什么 |
|---|---|---|
| 能否把正确预测排在错误预测前？ | correctness AUROC、配对排序 | 绝对 AUROC、变化、四角/配对分析 |
| 按该排序保留部分样本是否有用？ | risk–coverage 与固定 coverage 风险 | 曲线、实际 risk、相应增量；不要只给归一化比值 |
| 报告的概率能否对应经验正确率？ | confidence calibration、binary correctness scoring | 校准曲线、分箱口径、ECE/Brier/NLL；不能把排名提高称为概率校准已解决 |

选择性评估的对象及指标局限可参考 [R14][r14]、[R27][r27]、[R28][r28]、[R29][r29]。各指标可能方向不同，应解释任务目标，而非通过挑选有利指标“统一”结论。

### 6.2 新发现：现有名称并非严格 top-label calibration

Gupta 与 Ramdas 区分：

\[
\text{confidence calibration: } P(Y=c(X)\mid h(X))=h(X),
\]

\[
\text{top-label calibration: } P(Y=c(X)\mid h(X),c(X))=h(X).
\]

后者额外条件化预测标签。[R12][r12] 本项目 `top_label_ece(confidence, correctness, n_bins=15)` 与 adaptive 版本只在置信度上分箱，不接收预测标签，因此按该文献的严格定义属于 **confidence ECE / correctness-probability calibration**。

文献中存在名称不一致；应在正文明确实际公式，避免靠名称制造更强保证。局部候选 index 不是全数据共享的语义类标签，不能简单按 index 分箱就宣称修复了这个定义问题。

建议正文改用“置信度校准”，binary Brier 与 NLL 改用“预测正确性的二元概率评分”。保留旧 API 名称、历史协议名称与数值并注明兼容性即可。本轮只记录这一术语问题，没有重算指标或更改冻结计算。

### 6.3 温度：数学保证与项目反例各归其位

Guo 等的 temperature scaling 保持每个样本的 argmax，因此保持分类准确率。[R11][r11] 它没有保证不同样本的最大 softmax 概率排序不变。

本项目已测试的反例使用 logits `(2,0,0)` 与 `(1,0,−100)`：T=1 时前者 MSP 较高，T=10 时后者较高；两者 argmax 均不变。若正确性依次为 `(1,0)`，AUROC 从 1 变为 0。**这是项目数学反例，不是 Guo 原文中的实验。**

因此，“统一 logits 温度必然不改 MSP 排序”错误；“平均 MSP 仅漂移约 10%，所以尺度解释被排除”也没有逻辑依据。应引用当前 native/global-T/per-K-T 的实测指标，分别说明 accuracy、ranking、calibration 和 selective utility 如何变化。

### 6.4 E-AURC 与 RER：去掉某个基准不等于完全排除 accuracy

Traub 等指出 E-AURC 仍继承部分 AURC 评价问题。[R14][r14] 以下为本项目指标定义下的独立推导：随机置信度在总体上有常数风险 `1−a`，于是 `AURC=1−a`；减去连续 oracle 参考 `1−a+a log a` 后，`E-AURC=−a log a`，仍依赖准确率 a，而 AUROC 为 0.5（要求两类均有正概率）。

另一个已核验的有限样本例子：置信度 `(0.8,0.3)`、正确性 `(1,0)`，本项目 trapezoidal AURC 为 0.125，连续 oracle 参考为约 0.153426，E-AURC 为约 −0.028426。原因是有限积分规则与连续参考不同，不应裁剪负数或把该参考称为有限样本下界。总体 AURC 与有限估计量需要区分；[R15][r15] 提供这一研究背景，但没有验证本项目的 trapezoid 实现。

按 `RER(c)=(R(1)−R(c))/R(1)`，当 `0≤a<1`、`0<c≤1` 时，理想排序的上界为：若 `a≥c` 则为 1；否则为 `a(1−c)/(c(1−a))`。`R(1)=0` 时相对量不定义。故归一化仍保留 accuracy/coverage 依赖；RER 增益的百分点也不等于实际错误率下降的百分点。

### 6.5 ECE 的统计解释需要保守而准确

分箱、样本量和绝对值会影响经验 ECE 的偏差；equal-mass 在 Roelofs 等的分析中改善偏差，但不自动消除偏差。[R13][r13] Fang 与 Santos 研究方向可微函数的 bootstrap 条件。[R23][r23]

这两篇文献不能拼成“当前自适应 image bootstrap 的覆盖率已证明正确”，也不能拼成“当前所有 ECE 区间必然错误”。它们支持的是：覆盖率需要匹配具体估计器、极限情形和抽样假设。当前文件未推导这个特定自适应分箱估计器的完整理论。

因此不能把 ECE 区间下界为正直接当作总体失校准已被正式检验确认。经验校准曲线与误差可以如实报告；区间说明为既定程序下的条件重抽样不确定性，并明确没有额外覆盖率证明。

## 7. 特征增益与分数模型：可以支持什么？

### 7.1 Full−(S+Q) 回答有限学习流程中的增量效用

Williamson 等将内在变量重要性定义为总体上最优全特征与最优受限特征预测函数的表现差。[R16][r16] 本项目比较的则是固定标准化、Logistic、有限 C 网格与训练数据上的两个已选模型。

所以 Full−(S+Q) 可称为：“在指定学习流程和冻结 grounding 条件下，candidate–candidate 衍生特征提供额外的可靠性预测效用。”它不能直接证明 Q 分数在信息论上不充分、V 是必要信息、或语义交互是退化的因果原因。

新增特征时会同时改变系数、正则化与选中的 C；这不是只把现实中的某个原因置零的因果干预。若增益不稳定，也不能证明总体上不存在该信息。

已有证据摘要（`eval_split=__pooled_test__`，三个固定 seed 的平均效应）由 `information/information_bootstrap.csv` 的 `Full_minus_S_plus_Q` 行生成：

<!-- GENERATED_FEATURE_TABLE_START -->

| 条件 | Full−(S+Q) AUROC | 95% CI |
|---|---:|---:|
| random K5 | +0.000679 | [-0.001656, +0.002922] |
| random K50 | +0.013638 | [+0.011202, +0.016070] |
| matched hard K5 | +0.027089 | [+0.024338, +0.029744] |
| K10 dose m8 | +0.025464 | [+0.022670, +0.028348] |

<!-- GENERATED_FEATURE_TABLE_END -->

random K5 区间跨零，表中的另外三个条件区间为正。这支持“在若干已测条件下有增量”，尚不单独证明“hard 相对 random 增益显著扩大”：后者要直接计算两种增量的配对差值。[R08][r08]

这里还要区分两个不同的“交互”：四角 I 是标签块与置信度块对 AUROC 的非加性；Full−(S+Q) 是特征组的增量预测效用。不要混为同一个机制证据。

### 7.2 ScoreDeepSets 的失败不能证明分数没有足够信息

Deep Sets 的置换不变表达形式有理论条件；Wagstaff 等进一步讨论连续表示与集合大小/潜在维度限制。[R17][r17]、[R18][r18] 这些定理不保证有限训练下跨 K 泛化，也没有证明本项目 ScoreDeepSets 的误差由潜在维度造成。

现有 K50 证据如下，来自同一汇总文件的 pooled test、mean3 结果：

<!-- GENERATED_CROSS_K_TABLE_START -->

| K50 端点/对比 | AUROC 或配对差值 | 95% CI |
|---|---:|---:|
| SDS，K5/10 训练 | +0.678773 | [+0.668254, +0.689198] |
| SDS，K20/50 训练 | +0.760032 | [+0.749032, +0.770566] |
| Stats，K5/10 训练 | +0.791475 | [+0.781300, +0.801126] |
| Stats，K20/50 训练 | +0.795627 | [+0.785659, +0.805010] |
| 确定性 MSP | +0.789022 | [+0.778694, +0.798714] |
| SDS 大 K−小 K 训练配对差值 | +0.081259 | [+0.073471, +0.088991] |

<!-- GENERATED_CROSS_K_TABLE_END -->

大 K 训练显著改善 ScoreDeepSets 的 K50 AUROC，但点估计仍低于现有 Stats Logistic 与确定性 MSP。合理结论是“小 K 训练支持不足构成失败的一部分，现有流程尚未获得稳定竞争优势”。不能写成“分数表示必然不足”，也不能写成“只要大 K 训练即可解决”。

128 行子集拟合仅检验极小数据上的拟合表现，不能证明优化普遍充分；训练曲线和选型记录能约束判断，但不足以唯一分离所有表示、优化和分布因素。Ovadia 等提供 shift 下不确定性评估的背景，不直接解释本项目的 K 外推原因。[R19][r19]

论文保留失败结果即可。当前下一阶段不再增加架构搜索或训练实验；重点展示支持范围、剩余性能差距与无法作出的推论。

## 8. 统计、稳健性与事后分析的解释边界

### 8.1 当前区间的估计对象

同图像下多条 referring expression 不独立，image-cluster 抽样有明确的依赖结构依据；cluster bootstrap 的有效性仍依赖相应采样模型。[R20][r20] 本项目每次共享图像抽样，按 seed 计算完整效应，再对三个固定模型取均值。

这回答固定模型在相应图像抽样分布下的平均效应不确定性，不包含重新划分数据、重新训练、超参搜索或所有随机初始化的分布。seed SD 另行报告；不能把三个 seed 和 5,000 次测试 bootstrap 当作 5,000 次模型重复训练。[R21][r21]

DeLong 的相关 ROC 方法说明相关比较需要协方差，但其原始设置不是本项目的跨正确性标签、image-cluster、三 seed 均值；不能直接贴一个“DeLong”引用就宣称现有程序被其定理验证。[R10][r10]

### 8.2 当前补充分析的证据等级

训练/标准化/epoch 选择按 train/tune 隔离，是必要的实施约束；但后来根据已观察测试现象决定新增消融与分析，会影响证据的独立性。Dwork 等的工作解释了自适应复用数据的问题。[R22][r22]

因此继续明确标注“结果驱动的补充分析”。不能把输入哈希冻结改写为完整假设与分析预注册，也不能把本轮修补包装成未观察测试集上的独立确认。文献无法补发这类证据身份。

本轮众多端点的区间主要为逐项区间，不能直接宣称形成整体 95% 同时覆盖的确认结论。论文需确定主命题，次要条件用于描述边界；不要将每个正区间当作一项独立贡献。

### 8.3 候选敏感性不是一句“不显著”就完成

现有类别错配审计与冻结模型敏感性实验能量化错误规模，以及旧/新构造在共同可用 cohort 中的影响。须同时给出配对交集及排除规模，因为这两个量决定结果的适用对象。

差值区间跨零只说明当前分析未稳定区分其符号；等效结论需要有科学意义的等效界值和相应推断。[R24][r24] 对不同 family 或端点也不能统一说“完全没有影响”：有的区间略为正，有的跨零，应如实报告效应尺度。

当前审计足以否定“所有 proposal 与 target 类别匹配都天然正确”之类主张，并为已测构造提供敏感性记录；它没有证明所有未标注对象、所有自然候选或所有 detector 类别都被覆盖。

## 9. 文献支持下的论文主线与贡献定位

建议总论点：

> 在 GT 辅助、目标在场的受控候选构造下，候选扩张会改变预测正确性与置信度排序之间的关系；这种变化需同时审计训练支持、信息来源和指标分解。冻结嵌入衍生特征在部分条件下提供额外可靠性信号，但四角分摊及模型成败不足以识别唯一因果机制。

这里的第一分句也要限定到证据支持的 scorer/条件，不能由 B3 的 AUROC 降低推出每个 scorer 的 discrimination 都退化。

相关研究已覆盖 CLIP REC、selective prediction、temperature scaling、set representation 与 grounding 不确定性；这些不能作为本项目的新方法贡献。[R11][r11]、[R17][r17]、[R25][r25]、[R26][r26]、[R27][r27]、[R28][r28]

本项目更适合声称的贡献为：**问题明确的受控评估、带边界的增量信息证据、以及展示路径依赖并纠正过强解释的分析框架与可复核产物**。这是一条实证与分析贡献路线；本轮没有证明它是文献中首次。

| 正文章节中的问题 | 主证据 | 分析承担的任务 | 不能越过的边界 |
|---|---|---|---|
| 候选扩张下什么发生变化？ | accuracy、AUROC、risk–coverage、calibration | 区分结构性准确率限制与实测可靠性变化 | 不推广到任意自然候选和所有 scorer |
| 分数学习与训练支持能解释多少？ | Stats/SDS ID–OOD、MSP、训练曲线 | 解释先前模型比较的外推混淆 | 不把一个模型失败当作信息不可能性 |
| 新特征提供什么额外信号？ | S/S+Q/S+V/Full | 分离 alternate scoring 与候选间特征的增量 | 不称为语义机制的必要性证明 |
| 标签与置信度如何共同进入变化？ | 四角、两路径、I、S/F/E | 解释指标变化及对称平均为何掩盖抵消 | 不把混合四角称为已识别中介效应 |
| 哪些结论可靠、哪些仍有限？ | 类别敏感性、跨 family/表达条件、统计审计 | 将稳健性嵌入对应命题 | 不用一串附加实验替代核心论证 |

论文图表应服务于这条递进关系。完整实验历史、gate 修改前后、失败恢复及日志放附录；正文不按 Phase 编号写研究流水账。章节中的每个强结论都应能对应“数据产物＋所用定义＋适用范围”，而不仅是文献数量。

## 10. 下一阶段只用已有证据强化分析的任务

以下记录调研阶段提出的后续任务；本轮已落实第 2 项的路径与配对分析，具体状态见[理论完善记录](theory_analysis_completion_2026-10-04.md)。其余任务不因新增理论文档而自动完成，也不作为新实验完成记录：

1. 完成一张可直接进入论文的四角/双路径主图。保持有符号方向一致，展示 I 与对称分量的共享抽样区间；优先解释路径反号，弱化贡献百分比。
2. 基于现有逐样本预测落实 §5 的 S/F/E 配对恒等式，展示组规模、配对权重和排序胜率，核验其重构四角。需要统计时复用正式 cluster bootstrap，不重新训练模型。
3. 把“hard/dose 中增益更强”的比较改为对齐 cohort 后的增量差值；若不能对齐或结果不支持，收缩为条件内增益描述。这是直接检验已有比较，不扩展实验条件。
4. 把效用表补成“absolute AUROC＋配对增量＋实际 risk@coverage”，将 RER 与 E-AURC 定位为有准确率依赖的辅助摘要。
5. 建立正文主张表：已支持、部分支持、未识别；将 H1c、四角 I、特征增益的因变量分开，逐处消除“机制被否证/95%因果贡献/信息必然不足”等措辞。
6. 更新当前论文的 calibration 术语，保留历史函数名称及冻结指标身份；处理旧引用笔记中的明确书目信息错误。暂不修改历史协议或冻结计算。

验收目标是论证链条更清楚、指标解释更准确、边界更可检验，不是多造几条正结果。

## 11. 文献卡片与阅读留档

### R01 — Shapley，基础来源（A）

Lloyd S. Shapley. *A Value for N-Person Games*. RAND P-295，1952 报告版本。[官方记录][r01]；[扫描 PDF](https://www.rand.org/content/dam/rand/pubs/papers/2021/P295.pdf)。读到元数据和报告摘要；扫描正文未获得可可靠读取的文本，不声称精读证明。常见 1953 年引用对应后续书章，不能把 PDF 路径里的 2021 当作发表年。用途：来源追溯；本文件的现代价值函数与两玩家推导主要依赖 R02，而非未读的扫描定理。

### R02 — 价值函数决定“哪一种 Shapley”（F）

Mukund Sundararajan, Amir Najmi. *The Many Shapley Values for Model Explanation*. ICML 2020，PMLR 119:9269–9278。[会议记录][r02]；[全文](https://proceedings.mlr.press/v119/sundararajan20b/sundararajan20b.pdf)。读取 §2.1、§2.2，特别是排列平均定义与 Baseline Shapley 的输入替换价值函数。支持分摊依赖所定义游戏，不是仅凭算法名称就有唯一解释。项目适配见 §3；没有声称原文研究了 grounding 的数据集级 AUROC。

### R03 — 观测与干预解释不可混同（F）

Dominik Janzing, Lenon Minorics, Patrick Blöbaum. *Feature relevance quantification in explainable AI: A causal problem*. AISTATS 2020，PMLR 108:2907–2916。[记录][r03]；[全文](https://proceedings.mlr.press/v108/janzing20a/janzing20a.pdf)。读取引言及条件/干预分布讨论。用途：解释“缺失特征”操作的分布语义为何重要。不能把作者针对特征归因的立场概括成任何边际分摊都已经识别真实因果原因。

### R04 — Causal Shapley 的额外定义（F）

Tom Heskes, Evi Sijben, Ioan Gabriel Bucur, Tom Claassen. *Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models*. NeurIPS 2020。[记录][r04]；[全文](https://proceedings.neurips.cc/paper/2020/file/32e54441e6382a7fbacbbbaf3c450059-Paper.pdf)。读取 §2、公式 (1)–(4)，包含 `E[f(X)|do(XS=xS)]` 的价值函数。支持需要干预语义与因果知识。当前项目数组替换不满足该定义，不能以 Shapley 公理代替它。

### R05 — 直接与间接效应的定义（F）

Judea Pearl. *Direct and Indirect Effects*. UAI 2001，411–420。[作者全文][r05]。读取 §2 的 controlled/natural effect 定义，尤其 §2.2、§2.4。用途：区分可计算对比与反事实中介对象。原文扫描文本有部分符号识别瑕疵，本文件不逐字引用；不把 Pearl 文中具有反事实含义的 descriptive interpretation 等同于我们的描述性指标分摊。

### R06 — 中介识别假设（F）

Kosuke Imai, Luke Keele, Dustin Tingley. *A General Approach to Causal Mediation Analysis*. Psychological Methods 15(4):309–334，2010，DOI 10.1037/a0020761。[作者页面][r06]；[全文](https://imai.fas.harvard.edu/research/files/BaronKenny.pdf)。读取潜在结果定义、公式 (1)–(4) 与 Assumption 1，涉及 sequential ignorability。用途：明确中介效应与识别条件。相似的四项代数结构不能让本项目自动继承该文的因果解释或估计保证。

### R07 — 统计交互与机制交互（A）

Tyler J. VanderWeele, Mirjam J. Knol. *A Tutorial on Interaction*. Epidemiologic Methods 3(1):33–72，2014，DOI 10.1515/em-2013-0005。[出版社记录][r07]；[作者机构 PDF](https://content.sph.harvard.edu/wwwhsph/sites/603/2018/04/InteractionTutorial_EM.pdf)。读取检索可见的原始摘要/引言片段；出版社/PDF 直接访问失败。仅用于尺度与交互解释的原则，不声称逐项核验其机制交互定理，也不把流行病学风险尺度公式直接套在 AUROC 上。

### R08 — 显著性差异不是差异显著（F）

Andrew Gelman, Hal Stern. *The Difference Between “Significant” and “Not Significant” is not Itself Statistically Significant*. The American Statistician 60(4):328–331，2006。[作者全文][r08]，DOI 10.1198/000313006X152649。读取引言与示例。用途：hard 与 random 的特征增益要直接比较；不同端点区间是否跨零不是其增量差值的检验。

### R09 — AUC 的配对概率含义（A）

James A. Hanley, Barbara J. McNeil. *The meaning and use of the area under a receiver operating characteristic (ROC) curve*. Radiology 143(1):29–36，1982，DOI 10.1148/radiology.143.1.7063747。[原始摘要/元数据][r09]。摘要明确给出正负对象排序的概率解释和 Wilcoxon 联系。用途：§5 的基础；S/F/E 恒等式是本文件展开指标定义的推导，不是该文已有的 grounding 结果。

### R10 — 相关 ROC 比较（A）

Elizabeth R. DeLong, David M. DeLong, Daniel L. Clarke-Pearson. *Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach*. Biometrics 44(3):837–845，1988。[原始摘要/元数据][r10]。读取摘要中关于相关比较及 U-statistic 协方差的方法说明。这里只支持“配对比较要保留相关性”的背景；不拿其方法替代不同标签、聚类样本的统计方案。

### R11 — 温度缩放与校准（F）

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. *On Calibration of Modern Neural Networks*. ICML 2017，PMLR 70:1321–1330。[记录][r11]；[全文](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)。读取校准定义及 §4 temperature scaling，特别是 argmax/accuracy 不变的实际陈述。用途：纠正温度排序推断。跨样本 MSP 反例来自项目，不能标为原文结论。

### R12 — Confidence 与严格 top-label 的区别（F）

Chirag Gupta, Aaditya Ramdas. *Top-label calibration and multiclass-to-binary reductions*. ICLR 2022；arXiv 初稿 2021。[作者版本及版本记录][r12]；[可读 HTML](https://arxiv.org/html/2107.08353v3)。读取 §2 公式 (1)–(4) 及术语澄清。ICLR 接收版本链接被 OpenReview 验证码阻挡；分析基于可读作者版本，不能说已逐页读取接收 PDF。用途：当前 pooled ECE 的准确命名；不据其批评否定本项目 binary correctness 校准的实际用途。

### R13 — ECE 估计偏差（F）

Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, Michael C. Mozer. *Mitigating Bias in Calibration Error Estimation*. AISTATS 2022，PMLR 151:4036–4054。[记录][r13]；[全文](https://proceedings.mlr.press/v151/roelofs22a/roelofs22a.pdf)。读取引言、估计偏差及分箱相关分析。用途：equal-mass、经验 ECE 的局限。该文不提供本项目自适应 image-cluster percentile 区间的专用覆盖率定理。

### R14 — 选择性评价指标的缺陷（F）

Jeremias Traub et al. *Overcoming Common Flaws in the Evaluation of Selective Classification Systems*. NeurIPS 2024。[记录][r14]；[全文](https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf)。读取指标比较段落与 Fig. 2，涉及 AURC、E-AURC 和 AUGRC。支持不能单凭基准扣除宣称排除了基础性能影响。没有把该文提出的 AUGRC 偷换成现有结果，也没有将项目有限样本反例归为该文实验。

### R15 — Population AURC 与有限估计（A）

Han Zhou, Jordy Van Landeghem, Teodora Popordanoska, Matthew B. Blaschko. *A Novel Characterization of the Population Area Under the Risk Coverage Curve (AURC) and Rates of Finite Sample Estimators*. ICML 2025，PMLR 267:79226–79253。[原始摘要/记录][r15]。官方链接的全文访问失败。仅用于区分总体函数与有限样本估计的背景；不引用未读证明为当前积分规则或区间背书。

### R16 — 内在变量重要性与具体算法消融（F）

Brian D. Williamson, Peter B. Gilbert, Noah R. Simon, Marco Carone. *A general framework for inference on algorithm-agnostic variable importance*. JASA 118(543):1645–1658，2023 卷期；2022 年在线发表，DOI 10.1080/01621459.2021.2003200。[作者手稿][r16]。读取 §2.2 oracle predictiveness 和变量重要性定义。用途：区分最优函数差与两个有限 Logistic 模型的差。不声称我们采用了该文的高效估计或零重要性检验。

### R17 — Deep Sets（A）

Manzil Zaheer et al. *Deep Sets*. NeurIPS 2017。[作者摘要/版本][r17]。读取摘要及记录；本轮不完整审核所有表示定理。用途：架构来源和置换不变性动机。表达能力不能直接变成有限数据、有限优化或跨集合大小泛化保证。

### R18 — 连续集合表示的限制（F）

Edward Wagstaff, Fabian B. Fuchs, Martin Engelcke, Ingmar Posner, Michael A. Osborne. *On the Limitations of Representing Functions on Sets*. ICML 2019，PMLR 97:6487–6494。[记录][r18]；[全文](https://proceedings.mlr.press/v97/wagstaff19a/wagstaff19a.pdf)。读取 §4、Theorem 4.1 的连续 sum-decomposition 维度条件。该定理针对一般函数表示；不能据它断言当前含 mean/max 聚合的 ScoreDeepSets 失败由 latent 维度造成。

### R19 — Shift 下不确定性评价（A）

Yaniv Ovadia et al. *Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift*. NeurIPS 2019。[原始会议摘要][r19]。读取摘要及检索可见引言。用途：shift 下可靠性需要重新评估的动机。该文研究的 shift、模型与我们的候选操作不同，不提供候选扩张的普遍机制结论。

### R20 — Cluster bootstrap（A）

C. A. Field, A. H. Welsh. *Bootstrapping Clustered Data*. JRSS B 69(3):369–390，2007，DOI 10.1111/j.1467-9868.2007.00593.x。[原始摘要/期刊记录][r20]。全文受限，未读证明。用途：以 cluster 为抽样单元的依赖结构原则；其一致性结论有模型前提，不是任意复杂指标的通用担保。

### R21 — Benchmark 的多重方差来源（F）

Xavier Bouthillier et al. *Accounting for Variance in Machine Learning Benchmarks*. MLSys 2021。[会议记录][r21]；[全文](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf)。读取 §2 的变化来源、§3.2 的完整流程与固定选型估计之别。用途：明确当前三个固定模型的区间没有覆盖完整学习流程。会议页与 PDF 作者名单呈现有差异，本文件用 et al.，未擅自合并生成完整作者列表。

### R22 — 自适应复用数据（F）

Cynthia Dwork, Vitaly Feldman, Moritz Hardt, Toniann Pitassi, Omer Reingold, Aaron Roth. *Generalization in Adaptive Data Analysis and Holdout Reuse*. arXiv:1506.02629，2015。[作者版本][r22]；[全文](https://arxiv.org/pdf/1506.02629)。读取摘要与引言的 adaptive reuse 问题。用途：补充实验的证据身份。不要把这篇论文的标题写成同期另一篇 *The reusable holdout*，也不声称本项目实施了文中的复用保障算法。

### R23 — 非光滑函数与 bootstrap 条件（F）

Zheng Fang, Andres Santos. *Inference on Directionally Differentiable Functions*. Review of Economic Studies 86(1):377–412，2019；2018 在线发表。[期刊记录][r23]；[作者版本](https://arxiv.org/pdf/1404.3763)。读取引言及 Theorem 3.1 的条件与结论。用途：提醒非光滑统计量不能自动获得通常的 bootstrap 保证。未验证自适应 ECE 的所有条件，故不据该定理宣判当前程序必然有效或必然无效。

### R24 — 等效与未拒绝的区别（A）

Daniël Lakens. *Equivalence Tests: A Practical Primer for t Tests, Correlations, and Meta-Analyses*. Social Psychological and Personality Science 8(4):355–362，2017，DOI 10.1177/1948550617697177。[原始摘要/元数据][r24]。出版社直接全文访问失败。用途：需要等效边界才可作等效主张；不把其 t-test 操作直接照搬为本项目 AUROC 的具体检验。

### R25 — CLIP REC 的先行工作（A）

Sanjay Subramanian et al. *ReCLIP: A Strong Zero-Shot Baseline for Referring Expression Comprehension*. ACL 2022，5198–5215。[会议原始记录][r25]。读取摘要及引用元数据。用途：frozen CLIP grounding 的背景和创新边界。正确页码为 5198 起，不能写成 5199；本轮没有全面比较其任务/关系模块与本项目方法。

### R26 — 近期 grounding 不确定性工作（A）

Qingni Wang, Yue Fan, Xin Eric Wang. *SafeGround: Know When to Trust GUI Grounding Models via Uncertainty Calibration*. arXiv:2602.02419，2026。[作者摘要/记录][r26]。读取摘要及元数据，未给预印本捏造正式 venue。用途：说明 grounding uncertainty 已有直接研究；GUI 与本项目的 REC、候选协议不同，不能继承其风险保证或声称同设置全面超越。

### R27 — 预训练分类器上的选择性分类（A）

Yonatan Geifman, Ran El-Yaniv. *Selective Classification for Deep Neural Networks*. arXiv:1705.08500，2017。[原始摘要/版本][r27]。读取摘要：给定训练好的网络构造 selective classifier，并控制风险。用途：拒绝/coverage 的已有框架。纠正历史笔记将这项工作泛写成联合优化分类和拒绝；该训练思想应与 R28 区分。

### R28 — 联合训练与拒绝（A）

Yonatan Geifman, Ran El-Yaniv. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option*. ICML 2019，PMLR 97:2151–2159。[原始摘要/记录][r28]。摘要明确联合优化分类/回归与拒绝。用途：与 R27 区分，本项目未提出这类新训练算法。

### R29 — 不确定性评价的多维对象（A）

Yukun Ding, Jinglan Liu, Jinjun Xiong, Yiyu Shi. *Revisiting the Evaluation of Uncertainty Estimation and Its Application to Explore Model Complexity-Uncertainty Trade-Off*. CVPR Workshops 2020。[CVF 原始记录][r29]。读取原始摘要/检索引言；PDF 直接访问返回 403。本文件仅借其评价问题背景，不声称完整核验公式；不能把 workshops 写成 CVPR 主会。

### R30 — 进一步分解文献，暂不承担核心论证（M）

Anthony F. Shorrocks. *Decomposition procedures for distributional analysis: a unified framework based on the Shapley value*. Journal of Economic Inequality 11:99–126，2013 卷期，2012-01-07 在线发表。[出版社记录][r30]。全文受限，本轮仅核对书目信息。可作为后续补读入口，但不以其未读正文支持本项目的具体分解定理。

## 12. 历史引用与措辞纠错清单

| 当前/历史问题 | 本轮核实 | 后续正文处理 |
|---|---|---|
| Guo 2017 被记成另一段描述性标题 | 正式标题 *On Calibration of Modern Neural Networks*，R11 | 用正式题名与会议引用 |
| “top-label ECE” 被等同于严格标签条件校准 | 接口仅 confidence/correctness；R12 区分定义 | 正文用 confidence ECE，解释旧命名 |
| Geifman 2017 被写成联合训练分类与拒绝 | R27 给定训练好的网络；R28 是 SelectiveNet | 两项工作分开引用 |
| Zhou 2025 的题名/卷页未完整核对 | R15 提供官方题名、PMLR 267:79226–79253 | 可引用摘要支持的背景，不引用未读证明 |
| ReCLIP 页码可能误写为 5199 起 | 官方记录为 5198–5215，R25 | 采用官方记录 |
| Shapley 年份容易与 PDF 存放时间混淆 | RAND 1952 报告与 1953 书章不同 | 标明引用版本 |
| Dwork 2015 两篇相关工作容易混名 | R22 是 *Generalization in Adaptive Data Analysis and Holdout Reuse* | 不错配 Science 文章标题与 arXiv 链接 |
| “机制被否证”“框中无物体”“信息必然不足” | 当前证据与因变量不支持这些强表述 | 分别写“所测解释未获支持”“未匹配标注对象”“指定学习流程下表现不足” |

历史 `docs/literature_notes.md` 其余“需查证”条目仍不能直接进入论文引用。**本轮没有逐条验证整个历史列表，也没有确认所有看似正式的题名都真实存在。** 无法定位的条目应删除或保持待核实，不能用本文件的存在为它们整体背书。

## 13. 项目证据快照与复核记录

本文件的公式不改变冻结输入。表格从以下已有文件读取；SHA256 对应调研时的原始字节：

| 文件（相对仓库根） | SHA256 |
|---|---|
| `results/research_repair_v1/mechanism/four_corners_paths_interaction.csv` | `3f989d73e5d451081fe753c0cec6b020a1460e5d6195af0419f1824c3b612a72` |
| `results/research_repair_v1/mechanism/bootstrap_ci.csv` | `7d1b54e6d3dca246fc2fb8435189ec09cdb98c51bfd452486e6b4522b448018a` |
| `results/research_repair_v1/information/information_bootstrap.csv` | `0755e9427fde43b7b7527ac7c62ef49bdb12cde8127a05ae64517bd86e140760` |
| `src/ccg/metrics/calibration.py` | `1c935256562e22d870469c05316e61fdf44704e5874d8240a75dbc72f0020082` |

<!-- GENERATED_QA_START -->

文档复核：30 个文献 ID 与链接定义对应；阅读深度计数 F=15、A=14、M=1。12 行既有四角结果（9 个 seed 行、3 个 mean3 行）的两路径加和、交互与对称恒等式通过核对，最大残差 0.00e+00；四角、信息消融及 K50 模型表直接取自既有 CSV，保留正式配对 CI；四角表与既有 CI 点估计一致。上述 S/F/E 配对公式在 540 个含三组且含并列分数的有限示例上核对，最大残差 0.00e+00。这只是公式复核，不能替代真实预测的后续组间分析。四个来源文件 SHA256 保持一致。

<!-- GENERATED_QA_END -->

文献复核不是重新验收整个实验库。本轮不重新训练、不新增实测端点，也不把访问受限的全文标为已读。当前最有价值的改进是把四角变成对指标变化的可解释审计，并在正文清楚承认内部因果机制仍未识别。

[r01]: https://www.rand.org/pubs/papers/P295.html
[r02]: https://proceedings.mlr.press/v119/sundararajan20b.html
[r03]: https://proceedings.mlr.press/v108/janzing20a.html
[r04]: https://proceedings.neurips.cc/paper_files/paper/2020/hash/32e54441e6382a7fbacbbbaf3c450059-Abstract.html
[r05]: https://ftp.cs.ucla.edu/pub/stat_ser/R273-U.pdf
[r06]: https://imai.fas.harvard.edu/research/baronkenny/
[r07]: https://www.degruyterbrill.com/document/doi/10.1515/em-2013-0005/html
[r08]: https://sites.stat.columbia.edu/gelman/research/published/signif4.pdf
[r09]: https://pubmed.ncbi.nlm.nih.gov/7063747/
[r10]: https://pubmed.ncbi.nlm.nih.gov/3203132/
[r11]: https://proceedings.mlr.press/v70/guo17a.html
[r12]: https://arxiv.org/abs/2107.08353
[r13]: https://proceedings.mlr.press/v151/roelofs22a.html
[r14]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/047c84ec50bd8ea29349b996fc64af4b-Abstract-Conference.html
[r15]: https://proceedings.mlr.press/v267/zhou25y.html
[r16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10652709/
[r17]: https://arxiv.org/abs/1703.06114
[r18]: https://proceedings.mlr.press/v97/wagstaff19a.html
[r19]: https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html
[r20]: https://academic.oup.com/jrsssb/article-abstract/69/3/369/7109361
[r21]: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
[r22]: https://arxiv.org/abs/1506.02629
[r23]: https://academic.oup.com/restud/article/86/1/377/5094886
[r24]: https://pubmed.ncbi.nlm.nih.gov/28736600/
[r25]: https://aclanthology.org/2022.acl-long.357/
[r26]: https://arxiv.org/abs/2602.02419
[r27]: https://arxiv.org/abs/1705.08500
[r28]: https://proceedings.mlr.press/v97/geifman19a.html
[r29]: https://openaccess.thecvf.com/content_CVPRW_2020/html/w1/Ding_Revisiting_the_Evaluation_of_Uncertainty_Estimation_and_Its_Application_to_CVPRW_2020_paper.html
[r30]: https://link.springer.com/article/10.1007/s10888-011-9214-z
