# 论文组织与展示审查

2026-10-03。论文按问题和证据依赖推进，阶段代号、完整门控与实验历史放入复现索引。

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
