# V4 分析方法的文献留档

检索日期 2026-10-05。本轮只补充 W1 的解释依据；完整 grounding 新颖性定位继续引用 `reviews/v3_novelty_review.md`，四角与路径解释依据继续引用 `reviews/analysis_literature_review_2026-10-04.md`。不借 V4 重启理论命题扩张。

|检索式|纳入来源|阅读深度/正文位置|用途与边界|
|---|---|---|---|
|Hanley McNeil 1982 meaning use area receiver operating characteristic curve probability correct ranking|[Radiology 原文 DOI](https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747)，已有 R09 原始摘要核验|本轮 DOI 网页返回访问错误；沿用此前已核验的摘要/元数据，不冒充此次全文精读|AUROC 的随机正误 pair 排序解释；本项目实际pair计数、seed平均和bootstrap需自己核验，不是该文提供的grounding结论|
|Geifman El-Yaniv selective classification risk coverage 2017 arxiv|[作者论文](https://arxiv.org/abs/1705.08500)|原始摘要与版本记录，2017-06-01 v2；摘要明确给定训练网络、拒绝以控制风险|支持选择性分类背景，不代表我们的固定coverage结果有它的风险保证，也不构成新拒绝训练方法|
|risk coverage AURC accuracy dependence generalized risk selective classification 2024 paper|[Traub等，NeurIPS 2024全文](https://proceedings.neurips.cc/paper_files/paper/2024/file/047c84ec50bd8ea29349b996fc64af4b-Paper-Conference.pdf)|阅读全文的引言 pp1–2，以及前次记录的指标比较和Fig2；本轮重点复核引言关于working-point和多阈值评价的区别|AURC汇总不能直接当作某部署工作点的效用；报告固定coverage绝对风险及每1000接受/输入样本错误变化。其AUGRC不是本项目V4 AURC，不偷换已有口径|

排除：二手百科、Scribd镜像、搜索摘要中没有实际核实的2026新应用论文，均不用于本轮方法判断。搜索引擎结果只是发现入口。

本项目推导：V4连续并列积分、oracle及相同label下ΔE-AURC=ΔAURC来自展开已定义的风险函数；不是声称上述论文证明了我们的实现。合成测试、加权显式重采样对照和真实prediction计数分别验证公式、bootstrap计算和实际效应。数值稳定区间不等于应用成本收益，且显著的小效应不自动成为新机制。

十二维 V_pure 隔离的是计算依赖，不是条件互信息、统计正交或因果机制。重新拟合 Logistic 得到的差分也可能体现有限模型对输入变换的利用；见原分析文献记录对该问题的解释边界。自然候选集由图像产生、受控候选集上游由GT/表达参与，必须区分。
