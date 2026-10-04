# 理论完善与已有预测分析：完成记录

日期：2026-10-04。本轮目标是完善论证与解释，不扩大模型、数据集或候选条件。状态：理论稿和正式配对分析已完成；因果识别与未实施的其他分析仍保留限制。

## 1. 交付与修正

| 问题 | 本轮处理 | 产物及边界 |
|---|---|---|
| 四角只有数值，缺少定义、条件和证明 | 整理结构单调性、配对重构、路径、交互、幅度界、温度及函数类界 | [理论稿](../docs/theory_analysis_v1.md)；不声称新的通用 Shapley 方法 |
| 小标签分量容易被误读为错误转移无关 | 给出路径抵消反例，展开 L(p)=T_N+T_R | 正文需要同时展示两条路径与有符号项 |
| 不清楚路径为什么反号 | 复用历史组间 AUROC 点估计，补齐配对变化及路径项的共享抽样 CI | [配对报告](../results/research_repair_v1/theory_analysis/v1/analysis_report.md)；没有重新训练 |
| 正交互容易被解释成语义协同 | I 展开为不同组间排名变化的加权差 | GDINO 三种排名都下降时 I 仍为正，不能据此宣称协同增强 |
| 正确性单调性被误推广到 AUROC | 给出 AUROC 升/降反例，证明标签路径绝对值不超过 max(alpha,beta) 并展示可达到条件 | 小准确率损失不保证小 AUROC 标签路径 |
| 文献稿 F/E 命名与仓库相反 | F 统一为正确转错，E 为两端错误；修改公式、解释并追加更正记录 | 保留历史结果，不改 CSV 栏名或模型 |
| 上轮遗漏已经存在的配对结果 | 明确历史 `pairwise_auc_components.csv` 已保存点估计 | 新增贡献是路径解释及区间，不将旧恒等式再记为新实验 |
| 理论容易超过证据 | 区分确定性结构性质、精确指标分摊、条件实测模式与未识别因果机制 | 保留结果驱动、固定 seed、family 内 cohort 和非同时区间限制 |

## 2. 主要结果如何进入论文

RPN/DETR 的 S–F 排名改善，而 S–E、F–E 排名变差。GDINO 三种排名都变差。由配对权重，可以精确重构 C0/C1 的差异与标签路径的反号。

这支持“不同正确性组之间的配对排序发生异质变化”，比“平均置信度下降”更接近 AUROC 的定义。但它没有识别这些排名变化的唯一内部原因。

GDINO 的 p50 新比较项点估计为正但区间跨零，不能把它写成稳定正项。RPN 的正 L1 来自正负项净抵消，DETR 的两项均稳定为正；必须按各 family 的证据分别表述。

建议正文呈现顺序：四角/两路径 → S/F/E 配对图 → 路径项与交互重构 → 解释边界。结构证明和额外反例放定义节或附录，避免独立理论章堆砌符号。

## 3. 复现与验证

正式分析程序：[analyze_repair_pairwise.py](../scripts/analyze_repair_pairwise.py)。来源与运行记录：[manifest.json](../results/research_repair_v1/theory_analysis/v1/manifest.json)。

固定 5,000 次、seed=0、95% percentile image-cluster bootstrap。每次抽样共享图像重数，各 seed 内先计算组数量、权重、胜率和完整效应，再取三 seed 均值。保留无效值与原始分布，不静默填零。

采用 tie-aware 加权求和加速，验证其与显式复制样本一致。每一 replicate 的四角与 I 都对照既有修补 raw bootstrap，不仅对照最终 CI。历史配对点估计也逐 seed 核对。

[export_repair_pairwise_theory.py](../scripts/export_repair_pairwise_theory.py)从已有分布导出逐 seed CI、生成理论正文数据表，并对保存的全部抽样检查路径/交互恒等式；不进行第二次抽样。

<!-- GENERATED_VERIFICATION_START -->

- 正式输出：99 个 family–quantity 均值区间，297 个逐 seed 区间；所有抽样均有效。
- 历史配对点估计最大差：0.00e+00；修补四角点估计最大差：2.22e-16。
- 与旧 raw bootstrap 逐次比较的最大差：3.33e-16；所有保存抽样的新增恒等式最大残差：3.75e-16。
- 原冻结与版本化 supplemental 输入：834 个唯一文件通过哈希核验，变更列表为空。
- 完整测试集：1246 passed、2 个原有 opt-in skips，退出码 0；collect-only 核对 1248 项。
- 专项测试覆盖并列、重复抽样、空 F、单类、G 非空、540 种有限分组、幅度界达到条件与小准确率损失的大 AUROC 变化反例。
- 详细记录：[验证 JSON](../results/research_repair_v1/theory_analysis/v1/verification.json)、[完整输入核验](../results/research_repair_v1/theory_analysis/v1/input_verification/final_input_verification.json)、[逐 seed 区间](../results/research_repair_v1/theory_analysis/v1/bootstrap_per_seed_ci.csv)、[导出与恒等式核验](../results/research_repair_v1/theory_analysis/v1/integration_manifest.json)。

<!-- GENERATED_VERIFICATION_END -->

运行时 manifest 的 `git_head` 指开始时的仓库提交；本轮新增代码同时以独立 SHA256 标识，发布提交包含对应源文件。冻结预测和旧机制证据没有改变。

短运行保存在本地 `theory_analysis/pilot_20`，明确标为 PILOT，未进入正式结论。正式证据目录为 `theory_analysis/v1`；原有 59-job publication export 的计数、gate 和状态未被追改。新结果的入口在当前 README、论文蓝图与摘要中。

## 4. 尚未完成或不能宣称的事项

- 内部因果机制仍未识别；S/F/E 是结果定义的组，不是随机分配的因果分层。
- 不能由本轮分析宣称 Full−(S+Q) 在 hard 相对 random 中被显著放大。相关增量的直接差值尚未在本轮补齐。
- 不能由本轮分析宣称自然候选评估、target-absence 或新视觉域迁移已完成。
- 不以经验 ECE 区间代替特定估计器的覆盖率理论，不以当前模型失败证明分数信息必然不足。
- 候选内容、尺度、冗余与表示等算法原因尚不能由指标恒等式唯一排除或识别。
- 新篇章是一份可写入论文的理论与分析稿，完整论文正文和最终排版尚未完成。

本轮完成标准是：定义明确、推导有证明/反例、经验解释可回到逐样本冻结产物、现有论文材料不超越这些证据。它不是所有研究疑问均已解决的声明。
