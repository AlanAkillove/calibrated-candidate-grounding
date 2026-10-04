# V3 完成验收

状态：**COMPLETE — 研究、正式测量与科学验收完成。** 最终验收见 [final_acceptance.json](../results/v3_final_validation/final_acceptance.json)。正文结果为正、反向或证据不足均如实保留，不以全部假设成功作为完成条件。

## 已完成与待验收

| 缺口 | 修改及实测证据 | 当前状态 | 结论边界 |
|---|---|---|---|
| 文献定位宽泛 | [独立检索记录](v3_novelty_review.md)：日期、查询、正文位置、比较矩阵与阅读深度 | 完成 | 不声称首次研究 grounding 可靠性或首次使用候选上下文 |
| 确认数据身份不可信 | [数据审计](v3_data_exposure_audit.md)及 final_acceptance：3107 张新确认图像/7993 表达；200/522 开发；保留筛查和排除记录 | PASS | 是本项目流程隔离；基础模型预训练暴露未知，像素筛查覆盖和来源映射有限 |
| B/16 原语义表定义不同 | [复制报告](v3_replication_report.md)：重新生成 S17/Q8/V8/Full33、12 个 Logistic、6 项原容差锚点、5000 次正式抽样 | PASS | 复用已观察 COCO 测试集的跨配置复制，不是独立确认或 backbone 单因素 |
| 自然接口未实测 | [接口报告](v3_pipeline_report.md)、首次冻结确认预测及自然 bootstrap：所有输入保留、整体/覆盖条件分母、漏检/空候选/低 K 追踪 | PASS，实测完成 | 自然正例漏检不同于图像无目标；GT 不进入自然排序或可靠性特征；本轮自然无空/低 K 样本不能证明其他数据没有 |
| 最强结论缺独立确认 | 固定 C1–C4、各配置自己的 correctness、一次正式前向、四比较调整区间、写入前运行屏障 | PASS；C1 未确认，C2/C3/C4 支持指定正向效应 | 不通过平均配置、调参或扩样本挽救结果；不把 C1 区间跨零称作等效 |
| 统计与流程验收 | 共享图像抽样、固定 seed 内效应均值、边界同分风险、单类未定义；内容绑定完整测试、保存预测/原始分布实际验收 | PASS | 不把预测 ensemble 或平均端点区间当作效应推断；区间条件于现有模型 |
| 论文组织与证据一致 | [V3 蓝图](../docs/paper_blueprint_v3.md)、[论证草稿](../docs/paper_draft_v3.md)、当前摘要按问题推进；[产物生成索引](../results/v3_final_validation/evidence_index.md)及[正文表](../results/v3_final_validation/publication/key_findings.md)只读取正式产物 | 完成研究收束；学位排版另行完成 | 解释审计充分不等于因果机制识别；正文包含反向和未确认结果 |

## 当前成立的分析

B/16 大 K、matched hard 和 dose 复制支持在对应条件下加入候选间特征的增量。小 K 的增量不稳定；S+V 的绝对表现可能高于 Full，S+Q 可能低于 S。因此贡献是固定学习流程与数据条件下的信息来源比较，不是 Full 普遍最优、Q 普遍有益或真实语义交互的因果识别。

正式数字与区间读取 [证据索引](../results/v3_final_validation/evidence_index.md)，不在验收文字中手工复制一套独立数字。

## 确认与自然结果改变了什么

1. **规模退化收缩为条件现象。** FineCops 受控 C1 未确认 MSP 的 AUROC 退化，虽然准确率下降、固定覆盖风险上升。不能把源 COCO 的判别退化当作候选扩张的一般定律，也不能把本次跨零区间解释成无效或等效。
2. **来源增量获得受控新确认。** C2/C3 分别支持 K50 的 Full−S+Q；不是平均两个配置掩盖失败。B/16 的 S+V 点表现仍高于 Full，故结论是固定流程中的条件性增量，而非 Full 最佳或 Q 普遍有益。
3. **训练支持效应迁移，但模型优势有限。** C4 支持大 K 训练的既有 ScoreDeepSets 优于小 K 版本；大 K 版本的绝对表现仍低于 MSP/Stats。三个固定 seed 的效应大小不同；图像 bootstrap 不覆盖重新训练/搜索的随机性。
4. **自然候选是实测边界。** K5 下两配置的 Full−S+Q 为反向；K20 的整体 AUROC 增量主要体现在未覆盖错误排序，覆盖条件区间跨零。K50 的 B0 整体及覆盖条件增量获边际支持，B/16 证据不足。不能据“一个显著、另一个不显著”宣布配置交互。
5. **指标仍需分别回答。** B0 自然 K50 的 Risk@50 差分未获明确支持，Risk@80 改善获边际支持。自然覆盖增加不伴随全部准确率持续增加；MSP 整体 AUROC 的点趋势也不同于原受控退化。没有追加结果驱动的新比较来挽救结论。

四项主要 AUROC 的两个区间等级、全部绝对端点、选择性风险、无效抽样、排名贡献和实际分母均来自 [生成式证据索引](../results/v3_final_validation/evidence_index.md)。本文件只解释已存结果，不另手工维护数字表。

## 实际验收与落盘产物

- 确认身份：官方 positive val 经暴露及结构审计；全部表达身份保留；[暴露审计](v3_data_exposure_audit.md)与[数据验收](../results/v3_final_validation/exposure/final_acceptance.json) PASS。完全离开图像的官方框按确认前规则排除，原记录和仍相交的越界 GT 保留。
- 开发与准备：[原开发验收](../results/v3_final_validation/pipeline/dev_acceptance.json)、[元数据复核](../results/v3_final_validation/pipeline/dev_revalidation.json)、[关闭产物准备验收](../results/v3_final_validation/pipeline/confirmation/preparation_acceptance.json) PASS。准备只提取 proposal/嵌入和身份，不观察确认性能。
- 封存与首次前向：[protocol_freeze.json](../results/v3_final_validation/protocol_freeze.json) 在首次前向前封存；[唯一运行台账](../results/v3_final_validation/confirmation_run.json)为 INFERENCE_COMPLETE，corrections 为空。源权重、特征、温度、标准化与规则未根据确认性能改动。
- 正式推断：[bootstrap/index.json](../results/v3_final_validation/pipeline/confirmation/bootstrap/index.json) COMPLETE。受控及自然均实际执行冻结的 5000 次共享图像抽样，原始均值/逐 seed 分布、区间与未定义计数落盘。
- 独立计算核对：[final_acceptance.json](../results/v3_final_validation/final_acceptance.json) PASS：完整表达/seed/配置网格、每个候选身份与预封存清单、raw logits 指针和宽度、稳定并列 winner、官方 GT IoU 正确性、受控嵌套不出现错误→正确、自然全部样本流、两个区间等级、逐 seed 均值及排名恒等式均通过。受控排除原因分别追踪，不只给总数。
- 现有与新增测试：[tests_before_confirmation.json](../results/v3_final_validation/tests_before_confirmation.json)及原始 JUnit：1298 passed、2 原有 opt-in skips、exit 0；4 个原有 marker 警告保留。实际验收再核对所有已测试源文件字节不变；后来新增的证据验证/打包和图表生成脚本已执行并校验，不冒充已纳入之前完整测试。
- 原输入保护：[legacy_inputs_final.json](../results/v3_final_validation/legacy_inputs_final.json) PASS，834 文件未改，历史文档原前缀保留。确认和统计没有重新训练旧模型。
- 公开证据：逐样本预测无损 gzip，原始 scorer logits、正式 bootstrap NPZ、12 个 B/16 拟合模型、小型源模型 ZIP 和与原缓存逐字节一致的 RPN 几何库；原图、特征缓存与公共大权重不进入 Git。[复核指南](../docs/v3_reproduction.md)区分保存证据复核和完整前向重建。公开几何库路径也实际执行最终验收 PASS，而非只提供未执行命令。
- 论文表图：产物生成主表和三组新图已渲染检查；自然图明确分母和效应方向。复现错误的 CLI 打包参数在解析阶段失败，旧日志保留；正确参数只压缩已有字节，不重跑确认。

## 剩余限制与停止条件

项目暴露审计不证明基础模型预训练未见，近重复和历史像素覆盖有限；GQA/VG 命名空间不证明非 COCO 视觉域。表达生成、图像来源和筛选同时变化，不能单独归因为视觉域。

可靠性增量较小且依赖 K、候选模式和配置；自然分析使用边际区间，不能包装成额外 family-wise 主确认。V 的部分锚点间接依赖 query-conditioned 排名，来源组不正交；AUROC 增量不估计条件互信息。四角与漏检分解是统计记账，不是内部因果机制。自然接口限于冻结 RPN/N64；目标缺席、多目标任务、新 NONE 分类器、架构矩阵和更多 seed 未开展。M3 仍 UNVERIFIABLE。

文献定位可核查、项目确认身份可信、特征复制及自然实测完成，论文材料已依据实际结果收束。本轮停止新增实验。后续工作是学位格式、引文编号、附录编排和展示，不把这些排版工作混称为尚未完成的实验。

提交与远程推送由本轮 Git 记录提供；本验收保留实际执行证据，不在文件中写入循环依赖的自身提交哈希。
