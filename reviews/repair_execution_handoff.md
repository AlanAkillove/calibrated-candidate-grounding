# Research Repair v1 — 执行交接

这是执行交接，不是完成报告。用户已授权完整针对性修补、跨夜运行与三个 `gpt-6-luna/max` 执行 subagents。旧审查聊天创建新 agent 被工具以 `agent thread limit reached` 拒绝，不能使用旧模型冒充指定模型。

## 入口与保护

先读 `reviews/repair_protocol.md`、`reviews/strict_research_review_2026-10-03.md` 和用户批准的计划。工作区为 `F:\DL Projects\calibrated-candidate-grounding`。Python 固定为 `E:\conda\envs\deepminer\python.exe`，不安装依赖、不重新提取特征。

准备命令：`& 'E:/conda/envs/deepminer/python.exe' -B scripts/prepare_research_repair.py`。如果 `results/research_repair_v1/input_manifest.json` 已存在，只运行同一命令加 `--verify`，禁止删除或覆盖 baseline。正式执行前确认旧聊天的准备进程已结束。基线 Git HEAD 为 `07e15a66056436fcecb880c5aa2d75354beefe26`，原有未跟踪的 `audit/`、`docs/paper_blueprint_v1.md` 必须保留。

原始 results、模型、预测、候选及缓存保持字节不变。历史纠正追加到 `docs/research_protocol.md`、`docs/experiment_log.md`，必要时在修补目录保存关联原协议路径的纠正记录；不能改掉原协议的哈希。当前文档可修正，最终数字必须由新产物生成。

## 派遣方式

在有空余额度的会话调用 `collaboration.spawn_agent`，每次都指定 `model="gpt-6-luna"`、`reasoning_effort="max"`、`fork_turns="none"`。启动说明须包含工作区、协议文件、责任范围与以下各自任务。若失败，如实记录，不替换模型，不将该任务写成已调度。

### Agent A：统计纠错

独占公共 bootstrap 模块、统计汇总入口的必要修正、统计 runner 与对应 tests，输出 statistics/。先发布共享 image-cluster bootstrap 接口并通知 B/C。每次同一图像抽样覆盖所有固定 seed，先每 seed 效应后均值；95% percentile CI、5000次、seed0。relative、DoD、severity macro 均在 replicate 内计算。报告无效抽样、逐 seed CI 与 seed 标准差。覆盖协议列出的所有主表/gate，已有正确共享抽样只验证。优先无损预测；缺失时冻结恢复，必要时原参数确定性恢复，锚点容差不得放宽。CSV 缺 sentence_id 按 canonical 顺序逐项核验。补 B0 native/global-T/validation per-K 诊断，纠正 E-AURC/RER 与温度叙事。只改必要主逻辑，不做泛化代码审计。

### Agent B：信息来源与跨 K

独占信息实验 runner、实验模块、对应 tests，输出 information/。原图像划分、grounding、B3三seed 固定。按协议固定 S/Q/V 列，训练四组 S、S+Q、S+V、Full，主比较 Full−(S+Q)，测试 random K、matched hard 与 dose。沿用原 Logistic C 网格和 tie-break，标准化只训练拟合。Stats Logistic 与原 ScoreDeepSets 做 small K5/10、large K20/50 两种训练/选型，同图像同训练行数，所有测试不参与选择。原架构/网格/300epoch 上限，不无限调参。保存 train/tune 曲线、epoch、配置、模型及逐样本预测，补128行拟合诊断与 score→MSP 基线。已复现实现错误先最小测试再修复；先协调共享文件。正式 CI 等待 A 工具，不平均 CI 端点。

### Agent C：候选审计与机制

独占新候选/机制 runner 及对应 tests，输出 candidates/、mechanism/。真实 referring ann 类别与 target proposal 最近匹配对象类别逐项审计；hard/dose 报 proposal 数与 distinct GT-object 数、objectness、面积、IoU、重复。mismatch=0 仅审计；大于零版本化构造真实类别候选，仅原/新共同可用交集，报告排除，冻结模型前向不训练。原始缓存不可覆盖，按 K 分块。机制保存四角、两条路径、有符号贡献与 interaction，共享 bootstrap 5000次。H1c accuracy harm 不解释 AUROC；未匹配标注不是“nothing”；净贡献小不等于变量没有作用。

## 资源与接手后状态

主代理协调一个训练进程加一个重型统计/审计进程，所有 BLAS 两线程、GPU 串行。B 可开发时 C 审计，正式重统计互斥。A 的公共接口就绪前 B/C 不产正式区间。每阶段完成即落盘，失败可续跑。

`STATUS.json` 初始为 `PREPARED_DISPATCH_BLOCKED`，六个实验/集成/验收阶段为 `NOT_STARTED`。接手后仅在真实派遣成功时更新到执行态，保存 agent id、模型与责任；不凭协议存在更改实验状态。

主代理负责当前 README、摘要、论文蓝图、引用及事实纠正、总索引、完整测试和最终哈希核验。按协议验收所有数学、统计、特征/划分、类别/交集、四角与恢复锚点。所有必需实验完成、证据与文档一致后，才将状态设 COMPLETE 并交付 `reviews/repair_completion.md`；不可恢复项须有真实失败证据，不能用近似结果填空。
