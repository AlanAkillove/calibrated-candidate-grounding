# Research Repair v1 — 执行协议

日期：2026-10-03。用户已明确授权完整针对性修补及跨夜运行。依据本目录综合审查与用户批准的执行计划。此协议是在原结果已可见后制定的补充分析协议，不是对历史实验的追溯预注册。

## 范围与身份保护

- 不增加 backbone、数据集或架构，不扩展 target-absence 或自然候选评估。
- 原始 results、预测、模型、候选及缓存只读。新增结果全部写入 `results/research_repair_v1/`。
- `input_manifest.json` 记录旧结果与使用的缓存、标注、候选、模型文件 SHA256。结束时使用准备脚本 `--verify` 核验。
- 历史协议/日志追加纠正；当前 README、结果摘要和论文蓝图可修改，但须与新产物一致。
- 本研究不以所有 gate 转为正结果为完成标准。更弱、负向或不可恢复的结果如实保存。

## 环境与调度

- 工作区：`F:\DL Projects\calibrated-candidate-grounding`。
- Python：`E:\conda\envs\deepminer\python.exe`；已有 NumPy 1.26.4、Torch 2.5.1+cu121、sklearn 1.7.2，CUDA 可用。
- 单卡 RTX4060 Laptop，显存约8GB，系统内存约16GB；每进程 BLAS 默认两线程。
- 所有执行 subagents 使用 `model=gpt-6-luna`、`reasoning_effort=max`、`fork_turns=none`。不能替用其他模型。
- 最多三 subagents；最多同时一个训练进程加一个重型统计/审计进程，GPU串行。分块读取K缓存。
- 本审查聊天的旧子线程已占满工具额度，创建执行agent返回 `agent thread limit reached`。不得称已经派遣成功；须在具备空余子线程配额的执行聊天启动。

## 所有权与公共接口

主代理负责本协议、初始manifest、根目录当前文档、历史日志追加、最终索引及完整测试。未跟踪的既有 `audit/`、`docs/paper_blueprint_v1.md` 和审查文档不得删除。

| Agent | 独占修改范围 | 输出目录 |
|---|---|---|
| A statistics | `src/ccg/repairs/bootstrap.py`、统计模块、`scripts/repair_statistics.py`、新统计tests、既有统计汇总入口的必要修正 | statistics/ |
| B information | 新实验模块与runner、新实验tests；已复现训练错误的最小修正（提前通知主代理） | information/ |
| C candidates/mechanism | 新审计/敏感性与机制runner、对应tests；候选构造的版本化修正 | candidates/、mechanism/ |

不得多人编辑同一文件。A先发布公共bootstrap接口；B/C开发可并行，但正式统计调用统一实现。

公共工具接收共同cohort的 image_id 和每seed统计回调；每次抽样索引共享给所有seed/比较单元。返回点估计、95%CI、逐seed效应、seed标准差、有效/无效replicate数及原始replicate序列。具体实现不以平均预测替代平均模型效应。

所有预测必须携带 sentence_id、ref_id、image_id、eval_split、seed、cell/K、grounding_correct 与各模型confidence。CSV缺sentence_id时，与原canonical行序逐项核验后重建；不能按ref_id去重合并。保存恢复锚点、原容差和恢复误差。

## A — 统计纠正与温度

1. 正式 image-cluster bootstrap：5000次、seed0、95% percentile CI，抽样单位image。
2. 每seed先算效应，再在每replicate取均值；relative先各seed计算，再mean。DoD与跨severity macro也在replicate内部完成。
3. NaN/单类AUROC/零分母不能填0；报告无效次数，不足以形成可靠区间时标记不确定性不足。
4. 逐seed区间与seed间标准差单列，区分条件于固定三个模型的测试不确定性与训练随机性。
5. 核查所有进入摘要、主表、gate的端点平均：Phase05/1/1F、RefCOCOg、V2G/D/P、LCR/M25/M3。已有正确方法只验证，不无谓重跑。
6. 读取无损预测优先。无预测先恢复冻结模型前向；无模型才按原超参数/划分做确定性恢复。原锚点与容差不放宽；恢复失败标UNVERIFIABLE。
7. 保留E-AURC/RER数值定义，修正其完全accuracy-independent的解释；旧RouteA判定留历史，但不能作为独立discrimination证据。
8. B0已存分数比较native、冻结global-T和validation per-K温度；直接报告AUROC/校准/风险变化。per-K采用原诊断值，缺值时仅用validation拟合、标为诊断，不用test选温度。
9. 交付旧新点估计/CI/操作判定对照，不改变历史阈值，不自动把操作gate等同科学结论。

## B — 语义分组与ID/OOD

原train/tune图像划分、B3三个seed、grounding预测均保持固定。标准化仅拟合训练数据。Logistic C=[0.1,1,10]，沿用原selection tie-break（差小于0.002选更简单的先前值）。

### 语义分组（结果前固定，本轮不按结果删选）

S为原Stats17。Q含8列：`clip_top1,clip_top2,clip_margin12,clip_entropy,clip_normH,clip_rank_top1,q_top3,q_margin13`。

V含8列：`cand_vmax,cand_vmean,cand_vstd,cand_top12_sim,cand_top15_mean,density_070,density_080,cand_top13_sim`。

比较S、S+Q、S+V、Full=S+Q+V；train/tune只K5/K10。测试random K5/10/20/50、matched random/hard K5、固定K10 m=0/2/4/8。主比较Full−(S+Q)，同时保存S+V−S、Full−S与absolute指标。所有interval调用A工具。

若Full相对S+Q未取得稳定增量，不能维持交互核心主张。重跑S/Full先核对原reference，任何差异必须说明环境/精度/选型来源，不改旧数值。

### 分数模型ID/OOD

只使用既有Stats Logistic与ScoreDeepSets，分别小K=[5,10]、大K=[20,50]训练及选型；相同train/tune图像、相同行数预算，所有K test只最终评估。每设置独立train-only标准化；沿用原网格、architecture、max_epochs300、patience30，不追加混合训练或架构搜索。

保存模型、选择配置、train/tune loss/AUROC、逐epoch记录、选中epoch及test逐样本预测。补固定128训练行的拟合能力诊断与确定性score→MSP基线；拟合诊断不当作泛化证据。遇到已复现的实现错误，先最小测试，再修复和重跑受影响结果；否则保留模型失败，禁止无限调参。

## C — 候选与机制

1. 对照referring target ann_id的真实category与target proposal最大IoU对象category，保存mismatch逐表达式与汇总（类别/图像/物体）。
2. hard及dose各cell报告proposal数、distinct GT object数、objectness、area、IoU、重复程度。proposal条数不称独立物体数。
3. mismatch=0则无需新构造。mismatch>0则版本化生成真实target类别敏感性候选，用旧新都可用的配对交集，报告流失与availability；冻结B3/reliability只前向，不训练。
4. 保持原Shapley公式，新增四角、两路径、有符号贡献、I=A11−A10−A01+A00及共享抽样CI。不用净小D_label暗示label无作用。
5. H1c只解释accuracy harm，不能替代AUROC gap；“机制被否证”改为测得解释未获得支持，“nothing”改为未匹配所用标注对象。

## 主代理集成与验收

- 修正RQ编号、V2实际训练状态、freeze证据等级、旧报告数值错误和引用；公共首页明确受控GT条件与共享COCO视觉域。
- 最终指标/CI/图表从修正版产物自动生成，不手填。论文按问题依赖组织，稳健性嵌入命题，历史gate/hash/流水账进附录。
- 测试覆盖温度反例、E-AURC随机排序与RER上界、共享抽样/seed/重复表达/单类/relative、分组及split隔离、类别错配/同物体多proposal/敏感性交集、四角及interaction。
- 验证受影响测试及完整tests。旧错误叙事断言可改成科学不变量，但必须记录理由，不删除测试绕过失败。
- 验证旧输入hash不变，恢复原anchor容差不放宽。
- 最终 `reviews/repair_completion.md` 逐条标明改动、正式重跑产物、主张变化、剩余限制；只在所有必需工作完成时写COMPLETE。

## 当前状态

本文件写入时只完成执行准备，三名指定模型的执行agents尚未派遣成功。不能将协议与manifest的完成误读为实验修补完成。机器状态请以修补输出目录STATUS.json及实际产物为准。
