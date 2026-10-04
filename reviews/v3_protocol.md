# V3 独立确认与自然候选验证协议

执行起点：2026-10-04；历史基线 Git `947b8b2e87cb754853a6bb7ff1fd140a5229b3a5`。
本文记录确认结果观察前的协议。正式封存后本文只读；冻结身份、运行及完成状态分别以 `protocol_freeze.json`、`confirmation_run.json` 和 `STATUS.json` 为准。本文的假设不根据确认结果改写。

用户授权有限扩展、一到三个月：FineCops 未使用官方划分、既有 B/16 特征来源复制、自然候选评估；本阶段不新增架构、seed、proposal family、机制控制实验或稳健性矩阵。理论线收束。旧 FineCops STOP 与 M3 UNVERIFIABLE 保留。

## 研究对象与假设

受控接口沿用目标在场、唯一有效目标、嵌套随机扩展及最大 K 共同样本。自然接口取冻结 RPN 原生 NMS 后按 objectness 排序的 top-K，不强制目标、不清除其他合法目标框、不额外去重；稳定 proposal identity 解决并列。K=5/10/20/50，记录请求及实际 K。图像中有目标而候选未覆盖，与图像中不存在目标是不同事件；本轮只分析前者，不引入负文本、编辑图像或 NONE head。

新受控随机候选沿用历史 seed=20260927，每条 FineCops positive 表达使用 `numpy.random.default_rng([20260927, int(expr_id)])`。目标选最大 IoU 框，同分按稳定 proposal identity；其余 IoU>=0.5 框均不作 distractor。一次负候选 permutation 的前缀构成全部 K，三个 scorer seed 和两个 backbone 共用同一候选身份。表达身份另带来源及官方 split，禁止依赖 Python 的随机 hash。

四个主要确认比较仅在新确认集受控 random 共同样本内：

| 名称 | 每个固定 seed 内的效应 | 解释 |
|---|---|---|
| C1 | B0 MSP AUROC(K5) − AUROC(K50) | 指定 scorer 的规模效应迁移 |
| C2 | B0 Full AUROC(K50) − (S+Q) AUROC(K50) | 指定学习流程中的 V 增量 |
| C3 | B/16 Full AUROC(K50) − (S+Q) AUROC(K50) | 新配置的来源复制 |
| C4 | B0 large-K trained SDS AUROC(K50) − small-K trained SDS AUROC(K50) | 既有训练支持效应迁移 |

预期方向为正，但负值、跨零或未定义均是有效结果。C2/C3 分别判断，不平均配置以隐藏失败。FineCops 图像来源与表达生成流程共同变化，不把域迁移结果单独归因为视觉域。

自然接口的所有指标和官方 difficulty 分组为预先声明的次要分析，不要求 AUROC 随 K 下降，也不作为主要比较的替代胜利条件。自然接口覆盖率与选择正确率分别报告；条件 AUROC 和全体 AUROC 可因样本构成不同而变化。目标未覆盖子组全为错误，AUROC 未定义。

## 暴露审计与确认集

新数据只写入 `data/raw/v3_finecops/`，新产物 `results/v3_final_validation/`。历史缓存、模型、预测、候选和修补结果只读；既有未跟踪 `audit/` 不改动。

官方 positive train 中经审计的最多 200 张图像作为开发集（seed=20261004）；官方 positive val 全部剩余合格图像作为确认集。审计逐项区分 metadata_downloaded、image_downloaded、audited、trained、scored，保存文件与像素身份、来源映射、近重复规则和全部排除原因。未入本项目研究流程不等于基础模型预训练未见。未知或无法核查的曝光不能被自动视作干净。

保守排除全部历史 FineCops test 图像，及原 train/tune/test、RefCOCOg、人工审查等历史图像；跨数据源匹配不靠数值 image_id。开发与确认之间也排除重复及近重复。确认性能未计算前固定审计规则。没有可信确认数据就记录 BLOCKED_CONFIRMATION，不重划旧 test、不放宽旧 gate、不因结果选新数据源。

执行期发现：VG 官方图像元数据含 `coco_id` 与 `flickr_id`（https://visualgenome.org/api/v0/api_readme）。因此 GQA/VG 命名空间不同不足以证明非 COCO 图像来源或跨视觉域。来源审计必须报告新划分的 COCO 映射比例、历史使用重叠和最终保留图像来源；确认身份独立与视觉域外推是两个判断。原计划“非 COCO 数据源”的表述改为待审计命题，不能在结果出来前写成已满足。此处只纠正来源事实，不根据模型绩效换数据源或重新定义假设。

暴露验收证明的是冻结筛查流程通过：历史身份清单全部纳入排除、可用来源映射核对、全部本地历史图像与新图像完成规定指纹检查。VG 官方记录存在且 `coco_id=null` 与来源 ID 无法映射不同；前者如实记录来源字段为空，后者属于未解决的身份问题。历史元数据清单覆盖与实际像素覆盖分别报告；未保留像素的历史图像不能声称已进行像素比较。近重复筛查及来源映射仍可能漏检，论文不能把验收通过写成对基础模型预训练暴露或所有历史近重复的完备证明。

确认性能观察前固定标注结构规则：目标框必须有限、宽高为正并与实际图像有正面积交集；存在结构不可用目标框的图像整张排除，记录表达、原框、实际尺寸和理由。部分越界但仍有正交集的官方框保持原值，单独报告，不推定为无效或 amodal 标注。本轮此规则排除一张图像及其唯一表达，旧阻塞清单和原文件保留；不能据框不可用声称图像中不存在目标。最终确认 3107 张图像/7993 条表达，开发 200/522。

## 模型与接口

B0 与 B/16 使用各自对应既有 scorer 三个 seed 和既有源温度；FineCops 上不拟合温度、标准化、grounding 或可靠性模型。B/16 重新按修补版 S17、Q8、V8、Full33 定义生成特征，不复用旧 V2 sem14 模型；在原 523/224 train/tune 图像、K5/10 上选 C=0.1/1/10，沿用原 tie-break，标准化只训练拟合。

B/16 COCO 复制包括 random K5/50、matched random/hard K5、K10 m0/m8。Full−(S+Q) 增量、hard−random 与 m8−m0 增量差直接共享抽样计算。它仍是已观察测试集上的配置复制，不能升级为独立确认。

自然样本全部进入流失表。空候选计为无法定位；K_eff=min(K_requested,N_valid)，不补虚假候选。K_eff<5 的样本记录基本定位与原因，Full/组间可靠性指标不伪造。可靠性报告注明 K_eff>=5 的分母，并报告相对全部输入的可评分比例，不过滤成 N>=50 或 target-present。

逐样本保留 source-qualified expression/sentence identity、image identity、candidate mode、requested/effective K、proposal IDs、目标覆盖、有效目标框数、选中正确性、seed、模型概率、缺失原因。多表达不得按 ref_id 合并。GT 只进入受控构造及评估标注，不进入自然排序或可靠性特征。

自然接口中，候选未覆盖目标的预测必定错误。为检查 Full−S+Q 的整体增量是否主要来自这部分错误，使用同一配对可评分交集，分别报告覆盖条件下的增量、正确预测对未覆盖错误的 cross-AUROC 增量，以及按错误样本数加权的两个有符号贡献。恒等式为 overall-AUROC = covered-error 比例×covered-AUROC + uncovered-error 比例×cross-AUROC；增量同样分解，区间共享图像抽样。这是成对排名审计，不识别因果机制；不存在该类错误或正确预测时相应指标未定义，零权重贡献可为零。

## 统计与风险并列规则

正式 5000 次 image-cluster shared bootstrap，seed=0；所有比较使用共同图像抽样，在 seed 内算效应再平均三个固定 seed，不平均预测。95% 边际与四比较 Bonferroni 98.75% percentile CI 来自同一批原始抽样。保留 seed 点、seed SD、无效抽样与分母；有无效抽样的区间注明其条件性，不默认为有效确认。测试抽样不涵盖完整训练与搜索随机性。

V3 Risk@50%/80% 固定 **fractional_boundary_tie**：以名义接受量 cN 选高置信度样本，边界同分组接受所需比例，风险等于随机接受边界同分样本时的期望错误比例。这使指标对同分样本行序不敏感；每次 bootstrap 使用同一定义。历史 ceil(cN)+stable-sort 指标不改，跨版本不能混用风险端点。

空样本、单类 AUROC、非有限置信度及零分母拒绝或标未定义，不能填零。不显著不等于等效。不新增结果驱动 gate，不以修补使假设全部通过为目标。

## 冻结、调度与验收

主代理负责协议和冻结工具；A 文献/暴露审计，B B/16 复制，C 候选/推理/统计。全部执行 agent 为 gpt-6-luna、max、fork none，互不编辑同一文件。最多一个训练进程和一个重统计进程；GPU 串行、BLAS 两线程，候选嵌入分块读取。

正式 freeze-v1 绑定 code/model/cohort/config/exposure 的 SHA256 与真实 PASS 的暴露、开发、复制报告。`verify_freeze` 在确认前核对所有字节；确认运行 ledger 独占创建，避免静默第二次考试。流程屏障不等于不可绕过的访问隔离；只要发生确认结果暴露，必须如实记录。

确认前验收：身份与表达对齐；自然 GT 独立与多个合法框；空/低 K/流失；特征分配与 train-only 标准化；冻结预测锚点；bootstrap 与风险并列；源输入哈希；新增及完整现有测试。实现错误重跑须保留原运行、最小复现、原因和纠正记录，禁止绩效驱动重选模型或扩样本。

完成产物包括文献定位、暴露审计、复制结果、自然接口实测、正式确认、原始预测/分布、论文蓝图和 `reviews/v3_completion.md`。阻塞与未执行不能记作完成。
