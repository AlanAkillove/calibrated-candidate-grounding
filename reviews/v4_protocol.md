# V4 targeted strengthening：预冻结协议

日期：2026-10-05。基线 `a203744cba2a44b12514012a966a753f028ab1e2`；分支 `v4-targeted-strengthening`。主代理作 GO/STOP 决策，执行代理统一 gpt-6-luna / max。用户最新授权由主代理自主决策；本协议的冻结仍是运行前的必要条件。

## 研究地位与保护范围

V4 是观察 V3 后针对 W1/W2/W3 的补充研究。FineCops 同一批表达已经评估过，V4 在这批数据上的分析不是新的独立确认。新特征、指标、覆盖率与分层在 V4 新结果计算前固定；这不消除结果驱动选择研究问题造成的限制。

禁止写入 `results/v3_final_validation/`、`results/research_repair_v1/` 或原冻结模型、候选、缓存、预测、历史表。初始审计记录每个保护文件的完整 SHA256，终验逐项比较。所有新运行产物只在 `results/v4_targeted_strengthening/`。不修改用户已有 `audit/`。

## 冻结顺序

P0：本协议、机器可读配置、12 维特征实现及依赖审计、统计方法、输入审计哈希封存后，允许开发 runner 和源域训练；禁止计算 FineCops 正式 V4 端点。

P1：源域训练模型、标准化、选择日志、ambiguity 训练阈值、样本清单与最终分析代码封存后，允许一次正式 A/B 分析。代码错误可修正续跑，但保留原版本、失败记录、最小复现和原因；不得以结果不好为理由改设计。

C-pilot：先通过官方数据来源/权限/实际图像获取门槛，再冻结数据集及 400 张开发图像的 RPN-only 审计清单。此时不运行 CLIP/scorer/可靠性模型。

C-final：通过 proposal 门槛后，冻结外部正式清单、输入身份、模型、候选、统计及保留下来的主要比较，才允许正式 external forward。不经过 C-final 不生成外部确认成绩。

## 共用数据、模型和统计

A/B 受控主要样本：V3 FineCops 正式 positive val 中 controlled random K50 的共同样本，2888 张图、7004 条表达；保留 sentence/record identity 和全部重复表达。次要受控 K5/10/20 使用相同共同样本。自然接口使用 V3 相应 requested K 的全部可评分样本，不套用受控 K50 流失过滤；报告不可评分数量、覆盖及 covered-only 结果。

配置分别报告 B0（OpenCLIP B/32）与 B/16。各配置三个既有 scorer seed 1/2/3，原温度、grounding logits、候选身份、预测标签不变。绝不平均预测形成 ensemble。可靠性新模型只在原 COCO reliability train 523 图上训练、tune 224 图上选择；FineCops 不参与拟合、温度或选型。

5000 次 shared image-cluster bootstrap，seed=0。每个 draw 对同一图像内全部表达赋同一抽样多重度；所有模型、三个 seed、分层及差分共用抽样。在 seed 内求端点/效应再平均三个 seed。提供逐 seed 值、seed SD（ddof=1）、绝对端点、95% CI、调整 CI、原始分布与无效次数。单类 AUROC、空分母、空组标未定义，禁止填零。任何主要比较存在无效抽样，其可用抽样区间必须标 conditional，不能判为正式 SUPPORTED。

固定一个 A/B 主要 family，六项：A 的 Full−SQ ΔAURC（两配置，预期负）、A4 的 high−low ΔAUROC 差分（两配置，预期正）、B 的 Full_pure−SQ ΔAUROC controlled K50（两配置，预期正）。Bonferroni 99.1666666667% percentile CI，分位数 [0.0041666666667, 0.9958333333333]。同时报告边际 95%；判定不在两种区间之间择优。

次要分析：绝对 AUROC、各固定 coverage 的风险及差分、AURC/E-AURC、受控其他 K、自然 K5/20/50 overall/covered-only。次要 95% 区间不构成新的确认 family，不能以某项显著救回主要假设。不设置事后 deployment 成本或任意最小实用阈值。

## A：已有增量的实际大小

假设：已有 Full−SQ 排序增量可对应较低的选择性风险。估计对象为三个固定 seed 下同一评估总体的平均指标差分，不是因果干预效应。A 不训练任何模型。

读取 V3 无损保存预测。主要比较 controlled K50；固定 coverage 为 0.2/0.3/0.4/0.5/0.6/0.7/0.8/0.9。画 MSP、S、SQ、Full 的绝对风险曲线，同时报告 Full−SQ 差分。边界并列置信度使用 fractional expected acceptance；不以输入顺序破并列。不挑最好 coverage 作为唯一正文结论。

V4 的 AURC 使用连续 fractional-tie 解析积分，版本名 `AURC_V4_CONTINUOUS_FRACTIONAL_TIES`。某并列组之前接受 m 个样本、e 个错误，组大小 g、错误 h，p=h/g，其积分贡献 m>0 时为 `[p*g+(e-p*m)*log((m+g)/m)]/N`，m=0 时为 p*g/N。coverage=0 的风险本身未定义；图上若展示极限须注明。Oracle AURC=(1−a)+a log(a)，a=0/1 分别按极限处理。E-AURC=AURC−oracle，不宣称排除了 accuracy 影响；相同标签比较中 ΔE-AURC=ΔAURC。旧 AURC 不覆盖、不与新积分口径混算。

Pairwise：对每 seed 的真实正确/错误对 P=n_correct*n_error，分别统计 SQ 与 Full 的 win/tie/loss 以及 3×3 转移。win=1、tie=0.5、loss=0。净恢复 credit/P×10000 后再平均三个 seed；不把各 seed 不同分母混成 pooled pair estimator。真实 pair 数、净恢复及反向损失均保存。点估计分块精确计算；bootstrap 用加权 AUROC 求同一 pair-credit estimand，不重复 O(n²) 转移枚举。纳入同图与跨图对，抽样相关性由图像 cluster 保持。结果保存在 `practical_effect/pairwise_gain.csv`。

风险下降同时换算每 1000 接受样本及每 1000 输入表达减少的期望错误数，后者乘以固定 coverage。允许结论是指定条件和覆盖率下可测但有限的收益；禁止泛化为任何应用都值得部署。

### A4：唯一异质性分析

主要 ambiguity=候选 crop unit embedding 的 mean nearest-neighbor cosine（对角排除），对应 V_pure 第5维。不能读取 query、winner、rank、label、reliability score。

每配置分别在原 source TRAIN 的 controlled random K50 表达上按图像均衡权重定 tertile：每图总权重1，图内表达均分；采用加权经验 CDF 的 inverse quantile 1/3、2/3。不使用 tune/FineCops/test 效应定阈值。low≤t1、mid为(t1,t2]、high>t2，边界并列不拆分。阈值和源训练身份在 P1 封存。

正式比较 high−low 的 [AUROC(Full)−AUROC(SQ)]，在每次完整图像抽样内求组效应和差分。展示三组绝对端点、效应、阈值、图像/表达/正误数。不用“高组显著、低组不显著”证明交互。若训练阈值重合、任一极端组不足50张图或任一 seed 正/误样本各不足10，则 NOT ASSESSABLE，不重新划组。不得额外 subgroup mining。

## B：隔离候选集合函数依赖

四组：S17、S+Q25、S+V_pure29、Full_pure37。S/Q 保留修补后相同顺序/计算方式；Q 可以依赖 query 与 scorer 排序，绝不称 Q/V 统计正交。旧 V8 八列全部经 winner/rank 依赖 query，仅用于 A 和历史对照。

V_pure 输入严格只有 unordered crop_embeddings[K,D] 与 candidate_boxes[K,4]；无 objectness、query、GT、winner、rank、label 或 identity。float64 重新单位化，K≥5，无效/零向量/非有限输入/非正面积拒绝，不静默补值。12维固定顺序见 `src/ccg/v4/pure_features.py` 及依赖审计：

1. i<j cosine mean；2. population std；3. linear q50；4. linear q90；5. mean NN cosine；6. max NN cosine。
7. 单位 crop Gram eigenvalue entropy effective rank / min(K,D)，clip负特征值为0；不含 query 排名。
8. pair IoU mean；9. linear IoU q90；10. fraction pair IoU strictly>0.5。
11. mean pair center distance / enclosing-candidate-box diagonal；12. population std(log positive pixel area)。

给定同一个候选集合，这个函数与 query 无关；受控候选构造上游使用 GT/表达，因此不能声称端到端统计独立。自然同图同K跨表达必须得到相同 V_pure。单位化、列分配、排列不变性、无 query 输入和 grounding 预测一致性要验收。

复用精确冻结的 S/SQ 控制模型（不重复训练无变化控制）；新训 S+V_pure/Full_pure。source train K={5,10} 行数与 V3一致；仅 train 拟合 population mean/std，标准化分母 std+1e−8。Logistic C={0.1,1,10}，等权平均 K5/K10 tune AUROC，best 的0.002内优先小C，沿用原优化器/上限，不增网格。各配置3seed，共36次C拟合。保存每C train/tune loss/AUROC、选中参数、标准化、模型及逐表达预测。冻结 S/SQ 前向须回放 V3 锚点，不放宽既有容差。

主要效应 Full_pure−SQ AUROC controlled K50，各配置单独判。报告绝对端点与 Risk50/80/AURC。受控 K5/10/20、自然 K5/20/50 是次要边界分析。

正结果仅允许“给定候选集函数不读取 query 的结构特征，在此冻结可靠性模型下有条件增量”。不支持因果机制、语义必要性或普遍收益。负/不确定结果只能说此12维特征和训练流程未得到相应支持；不能据此证明所有纯候选信号无效，也不能证明旧 V8 的信号必然全部来自 query。不另造 V2/V3 特征，不 residualize，不换架构。

## C：外部可行性与门槛

官方论文/作者仓库核实 ReferItGame/SAIAPR、RefEgo/Ego4D、Talk2Car/nuScenes、Flickr30k Entities。审计矩阵保留日期、查询式、URL、正文位置、阅读深度、纳入/排除理由。图像与标注的许可分别记录；链接存在、HEAD可达、完整下载、实际解析和合法使用分别标状态。官方总量不冒充本地可用量。

预先优先次序 ReferIt→RefEgo→Talk2Car→Flickr，仅在未算 proposal/模型成绩的来源、任务、权限、访问门槛失败时进入下一项；不能按 recall 或结果更换。正式 W3 必须图像来源清晰不依赖历史 COCO 集，且视觉域确实不同。经典自然照片不同语料库本身不证明视觉域差异；ReferIt/Flickr可判任务适配但 W3 证据不足。不把基础模型预训练未见当作可证明事实。

至少约500图/1000正表达且图像真实、expression→box可无歧义适配。图像本身有目标但proposal漏检可纳入；不加入负表达、编辑负图或NONE分类器。历史曝光审计含source ID、文件hash、解码像素hash、冻结近重复规则；不满足清洁样本定义则停止正式确认。

如果实际图像权限/访问或明确域差异不成立，STOP external，W3=NOT ASSESSABLE，而非“迁移失败”。不会为了报告完整而下载不合适镜像、代签个人许可或换十个数据集。

只有数据前门槛成立才运行400张官方开发图像（seed20261005）的小 RPN 审计；V3 Faster R-CNN R50-FPN COCO_V1 frozen RPN，N64、原生NMS、无额外去重。报告recall@.5/.7、natural topK覆盖、K5/10/20/50可用率、proposal数、pairIoU重复率及表达/图像流失。controlled GO门槛：bank target recall@.5≥.70、K50可构造表达≥50%、400图中≥200图可构造，且正式预计共同样本≥500图/1000表达。任何失败STOP controlled；不得通过更换detector或调候选规则救回。natural-only若可做只能描述边界，不能冒充External-C2。

若 GO，C-final补充冻结具体 official split、全部符合身份/有效标注规则的固定正式样本（不按模型性能筛选）、GT adapter、模型和预计成本。受控与自然规则沿用V3，不改变IoU .5。视频/驾驶序列需原clip/scene身份：保留image-cluster主区间及clip/scene-cluster保守敏感性，任何迁移支持要求两者调整区间均支持；若无法恢复序列ID则NOT ASSESSABLE。

External-C2（不重复C1）的 surviving claim选择规则在 B 结果前固定：两配置B主要调整CI均>0，则两者验证Full_pure−SQ；仅一配置成立，则仅该配置验证pure，仍保留两family槽位（97.5%CI）；两配置均不成立，则两者验证已有V3 legacy Full−SQ这一预定较稳健结论。无其他候选 endpoint。外部 family与开发A/B分别报告、最多两项，Bonferroni97.5%、边际95%、5000共享cluster draws、seed0；不利用外部结果再选择模型。

## 预算、资源及停止

A/B 只用已存预测/embedding，CPU Logistic，GPU预期0；最多一训练进程与一重型统计进程，BLAS每进程2线程，bootstrap先分块缓存排序避免每draw重复O(n²)。CPU预计4–24小时、新增磁盘0.5–5GB。磁盘初查空余约1.47TB，预算依据实际字节持续记账。

C若GO，小RPN+两配置冻结crop/text预计1–6GPU小时，新增下载/缓存控制30GB以内，先核实官方包容量；硬预算20GPU小时，12小时预警，最多1GPU进程。不跑新seed/架构/二维机制/全网格，也不做额外proposal-family、residual诊断。当前没有外部性能或proposal实测，预算不是完成事实。

|路线|当前决策|进入下一阶段条件|停止时如何报告|
|---|---|---|---|
|保护审计|GO，初始PASS|终验同一SHA清单|任何变化STOP验收|
|A固定效应解释|GO，P0后开发/P1后正式|保存预测对齐与新数学测试通过|不扩coverage/调参救效应|
|A4唯一分层|条件GO|训练阈值封存、极端组可评估|不足则NOT ASSESSABLE|
|B纯集合特征|GO，P0后源域训练|特征测试、原锚点与split隔离通过，P1封存|负结果保留，不新造特征|
|C外部|审计中；正式NO-GO|来源/明确域/权限/访问/规模→RPN门槛→C-final|无法满足则NOT ASSESSABLE|
|额外proposal/residual/模型|STOP|本轮不安排|保留single-family边界|

## 判定与终验

SUPPORTED：预定方向主要调整CI支持且无未说明invalid/bootstrap/身份问题。NOT SUPPORTED：主要调整CI支持相反方向或该路线在本流程下未获支持（必须另述是否只是未排除0）。INCONCLUSIVE：区间跨0且可计算；不能写等效/无信号。NOT ASSESSABLE：数据、身份、组样本或接口不足未形成可信检验。

终验覆盖合成数学/并列风险/pair转移一致性、共享图像抽样/重复表达/seed对齐/单类与差分、特征输入与排列/自然跨query一致、训练选型隔离、grounding复现锚点、保护输入哈希不变、受影响和完整既有测试通过。每项记录状态；统计或文件还未产生不得记完成。

最终交付实际W1/W2/W3科学评估、绝对数字与证据索引、可复现命令、执行预算和限制，及论文draft/blueprint V4。论文主线保留受控现象→训练支持/有条件信号→解释审计→V3独立确认→自然边界，V4嵌入对应命题，不单独罗列实验流水账。四角/路径分解仍是解释审计，不识别因果机制。达到本协议GO路线验收、STOP路线有证据后停止加实验；负结果有效，W3未取得确认必须保留未解决标记。
