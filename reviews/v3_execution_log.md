# V3 执行与纠正记录

本文件记录开发期的事实纠正和实现修正；正式确认运行及错误续跑另存不可覆盖的运行记录。确认性能尚未观察，以下开发修正不构成第二次正式确认。

## 2026-10-04：来源事实纠正

GQA/Visual Genome 图像命名空间与 COCO 不同，但 Visual Genome 官方图像元数据包含 `coco_id`，不能据此声称新图像来自完全不同视觉域。独立确认的身份审计与视觉域外推分开判断。保留原数据源和主要比较，不根据模型表现更换确认集。

依据：[Visual Genome 官方 API 说明](https://visualgenome.org/api/v0/api_readme)，本地逐项映射计数见 `results/v3_final_validation/exposure/source_identity_counts.json`。映射只是来源线索，最终是否历史暴露还要核对使用清单、实际像素与近重复。

## 2026-10-04：可靠性标签按配置区分

开发期检查发现 C3 的计算接口最初复用了 B0 的正确性标签。B0 与 B/16 虽然共享候选，但各自 scorer 可能选择不同框。因此接口改为 `correctness_by_seed[seed][backbone][K]`，C3 使用 B/16 自己的标签。此问题在确认推理前发现；验收要求包含不同配置标签相反的最小反例。

## 2026-10-04：B/16 旧缓存的样本范围

Phase 0A 通用候选缓存有 21,373 条可用表达，冻结 scorer 的 canonical cohort 有 20,799 条。最初错误地要求两集合相等，造成开发运行停止。修正后要求每条 canonical sentence identity 在缓存中唯一存在，逐项核对 ref/image/split，忽略缓存中不属于 canonical cohort 的 574 条。没有改变 scorer cohort、容差、模型或候选规则。

后续六项 K5/K10 冻结打分器前向检查均满足原 `atol=1e-4`；证据来自 `results/v3_final_validation/replication/anchor_forward.json`。冻结输入核对最初通过 834 个原文件；最终集成后仍须重新核对。

## 2026-10-04：历史审计清单角色纠正

`data/audit_subset.csv` 是 COCO/RefCOCO+ 审查子集；旧 FineCops RPN 审查的实际清单是 `results/phase1e_finecops/audit_subset.csv`。暴露登记须区分二者，不能因文件同名混淆数据来源。历史本地图像指纹已覆盖 22,094 个文件，未发生解码失败；新确认集整体是否通过仍以完整暴露验收为准。

## 2026-10-04：风险指标的版本约定

V3 使用协议预先规定的 `fractional_boundary_tie`，对边界同分组作所需比例的随机接受，计算期望风险；历史 ceil/stable-sort 风险不追改。所有表明确版本，不能将两种端点直接混为同一指标。

## 2026-10-04：ScoreDeepSets 前向宽度纠正

新接口的初稿将所有 K 的分数输入填到 K50，与原修补模型按实际 K 进行前向不同。源模型锚点回放在确认推理前发现该差异。修正输入宽度和标准化维度检查后，源 B0 scorer、S/Full Logistic、small/large ScoreDeepSets 各项锚点通过；权重、标准化参数、选型与温度不变。`pipeline/dev/source_model_anchors.json` 留存逐项对照，新的概率回放标准 `1e-9` 与既有原分数标准 `1e-4` 分别标记，不混称原有同一容差。

## 2026-10-04：官方边界框与坐标完整性

预检查发现一部分官方框越过元数据图像右边或下边，大部分不超过 1 像素。原本“所有框必须完全位于图像内”的开发检查过度限制了数据接口，不能据此悄悄裁剪、改写官方 GT 或删除表达。应核对实际 JPEG 尺寸、官方图像尺寸、目标身份与坐标一致性；有限正面积且与图像正面积相交的原框保留并记录越界程度，坐标错配等真实错误另列未解决问题。

依据：[FineCops 官方说明](https://github.com/liujunzhuo/FineCops-Ref#usage)规定 COCO 格式框为 xywh，并提供直接加载接口；[COCO 官方加载实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py)保留标注数值。这些来源支持保留官方数值，不能据此声称每个越界框都是正确的截断或非模态标注。具体数量和坐标核对结果由暴露/标注审计保存。该约定在任何确认性能暴露前确定。

## 2026-10-04：确认前验收与准备授权

最终暴露审计 PASS，确认 3107 图像/7993 表达，开发 200/522。完全离开实际图像的官方目标框按统一结构规则排除整张图像；保留官方坐标、原图、旧阻塞清单和 prescore_annotation_exclusions。不能由框错误推断图像不存在目标。99 个仍与图像相交的越界表达保持原 GT 并报告。

完整测试实际执行 1298 passed、2 原有可选 skipped、exit 0，测试前后源文件哈希不变。测试脚本的展示命令硬编码了 -B，但实际外层调用未带 -B；验收追加 actual_invocation 和说明，计算与源文件未变。这只影响字节码抑制，不改变测试结果。

最终暴露 CSV 比旧开发验收多一次确认样本排除，导致其旧哈希过时。原 dev_acceptance 不覆盖；dev_revalidation 保存旧 CSV 身份，确认差分仅删除一条 val 记录，522 个开发表达逐行、200 个 JPEG 及全部模型/特征/预测/代码均不变。开发复核 PASS。

3291 个源输入已写入 preparation-only authorization；仅开放 RPN、冻结 CLIP 特征和候选清单，不开放 grounding、可靠性前向或性能聚合。B/16 复制与文献审查已完成。确认输入全部封存、唯一正式推理、统计、自然结果与论文整合尚未完成，本记录不宣告 V3 完成。

## 2026-10-04：额度中断后的主控接管

Luna/max 自然接口代理在监督准备过程中因额度限制退出，已启动的准备进程继续运行并于 10:47 UTC 保存 preparation_manifest。用户再次要求继续后，主代理接管固定 CLI 的监督和后续正式运行，没有以其他模型替换执行子代理，没有重训或修改冻结源代码。

主代理在任何 scorer 调用前复核关闭后的候选库、两个完整特征缓存、全部候选身份和既有测试源文件。实际准备验收 PASS：3107 图像、7993 表达、198848 proposal；每图 64 条；两个配置均无无效 crop/文本截断，所有候选身份与源规则重建一致。详情见 pipeline/confirmation/preparation_acceptance.json。旧准备授权和原产物保留，正式运行台账当时仍不存在。


## 2026-10-05（本地）：首次确认、正式统计与最终验收完成

关闭后的准备通过验收后，正式 freeze 绑定 3307 个输入，SHA256 e1d3e33c61b4ff82c5e71ab2cb9953b618418b1bc3d84f1830bc1197d3728613。首次确认于 2026-10-04 15:51:53 UTC 建立独占台账，16:37:55 UTC 前向完成。corrections 为空；未重复正式前向，未因性能重选模型。383664 行预测的原 SHA 为 a4e909abf4cc7ab520667685a35d884f70fa6c601c43abf1a4d384f4e1d0c173。

统计独立读取保存预测，16:39:37–16:50:28 UTC 实际完成两个 5000 次共享图像簇 bootstrap。最终保存分布验收 PASS：逐 seed 均值、95%/98.75% 同分布 quantile、自然有符号排名恒等式、每条候选身份/稳定并列 winner/GT IoU、完整样本流均核对。60 个受控和592 个自然端点均无无效抽样。原有 834 冻结输入未改，全部已测试源字节仍等于确认前快照。后续仅新增证据验收/打包/展示脚本，不修改冻结计算实现；实际执行及 py_compile 通过，不称其已包含在早先完整测试中。

C1 未确认规模判别退化；C2/C3/C4 的调整区间支持指定正效应。自然 K5 反向，K20 增量主要记入漏检排序，K50 B0 获边际支持而 B16 证据不足。所有当前文档按实际结果收束，不追加挽救实验。正文数字来自自动生成产物，完整限制见 v3_completion.md。

打包 CLI 最初误用 pack 参数，在 argparse 阶段退出，日志 packaging_console.log 保留；改用已定义的 package 后仅压缩保存字节，gzip 解压哈希与首次预测一致，没有 scorer 调用、bootstrap 重采样或确认纠正。


## 最终展示源文件边界检查

图例/刻度修订一度改动已纳入完整测试快照的 v3_plot_results.py。发布一致性核对发现这一变更；原脚本已恢复到测试快照 SHA 的逐字节版本，最终显示布局另存为 build_v3_figures.py，并实际生成/检查图像。该展示脚本未纳入早先完整测试，独立执行和编译核验；没有修改冻结统计、预测、选型或输入，也没有重跑模型。所有已测试源字节再次核对一致。原最终验收副本保留为 pre_publication_acceptance.json，发布验收从相同保存结果复核。
