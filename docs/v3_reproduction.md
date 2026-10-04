# V3 证据复核与复现

当前实验完成状态以 [STATUS](../results/v3_final_validation/STATUS.json) 和 [验收](../reviews/v3_completion.md) 为准。以下命令不将未完成阶段写成已有结果。

## 不运行模型的复核

先读 [产物生成的索引](../results/v3_final_validation/evidence_index.md)、[文献定位](../reviews/v3_novelty_review.md)、[数据审计](../reviews/v3_data_exposure_audit.md)、[复制验收](../reviews/v3_replication_report.md)。正式比较的对象、分母、区间和无效抽样在对应 JSON 中。

逐样本预测以无损 gzip 保存，`predictions_package.json` 含压缩与原文件的 SHA256、字节数和恢复命令。恢复只解压已保存字节，不运行模型，不覆盖身份不同的现存文件。例如：

```powershell
python scripts/v3_package_predictions.py restore results/v3_final_validation/pipeline/dev/inference/predictions_package.json
```

正式确认已完成，其预测位于 `pipeline/confirmation/inference/`。B/16 复制的逐样本预测为 `replication/semantic_ablation_predictions.csv.gz`，其身份见复制验收。

```powershell
python scripts/v3_package_predictions.py restore results/v3_final_validation/pipeline/confirmation/inference/predictions_package.json
```

预测 schema 的 `target_coverage` 明确表示候选是否覆盖目标；`target_present` 是历史接口保留的候选覆盖别名，不能读成图像是否含目标。`bank_target_coverage` 指完整原生库，`presented_valid_target_count` 指呈现前缀中的合法框条数。所有本轮 FineCops 表达来自官方正例，候选未覆盖或空候选属于 proposal/定位流程失败，不是新 NONE 任务。

冻结 cohort 清单中的 `model_scored=false` 等字段是审计/封存时状态，保持原字节用于身份核验；确认完成后的真实状态由运行台账与 STATUS 记录，不回写冻结清单。

共享抽样原始分布以 NPZ 保存，summary 的 `raw_draws.keys` 给出每个效应的均值与逐 seed 数组对应关系。可直接从有限抽样数组重算 percentile 区间，而不下载图像或调用 GPU：

```python
import json
from pathlib import Path
import numpy as np

folder = Path("results/v3_final_validation/pipeline/confirmation/bootstrap/controlled")
summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
with np.load(folder / summary["raw_draws"]["path"]) as draws:
    values = draws[summary["raw_draws"]["keys"]["C1/auroc_correct"]["mean"]]
    finite = values[np.isfinite(values)]
    print("invalid", len(values) - len(finite))
    print("95%", np.quantile(finite, [.025, .975]))
    print("98.75%", np.quantile(finite, [.00625, .99375]))
```

对应正式产物已完成并通过保存分布验收。有限抽样区间条件于可计算抽样，单类情况不能填零。自然分析没有四比较的 family-wise 调整，不将其 95% 区间称为主要确认区间。

确认的 RPN 几何库还保存为 `pipeline/confirmation/frozen_proposals.h5`，与原缓存逐字节相同，身份见 `candidate_geometry_identity.json` 和确认 freeze。这只含 proposal 坐标、objectness 及索引，没有原图或 CLIP 嵌入。恢复确认 JSONL 后，可运行 `python scripts/v3_validate_final_evidence.py` 核验每个 winner、GT IoU、稳定并列、候选身份和原始抽样分布，无需 GPU 或 JPEG。它检查的是保存证据的一致性，不能代替完整像素暴露审计和模型前向重建。

## 精确模型身份

`model_replay/fitted_source_models.zip` 保存 56 个直接使用的小型 scorer、可靠性模型和元数据文件；`manifest.json` 记录全部 59 个输入身份。三个大型公共权重（两个 OpenCLIP、一个 RPN）不随仓库发布，以原路径、SHA256 和原模型元数据定位。压缩包只打包已有权重，没有新增训练。

```powershell
python scripts/v3_package_models.py verify
python scripts/v3_package_models.py restore
```

restore 对现有文件先检查身份，不替换不同字节。原权重和元数据仍是主输入，压缩包不是另一套模型。部分源元数据保留原 Windows 绝对路径，跨机器复现时必须显式记录路径适配，建立新的运行清单，不能把改写后的字节冒充原冻结身份。

## 完整重建

运行环境见 [environment.json](../results/v3_final_validation/environment.json)。原实验使用现有 `deepminer`、RTX 4060 Laptop GPU、两个 BLAS 线程。源训练/调参/测试图像和已有源模型需按照原项目下载及训练流程恢复，并满足原有锚点；不能放宽容差。

官方 FineCops positive train/val、Visual Genome 来源映射、历史暴露清单及图像获取步骤由 `scripts/v3_audit_exposure.py` 和数据审计记录定义。原图不随 Git 发布；FineCops 标注许可不能覆盖各图像的原许可。完整像素复核需要实际图像；只有清单不能代替像素检查。

准备、封存、推理、统计是不同命令：

1. 审计最终数据并在 dev 上通过接口、锚点和完整测试；B/16 复制实际完成。
2. 主控 `v3_prepare_authorization.py` 将实际 PASS 与源图像/模型/代码绑定，只开放 proposal/CLIP/candidate 准备。
3. `v3_run_pipeline.py prepare --stage confirmation --authorization ...` 保存派生输入，不计算性能。
4. `v3_seal_confirmation.py --authorization ... --preparation ...` 封存全部派生输入和协议。
5. `v3_run_pipeline.py infer --stage confirmation --freeze ... --ledger ...` 独占建立运行台账，首次正式前向。
6. `v3_bootstrap_confirmation.py --predictions ... --freeze ... --ledger ...` 只读保存预测，以 5000 次共享图像抽样生成统计。
7. `build_v3_report.py`、`build_v3_figures.py` 只读取正式结果生成最终论文表图。原 `v3_plot_results.py` 保留确认前测试快照字节；最终展示驱动只修改图例、刻度与轴标题，不重新推断。
8. `v3_validate_final_evidence.py` 使用公开的原 proposal 库副本、确认 GT 和保存预测核验原始分布、winner/GT、冻结候选身份与现有源文件。对所有封存图像/嵌入/模型的完整输入核验仍要求原缓存，不能混称保存结果复核与全流程复现。

原确认台账不能复用作第二次考试。复现既有结果属于复制研究，使用新的输出路径、运行身份和“已知结果的复制”标签。实现错误续跑必须另保留错误证据、旧输出、修改原因和新运行记录。

统计运行的完整输入验证需要全部原冻结文件，包括图像与特征缓存；精简 Git checkout 不会自动满足这些条件。精简仓库可以复核公开预测、已保存抽样分布、指标和源模型身份，完整前向重建还需要数据和公共权重。不要把这两种复核能力混称为完全复现。

源文件验收比较原始字节，Git 换行策略也是复现身份的一部分。若其他平台的 checkout 改变了 CRLF/LF，必须按原快照恢复字节或另立来源适配记录，不能悄悄放宽哈希检查。新哈希绑定的 V3 源文件和结果通过 `.gitattributes` 保留原字节。
