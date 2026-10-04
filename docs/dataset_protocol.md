# Dataset & Proposal Design Audit — `docs/dataset_protocol.md`

对应原始需求 **Task 3**（数据与 proposal design audit）。本文件在**不下载大文件**的前提下，
先核实数据组织、依赖、license、proposal generator 选型、存储与显存预算，并给出风险清单。

写作日期：2026-09-26。状态：pre-download audit（Phase 0 之前）。
实测更新：2026-09-27（下载与解析后核验完成；见 §1.4 实测更新与各小节行内标注——原 `[待下载后验证]`
相关猜测已按实测结果标注为 **已确认 / 已证伪**）。proposal audit 结果见 `docs/experiment_log.md`
条目 `audit-proposal-001` 与 `docs/research_protocol.md` **Amendment A3**。

本文件中所有来自外部检索的事实都标出来源；凡是**无法在下载后核验之前确认**的具体数字，
一律显式标注 **`[待下载后验证]`**（2026-09-27 已核验的条目标注“已实测”），不允许写成已确定结论。

---

## 1. Primary dataset：RefCOCO+（UNC image-level split）

### 1.1 数据集事实（已核实）

| 项目 | 结论 | 来源 |
|---|---|---|
| 数据归属 | RefCOCO / RefCOCO+ / RefCOCOg 由 UNC（Kazemzadeh et al. EMNLP 2014 采集协议；Yu et al. ECCV 2016 "Modeling Context in Referring Expressions" 定义 split 与 REF/REC 任务）发布，图像来自 **MS COCO** | `github.com/lichengunc/refer` README |
| 官方分发形式 | 每个数据集一个 zip：`https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip`（README 明确写出该 URL）。**2026-09-27 实测：官方直链 SSL 失败（refer issue #14 已知）→ 改用 Wayback 存档成功，见 §1.4 / R13** | 同上 |
| 服务器可用性 | UNC README 自述 "As the webserver is broken (sry about this), please check this Issue for all datasets downloading" → 官方直链可能失效，需走 GitHub issue 列出的镜像（社区镜像如 HuggingFace / OpenDataLab / TensorFlow Datasets `ref_coco`） | 同上 |
| 图像依赖 | README "Prepare Images"：把 `mscoco` 图像放入 `data/images`；RefCOCO、RefCOCO+、RefCOCOg 使用 COCO 图像 | 同上 |
| API / split 选择 | `refer.py` 支持 `dataset='refcoco+', splitBy='unc'`；同一 API 亦支持 `refcoco` 的 `unc`/`google`、`refcocog` 的 `google`/`umd`、`refclef` 的 `unc`/`berkeley` | 同上 |
| **UNC split 是 image-level** | TFDS 文档："`unc`" 和 "`umd`" splits **partition images** between train / validation / test（即图像不跨 split 出现）；而 Google-style split 不保证图像不相交 | `tensorflow.org/datasets/catalog/ref_coco`（经 OpenDataLab 镜像页复核） |
| RefCOCOg 的 google split 无 canonical test（val 常被论文当 test 报告，称 "val\*"） | 同上 | 同上 |
| **RefCOCO+ 禁用绝对位置词（非“无空间介词/纯外观”）** | TFDS 文档原文（其 “strictly appearance based” 属过强表述，见右注）：“RefCoco+ expressions are strictly appearance based descriptions, which they enforced by preventing raters from using location based descriptions”。**修正表述**：RefCOCO+ 标注采集禁止绝对位置词，降低了对简单绝对位置捷径的依赖，但关系型与上下文型表达仍可出现。> RefCOCO+ annotation collection prohibits absolute location words, reducing reliance on simple absolute-position shortcuts, while relational and contextual expressions can still occur. | 同上；第三方标注文档复核 |
| 规模（常引用值） | RefCOCO：142,210 expressions / 50,000 objects / 19,994 images；**RefCOCO+：141,565 expressions / 49,856 objects / 19,992 images**（**2026-09-27 实测确认：`refs(unc).p` 含 49,856 条 region 级 refs；`instances.json` 含 19,992 图像**） | Liao et al., "A Real-Time Cross-Modality Correlation Filtering Method for Referring Expression Comprehension", CVPR 2020（openaccess.thecvf.com） |
| RefCOCO+ 的 UNC split 各子集表达数 | **【2026-09-27 已实测（本项目口径）】**UNC split 内置在 refs 的 `split` 字段：**train 42,278 / val 3,805 / testA 1,975 / testB 1,798 refs**。原“常见表格值”（train 120,624 / val 10,758 / testA 5,726 / testB 4,889）保留为历史引用，不再作为项目口径 | 实测来源：`refs(unc).p`（2026-09-27）；原文献值 arXiv:2312.08007 Table 1 等 |
| testA / testB 语义 | 社区普遍描述为 RefCOCO/RefCOCO+ 的 testA 以 person 图像为主、testB 为非 person；本项目的 `testA`/`testB` 命名沿用官方文件中的 split 字段，**不自行重新定义** | OpenDataLab/TFDS 描述；**`[待下载后验证]`** |

**对本项目的影响**：
- 采用 **UNC image-level split** 是协议级决定（见 `research_protocol.md` §9），因为本项目缓存
  image/proposal feature，若同一图像跨 split 会造成 calibration/selection 污染。
- RefCOCO+ **标注采集禁止绝对位置词**（注意：这不是“无空间介词 / 纯外观描述”——关系型与
  上下文型表达仍可出现，见 §1.1 修正表述）这一倾向是一个有利条件：本项目 backbone 是 CLIP
  家族（对空间关系本就弱），主数据集降低对简单绝对位置捷径的依赖，可以减少 "失效究竟来自
  candidate-set shift 还是来自 CLIP 空间盲区" 的混淆。代价：RefCOCO+ 的表达更侧重外观，
  hardness 更多来自同类外观相似对象，这与本项目 same-category hard negatives 的定义天然契合
  （但也带来 §7 所述的 GT 信息泄漏风险）。

### 1.2 Annotation parser 方案对比（已核实）

| 方案 | 形式 | 优点 | 缺点 / 风险 | 选型 |
|---|---|---|---|---|
| **官方 `lichengunc/refer`（`refer.py` + `refs.json`）** | zip 内含 refs 元数据 + split 信息；API 按 `splitBy='unc'` 加载。**2026-09-27 实测更正：zip 内实际为 `refs(unc).p`（Python pickle，需 `encoding="latin1"`）而非 `refs.json`——“JSON 主格式”假设已证伪，实际以 pickle 为准** | 权威、与论文口径一致；UNC/Google/UMD split 同一 API 可选，便于**交叉核对 split 是否被误用** | 原始代码为 Python 2 时代风格；`make` 会编译 `_mask.c/_mask.so`（复制自 mscoco API），在 Windows 上是额外摩擦；包含 mask 功能但本项目只需 box | **采用（自写只读 parser：`refs(unc).p` pickle + `instances.json` 解析逻辑，不编译 mask 扩展）** |
| **Lake/ITSC "referring coco" `.mats`**（SCAN / RESCON / MATCHING 系列使用，如 `refcoco+ train_splitA.mat`） | MATLAB 导出的 `.mat`，需 `scipy.io.loadmat` | 大量开源 baseline 直接可用，字段扁平 | 属于**第三方二次预处理**，split 字段命名（splitA/splitB/splitC）与官方 `unc` split 的对应关系必须实证核对；混用有污染风险 | 仅作 **cross-check**（若可用），不作为主输入 |
| **JSON 转换版（`instances.json` + `refs.json` / `refcoco+.json`，常见于 ALBEF / LLaVA / PropVG 等仓库）** | COCO-style `instances.json` + 表达列表 | 与 pycocotools 生态兼容，字段清晰 | 不同仓库转换脚本字段不一致（有的把 `split` 写成字符串，有的写成 `.p` 索引文件）；来源可信度需逐仓库确认 | 备选（若官方 zip 不可达时的镜像路线） |
| `pycocotools` + 官方 COCO annotations | 只做 COCO GT 侧（IoU metadata、同类别查询） | 权威、必需 | 不含 referring expression 与 split | **必需辅件** |

**最终选择（冻结）**：以官方 `refcoco+.zip`（UNC split，`splitBy='unc'`）为唯一真值来源；
自行实现只读 parser 输出 `ReferringExample` dataclass（Task 4）。**【2026-09-27 实测更正】**
zip 内实际为 `refs(unc).p`（Python pickle，需 `encoding="latin1"`；49,856 条 region 级 refs，字段
`ref_id/image_id/split/sentences/ann_id/category_id/file_name`）+ `instances.json`（COCO 风格：
`ann_id → bbox(xywh)/category_id`，19,992 图像）——原“`refs.json` / JSON 主格式”假设**已证伪**，
本 parser 以 **pickle 为准**；COCO2014 `instances_trainval2014.json` 仍作 COCO GT 辅件。
**禁止**在同一份实验里混用 `.mats` 第三方 split 与官方
split 的样本成员定义；若为了对齐 baseline 需要引用 `.mats`，只能作为额外的
`split_provenance` 标签列，不作为 split 判定依据。

### 1.3 COCO 图像依赖与获取步骤（2026-09-27 实测更新）

- **【已实测修正】**RefCOCO+ 全部 **19,992 张**图像位于 COCO **train2014**（1500 张 val-split
  抽样图在 `instances_val2014` 中 **0 命中**）。本项目**只需 COCO train2014 图像**；val2014
  图像**非必需**（`instances_val2014.json` 仅用于交叉核验）。原“train2014 + val2014 双依赖”
  表述据此修正。常引用规模：train2014 = 82,783 张、val2014 = 40,504 张（COCO 官方页面）。
- **注册要求**：COCO 官网下载需注册（填邮箱 → 邮件确认 → 获得带令牌的下载链接），
  `images_train2014.zip` 与 `images_val2014.zip` 各约 20GB / 6GB 量级。
- **License 注意事项**：COCO **annotations** 为 CC BY 4.0；COCO **images** 源自 Flickr，
  每张图各自保留其原始 Creative Commons 许可（部分含 BY-NC / BY-ND 等限制），因此
  `cocoapi` issue #81 指出 "COCO images do not adhere to license.txt"（不存在统一图像 license）。
  → 本项目 README/data 文档必须写明：图像由使用者自行从 COCO 官方获取，遵守逐图 CC 许可；
  仓库**不得**再分发图像；若发布派生数据（feature cache）需说明其依赖 COCO 图像许可。
- 下载步骤说明（2026-09-27 实测执行记录）：
  1. 注册 cocodataset.org → 下载 `annotations_trainval2014.zip`（**实测官方 SCDN 正常：
     252,872,794 bytes**）、`images_train2014.zip` 至 `data/mscoco/`；**val2014 图像非必需**；
  2. 下载 `refcoco+.zip` —— **实测官方 UNC 直链 SSL 失败（refer issue #14 已知）**，改用 Wayback
     存档 `https://web.archive.org/web/20220413011656id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip`
     （**45,613,210 bytes**）成功，解压到 `data/raw/refcoco+/refcoco+/`；
  3. audit 图像按 manifest 逐张从 `images.cocodataset.org` 下载（**1500 张，首轮 6 张失败、
     重试后 1500/1500 成功**）；
  4. 运行 `scripts/prepare_refcoco.py` 做完整性校验（见 §6 artifact）；
  5. **不下载** RefCOCO / RefCOCOg / gRefCOCO / Ref-L4（Phase 0 禁止多数据集全矩阵）。

### 1.4 实测更新（2026-09-27，下载与解析后核验）

> 本节汇总下载完成后的实测事实；与 §1.1–§1.3 原文冲突处，以本节 + 对应行内标注为准。

1. **数据格式**：`data/raw/refcoco+/refcoco+/` 下为 `refs(unc).p`（Python pickle，需
   `encoding="latin1"`；**49,856 条 region 级 refs**，字段 `ref_id/image_id/split/sentences/
   ann_id/category_id/file_name`）+ `instances.json`（COCO 风格：`ann_id → bbox(xywh)/category_id`，
   **19,992 图像**）。**“refs.json / JSON 主格式”假设已证伪**——实际以 **pickle 为准**（见 §1.2）。
2. **Split 计数（实测，项目口径）**：UNC split 内置在 refs 的 `split` 字段：
   **train 42,278 / val 3,805 / testA 1,975 / testB 1,798 refs**（见 §1.1）。
3. **图像来源（实测）**：全部 **19,992** 张 RefCOCO+ 图像位于 COCO **train2014**；1500 张
   val-split 抽样图在 `instances_val2014` 中 **0 命中**。本项目只需 **train2014** 图像；
   val2014 图像非必需（`instances_val2014.json` 仅用于交叉核验）——原“train2014+val2014
   双依赖”表述已修正（见 §1.3）。
4. **下载途径（实测）**：
   - RefCOCO+：官方 UNC 直链 **SSL 失败**（refer issue #14 已知）→ Wayback 存档
     `https://web.archive.org/web/20220413011656id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip`
     成功（45,613,210 bytes）；
   - COCO annotations：官方 SCDN 正常（`annotations_trainval2014.zip`，252,872,794 bytes）；
   - audit 图像：按 manifest 逐张从 `images.cocodataset.org` 下载（1500 张；首轮 6 张失败，
     重试后 1500/1500 成功）。
5. **Audit subset（冻结）**：`data/audit_subset.csv`——1500 张（train 池 1000 + val_select 500；
   val_select = val 池按 seed+1 permutation 前半；抽样 seed 20260927）；冻结复现：
   `scripts/build_audit_subset.py` 已实现字节级一致再生成（详见 §10.1）。

## 2. External stress test：FineCops-Ref（**已实测核实 2026-09-29**）

- Liu, Yang, Li, Wang. "FineCops-Ref: A new Dataset and Task for Fine-Grained Compositional
  Referring Expression Comprehension." **EMNLP 2024 main**（arXiv:2409.14750，v2 2025-01-11）。
- **官方分发渠道（实测）**：figshare article **26048050**（`api.figshare.com/v2/articles/26048050`），
  license 字段 **CC BY 4.0**（`creativecommons.org/licenses/by/4.0`，非推断）。该 article 共 15 个文件：
  test（`test_expression_pos.json` 3,075,530 B / `test_expression_all.json` 12,127,597 B + 两个
  `*_coco_format.json`）、train/val 标注（`expression_all_train_set.json` 74,999,544 B、
  `expression_all_val_set.json` 8,429,891 B 及各 coco_format / pos_only 版本）、`neg_images.tgz`
  567,178,672 B。本地只下载 **test 的 4 个 json** + GQA `sceneGraphs.zip`（44,824,862 B，含
  `train_sceneGraphs.json` / `val_sceneGraphs.json` 两个 entry），逐文件 sha256 见
  `data/raw/finecops/dataset_card.json`。
- **实测 counts（positive test，官方数字已被本地重核）**：**9,605 expressions / 4,313 unique images**；
  level 1/2/3 = **5,730 / 3,404 / 471**（59.66% / 35.44% / 4.90%）；
  tuple_type = 0_hop 2,333 / 1_hop 2,146 / 2_hop 2,555 / and 1,639 / same_attr 705 /
  **same_attr_two_hop 227**（< 300，按指令 §22 不可单独报告）。
  negative：**9,814 negative text + 8,507 negative image**（`test_expression_all.json` 合计
  27,926 行；negative_type = object 8,122 / attribute 3,569 / order 2,029 / relation 1,891 /
  flip 1,555 / swap_attr 1,155）。以上均为**解析实测值**，见
  `results/phase1e_finecops/metadata_audit.json`。
- 论文自述结论之一：specialist models 与 MLLMs 在该数据集上仍有明显差距（我们只引用其
  数据集动机，不引用性能结论）。
- **对本项目的意义与限制（原 `[待下载后验证]` 三项已关闭）**：
  - 它承担 prompt 中 "relation confusion / compositional difficulty / negative selection" 的
    stress test 角色，因此本项目**不**人工生成 relation negatives。
  - ~~是否为 evaluation-only（无 train split）~~ → **实测：不是 evaluation-only**，figshare 上确有
    train/val 标注文件。但本项目按指令 §3 **只用官方 test split**，train/val 标注**刻意不下载**
    （记录于 `dataset_card.json` 的 `deliberately_not_fetched`），因此不存在任何用 FineCops
    train/val 训练/调参/校准的可能路径；FineCops 角色仍是 frozen external evaluation。
  - **来源纠正（2026-10-04 V3）：GQA/VG 身份不等于非 COCO 图像来源。**
    标注的 `file_name` 为 `<gqa_image_id>.jpg`，4,313 个 image id **100% 可在 GQA
    `val_sceneGraphs.json` 中解析**（`n_images_without_graph = 0`）；这只确认 GQA/VG 标注与命名空间。
    VG 官方图像元数据还包含 `coco_id` 与 `flickr_id`，必须核查跨来源重叠，不能据此直接
    宣称“跨视觉域”或与现有图像独立。[VG 官方数据定义](https://visualgenome.org/api/v0/api_readme)。
    冻结 RPN 在该图像集合上重新生成 bank，并单独报告 proposal recall；与 RefCOCO+ 的差值
    是这些数据集合上的实测差异，不能单独识别视觉域偏移的因果贡献。历史 EXTERNAL STOP 不变；
    新划分及独立性核验采用[独立 V3 协议](../reviews/v3_protocol.md)。
    来源计数与身份映射由 [V3 生成式审计工件](../results/v3_final_validation/exposure/source_identity_counts.json)
    留档；映射结果不是确认集性能结果，也不代表暴露审计已经通过。
  - 图像获取：不下载 GQA **21,817,965,542 B**（148,855 members）`images.zip` 整包；该 host 支持
    HTTP `Range`，`ccg/external/gqa_images.py` 只读一次 zip 尾部中心目录，再按成员定位抽取。
    实测两个约束：(a) host **按 IP 限流**（额外并行流立即返回 **503**），因此并发固定 3 +
    指数退避 + 幂等 skip-existing；(b) 旧 `zipfile` 路径每图平均读回 **3.5 MB**（≈23× 读放大），
    改为「每线程一条 keep-alive HTTPS 连接 + 每图单个合并 Range + raw-deflate（`zlib wbits=-15`）
    解码 + CRC32/长度校验」后 **wire == payload**（实测 642 图 = 642 请求，
    85,047,571 B out / 84,513,451 B in，比值 1.006）。
  - 类别信息：COCO-format 标注的 `categories` 只有 1 个占位条目（无真实类别），因此外部
    same-category 的等价量只能是 **GQA scene-graph object `name` 的精确同名**；禁止自造
    COCO category mapping（见 `research_protocol.md` Amendment A9.4）。
  - negative images 来自 **AI 编辑/生成**；本轮只 download/parse/count，不建模、不训练 NONE
    head（指令 §2/§23/§24）。

- **F2/F3 实测结果（2026-09-29，冻结 N=64 RPN + 1,000 图 / 2,235 条 positive test 审计）**
  数据源：`results/phase1e_finecops/{rpn_audit_summary.json,recall_by_target_size.csv,external_branch_decision.json}`。

  | 量 | FineCops-Ref（GQA 域） | RefCOCO+（COCO 域，同一冻结 RPN） |
  |---|---|---|
  | target proposal recall@0.5 | **0.7579** [0.7398, 0.7752] | 0.9859 |
  | target proposal recall@0.7 | **0.6376** [0.6174, 0.6573] | 0.8998 |
  | natural omission@0.5 | **0.2421** | 0.0141 |
  | same-category K5 availability | **0.1861**（GQA 精确同名口径） | 0.9003（COCO GT 类别口径） |
  | GT/对象口径 recall@0.5 | 0.4313（全部 VG 对象） | 0.8085 |

  其它实测：bank 恒 64（min 64）、mean valid distractors 59.73、invalid crop 1.54%、
  冗余（pair IoU>0.7）0.12%、几何 0 例外（RELEASED BOXES IN JPEG PIXEL SPACE）。
  recall 随目标尺寸单调上升（<32px 0.227 → ≥256px 0.903，49.1% 目标边长 <128px），
  但最大桶仍不及 RefCOCO+ 总体水平 → 尺寸只解释部分缺口，余下为全局感知/域差距；
  且 recall 缺口与官方 level 无关（L1 0.776 / L2 0.729 / L3 0.770）。
  **判定：EXTERNAL STOP（§6 recall < 0.80）+ regime `level_primary_only`**；详见
  `docs/research_protocol.md` **A9.13** 与 `docs/experiment_log.md`
  条目 `p1e-finecops-external-feasibility-audit-20260929-01`。

## 2B. External confirmation 候选：RefCOCOg — UMD split（**已实测核实 2026-09-29**）

A9 EXTERNAL STOP 后的替代候选。目的不是找一个“更不同的视觉域”，而是在**保持 COCO 图像域 /
COCO object ontology / 冻结 proposal pipeline 可比性**的前提下引入**语言与标注协议 shift**。
因此它的合法定位是 **cross-dataset external validation under a shared COCO visual domain**，
**不是** cross-visual-domain generalization（指令 §24）。

- **来源与指纹**：与 RefCOCO+ 同一分发点 `bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip`；
  官方直链已失效（SSL failure，与 `refcoco+.zip` 上已经历过的同一故障），因此改取 Internet Archive
  对**同一官方 URL** 的快照 `20220413012904id_`（`id_` 修饰符返回未经改写的原始字节；
  zip 56,712,951 B，sha256 `3d1f7e5b2ff22059…`；CDX `response_length` 56,715,268 含 HTTP 头部，
  故只作下限校验，完整性以响应自身的 Content-Length 为准）。解压后逐文件记录：
  `refs(umd).p` 33,853,676 B / sha256_16 `0331c7533537b67c`，`refs(google).p` 33,853,786 B /
  `e4d8320dfd15fc21`（**仅用于证明未误用 Google object split**），`instances.json` 124,416,571 B /
  `96c89b426c657f2f`。见 `data/raw/refcocog/dataset_card.json`。
- **G0 实测（不引用网络文献数字）**：image-level split 确认（`refer.py` 的 `splitBy='umd'` 口径；
  本文只记录实测文件事实，该 split 的原始论文出处列在 `docs/literature_notes.md` §8 待核对）。`image_level_split: true`，
  三个 split 间 **0** 图重叠。test **9,602 expr / 5,023 refs / 5,023 objects / 2,600 imgs / 76 类**；
  train 80,512 / 42,226 / 42,224 / 21,899；val 4,896 / 2,573 / 2,573 / 1,300；合计 95,010 / 49,822 / 49,820 / 25,799。
  test 每 ref 句数：1 句 452 / 2 句 4,563 / 3 句 8。
- **目标框来源（必须审计而非假定）**：UMD refs **不携带 box**，target 由 `ann_id` join `instances.json`。
  archive 侧 vs 官方 `instances_train2014.json` 逐 ref 比较：n=5,023、IoU **min=mean=max=1.0**、
  n_identical **5,023**、类别不一致 **0**、未匹配 **0** → **ARCHIVE AND OFFICIAL BOXES IDENTICAL**。
  GT 对象集（same-category 候选池）同 RefCOCO+ 侧 `run_proposal_audit.py` 先例，取官方 train2014 注解。
- **G1 image overlap（本数据集最大的风险）**：RefCOCOg UMD test 与 RefCOCO+ 共用 COCO 图像，
  重叠 train **1,257** / val_select 65 / val_calib 58 / **development 1,380** / testA 47 / testB 71 /
  **ALL RefCOCO+ 1,498**（共 2,600）。→ RefCOCOg test **本身不是** external。
- **G2 image-disjoint 子集**：
  `rg_external_strict`（减 ALL RefCOCO+，含已被反复查看的 testA/testB）= **2,909 expr / 1,102 imgs / 1,512 refs**；
  `rg_external_devdisjoint`（减 train ∪ val_select ∪ val_calib）= **3,448 expr / 1,220 imgs / 1,796 refs**
  （cumulative，dev-only extra 539 行 / 118 图）。size gate：**STRICT PRIMARY**（≥ 1500 expr / ≥ 500 imgs）。
  图像获取：strict 所需 1,102 张全部下载（178,258,375 B / 126.6 s / failed 0），dev-only 118 张已在盘且
  其 proposals **直接复用冻结 RefCOCO+ bank**（同域、同模型、同 N，不是重提）。
- **G3 冻结 proposal 审计（参数一字未改）**：recall@0.5 **0.972155**、recall@0.7 **0.884840**、
  natural omission **0.027845**、K5 = K10 random availability **0.972155**、gt_object_recall@0.5 0.913232
  → **EXTERNAL GO**（同域 gate：recall ≥ 0.90 ∧ K5 availability ≥ 0.95）。
  RefCOCO+ 冻结参照（读盘）0.9858742 → Δ **−0.013719**（对比 FineCops 的 −0.2279）。
- **G4 same-category（A8 规则逐字复用）**：≥1 **0.913029** / ≥2 0.828463 / ≥4（K5）**0.649708**
  [0.632184, 0.666836] / ≥9 0.270540；hard cohort **1,890 expr / 755 imgs / 978 refs / 69 类**（功效门通过）。
  **不要求**复制 RefCOCO+ 的 0.9003（Δ −0.250612 作为 external shift 记录，第一版不 reweight）。
- **G4 分布 shift（external value 的来源）**：tokens mean 3.5348 → 8.3875、vocab 2,942 → 4,038、
  spatial rate 0.4268 → 0.7867、absolute-position rate 0.0330 → 0.1858；
  hard-cohort 类别 entropy 2.5329 → 2.9210、person fraction 0.5224 → 0.3852。
- **G5 判定：Branch A — CLEAN EXTERNAL**（`next_stage_allowed = true`）。本轮**未前向任何模型**：
  无 CLIP extraction / B3 / Stats / E1b / bootstrap，RefCOCOg train 完全未用（0 个新训练参数）。
  详见 `docs/research_protocol.md` **A10.13** 与 `docs/experiment_log.md` 条目
  `p1e-refcocog-external-feasibility-audit-20260929-01`。

## 3. 后续扩展数据集（仅记录，不进入 Phase 0）

- **gRefCOCO**：来自 **GRES: Generalized Referring Expression Segmentation, CVPR 2023**
  （Liu et al.；代码与数据 `github.com/henghuiding/grefcoco`，项目页 `henghuiding.com/GRES`），
  支持 zero-target（无匹配目标）与 multi-target 表达，并给出 GREC/GRES 评测协议。
  → **更正记录**：早期资料常把 "REF / referent-absent" 归给更早的工作；本项目文档以 GRES/gRefCOCO
  （CVPR 2023）为出处，具体 zero-target 样本数 **`[待下载后验证]`**。
- **Ref-L4**：面向 instruction-tuned LVLM 的 grounding/推理长度标注基准（检索未能定位到权威
  页面，作者与年份 **需查证**）。仅作为后续 extension 候选。
- 第一阶段禁止把上述数据集铺成多 dataset 全矩阵。

---

## 4. Proposal generator 选型（已核实 + 决策）

需求：单个**冻结的 COCO-pretrained detector / RPN**，对每张图产生 **N ≈ 64** 个 proposals，
保存 box + objectness + crop + CLIP embedding + 与 GT 的 IoU metadata，且必须在 8GB VRAM 上
可推理。

| 候选 | 权重规模 | 8GB 推理 | 依赖摩擦 | proposal 可控性 | 评价 |
|---|---|---|---|---|---|
| **torchvision `fasterrcnn_resnet50_fpn`（`FasterRCNN_ResNet50_FPN_Weights.COCO_V1`）** | 官方文档标注 **File size 159.7 MB** | 可以（FP16/bf16 或 FP32，单图） | **最低**：`torchvision` 已在 `pyproject.toml` 依赖内 | RPN 输出经 NMS 后取 top-K，可按 objectness 截断到 64 | **选用** |
| torchvision `fasterrcnn_resnet50_fpn_v2`（COCO_V1 weights） | 约同量级（**具体大小 `[待下载后验证]`**） | 可以 | 极低 | 同上，AR 通常更高 | 备选；换用需 amendment |
| Detectron2 `Mask R-CNN R50-FPN, 3x`（MODEL_ZOO） | 约 170–200 MB（**`[待下载后验证]`**） | 可以（需 `fp16` 或较小 batch） | **高**：detectron2 在 Windows 需编译/特定 wheel，且与现有 conda 环境冲突风险 | RPN `POST_NMS_TOP_K_*` 可配；官方多输出 instance mask（本项目不需要） | 不作为 Phase 0 主线 |
| MMDetection | 类似 | 可以 | 高（版本耦合、编译） | 可配 | 不采用 |
| Grounding DINO / 大型 detector | > 8GB 风险、需要 license 确认 | 紧张 | 高 | — | **禁止**（prompt 第二十九节：不做 Grounding DINO fine-tuning；且体积与目标不符） |

**最终选择与理由（冻结）**：**torchvision Faster R-CNN R50-FPN（COCO 预训练权重，159.7 MB）**，
完全冻结、仅推理、query-independent。理由：
1. 权重体积与官方文档明确（约 160 MB，符合 prompt 所述 160–200 MB 量级）；
2. 依赖已在项目内，无 Windows 安装摩擦，符合 "适合单张 RTX 4060 Laptop / 8GB" 的约束；
3. RPN 的 objectness 分数可直接作为 candidate 的 `g_i` 可选特征，语义清晰；
4. COCO-pretrained ⇒ 与 RefCOCO+ 的 COCO 图像域和 COCO 类别体系一致，使
   "same-category hard negatives"（依赖 COCO GT 类别）可以被定义和计算。

**实现要点（须在代码中显式固定，避免隐式漂移）**：
- 保留 torchvision 官方推理预处理（短边 resize 800、长边上限 1333、ImageNet 归一化）；
  该预处理同时决定 proposal 几何，**不得**与 CLIP 侧的 crop 缩放混用；
- 取 RPN 输出（已 NMS）按 objectness 降序截断至 **N = 64**；若某图不足 64，则保留实际数量
  并在 audit 中统计（不得用 padding 伪造 proposal）；
- IoU 阈值与 target assignment 严格按 `research_protocol.md` §5；
- 保存 `post_nms_top_n`、`nms_thresh`、`score_thresh`（若用于过滤需明示）、权重 URL 与
  sha256（下载后写入 `data/MANIFEST.json`）。

## 5. Proposal recall 的文献参考值与本项目立场

- 文献普遍以 **AR@k（average recall at k proposals）** 作为 proposal 质量指标，Hosang et al.
  2016 "What Makes for Effective Detection Proposals?"（PAMI）系统比较了 Selective Search /
  EdgeBox / RPN 在不同 proposal 预算下的 AR；FPN（Lin et al. CVPR 2017）在 1000 proposals 预算下
  报告 box/segment AR（其表格明确 "always for 1000 proposals"）。
- 检索到的共识是：现代 RPN 在 **1000** proposals 预算下 recall 很高，而在 **~100–300** proposals
  预算下 AR@0.5 通常仍在 0.9 量级；但**本项目用的是 N=64 且 GT 是 RefCOCO+ 的 referring target
  （含小目标、非 COCO 主类别实例）**，因此文献值只能当作 "量级参考"，不能当作预期结论。
  具体数值（例如 FPN 论文 AR@100 的精确数字）**需查证**。
- 因此协议规定：**Phase 0 的第一件产出物是 proposal quality audit**
  （recall@64、recall@32、natural miss rate、IoU distribution、candidate availability
  distribution），并且必须在任何 decision-model 结果之前完成、写入日志。**（2026-09-27 已完成：结果见
  `docs/experiment_log.md` 条目 `audit-proposal-001`；protocol 层 outcome 见 `docs/research_protocol.md`
  Amendment A3。）**
- 若 audit 显示 recall 过低（本项目预注册的操作性阈值，写在此处以免事后随意改）：
  - `recall@64 < 0.85`：必须报告为数据侧限制，并考虑以 **amendment** 方式更换 generator
    （保留原 generator 全部审计数字），同时把 natural-omission split 的统计单独报告；
    （**2026-09-27 实测：RefCOCO+ referring target 级 N=64 recall@IoU0.5 = 0.9858742004264393，未触发本处置流程**）
  - 任何情况下**不得**通过提高 IoU 阈值、或从 RefCOCO+ 中静默删除 target-missing 样本来 "修复" recall。
  - `research_protocol.md` §5/§6 的 target 定义、嵌套性质、K∈{5,10,20,50} 网格不因 generator 更换而改变。

## 6. 规模、存储与显存预算

| 项 | 估算 | 依据 |
|---|---|---|
| RefCOCO+ 使用图像数 | **19,992**（**2026-09-27 实测确认**，全部位于 COCO train2014） | §1.4 实测更新；原文献值 |
| Proposal crops 总数 | ~20k × 64 ≈ **1.28M** | prompt §二十四 |
| Region embeddings 体积 | 1.28M × 512 × 2 bytes (FP16) ≈ **1.31 GB** | prompt §二十四 |
| Global image features | ~20k × 512 × 2B ≈ 20 MB | 估算 |
| Proposal boxes / objectness / IoU metadata | 数十 MB（FP32 boxes + FP16 scores + int32 assignment + IoU 矩阵按需） | 估算 |
| Query embeddings | 表达数 ~141.5k × 512 × 2B ≈ **~145 MB**（按唯一文本去重会更小）→ "数百 MB" 量级 | 文献值 + 估算 |
| **总 cache 目标** | **< 3 GB**（不含原始图像与 COCO 标注 zip） | prompt §二十四 |
| Candidate sets（indices） | ~141.5k refs × (4 K 值 × 2 hardness × presence 变体) × int16 indices ≈ 数十 MB | 估算 |
| 存储格式 | HDF5（单一 `.h5`，按 image_id random access）；备选 `.npz` + 索引 | prompt §二十四（选一种简单、支持 random access 的实现，避免几百万小文件） |
| Feature extraction 显存 | FP16/autocast，**batch 64 起步**，OOM 降 32；detector 推理单独 batch（8–16） | prompt §二十四 |
| 决策模型训练显存 | 在缓存 embedding 上运行（MLP <1M、DeepSets <0.5M），预计 <2 GB | 估算 |
| 不做的事 | 不为 batch size 做 gradient tricks（无训练）；不缓存 per-expression 视觉特征 | prompt §七/§二十四 |

**小文件禁令**：禁止为每个 crop 写一个 `.jpg`。若后续需要可视化，只在 audit 阶段导出少量样例。

## 7. 风险清单核查（Task 3 强制项）

| # | 风险 | 现状判断 | 缓解 / 处置 |
|---|---|---|---|
| R1 | **UNC / Google split 混用污染** | 真实存在。同一 `refer.py` 可加载两种 split，极易在换 baseline 代码时被替换；Google-style 为 object-level，图像跨 split | 只在 `splits.py` 一处读取 split；把 `splitBy='unc'` 写入每个 artifact 的 provenance 字段；新增单测断言 "train/val/test 图像集合两两不相交"；任何引用 `.mats` 第三方 split 的对照都单独标注，不进入主结论 |
| R2 | **RefCOCO+ annotation leakage** | 存在多个层面：(a) 同一 image 的多条表达天然共享图像与 GT；(b) 若 split 用 object-level 则同图跨 split；(c) `val_select`/`val_calib` 若按表达随机切分会继承同图相关性 | image-level split（R1）；`val_select`/`val_calib` 也**按 image** 划分；bootstrap 一律 **image-level resampling**（§12 协议）；同一表达文本在多个 split 重复出现的比例需 `[待下载后验证]` |
| R3 | **Proposal recall 过低** | **已实测（2026-09-27，audit subset 1500 图）**：RefCOCO+ referring target 级（3752 expressions）N=64 recall@0.5 = 0.9858742004264393 ≥ 0.85 → 不触发 §5 处置流程；GT-object 级（15349 个 iscrowd=0 GT objects）recall@0.5 = 0.8085217277998566（N=64）/ 0.8801876343735748（N=128）（参考口径） | 已由 §5 的 audit 执行完毕：`results/proposal_audit/`（summary.json + 5 CSV + 5 figures）；完整数字含 recall@0.7 与 CI，均未静默过滤；见 `docs/experiment_log.md` `audit-proposal-001` |
| R4 | **Natural omission 样本量过少** | **已确认发生（2026-09-27）**：expression 级 P(max IoU<0.5) = 53/3752 = 0.014125799573560768（N=64）/ 22/3752 = 0.005863539445628998（N=128） | 预先声明：natural_omission 为**描述性/小样本**结果，CI 必报；若样本量 < 足以支撑 3-seed 比较，则明确写 "该结论不显著"；主要 NONE 能力评估放在 synthetic omission 上，并显式声明 synthetic 的局限。**追加（2026-09-27）：RQ4 的 natural omission split 统计功效有限，需在 Phase 2 前做功效评估（不影响 Gate；见 Amendment A3）** |
| R5 | **K=50 时 proposal 数量不足** | **已实测（2026-09-27）**：valid distractors = N − \|{target}∪to_remove\| ≥ K−1 口径下，N=64：K=5/10/20 = 1.0（3752/3752）；K=50 = 3719/3752 = 0.9912046908315565 [0.9876741623023597, 0.9937303782938145]；N=128：全部 1.0 | 先报告 **candidate availability distribution**（每图可用候选数直方图、按 split/类别分层），再决定：仅在候选数 ≥ K 的样本上评估该 K，并**同时报告被排除样本的数量与特征**（不得静默过滤）；必要时把主网格的最大 K 降为 20，并以 amendment 记录。**处置结果（2026-09-27）：保留 K=50；33/3752 = 0.88% 无法支持 K=50 的 expression 必须在 K=50 cells 显式报告排除计数，禁止静默过滤（见 Amendment A3 §A3.2）** |
| R6 | **Candidate set 中存在多个正确 proposal** | 若不处理则 accuracy/ECE 不可解释 | 协议 §5 已冻结：target = 唯一 max-IoU proposal，其他 IoU ≥ 0.5 的 proposal 一律移除 |
| R7 | **Hard negative 使用 GT 信息导致不公平** | **需要注意**：same-category hard negatives 依赖 COCO GT 类别与 IoU，属于 evaluation 时不可得的信息 | 处理：该 regime 与 CLIP-hard 一样按 **diagnostic evaluation regime** 报告（真实 detector 无法使用 GT）；同时保证 **target 的定义只用一次 assignment**，hard-negative 选择不得引入任何 "该候选是否为答案" 的信息；在结果表格中每列都带 regime 标签 |
| R8 | **CLIP-hard negatives 被误读为部署分布** | 存在 | 协议 §6 已声明 diagnostic/adversarial regime；文档与结果命名统一使用 `clip_hard_diagnostic` |
| R9 | **Calibration split 被用于 model selection** | 常见错误 | `val_select` 与 `val_calib` 按 image 分离并在代码中用不同 loader；temperature/threshold 拟合函数只接受 `val_calib`；新增检查：训练脚本若检测到 test/calib 混用则抛异常 |
| R10 | **softmax 的 K 依赖被误当成 calibration failure** | 存在（概念风险） | 协议 §2 RQ1 已写入；主指标使用 top-label ECE / correctness NLL / Brier 与 risk–coverage，并同时报告 ΔAcc 与 ΔCalibration 两条线 |
| R11 | **Crop context 丢失 / CLIP 空间弱** | 存在：candidate 用 crop embedding，丢失全局上下文；CLIP 对空间与小目标弱 | Phase 0 冻结为 **crop-only embedding**；不做 context-expansion、不加 image-level 特征融合（属于 protocol 变更，需 amendment）；把 CLIP 空间弱点作为**解释性限制**写入结论段，并允许 FineCops-Ref 作为外部压力测试来暴露该限制 |
| R12 | **同图表达在 evaluation 中的统计相关性** | 存在（RefCOCO+ 每对象平均 ~2.8 条表达） | 与 R2 相同：image-level bootstrap + 报告 per-image 聚合的敏感性分析（次要、不改判据） |
| R13 | **官方直链失效 / 镜像可信度** | **已确认（2026-09-27）**：官方 UNC 直链 SSL 失败（refer issue #14 已知）；改用 Wayback 存档成功（45,613,210 bytes，URL 见 §1.4）；COCO annotations 官方 SCDN 正常（252,872,794 bytes） | 优先官方链接；若不可达，选用可校验镜像并把文件 sha256 写入 `data/MANIFEST.json`；禁止使用无法追溯来源的第三方打包（实际执行：Wayback 存档，可追溯官方 URL；文件字节数已记录于 §1.4） |
| R14 | **License 合规** | COCO 图像逐图 CC（含 NC/ND 变体）；RefCOCO 需遵守其引用要求；FineCops-Ref license **已核实（2026-09-29）**：figshare article 26048050 的 `license` 字段 = **CC BY 4.0**（API 实测，非网页推断）；底层 GQA/Visual Genome 图像另受其自身条款约束（GQA 为 Scene Graph API 公开数据集） | 仓库不分发图像/原始标注；README 与 `data/README.md` 写明注册要求与 attribution（Yu et al. ECCV 2016、Kazemzadeh et al. EMNLP 2014、COCO、FineCops-Ref/EMNLP 2024） |

## 8. 本文件的最终决定摘要

1. **Primary data**：RefCOCO+，**UNC image-level split**，官方 `refcoco+.zip`（**实测经 Wayback 存档成功获取**）+ 自写只读
   parser（**实测以 `refs(unc).p` pickle 为准；“refs.json / JSON 主格式”假设已证伪**，见 §1.2/§1.4）；`.mats` 第三方版本仅作 cross-check，不作为 split 真值。
2. **Images**：COCO2014 **`train2014`**（**已实测确认**：全部 19,992 张 RefCOCO+ 图像位于 train2014；val2014 图像**非必需**，`instances_val2014.json` 仅用于交叉核验；注册下载，逐图 CC license，仓库不分发）。
3. **Proposal generator**：**冻结的 torchvision Faster R-CNN R50-FPN（COCO_V1，159.7 MB）**，
   **N = 64（2026-09-27 按 A2.4 预注册规则选定：N=64 支持 K=50 比例 0.9912046908315565 ≥ 90%；见 `research_protocol.md` Amendment A3）**，取 RPN NMS 后按 objectness top-64；IoU ≥ 0.5 的等价 proposal 移除，target 为唯一
   max-IoU proposal。
4. **External stress test**：FineCops-Ref（EMNLP 2024；**实测**：figshare 26048050 / CC BY 4.0 /
   test 9,605 pos + 9,814 neg text + 8,507 neg image / GQA-Visual Genome 标注来源，含 COCO 图像关联 /
   历史阶段存在 train-val 标注但刻意不下载；V3 另行审计新划分，详见 §2）→ **F0–F4 可行性审计已完成并判为 EXTERNAL STOP**：
   同一冻结 N=64 RPN 在 GQA 上 target recall@0.5 仅 **0.7579**（< 0.80 停止线），
   same-category K5 可用性仅 **0.1861** → F5–F10 未获授权（见 **A9.13**）。
5. **预算**：feature cache 约 1.3 GB region embeddings (FP16) + ~0.15 GB query embeddings，
   总 **< 3 GB**；提取 FP16 batch 64 起步、OOM 降 32。
6. **Gate 前置条件**：proposal quality audit（recall、natural miss rate、IoU 分布、
   candidate availability 分布）必须先于任何 decision-model 结果完成并入库。**（已完成 2026-09-27：1500 图；
   见 `docs/experiment_log.md` `audit-proposal-001` 与 `docs/research_protocol.md` Amendment A3）**

## 9. 产出物（`scripts/prepare_refcoco.py` / `audit_proposals.py`）

| Artifact | 内容 |
|---|---|
| `data/MANIFEST.json` | 每个下载文件的来源 URL、大小、sha256、license 备注、下载日期 |
| `data/refcoco+/parsed/referring_examples.jsonl` | `ref_id, image_id, text, gt_box, gt_object_id, split, split_provenance` |
| `data/refcoco+/splits/{train,val_select,val_calib,testA,testB}_image_ids.json` | image-level split 名单（供污染单测） |
| `cache/proposal_bank.h5` | `boxes[N,4]`, `objectness[N]`, `gt_iou[N]`, `gt_assignment`, `image_id` |
| `results/phase0/proposal_audit.json` | recall@32、recall@64、natural miss rate、IoU 直方图、候选可用数分布、按类别/尺寸分层 |
| `results/phase0/proposal_audit.md` | 人读版本 + 风险清单核查结论 |

## 10. Proposal-system audit design（本轮预注册设计，不含结果）

> 本小节只写 **audit 设计**（预注册），不写任何结果；具体数值待 `scripts/audit_proposals.py`
> 实跑后按 `docs/experiment_log.md` 规范另写入日志。预注册依据：`docs/research_protocol.md`
> **Amendment A2**（class-agnostic RPN proposal bank 与 N-selection 工程判据）。
> **2026-09-27 注**：本轮 audit 已运行完成；结果数字不复制到本小节，见 `docs/experiment_log.md`
> 条目 `audit-proposal-001` 与 `docs/research_protocol.md` Amendment A3。

### 10.1 Audit 样本与隔离

- **audit subset**：从 `train` / `val_select` 的 **image 级**确定性抽样 **1,000–2,000 张**；
  固定随机种子（写入 artifact 的 `seed` 字段）；抽样单位是 image，不是 expression。
  **【2026-09-27 实测更新（冻结）】**audit subset 冻结为 `data/audit_subset.csv`：**1500 张**
  （train 池 **1000** + val_select **500**；val_select = val 池按 seed+1 permutation 前半；
  抽样 seed **20260927**）；冻结复现：`scripts/build_audit_subset.py` 已实现**字节级一致**再生成。
- **硬约束**：本 audit **不查看 `testA` / `testB`**（完全隔离，与 `research_protocol.md` §9 一致）；
  `val_calib` 也不用于 proposal 工程选型（避免与 calibration 用途重叠）。
- **对比**：对同一 subset 同时报告 **N=64 与 N=128** 两档，以支撑 A2.4 的 N-selection 预注册判据
  （`>=90%` target-present 例子能构造 K=50）。

### 10.2 报告指标清单（N=64 / N=128 两档均报）

- **proposal recall@IoU**：`recall@0.5`、`recall@0.7`（以 RefCOCO+ referring target GT box 为基准）；
- **max-IoU 分布**：`max IoU` 的 median / P25 / P75；
- **natural miss**：`P(max IoU < 0.5)`（即 natural proposal miss 率，对应 §5 / R3）；
- **有效 distractor 可用性**（**在移除 target-equivalent（与 GT box IoU≥0.5）proposal 之后**）：
  `P(valid distractors >= 4 / 9 / 19 / 49)`，分别对应 `K=5 / 10 / 20 / 50`（对应 R5：K=50 候选不足风险）；
- **冗余度**：proposal 间 `IoU>0.7` 与 `IoU>0.9` 的比例（近重复密度；用于 A2.5 判断是否引入更严格 duplicate suppression）；
- **remaining candidate count distribution**：每个 (image, K) 去除 target-equivalent 后的剩余候选数直方图（A2.5 明确：此结果未出之前不得引入新的 IoU threshold）；
- **same-category distractor availability**：同类 distractor 可用性 `>= 1 / 2 / 4 / 9`（GT-assisted，仅诊断用途，见 A1.2 的 GT-assisted diagnostic 定位）。

### 10.3 Artifact 清单

```text
results/proposal_audit/summary.json                       本次 audit 总览（含 seed/subset 大小/N 档位/provenance）
results/proposal_audit/recall_by_N.csv                    recall@0.5 / @0.7 按 N=64/128
results/proposal_audit/candidate_availability_by_N.csv     P(valid distractors>=4/9/19/49) 与 remaining count 分布
results/proposal_audit/same_category_availability.csv      same-category distractor availability >=1/2/4/9
results/proposal_audit/natural_omission.csv               P(max IoU<0.5) 与 natural miss 样本列表
results/proposal_audit/proposal_iou_statistics.csv        max-IoU median/P25/P75、冗余 IoU>0.7/0.9 比例
+ 5 diagnostic figures（recall-vs-N 曲线、max-IoU 直方图、candidate availability 曲线、
  remaining candidate count 直方图、same-category availability 曲线）
```

**与 gate 的关系**：本 audit 为 proposal engineering 产出，不产生任何 Gate Q1/Q2/Q3 结论；
其唯一作用于 A2.4 决定 N=64 还是 N=128，并为 A2.5 的 duplicate-suppression 决策提供
remaining candidate count distribution 依据。
