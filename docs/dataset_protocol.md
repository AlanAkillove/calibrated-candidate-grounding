# Dataset & Proposal Design Audit — `docs/dataset_protocol.md`

对应原始需求 **Task 3**（数据与 proposal design audit）。本文件在**不下载大文件**的前提下，
先核实数据组织、依赖、license、proposal generator 选型、存储与显存预算，并给出风险清单。

写作日期：2026-09-26。状态：pre-download audit（Phase 0 之前）。

本文件中所有来自外部检索的事实都标出来源；凡是**无法在下载后核验之前确认**的具体数字，
一律显式标注 **`[待下载后验证]`**，不允许写成已确定结论。

---

## 1. Primary dataset：RefCOCO+（UNC image-level split）

### 1.1 数据集事实（已核实）

| 项目 | 结论 | 来源 |
|---|---|---|
| 数据归属 | RefCOCO / RefCOCO+ / RefCOCOg 由 UNC（Kazemzadeh et al. EMNLP 2014 采集协议；Yu et al. ECCV 2016 "Modeling Context in Referring Expressions" 定义 split 与 REF/REC 任务）发布，图像来自 **MS COCO** | `github.com/lichengunc/refer` README |
| 官方分发形式 | 每个数据集一个 zip：`https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip`（README 明确写出该 URL） | 同上 |
| 服务器可用性 | UNC README 自述 "As the webserver is broken (sry about this), please check this Issue for all datasets downloading" → 官方直链可能失效，需走 GitHub issue 列出的镜像（社区镜像如 HuggingFace / OpenDataLab / TensorFlow Datasets `ref_coco`） | 同上 |
| 图像依赖 | README "Prepare Images"：把 `mscoco` 图像放入 `data/images`；RefCOCO、RefCOCO+、RefCOCOg 使用 COCO 图像 | 同上 |
| API / split 选择 | `refer.py` 支持 `dataset='refcoco+', splitBy='unc'`；同一 API 亦支持 `refcoco` 的 `unc`/`google`、`refcocog` 的 `google`/`umd`、`refclef` 的 `unc`/`berkeley` | 同上 |
| **UNC split 是 image-level** | TFDS 文档："`unc`" 和 "`umd`" splits **partition images** between train / validation / test（即图像不跨 split 出现）；而 Google-style split 不保证图像不相交 | `tensorflow.org/datasets/catalog/ref_coco`（经 OpenDataLab 镜像页复核） |
| RefCOCOg 的 google split 无 canonical test（val 常被论文当 test 报告，称 "val\*"） | 同上 | 同上 |
| **RefCOCO+ 不含空间介词** | TFDS 文档："RefCoco+ expressions are strictly appearance based descriptions, which they enforced by preventing raters from using location based descriptions"（禁用 "person on the left" 这类绝对空间描述词） | 同上；第三方标注文档复核 |
| 规模（常引用值） | RefCOCO：142,210 expressions / 50,000 objects / 19,994 images；**RefCOCO+：141,565 expressions / 49,856 objects / 19,992 images** | Liao et al., "A Real-Time Cross-Modality Correlation Filtering Method for Referring Expression Comprehension", CVPR 2020（openaccess.thecvf.com） |
| RefCOCO+ 的 UNC split 各子集表达数 | 常见表格值：train 120,624 / val 10,758 / testA 5,726 / testB 4,889（**注意**：四项相加 =141,997，与 141,565 不一致，可能因不同论文对 duplicate/unanswerable 表达的处理不同） | arXiv:2312.08007 Table 1 等；**`[待下载后验证]`** |
| testA / testB 语义 | 社区普遍描述为 RefCOCO/RefCOCO+ 的 testA 以 person 图像为主、testB 为非 person；本项目的 `testA`/`testB` 命名沿用官方文件中的 split 字段，**不自行重新定义** | OpenDataLab/TFDS 描述；**`[待下载后验证]`** |

**对本项目的影响**：
- 采用 **UNC image-level split** 是协议级决定（见 `research_protocol.md` §9），因为本项目缓存
  image/proposal feature，若同一图像跨 split 会造成 calibration/selection 污染。
- RefCOCO+ **不含绝对空间介词**这一点是一个有利条件：本项目 backbone 是 CLIP 家族（对空间
  关系本就弱），主数据集避免依赖空间介词可以减少 "失效究竟来自 candidate-set shift 还是来自
  CLIP 空间盲区" 的混淆。代价：RefCOCO+ 的表达偏外观，hardness 主要来自同类外观相似对象，
  这与本项目 same-category hard negatives 的定义天然契合（但也带来 §7 所述的 GT 信息泄漏风险）。

### 1.2 Annotation parser 方案对比（已核实）

| 方案 | 形式 | 优点 | 缺点 / 风险 | 选型 |
|---|---|---|---|---|
| **官方 `lichengunc/refer`（`refer.py` + `refs.json`）** | zip 内为 JSON 元数据 + split 信息；API 按 `splitBy='unc'` 加载 | 权威、与论文口径一致；UNC/Google/UMD split 同一 API 可选，便于**交叉核对 split 是否被误用** | 原始代码为 Python 2 时代风格；`make` 会编译 `_mask.c/_mask.so`（复制自 mscoco API），在 Windows 上是额外摩擦；包含 mask 功能但本项目只需 box | **采用（只取 JSON 解析逻辑，自己写只读 parser，不编译 mask 扩展）** |
| **Lake/ITSC "referring coco" `.mats`**（SCAN / RESCON / MATCHING 系列使用，如 `refcoco+ train_splitA.mat`） | MATLAB 导出的 `.mat`，需 `scipy.io.loadmat` | 大量开源 baseline 直接可用，字段扁平 | 属于**第三方二次预处理**，split 字段命名（splitA/splitB/splitC）与官方 `unc` split 的对应关系必须实证核对；混用有污染风险 | 仅作 **cross-check**（若可用），不作为主输入 |
| **JSON 转换版（`instances.json` + `refs.json` / `refcoco+.json`，常见于 ALBEF / LLaVA / PropVG 等仓库）** | COCO-style `instances.json` + 表达列表 | 与 pycocotools 生态兼容，字段清晰 | 不同仓库转换脚本字段不一致（有的把 `split` 写成字符串，有的写成 `.p` 索引文件）；来源可信度需逐仓库确认 | 备选（若官方 zip 不可达时的镜像路线） |
| `pycocotools` + 官方 COCO annotations | 只做 COCO GT 侧（IoU metadata、同类别查询） | 权威、必需 | 不含 referring expression 与 split | **必需辅件** |

**最终选择（冻结）**：以官方 `refcoco+.zip`（UNC split，`splitBy='unc'`）为唯一真值来源；
自行实现只读 parser 输出 `ReferringExample` dataclass（Task 4），仅依赖 `refs.json` 类元数据 +
COCO2014 `instances_trainval2014.json`。**禁止**在同一份实验里混用 `.mats` 第三方 split 与官方
split 的样本成员定义；若为了对齐 baseline 需要引用 `.mats`，只能作为额外的
`split_provenance` 标签列，不作为 split 判定依据。

### 1.3 COCO 图像依赖与获取步骤（不实际下载）

- RefCOCO+ 图像来自 COCO **train2014 + val2014**（官方 README 要求把 `mscoco` 放入 images
  目录）。常引用规模：train2014 = 82,783 张、val2014 = 40,504 张（COCO 官方页面）；
  RefCOCO+ 实际用到的图像约 **19,992 张**（见 §1.1），即 **约 20k 量级**。
- **注册要求**：COCO 官网下载需注册（填邮箱 → 邮件确认 → 获得带令牌的下载链接），
  `images_train2014.zip` 与 `images_val2014.zip` 各约 20GB / 6GB 量级。
- **License 注意事项**：COCO **annotations** 为 CC BY 4.0；COCO **images** 源自 Flickr，
  每张图各自保留其原始 Creative Commons 许可（部分含 BY-NC / BY-ND 等限制），因此
  `cocoapi` issue #81 指出 "COCO images do not adhere to license.txt"（不存在统一图像 license）。
  → 本项目 README/data 文档必须写明：图像由使用者自行从 COCO 官方获取，遵守逐图 CC 许可；
  仓库**不得**再分发图像；若发布派生数据（feature cache）需说明其依赖 COCO 图像许可。
- 下载步骤说明（未来由人工执行）：
  1. 注册 cocodataset.org → 下载 `annotations_trainval2014.zip`、`images_train2014.zip`、
     `images_val2014.zip` 至 `data/mscoco/`；
  2. 从 UNC 官方链接（或其 issue 列出的镜像）下载 `refcoco+.zip`，解压到 `data/refcoco+/`；
  3. 运行 `scripts/prepare_refcoco.py` 做完整性校验（见 §6 artifact）；
  4. **不下载** RefCOCO / RefCOCOg / gRefCOCO / Ref-L4（Phase 0 禁止多数据集全矩阵）。

## 2. External stress test：FineCops-Ref（已核实）

- Liu, Yang, Li, Wang. "FineCops-Ref: A new Dataset and Task for Fine-Grained Compositional
  Referring Expression Comprehension." **EMNLP 2024 main**（arXiv:2409.14750，v2 2025-01-11）。
- 检索到的构成（**test set**）：**9,605 positive expressions、9,814 negative expressions、
  8,507 negative images**；特点是可控难度分级（object category / attribute / multi-hop
  relation）以及通过**细粒度编辑与生成**构造 negative text 与 negative images，用于测试模型
  在目标不可见时的 **reject** 能力。
- 论文自述结论之一：specialist models 与 MLLMs 在该数据集上仍有明显差距（我们只引用其
  数据集动机，不引用性能结论）。
- **对本项目的意义与限制**：
  - 它承担 prompt 中 "relation confusion / compositional difficulty / negative selection" 的
    stress test 角色，因此本项目**不**人工生成 relation negatives。
  - 检索显示其规模是 **test-set** 描述；是否为 **evaluation-only（无 train split）**
    **`[待下载后验证]`**。若是 evaluation-only，则它只能做 zero-shot 外部压力测试（与本项目
    "不参与任何 model/calibration 选择" 的协议一致）。
  - 其 negative images 来自 **AI 编辑/生成**，与 RefCOCO+ 的真实 COCO 照片存在明显 domain
    gap；同时其图像是否落在 COCO train2014/val2014 之内（能否复用 proposal bank）**`[待下载后验证]`**。
    若不在 COCO 图像域内，则外部 stress test 必须用同一冻结 detector 重新生成 proposal bank
    （允许，因为 generator 冻结且 query-independent），并单独报告其 proposal recall。

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
  distribution），并且必须在任何 decision-model 结果之前完成、写入日志。
- 若 audit 显示 recall 过低（本项目预注册的操作性阈值，写在此处以免事后随意改）：
  - `recall@64 < 0.85`：必须报告为数据侧限制，并考虑以 **amendment** 方式更换 generator
    （保留原 generator 全部审计数字），同时把 natural-omission split 的统计单独报告；
  - 任何情况下**不得**通过提高 IoU 阈值、或从 RefCOCO+ 中静默删除 target-missing 样本来 "修复" recall。
  - `research_protocol.md` §5/§6 的 target 定义、嵌套性质、K∈{5,10,20,50} 网格不因 generator 更换而改变。

## 6. 规模、存储与显存预算

| 项 | 估算 | 依据 |
|---|---|---|
| RefCOCO+ 使用图像数 | ~19,992（约 **20k** 量级） | §1.1 文献值，`[待下载后验证]` |
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
| R3 | **Proposal recall 过低** | 未知，取决于 N=64 与 RefCOCO+ target 尺寸分布 | 由 §5 的 audit + 预注册操作阈值处理；必须报告 recall@32/recall@64/IoU 直方图，不允许静默过滤 |
| R4 | **Natural omission 样本量过少** | **高概率发生**（冻结 detector 在 COCO 上通常召回较高 → natural miss 少），导致 RQ4 的 natural split 统计功效不足 | 预先声明：natural_omission 为**描述性/小样本**结果，CI 必报；若样本量 < 足以支撑 3-seed 比较，则明确写 "该结论不显著"；主要 NONE 能力评估放在 synthetic omission 上，并显式声明 synthetic 的局限 |
| R5 | **K=50 时 proposal 数量不足** | 高概率发生（移除 IoU≥0.5 等价 proposal 之后，剩余候选数下降） | 先报告 **candidate availability distribution**（每图可用候选数直方图、按 split/类别分层），再决定：仅在候选数 ≥ K 的样本上评估该 K，并**同时报告被排除样本的数量与特征**（不得静默过滤）；必要时把主网格的最大 K 降为 20，并以 amendment 记录 |
| R6 | **Candidate set 中存在多个正确 proposal** | 若不处理则 accuracy/ECE 不可解释 | 协议 §5 已冻结：target = 唯一 max-IoU proposal，其他 IoU ≥ 0.5 的 proposal 一律移除 |
| R7 | **Hard negative 使用 GT 信息导致不公平** | **需要注意**：same-category hard negatives 依赖 COCO GT 类别与 IoU，属于 evaluation 时不可得的信息 | 处理：该 regime 与 CLIP-hard 一样按 **diagnostic evaluation regime** 报告（真实 detector 无法使用 GT）；同时保证 **target 的定义只用一次 assignment**，hard-negative 选择不得引入任何 "该候选是否为答案" 的信息；在结果表格中每列都带 regime 标签 |
| R8 | **CLIP-hard negatives 被误读为部署分布** | 存在 | 协议 §6 已声明 diagnostic/adversarial regime；文档与结果命名统一使用 `clip_hard_diagnostic` |
| R9 | **Calibration split 被用于 model selection** | 常见错误 | `val_select` 与 `val_calib` 按 image 分离并在代码中用不同 loader；temperature/threshold 拟合函数只接受 `val_calib`；新增检查：训练脚本若检测到 test/calib 混用则抛异常 |
| R10 | **softmax 的 K 依赖被误当成 calibration failure** | 存在（概念风险） | 协议 §2 RQ1 已写入；主指标使用 top-label ECE / correctness NLL / Brier 与 risk–coverage，并同时报告 ΔAcc 与 ΔCalibration 两条线 |
| R11 | **Crop context 丢失 / CLIP 空间弱** | 存在：candidate 用 crop embedding，丢失全局上下文；CLIP 对空间与小目标弱 | Phase 0 冻结为 **crop-only embedding**；不做 context-expansion、不加 image-level 特征融合（属于 protocol 变更，需 amendment）；把 CLIP 空间弱点作为**解释性限制**写入结论段，并允许 FineCops-Ref 作为外部压力测试来暴露该限制 |
| R12 | **同图表达在 evaluation 中的统计相关性** | 存在（RefCOCO+ 每对象平均 ~2.8 条表达） | 与 R2 相同：image-level bootstrap + 报告 per-image 聚合的敏感性分析（次要、不改判据） |
| R13 | **官方直链失效 / 镜像可信度** | UNC README 自述 server broken | 优先官方链接；若不可达，选用可校验镜像并把文件 sha256 写入 `data/MANIFEST.json`；禁止使用无法追溯来源的第三方打包 |
| R14 | **License 合规** | COCO 图像逐图 CC（含 NC/ND 变体）；RefCOCO 需遵守其引用要求；FineCops-Ref license **`[待下载后验证]`** | 仓库不分发图像/原始标注；README 与 `data/README.md` 写明注册要求与 attribution（Yu et al. ECCV 2016、Kazemzadeh et al. EMNLP 2014、COCO、FineCops-Ref/EMNLP 2024） |

## 8. 本文件的最终决定摘要

1. **Primary data**：RefCOCO+，**UNC image-level split**，官方 `refcoco+.zip` + 自写只读
   JSON parser；`.mats` 第三方版本仅作 cross-check，不作为 split 真值。
2. **Images**：COCO2014 `train2014` + `val2014`（注册下载，逐图 CC license，仓库不分发）；
   实际使用约 **20k** 张量级 `[待下载后验证]`。
3. **Proposal generator**：**冻结的 torchvision Faster R-CNN R50-FPN（COCO_V1，159.7 MB）**，
   N = 64，取 RPN NMS 后按 objectness top-64；IoU ≥ 0.5 的等价 proposal 移除，target 为唯一
   max-IoU proposal。
4. **External stress test**：FineCops-Ref（EMNLP 2024，test-only 特征待验证）。
5. **预算**：feature cache 约 1.3 GB region embeddings (FP16) + ~0.15 GB query embeddings，
   总 **< 3 GB**；提取 FP16 batch 64 起步、OOM 降 32。
6. **Gate 前置条件**：proposal quality audit（recall@32/@64、natural miss rate、IoU 分布、
   candidate availability 分布）必须先于任何 decision-model 结果完成并入库。

## 9. 产出物（`scripts/prepare_refcoco.py` / `audit_proposals.py`）

| Artifact | 内容 |
|---|---|
| `data/MANIFEST.json` | 每个下载文件的来源 URL、大小、sha256、license 备注、下载日期 |
| `data/refcoco+/parsed/referring_examples.jsonl` | `ref_id, image_id, text, gt_box, gt_object_id, split, split_provenance` |
| `data/refcoco+/splits/{train,val_select,val_calib,testA,testB}_image_ids.json` | image-level split 名单（供污染单测） |
| `cache/proposal_bank.h5` | `boxes[N,4]`, `objectness[N]`, `gt_iou[N]`, `gt_assignment`, `image_id` |
| `results/phase0/proposal_audit.json` | recall@32、recall@64、natural miss rate、IoU 直方图、候选可用数分布、按类别/尺寸分层 |
| `results/phase0/proposal_audit.md` | 人读版本 + 风险清单核查结论 |
