"""V2-D2 RefCOCO language-distribution transfer: cohort, corpus and audit primitives.

What D2 actually is (and is not)
--------------------------------
D1 froze that the two core V1/V2 findings survive *人工复核* of the RefCOCO+
annotations.  The original D2 brief asked for a **strict image-disjoint**
RefCOCO cohort.  That premise is *false on this data*: RefCOCO and RefCOCO+ are
two re-annotations of the **same** COCO ``train2014`` regions - RefCOCO+ only
removes spatial-mention words - so a RefCOCO ``testA``/``testB`` image is, with
one exception, *the very same physical image* the RefCOCO+ study was fitted and
selected on.  A non-empty image-disjoint RefCOCO cohort does not exist (see the
Phase-1 audit artifact).  The user re-scoped D2 accordingly:

    D2  =  expression-language-distribution transfer on shared COCO images.

Same 1,500 testA/testB images, same *frozen* COCO RPN (``N = 64``,
``cache/proposals.h5``), same region/global CLIP features, same frozen V1
reliability bundle.  The **only** new quantity is the RefCOCO ``sentences`` text,
re-encoded with the frozen B0 OpenCLIP ViT-B/32 text tower.  Nothing is trained
or refit; every candidate-set / target-assignment / same-category rule is the
frozen RefCOCO+ one, imported rather than restated.

Reuse map (nothing measured here is re-implemented)
---------------------------------------------------
* candidate construction - :func:`ccg.external.refcocog_manifests.build_expression_manifest`
  (dataset-agnostic: it only duck-types ``image_id/ann_id/expr_id/gt_box`` off the
  expression and reads ``gt_index.objects``).  It emits one ordering per row,
  capped at ``N-1 = 63``, and every ``C_K`` is the prefix ``[target] + order[:K-1]``
  (:meth:`ccg.models.b3_data.B3Corpus.candidate_bank_indices`), so a *single*
  stored ordering serves ``K = 5, 10, 20, 50`` with the frozen nesting.
* target assignment / same-category - :func:`ccg.data.proposals.assign_target`,
  :func:`ccg.data.audit.assign_gt_category` / :func:`same_category_counts` (via the
  manifest builder), GT basis = official ``instances_train2014.json`` restricted
  to the cohort images (``non_crowd()``, exactly A11's ``read_gt_basis``).
* proposal feasibility - :func:`ccg.external.feasibility.audit_expression` /
  :func:`summarise_rows` over ``K = 5, 10, 20, 50``.
* frozen scoring - :mod:`ccg.semantic.hard_scores` + the no-refit
  :class:`ccg.semantic.frozen_load.FrozenExternalModels` bundle.

The two containers below (:class:`LangSample`, :class:`LangCohort`) exist only
because the A11 :class:`~ccg.external.refcocog_manifests.RefCOCOGSample` hard-codes
``K = 5``; they keep the *full* ``DMAX``-1 ordering so ``materialise_examples`` can
slice any cardinality, and are otherwise field-for-field the A11 duck type.

Boundary that must not drift
----------------------------
Even a full replication here is *cross-dataset transfer under a shared COCO
visual domain and a shared frozen proposal system*.  It is **not** cross-visual-
domain generalisation and must never be reported as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.bank import read_bank_image
from ..data.coco import load_coco_index
from ..data.manifests import MANIFEST_SEED, PRIMARY_KS, ManifestEntry, ManifestFile
from ..data.proposals import DEFAULT_IOU_THRESH, assign_target, xywh_to_xyxy
from ..data.refcoco import load_instances_json, load_refs_pickle
from ..features.cache import FeatureCache
from ..models import b3_data
from . import refcocog_manifests as rman

__all__ = [
    "C1_KS",
    "DMAX",
    "HARD_MIN_SAME_CATEGORY",
    "TEST_SPLITS",
    "REFCOCO_REFS_PICKLE",
    "REFCOCO_INSTANCES_JSON",
    "REFCOCO_PLUS_REFS_PICKLE",
    "PLUS_DEV_MANIFEST_DIR",
    "MAIN_FEATURES_ROOT",
    "LANG_CACHE_ROOT",
    "FROZEN_BANK",
    "IMAGE_SIZES_NPZ",
    "COCO_TRAIN2014_GT",
    "FROZEN_MODELS_ROOT",
    "B3_ROOT",
    "B3_SEEDS",
    "LangExpression",
    "LangSample",
    "LangCohort",
    "RefCOCOLangFeatureCorpus",
    "load_refcoco_test_expressions",
    "target_present_expressions",
    "build_cohorts",
    "cohort_from_manifests",
    "load_image_sizes_map",
    "text_corpus_from_cohort",
    "load_refcoco_plus_image_sets",
    "load_cohort_csv",
]

# --- frozen evaluation geometry ---------------------------------------------
#: The cardinality stress set (== ``ccg.data.manifests.PRIMARY_KS`` / ``EVAL_KS``).
C1_KS: Tuple[int, ...] = tuple(int(k) for k in PRIMARY_KS)  # (5, 10, 20, 50)
#: Largest ``C_K`` D2 materialises; the stored ordering keeps ``DMAX - 1`` distractors.
DMAX: int = max(C1_KS)
#: The C4 matched cohort requires this many same-category distractors at ``K = 5``.
HARD_MIN_SAME_CATEGORY: int = 5 - 1
#: RefCOCO test splits that form the transfer cohort.
TEST_SPLITS: Tuple[str, ...] = ("testA", "testB")

# --- frozen on-disk inputs --------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
REFCOCO_REFS_PICKLE = REPO / "data" / "raw" / "refcoco" / "refcoco" / "refs(unc).p"
REFCOCO_INSTANCES_JSON = REPO / "data" / "raw" / "refcoco" / "refcoco" / "instances.json"
REFCOCO_PLUS_REFS_PICKLE = REPO / "data" / "raw" / "refcoco+" / "refcoco+" / "refs(unc).p"
#: RefCOCO+ frozen candidate manifests (train/val/testA/testB) - the exposure truth.
PLUS_DEV_MANIFEST_DIR = REPO / "cache" / "manifests"
#: The RefCOCO+ main region/global feature cache (reused verbatim on the shared images).
MAIN_FEATURES_ROOT = REPO / "cache" / "features"
#: Independent namespace for the *new* RefCOCO text features only (never touches main).
LANG_CACHE_ROOT = REPO / "cache" / "refcoco_lang"
#: The frozen COCO RPN ``N = 64`` bank every RefCOCO+ / A11 stage reads.
FROZEN_BANK = REPO / "cache" / "proposals.h5"
IMAGE_SIZES_NPZ = REPO / "cache" / "image_sizes.npz"
#: Official COCO GT basis for the A8 same-category rule (``non_crowd()``), as A11 froze.
COCO_TRAIN2014_GT = REPO / "data" / "raw" / "annotations" / "instances_train2014.json"
#: The frozen V1 reliability bundle (R1 / E1b / temperature), load-and-predict only.
FROZEN_MODELS_ROOT = REPO / "results" / "phase1e_refcocog_external" / "frozen_models"
#: The frozen Phase-0B B3 checkpoints (V1 scorer).
B3_ROOT = REPO / "results" / "phase0b_independent"
B3_SEEDS: Tuple[int, ...] = (1, 2, 3)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LangExpression:
    """One RefCOCO expression in the exact duck shape ``build_expression_manifest`` reads.

    ``expr_id`` is the RefCOCO region-pickle ``sent_id`` (globally unique across
    the 142,210 expressions), ``gt_box`` is ``xyxy`` joined from ``instances.json``
    through ``ann_id``.  ``level`` / ``tuple_type`` fill the two slots
    :func:`ccg.external.feasibility.audit_expression` groups by; here they carry
    the split and a fixed language class (RefCOCO has no difficulty levels).
    """

    expr_id: int
    image_id: int
    ann_id: int
    category_id: int
    ref_id: int
    target_name: str
    gt_box: Tuple[float, float, float, float]
    width: int
    height: int
    text: str
    split: str
    level: int = 1
    tuple_type: str = "refcoco_test"

    @property
    def sentence_id(self) -> int:  # hard_scores reads this off the *sample*, kept for symmetry
        return int(self.expr_id)


@dataclass(frozen=True)
class LangSample:
    """One cohort row's candidate layout, duck-typed for ``materialise_examples``.

    Field-for-field the A11 :class:`~ccg.external.refcocog_manifests.RefCOCOGSample`
    minus the ``K = 5`` hard-code: ``distractor_order`` keeps the full ``DMAX - 1``
    frozen ordering and ``candidate_bank_indices(sample, K)`` slices any prefix.
    Local candidate index 0 is always the target.
    """

    sentence_id: int
    ref_id: int
    image_id: int
    target_index: int
    distractor_order: np.ndarray
    regime: str
    expr_id: int
    category: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "sentence_id", int(self.sentence_id))
        object.__setattr__(self, "ref_id", int(self.ref_id))
        object.__setattr__(self, "image_id", int(self.image_id))
        object.__setattr__(self, "target_index", int(self.target_index))
        object.__setattr__(self, "expr_id", int(self.expr_id))
        order = np.ascontiguousarray(self.distractor_order, dtype=np.int64).reshape(-1)
        if order.size != DMAX - 1:
            raise ValueError(
                f"expr {self.expr_id}: distractor_order has {order.size} rows, need {DMAX - 1}"
            )
        if int(self.target_index) in set(int(v) for v in order):
            raise ValueError(f"expr {self.expr_id}: the target appears among its own distractors")
        object.__setattr__(self, "distractor_order", order)


@dataclass(frozen=True)
class LangCohort:
    """The matched D2 cohort: one row per expression, both regimes side by side."""

    expr_id: np.ndarray
    ref_id: np.ndarray
    image_id: np.ndarray
    ann_id: np.ndarray
    target_index: np.ndarray
    rand_order: np.ndarray  # [n, DMAX-1]
    hard_order: np.ndarray  # [n, DMAX-1]
    category: List[str] = field(default_factory=list)
    category_id: np.ndarray = None  # type: ignore[assignment]
    target_best_iou: np.ndarray = None  # type: ignore[assignment]
    n_valid_distractors: np.ndarray = None  # type: ignore[assignment]
    n_same_category: np.ndarray = None  # type: ignore[assignment]
    n_target_equiv: np.ndarray = None  # type: ignore[assignment]
    text: List[str] = field(default_factory=list)
    split: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = int(np.asarray(self.expr_id).reshape(-1).size)
        for name in (
            "expr_id", "ref_id", "image_id", "ann_id", "target_index",
            "category_id", "n_valid_distractors", "n_same_category", "n_target_equiv",
        ):
            object.__setattr__(self, name, np.ascontiguousarray(
                np.asarray(getattr(self, name)).reshape(-1), dtype=np.int64))
        iou = np.ascontiguousarray(np.asarray(self.target_best_iou).reshape(-1), dtype=np.float64)
        object.__setattr__(self, "target_best_iou", iou)
        for name in ("rand_order", "hard_order"):
            matrix = np.ascontiguousarray(getattr(self, name), dtype=np.int64)
            if matrix.shape != (n, DMAX - 1):
                raise ValueError(f"{name} must be [{n}, {DMAX - 1}], got {matrix.shape}")
            object.__setattr__(self, name, matrix)
        for name in ("category", "text", "split"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} entries, cohort has {n}")
        for name in ("expr_id", "ref_id", "image_id", "ann_id", "target_index",
                     "category_id", "n_valid_distractors", "n_same_category",
                     "n_target_equiv", "target_best_iou", "rand_order", "hard_order"):
            v = getattr(self, name)
            rows = v.shape[0] if name.endswith("order") else v.size
            if rows != n:
                raise ValueError(f"{name} has {rows} rows, cohort has {n}")

    def __len__(self) -> int:
        return int(self.expr_id.size)

    @property
    def n_images(self) -> int:
        return int(np.unique(self.image_id).size)

    @property
    def n_refs(self) -> int:
        return int(np.unique(self.ref_id).size)

    def samples(self, regime: str) -> List[LangSample]:
        order = self.rand_order if regime == "random" else self.hard_order
        return [
            LangSample(
                sentence_id=int(self.expr_id[row]),
                ref_id=int(self.ref_id[row]),
                image_id=int(self.image_id[row]),
                target_index=int(self.target_index[row]),
                distractor_order=order[row],
                regime=regime,
                expr_id=int(self.expr_id[row]),
                category=str(self.category[row]),
            )
            for row in range(len(self))
        ]

    def samples_for(self, regime: str, rows: Sequence[int] | np.ndarray) -> List[LangSample]:
        """Duck-typed B3 samples for a subset of cohort rows, in the given order."""
        order = self.rand_order if regime == "random" else self.hard_order
        return [
            LangSample(
                sentence_id=int(self.expr_id[row]),
                ref_id=int(self.ref_id[row]),
                image_id=int(self.image_id[row]),
                target_index=int(self.target_index[row]),
                distractor_order=order[int(row)],
                regime=regime,
                expr_id=int(self.expr_id[row]),
                category=str(self.category[row]),
            )
            for row in np.asarray(rows, dtype=np.int64).reshape(-1)
        ]

    def subset(self, rows: Sequence[int] | np.ndarray) -> "LangCohort":
        index = np.asarray(rows, dtype=np.int64).reshape(-1)
        return LangCohort(
            expr_id=self.expr_id[index],
            ref_id=self.ref_id[index],
            image_id=self.image_id[index],
            ann_id=self.ann_id[index],
            target_index=self.target_index[index],
            rand_order=self.rand_order[index],
            hard_order=self.hard_order[index],
            category=[self.category[int(r)] for r in index],
            category_id=self.category_id[index],
            target_best_iou=self.target_best_iou[index],
            n_valid_distractors=self.n_valid_distractors[index],
            n_same_category=self.n_same_category[index],
            n_target_equiv=self.n_target_equiv[index],
            text=[self.text[int(r)] for r in index],
            split=[self.split[int(r)] for r in index],
        )


# ---------------------------------------------------------------------------
# expression loading (region pickle -> expression records)
# ---------------------------------------------------------------------------
def _category_names(instances_path: Path) -> Dict[int, str]:
    with Path(instances_path).open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    return {int(c["id"]): str(c["name"]) for c in root.get("categories", [])}


def load_refcoco_test_expressions(
    *,
    refs_pickle: Path = REFCOCO_REFS_PICKLE,
    instances_json: Path = REFCOCO_INSTANCES_JSON,
    sizes: Optional[Mapping[int, Tuple[int, int]]] = None,
    splits: Sequence[str] = TEST_SPLITS,
) -> List[LangExpression]:
    """Expand RefCOCO ``refs(unc).p`` testA/testB into expression-level records.

    One :class:`LangExpression` per sentence; the box is joined from the archive's
    ``instances.json`` (``xywh`` -> ``xyxy``) through ``ann_id``.  ``sizes`` supplies
    the true ``(W, H)`` :func:`feasibility.audit_expression` reports (from the frozen
    ``image_sizes.npz``); an image without a measured size is skipped, never guessed.
    """
    wanted = {str(s) for s in splits}
    recs = load_refs_pickle(Path(refs_pickle))
    anns = load_instances_json(Path(instances_json))
    cat_names = _category_names(Path(instances_json))
    sizes = {} if sizes is None else {int(k): (int(v[0]), int(v[1])) for k, v in sizes.items()}
    out: List[LangExpression] = []
    for rec in recs:
        if str(rec.get("split")) not in wanted:
            continue
        image_id = int(np.asarray(rec["image_id"]).item())
        ann_id = int(np.asarray(rec["ann_id"]).item())
        size = sizes.get(image_id)
        if size is None:
            continue
        ann = anns.get(str(ann_id))
        if ann is None:
            raise KeyError(f"ann_id {ann_id} missing from {instances_json.name}")
        gt_box = tuple(float(v) for v in xywh_to_xyxy(np.asarray(ann["bbox"], dtype=np.float32)))
        category_id = int(ann["category_id"])
        width, height = size
        for sentence in rec["sentences"]:
            text = str(sentence.get("raw") or sentence.get("sent") or "").strip()
            if not text:
                raise ValueError(f"region ref_id={rec.get('ref_id')} sentence has no text")
            expr_id = int(np.asarray(sentence["sent_id"]).item())
            out.append(
                LangExpression(
                    expr_id=expr_id, image_id=image_id, ann_id=ann_id, category_id=category_id,
                    ref_id=int(np.asarray(rec["ref_id"]).item()),
                    target_name=cat_names.get(category_id, ""),
                    gt_box=gt_box, width=width, height=height,
                    text=text, split=str(rec["split"]),
                )
            )
    out.sort(key=lambda e: (e.image_id, e.ann_id, e.expr_id))
    return out


def target_present_expressions(
    expressions: Sequence[LangExpression],
    *,
    bank_path: Path = FROZEN_BANK,
    iou_thresh: float = DEFAULT_IOU_THRESH,
) -> Tuple[List[LangExpression], Dict[str, Any]]:
    """Keep only rows whose target reaches ``iou_thresh`` in the frozen bank.

    ``build_expression_manifest`` raises on a target-absent row (a natural omission
    must never become a candidate row), so the cohort is the target-present subset.
    The dropped count is reported, not swallowed - it is the Phase-2 ``natural
    omission`` number.
    """
    present: List[LangExpression] = []
    absent = 0
    best_iou_sum = 0.0
    for expr in expressions:
        boxes, objectness = read_bank_image(Path(bank_path), int(expr.image_id))
        bank = _bank(expr.image_id, boxes, objectness)
        assignment = assign_target(bank, np.asarray(expr.gt_box, dtype=np.float32), iou_thresh=iou_thresh)
        best_iou_sum += float(assignment.best_iou)
        if assignment.target_proposal_idx is None:
            absent += 1
        else:
            present.append(expr)
    n = len(expressions)
    return present, {
        "n_expressions": n,
        "n_target_present": len(present),
        "natural_omission_at_iou_thresh": absent,
        "natural_omission_rate": (absent / n) if n else None,
        "mean_target_best_iou": (best_iou_sum / n) if n else None,
        "iou_thresh": float(iou_thresh),
    }


def _bank(image_id: int, boxes: np.ndarray, objectness: np.ndarray):
    from ..data.types import ProposalBank

    return ProposalBank(image_id=int(image_id), boxes=boxes, objectness=objectness)


# ---------------------------------------------------------------------------
# cohort construction (candidate sets for every K, both regimes)
# ---------------------------------------------------------------------------
def build_cohorts(
    expressions: Sequence[LangExpression],
    *,
    gt_index: Any,
    bank_paths: Optional[Mapping[str, Path]] = None,
    gt_basis: str = rman.GT_BASIS_PRIMARY,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    seed: int = MANIFEST_SEED,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[LangCohort, ManifestFile, ManifestFile]:
    """Materialise the matched D2 cohort (random + same_category) at ``DMAX`` width.

    Returns ``(cohort, random_manifest, hard_manifest)``.  ``expressions`` must all be
    target-present (filter with :func:`target_present_expressions` first).  Reuses the
    frozen manifest builder; only the cohort assembly is D2's own because it keeps the
    full ``DMAX - 1`` ordering instead of A11's ``K5`` slice.
    """
    emit = log or (lambda _m: None)
    bank_paths = dict(DEFAULT_BANK_PATHS if bank_paths is None else bank_paths)
    rman.configure_bank_source(bank_paths)
    common = dict(bank_paths=bank_paths, gt_index=gt_index, gt_basis=gt_basis,
                  iou_thresh=iou_thresh, seed=seed, eval_split="testA_testB")
    rand_manifest = rman.build_expression_manifest(expressions, regime="random", **common)
    hard_manifest = rman.build_expression_manifest(expressions, regime="same_category", **common)
    emit(f"[d2-cohort] manifests: random={len(rand_manifest.entries)} hard={len(hard_manifest.entries)}")
    cohort = cohort_from_manifests(rand_manifest, hard_manifest, expressions)
    emit(f"[d2-cohort] matched rows={len(cohort)} images={cohort.n_images} refs={cohort.n_refs}")
    return cohort, rand_manifest, hard_manifest


#: A11 reads the RefCOCOg strict bank first, then the frozen RefCOCO+ bank.  D2 needs
#: only the frozen RefCOCO+ bank (the shared testA/testB images live there).
DEFAULT_BANK_PATHS: Dict[str, Path] = {"frozen_plus": FROZEN_BANK}


def _entry_map(manifest: ManifestFile) -> Dict[int, ManifestEntry]:
    out: Dict[int, ManifestEntry] = {}
    for entry in manifest.entries:
        key = int(entry.ref_id)  # the manifest builder stores expr_id in ref_id
        if key in out:
            raise ValueError(f"duplicate manifest row {key}")
        out[key] = entry
    return out


def cohort_from_manifests(
    random_manifest: ManifestFile,
    hard_manifest: ManifestFile,
    expressions: Sequence[LangExpression],
) -> LangCohort:
    """Join the two regime manifests into the matched ``DMAX``-width cohort."""
    rand = _entry_map(random_manifest)
    hard = _entry_map(hard_manifest)
    rows = sorted(set(rand) & set(hard))
    if len(rows) != len(rand) or len(rows) != len(hard):
        raise AssertionError(
            f"regime manifests cover different rows: random={len(rand)} hard={len(hard)} "
            f"intersection={len(rows)}"
        )
    by_expr = {int(e.expr_id): e for e in expressions}
    if len(by_expr) != len(expressions):
        raise ValueError("expressions carry duplicate expr_id")

    def _order(entry: ManifestEntry) -> np.ndarray:
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        if order.size >= DMAX - 1:
            return order[: DMAX - 1]
        pad = np.full(DMAX - 1 - order.size, -1, dtype=np.int64)
        return np.concatenate([order, pad])

    rand_orders = np.stack([_order(rand[int(r)]) for r in rows])
    hard_orders = np.stack([_order(hard[int(r)]) for r in rows])
    cohort = LangCohort(
        expr_id=np.asarray(rows, dtype=np.int64),
        ref_id=np.asarray([int(by_expr[int(r)].ref_id) for r in rows], dtype=np.int64),
        image_id=np.asarray([int(rand[int(r)].image_id) for r in rows], dtype=np.int64),
        ann_id=np.asarray([int(by_expr[int(r)].ann_id) for r in rows], dtype=np.int64),
        target_index=np.asarray([int(rand[int(r)].target_index) for r in rows], dtype=np.int64),
        rand_order=rand_orders,
        hard_order=hard_orders,
        category=[str(by_expr[int(r)].target_name) for r in rows],
        category_id=np.asarray([int(by_expr[int(r)].category_id) for r in rows], dtype=np.int64),
        target_best_iou=np.asarray([float(rand[int(r)].target_max_iou) for r in rows]),
        n_valid_distractors=np.asarray([int(rand[int(r)].n_valid_distractors) for r in rows]),
        n_same_category=np.asarray([int(hard[int(r)].n_same_category_available) for r in rows]),
        n_target_equiv=np.asarray([int(rand[int(r)].n_target_equiv_removed) for r in rows]),
        text=[str(by_expr[int(r)].text) for r in rows],
        split=[str(by_expr[int(r)].split) for r in rows],
    )
    _validate_cohort(cohort, rand, hard)
    return cohort


def _validate_cohort(
    cohort: LangCohort, rand: Dict[int, ManifestEntry], hard: Dict[int, ManifestEntry]
) -> None:
    """Frozen invariants: identity, target-leak, duplicates, bank range, ordering legality."""
    for row in range(len(cohort)):
        target = int(cohort.target_index[row])
        for order in (cohort.rand_order[row], cohort.hard_order[row]):
            valid = order[order >= 0]
            if valid.size != order.size:
                # padded rows exist only when n_valid < DMAX-1; they must sort last
                if np.any(order[order < 0] != -1) or np.any(order[: valid.size] < 0):
                    raise AssertionError(f"expr {cohort.expr_id[row]}: padding not sorted last")
            if target in set(int(v) for v in valid):
                raise AssertionError(f"expr {cohort.expr_id[row]}: target among its distractors")
            if valid.size and np.unique(valid).size != valid.size:
                raise AssertionError(f"expr {cohort.expr_id[row]}: duplicate distractor bank row")
            if valid.size and (int(valid.min()) < 0 or int(valid.max()) >= rman.N_PROPOSALS):
                raise AssertionError(f"expr {cohort.expr_id[row]}: bank row outside [0, N)")
        if int(cohort.n_valid_distractors[row]) < cohort.rand_order[row][cohort.rand_order[row] >= 0].size:
            raise AssertionError(f"expr {cohort.expr_id[row]}: n_valid_distractors disagrees with order")


# ---------------------------------------------------------------------------
# cohort slicing helpers used by the C1 / C4 analyses
# ---------------------------------------------------------------------------
def common_cohort_rows(cohort: LangCohort) -> np.ndarray:
    """Rows eligible at ``K = DMAX`` (identical row set for every ``C1_KS`` prefix)."""
    return np.asarray(
        [row for row in range(len(cohort)) if int(cohort.n_valid_distractors[row]) >= DMAX - 1],
        dtype=np.int64,
    )


def matched_hard_rows(cohort: LangCohort) -> np.ndarray:
    """Rows carrying ``HARD_MIN_SAME_CATEGORY`` same-category distractors (the C4 cohort)."""
    return np.asarray(
        [row for row in range(len(cohort)) if int(cohort.n_same_category[row]) >= HARD_MIN_SAME_CATEGORY],
        dtype=np.int64,
    )


# ---------------------------------------------------------------------------
# the read-only feature corpus the frozen scorer walks (region + text from two roots)
# ---------------------------------------------------------------------------
@dataclass
class RefCOCOLangFeatureCorpus:
    """``B3Corpus`` duck type over *two* caches: regions from main, text from D2's.

    The RefCOCO testA/testB images are physically the RefCOCO+ test images, so their
    region features already live in ``cache/features`` and the proposal bank in
    ``cache/proposals.h5`` - reused verbatim.  Only the RefCOCO expression *text* is
    new, read from ``cache/refcoco_lang``.  Image sizes come from ``image_sizes.npz``.
    Region / text are widened to float32 exactly as :class:`B3Corpus` / the A11 corpus
    do, so the frozen forward sees the identical dtype contract.
    """

    region_root: Path
    text_root: Path
    bank_paths: Mapping[str, Path]
    image_sizes: Mapping[int, Tuple[int, int]]

    def __post_init__(self) -> None:
        self.region_root = Path(self.region_root)
        self.text_root = Path(self.text_root)
        self.bank_paths = {str(k): Path(v) for k, v in dict(self.bank_paths).items()}
        self.image_sizes = {int(k): (int(v[0]), int(v[1])) for k, v in dict(self.image_sizes).items()}
        self._region_cache: Optional[FeatureCache] = None
        self._text_cache: Optional[FeatureCache] = None
        self._bank_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _regions(self) -> FeatureCache:
        if self._region_cache is None:
            self._region_cache = FeatureCache.open(self.region_root)
        return self._region_cache

    def _texts(self) -> FeatureCache:
        if self._text_cache is None:
            self._text_cache = FeatureCache.open(self.text_root)
        return self._text_cache

    @property
    def feature_dim(self) -> int:
        return int(self._regions().feature_dim)

    def _region(self, image_id: int) -> np.ndarray:
        return np.asarray(self._regions().region_features(int(image_id)), dtype=np.float32)

    def _text(self, sentence_id: int) -> np.ndarray:
        return np.asarray(self._texts().text_features(int(sentence_id)), dtype=np.float32)

    def _bank(self, image_id: int) -> Tuple[np.ndarray, np.ndarray]:
        image_id = int(image_id)
        hit = self._bank_cache.get(image_id)
        if hit is None:
            last: Optional[Exception] = None
            for path in self.bank_paths.values():
                try:
                    hit = read_bank_image(Path(path), image_id)
                    break
                except KeyError as exc:
                    last = exc
            if hit is None:
                raise KeyError(f"image {image_id} in no bank; last error {last}")
            hit = (
                np.ascontiguousarray(hit[0], dtype=np.float32),
                np.ascontiguousarray(hit[1], dtype=np.float32),
            )
            self._bank_cache[image_id] = hit
        return hit

    def image_size(self, image_id: int) -> Tuple[int, int]:
        image_id = int(image_id)
        if image_id not in self.image_sizes:
            raise KeyError(f"image {image_id} has no measured size in the D2 corpus")
        return self.image_sizes[image_id]

    def close(self) -> None:
        for handle in (self._region_cache, self._text_cache):
            if handle is not None:
                handle.close()
        self._region_cache = None
        self._text_cache = None

    def __enter__(self) -> "RefCOCOLangFeatureCorpus":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ---------------------------------------------------------------------------
# shared data helpers
# ---------------------------------------------------------------------------
def load_image_sizes_map(npz_path: Path = IMAGE_SIZES_NPZ) -> Dict[int, Tuple[int, int]]:
    """``{image_id: (W, H)}`` from the frozen per-image-string-key ``image_sizes.npz``."""
    arr = np.load(Path(npz_path), allow_pickle=False)
    out: Dict[int, Tuple[int, int]] = {}
    for key in arr.files:
        try:
            image_id = int(key)
        except ValueError:
            continue
        value = np.asarray(arr[key]).reshape(-1)
        if value.size >= 2:
            out[image_id] = (int(value[0]), int(value[1]))
    return out


def official_gt_index(
    image_ids: Sequence[int], gt_file: Path = COCO_TRAIN2014_GT
) -> Any:
    """Official COCO index restricted to ``image_ids`` (A8 / A11 GT basis)."""
    return load_coco_index([Path(gt_file)], only_image_ids=[int(i) for i in image_ids])


def text_corpus_from_cohort(cohort: LangCohort) -> List[dict]:
    """``run_text_extraction`` rows, keyed by ``expr_id`` (== the B3 ``sentence_id``)."""
    rows: List[dict] = []
    for row in range(len(cohort)):
        text = str(cohort.text[row]).strip()
        if not text:
            raise AssertionError(f"expr {int(cohort.expr_id[row])}: empty expression text")
        rows.append(
            {
                "sentence_id": int(cohort.expr_id[row]),
                "ref_id": int(cohort.ref_id[row]),
                "sent_id": 0,
                "image_id": int(cohort.image_id[row]),
                "split": "test",
                "text": text,
            }
        )
    return rows


def load_refcoco_plus_image_sets(
    manifest_dir: Path = PLUS_DEV_MANIFEST_DIR,
    *,
    refs_pickle: Path = REFCOCO_PLUS_REFS_PICKLE,
) -> Dict[str, Any]:
    """RefCOCO+ exposure image sets (from the frozen manifests) + full-file image ids.

    Wraps :func:`ccg.external.refcocog.load_development_image_sets` and adds the
    union of every RefCOCO+ image (train/val/testA/testB) so the Phase-1 audit can
    report the honest overlap that kills the ``strict image-disjoint`` premise.
    """
    from .refcocog import load_development_image_sets

    dev = load_development_image_sets(manifest_dir)
    recs = load_refs_pickle(Path(refs_pickle))
    all_plus = {int(np.asarray(r["image_id"]).item()) for r in recs}
    out: Dict[str, Any] = {k: sorted(v) for k, v in dev.items()}
    out["all_refcoco_plus_ids"] = sorted(all_plus)
    return out


def load_cohort_csv(path: Path) -> LangCohort:
    """Read the Phase-1 frozen ``cohort.csv`` sidecar back into a :class:`LangCohort`.

    Downstream stages re-read the persisted cohort instead of re-deriving the
    candidate construction, so the row set / ordering every stage scores is exactly
    the one Phase 1 froze.  ``rand_distractors`` / ``hard_distractors`` are stored
    as the ``>= 0`` prefix (padded -1 dropped), so each is re-padded to ``DMAX - 1``;
    ``candidate_bank_indices`` only ever slices a prefix of length ``<= n_valid``,
    which stays inside the real (non-padded) entries.
    """
    import csv

    expr_id, ref_id, image_id, ann_id, target_index = [], [], [], [], []
    category, category_id, split, text = [], [], [], []
    target_best_iou, n_valid, n_same, n_equiv = [], [], [], []
    rand_orders, hard_orders = [], []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            expr_id.append(int(row["expr_id"]))
            ref_id.append(int(row["ref_id"]))
            image_id.append(int(row["image_id"]))
            ann_id.append(int(row["ann_id"]))
            target_index.append(int(row["target_index"]))
            category.append(str(row["category"]))
            category_id.append(int(row["category_id"]))
            split.append(str(row["split"]))
            text.append(str(row["text"]))
            target_best_iou.append(float(row["target_best_iou"]))
            n_valid.append(int(row["n_valid_distractors"]))
            n_same.append(int(row["n_same_category"]))
            n_equiv.append(int(row["n_target_equiv"]))
            for source, sink in ((row["rand_distractors"], rand_orders),
                                 (row["hard_distractors"], hard_orders)):
                order = np.asarray(json.loads(source), dtype=np.int64)
                padded = np.full(DMAX - 1, -1, dtype=np.int64)
                padded[: min(order.size, DMAX - 1)] = order[: DMAX - 1]
                sink.append(padded)
    return LangCohort(
        expr_id=np.asarray(expr_id, dtype=np.int64),
        ref_id=np.asarray(ref_id, dtype=np.int64),
        image_id=np.asarray(image_id, dtype=np.int64),
        ann_id=np.asarray(ann_id, dtype=np.int64),
        target_index=np.asarray(target_index, dtype=np.int64),
        rand_order=np.stack(rand_orders),
        hard_order=np.stack(hard_orders),
        category=category,
        category_id=np.asarray(category_id, dtype=np.int64),
        target_best_iou=np.asarray(target_best_iou, dtype=np.float64),
        n_valid_distractors=np.asarray(n_valid, dtype=np.int64),
        n_same_category=np.asarray(n_same, dtype=np.int64),
        n_target_equiv=np.asarray(n_equiv, dtype=np.int64),
        text=text,
        split=split,
    )
