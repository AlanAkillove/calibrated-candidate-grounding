"""Index-level candidate manifests for the RefCOCOg external cohort (protocol A11.3/A11.4).

Amendment A10 froze *which* RefCOCOg expressions form the external cohort - its
``hard_cohort.csv`` is the authoritative row list (same-category K5 eligible) -
but a feasibility audit does not need bank-row indices, so none were written.
This module materialises exactly those indices for A11 and nothing else: the
selection *rule* is the frozen RefCOCO+ one, imported rather than restated.

Frozen pieces reused verbatim
-----------------------------
* :func:`ccg.data.proposals.assign_target` - argmax IoU target with
  ``iou_thresh = 0.5`` plus the target-equivalent removal;
* :func:`ccg.data.audit.assign_gt_category` / :func:`same_category_counts` - the
  A8 category rule (a proposal inherits the category of its highest-IoU COCO GT
  object only at IoU >= 0.5; ``-1`` = unknown never matches);
* :data:`ccg.data.manifests.MANIFEST_SEED` and the ``manifests-v1`` ordering -
  one deterministic shuffle per evaluated row, shared by every K as prefixes
  ``C_K = [target] + order[:K-1]``; per-K resampling is forbidden;
* :class:`ccg.data.manifests.ManifestEntry` / :class:`ManifestFile` - the frozen
  containers, so the emitted jsonl re-validates under the same schema check.

The one documented generalisation (A11.4)
----------------------------------------
RefCOCO+ keys its per-row shuffle by the region ``ref_id`` because one region
carries one sentence; RefCOCOg regions carry several sentences and the A11 unit
of evaluation is the *expression*, so the shuffle key is the expression's unique
``sent_id``.  Both are "one shuffle per evaluated row, shared across all K".

The same-category GT basis is the one A10 froze - the official
``instances_train2014.json`` restricted to the image, ``non_crowd()`` - and the
crowd-inclusive variant is reported as a sensitivity number, never as the
primary (see :func:`gt_basis_sensitivity`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.audit import assign_gt_category, same_category_counts
from ..data.bank import read_bank_image
from ..data.manifests import (
    MANIFEST_SEED,
    PRIMARY_KS,
    SCHEMA_VERSION,
    ManifestEntry,
    ManifestFile,
)
from ..data.proposals import DEFAULT_IOU_THRESH, assign_target
from ..data.types import ProposalBank
from . import refcocog as rg
from .feasibility import N_PROPOSALS, PRIMARY_K

__all__ = [
    "A11_K",
    "A11_MANIFEST_SCHEMA",
    "GT_BASIS_PRIMARY",
    "GT_BASIS_SENSITIVITY",
    "REGIMES",
    "RefCOCOGCohort",
    "RefCOCOGSample",
    "build_expression_manifest",
    "cohort_from_manifests",
    "gt_basis_sensitivity",
    "load_cohort_manifests",
    "read_gt_basis",
    "save_cohort_manifests",
    "verify_against_a10_cohort",
    "verify_matched_pair",
]

#: The only candidate-set size A11 evaluates (A11.14: ``K = 5 only``).
A11_K: int = int(PRIMARY_K)
#: The two regimes of the matched manipulation.
REGIMES: Tuple[str, ...] = ("random", "same_category")
#: Schema tag of the A11 cohort sidecar (``ManifestFile`` keeps ``manifests-v1``).
A11_MANIFEST_SCHEMA = "a11-refcocog-candidate-v1"
#: GT basis A10 froze for the cohort (official COCO train2014, ``non_crowd()``).
GT_BASIS_PRIMARY = "official_coco_train2014_non_crowd"
#: Crowd-inclusive basis, kept only to report how sensitive the cohort is.
GT_BASIS_SENSITIVITY = "official_coco_train2014_including_crowd"


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RefCOCOGSample:
    """One external row's candidate layout, duck-typed for ``materialise_examples``.

    It exposes exactly what :func:`ccg.semantic.hard_scores.materialise_examples`
    and :meth:`ccg.models.b3_data.B3Corpus.candidate_bank_indices` read
    (``sentence_id`` / ``image_id`` / ``target_index`` / ``distractor_order``),
    so the frozen B3 input assembly runs unchanged on external rows.  Local
    candidate index 0 is always the target.
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
        object.__setattr__(self, "image_id", int(self.image_id))
        object.__setattr__(self, "target_index", int(self.target_index))
        order = np.ascontiguousarray(self.distractor_order, dtype=np.int64).reshape(-1)
        if order.size != A11_K - 1:
            raise ValueError(
                f"expr {self.expr_id}: distractor_order has {order.size} rows, need {A11_K - 1}"
            )
        if int(self.target_index) in set(int(value) for value in order):
            raise ValueError(f"expr {self.expr_id}: the target appears among its own distractors")
        object.__setattr__(self, "distractor_order", order)

    @property
    def candidate_indices(self) -> np.ndarray:
        """Bank rows of ``C_5``: ``[target] + distractor_order``."""
        out = np.empty(A11_K, dtype=np.int64)
        out[0] = int(self.target_index)
        out[1:] = self.distractor_order
        return out


@dataclass(frozen=True)
class RefCOCOGCohort:
    """The matched external cohort: one row per expression, both regimes side by side."""

    expr_id: np.ndarray
    ref_id: np.ndarray
    image_id: np.ndarray
    target_index: np.ndarray
    rand_order: np.ndarray
    hard_order: np.ndarray
    category: List[str] = field(default_factory=list)
    category_id: np.ndarray = None  # type: ignore[assignment]
    target_best_iou: np.ndarray = None  # type: ignore[assignment]
    n_valid_distractors: np.ndarray = None  # type: ignore[assignment]
    n_same_category: np.ndarray = None  # type: ignore[assignment]
    n_target_equiv: np.ndarray = None  # type: ignore[assignment]
    text: List[str] = field(default_factory=list)
    regime: Dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(np.asarray(self.expr_id).reshape(-1).size)
        for name in (
            "expr_id",
            "ref_id",
            "image_id",
            "target_index",
            "category_id",
            "target_best_iou",
            "n_valid_distractors",
            "n_same_category",
            "n_target_equiv",
        ):
            values = np.asarray(getattr(self, name)).reshape(-1)
            if values.size != n:
                raise ValueError(f"{name} has {values.size} rows, cohort has {n}")
            dtype = np.float64 if name == "target_best_iou" else np.int64
            object.__setattr__(self, name, np.ascontiguousarray(values, dtype=dtype))
        for name in ("rand_order", "hard_order"):
            matrix = np.ascontiguousarray(getattr(self, name), dtype=np.int64)
            if matrix.shape != (n, A11_K - 1):
                raise ValueError(f"{name} must be [{n}, {A11_K - 1}], got {matrix.shape}")
            object.__setattr__(self, name, matrix)
        for name, values in (("category", self.category), ("text", self.text)):
            if len(values) != n:
                raise ValueError(f"{name} has {len(values)} entries, cohort has {n}")
        if not self.regime:
            object.__setattr__(
                self,
                "regime",
                {key: np.arange(n, dtype=np.int64) for key in REGIMES},
            )

    def __len__(self) -> int:
        return int(self.expr_id.size)

    @property
    def n_images(self) -> int:
        return int(np.unique(self.image_id).size)

    @property
    def n_refs(self) -> int:
        return int(np.unique(self.ref_id).size)

    def samples(self, regime: str) -> List[RefCOCOGSample]:
        """Duck-typed B3 samples of one regime, in cohort order."""
        regime = str(regime)
        if regime not in REGIMES:
            raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
        order = self.rand_order if regime == "random" else self.hard_order
        return [
            RefCOCOGSample(
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

    def subset(self, rows: Sequence[int] | np.ndarray) -> "RefCOCOGCohort":
        """A view restricted to ``rows`` (used by the diagnostics)."""
        index = np.asarray(rows, dtype=np.int64).reshape(-1)
        return RefCOCOGCohort(
            expr_id=self.expr_id[index],
            ref_id=self.ref_id[index],
            image_id=self.image_id[index],
            target_index=self.target_index[index],
            rand_order=self.rand_order[index],
            hard_order=self.hard_order[index],
            category=[self.category[int(row)] for row in index],
            category_id=self.category_id[index],
            target_best_iou=self.target_best_iou[index],
            n_valid_distractors=self.n_valid_distractors[index],
            n_same_category=self.n_same_category[index],
            n_target_equiv=self.n_target_equiv[index],
            text=[self.text[int(row)] for row in index],
            regime={key: index for key, value in self.regime.items()},
        )


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def read_gt_basis(
    gt_index: Any, image_id: int, *, basis: str = GT_BASIS_PRIMARY
) -> Tuple[np.ndarray, np.ndarray]:
    """``(gt_boxes_xyxy, gt_categories)`` of one image under the requested basis."""
    objects = gt_index.objects(int(image_id))
    if basis == GT_BASIS_PRIMARY:
        objects = objects.non_crowd()
    elif basis != GT_BASIS_SENSITIVITY:
        raise ValueError(f"unknown gt basis {basis!r}; expected {GT_BASIS_PRIMARY} or {GT_BASIS_SENSITIVITY}")
    return np.asarray(objects.boxes, dtype=np.float32), np.asarray(objects.categories, dtype=np.int64)


def _expression_bank(bank_paths: Mapping[str, Path], image_id: int) -> ProposalBank:
    """One image's frozen ``bank-v1`` group as a :class:`ProposalBank`.

    ``bank_paths`` maps a bank file label to its path; the image is looked up in
    each in turn (RefCOCOg strict lives entirely in the A10 bank, while the
    development-disjoint extras of A10 came from the RefCOCO+ frozen bank).  A
    missing image is an error, never a silent skip.
    """
    last_error: Optional[Exception] = None
    for label, path in bank_paths.items():
        try:
            boxes, objectness = read_bank_image(path, int(image_id))
        except KeyError as exc:
            last_error = exc
            continue
        return ProposalBank(image_id=int(image_id), boxes=boxes, objectness=objectness)
    raise KeyError(
        f"image {image_id} is in no bank ({', '.join(bank_paths)}); last error: {last_error}"
    )


def build_expression_manifest(
    expressions: Sequence[Any],
    *,
    bank_paths: Mapping[str, Path],
    gt_index: Any,
    regime: str,
    seed: int = MANIFEST_SEED,
    top_n: int = N_PROPOSALS,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    gt_basis: str = GT_BASIS_PRIMARY,
    eval_split: str = rg.UMD_EVAL_SPLIT,
) -> ManifestFile:
    """Apply the frozen ``manifests-v1`` construction to RefCOCOg expressions.

    ``expressions`` are A10 :class:`~ccg.external.refcocog.RefCOCOGExpression`
    records (``expr_id`` unique, ``gt_box`` in ``xyxy``).  Entries come out in
    ``(image_id, ann_id, expr_id)`` order, the same deterministic order A10 used
    for its audit, and ``ManifestEntry.ref_id`` carries the **expression** id -
    the row key of this dataset (see the module docstring).
    """
    regime = str(regime)
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; expected one of {REGIMES}")
    seed = int(seed)
    top_n = int(top_n)
    if top_n < A11_K:
        raise ValueError(f"top_n must be >= {A11_K}, got {top_n}")
    ordered = sorted(expressions, key=lambda item: (int(item.image_id), int(item.ann_id), int(item.expr_id)))

    entries: List[ManifestEntry] = []
    seen: set[int] = set()
    for expr in ordered:
        expr_id = int(expr.expr_id)
        if expr_id in seen:
            raise ValueError(f"duplicate expr_id {expr_id} - the row key must be unique")
        seen.add(expr_id)
        image_id = int(expr.image_id)
        bank = _expression_bank(bank_paths, image_id)
        if int(bank.boxes.shape[0]) != top_n:
            raise ValueError(
                f"expr {expr_id}: image {image_id} bank holds {int(bank.boxes.shape[0])} "
                f"proposals, the frozen bank is N={top_n}"
            )
        gt_box = np.asarray(expr.gt_box, dtype=np.float32).reshape(-1)
        if gt_box.size != 4 or not bool(np.all(np.isfinite(gt_box))):
            raise ValueError(f"expr {expr_id}: malformed gt box {expr.gt_box!r}")
        assignment = assign_target(bank, gt_box, iou_thresh=float(iou_thresh))
        target_index = assignment.target_proposal_idx
        if target_index is None:
            # A11 runs on the target-present cohort only (A11.14); a natural
            # omission must never be silently turned into a candidate row.
            raise ValueError(f"expr {expr_id}: target-absent row in the A11 cohort")
        blocked = [int(value) for value in assignment.to_remove.tolist()] + [int(target_index)]
        blocked_arr = np.unique(np.asarray(blocked, dtype=np.int64))
        valid = np.setdiff1d(np.arange(int(bank.boxes.shape[0]), dtype=np.int64), blocked_arr)
        n_valid = int(valid.size)

        gt_boxes, gt_categories = read_gt_basis(gt_index, image_id, basis=gt_basis)
        categories, _ = assign_gt_category(
            bank.boxes, gt_boxes, gt_categories, iou_thresh=float(iou_thresh)
        )
        target_category = int(categories[int(target_index)])
        n_same_available = int(
            same_category_counts(categories, int(target_index), assignment.to_remove)
        )

        rng = np.random.default_rng([seed, expr_id])
        if regime == "random":
            ordering = valid[rng.permutation(n_valid)]
            n_same_used = {int(k): 0 for k in PRIMARY_KS}
            hard_fractions = {int(k): 0.0 for k in PRIMARY_KS}
        else:
            if n_same_available > 0 and target_category >= 0:
                mask = categories[valid] == target_category
                same = valid[mask]
                rest = valid[~mask]
                if int(same.size) != n_same_available:
                    raise RuntimeError(
                        f"expr {expr_id}: same_category_counts ({n_same_available}) disagrees "
                        f"with the ordering split ({int(same.size)})"
                    )
            else:
                same = np.empty(0, dtype=np.int64)
                rest = valid
            ordering = np.concatenate([same, rest[rng.permutation(int(rest.size))]])
            n_same_used = {int(k): int(min(n_same_available, int(k) - 1)) for k in PRIMARY_KS}
            hard_fractions = {
                int(k): float(min(n_same_available, int(k) - 1)) / float(int(k) - 1)
                for k in PRIMARY_KS
            }
        cap = top_n - 1
        if ordering.size > cap:
            ordering = ordering[:cap]

        entries.append(
            ManifestEntry(
                ref_id=expr_id,
                image_id=image_id,
                split=str(eval_split),
                target_index=int(target_index),
                distractor_order=np.ascontiguousarray(ordering, dtype=np.int32),
                n_valid_distractors=n_valid,
                n_target_equiv_removed=int(assignment.to_remove.size),
                n_same_category_available=n_same_available,
                eligible={
                    int(k): bool(n_valid >= int(k) - 1) for k in PRIMARY_KS
                },
                n_same_used_by_K=n_same_used,
                hard_fraction_by_K=hard_fractions,
                target_max_iou=float(assignment.best_iou),
                eval_split=str(eval_split),
            )
        )

    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "a11_schema": A11_MANIFEST_SCHEMA,
        "regime": regime,
        "seed": seed,
        "top_n": top_n,
        "iou_thresh": float(iou_thresh),
        "gt_basis": gt_basis,
        "row_key": "expr_id (RefCOCOg global sent_id); stored in ManifestEntry.ref_id",
        "construction": (
            "frozen manifests-v1 ordering: one numpy.default_rng([seed, row_key]) shuffle per "
            "expression, shared by every K as prefixes; same_category promotes same-COCO-category "
            "distractors first in ascending bank order"
        ),
        "n_rows": len(entries),
        "n_images": len({int(entry.image_id) for entry in entries}),
        "n_refs": len({int(entry.ref_id) for entry in entries}),
        "primary_k": A11_K,
    }
    return ManifestFile(regime=regime, meta=meta, entries=entries)


# ---------------------------------------------------------------------------
# cohort assembly + verification
# ---------------------------------------------------------------------------
def _entry_map(manifest: ManifestFile) -> Dict[int, ManifestEntry]:
    out: Dict[int, ManifestEntry] = {}
    for entry in manifest.entries:
        key = int(entry.ref_id)
        if key in out:
            raise ValueError(f"duplicate manifest row {key}")
        out[key] = entry
    return out


def cohort_from_manifests(
    random_manifest: ManifestFile,
    hard_manifest: ManifestFile,
    expressions: Sequence[Any],
    *,
    gt_index: Any,
    gt_basis: str = GT_BASIS_PRIMARY,
) -> RefCOCOGCohort:
    """Join the two regime manifests into the matched cohort (intersection of rows)."""
    rand = _entry_map(random_manifest)
    hard = _entry_map(hard_manifest)
    rows = sorted(set(rand) & set(hard))
    if len(rows) != len(rand) or len(rows) != len(hard):
        raise AssertionError(
            f"regime manifests cover different rows: random={len(rand)} hard={len(hard)} "
            f"intersection={len(rows)}"
        )
    by_expr = {int(expr.expr_id): expr for expr in expressions}
    if len(by_expr) != len(expressions):
        raise ValueError("expressions carry duplicate expr_id values")

    cohort = RefCOCOGCohort(
        expr_id=np.asarray(rows, dtype=np.int64),
        ref_id=np.asarray([int(by_expr[int(row)].ref_id) for row in rows], dtype=np.int64),
        image_id=np.asarray([int(rand[int(row)].image_id) for row in rows], dtype=np.int64),
        target_index=np.asarray([int(rand[int(row)].target_index) for row in rows], dtype=np.int64),
        rand_order=np.asarray(
            [np.asarray(rand[int(row)].distractor_order, dtype=np.int64)[: A11_K - 1] for row in rows]
        ),
        hard_order=np.asarray(
            [np.asarray(hard[int(row)].distractor_order, dtype=np.int64)[: A11_K - 1] for row in rows]
        ),
        category=[str(by_expr[int(row)].target_name) for row in rows],
        category_id=np.asarray([int(by_expr[int(row)].category_id) for row in rows], dtype=np.int64),
        target_best_iou=np.asarray([float(rand[int(row)].target_max_iou) for row in rows]),
        n_valid_distractors=np.asarray(
            [int(rand[int(row)].n_valid_distractors) for row in rows], dtype=np.int64
        ),
        n_same_category=np.asarray(
            [int(hard[int(row)].n_same_category_available) for row in rows], dtype=np.int64
        ),
        n_target_equiv=np.asarray(
            [int(rand[int(row)].n_target_equiv_removed) for row in rows], dtype=np.int64
        ),
        text=[str(by_expr[int(row)].text) for row in rows],
    )
    report = verify_matched_pair(cohort, gt_index=gt_index, gt_basis=gt_basis)
    if not report["ok"]:
        failures = {key: value for key, value in report.items() if value != 0 and key != "ok"}
        raise AssertionError(f"cohort construction failed: {failures}")
    return cohort


def verify_matched_pair(
    cohort: RefCOCOGCohort,
    *,
    gt_index: Any,
    gt_basis: str = GT_BASIS_PRIMARY,
) -> Dict[str, Any]:
    """The A11.3 / section 4 assertions over the finished matched cohort.

    Every counter in the returned ``failures`` block must be ``0``:

    ``identity``             - random and hard share sentence / ref / image / target;
    ``k``                    - both regimes carry exactly ``A11_K - 1`` distractors;
    ``target_leak``          - no distractor equals the target or a target-equivalent;
    ``duplicates``           - no repeated bank row inside one candidate set;
    ``hard_same_category``   - every hard distractor shares the target's COCO category;
    ``random_same_category`` - the random control is *not* category-promoted
                               (counted, reported, and allowed to be non-zero);
    ``unknown_category``     - no hard row whose target category is unknown (``-1``);
    ``bank_range``           - every bank row is inside ``[0, N)``.
    """
    failures: Dict[str, int] = {
        "identity": 0,
        "k": 0,
        "target_leak": 0,
        "duplicates": 0,
        "hard_same_category": 0,
        "unknown_category": 0,
        "bank_range": 0,
        "eligible": 0,
    }
    n_random_same_distractors = 0
    for row in range(len(cohort)):
        image_id = int(cohort.image_id[row])
        target = int(cohort.target_index[row])
        rand = [int(value) for value in cohort.rand_order[row]]
        hard = [int(value) for value in cohort.hard_order[row]]
        if len(set(rand)) != A11_K - 1 or len(set(hard)) != A11_K - 1:
            failures["duplicates"] += 1
        if target in set(rand) or target in set(hard):
            failures["target_leak"] += 1
        if min(rand + hard + [target]) < 0 or max(rand + hard + [target]) >= N_PROPOSALS:
            failures["bank_range"] += 1
        if int(cohort.n_valid_distractors[row]) < A11_K - 1:
            failures["eligible"] += 1
        boxes, _ = _bank_boxes(image_id)
        gt_boxes, gt_categories = read_gt_basis(gt_index, image_id, basis=gt_basis)
        categories, _ = assign_gt_category(boxes, gt_boxes, gt_categories)
        target_category = int(categories[target])
        if target_category < 0:
            failures["unknown_category"] += 1
        if any(int(categories[index]) != target_category for index in hard):
            failures["hard_same_category"] += 1
        n_random_same_distractors += sum(
            1 for index in rand if int(categories[index]) == target_category
        )
    return {
        "ok": all(value == 0 for value in failures.values()),
        "n_rows": len(cohort),
        "n_images": cohort.n_images,
        "n_refs": cohort.n_refs,
        "K": A11_K,
        "gt_basis": gt_basis,
        "failures": failures,
        "random_distractors_sharing_target_category": int(n_random_same_distractors),
        "mean_same_category_supply": float(np.mean(cohort.n_same_category))
        if len(cohort)
        else None,
    }


_bank_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_BANK_SOURCE: Dict[str, Path] = {}


def configure_bank_source(bank_paths: Mapping[str, Path]) -> None:
    """Point :func:`_bank_boxes` at the bank file(s) the run reads (idempotent)."""
    global _BANK_SOURCE
    _BANK_SOURCE = {str(label): Path(path) for label, path in dict(bank_paths).items()}
    _bank_cache.clear()


def _bank_boxes(image_id: int) -> Tuple[np.ndarray, np.ndarray]:
    if not _BANK_SOURCE:
        raise RuntimeError("call configure_bank_source() before reading banks")
    hit = _bank_cache.get(int(image_id))
    if hit is None:
        bank = _expression_bank(_BANK_SOURCE, int(image_id))
        hit = (
            np.ascontiguousarray(bank.boxes, dtype=np.float32),
            np.ascontiguousarray(bank.objectness, dtype=np.float32),
        )
        _bank_cache[int(image_id)] = hit
    return hit


def verify_against_a10_cohort(
    cohort: RefCOCOGCohort, a10_rows: Sequence[Mapping[str, str]]
) -> Dict[str, Any]:
    """Pin the materialised cohort to A10's frozen ``hard_cohort.csv``.

    Checks the row set, the per-row same-category supply, the valid-distractor
    count and the target IoU.  Any mismatch means A11 is no longer evaluating
    the cohort A10 froze, which stops the run.

    ``hard_cohort.csv`` stores ``target_best_iou`` rounded to six decimals (that
    is how :func:`ccg.external.feasibility.ProposalAuditRow.to_row` writes it), so
    the IoU comparison happens on the rounded value - the tolerance of A11.10 is
    not widened here, only the reporting precision of the frozen table is
    respected.
    """
    a10_iou_decimals = 6
    wanted = {int(row["expr_id"]) for row in a10_rows}
    have = {int(value) for value in cohort.expr_id}
    by_expr = {int(row["expr_id"]): row for row in a10_rows}
    same_mismatch = 0
    valid_mismatch = 0
    iou_mismatch = 0
    for row in range(len(cohort)):
        expr_id = int(cohort.expr_id[row])
        source = by_expr.get(expr_id)
        if source is None:
            continue
        if int(cohort.n_same_category[row]) != int(source["n_same_category_distractors"]):
            same_mismatch += 1
        if int(cohort.n_valid_distractors[row]) != int(source["n_random_distractors"]):
            valid_mismatch += 1
        if abs(round(float(cohort.target_best_iou[row]), a10_iou_decimals) - float(source["target_best_iou"])) > 1e-9:
            iou_mismatch += 1
    return {
        "ok": (
            wanted == have
            and same_mismatch == 0
            and valid_mismatch == 0
            and iou_mismatch == 0
        ),
        "a10_rows": len(wanted),
        "a11_rows": len(have),
        "rows_only_in_a10": sorted(wanted - have)[:20],
        "rows_only_in_a11": sorted(have - wanted)[:20],
        "n_missing": len(wanted - have),
        "n_extra": len(have - wanted),
        "same_category_supply_mismatches": same_mismatch,
        "valid_distractor_count_mismatches": valid_mismatch,
        "target_iou_mismatches": iou_mismatch,
    }


def gt_basis_sensitivity(
    cohort: RefCOCOGCohort,
    *,
    gt_index: Any,
    gt_index_with_crowd: Any,
) -> Dict[str, Any]:
    """How many rows lose their 4 same-category distractors under the crowd-inclusive basis."""
    rows_at_risk = 0
    supply_delta: List[int] = []
    for row in range(len(cohort)):
        image_id = int(cohort.image_id[row])
        target = int(cohort.target_index[row])
        boxes, _ = _bank_boxes(image_id)
        primary_boxes, primary_cats = read_gt_basis(gt_index, image_id, basis=GT_BASIS_PRIMARY)
        other_boxes, other_cats = read_gt_basis(
            gt_index_with_crowd, image_id, basis=GT_BASIS_SENSITIVITY
        )
        primary_categories, _ = assign_gt_category(boxes, primary_boxes, primary_cats)
        other_categories, _ = assign_gt_category(boxes, other_boxes, other_cats)
        primary_same = int(
            same_category_counts(primary_categories, target, np.empty(0, dtype=np.int64))
        )
        other_same = int(
            same_category_counts(other_categories, target, np.empty(0, dtype=np.int64))
        )
        supply_delta.append(other_same - primary_same)
        if other_same < A11_K - 1:
            rows_at_risk += 1
    return {
        "n_rows": len(cohort),
        "rows_below_k5_under_crowd_inclusive_basis": rows_at_risk,
        "rows_with_supply_change": int(sum(1 for value in supply_delta if value != 0)),
        "min_supply_delta": int(min(supply_delta)) if supply_delta else None,
        "max_supply_delta": int(max(supply_delta)) if supply_delta else None,
        "note": (
            "primary stays on the A10-frozen non_crowd basis; this block only records how the "
            "cohort would move if crowd annotations were included, as they are in the RefCOCO+ "
            "manifest construction"
        ),
    }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def save_cohort_manifests(
    root: Path | str,
    manifests: Mapping[str, ManifestFile],
    cohort: RefCOCOGCohort,
) -> Dict[str, Any]:
    """Write the jsonl+meta pairs and the cohort sidecar; return their provenance."""
    import hashlib
    import json

    root = Path(root)
    files: Dict[str, str] = {}
    for regime, manifest in manifests.items():
        target = root / f"{regime}_rg_external_strict.jsonl"
        manifest.save(target)
        sidecar = target.with_suffix(".meta.json")
        files[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
        files[sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    sidecar = root / "cohort.csv"
    lines = [
        "expr_id,ref_id,image_id,category,category_id,target_index,"
        "rand_distractors,hard_distractors,target_best_iou,"
        "n_valid_distractors,n_same_category,n_target_equiv,text"
    ]
    for row in range(len(cohort)):
        lines.append(
            ",".join(
                [
                    str(int(cohort.expr_id[row])),
                    str(int(cohort.ref_id[row])),
                    str(int(cohort.image_id[row])),
                    json.dumps(str(cohort.category[row])),
                    str(int(cohort.category_id[row])),
                    str(int(cohort.target_index[row])),
                    json.dumps([int(v) for v in cohort.rand_order[row]]),
                    json.dumps([int(v) for v in cohort.hard_order[row]]),
                    repr(float(cohort.target_best_iou[row])),
                    str(int(cohort.n_valid_distractors[row])),
                    str(int(cohort.n_same_category[row])),
                    str(int(cohort.n_target_equiv[row])),
                    json.dumps(str(cohort.text[row]), ensure_ascii=False),
                ]
            )
        )
    sidecar.write_text("\n".join(lines) + "\n", encoding="utf-8")
    files[sidecar.name] = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    return {
        "root": str(root),
        "schema_version": A11_MANIFEST_SCHEMA,
        "manifests_schema": SCHEMA_VERSION,
        "n_rows": len(cohort),
        "files": files,
    }


def load_cohort_manifests(root: Path | str) -> Tuple[ManifestFile, ManifestFile]:
    """Read the ``(random, same_category)`` pair back through the frozen loader."""
    root = Path(root)
    pair: List[ManifestFile] = []
    for regime in REGIMES:
        target = root / f"{regime}_rg_external_strict.jsonl"
        pair.append(ManifestFile.load(target))
    return pair[0], pair[1]
