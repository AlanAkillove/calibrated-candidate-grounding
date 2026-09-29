"""A10 external candidate: RefCOCOg (UMD split) as an image-disjoint test set.

What this module owns
---------------------
Everything that has to be decided *before* a proposal bank is touched:

* :func:`load_refs_umd` / :func:`expand_expressions` - parse the official
  ``refs(umd).p`` annotation archive and join the missing box field from the
  COCO instances (G0);
* :func:`load_development_image_sets` - which COCO images RefCOCO+ development
  actually exposed (train / val_select / val_calib), read back from the frozen
  candidate manifests rather than recomputed (G1);
* :func:`build_subsets` - the two image-disjoint external subsets
  ``rg_external_strict`` and ``rg_external_devdisjoint`` plus the pre-registered
  size gate of instruction section 9 (G2);
* the distribution audits of instruction sections 18/19 (language shift, target
  category shift);
* the A10 decision rules (sections 11/14/16/22), whose thresholds are frozen
  here *before* any RefCOCOg number exists.

Deliberate reuse
----------------
No measurement primitive is re-implemented.  The proposal audit itself is done
with :func:`ccg.external.feasibility.audit_expression`, i.e. the same target
assignment (``assign_target``, IoU >= 0.5, inclusive), the same
target-equivalent removal, the same ``assign_gt_category`` rule that maps a
proposal to the category of its best-overlapping COCO GT object, and the same
``same_category_counts`` distractor supply that produced the A8 cohort.  That
identity is what allows a RefCOCOg number to be compared with a RefCOCO+ number
at all; "the same manipulation" is a claim about code, not about prose.

Boundary that must not drift (instruction section 24)
----------------------------------------------------
RefCOCOg shares the COCO image domain and COCO object ontology with RefCOCO+.
This is a *cross-dataset* replication under a *shared visual domain*, NOT
cross-domain visual generalisation.  Everything measured here therefore speaks
about language / annotation-distribution transfer, and the frozen COCO-pretrained
RPN is expected (section 11) to carry the target here the way it does on
RefCOCO+; if it does not, the pipeline is broken and must be debugged, not
re-tuned.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..data.audit import wilson_ci
from ..data.manifests import ManifestFile
from ..data.proposals import iou_matrix, xywh_to_xyxy
from ..data.splits import (
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_TRAIN,
    SPLIT_VAL_CALIB,
    SPLIT_VAL_SELECT,
    normalize_split_name,
)
from .feasibility import N_PROPOSALS, PRIMARY_K  # the frozen budget the audit reuses

__all__ = [
    # frozen A10 criteria
    "UMD_EVAL_SPLIT",
    "STRICT_MIN_EXPRESSIONS",
    "STRICT_MIN_IMAGES",
    "DEV_MIN_EXPRESSIONS",
    "DEV_MIN_IMAGES",
    "A10_GO_RECALL_MIN",
    "A10_GO_RANDOM_K5_MIN",
    "A10_GRAY_RECALL_MIN",
    "A10_SAMECAT_PRIMARY_MIN",
    "A10_SAMECAT_GOOD_MIN",
    "A10_SAMECAT_STOP_MAX",
    "A10_HARD_MIN_EXPRESSIONS",
    "A10_HARD_MIN_IMAGES",
    "DISTRACTOR_COVERAGE_LEVELS",
    "SUBSET_STRICT",
    "SUBSET_DEV",
    "SUBSET_EXCLUDED",
    "SPATIAL_TOKENS",
    "ABSOLUTE_POSITION_TOKENS",
    "N_PROPOSALS",
    "PRIMARY_K",
    # records
    "RefCOCOGRegion",
    "RefCOCOGExpression",
    "SubsetStats",
    # parsing
    "load_refs_umd",
    "canonical_image_file_name",
    "expand_expressions",
    "summarise_dataset",
    # overlap and subsets
    "image_ids_by_split",
    "load_development_image_sets",
    "overlap_rows",
    "build_subsets",
    "size_gate",
    # distributions
    "whitespace_tokens",
    "expression_length_rows",
    "category_rows",
    "same_category_coverage_rows",
    "target_box_agreement",
    "matched_hard_rows",
    # decisions
    "engineering_verdict_a10",
    "same_category_verdict_a10",
    "hard_cohort_gate_a10",
    "branch_decision_a10",
]

# ---------------------------------------------------------------------------
# frozen A10 criteria (instruction sections 9/11/14/16/22)
# ---------------------------------------------------------------------------
#: The UMD split is image-level, and only ``test`` is ever treated as external.
UMD_EVAL_SPLIT = "test"

#: Section 9: strict-disjoint is primary when it reaches these sizes.
STRICT_MIN_EXPRESSIONS = 1500
STRICT_MIN_IMAGES = 500
#: Section 9: the fallback development-disjoint subset needs *more* material to
#: compensate for the images it borrowed from development.
DEV_MIN_EXPRESSIONS = 2000
DEV_MIN_IMAGES = 700

#: Section 11: stricter than A9 on purpose - same image domain, so a low recall
#: means a broken annotation/target/split pipeline, not a domain shift.
A10_GO_RECALL_MIN = 0.90
A10_GO_RANDOM_K5_MIN = 0.95
A10_GRAY_RECALL_MIN = 0.85

#: Section 14: same-category K5 availability for the external hard regime.
A10_SAMECAT_PRIMARY_MIN = 0.60
A10_SAMECAT_GOOD_MIN = 0.75
A10_SAMECAT_STOP_MAX = 0.50

#: Section 16: minimum statistical power of the hard cohort.
A10_HARD_MIN_EXPRESSIONS = 1000
A10_HARD_MIN_IMAGES = 300

#: Section 14: distractor-supply coverage reported (>= k same-category objects).
DISTRACTOR_COVERAGE_LEVELS: Tuple[int, ...] = (1, 2, 4, 9)

SUBSET_STRICT = "rg_external_strict"
SUBSET_DEV = "rg_external_devdisjoint"
#: A test image that survives neither definition.  Distinct from ``SUBSET_STRICT``
#: on purpose: defaulting an excluded image to "strict" would silently advertise
#: every RefCOCOg test expression as external material.
SUBSET_EXCLUDED = "excluded"

#: Section 18: a deliberately small, closed word list - prevalence of spatial /
#: absolute-position wording.  No NLP taxonomy, no parsing, no new model.
SPATIAL_TOKENS: frozenset[str] = frozenset(
    {
        "on", "in", "at", "of", "to", "from", "with", "without", "above", "below",
        "over", "under", "inside", "outside", "beside", "between", "near", "next",
        "left", "right", "front", "back", "middle", "center", "top", "bottom",
        "up", "down", "into", "onto", "around", "across", "behind", "before",
    }
)
#: Words RefCOCO+ collection prohibited (absolute position); their prevalence is
#: the cleanest single evidence that the two annotation distributions differ.
ABSOLUTE_POSITION_TOKENS: frozenset[str] = frozenset(
    {"left", "right", "middle", "center", "top", "bottom", "up", "down", "corner", "edge"}
)


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RefCOCOGRegion:
    """One record of ``refs(umd).p``: one referred COCO object."""

    ref_id: int
    image_id: int
    ann_id: int
    category_id: int
    file_name: str
    split: str
    sentences: Tuple[Tuple[int, str, Tuple[str, ...]], ...] = ()

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)


@dataclass
class RefCOCOGExpression:
    """One *expression* (sentence) of RefCOCOg, in the audit's attribute shape.

    The field names mirror :class:`ccg.external.finecops.FineCopsExpression`
    exactly where :func:`ccg.external.feasibility.audit_expression` reads them
    (``expr_id`` / ``image_id`` / ``level`` / ``tuple_type`` / ``target_name`` /
    ``width`` / ``height`` / ``gt_box``), so the very same audit function - and
    therefore the same definitions of "target present", "valid distractor" and
    "same-category supply" - runs on both datasets.

    ``level`` and ``tuple_type`` carry A10 meanings instead of FineCops'
    difficulty levels and tuple types: ``level`` selects the external subset a
    row belongs to (1 = strict-disjoint, 2 = added by the looser
    development-disjoint definition, 0 = a test image excluded by both) and
    ``tuple_type`` records the RefCOCOg expression class, so ``summarise_rows``
    groups by subset / language class in the same slots it grouped by difficulty
    for FineCops.
    """

    expr_id: int
    image_id: int
    level: int
    tuple_type: str
    target_name: str
    width: int
    height: int
    gt_box: Tuple[float, float, float, float]
    ann_id: int = 0
    category_id: int = -1
    ref_id: int = 0
    region_ref_id: int = 0
    split: str = UMD_EVAL_SPLIT
    text: str = ""
    tokens: Tuple[str, ...] = ()
    file_name: str = ""
    subset: str = SUBSET_EXCLUDED

    def to_row(self) -> Dict[str, Any]:
        return {
            "expr_id": self.expr_id,
            "region_ref_id": self.region_ref_id,
            "image_id": self.image_id,
            "ann_id": self.ann_id,
            "category_id": self.category_id,
            "target_name": self.target_name,
            "subset": self.subset,
            "level": self.level,
            "tuple_type": self.tuple_type,
            "split": self.split,
            "n_tokens": len(self.tokens),
            "expression": self.text,
        }


@dataclass
class SubsetStats:
    """Size of one candidate external subset (section 8/9).

    ``rg_external_devdisjoint`` is reported as the *superset* it is: every image
    that survives the strict definition also survives the looser
    development-disjoint one, so its counts include the strict rows.
    """

    name: str
    n_expressions: int = 0
    n_refs: int = 0
    n_images: int = 0
    n_objects: int = 0
    excluded_images: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_expressions": self.n_expressions,
            "n_refs": self.n_refs,
            "n_images": self.n_images,
            "n_objects": self.n_objects,
            "excluded_images": self.excluded_images,
            "meets_strict_criterion": bool(
                self.n_expressions >= STRICT_MIN_EXPRESSIONS
                and self.n_images >= STRICT_MIN_IMAGES
            ),
            "meets_dev_criterion": bool(
                self.n_expressions >= DEV_MIN_EXPRESSIONS and self.n_images >= DEV_MIN_IMAGES
            ),
        }


# ---------------------------------------------------------------------------
# G0 - parsing
# ---------------------------------------------------------------------------
def load_refs_umd(path: str | Path) -> List[dict]:
    """Unpickle the official ``refs(umd).p``.

    .. warning::
       ``pickle`` executes arbitrary objects while loading.  Point this only at
       the archive this repository downloaded and hash-pinned in
       ``data/raw/refcocog/dataset_card.json`` (the official referit zip, served
       by the Internet Archive copy of the broken UNC URL); never at an uploaded
       or network-streamed file.  No json form of the UMD split exists, which is
       why this one pickle load is unavoidable.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found - run tools/download_refcocog.py first (A10 G0)"
        )
    with open(p, "rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise ValueError(f"{p}: expected a list of dict records, got {type(raw).__name__}")
    return raw


def parse_regions(raw: Sequence[Mapping[str, Any]]) -> List[RefCOCOGRegion]:
    """``refs(umd).p`` records -> :class:`RefCOCOGRegion`, sorted and validated.

    Layout measured on the downloaded archive (2026-09-29): every record carries
    ``ref_id``, ``image_id``, ``ann_id``, ``category_id``, ``file_name``,
    ``split`` and ``sentences[{tokens, raw, sent, sent_id}]``.  There is **no**
    box field, exactly like RefCOCO+'s ``refs(unc).p`` - the target box is joined
    from the COCO instances through ``ann_id``.
    """
    wanted = {"ref_id", "image_id", "ann_id", "category_id", "file_name", "split", "sentences"}
    regions: List[RefCOCOGRegion] = []
    for rec in raw:
        missing = wanted - set(rec)
        if missing:
            raise ValueError(f"record {rec.get('ref_id')!r} lacks fields {sorted(missing)}")
        sentences: List[Tuple[int, str, Tuple[str, ...]]] = []
        for sent in rec["sentences"] or []:
            text = str(sent.get("sent") or sent.get("raw") or "").strip()
            if not text:
                continue
            tokens = tuple(str(t).lower() for t in (sent.get("tokens") or text.split()))
            sentences.append((int(sent.get("sent_id", -1)), text, tokens))
        regions.append(
            RefCOCOGRegion(
                ref_id=int(rec["ref_id"]),
                image_id=int(rec["image_id"]),
                ann_id=int(rec["ann_id"]),
                category_id=int(rec["category_id"]),
                file_name=str(rec["file_name"]),
                split=normalize_split_name(rec["split"]),
                sentences=tuple(sentences),
            )
        )
    regions.sort(key=lambda r: (r.image_id, r.ann_id, r.ref_id))
    dup = [k for k, n in Counter((r.image_id, r.ann_id, r.ref_id) for r in regions).items() if n > 1]
    if dup:
        raise ValueError(f"{len(dup)} duplicated (image_id, ann_id, ref_id) keys, e.g. {dup[:3]}")
    return regions


def canonical_image_file_name(file_name: str, image_id: int) -> str:
    """RefCOCOg's per-object ``file_name`` -> the real COCO file name.

    Measured: ``refs(umd).p`` stores ``COCO_train2014_000000380440_491042.jpg``,
    i.e. the COCO name with the annotation id appended (RefCOCOg names each
    *referred object* as if it were its own image).  The canonical form drops
    that suffix; ``image_id`` is checked against the remaining zero-padded COCO
    id so a silent mismatch between the two id systems is impossible.
    """
    name = Path(str(file_name)).name
    stem, _, ext = name.partition(".")
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "COCO":
        raise ValueError(f"{file_name!r} is not a COCO-style RefCOCOg file name")
    numeric = [p for p in parts[2:] if p.isdigit()]
    if not numeric:
        raise ValueError(f"{file_name!r} carries no numeric COCO id")
    coco_id = int(numeric[0])
    if coco_id != int(image_id):
        raise ValueError(
            f"{file_name!r} encodes COCO id {coco_id} but the record says image_id={image_id}"
        )
    keep = parts[:3] if len(parts) >= 3 else parts
    return "_".join(keep) + "." + (ext or "jpg")


def expand_expressions(
    regions: Sequence[RefCOCOGRegion],
    *,
    boxes_by_ann: Mapping[int, Sequence[float]],
    image_meta: Mapping[int, Tuple[int, int]],
    category_names: Mapping[int, str],
    splits: Optional[Iterable[str]] = None,
    subsets_by_image: Optional[Mapping[int, str]] = None,
) -> List[RefCOCOGExpression]:
    """Region records -> expression-level audit inputs.

    ``boxes_by_ann`` maps a COCO annotation id to its ``xywh`` box (the join that
    replaces the missing ``bbox`` field); ``image_meta`` maps a COCO image id to
    ``(width, height)``.  A region whose annotation is absent from
    ``boxes_by_ann`` is reported, never guessed at, by
    :func:`summarise_dataset` - so this function simply skips it.
    """
    wanted = {normalize_split_name(s) for s in splits} if splits else None
    out: List[RefCOCOGExpression] = []
    for reg in regions:
        if wanted is not None and reg.split not in wanted:
            continue
        box = boxes_by_ann.get(int(reg.ann_id))
        meta = image_meta.get(int(reg.image_id))
        if box is None or meta is None:
            continue
        x, y, w, h = (float(v) for v in box)
        width, height = int(meta[0]), int(meta[1])
        subset = (subsets_by_image or {}).get(int(reg.image_id), SUBSET_EXCLUDED)
        level = {
            SUBSET_STRICT: 1,
            SUBSET_DEV: 2,
            SUBSET_EXCLUDED: 0,
        }[subset]
        for sent_id, text, tokens in reg.sentences:
            out.append(
                RefCOCOGExpression(
                    expr_id=int(sent_id),
                    image_id=int(reg.image_id),
                    level=level,
                    tuple_type=_expression_class(tokens),
                    target_name=str(category_names.get(int(reg.category_id), f"cat_{reg.category_id}")),
                    width=width,
                    height=height,
                    gt_box=(x, y, x + w, y + h),
                    ann_id=int(reg.ann_id),
                    category_id=int(reg.category_id),
                    ref_id=int(reg.ref_id),
                    region_ref_id=int(reg.ref_id),
                    split=reg.split,
                    text=text,
                    tokens=tokens,
                    subset=subset,
                )
            )
    out.sort(key=lambda e: (e.image_id, e.ann_id, e.expr_id))
    seen: Set[int] = set()
    for expr in out:
        if expr.expr_id in seen:
            raise ValueError(f"sent_id {expr.expr_id} is not unique; expr_id cannot key the audit")
        seen.add(expr.expr_id)
    return out


def _expression_class(tokens: Sequence[str]) -> str:
    """Coarse, pre-registered expression class (section 18, no NLP taxonomy).

    ``spatial`` uses a relational / positional word, ``attribute`` a colour-like
    modifier with no relation, ``simple`` neither.  The point is only to show the
    two datasets do not share one annotation distribution; it is not a taxonomy
    claim.
    """
    toks = {str(t).lower() for t in tokens}
    if toks & SPATIAL_TOKENS:
        return "spatial"
    if toks & _COLOR_TOKENS:
        return "attribute"
    return "simple"


#: Minimal colour vocabulary, used only by :func:`_expression_class`.
_COLOR_TOKENS: frozenset[str] = frozenset(
    {
        "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey",
        "orange", "pink", "purple", "tan", "dark", "light", "colored",
    }
)


def summarise_dataset(
    regions: Sequence[RefCOCOGRegion],
    *,
    instances_member: Mapping[str, Any],
    provenance: Mapping[str, Any],
    boxes_by_ann: Mapping[int, Sequence[float]],
    n_unmatched_ann: int,
) -> Dict[str, Any]:
    """G0 dataset-integrity summary, measured from the files (section 6)."""
    by_split: Dict[str, List[RefCOCOGRegion]] = {}
    for reg in regions:
        by_split.setdefault(reg.split, []).append(reg)
    splits: Dict[str, Any] = {}
    for name, regs in sorted(by_split.items()):
        splits[name] = {
            "n_refs": len(regs),
            "n_expressions": sum(r.n_sentences for r in regs),
            "n_images": len({r.image_id for r in regs}),
            "n_objects": len({(r.image_id, r.ann_id) for r in regs}),
            "n_categories": len({r.category_id for r in regs}),
            "sentences_per_ref": {
                str(k): v for k, v in sorted(Counter(r.n_sentences for r in regs).items())
            },
        }
    all_images = {r.image_id for r in regions}
    split_images = {n: {r.image_id for r in regs} for n, regs in by_split.items()}
    return {
        "provenance": dict(provenance),
        "records": {
            "n_refs_total": len(regions),
            "n_expressions_total": sum(r.n_sentences for r in regions),
            "n_images_total": len(all_images),
            "n_objects_total": len({(r.image_id, r.ann_id) for r in regions}),
            "splits": splits,
            "split_labels": sorted(by_split),
            "image_level_split": bool(
                all(
                    len(a & b) == 0
                    for i, (n1, a) in enumerate(split_images.items())
                    for n2, b in list(split_images.items())[i + 1 :]
                )
            ),
            "shared_images_between_splits": {
                f"{n1}|{n2}": len(a & b)
                for i, (n1, a) in enumerate(split_images.items())
                for n2, b in list(split_images.items())[i + 1 :]
                if (a & b)
            },
        },
        "annotation_file": {
            "images": int(len(instances_member.get("images", []))),
            "annotations": int(len(instances_member.get("annotations", []))),
            "categories": int(len(instances_member.get("categories", []))),
        },
        "target_box_join": {
            "n_boxes_available": int(len(boxes_by_ann)),
            "n_refs_unmatched_in_coco": int(n_unmatched_ann),
            "unmatched_rate": round(n_unmatched_ann / max(1, len(regions)), 6),
        },
    }


# ---------------------------------------------------------------------------
# G1 - image overlap
# ---------------------------------------------------------------------------
def image_ids_by_split(regions: Sequence[RefCOCOGRegion]) -> Dict[str, Set[int]]:
    buckets: Dict[str, Set[int]] = {}
    for reg in regions:
        buckets.setdefault(reg.split, set()).add(int(reg.image_id))
    return buckets


def load_development_image_sets(manifest_dir: str | Path) -> Dict[str, Set[int]]:
    """RefCOCO+ image ids per development split, from the frozen manifests.

    Reading the manifests (instead of re-deriving the val sub-split) means the
    answer comes from *what the study actually used*: every entry carries the
    ``eval_split`` label its images were routed to when B3 / the reliability
    models / the calibrators were fitted.

    Returns keys ``train``, ``val_select``, ``val_calib``, ``testA``, ``testB``,
    plus the composites ``development`` (train + val_select + val_calib) and
    ``all_refcoco_plus``.
    """
    root = Path(manifest_dir)
    files = sorted(root.glob("random_*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no random_*.jsonl manifests under {root} - the RefCOCO+ split image sets "
            "cannot be established without them"
        )
    sets: Dict[str, Set[int]] = {
        SPLIT_TRAIN: set(),
        SPLIT_VAL_SELECT: set(),
        SPLIT_VAL_CALIB: set(),
        SPLIT_TEST_A: set(),
        SPLIT_TEST_B: set(),
    }
    for path in files:
        mf = ManifestFile.load(path)
        for entry in mf.entries:
            name = entry.eval_split or normalize_split_name(entry.split)
            if name in sets:
                sets[name].add(int(entry.image_id))
    sets["development"] = set(sets[SPLIT_TRAIN]) | set(sets[SPLIT_VAL_SELECT]) | set(sets[SPLIT_VAL_CALIB])
    sets["all_refcoco_plus"] = (
        set(sets["development"]) | set(sets[SPLIT_TEST_A]) | set(sets[SPLIT_TEST_B])
    )
    return sets


def overlap_rows(
    rg_images: Mapping[str, Set[int]],
    plus_sets: Mapping[str, Set[int]],
    expressions_by_image: Mapping[int, int],
    refs_by_image: Mapping[int, int],
    *,
    compared_splits: Sequence[str] = (UMD_EVAL_SPLIT,),
) -> List[Dict[str, Any]]:
    """G1 matrix: every RefCOCOg split x every RefCOCO+ development set.

    ``n_expressions_remaining`` / ``n_refs_remaining`` are what survives on the
    *non-overlapping* images, i.e. the raw material of the external subset.
    """
    rows: List[Dict[str, Any]] = []
    for rg_split in compared_splits:
        rg = set(rg_images.get(rg_split, set()))
        for plus_name in (
            SPLIT_TRAIN,
            SPLIT_VAL_SELECT,
            SPLIT_VAL_CALIB,
            "development",
            SPLIT_TEST_A,
            SPLIT_TEST_B,
            "all_refcoco_plus",
        ):
            plus = set(plus_sets.get(plus_name, set()))
            inter = rg & plus
            rest = rg - plus
            rows.append(
                {
                    "refcocog_split": rg_split,
                    "refcocog_images": len(rg),
                    "refcoco_plus_set": plus_name,
                    "refcoco_plus_images": len(plus),
                    "overlapping_images": len(inter),
                    "non_overlapping_images": len(rest),
                    "overlap_rate": round(len(inter) / len(rg), 6) if rg else 0.0,
                    "expressions_remaining": sum(expressions_by_image.get(i, 0) for i in rest),
                    "refs_remaining": sum(refs_by_image.get(i, 0) for i in rest),
                    "expressions_total": sum(expressions_by_image.get(i, 0) for i in rg),
                    "refs_total": sum(refs_by_image.get(i, 0) for i in rg),
                }
            )
    return rows


def subset_membership(
    rg_images: Mapping[str, Set[int]], plus_sets: Mapping[str, Set[int]], *, split: str = UMD_EVAL_SPLIT
) -> Dict[int, str]:
    """``image_id -> subset`` for the images that survive either definition.

    Strict (level 1) wins when an image qualifies for both, so ``level`` is a
    partition and the development-disjoint figures are cumulative.
    """
    rg = set(rg_images.get(split, set()))
    strict_keep = rg - set(plus_sets.get("all_refcoco_plus", set()))
    dev_keep = rg - set(plus_sets.get("development", set()))
    mapping = {int(i): SUBSET_DEV for i in dev_keep}
    mapping.update({int(i): SUBSET_STRICT for i in strict_keep})
    return mapping


def build_subsets(
    expressions: Sequence[RefCOCOGExpression],
    regions: Sequence[RefCOCOGRegion],
    membership: Mapping[int, str],
    *,
    split: str = UMD_EVAL_SPLIT,
) -> Dict[str, SubsetStats]:
    """Section 8: the two disjoint external subsets, and what is left in each."""
    stats = {
        SUBSET_STRICT: SubsetStats(name=SUBSET_STRICT),
        SUBSET_DEV: SubsetStats(name=SUBSET_DEV),
    }
    images: Dict[str, Set[int]] = {SUBSET_STRICT: set(), SUBSET_DEV: set()}
    objects: Dict[str, Set[Tuple[int, int]]] = {SUBSET_STRICT: set(), SUBSET_DEV: set()}
    refs: Dict[str, Set[Tuple[int, int, int]]] = {SUBSET_STRICT: set(), SUBSET_DEV: set()}
    for reg in regions:
        if reg.split != split:
            continue
        subset = membership.get(int(reg.image_id))
        keys = (SUBSET_STRICT, SUBSET_DEV) if subset == SUBSET_STRICT else (subset,)
        for name in keys:
            if name is None:
                continue
            refs[name].add((int(reg.image_id), int(reg.ann_id), int(reg.ref_id)))
            objects[name].add((int(reg.image_id), int(reg.ann_id)))
            images[name].add(int(reg.image_id))
    for expr in expressions:
        if expr.split != split:
            continue
        subset = membership.get(int(expr.image_id))
        if subset is None:
            continue
        stats[subset].n_expressions += 1
        if subset == SUBSET_STRICT:
            stats[SUBSET_DEV].n_expressions += 1
    test_images = {int(r.image_id) for r in regions if r.split == split}
    for name, s in stats.items():
        s.n_refs = len(refs[name])
        s.n_objects = len(objects[name])
        s.n_images = len(images[name])
        s.excluded_images = len(test_images - images[name])
    return stats


def size_gate(stats: Mapping[str, SubsetStats]) -> Dict[str, Any]:
    """Section 9: strict first, development-disjoint only as a gray zone."""
    strict = stats.get(SUBSET_STRICT)
    dev = stats.get(SUBSET_DEV)
    if strict is None or dev is None:
        raise ValueError("size_gate needs both rg_external_strict and rg_external_devdisjoint")
    if strict.n_expressions >= STRICT_MIN_EXPRESSIONS and strict.n_images >= STRICT_MIN_IMAGES:
        primary, decision, note = SUBSET_STRICT, "STRICT PRIMARY", (
            f"strict-disjoint reaches {strict.n_expressions} expressions / {strict.n_images} "
            f"images, above the {STRICT_MIN_EXPRESSIONS}/{STRICT_MIN_IMAGES} floor"
        )
    elif dev.n_expressions >= DEV_MIN_EXPRESSIONS and dev.n_images >= DEV_MIN_IMAGES:
        primary, decision, note = SUBSET_DEV, "EXTERNAL GRAY ZONE", (
            f"strict-disjoint is too small ({strict.n_expressions} expressions / "
            f"{strict.n_images} images); development-disjoint reaches "
            f"{dev.n_expressions}/{dev.n_images} - report before running"
        )
    else:
        primary, decision, note = "", "REFCOCOG EXTERNAL STOP", (
            f"neither subset is large enough (strict {strict.n_expressions}/"
            f"{strict.n_images}, dev {dev.n_expressions}/{dev.n_images})"
        )
    return {
        "primary_subset": primary,
        "decision": decision,
        "criteria": {
            "strict_min_expressions": STRICT_MIN_EXPRESSIONS,
            "strict_min_images": STRICT_MIN_IMAGES,
            "dev_min_expressions": DEV_MIN_EXPRESSIONS,
            "dev_min_images": DEV_MIN_IMAGES,
        },
        "measured": {SUBSET_STRICT: strict.to_dict(), SUBSET_DEV: dev.to_dict()},
        "note": note,
    }


# ---------------------------------------------------------------------------
# G4 / section 18-19 distributions
# ---------------------------------------------------------------------------
def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": round(float(arr.mean()), 4),
        "p10": round(float(np.quantile(arr, 0.10)), 4),
        "median": round(float(np.median(arr)), 4),
        "p75": round(float(np.quantile(arr, 0.75)), 4),
        "p90": round(float(np.quantile(arr, 0.90)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def whitespace_tokens(text: str) -> Tuple[str, ...]:
    """One tokenizer for *both* datasets in the section 18 comparison.

    RefCOCOg ships a ``tokens`` field, RefCOCO+ does not, so a length or
    vocabulary difference could always be a tokeniser artefact.  Comparing
    whitespace-split lower-case text removes that confound; the archive tokens
    are still what :func:`expand_expressions` classifies with.
    """
    return tuple(str(text or "").lower().split())


def expression_length_rows(
    groups: Mapping[str, Sequence[Tuple[str, Sequence[str]]]],
) -> List[Dict[str, Any]]:
    """One row per expression group: length, vocabulary, spatial-word use.

    A group is ``(text, tokens)`` pairs, not dataset records, so RefCOCO+ and
    RefCOCOg rows come out of the same code - a length difference only means
    something if both sides are counted identically.
    """
    rows: List[Dict[str, Any]] = []
    for name, exprs in sorted(groups.items()):
        if not exprs:
            rows.append({"group": name, "n_expressions": 0})
            continue
        token_sets = [set(str(t).lower() for t in toks) for _text, toks in exprs]
        lengths = [len(tuple(toks)) for _text, toks in exprs]
        vocab: Counter[str] = Counter()
        for toks in token_sets:
            vocab.update(toks)
        spatial = sum(1 for s in token_sets if s & SPATIAL_TOKENS)
        absolute = sum(1 for s in token_sets if s & ABSOLUTE_POSITION_TOKENS)
        rows.append(
            {
                "group": name,
                "n_expressions": len(exprs),
                "tokens_per_expression": _quantiles(lengths),
                "mean_characters": round(float(np.mean([len(str(t)) for t, _x in exprs])), 4),
                "vocabulary_size": len(vocab),
                "hapax_legomena": sum(1 for _, n in vocab.items() if n == 1),
                "spatial_token_expressions": spatial,
                "spatial_token_rate": round(spatial / len(exprs), 6),
                "absolute_position_expressions": absolute,
                "absolute_position_rate": round(absolute / len(exprs), 6),
            }
        )
    return rows


def category_rows(
    groups: Mapping[str, Sequence[Tuple[str, int]]], *, top_k: int = 10
) -> List[Dict[str, Any]]:
    """Target-category frequency, entropy and person share per group (section 19).

    ``groups`` maps a name to ``(category_name, image_id)`` pairs, again
    dataset-agnostic so that both cohorts are measured by the same code.
    """
    rows: List[Dict[str, Any]] = []
    for name, exprs in sorted(groups.items()):
        if not exprs:
            rows.append({"group": name, "n_expressions": 0})
            continue
        counts: Counter[str] = Counter(str(cat) for cat, _img in exprs)
        total = float(sum(counts.values()))
        entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
        top = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k])
        rows.append(
            {
                "group": name,
                "n_expressions": int(total),
                "n_images": len({int(img) for _cat, img in exprs}),
                "n_categories": len(counts),
                "category_entropy_bits": round(entropy, 6),
                "top_category_share": round(max(counts.values()) / total, 6),
                "person_fraction": round(counts.get("person", 0) / total, 6),
                "top_categories": top,
            }
        )
    return rows


def same_category_coverage_rows(
    n_same: Sequence[int],
    *,
    n_valid: Optional[Sequence[int]] = None,
    levels: Sequence[int] = DISTRACTOR_COVERAGE_LEVELS,
    group: str = "all",
) -> List[Dict[str, Any]]:
    """Share of expressions with ``>= k`` same-category distractors (section 14).

    A K = 5 cohort needs ``k = 4`` distractors *in addition to* the target, so
    the ``k = 4`` row is the availability of the hard regime itself.
    """
    arr = np.asarray(list(n_same), dtype=np.int64)
    if arr.size == 0:
        return [{"group": group, "n_rows": 0}]
    valid = np.asarray(n_valid, dtype=np.int64) if n_valid is not None else None
    rows: List[Dict[str, Any]] = []
    for k in levels:
        hits = int(np.count_nonzero(arr >= int(k)))
        lo, hi = wilson_ci(hits, int(arr.size))
        row: Dict[str, Any] = {
            "group": group,
            "distractor_level": int(k),
            "cohort_size": f"K{int(k) + 1}" if int(k) < 10 else "",
            "n_rows": int(arr.size),
            "n_at_least": hits,
            "availability": round(hits / int(arr.size), 6),
            "wilson_low": round(lo, 6),
            "wilson_high": round(hi, 6),
            "mean_same_category": round(float(arr.mean()), 4),
            "median_same_category": round(float(np.median(arr)), 4),
        }
        if valid is not None and valid.size == arr.size:
            # the *distractor pool* only: "at least k proposals survive as usable
            # distractors".  It is not the section 11 K5 availability, which also
            # requires the target proposal itself to exist - conflating the two
            # would report ~1.0 where the gate reads ~0.97.
            row["distractor_pool_availability"] = round(float(np.mean(valid >= int(k))), 6)
        rows.append(row)
    return rows


def target_box_agreement(
    pairs: Sequence[Tuple[Sequence[float], Sequence[float]]],
) -> Dict[str, Any]:
    """IoU(archive ``instances.json`` box, official COCO box) per referred object.

    Section 11 makes a same-domain recall shortfall a *plumbing* suspicion, so
    the first thing to establish is which box the target actually is.  RefCOCO+
    set the precedent (``scripts/run_proposal_audit.py``): the referred target
    box is joined from the dataset's own ``instances.json`` through ``ann_id``,
    while the COCO GT object set used for category assignment comes from the
    official ``instances_train2014.json``.  This function measures whether that
    two-source arrangement is a real difference or an identity, so the choice
    cannot quietly change what "target" means between the two datasets.
    """
    if not pairs:
        return {"n": 0}
    a = xywh_to_xyxy(np.asarray([list(p[0]) for p in pairs], dtype=np.float32))
    b = xywh_to_xyxy(np.asarray([list(p[1]) for p in pairs], dtype=np.float32))
    ious = np.diag(iou_matrix(a, b)).astype(np.float64)
    return {
        "n": int(ious.size),
        "iou": _quantiles(ious),
        "n_below_05": int(np.count_nonzero(ious < 0.5)),
        "n_below_09": int(np.count_nonzero(ious < 0.9)),
        "n_identical": int(np.count_nonzero(ious >= 1.0 - 1e-6)),
        "verdict": (
            "ARCHIVE AND OFFICIAL BOXES IDENTICAL"
            if int(np.count_nonzero(ious >= 1.0 - 1e-6)) == int(ious.size)
            else "BOX SOURCES DIFFER - state which one defines the target"
        ),
    }


def matched_hard_rows(
    rows: Sequence[Any],
    ref_id_by_expr: Mapping[int, int],
    *,
    k: int = PRIMARY_K,
) -> List[Dict[str, Any]]:
    """Section 17: the matched random / same-category cohort, one row per expression.

    A hard-eligible expression is emitted together with the *random* view of the
    very same cell: identical sentence, image, target proposal and scorer seed,
    only the distractor composition differs.  That identity is what makes the
    ``Delta_hard - Delta_random`` contrast of section 23 a matched comparison
    rather than a two-cohort comparison, so both availability flags are written
    out and checked, not assumed.
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not bool(r.same_name_available.get(int(k), False)):
            continue
        out.append(
            {
                "expr_id": int(r.expr_id),
                "ref_id": int(ref_id_by_expr.get(int(r.expr_id), -1)),
                "image_id": int(r.image_id),
                "category": str(r.target_name),
                "target_best_iou": round(float(r.target_best_iou), 6),
                "n_same_category_distractors": int(r.n_same_name_distractors),
                "n_random_distractors": int(r.n_valid_distractors),
                "hard_k5_available": 1,
                "matched_random_k5_available": int(bool(r.available.get(int(k), False))),
            }
        )
    out.sort(key=lambda row: (row["image_id"], row["expr_id"]))
    return out


# ---------------------------------------------------------------------------
# G5 - decision rules (frozen before measurement)
# ---------------------------------------------------------------------------
def engineering_verdict_a10(recall_at_05: float, random_k5_availability: float) -> Dict[str, Any]:
    """Section 11: GO needs *both* recall and K5 availability; <0.85 stops."""
    r = float(recall_at_05)
    a = float(random_k5_availability)
    conditions = {
        f"recall_at_05_ge_{A10_GO_RECALL_MIN:.2f}": r >= A10_GO_RECALL_MIN,
        f"k5_availability_ge_{A10_GO_RANDOM_K5_MIN:.2f}": a >= A10_GO_RANDOM_K5_MIN,
    }
    if not (math.isfinite(r) and math.isfinite(a)):
        verdict = "AUDIT INCOMPLETE"
    elif all(conditions.values()):
        verdict = "EXTERNAL GO"
    elif r < A10_GRAY_RECALL_MIN:
        verdict = "EXTERNAL STOP"
    else:
        verdict = "ENGINEERING GRAY ZONE"
    return {
        "criteria": {
            "go_recall_min": A10_GO_RECALL_MIN,
            "go_random_k5_min": A10_GO_RANDOM_K5_MIN,
            "gray_recall_min": A10_GRAY_RECALL_MIN,
        },
        "measured": {"recall_at_05": r, "k5_random_availability": a},
        "conditions": conditions,
        "verdict": verdict,
        "note": (
            "same COCO image domain as RefCOCO+, so a recall shortfall here points at "
            "annotation / target-definition / split plumbing and must be debugged, not "
            "dodged by changing the generator (instruction sections 10/11)"
        ),
    }


def same_category_verdict_a10(k5_availability: float) -> Dict[str, Any]:
    """Section 14: >=0.60 primary, >=0.75 comfortable, <0.50 kills the branch."""
    a = float(k5_availability)
    if not math.isfinite(a):
        verdict, branch = "AUDIT INCOMPLETE", "unknown"
    elif a >= A10_SAMECAT_GOOD_MIN:
        verdict, branch = "SAME-CATEGORY PRIMARY (COMFORTABLE)", "same_category_primary"
    elif a >= A10_SAMECAT_PRIMARY_MIN:
        verdict, branch = "SAME-CATEGORY PRIMARY", "same_category_primary"
    elif a >= A10_SAMECAT_STOP_MAX:
        verdict, branch = "SAME-CATEGORY DIAGNOSTIC ONLY", "same_category_diagnostic"
    else:
        verdict, branch = "HARD REGIME STOP", "level_free_random_only"
    return {
        "criteria": {
            "primary_min": A10_SAMECAT_PRIMARY_MIN,
            "good_min": A10_SAMECAT_GOOD_MIN,
            "stop_below": A10_SAMECAT_STOP_MAX,
        },
        "measured": {"k5_same_category_availability": a},
        "verdict": verdict,
        "hard_regime_branch": branch,
    }


def hard_cohort_gate_a10(n_expressions: int, n_images: int) -> Dict[str, Any]:
    """Section 16: power floor of the same-category cohort."""
    ok = int(n_expressions) >= A10_HARD_MIN_EXPRESSIONS and int(n_images) >= A10_HARD_MIN_IMAGES
    return {
        "criteria": {"min_expressions": A10_HARD_MIN_EXPRESSIONS, "min_images": A10_HARD_MIN_IMAGES},
        "measured": {"n_expressions": int(n_expressions), "n_images": int(n_images)},
        "verdict": "HARD COHORT OK" if ok else "HARD EXTERNAL UNDERPOWERED",
        "eligible": ok,
    }


def branch_decision_a10(
    *,
    size_decision: str,
    engineering: Mapping[str, Any],
    same_category: Mapping[str, Any],
    hard_cohort: Mapping[str, Any],
) -> Dict[str, Any]:
    """Section 22: exactly one of Branch A / B / C, from the three gates above."""
    recall = float(engineering.get("measured", {}).get("recall_at_05", float("nan")))
    verdict = str(engineering.get("verdict", ""))
    sc_branch = str(same_category.get("hard_regime_branch", ""))
    powered = bool(hard_cohort.get("eligible", False))
    if verdict == "EXTERNAL STOP" or recall < A10_GRAY_RECALL_MIN:
        branch, reason = "C", "proposal recall below the A10 stop line"
    elif sc_branch == "level_free_random_only":
        branch, reason = "C", "same-category hard cohort not constructible"
    elif not powered:
        branch, reason = "C", "hard cohort underpowered"
    elif (
        verdict == "EXTERNAL GO"
        and sc_branch.startswith("same_category")
        and size_decision == SUBSET_STRICT
    ):
        branch, reason = "A", "strict-disjoint subset, GO recall, hard regime constructible"
    else:
        branch, reason = "B", (
            "limited: "
            + "; ".join(
                filter(
                    None,
                    [
                        "" if size_decision == SUBSET_STRICT else f"primary subset is {size_decision or 'none'}",
                        "" if verdict == "EXTERNAL GO" else f"engineering verdict {verdict}",
                        "" if sc_branch == "same_category_primary" else f"same-category branch {sc_branch}",
                    ],
                )
            )
        )
    return {
        "branch": branch,
        "reason": reason,
        "inputs": {
            "size_decision": size_decision,
            "engineering_verdict": verdict,
            "same_category_branch": sc_branch,
            "hard_cohort_verdict": hard_cohort.get("verdict"),
        },
        "next_stage_allowed": branch == "A",
        "report_only": branch != "A",
    }
