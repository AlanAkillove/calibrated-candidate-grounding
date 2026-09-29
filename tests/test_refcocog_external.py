"""Phase 1E / Amendment A10 RefCOCOg external feasibility tests (section 26).

Everything is offline: the released ``refs(umd).p`` shape and the RefCOCO+
development manifests are reproduced by small synthetic fixtures, so the
external data contract is pinned without touching the GPU or the 20 GB of COCO
JPEGs.

The fifteen pre-registered items and where they live:

1.  UMD split parser ......................... ``test_parse_regions_*``
2.  image-id canonicalisation ................ ``test_canonical_image_file_name_*``
3.  overlap exclusion ....................... ``test_overlap_rows_*``
4.  development-disjoint construction ........ ``test_membership_development_disjoint_*``
5.  strict-disjoint construction ............. ``test_membership_strict_*``
6.  zero-overlap assertion ................... ``test_external_subsets_have_zero_image_overlap_*``
7.  frozen ``N = 64`` RPN ................... ``test_frozen_proposal_budget_is_64_and_shared_*``
8.  COCO category assignment ................ ``test_category_is_assigned_only_at_iou_05``
9.  same-category candidates ................ ``test_same_category_*``
10. no target-equivalent proposals .......... ``test_target_equivalents_are_removed_*``
11. matched random / hard identity .......... ``test_matched_hard_rows_*``
12. no RefCOCOg label enters training ........ ``test_only_the_umd_test_split_is_reachable``
                                               ``test_a10_code_fits_nothing_and_forwards_no_model``
13. deterministic manifest .................. ``test_expand_expressions_is_deterministic_*``
14. sample-size gate ........................ ``test_size_gate_*`` / ``test_hard_cohort_gate_*``
15. external branch logic ................... ``test_branch_decision_a10_*``

Extras: the archive-vs-official target-box control (the measurement that killed
the "two box sources" suspicion), the single whitespace tokenizer that keeps the
section 18 length comparison free of a tokeniser confound, and the deterministic
COCO image-fetch layer.
"""

from __future__ import annotations

import ast
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _extra in (_REPO / "src", _REPO / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from ccg.data.manifests import ManifestEntry, ManifestFile, SCHEMA_VERSION  # noqa: E402
from ccg.data.splits import (  # noqa: E402
    SPLIT_TEST,
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_TRAIN,
    SPLIT_VAL,
    SPLIT_VAL_CALIB,
    SPLIT_VAL_SELECT,
)
from ccg.data.types import ProposalBank  # noqa: E402
from ccg.external import coco_images as ci  # noqa: E402
from ccg.external import feasibility as fe  # noqa: E402
from ccg.external import refcocog as rg  # noqa: E402

_DRIVER = _REPO / "scripts" / "run_phase1e_refcocog_feasibility.py"

# ---------------------------------------------------------------------------
# the synthetic cohort: nine RefCOCOg UMD test images, six of them RefCOCO+
# ---------------------------------------------------------------------------
_TEST_IMAGES = tuple(range(1, 10))          # 1..9 all sit in RefCOCOg "test"
_PLUS_SETS: Dict[str, set[int]] = {
    SPLIT_TRAIN: {1, 2},
    SPLIT_VAL_SELECT: {3},
    SPLIT_VAL_CALIB: {4},
    SPLIT_TEST_A: {5},
    SPLIT_TEST_B: {6},
}
_PLUS_SETS["development"] = (
    _PLUS_SETS[SPLIT_TRAIN] | _PLUS_SETS[SPLIT_VAL_SELECT] | _PLUS_SETS[SPLIT_VAL_CALIB]
)
_PLUS_SETS["all_refcoco_plus"] = (
    _PLUS_SETS["development"] | _PLUS_SETS[SPLIT_TEST_A] | _PLUS_SETS[SPLIT_TEST_B]
)

#: what the two definitions must leave behind (section 8)
_STRICT_EXPECTED = {7, 8, 9}
_DEV_EXPECTED = {5, 6, 7, 8, 9}

_CATEGORIES = {17: "person", 18: "bicycle", 20: "bird"}


def _umd_record(
    ref_id: int,
    image_id: int,
    ann_id: int,
    *,
    category_id: int = 17,
    split: str = SPLIT_TEST,
    n_sentences: int = 2,
) -> Dict[str, Any]:
    """One ``refs(umd).p`` record - the layout measured on the real archive.

    The archive stores *no* box field; ``file_name`` carries the referred object
    id as a suffix, which is exactly what :func:`canonical_image_file_name` has
    to undo.
    """
    return {
        "ref_id": ref_id,
        "image_id": image_id,
        "ann_id": ann_id,
        "category_id": category_id,
        "file_name": f"COCO_train2014_{image_id:012d}_{ann_id:06d}.jpg",
        "split": split,
        "sentences": [
            {
                "sent_id": ref_id * 100 + k,
                "raw": f"the bird on the left number {ref_id} {k}",
                "sent": f"the bird on the left number {ref_id} {k}",
                "tokens": ["the", "bird", "on", "the", "left", "number", str(ref_id), str(k)],
            }
            for k in range(n_sentences)
        ],
    }


def _umd_records() -> List[Dict[str, Any]]:
    records = [
        _umd_record(101, 7, 907),                      # strict-disjoint
        _umd_record(102, 8, 908, n_sentences=1),      # strict-disjoint
        _umd_record(103, 9, 909),                     # strict-disjoint
        _umd_record(104, 5, 905),                     # RefCOCO+ testA: dev-only
        _umd_record(105, 6, 906, n_sentences=1),      # RefCOCO+ testB: dev-only
        _umd_record(106, 1, 901),                     # RefCOCO+ train: excluded
        _umd_record(107, 3, 903, split=SPLIT_VAL),    # not an eval split at all
    ]
    return records


def _regions() -> List[rg.RefCOCOGRegion]:
    return rg.parse_regions(_umd_records())


def _boxes_by_ann() -> Dict[int, List[float]]:
    return {901: [0, 0, 40, 40], 903: [5, 5, 30, 30], 905: [10, 10, 20, 20],
            906: [10, 10, 20, 20], 907: [100, 100, 80, 60], 908: [0, 0, 10, 10],
            909: [50, 50, 20, 20]}


def _image_meta() -> Dict[int, Tuple[int, int]]:
    return {i: (320, 240) for i in _TEST_IMAGES} | {3: (200, 200)}


def _membership() -> Dict[int, str]:
    return rg.subset_membership(rg.image_ids_by_split(_regions()), _PLUS_SETS)


# ---------------------------------------------------------------------------
# 1 - UMD split parser
# ---------------------------------------------------------------------------
def test_parse_regions_reads_the_umd_records(tmp_path: Path) -> None:
    payload = _umd_records()
    path = tmp_path / "refs(umd).p"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)

    regions = rg.parse_regions(rg.load_refs_umd(path))
    assert len(regions) == 7
    # sorted by (image, annotation, ref): the audit's row order is a fact of the
    # parser, never of the pickle's dict ordering
    assert [(r.image_id, r.ann_id) for r in regions] == sorted(
        (r.image_id, r.ann_id) for r in regions
    )
    assert {r.split for r in regions} == {SPLIT_TEST, SPLIT_VAL}
    assert regions[0].n_sentences == 2
    # the box field the archive does not carry stays absent from the region
    assert not hasattr(regions[0], "bbox")


def test_parse_regions_rejects_malformed_records() -> None:
    broken = dict(_umd_records()[0])
    del broken["ann_id"]
    with pytest.raises(ValueError, match="lacks fields"):
        rg.parse_regions([broken])

    dup = _umd_records()[0]
    with pytest.raises(ValueError, match="duplicated"):
        rg.parse_regions([dup, dict(dup)])

    empty = _umd_records()[0]
    empty["sentences"] = [{"sent_id": 1, "raw": "   ", "sent": "", "tokens": []}]
    assert rg.parse_regions([empty])[0].n_sentences == 0  # blank text is dropped


def test_load_refs_umd_refuses_the_wrong_shape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rg.load_refs_umd(tmp_path / "absent.p")
    bad = tmp_path / "refs(umd).p"
    with bad.open("wb") as fh:
        pickle.dump({"not": "a list"}, fh)
    with pytest.raises(ValueError, match="expected a list"):
        rg.load_refs_umd(bad)


# ---------------------------------------------------------------------------
# 2 - image-id canonicalisation
# ---------------------------------------------------------------------------
def test_canonical_image_file_name_strips_the_object_suffix() -> None:
    assert (
        rg.canonical_image_file_name("COCO_train2014_000000380440_491042.jpg", 380440)
        == "COCO_train2014_000000380440.jpg"
    )
    # idempotent: the canonical form survives a second pass unchanged
    once = rg.canonical_image_file_name("COCO_train2014_000000000007_000907.jpg", 7)
    assert rg.canonical_image_file_name(once, 7) == once == "COCO_train2014_000000000007.jpg"


def test_canonical_image_file_name_refuses_an_id_mismatch() -> None:
    with pytest.raises(ValueError, match="encodes COCO id"):
        rg.canonical_image_file_name("COCO_train2014_000000000007_000907.jpg", 8)
    with pytest.raises(ValueError, match="not a COCO-style"):
        rg.canonical_image_file_name("n02419459_1.jpg", 7)


# ---------------------------------------------------------------------------
# 3 / 4 / 5 / 6 - overlap, the two subsets, the zero-overlap assertion
# ---------------------------------------------------------------------------
def test_membership_strict_and_development_disjoint() -> None:
    membership = _membership()
    strict = {i for i, name in membership.items() if name == rg.SUBSET_STRICT}
    dev_only = {i for i, name in membership.items() if name == rg.SUBSET_DEV}
    assert strict == _STRICT_EXPECTED
    # membership is a *partition*: strict wins where an image qualifies for both,
    # so the development-disjoint set is only reported through its extra images
    assert dev_only == _DEV_EXPECTED - _STRICT_EXPECTED
    assert strict & dev_only == set()
    assert strict | dev_only == _DEV_EXPECTED
    # the excluded images are absent, not silently labelled strict
    assert not ({1, 2, 3, 4} & set(membership))


def test_overlap_rows_reports_remaining_material() -> None:
    regions = _regions()
    expr_by_image: Dict[int, int] = {}
    refs_by_image: Dict[int, int] = {}
    for reg in regions:
        refs_by_image[reg.image_id] = refs_by_image.get(reg.image_id, 0) + 1
        expr_by_image[reg.image_id] = expr_by_image.get(reg.image_id, 0) + reg.n_sentences
    rows = rg.overlap_rows(rg.image_ids_by_split(regions), _PLUS_SETS, expr_by_image, refs_by_image)
    by_name = {r["refcoco_plus_set"]: r for r in rows}
    assert len(rows) == 7                      # the test split x every RefCOCO+ set
    assert by_name[SPLIT_TRAIN]["overlapping_images"] == 1        # image 1
    # image 3 carries a *val*-split region, so it is not test material at all
    assert by_name["development"]["overlapping_images"] == 1      # image 1
    assert by_name["development"]["non_overlapping_images"] == 5
    assert by_name["all_refcoco_plus"]["non_overlapping_images"] == 3
    # remaining counts only cover the *test* split's surviving images
    assert by_name["all_refcoco_plus"]["refs_remaining"] == 3
    assert by_name["development"]["refs_remaining"] == 5
    assert by_name[SPLIT_TEST_A]["overlapping_images"] == 1
    assert by_name[SPLIT_TEST_B]["overlapping_images"] == 1


def test_external_subsets_have_zero_image_overlap_with_refcoco_plus() -> None:
    """The single claim that makes this "external" at all (section 3/4)."""
    regions = _regions()
    membership = _membership()
    expressions = rg.expand_expressions(
        regions,
        boxes_by_ann=_boxes_by_ann(),
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        splits=(rg.UMD_EVAL_SPLIT,),
        subsets_by_image=membership,
    )
    strict_images = {e.image_id for e in expressions if e.level == 1}
    dev_images = {e.image_id for e in expressions if e.level in (1, 2)}
    assert strict_images == _STRICT_EXPECTED
    assert dev_images == _DEV_EXPECTED
    assert not (strict_images & _PLUS_SETS["all_refcoco_plus"])
    assert not (dev_images & _PLUS_SETS["development"])
    # and the images that *are* RefCOCO+ development images carry no external row
    assert not any(e.level > 0 for e in expressions if e.image_id in _PLUS_SETS["development"])


def test_build_subsets_is_cumulative_and_counts_refs_and_objects() -> None:
    regions = _regions()
    membership = _membership()
    expressions = rg.expand_expressions(
        regions,
        boxes_by_ann=_boxes_by_ann(),
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        splits=(rg.UMD_EVAL_SPLIT,),
        subsets_by_image=membership,
    )
    stats = rg.build_subsets(expressions, regions, membership)
    strict, dev = stats[rg.SUBSET_STRICT], stats[rg.SUBSET_DEV]
    assert strict.n_images == 3 and strict.n_refs == 3 and strict.n_objects == 3
    assert strict.n_expressions == 5                     # 2 + 1 + 2 sentences
    assert dev.n_images == 5 and dev.n_refs == 5 and dev.n_expressions == 8
    assert dev.n_expressions >= strict.n_expressions     # reported as the superset
    # the RefCOCOg test split holds six images here; four are RefCOCO+ development
    # images and six are RefCOCO+ images of some kind
    assert strict.excluded_images == 3 and dev.excluded_images == 1


def test_an_unlisted_image_is_excluded_not_defaulted_to_strict() -> None:
    """Regression: the old ``default=SUBSET_STRICT`` advertised excluded images."""
    regions = _regions()
    expressions = rg.expand_expressions(
        regions,
        boxes_by_ann=_boxes_by_ann(),
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        splits=(rg.UMD_EVAL_SPLIT,),
        subsets_by_image={},                     # nothing was ever classified
    )
    assert expressions and all(e.level == 0 for e in expressions)
    assert all(e.subset == rg.SUBSET_EXCLUDED for e in expressions)
    assert rg.build_subsets(expressions, regions, {})  # no subset material at all
    assert all(s.n_expressions == 0 for s in rg.build_subsets(expressions, regions, {}).values())


# ---------------------------------------------------------------------------
# 14a - the size gate (section 9)
# ---------------------------------------------------------------------------
def _stats(strict_expr: int, strict_img: int, dev_expr: int, dev_img: int) -> Dict[str, rg.SubsetStats]:
    return {
        rg.SUBSET_STRICT: rg.SubsetStats(rg.SUBSET_STRICT, strict_expr, 1, strict_img, 1),
        rg.SUBSET_DEV: rg.SubsetStats(rg.SUBSET_DEV, dev_expr, 1, dev_img, 1),
    }


@pytest.mark.parametrize(
    "sizes,decision,primary",
    [
        ((1500, 500, 1500, 500), "STRICT PRIMARY", rg.SUBSET_STRICT),      # exact floor
        ((2000, 900, 2000, 900), "STRICT PRIMARY", rg.SUBSET_STRICT),
        ((1499, 900, 2000, 700), "EXTERNAL GRAY ZONE", rg.SUBSET_DEV),     # expr short
        ((1500, 499, 2000, 700), "EXTERNAL GRAY ZONE", rg.SUBSET_DEV),     # images short
        ((100, 10, 1999, 800), "REFCOCOG EXTERNAL STOP", ""),              # dev too small
        ((100, 10, 100, 10), "REFCOCOG EXTERNAL STOP", ""),
    ],
)
def test_size_gate(sizes, decision, primary) -> None:
    gate = rg.size_gate(_stats(*sizes))
    assert gate["decision"] == decision
    assert gate["primary_subset"] == primary


def test_size_gate_requires_both_strict_conditions() -> None:
    stats = _stats(10_000, 10, 10_000, 800)         # plenty of expressions, no images
    assert stats[rg.SUBSET_STRICT].to_dict()["meets_strict_criterion"] is False
    assert rg.size_gate(stats)["decision"] == "EXTERNAL GRAY ZONE"


# ---------------------------------------------------------------------------
# 7 - the frozen proposal budget
# ---------------------------------------------------------------------------
def test_frozen_proposal_budget_is_64_and_shared_with_refcoco_plus() -> None:
    assert fe.N_PROPOSALS == 64
    assert rg.N_PROPOSALS is fe.N_PROPOSALS and rg.N_PROPOSALS == 64
    assert rg.PRIMARY_K == fe.PRIMARY_K == 5
    src = _DRIVER.read_text(encoding="utf-8")
    assert "top_n=fe.N_PROPOSALS" in src           # never rpn.py's 128 default
    assert "top_n=128" not in src and "DEFAULT_TOP_N" not in src
    assert "iou_thresh=" not in src               # DEFAULT_IOU_THRESH = 0.5, unforked
    assert "Ks=" not in src                       # the reported Ks stay (5, 10) too
    assert '"n_proposals": fe.N_PROPOSALS' in src


def test_the_development_proposal_bank_is_read_not_rebuilt() -> None:
    """Section 10: the frozen pipeline is *reused*; images RefCOCO+ saw keep their bank."""
    src = _DRIVER.read_text(encoding="utf-8")
    assert "_read_bank_as_proposal(Path(args.frozen_bank), iid)" in src
    assert "write_bank_entry" in src and "finalize_bank" in src
    assert "cache/proposals.h5" in src


# ---------------------------------------------------------------------------
# 8 / 9 / 10 - category assignment, same-category supply, equivalent removal
# ---------------------------------------------------------------------------
_PERSON, _BICYCLE = 17, 18


def _expression(**overrides: Any) -> rg.RefCOCOGExpression:
    base = dict(
        expr_id=1,
        image_id=7,
        level=1,
        tuple_type="spatial",
        target_name="person",
        width=640,
        height=480,
        gt_box=(100.0, 100.0, 180.0, 160.0),        # xywh(100,100,80,60) -> xyxy
        ann_id=907,
        category_id=_PERSON,
        ref_id=101,
        subset=rg.SUBSET_STRICT,
    )
    base.update(overrides)
    return rg.RefCOCOGExpression(**base)


def _bank(boxes: Sequence[Sequence[float]], scores: Sequence[float]) -> ProposalBank:
    return ProposalBank(
        image_id=7,
        boxes=np.asarray(list(boxes), dtype=np.float32),
        objectness=np.asarray(list(scores), dtype=np.float32),
    )


_GT_BOXES = np.asarray(
    [
        [100, 100, 180, 160],   # the referred person
        [300, 100, 380, 160],   # a second person
        [40, 40, 70, 70],       # a bicycle
    ],
    dtype=np.float32,
)
_GT_CODES = np.asarray([_PERSON, _PERSON, _BICYCLE], dtype=np.int64)


def _audit(boxes, scores, expr=None):
    """The frozen audit, but read out at three cardinalities.

    ``Ks`` only chooses which availability flags get *reported*; it is not part of
    the frozen pipeline (the driver keeps the module default ``(5, 10)``), and the
    extra levels exist so the tests can pin the ``k - 1`` distractor arithmetic.
    """
    return fe.audit_expression(
        expr or _expression(),
        _bank(boxes, scores),
        gt_object_boxes_xyxy=_GT_BOXES,
        gt_object_name_codes=_GT_CODES,
        Ks=(2, 4, fe.PRIMARY_K),
    )


def test_category_is_assigned_only_at_iou_05() -> None:
    """A8 rule: proposal -> highest-IoU COCO GT, and only ``IoU >= 0.5`` gets a category."""
    row = _audit(
        [
            [100, 100, 180, 160],   # target, person
            [300, 100, 380, 160],   # person          -> same-category distractor
            [40, 40, 70, 70],       # bicycle         -> other category
            [60, 60, 75, 80],       # IoU with the bicycle GT < 0.5 -> *no* category
            [200, 200, 260, 250],   # overlaps nothing
        ],
        [0.9, 0.8, 0.7, 0.6, 0.5],
    )
    assert row.target_present and row.n_same_name_distractors == 1
    assert row.n_proposals == 5
    assert row.same_name_available[2] is True         # target + 1 same-category
    assert row.same_name_available[fe.PRIMARY_K] is False


def test_target_equivalents_are_removed_before_counting_distractors() -> None:
    """A second proposal that also matches the target is a second correct answer."""
    row = _audit(
        [
            [101, 101, 179, 159],   # IoU ~ 0.94 -> equivalent, must be removed
            [100, 100, 180, 160],   # IoU = 1.0  -> the argmax target
            [110, 110, 190, 170],   # IoU ~ 0.57 -> equivalent, must be removed
            [300, 100, 380, 160],   # the other person
            [40, 40, 70, 70],       # the bicycle
            [5, 5, 8, 8],           # degenerate crop
        ],
        [0.9, 0.95, 0.8, 0.7, 0.6, 0.5],
    )
    assert row.n_equivalent == 3                     # three proposals reach 0.5
    assert row.target_best_iou > 0.99
    assert row.n_valid_distractors == 3              # 6 - target - 2 equivalents
    # the removed equivalents share the target's category; counting them would
    # have turned a 1-distractor image into a 3-distractor "hard" cohort
    assert row.n_same_name_distractors == 1
    assert row.n_invalid_crop == 1
    assert row.available[4] is True and row.available[fe.PRIMARY_K] is False


def test_natural_omission_is_measured_and_not_repaired() -> None:
    row = _audit([[0, 0, 20, 20], [300, 100, 380, 160]], [0.5, 0.4])
    assert not row.target_present and row.target_best_iou < 0.5
    assert not any(row.available.values()) and not any(row.same_name_available.values())
    # ``same_category_counts`` answers "no matchable category" with 0 when the
    # target proposal is absent, so an omitted target cannot donate a hard cohort
    assert row.n_same_name_distractors == 0


def test_same_category_coverage_rows_levels_and_bounds() -> None:
    n_same = [0, 1, 2, 4, 4, 9, 12]
    rows = rg.same_category_coverage_rows(n_same, n_valid=[60] * len(n_same))
    by_level = {int(r["distractor_level"]): r for r in rows}
    assert sorted(by_level) == list(rg.DISTRACTOR_COVERAGE_LEVELS) == [1, 2, 4, 9]
    assert [by_level[k]["n_at_least"] for k in (1, 2, 4, 9)] == [6, 5, 4, 2]
    assert by_level[4]["cohort_size"] == "K5"        # K=5 needs four distractors
    assert by_level[4]["availability"] == pytest.approx(4 / 7)
    for key in ("availability", "wilson_low", "wilson_high"):
        assert by_level[1][key] >= by_level[4][key] >= by_level[9][key]
    # the pool column is *not* the section 11 K5 availability: it ignores whether
    # the target proposal exists, so it must keep its own name
    assert by_level[9]["distractor_pool_availability"] == 1.0
    assert "random_availability" not in by_level[9]
    assert rg.same_category_coverage_rows([])[0]["n_rows"] == 0


# ---------------------------------------------------------------------------
# 11 - the matched random / hard cohort (section 17)
# ---------------------------------------------------------------------------
def _row(expr_id: int, image_id: int, *, hard: bool, random_ok: bool = True,
         same: int = 4) -> fe.ProposalAuditRow:
    return fe.ProposalAuditRow(
        expr_id=expr_id,
        image_id=image_id,
        level=1,
        tuple_type="spatial",
        target_name="person",
        n_proposals=64,
        target_best_iou=0.91,
        target_present=True,
        n_equivalent=0,
        n_valid_distractors=63,
        n_invalid_crop=0,
        n_same_name_distractors=same,
        available={5: random_ok, 10: random_ok},
        same_name_available={5: hard, 10: hard},
    )


def test_matched_hard_rows_preserve_the_cell_identity() -> None:
    rows = [_row(1, 7, hard=True), _row(2, 7, hard=False), _row(3, 8, hard=True)]
    ref_id_by_expr = {1: 101, 2: 102, 3: 103}
    hard = rg.matched_hard_rows(rows, ref_id_by_expr, k=fe.PRIMARY_K)
    assert [h["expr_id"] for h in hard] == [1, 3]         # only hard-eligible cells
    assert [h["image_id"] for h in hard] == [7, 8]
    assert all(h["hard_k5_available"] == 1 for h in hard)
    assert all(h["matched_random_k5_available"] == 1 for h in hard)
    assert [h["ref_id"] for h in hard] == [101, 103]
    assert all(h["n_random_distractors"] == 63 for h in hard)
    assert all(h["n_same_category_distractors"] == 4 for h in hard)


def test_matched_hard_rows_flag_a_broken_random_control() -> None:
    """The random view of a hard cell must exist, or the matched contrast is void."""
    rows = [_row(1, 7, hard=True), _row(2, 8, hard=True, random_ok=False)]
    hard = rg.matched_hard_rows(rows, {1: 101, 2: 102}, k=fe.PRIMARY_K)
    assert [h["matched_random_k5_available"] for h in hard] == [1, 0]
    unknown = rg.matched_hard_rows([_row(9, 7, hard=True)], {}, k=fe.PRIMARY_K)
    assert unknown[0]["ref_id"] == -1                    # never silently 0


# ---------------------------------------------------------------------------
# 12 - no RefCOCOg label can reach a training path
# ---------------------------------------------------------------------------
def test_only_the_umd_test_split_is_reachable(tmp_path: Path) -> None:
    regions = _regions()
    kwargs = dict(
        boxes_by_ann=_boxes_by_ann(),
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        subsets_by_image=_membership(),
    )
    every = rg.expand_expressions(regions, splits=(SPLIT_TEST, SPLIT_VAL, SPLIT_TRAIN), **kwargs)
    test_only = rg.expand_expressions(regions, splits=(rg.UMD_EVAL_SPLIT,), **kwargs)
    assert len(every) > len(test_only)
    assert {e.split for e in test_only} == {SPLIT_TEST}
    assert all(e.image_id != 3 for e in test_only)       # the val-split region is gone


def test_a10_code_fits_nothing_and_forwards_no_model() -> None:
    """Sections 20/21/29: zero new parameters, no calibration, no reliability forward."""
    forbidden = (
        ".fit(", "normalize_fit", "fit_temperature", "IsotonicRegression",
        "LogisticRegression", "GridSearchCV", "apply_frozen", "FrozenSeedModels",
        "train_b3", "extract_features", "paired_bootstrap", "reweight",
        "groundingdino", "GroundingDINO", "maskrcnn",
    )
    for source in (
        (_REPO / "src" / "ccg" / "external" / "refcocog.py").read_text(encoding="utf-8"),
        (_REPO / "src" / "ccg" / "external" / "coco_images.py").read_text(encoding="utf-8"),
        _DRIVER.read_text(encoding="utf-8"),
    ):
        for token in forbidden:
            assert token not in source, f"{token!r} appeared in the A10 sources"


# ---------------------------------------------------------------------------
# 13 - deterministic manifest
# ---------------------------------------------------------------------------
def test_expand_expressions_is_deterministic_under_input_order() -> None:
    regions = _regions()
    kwargs = dict(
        boxes_by_ann=_boxes_by_ann(),
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        splits=(rg.UMD_EVAL_SPLIT,),
        subsets_by_image=_membership(),
    )
    first = rg.expand_expressions(regions, **kwargs)
    second = rg.expand_expressions(list(reversed(regions)), **kwargs)
    assert [e.expr_id for e in first] == [e.expr_id for e in second]
    assert [e.to_row() for e in first] == [e.to_row() for e in second]
    assert len({e.expr_id for e in first}) == len(first)


def test_duplicate_sent_id_is_refused_not_absorbed() -> None:
    regions = _regions()
    kwargs = dict(
        boxes_by_ann=_boxes_by_ann(),
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        splits=(rg.UMD_EVAL_SPLIT,),
    )
    collide = rg.parse_regions(
        [_umd_record(201, 7, 907), _umd_record(202, 8, 908)]  # sent_id 2010/2011 vs 2020
    )
    assert rg.expand_expressions(collide, **kwargs)         # unique, so fine
    poison = _umd_records()
    poison[1]["sentences"][0]["sent_id"] = poison[0]["sentences"][0]["sent_id"]
    with pytest.raises(ValueError, match="not unique"):
        rg.expand_expressions(rg.parse_regions(poison), **kwargs)


def test_a_region_without_a_coco_annotation_is_skipped_and_counted() -> None:
    regions = _regions()
    kwargs = dict(
        boxes_by_ann={907: [100, 100, 80, 60]},            # only one annotation survives
        image_meta=_image_meta(),
        category_names=_CATEGORIES,
        splits=(rg.UMD_EVAL_SPLIT,),
    )
    kept = rg.expand_expressions(regions, **kwargs)
    assert {e.ann_id for e in kept} == {907}
    summary = rg.summarise_dataset(
        regions,
        instances_member={"images": [], "annotations": [], "categories": []},
        provenance={"source": "unit"},
        boxes_by_ann=kwargs["boxes_by_ann"],
        n_unmatched_ann=len(regions) - 1,
    )
    assert summary["target_box_join"]["n_refs_unmatched_in_coco"] == 6


# ---------------------------------------------------------------------------
# the target-box control (the suspicion that had to be killed before G3)
# ---------------------------------------------------------------------------
def test_target_box_agreement_detects_identity_and_unit_change() -> None:
    identical = rg.target_box_agreement([([10, 20, 30, 40], [10, 20, 30, 40])] * 5)
    assert identical["verdict"] == "ARCHIVE AND OFFICIAL BOXES IDENTICAL"
    assert identical["n_identical"] == identical["n"] == 5
    assert identical["n_below_05"] == 0

    # one source storing xywh where the other stores xyxy is the *unit* bug this
    # control catches: the same box (100,100) + 20x20 written as xywh on one side
    # and as xyxy on the other is read as two different boxes, IoU 400 / 14400
    shifted = rg.target_box_agreement([([100, 100, 20, 20], [100, 100, 120, 120])])
    assert shifted["verdict"] == "BOX SOURCES DIFFER - state which one defines the target"
    assert shifted["n_below_05"] == 1 and shifted["n_identical"] == 0
    assert shifted["iou"]["max"] == pytest.approx(400 / 14400, abs=1e-4)
    assert rg.target_box_agreement([]) == {"n": 0}


def test_whitespace_tokens_is_one_tokenizer_for_both_datasets() -> None:
    assert rg.whitespace_tokens("The  Dog ON the LEFT") == ("the", "dog", "on", "the", "left")
    assert rg.whitespace_tokens("") == () and rg.whitespace_tokens(None) == ()
    rows = rg.expression_length_rows(
        {
            "a::short": [("the dog", ("the", "dog"))],
            "b::long": [("the dog on the left", rg.whitespace_tokens("the dog on the left"))],
        }
    )
    by_group = {r["group"]: r for r in rows}
    # the nested shape is the module's; only the driver flattens it for the CSV
    lengths = by_group["a::short"]["tokens_per_expression"]
    assert lengths["mean"] == 2 and lengths["median"] == 2
    assert by_group["b::long"]["tokens_per_expression"]["mean"] == 5
    assert by_group["b::long"]["spatial_token_rate"] == 1.0
    assert by_group["a::short"]["absolute_position_rate"] == 0.0
    assert by_group["b::long"]["absolute_position_rate"] == 1.0
    assert rg.expression_length_rows({"x::empty": []})[0]["n_expressions"] == 0


def test_category_rows_report_entropy_and_person_share() -> None:
    rows = rg.category_rows(
        {
            "plus::hard": [("person", 1), ("person", 1), ("bowl", 2)],
            "rg::hard": [("person", 3), ("chair", 4), ("car", 5)],
        }
    )
    by_group = {r["group"]: r for r in rows}
    assert by_group["plus::hard"]["person_fraction"] == pytest.approx(2 / 3)
    assert by_group["rg::hard"]["person_fraction"] == pytest.approx(1 / 3)
    assert by_group["plus::hard"]["category_entropy_bits"] < by_group["rg::hard"]["category_entropy_bits"]
    assert by_group["plus::hard"]["top_categories"].startswith("person:2")
    assert by_group["rg::hard"]["n_images"] == 3


# ---------------------------------------------------------------------------
# 14b / 15 - the A10 decision rules
# ---------------------------------------------------------------------------
def test_engineering_verdict_a10_is_stricter_than_the_finecops_gate() -> None:
    assert rg.A10_GO_RECALL_MIN == 0.90 and rg.A10_GRAY_RECALL_MIN == 0.85
    # the whole point of section 11: the stop line sits *above* A9's stop line
    assert rg.A10_GRAY_RECALL_MIN > fe.STOP_RECALL_MAX
    assert rg.A10_GO_RANDOM_K5_MIN > fe.GO_AVAILABILITY_MIN

    assert rg.engineering_verdict_a10(0.97, 0.97)["verdict"] == "EXTERNAL GO"
    assert rg.engineering_verdict_a10(0.90, 0.95)["verdict"] == "EXTERNAL GO"      # floor is >=
    assert rg.engineering_verdict_a10(0.87, 0.99)["verdict"] == "ENGINEERING GRAY ZONE"
    assert rg.engineering_verdict_a10(0.95, 0.90)["verdict"] == "ENGINEERING GRAY ZONE"
    assert rg.engineering_verdict_a10(0.849, 0.99)["verdict"] == "EXTERNAL STOP"
    assert rg.engineering_verdict_a10(float("nan"), 0.99)["verdict"] == "AUDIT INCOMPLETE"


def test_same_category_verdict_a10_bands() -> None:
    assert rg.same_category_verdict_a10(0.80)["hard_regime_branch"] == "same_category_primary"
    assert rg.same_category_verdict_a10(0.60)["verdict"] == "SAME-CATEGORY PRIMARY"
    assert rg.same_category_verdict_a10(0.55)["hard_regime_branch"] == "same_category_diagnostic"
    assert rg.same_category_verdict_a10(0.49)["hard_regime_branch"] == "level_free_random_only"


def test_hard_cohort_gate_a10_power_floor() -> None:
    assert rg.hard_cohort_gate_a10(1000, 300)["verdict"] == "HARD COHORT OK"
    assert rg.hard_cohort_gate_a10(999, 10_000)["verdict"] == "HARD EXTERNAL UNDERPOWERED"
    assert rg.hard_cohort_gate_a10(50_000, 299)["eligible"] is False


@pytest.fixture()
def _gates() -> Dict[str, Dict[str, Any]]:
    return {
        "go": rg.engineering_verdict_a10(0.97, 0.97),
        "gray": rg.engineering_verdict_a10(0.87, 0.97),
        "stop": rg.engineering_verdict_a10(0.80, 0.97),
        "same_primary": rg.same_category_verdict_a10(0.65),
        "same_stop": rg.same_category_verdict_a10(0.40),
        "powered": rg.hard_cohort_gate_a10(1890, 755),
        "underpowered": rg.hard_cohort_gate_a10(300, 40),
    }


def test_branch_decision_a10_clean_external(_gates) -> None:
    branch = rg.branch_decision_a10(
        size_decision=rg.SUBSET_STRICT,
        engineering=_gates["go"],
        same_category=_gates["same_primary"],
        hard_cohort=_gates["powered"],
    )
    assert branch["branch"] == "A" and branch["next_stage_allowed"] is True
    assert branch["report_only"] is False


def test_branch_decision_a10_limited_external(_gates) -> None:
    # gray recall band -> B, never A, even with everything else passing
    assert rg.branch_decision_a10(
        size_decision=rg.SUBSET_STRICT,
        engineering=_gates["gray"],
        same_category=_gates["same_primary"],
        hard_cohort=_gates["powered"],
    )["branch"] == "B"
    # the looser development-disjoint primary is also a B (section 22)
    assert rg.branch_decision_a10(
        size_decision=rg.SUBSET_DEV,
        engineering=_gates["go"],
        same_category=_gates["same_primary"],
        hard_cohort=_gates["powered"],
    )["branch"] == "B"


def test_branch_decision_a10_stops(_gates) -> None:
    for kwargs in (
        dict(
            size_decision=rg.SUBSET_STRICT, engineering=_gates["stop"],
            same_category=_gates["same_primary"], hard_cohort=_gates["powered"],
        ),
        dict(
            size_decision=rg.SUBSET_STRICT, engineering=_gates["go"],
            same_category=_gates["same_stop"], hard_cohort=_gates["powered"],
        ),
        dict(
            size_decision=rg.SUBSET_STRICT, engineering=_gates["go"],
            same_category=_gates["same_primary"], hard_cohort=_gates["underpowered"],
        ),
    ):
        branch = rg.branch_decision_a10(**kwargs)
        assert branch["branch"] == "C", kwargs
        assert branch["next_stage_allowed"] is False and branch["report_only"] is True


def test_branch_decision_a10_underpowered_beats_a_clean_recall(_gates) -> None:
    """The three gates are conjunctive: power is not traded against recall."""
    branch = rg.branch_decision_a10(
        size_decision=rg.SUBSET_STRICT,
        engineering=_gates["go"],
        same_category=_gates["same_primary"],
        hard_cohort=_gates["underpowered"],
    )
    assert "underpowered" in branch["reason"]


# ---------------------------------------------------------------------------
# the RefCOCO+ development image sets, read back from the frozen manifests
# ---------------------------------------------------------------------------
def _entry(ref_id: int, image_id: int, split: str, eval_split: str) -> ManifestEntry:
    return ManifestEntry(
        ref_id=ref_id,
        image_id=image_id,
        split=split,
        target_index=0,
        distractor_order=np.asarray([1, 2, 3, 4], dtype=np.int32),
        n_valid_distractors=4,
        n_target_equiv_removed=0,
        n_same_category_available=2,
        eligible={5: True},
        n_same_used_by_K={5: 2},
        hard_fraction_by_K={5: 0.5},
        target_max_iou=0.91,
        eval_split=eval_split,
    )


def test_load_development_image_sets_reads_the_frozen_manifests(tmp_path: Path) -> None:
    files = {
        "random_train.jsonl": [_entry(1, 1, SPLIT_TRAIN, SPLIT_TRAIN)],
        "random_val.jsonl": [
            _entry(2, 3, SPLIT_VAL, SPLIT_VAL_SELECT), _entry(3, 4, SPLIT_VAL, SPLIT_VAL_CALIB)
        ],
        "random_testA.jsonl": [_entry(4, 5, SPLIT_TEST_A, SPLIT_TEST_A)],
        "random_testB.jsonl": [_entry(5, 6, SPLIT_TEST_B, SPLIT_TEST_B)],
    }
    for name, entries in files.items():
        ManifestFile(regime="random", meta={"schema_version": SCHEMA_VERSION, "regime": "random"},
                     entries=entries).save(tmp_path / name)

    sets = rg.load_development_image_sets(tmp_path)
    assert sets[SPLIT_TRAIN] == {1}
    assert sets[SPLIT_VAL_SELECT] == {3} and sets[SPLIT_VAL_CALIB] == {4}
    assert sets["development"] == {1, 3, 4}          # training + tuning + calibration
    assert sets["all_refcoco_plus"] == {1, 3, 4, 5, 6}
    assert sets[SPLIT_TEST_A] == {5}                 # seen by the study, never trained on


def test_load_development_image_sets_needs_the_manifests(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="random_\\*.jsonl"):
        rg.load_development_image_sets(tmp_path)


def test_membership_drops_only_what_the_definitions_exclude() -> None:
    """val_select / val_calib are already inside ``val``; the implementation must
    not have to name them twice (section 8's correctness note)."""
    membership = _membership()
    assert membership.get(3) is None and membership.get(4) is None    # val_select / val_calib
    assert membership.get(5) == rg.SUBSET_DEV and membership.get(6) == rg.SUBSET_DEV


# ---------------------------------------------------------------------------
# the COCO image fetch layer (the only network-facing part of A10)
# ---------------------------------------------------------------------------
def test_local_image_path_uses_the_coco_2014_layout() -> None:
    path = ci.local_image_path("root", "COCO_train2014_000000000007.jpg")
    assert path.as_posix() == "root/train2014/COCO_train2014_000000000007.jpg"
    assert ci.local_image_path("root", Path("x/COCO_val2014_000000000007.jpg")).parent.name == "val2014"
    with pytest.raises(ValueError, match="not a COCO 2014 file name"):
        ci.local_image_path("root", "n02419459_1.jpg")


def test_fetch_report_counts_and_deterministic_failure_order() -> None:
    report = ci.ImageFetchReport(requested=4)
    report.fetched = 2
    report.skipped_existing = 1
    report.failed = [("z.jpg", "http-404"), ("a.jpg", "not-jpeg")]
    payload = report.to_dict()
    assert payload["present"] == 3 and payload["n_failed"] == 2
    assert payload["failure_reasons"] == {"http-404": 1, "not-jpeg": 1}
    names = sorted(n for n, _ in report.failed)          # the artifact sorts
    assert names == ["a.jpg", "z.jpg"]
    assert ci.fetch_coco_images([], "root").requested == 0


def _imported_modules(path: Path) -> set:
    """Module names ``path`` actually imports.

    Prose is not a dependency: A10's docstrings *must* be able to say "FineCops
    A9 stopped here" and "the GQA object-name analogue", so the boundary of
    section 24 is checked on the import graph, not on the text.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_gqa_domain_modules_are_not_reused_for_refcocog() -> None:
    """Section 24's boundary in code form: A10 is COCO-native, not GQA-native."""
    for path in (_REPO / "src" / "ccg" / "external" / "refcocog.py", _DRIVER):
        imported = {name.lower() for name in _imported_modules(path)}
        assert not [name for name in imported if "finecops" in name], path
        assert not [name for name in imported if "gqa" in name], path
    src = (_REPO / "src" / "ccg" / "external" / "refcocog.py").read_text(encoding="utf-8")
    assert "from .feasibility import" in src     # reuse, never re-implement
