"""Validation semantics of the pure data containers in :mod:`ccg.data.types`."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.data.types import (
    HARDNESS_VALUES,
    CandidateSet,
    ProposalBank,
    ReferringExample,
    box_area,
    box_xywh,
)


def make_example(**overrides):
    payload = dict(
        ref_id=7,
        image_id=42,
        text="the person on the left",
        gt_box=np.array([10.0, 20.0, 60.0, 90.0], dtype=np.float32),
        gt_object_id=3,
        split="val",
    )
    payload.update(overrides)
    return ReferringExample(**payload)


def make_bank(n: int = 8, image_id: int = 42, with_gt: bool = True):
    boxes = np.stack(
        [
            np.array([i * 5.0, i * 5.0, i * 5.0 + 20.0, i * 5.0 + 20.0], dtype=np.float32)
            for i in range(n)
        ]
    )
    objectness = np.linspace(0.9, 0.1, n, dtype=np.float32)
    gt_ious = np.linspace(0.0, 1.0, n, dtype=np.float32) if with_gt else None
    gt_assignment = np.arange(n, dtype=np.int64) if with_gt else None
    return ProposalBank(
        image_id=image_id,
        boxes=boxes,
        objectness=objectness,
        gt_ious=gt_ious,
        gt_assignment=gt_assignment,
    )


# ---------------------------------------------------------------------------
# ReferringExample
# ---------------------------------------------------------------------------
def test_referring_example_accepts_and_normalises():
    ex = make_example(gt_box=[1, 2, 5, 9])  # plain list is fine
    assert ex.gt_box.dtype == np.float32
    assert ex.gt_box.shape == (4,)
    assert ex.ref_id == 7 and ex.image_id == 42
    assert ex.area == pytest.approx((5 - 1) * (9 - 2))


@pytest.mark.parametrize(
    "bad_box",
    [
        [1.0, 2.0, 3.0],              # wrong length
        [1.0, 2.0, 1.0, 5.0],         # zero width  (x2 == x1)
        [1.0, 2.0, 0.5, 6.0],         # inverted x
        [1.0, 5.0, 4.0, 2.0],         # inverted y
        [np.nan, 0.0, 5.0, 5.0],      # non-finite
    ],
)
def test_referring_example_rejects_bad_boxes(bad_box):
    with pytest.raises(ValueError):
        make_example(gt_box=bad_box)


def test_referring_example_requires_text_and_ids():
    with pytest.raises(ValueError, match="non-empty"):
        make_example(text="   ")
    with pytest.raises(ValueError, match="text"):
        make_example(text=123)
    with pytest.raises(ValueError, match="ref_id"):
        make_example(ref_id=-1)
    with pytest.raises(ValueError, match="image_id"):
        make_example(image_id=-5)
    with pytest.raises(ValueError, match="whitespace"):
        make_example(split="val calib")


def test_referring_example_object_id_optional():
    assert make_example(gt_object_id=None).gt_object_id is None
    assert make_example(gt_object_id="9").gt_object_id == 9  # coerced to int


# ---------------------------------------------------------------------------
# ProposalBank
# ---------------------------------------------------------------------------
def test_proposal_bank_shape_and_dtype():
    bank = make_bank(n=8)
    assert bank.N == 8
    assert bank.boxes.shape == (8, 4) and bank.boxes.dtype == np.float32
    assert bank.objectness.dtype == np.float32
    assert bank.gt_assignment.dtype == np.int64
    # features are a *reference*, never an in-memory array
    assert bank.feature_ref is None
    assert not hasattr(bank, "proposal_features")


def test_proposal_bank_rejects_mismatched_lengths():
    bank = make_bank(n=6)
    with pytest.raises(ValueError, match="objectness"):
        ProposalBank(image_id=1, boxes=bank.boxes, objectness=np.zeros(5, dtype=np.float32))
    with pytest.raises(ValueError, match="gt_ious"):
        ProposalBank(
            image_id=1,
            boxes=bank.boxes,
            objectness=bank.objectness,
            gt_ious=np.zeros(3, dtype=np.float32),
            gt_assignment=np.zeros(3, dtype=np.int64),
        )


def test_proposal_bank_requires_boxes_matrix():
    with pytest.raises(ValueError, match=r"\[N,4\]"):
        ProposalBank(image_id=1, boxes=np.zeros((4,), dtype=np.float32), objectness=np.zeros(1))
    with pytest.raises(ValueError, match="at least one proposal"):
        ProposalBank(image_id=1, boxes=np.zeros((0, 4), dtype=np.float32), objectness=np.zeros(0))


def test_proposal_bank_gt_fields_come_in_pairs():
    bank = make_bank(n=4)
    with pytest.raises(ValueError, match="together"):
        ProposalBank(image_id=1, boxes=bank.boxes, objectness=bank.objectness, gt_ious=bank.gt_ious)
    with pytest.raises(ValueError, match="together"):
        ProposalBank(
            image_id=1,
            boxes=bank.boxes,
            objectness=bank.objectness,
            gt_assignment=bank.gt_assignment,
        )


def test_proposal_bank_gt_ious_range_and_sentinel():
    bank = make_bank(n=4)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ProposalBank(
            image_id=1,
            boxes=bank.boxes,
            objectness=bank.objectness,
            gt_ious=np.array([0.1, 0.2, 0.3, 1.5], dtype=np.float32),
            gt_assignment=np.arange(4),
        )
    with pytest.raises(ValueError, match="-1"):
        ProposalBank(
            image_id=1,
            boxes=bank.boxes,
            objectness=bank.objectness,
            gt_ious=bank.gt_ious,
            gt_assignment=np.array([-1, -2, 0, 1], dtype=np.int64),
        )
    # -1 is the legal "unassigned" sentinel
    ok = ProposalBank(
        image_id=1,
        boxes=bank.boxes,
        objectness=bank.objectness,
        gt_ious=bank.gt_ious,
        gt_assignment=np.array([-1, 0, 1, 2], dtype=np.int64),
    )
    assert ok.gt_assignment[0] == -1


def test_proposal_bank_top_k_order_is_objectness_descending():
    bank = make_bank(n=10)
    order = bank.top_k_order(3)
    assert order.tolist() == [0, 1, 2]  # objectness decreasing with row index
    assert bank.top_k_order(None).size == 10


def test_proposal_bank_subset_preserves_rows_and_reference():
    bank = ProposalBank(
        image_id=9,
        boxes=make_bank(n=6).boxes,
        objectness=make_bank(n=6).objectness,
        feature_ref="cache://p.h5#/images/9/features",
    )
    sub = bank.subset([4, 0, 2])
    assert sub.N == 3
    np.testing.assert_allclose(sub.boxes[0], bank.boxes[4])
    assert sub.feature_ref is not None and sub.feature_ref != bank.feature_ref
    with pytest.raises(ValueError, match="out of range"):
        bank.subset([0, 99])


# ---------------------------------------------------------------------------
# CandidateSet
# ---------------------------------------------------------------------------
def test_candidate_set_derives_k_and_presence():
    cs = CandidateSet(ref_id=1, candidate_indices=[5, 2, 9, 0], target_index=2)
    assert cs.K == 4
    assert cs.target_present is True
    assert cs.target_proposal_index == 9
    assert cs.distractor_indices.tolist() == [5, 2, 0]
    assert cs.has_proposal(5) and not cs.has_proposal(7)


def test_candidate_set_explicit_k_must_match_length():
    with pytest.raises(ValueError, match="inconsistent with len"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2, 3], K=4, target_index=0)
    with pytest.raises(ValueError, match="inconsistent with len"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2, 3], K=2, target_index=0)


def test_candidate_set_target_present_must_agree_with_target_index():
    with pytest.raises(ValueError, match="target_present"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2, 3], target_index=None, target_present=True)
    with pytest.raises(ValueError, match="target_present"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2, 3], target_index=1, target_present=False)
    # derived when None
    absent = CandidateSet(ref_id=1, candidate_indices=[1, 2, 3], target_index=None)
    assert absent.target_present is False and absent.target_proposal_index is None
    assert absent.distractor_indices.tolist() == [1, 2, 3]


def test_candidate_set_rejects_bad_indices():
    with pytest.raises(ValueError, match="duplicates"):
        CandidateSet(ref_id=1, candidate_indices=[2, 2, 3], target_index=0)
    with pytest.raises(ValueError, match="non-negative"):
        CandidateSet(ref_id=1, candidate_indices=[-1, 3], target_index=0)
    with pytest.raises(ValueError, match="at least one"):
        CandidateSet(ref_id=1, candidate_indices=[], K=0)
    with pytest.raises(ValueError, match="1-D"):
        CandidateSet(ref_id=1, candidate_indices=[[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="outside"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2, 3], target_index=3)


def test_candidate_set_hardness_must_be_known():
    for hardness in HARDNESS_VALUES:
        CandidateSet(ref_id=1, candidate_indices=[1, 2], target_index=0, hardness=hardness)
    with pytest.raises(ValueError, match="hardness"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2], target_index=0, hardness="relation_hard")
    with pytest.raises(ValueError, match="regime"):
        CandidateSet(ref_id=1, candidate_indices=[1, 2], target_index=0, regime="")


def test_candidate_set_indices_are_int64_even_from_floats():
    cs = CandidateSet(ref_id=1, candidate_indices=np.array([1.0, 2.0, 3.0]), target_index=0)
    assert cs.candidate_indices.dtype == np.int64


def test_box_helpers():
    boxes = np.array([[0.0, 0.0, 10.0, 5.0], [1.0, 2.0, 4.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(box_xywh(boxes), [[0, 0, 10, 5], [1, 2, 3, 4]])
    np.testing.assert_allclose(box_area(boxes), [50.0, 12.0])
