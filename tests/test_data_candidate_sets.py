"""Nested / hard / omitted candidate-set construction (pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.data.candidate_sets import (
    DEFAULT_KS,
    CandidateAvailability,
    CandidateShortage,
    assert_nested,
    build_clip_hard_sets,
    build_nested_random_sets,
    build_ranked_nested_sets,
    build_same_category_hard_sets,
    cosine_similarity,
    select_clip_hard_negatives,
    select_same_category_hard_negatives,
    synthetic_omit,
)
from ccg.data.types import CandidateSet, ProposalBank

POOL = 60


def big_pool(n: int = POOL, target: int = 7):
    return [i for i in range(n) if i != target]


def small_bank(n: int = 10):
    boxes = np.stack(
        [np.array([i * 4.0, i * 4.0, i * 4.0 + 16.0, i * 4.0 + 16.0]) for i in range(n)]
    ).astype(np.float32)
    return ProposalBank(
        image_id=1,
        boxes=boxes,
        objectness=np.linspace(1.0, 0.0, n, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# nesting
# ---------------------------------------------------------------------------
def test_nested_random_sets_are_nested_and_keep_the_target():
    target = 7
    sets = build_nested_random_sets(target, big_pool(), DEFAULT_KS, 0)
    assert set(sets) == set(DEFAULT_KS)
    built = sets.built()
    assert sorted(built) == list(DEFAULT_KS)
    assert sets.max_built_K == 50

    previous = None
    for k in DEFAULT_KS:
        cs = built[k]
        assert cs.K == k and cs.candidate_indices.size == k
        assert cs.target_present is True
        assert cs.target_proposal_index == target, "target must be constant across K"
        assert cs.hardness == "random"
        assert 0 <= cs.target_index < k
        if previous is not None:
            assert set(previous.candidate_indices) <= set(cs.candidate_indices)
        previous = cs
    assert_nested(sets, target_proposal_idx=target)  # the library's own invariant check


def test_nested_random_sets_are_deterministic_per_seed():
    a = build_nested_random_sets(3, big_pool(), DEFAULT_KS, 42)
    b = build_nested_random_sets(3, big_pool(), DEFAULT_KS, 42)
    for k in DEFAULT_KS:
        np.testing.assert_array_equal(a[k].candidate_indices, b[k].candidate_indices)


def test_target_position_is_shuffled_but_resolvable():
    sets = build_nested_random_sets(3, big_pool(), (10,), 1)
    cs = sets[10]
    assert cs.candidate_indices[cs.target_index] == 3
    fixed = build_nested_random_sets(3, big_pool(), (10,), 1, shuffle_positions=False)
    assert fixed[10].target_index == 0
    assert fixed[10].candidate_indices[0] == 3


def test_pool_deduplication_and_target_removal():
    # the target appearing in the pool is dropped defensively, repeats too
    pool = [1, 2, 2, 5, 7, 7]
    sets = build_nested_random_sets(7, pool, (4,), 0, shuffle_positions=False)
    cs = sets[4]
    assert cs.candidate_indices[0] == 7 and cs.target_index == 0
    assert set(cs.candidate_indices.tolist()) == {7, 1, 2, 5}
    # one distractor short and the request becomes a reported shortage
    shortage = build_nested_random_sets(7, pool, (5,), 0)
    assert shortage[5] is None and shortage.availability.missing[5] == 1


# ---------------------------------------------------------------------------
# availability reporting (never a silent filter)
# ---------------------------------------------------------------------------
def test_short_pool_reports_missing_k_instead_of_dropping_it():
    target = 0
    pool = list(range(1, 7))  # 6 distractors -> at most K = 7
    report = CandidateAvailability(requested_Ks=DEFAULT_KS)
    sets = build_nested_random_sets(target, pool, DEFAULT_KS, 0, availability=report)

    assert sorted(sets) == list(DEFAULT_KS), "requested K keys must survive"
    assert sets[5] is not None and sets[5].K == 5
    assert sets[10] is None and sets[20] is None and sets[50] is None
    assert sets.max_built_K == 5

    payload = sets.availability.to_dict()
    assert payload["num_shortages"] == 3
    assert payload["missing"]["10"] == 1 and payload["missing"]["50"] == 1
    assert payload["availability"]["5"] == pytest.approx(1.0)
    assert payload["example_shortages"][0]["max_possible_K"] == 7
    assert payload["example_shortages"][0]["pool_size"] == 6
    assert sets.availability is report  # the caller's object is filled in place


def test_availability_report_can_be_aggregated_across_examples():
    report = CandidateAvailability(requested_Ks=(5, 10))
    build_nested_random_sets(0, list(range(1, 30)), (5, 10), 0, availability=report)
    build_nested_random_sets(0, list(range(1, 4)), (5, 10), 0, availability=report)
    assert report.num_requests == 2
    assert report.counts[5] == 1 and report.missing[5] == 1
    assert report.counts[10] == 1 and report.missing[10] == 1
    assert report.availability[5] == pytest.approx(0.5)
    payload = report.to_dict()
    assert payload["counts"]["5"] == 1 and payload["missing"]["10"] == 1
    assert payload["num_shortages"] == 2


def test_availability_merge_requires_the_same_grid():
    a = CandidateAvailability(requested_Ks=(5, 10))
    b = CandidateAvailability(requested_Ks=(5, 20))
    with pytest.raises(ValueError, match="requested_Ks"):
        a.merge(b)


def test_shortage_record_is_dataclass_shaped():
    shortage = CandidateShortage(ref_id=4, requested_K=50, pool_size=9, max_possible_K=10)
    assert shortage.to_dict() == {
        "ref_id": 4,
        "requested_K": 50,
        "pool_size": 9,
        "max_possible_K": 10,
    }


def test_ranked_builder_rejects_bad_arguments():
    with pytest.raises(ValueError, match="Ks must all be >= 2"):
        build_ranked_nested_sets(0, [1, 2, 3], (1,))
    with pytest.raises(ValueError, match="exclude the target"):
        build_ranked_nested_sets(0, [0, 1, 2], (3,))
    with pytest.raises(ValueError, match="non-negative"):
        build_ranked_nested_sets(-1, [1, 2], (2,))


# ---------------------------------------------------------------------------
# same-category hard negatives
# ---------------------------------------------------------------------------
def make_gt_bank():
    n = 7
    boxes = np.stack(
        [np.array([i * 3.0, i * 3.0, i * 3.0 + 12.0, i * 3.0 + 12.0]) for i in range(n)]
    ).astype(np.float32)
    gt_assignment = np.array([10, 11, 11, 20, 30, -1, 11], dtype=np.int64)
    gt_ious = np.array([0.2, 0.9, 0.5, 0.8, 0.7, 1.0, 0.5], dtype=np.float32)
    return ProposalBank(
        image_id=5,
        boxes=boxes,
        objectness=np.full(n, 0.5, dtype=np.float32),
        gt_ious=gt_ious,
        gt_assignment=gt_assignment,
    )


CATEGORIES = {10: 1, 11: 1, 20: 2, 30: 3}


def test_same_category_hard_negatives_rank_by_iou_deterministically():
    bank = make_gt_bank()
    ranked = select_same_category_hard_negatives(
        bank, 10, CATEGORIES, target_proposal_idx=0
    )
    # eligible = same category (1) and not the target object -> rows 1, 2, 6
    # sorted by IoU desc, ties by row asc (rows 2 and 6 both have 0.5)
    assert ranked.tolist() == [1, 2, 6]
    assert select_same_category_hard_negatives(bank, 10, CATEGORIES, k=2).tolist() == [1, 2]


def test_same_category_hard_negatives_min_iou_and_object_array_form():
    # array-indexed categories: object id is the row of the array, so this bank
    # uses small object ids (0..3) instead of the COCO-scale ids of make_gt_bank
    n = 6
    boxes = np.stack(
        [np.array([i * 3.0, i * 3.0, i * 3.0 + 12.0, i * 3.0 + 12.0]) for i in range(n)]
    ).astype(np.float32)
    bank = ProposalBank(
        image_id=6,
        boxes=boxes,
        objectness=np.full(n, 0.5, dtype=np.float32),
        gt_ious=np.array([0.2, 0.9, 0.5, 0.8, 0.7, 1.0], dtype=np.float32),
        gt_assignment=np.array([0, 1, 1, 2, 3, -1], dtype=np.int64),
    )
    as_array = np.array([7, 7, 8, 9], dtype=np.int64)
    ranked = select_same_category_hard_negatives(bank, 0, as_array, min_iou=0.6)
    assert ranked.tolist() == [1]
    assert select_same_category_hard_negatives(bank, 0, as_array).tolist() == [1, 2]
    # nothing passes a 0.95 filter: an honest empty answer, not an exception
    assert select_same_category_hard_negatives(bank, 0, as_array, min_iou=0.95).size == 0
    # an object id outside the array is an error, not a silent "unknown category"
    with pytest.raises(ValueError, match="out of range"):
        select_same_category_hard_negatives(bank, 42, as_array)


def test_same_category_hard_negatives_require_metadata():
    bank = small_bank()
    with pytest.raises(ValueError, match="gt_ious"):
        select_same_category_hard_negatives(bank, 10, CATEGORIES)
    with pytest.raises(ValueError, match="unknown"):
        select_same_category_hard_negatives(make_gt_bank(), 999, CATEGORIES)


def test_same_category_hard_sets_are_nested_and_tagged():
    bank = make_gt_bank()
    sets = build_same_category_hard_sets(
        bank,
        target_proposal_idx=0,
        target_object_id=10,
        object_categories=CATEGORIES,
        Ks=(2, 4, 7),
        rng=0,
        image_id=bank.image_id,
    )
    assert sets[7] is not None
    for k in (2, 4, 7):
        assert sets[k].hardness == "same_category_hard"
        assert sets[k].target_proposal_index == 0
    assert_nested(sets, target_proposal_idx=0)
    # the hard prefix (rows 1, 2, 6) is inside every set that has room for it
    assert set([1, 2]).issubset(set(sets[4].candidate_indices.tolist()))
    assert any("same-category negatives available" in note for note in sets.availability.notes)


# ---------------------------------------------------------------------------
# CLIP-hard negatives
# ---------------------------------------------------------------------------
def make_features():
    # row i has first coordinate proportional to its similarity with the query
    rows = [
        [1.0, 0.0],
        [0.9, 0.1],
        [0.8, 0.2],
        [0.1, 0.9],
        [0.0, 0.0],   # zero-norm crop -> treated as uninformative (sim 0)
        [-0.5, 0.5],
    ]
    return np.asarray(rows, dtype=np.float32)


def test_cosine_similarity_basics():
    sims = cosine_similarity(np.array([1.0, 0.0]), make_features())
    assert sims[0] == pytest.approx(1.0)
    assert sims[4] == pytest.approx(0.0, abs=1e-6)
    assert sims[5] < 0.0
    with pytest.raises(ValueError, match="2-D"):
        cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="dim mismatch"):
        cosine_similarity(np.array([1.0, 0.0, 0.0]), make_features())
    with pytest.raises(ValueError, match="zero norm"):
        cosine_similarity(np.array([0.0, 0.0]), make_features())


def test_clip_hard_negatives_exclude_and_order():
    features = make_features()
    ranked = select_clip_hard_negatives(np.array([1.0, 0.0]), features, exclude_idx=[0])
    assert 0 not in ranked.tolist()
    assert ranked.tolist() == [1, 2, 3, 4, 5]
    truncated = select_clip_hard_negatives(np.array([1.0, 0.0]), features, exclude_idx=(0,), k=2)
    assert truncated.tolist() == [1, 2]
    filtered = select_clip_hard_negatives(
        np.array([1.0, 0.0]), features, exclude_idx=(0,), min_similarity=0.5
    )
    assert filtered.tolist() == [1, 2]
    with pytest.raises(ValueError, match="out-of-range"):
        select_clip_hard_negatives(np.array([1.0, 0.0]), features, exclude_idx=[99])


def test_clip_hard_sets_are_nested_with_constant_target():
    features = make_features()
    sets = build_clip_hard_sets(
        ref_id=11,
        target_proposal_idx=0,
        query_feature=np.array([1.0, 0.0]),
        proposal_features=features,
        Ks=(2, 3, 6),
        rng=0,
        exclude_idx=(),
    )
    assert_nested(sets, target_proposal_idx=0)
    for k in (2, 3, 6):
        assert sets[k].hardness == "clip_hard"
        assert sets[k].ref_id == 11
    # hardest negatives come first once the target is out of the way
    assert set(sets[3].candidate_indices.tolist()) >= {0, 1, 2}


# ---------------------------------------------------------------------------
# synthetic omission
# ---------------------------------------------------------------------------
def test_synthetic_omit_removes_target_only():
    cs = CandidateSet(ref_id=1, candidate_indices=[4, 9, 2, 7], target_index=1, hardness="clip_hard")
    omitted = synthetic_omit(cs)
    assert omitted.target_index is None
    assert omitted.target_present is False
    assert omitted.K == 3 == len(omitted.candidate_indices)
    assert 9 not in omitted.candidate_indices.tolist()
    assert omitted.hardness == "clip_hard", "difficulty of the distractors is unchanged"
    assert omitted.regime == "random_omit"
    assert omitted.ref_id == 1

    explicit = synthetic_omit(cs, regime="synthetic_omit")
    assert explicit.regime == "synthetic_omit"


def test_synthetic_omit_rejects_double_omission_and_empty_sets():
    absent = CandidateSet(ref_id=2, candidate_indices=[1, 2, 3], target_index=None)
    with pytest.raises(ValueError, match="already has no target"):
        synthetic_omit(absent)
    singleton = CandidateSet(ref_id=3, candidate_indices=[5], target_index=0)
    with pytest.raises(ValueError, match="empty candidate set"):
        synthetic_omit(singleton)


def test_omit_of_nested_family_keeps_nesting_of_distractors():
    sets = build_nested_random_sets(7, big_pool(), (5, 10), 0)
    omitted = {k: synthetic_omit(cs) for k, cs in sets.built().items()}
    small = set(omitted[5].candidate_indices.tolist())
    large = set(omitted[10].candidate_indices.tolist())
    assert small <= large
    assert all(cs.target_present is False for cs in omitted.values())


# ---------------------------------------------------------------------------
# the invariant checker itself
# ---------------------------------------------------------------------------
def test_assert_nested_detects_violations():
    ok = {
        2: CandidateSet(ref_id=1, candidate_indices=[7, 1], target_index=0),
        3: CandidateSet(ref_id=1, candidate_indices=[7, 1, 2], target_index=0),
    }
    assert_nested(ok, target_proposal_idx=7)

    broken = {
        2: CandidateSet(ref_id=1, candidate_indices=[7, 1], target_index=0),
        3: CandidateSet(ref_id=1, candidate_indices=[7, 2, 3], target_index=0),
    }
    with pytest.raises(AssertionError, match="subset"):
        assert_nested(broken)

    moved_target = {
        2: CandidateSet(ref_id=1, candidate_indices=[7, 1], target_index=0),
        3: CandidateSet(ref_id=1, candidate_indices=[8, 7, 1], target_index=1),
    }
    with pytest.raises(AssertionError, match="target changed"):
        assert_nested(moved_target, target_proposal_idx=8)
