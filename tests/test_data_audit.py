"""Hand-computed assertions for the frozen audit statistics (ccg.data.audit).

Every expected number below is derived by hand from the box geometry written
in the comments (areas / intersections / unions), never by running the code
under test first: these tests freeze the contract, they do not snapshot the
implementation.

Frozen conventions exercised here:

* ``IoU == iou_thresh`` counts as *covered* (``>=`` on recall / assignment,
  strictly ``<`` for natural omission, strictly ``>`` for redundancy);
* unknown category is ``-1`` and never matches anything, not even itself;
* a pool that is too small returns ``available=False`` **and** the real
  remaining count - nothing is dropped, padded or invented;
* the empty-input conventions are the ones documented in ``ccg.data.audit``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ccg.data.audit import (
    DEFAULT_KS,
    aggregate_availability,
    assign_gt_category,
    iou_quantiles,
    natural_omission_rate,
    proposal_recall,
    redundancy_stats,
    remaining_candidate_count,
    same_category_counts,
    target_availability,
    wilson_ci,
)


def box(x0: float, y0: float, x1: float, y1: float) -> list:
    """``[x0, y0, x1, y1]`` helper so the geometry stays readable."""
    return [float(x0), float(y0), float(x1), float(y1)]


def spread(n: int, *, step: float = 10.0) -> list:
    """``n`` pairwise-disjoint boxes (geometry irrelevant for counting tests)."""
    return [box(i * step, 0.0, i * step + 5.0, 5.0) for i in range(n)]


# ---------------------------------------------------------------------------
# assign_gt_category: hand-computed IoU table, 2 GT x 3 proposals
# ---------------------------------------------------------------------------
GT_A = box(0, 0, 10, 10)  # area 100, category 7
GT_B = box(20, 0, 30, 10)  # area 100, category 9


def test_default_ks_is_the_protocol_grid():
    assert DEFAULT_KS == (5, 10, 20, 50)


def test_assign_gt_category_picks_best_gt_with_threshold():
    # p0 = x0y0x5y5: inter 25, union 100+25-25 -> IoU 0.25 < 0.5 -> unknown
    # p1 = GT_A itself: IoU 1.0 -> category 7
    # p2 = x20y0x30y5: inter 50, union 100+50-50 -> IoU exactly 0.5 -> accepted
    proposals = [box(0, 0, 5, 5), box(0, 0, 10, 10), box(20, 0, 30, 5)]
    categories, best_ious = assign_gt_category(proposals, [GT_A, GT_B], [7, 9], iou_thresh=0.5)

    assert categories.dtype == np.int64
    assert best_ious.dtype == np.float32
    assert categories.tolist() == [-1, 7, 9]
    np.testing.assert_allclose(best_ious, [0.25, 1.0, 0.5], atol=1e-6)


def test_assign_gt_category_keeps_best_iou_even_when_unknown():
    # a near miss stays auditable: -1 category but the real 0.25 in best_ious
    categories, best_ious = assign_gt_category([box(0, 0, 5, 5)], [GT_A, GT_B], [7, 9])
    assert categories.tolist() == [-1]
    assert best_ious[0] == pytest.approx(0.25, abs=1e-6)


def test_assign_gt_category_empty_proposals_return_empty_arrays():
    categories, best_ious = assign_gt_category(np.zeros((0, 4)), [GT_A], [7])
    assert categories.shape == (0,) and categories.dtype == np.int64
    assert best_ious.shape == (0,) and best_ious.dtype == np.float32


def test_assign_gt_category_without_gt_is_all_unknown():
    categories, best_ious = assign_gt_category([box(0, 0, 4, 4), box(50, 50, 60, 60)], [], [])
    assert categories.tolist() == [-1, -1]
    assert best_ious.tolist() == [0.0, 0.0]


def test_assign_gt_category_misaligned_inputs_raise():
    with pytest.raises(ValueError):
        assign_gt_category([GT_A], [GT_A, GT_B], [7])  # 2 boxes, 1 category
    with pytest.raises(ValueError):
        assign_gt_category([GT_A], [GT_A], [7], iou_thresh=1.5)


# ---------------------------------------------------------------------------
# proposal_recall: 5 proposals / 2 GT objects, threshold boundary
# ---------------------------------------------------------------------------
def test_proposal_recall_threshold_boundary_is_inclusive():
    gts = [box(0, 0, 10, 10), box(100, 100, 110, 110)]
    covered = [
        box(0, 0, 10, 10),  # GT1: IoU 1.0
        box(100, 100, 110, 105),  # GT2: inter 50, union 100 -> IoU exactly 0.5
        box(200, 200, 210, 210),  # far away
        box(300, 300, 310, 310),
        box(400, 400, 410, 410),
    ]
    # IoU == 0.5 counts as covered
    assert proposal_recall(covered, gts, 0.5) == 1.0
    # one ulp above 0.5: GT2 slips below the boundary -> 1 of 2 objects covered
    assert proposal_recall(covered, gts, 0.5000001) == 0.5

    partial = list(covered)
    partial[1] = box(100, 100, 110, 104)  # GT2: inter 40, union 100 -> IoU 0.4
    assert proposal_recall(partial, gts, 0.5) == 0.5


def test_proposal_recall_empty_conventions():
    gt = box(0, 0, 10, 10)
    assert proposal_recall([box(0, 0, 1, 1)], [], 0.5) == 1.0  # no GT: vacuous
    assert proposal_recall([], [gt], 0.5) == 0.0  # GT present, no proposals
    assert proposal_recall([], [], 0.5) == 1.0


# ---------------------------------------------------------------------------
# availability / remaining count: 5 proposals, hand-computed P(>= K-1)
# ---------------------------------------------------------------------------
def test_target_availability_five_proposals_needs_four_distractors():
    proposals = spread(5)
    available = target_availability(proposals, target_idx=1, to_remove_idx=[])
    # 5 proposals, target 1 removed -> 4 valid distractors
    # K=5 needs >= 4 -> True; K=10/20/50 need >= 9/19/49 -> False
    assert available == {5: True, 10: False, 20: False, 50: False}
    assert all(isinstance(flag, bool) for flag in available.values())
    assert remaining_candidate_count(proposals, 1, []) == 4


def test_target_availability_twenty_one_proposals_supports_k20():
    proposals = spread(21)
    available = target_availability(proposals, 0, [])
    # 21 - 1 target = 20 valid: K=20 needs 19 -> True, K=50 needs 49 -> False
    assert available[5] is True
    assert available[10] is True
    assert available[20] is True
    assert available[50] is False
    assert remaining_candidate_count(proposals, 0, []) == 20


def test_no_silent_filtering_small_pool_returns_false_and_real_count():
    proposals = spread(3)
    available = target_availability(proposals, 0, [1])
    # pool 3, target 0 and its equivalent 1 removed -> 1 real distractor
    assert available[5] is False
    # the measured count is 1 - not padded up to K=5's 4, not dropped
    assert remaining_candidate_count(proposals, 0, [1]) == 1


def test_to_remove_is_deduplicated_and_may_contain_the_target():
    proposals = spread(5)
    assert remaining_candidate_count(proposals, 0, [0, 0, 2]) == 3  # dupes collapse
    assert remaining_candidate_count(proposals, 1, [1, 2]) == 3  # target may appear twice
    assert remaining_candidate_count(proposals, None, [0, 2]) == 3  # natural miss


def test_target_availability_natural_miss_counts_everything_else():
    proposals = spread(6)
    available = target_availability(proposals, None, [])
    # no target to reserve a slot for: all 6 count -> P(K=5) True, P(K=10) False
    assert available == {5: True, 10: False, 20: False, 50: False}


def test_availability_custom_ks_and_validation():
    proposals = spread(5)
    assert target_availability(proposals, 0, [], Ks=(2, 3)) == {2: True, 3: True}
    with pytest.raises(ValueError):
        target_availability(proposals, 0, [], Ks=(0,))
    with pytest.raises(ValueError):
        target_availability(proposals, 5, [])  # target out of range
    with pytest.raises(ValueError):
        remaining_candidate_count(proposals, 0, [7])  # removal out of range


def test_aggregate_availability_rates():
    assert aggregate_availability([{5: True, 10: True}, {5: True, 10: False}], Ks=(5, 10)) == {
        5: 1.0,
        10: 0.5,
    }


def test_aggregate_availability_missing_key_counts_as_false():
    # capacity that was never measured must not be fabricated
    sparse = [{5: True}, {5: True, 10: True}]
    assert aggregate_availability(sparse, Ks=(5, 10)) == {5: 1.0, 10: 0.5}


def test_aggregate_availability_accepts_string_keys():
    # a JSON round-trip stringifies the int keys; the rate must survive it
    assert aggregate_availability([{"5": True, "10": False}], Ks=(5, 10)) == {5: 1.0, 10: 0.0}


def test_aggregate_availability_empty_population_is_zero():
    assert aggregate_availability([], Ks=(5, 10)) == {5: 0.0, 10: 0.0}


# ---------------------------------------------------------------------------
# redundancy_stats: three identical boxes -> all 3 pairs at IoU 1.0
# ---------------------------------------------------------------------------
def test_redundancy_stats_identical_boxes():
    a = box(0, 0, 2, 2)
    stats = redundancy_stats([a, list(a), list(a)])
    assert stats["n_pairs"] == 3
    assert stats["frac_pairs_gt_mid"] == 1.0
    assert stats["frac_pairs_gt_high"] == 1.0


def test_redundancy_stats_one_duplicate_of_three_pairs():
    a = box(0, 0, 2, 2)
    far = box(100, 100, 102, 102)
    # pair IoUs: (a,a)=1.0, (a,far)=0.0, (a,far)=0.0 -> 1 of 3 pairs redundant
    stats = redundancy_stats([a, list(a), far])
    assert stats["n_pairs"] == 3
    assert stats["frac_pairs_gt_mid"] == pytest.approx(1.0 / 3.0)
    assert stats["frac_pairs_gt_high"] == pytest.approx(1.0 / 3.0)


def test_redundancy_stats_threshold_comparison_is_strict():
    a = box(0, 0, 2, 2)  # area 4
    b = box(0, 0, 2, 1)  # area 2, inter 2, union 4 -> IoU exactly 0.5
    stats = redundancy_stats([a, b], thresh_mid=0.5)
    assert stats["frac_pairs_gt_mid"] == 0.0  # 0.5 > 0.5 is False


def test_redundancy_stats_fewer_than_two_boxes():
    stats = redundancy_stats([box(0, 0, 2, 2)])
    assert stats == {"frac_pairs_gt_mid": 0.0, "frac_pairs_gt_high": 0.0, "n_pairs": 0}
    assert redundancy_stats(np.zeros((0, 4)))["n_pairs"] == 0


# ---------------------------------------------------------------------------
# same_category_counts: unknown (-1) is never matchable
# ---------------------------------------------------------------------------
def test_same_category_counts_excludes_removed_and_unknown():
    categories = [3, 3, 3, -1, 5]
    # candidates with a removal: idx2 (3 == 3 valid); idx3 is -1, idx4 is 5
    assert same_category_counts(categories, target_idx=0, to_remove_idx=[1]) == 1
    assert same_category_counts(categories, target_idx=0, to_remove_idx=[2]) == 1


def test_same_category_counts_never_matches_unknown():
    # an unknown target has no matchable category, even against other unknowns
    assert same_category_counts([-1, 3, 3], target_idx=0, to_remove_idx=[]) == 0
    assert same_category_counts([-1, -1, -1], target_idx=1, to_remove_idx=[]) == 0
    # natural miss: no target -> no category to match
    assert same_category_counts([3, 3, 3], target_idx=None, to_remove_idx=[]) == 0


def test_same_category_counts_all_candidates_removed():
    assert same_category_counts([3, 3, 3], target_idx=0, to_remove_idx=[1, 2]) == 0
    count = same_category_counts([3, 3, 3, 3], target_idx=0, to_remove_idx=[1])
    assert isinstance(count, int) and count == 2


# ---------------------------------------------------------------------------
# natural_omission_rate: strictly below the threshold
# ---------------------------------------------------------------------------
def test_natural_omission_rate_strictly_below_threshold():
    assert natural_omission_rate([0.2, 0.5, 0.9], iou_thresh=0.5) == pytest.approx(1.0 / 3.0)
    # IoU exactly at the threshold is *covered*, not omitted
    assert natural_omission_rate([0.5], iou_thresh=0.5) == 0.0
    assert natural_omission_rate([0.4999], iou_thresh=0.5) == 1.0
    assert natural_omission_rate([0.2, 0.5, 0.9], iou_thresh=0.9) == pytest.approx(2.0 / 3.0)


def test_natural_omission_rate_empty_is_zero():
    assert natural_omission_rate([]) == 0.0
    assert natural_omission_rate([0.0], iou_thresh=0.0) == 0.0


# ---------------------------------------------------------------------------
# wilson_ci: textbook 95% intervals (hand-checked)
# ---------------------------------------------------------------------------
def test_wilson_ci_known_values():
    # k=0, n=10: upper = 2 * (z^2 / 2n) / (1 + z^2/n) = 0.38416 / 1.38416
    low, high = wilson_ci(0, 10)
    assert low == 0.0
    assert high == pytest.approx(0.2775, abs=1e-3)

    low, high = wilson_ci(10, 10)  # mirror image of k=0
    assert low == pytest.approx(0.7225, abs=1e-3)
    assert high == 1.0  # exact endpoint at k = n (no float round-off)

    low, high = wilson_ci(5, 10)  # symmetric around 0.5
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)


def test_wilson_ci_degenerate_and_invalid():
    assert wilson_ci(0, 0) == (0.0, 1.0)  # no observations, no information
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(1, 10, z=0.0)


# ---------------------------------------------------------------------------
# iou_quantiles: linear interpolation, keyed by q
# ---------------------------------------------------------------------------
def test_iou_quantiles_hand_computed():
    result = iou_quantiles([0.0, 0.5, 1.0])
    assert result == pytest.approx({0.25: 0.25, 0.5: 0.5, 0.75: 0.75})
    assert iou_quantiles([0.0, 1.0, 2.0, 3.0], qs=(0.5,)) == pytest.approx({0.5: 1.5})


def test_iou_quantiles_keys_and_empty_sample():
    result = iou_quantiles([0.1, 0.2])
    assert set(result) == {0.25, 0.5, 0.75}  # keys follow the requested q values
    empty = iou_quantiles([])
    assert set(empty) == {0.25, 0.5, 0.75}
    assert all(math.isnan(value) for value in empty.values())


def test_iou_quantiles_invalid_q_raises():
    with pytest.raises(ValueError):
        iou_quantiles([0.1], qs=(1.5,))
    with pytest.raises(ValueError):
        iou_quantiles([0.1], qs=(-0.1,))
