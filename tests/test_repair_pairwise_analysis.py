"""Scientific invariants for the new frozen-input pair accounting."""
import itertools

import numpy as np
import pytest

from ccg.metrics.discrimination import auroc_correct
from ccg.repairs.mechanism import four_corner_paths
from ccg.repairs.pairwise_analysis import FrozenPairAccounting


def test_weighted_accounting_matches_explicit_cluster_duplicates_and_ties():
    p5 = np.array([.8, .8, .1, .5, .3, .3])
    p50 = np.array([.5, .3, .3, .8, .1, .8])
    r5 = np.array([1, 1, 0, 1, 1, 0])
    r50 = np.array([1, 0, 0, 1, 0, 0])
    cache = FrozenPairAccounting(p5, r5, p50, r50)
    weights = np.array([2, 1, 3, 0, 2, 1])
    draw = np.repeat(np.arange(6), weights)
    actual = cache.evaluate(weights)
    expected = four_corner_paths(p5[draw], r5[draw], p50[draw], r50[draw])
    for name in ("A00", "A10", "A01", "A11", "interaction_I"):
        assert actual[name] == pytest.approx(expected[name], abs=1e-12)
    for tag in ("p5", "p50"):
        assert actual[f"label_path_{tag}"] == pytest.approx(
            actual[f"new_comparison_{tag}"] + actual[f"removed_comparison_{tag}"], abs=1e-12)
    assert actual["C0"] == pytest.approx(actual["C0_SE"] + actual["C0_FE"], abs=1e-12)
    assert actual["C1"] == pytest.approx(actual["C1_SE"] + actual["C1_SF"], abs=1e-12)


def test_pair_formulas_exhaust_all_three_group_assignments_with_ties():
    scores = np.array([0., 1., 1., 2., 0., 3.])
    cases = 0
    for assignment in itertools.product(range(3), repeat=6):
        groups = np.asarray(assignment)
        if np.unique(groups).size != 3:
            continue
        r5 = (groups != 2).astype(int)  # S,F positive; E negative
        r50 = (groups == 0).astype(int)
        result = FrozenPairAccounting(scores, r5, scores[::-1], r50).evaluate()
        expected = four_corner_paths(scores, r5, scores[::-1], r50)
        for name in ("A00", "A10", "A01", "A11", "interaction_I"):
            assert result[name] == pytest.approx(expected[name], abs=1e-12)
        cases += 1
    assert cases == 540


def test_absent_flip_group_keeps_corners_defined_without_inventing_pair_auc():
    result = FrozenPairAccounting([.8, .2], [1, 0], [.7, .3], [1, 0]).evaluate()
    assert np.isnan(result["U_SF_p5"])
    assert result["A00"] == result["A10"] == 1
    assert result["label_path_p5"] == result["new_comparison_p5"] == 0
    assert result["removed_comparison_p5"] == 0


def test_single_class_is_undefined_and_gained_correct_is_rejected():
    result = FrozenPairAccounting([.8, .2], [1, 1], [.7, .3], [1, 0]).evaluate()
    assert np.isnan(result["A00"])
    assert np.isnan(result["label_path_p5"])
    assert np.isfinite(result["A10"])
    with pytest.raises(ValueError, match="G is nonempty"):
        FrozenPairAccounting([.8, .2], [1, 0], [.7, .3], [1, 1])
    with pytest.raises(ValueError, match="nonnegative"):
        FrozenPairAccounting([.8, .2], [1, 0], [.7, .3], [1, 0]).evaluate([1, -1])


def test_accuracy_decline_does_not_force_auc_decline():
    p = np.array([.9, .1, .5])  # S,F,E
    r5, r50 = np.array([1, 1, 0]), np.array([1, 0, 0])
    assert r50.mean() < r5.mean()
    assert auroc_correct(p, r50) == 1
    assert auroc_correct(p, r5) == .5


@pytest.mark.parametrize("counts", [(2, 3, 6), (3, 6, 2)])
def test_label_path_bound_is_attained_for_both_weight_orderings(counts):
    ns, nf, ne = counts
    alpha, beta = nf/(ns+nf), nf/(nf+ne)
    # alpha>=beta: S>E>F; beta>=alpha: E>S>F.
    group_scores = [.9, .1, .5] if alpha >= beta else [.5, .1, .9]
    groups = np.repeat([0, 1, 2], counts)
    p = np.repeat(group_scores, counts)
    r5, r50 = (groups != 2).astype(int), (groups == 0).astype(int)
    result = FrozenPairAccounting(p, r5, 1-p, r50).evaluate()
    bound = max(alpha, beta)
    assert result["label_path_p5"] == pytest.approx(bound)
    assert result["label_path_p50"] == pytest.approx(-bound)
    assert abs(result["interaction_I"]) == pytest.approx(2*bound)


def test_one_percent_accuracy_loss_can_change_auc_by_half():
    groups = np.repeat([0, 1, 2], [98, 1, 1])
    p = np.repeat([.5, .1, .9], [98, 1, 1])
    r5, r50 = (groups != 2).astype(int), (groups == 0).astype(int)
    result = FrozenPairAccounting(p, r5, p, r50).evaluate()
    assert r5.mean()-r50.mean() == pytest.approx(.01)
    assert result["label_path_p5"] == .5
