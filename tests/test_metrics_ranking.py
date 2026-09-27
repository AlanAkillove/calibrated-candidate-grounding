"""Synthetic tests for :mod:`ccg.metrics.ranking` (Task 5: perfect ranking & co)."""

from __future__ import annotations

import numpy as np
import pytest

from ccg.metrics.ranking import accuracy_by_k, mrr, top1_accuracy, topk_accuracy, target_rank


def _perfect_ranking(n: int = 50, k: int = 10, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Every target score is strictly the largest of its candidate set."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1.0, size=(n, k))
    target = rng.integers(0, k, size=n)
    rows = np.arange(n)
    scores[rows, target] = scores.max(axis=1) + 1.0
    return scores, target


def test_perfect_ranking_top1_and_mrr_are_one() -> None:
    scores, target = _perfect_ranking()
    assert top1_accuracy(scores, target) == pytest.approx(1.0)
    assert mrr(scores, target) == pytest.approx(1.0)
    assert topk_accuracy(scores, target, 1) == pytest.approx(1.0)
    assert topk_accuracy(scores, target, 5) == pytest.approx(1.0)
    assert np.all(target_rank(scores, target) == 1)


def test_inverse_ranking_is_zero() -> None:
    scores = np.array([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
    target = np.array([2, 0])  # the worst candidate in both rows -> rank 3
    assert top1_accuracy(scores, target) == pytest.approx(0.0)
    assert mrr(scores, target) == pytest.approx(1.0 / 3.0)
    assert topk_accuracy(scores, target, 2) == pytest.approx(0.0)
    assert topk_accuracy(scores, target, 3) == pytest.approx(1.0)


def test_top1_matches_manual_example() -> None:
    scores = np.array([[0.4, 0.1, 0.5], [0.9, 0.05, 0.05], [0.2, 0.7, 0.1]])
    target = np.array([0, 0, 1])
    assert top1_accuracy(scores, target) == pytest.approx(2.0 / 3.0)
    assert np.all(target_rank(scores, target) == np.array([2, 1, 1]))
    assert mrr(scores, target) == pytest.approx((0.5 + 1.0 + 1.0) / 3.0)


def test_accuracy_by_k_matches_topk_accuracy() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(120, 20))
    target = rng.integers(0, 20, size=120)
    table = accuracy_by_k(scores, target, ks=[1, 2, 5, 20, 50])
    assert set(table) == {1, 2, 5, 20, 50}
    for k in (1, 2, 5, 20):
        assert table[k] == pytest.approx(topk_accuracy(scores, target, k))
    assert table[50] == pytest.approx(1.0)  # k clamped to K
    assert table[1] <= table[2] <= table[5] <= table[20]


def test_ties_are_broken_towards_smallest_index() -> None:
    scores = np.ones((3, 4))
    assert top1_accuracy(scores, np.array([0, 0, 0])) == pytest.approx(1.0)
    # candidate 1 ties with candidate 0 but loses the stable-order tie-break
    assert top1_accuracy(scores, np.array([1, 1, 1])) == pytest.approx(0.0)
    assert topk_accuracy(scores, np.array([1, 1, 1]), 2) == pytest.approx(1.0)
    assert mrr(scores, np.array([3, 3, 3])) == pytest.approx(1.0 / 4.0)


def test_target_absent_raises_by_default() -> None:
    scores, _ = _perfect_ranking(n=4, k=5)
    target = np.array([0, 1, -1, 3])
    with pytest.raises(ValueError, match="target absent"):
        top1_accuracy(scores, target)
    with pytest.raises(ValueError, match="target absent"):
        mrr(scores, target)


def test_target_absent_can_be_masked_or_dropped() -> None:
    scores = np.array([[1.0, 0.0], [0.0, 1.0], [5.0, 1.0], [1.0, 5.0]])
    target = np.array([0, 1, -1, -1])
    present = np.array([True, True, False, False])
    # explicit mask
    assert top1_accuracy(scores, target, mask=present) == pytest.approx(1.0)
    # automatic drop
    assert top1_accuracy(scores, target, strict=False) == pytest.approx(1.0)
    assert topk_accuracy(scores, target, 1, strict=False) == pytest.approx(1.0)
    assert accuracy_by_k(scores, target, [1], strict=False) == {1: pytest.approx(1.0)}
    # a mask that selects only absent rows still raises, an empty mask is rejected
    with pytest.raises(ValueError, match="target absent"):
        top1_accuracy(scores, target, mask=~present)
    with pytest.raises(ValueError, match="No rows left"):
        top1_accuracy(scores, np.array([0, 1, 0, 1]), mask=np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="differ in length"):
        top1_accuracy(scores, target, mask=np.array([True, False]))


def test_shape_and_range_validation() -> None:
    with pytest.raises(ValueError, match="2-D"):
        top1_accuracy(np.ones(4), np.array([0]))
    with pytest.raises(ValueError, match="1-D"):
        top1_accuracy(np.ones((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="outside"):
        top1_accuracy(np.ones((2, 3)), np.array([0, 5]))
    with pytest.raises(ValueError, match="empty"):
        top1_accuracy(np.ones((2, 3)), np.array([]))
    with pytest.raises(ValueError, match="k"):
        topk_accuracy(np.ones((2, 3)), np.array([0, 1]), 0)


def test_metrics_are_finite_for_float32_inputs() -> None:
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(30, 6)).astype(np.float32)
    target = rng.integers(0, 6, size=30).astype(np.int32)
    acc = top1_accuracy(scores, target)
    assert np.isfinite(acc) and 0.0 <= acc <= 1.0
    assert np.isfinite(mrr(scores, target))
