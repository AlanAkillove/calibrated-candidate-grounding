"""Unit tests for :mod:`ccg.semantic.features` (E1 frozen 16-d stats, A7.3).

Pure NumPy: no real caches and no torch. Small hand-computable examples use
unit basis vectors so every column can be checked analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from ccg.semantic.features import (
    DENSITY_TAUS,
    SEMANTIC_STAT_NAMES,
    b3_order,
    descending_average_ranks,
    semantic_stats,
    top5_indices,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _unit_rows(rng: np.random.Generator, n: int, d: int) -> np.ndarray:
    """Return ``[n, d]`` float64 with each row L2-normalised."""
    v = rng.normal(size=(n, d))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _stats_row(out: np.ndarray) -> dict[str, float]:
    return dict(zip(SEMANTIC_STAT_NAMES, out[0]))


def _stat_index(name: str) -> int:
    return SEMANTIC_STAT_NAMES.index(name)


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
def test_frozen_constants():
    assert len(SEMANTIC_STAT_NAMES) == 16
    assert len(set(SEMANTIC_STAT_NAMES)) == 16
    assert DENSITY_TAUS == (0.7, 0.8)


# ---------------------------------------------------------------------------
# b3_order / top5_indices
# ---------------------------------------------------------------------------
def test_b3_order_stable_descending():
    scores = np.array([[0.5, 0.5, 0.9, 0.1, 0.9]])
    order = b3_order(scores)
    # 0.9 at idx 2 & 4 -> smaller index first; then 0.5 at 0 & 1; then 0.1 at 3
    assert order.tolist() == [[2, 4, 0, 1, 3]]
    with pytest.raises(ValueError):
        b3_order(np.zeros((2, 1)))


def test_top5_indices_selection_and_stability():
    scores = np.array(
        [
            [0.1, 0.9, 0.2, 0.8, 0.7, 0.3],
            [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        ]
    )
    out = top5_indices(scores)
    assert out.shape == (2, 5)
    assert out.dtype == np.int64
    assert out[0].tolist() == [1, 3, 4, 5, 2]
    # all-equal row -> stable -> first five indices
    assert out[1].tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        top5_indices(np.zeros((2, 4)))


def test_top5_indices_are_the_five_largest():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=(9, 8))
    top = top5_indices(scores)
    for i in range(scores.shape[0]):
        chosen = set(top[i].tolist())
        rest = [j for j in range(scores.shape[1]) if j not in chosen]
        assert min(scores[i, list(chosen)]) >= max(scores[i, rest])


# ---------------------------------------------------------------------------
# descending_average_ranks
# ---------------------------------------------------------------------------
def test_descending_average_ranks_ties():
    vals = np.array([[2.0, 2.0, 1.0]])
    ranks = descending_average_ranks(vals)
    assert np.allclose(ranks, [[1.5, 1.5, 3.0]])


def test_descending_average_ranks_distinct():
    vals = np.array([[3.0, 1.0, 2.0]])
    assert descending_average_ranks(vals).tolist() == [[1.0, 3.0, 2.0]]
    # a full tie gives the middle rank for every element
    full = np.ones((1, 5))
    assert np.allclose(descending_average_ranks(full), [[3.0] * 5])


# ---------------------------------------------------------------------------
# manual small examples
# ---------------------------------------------------------------------------
def test_manual_uniform_orthogonal_case():
    # z_q = e0; every candidate is orthogonal to z_q -> a is uniform (all zeros)
    z_q = np.array([[1.0, 0.0, 0.0, 0.0]])
    z_i = np.array(
        [
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ]
    )
    scores = np.array([[0.9, 0.8, 0.7, 0.6, 0.5]])
    stat = _stats_row(semantic_stats(z_q, z_i, scores))

    assert stat["clip_top1"] == pytest.approx(0.0)
    assert stat["clip_top2"] == pytest.approx(0.0)
    assert stat["clip_margin12"] == pytest.approx(0.0)
    # uniform softmax -> H = log K
    assert stat["clip_entropy"] == pytest.approx(np.log(5.0), abs=1e-12)
    assert stat["clip_normH"] == pytest.approx(0.0, abs=1e-12)
    # all a tied -> winner average rank = 3 -> 3/5
    assert stat["clip_rank_top1"] == pytest.approx(3.0 / 5.0)
    # winner t = idx0 = [0,1,0,0]; v_t = [1, 1, 0, 0, 0]; over j != t -> [1, 0, 0, 0]
    assert stat["cand_vmax"] == pytest.approx(1.0)
    assert stat["cand_vmean"] == pytest.approx(0.25)
    assert stat["cand_vstd"] == pytest.approx(np.sqrt(0.1875))
    assert stat["cand_top12_sim"] == pytest.approx(1.0)
    assert stat["cand_top15_mean"] == pytest.approx(0.25)
    assert stat["density_070"] == pytest.approx(0.25)
    assert stat["density_080"] == pytest.approx(0.25)
    assert stat["q_top3"] == pytest.approx(0.0)
    assert stat["q_margin13"] == pytest.approx(0.0)
    assert stat["cand_top13_sim"] == pytest.approx(0.0)


def test_manual_margin_and_density_boundary_case():
    # winner t = idx0 = e0; candidate idx1 is a unit vector with cos(t, .) = 0.7
    cos_val = 0.7
    sin_val = np.sqrt(1.0 - cos_val * cos_val)
    z_q = np.array([[1.0, 0.0, 0.0]])
    z_i = np.array(
        [
            [
                [1.0, 0.0, 0.0],
                [cos_val, sin_val, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    scores = np.array([[0.9, 0.8, 0.5, 0.3, 0.2]])
    stat = _stats_row(semantic_stats(z_q, z_i, scores))

    # a = [1, 0.7, 0, 0, 0]; order = [0, 1, 2, 3, 4]
    assert stat["clip_top1"] == pytest.approx(1.0)
    assert stat["clip_top2"] == pytest.approx(0.7)
    assert stat["clip_margin12"] == pytest.approx(0.3)
    # ranks in a: [1, 2, 4, 4, 4] -> winner rank 1 -> 1/5
    assert stat["clip_rank_top1"] == pytest.approx(1.0 / 5.0)
    assert stat["cand_vmax"] == pytest.approx(0.7)
    assert stat["cand_vmean"] == pytest.approx(0.7 / 4.0)
    assert stat["cand_vstd"] == pytest.approx(np.std([0.7, 0.0, 0.0, 0.0]))
    assert stat["cand_top12_sim"] == pytest.approx(0.7)
    assert stat["cand_top15_mean"] == pytest.approx(0.7 / 4.0)
    assert stat["cand_top13_sim"] == pytest.approx(0.0)
    # strict greater: v_t1 == 0.7 exactly is NOT counted
    assert stat["density_070"] == pytest.approx(0.0)
    assert stat["density_080"] == pytest.approx(0.0)
    assert stat["q_top3"] == pytest.approx(0.0)
    assert stat["q_margin13"] == pytest.approx(1.0)
    # entropy reference via an independent stable softmax formula (A7.3: T = 0.01)
    a = np.array([1.0, 0.7, 0.0, 0.0, 0.0]) / 0.01
    p = np.exp(a - a.max())
    p = p / p.sum()
    assert stat["clip_entropy"] == pytest.approx(float(-(p * np.log(p)).sum()))
    assert stat["clip_entropy"] < np.log(5.0)  # CLIP temperature sharpens the softmax


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------
def test_semantic_stats_is_bit_identical():
    rng = np.random.default_rng(0)
    n, k, d = 7, 6, 8
    z_q = _unit_rows(rng, n, d)
    z_i = _unit_rows(rng, n * k, d).reshape(n, k, d)
    scores = rng.normal(size=(n, k)).astype(np.float32)
    out1 = semantic_stats(z_q, z_i, scores)
    out2 = semantic_stats(z_q, z_i, scores)
    assert np.array_equal(out1, out2)
    assert out1.shape == (n, len(SEMANTIC_STAT_NAMES))
    assert out1.dtype == np.float64


# ---------------------------------------------------------------------------
# permutation invariance
# ---------------------------------------------------------------------------
def test_permutation_invariance():
    rng = np.random.default_rng(1)
    n, k, d = 5, 6, 10
    z_q = _unit_rows(rng, n, d)
    z_i = _unit_rows(rng, n * k, d).reshape(n, k, d)
    # distinct scores per row -> no tie-break ambiguity
    scores = np.broadcast_to(np.arange(1.0, k + 1.0), (n, k)).copy()

    perm = np.array([3, 0, 5, 1, 4, 2])
    out = semantic_stats(z_q, z_i, scores)
    out_perm = semantic_stats(z_q, z_i[:, perm, :], scores[:, perm])
    assert np.allclose(out, out_perm, atol=1e-12)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def test_validation_errors():
    rng = np.random.default_rng(2)
    z_q = _unit_rows(rng, 4, 8)
    z_i = _unit_rows(rng, 4 * 5, 8).reshape(4, 5, 8)
    scores = rng.normal(size=(4, 5))

    # K < 5
    with pytest.raises(ValueError):
        semantic_stats(z_q, z_i[:, :4, :], scores[:, :4])
    # z_q batch mismatch
    with pytest.raises(ValueError):
        semantic_stats(z_q[:3], z_i, scores)
    # scores batch mismatch
    with pytest.raises(ValueError):
        semantic_stats(z_q, z_i, scores[:3])
    # z_i not 3-D
    with pytest.raises(ValueError):
        semantic_stats(z_q, z_i[0], scores)
    # z_q not 2-D
    with pytest.raises(ValueError):
        semantic_stats(z_q[0], z_i, scores)
    # scores not 2-D
    with pytest.raises(ValueError):
        semantic_stats(z_q, z_i, scores[0])
    # candidate-count mismatch between z_i and scores
    with pytest.raises(ValueError):
        semantic_stats(z_q, z_i, rng.normal(size=(4, 6)))
    # embedding-dim mismatch between z_q and z_i
    with pytest.raises(ValueError):
        semantic_stats(z_q, _unit_rows(rng, 4 * 5, 9).reshape(4, 5, 9), scores)


# ---------------------------------------------------------------------------
# value ranges
# ---------------------------------------------------------------------------
def test_value_ranges():
    rng = np.random.default_rng(3)
    n, k, d = 12, 20, 16
    z_q = _unit_rows(rng, n, d)
    z_i = _unit_rows(rng, n * k, d).reshape(n, k, d)
    scores = rng.normal(size=(n, k))
    out = semantic_stats(z_q, z_i, scores)

    entropy = out[:, _stat_index("clip_entropy")]
    assert np.all(entropy >= -1e-12)
    assert np.all(entropy <= np.log(k) + 1e-9)

    norm_h = out[:, _stat_index("clip_normH")]
    assert np.all(norm_h >= -1e-12)
    assert np.all(norm_h <= 1.0 + 1e-12)

    for name in ("density_070", "density_080"):
        col = out[:, _stat_index(name)]
        assert np.all(col >= 0.0)
        assert np.all(col <= 1.0)

    rank_top1 = out[:, _stat_index("clip_rank_top1")]
    assert np.all(rank_top1 > 0.0)
    assert np.all(rank_top1 <= 1.0 + 1e-12)
