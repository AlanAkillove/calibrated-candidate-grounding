"""Synthetic contract checks for strictly query-independent V_pure features."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ccg.v4.pure_features import PURE_FEATURE_NAMES, pure_features  # noqa: E402


def _synthetic_set(seed: int = 31, *, k: int = 7, d: int = 5):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(k, d)).astype(np.float32)
    xy1 = rng.uniform(-20.0, 10.0, size=(k, 2))
    wh = rng.uniform(1.0, 8.0, size=(k, 2))
    boxes = np.column_stack([xy1, xy1 + wh])
    return embeddings, boxes


def test_name_order_dimension_and_query_free_signature():
    embeddings, boxes = _synthetic_set()
    features = pure_features(embeddings, boxes)

    assert PURE_FEATURE_NAMES == (
        "cos_mean",
        "cos_std",
        "cos_q50",
        "cos_q90",
        "nn_cos_mean",
        "nn_cos_max",
        "spectral_effective_rank_normalized",
        "pair_iou_mean",
        "pair_iou_q90",
        "pair_iou_fraction_gt_05",
        "center_distance_mean_normalized",
        "log_area_std",
    )
    assert features.shape == (12,)
    assert features.dtype == np.float64
    assert np.all(np.isfinite(features))
    assert tuple(inspect.signature(pure_features).parameters) == (
        "crop_embeddings",
        "candidate_boxes",
    )


def test_permutation_invariance_uses_the_whole_unordered_candidate_set():
    embeddings, boxes = _synthetic_set(k=9)
    expected = pure_features(embeddings, boxes)
    permutation = np.asarray([8, 2, 6, 0, 5, 1, 7, 4, 3])

    actual = pure_features(embeddings[permutation], boxes[permutation])

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_pair_cosines_use_only_i_less_than_j_and_linear_quantiles():
    embeddings = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    boxes = np.asarray(
        [[0, 0, 1, 1], [3, 0, 4, 1], [6, 0, 7, 1], [9, 0, 10, 1], [12, 0, 13, 1]],
        dtype=np.float64,
    )
    z = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    i, j = np.triu_indices(5, k=1)
    pair = (z @ z.T)[i, j]

    features = pure_features(embeddings, boxes)

    assert features[0] == pytest.approx(np.mean(pair))
    assert features[1] == pytest.approx(np.std(pair, ddof=0))
    assert features[2] == pytest.approx(np.quantile(pair, 0.50, method="linear"))
    assert features[3] == pytest.approx(np.quantile(pair, 0.90, method="linear"))


def test_repeated_vectors_and_boxes_exclude_diagonal_and_give_expected_rank():
    embeddings = np.tile(np.asarray([[2.0, -1.0, 4.0, 3.0, 7.0]]), (5, 1))
    boxes = np.tile(np.asarray([[1.0, 2.0, 5.0, 8.0]]), (5, 1))

    features = pure_features(embeddings, boxes)

    np.testing.assert_allclose(features[:6], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0], atol=1e-12)
    assert features[6] == pytest.approx(1.0 / 5.0, abs=1e-12)
    np.testing.assert_allclose(features[7:10], [1.0, 1.0, 1.0], atol=1e-12)
    assert features[10] == pytest.approx(0.0)
    assert features[11] == pytest.approx(0.0)


def test_spectral_effective_rank_is_one_for_orthonormal_rows():
    embeddings = np.eye(5, dtype=np.float64)
    boxes = np.asarray(
        [[0, 0, 1, 1], [3, 0, 4, 1], [6, 0, 7, 1], [9, 0, 10, 1], [12, 0, 13, 1]],
        dtype=np.float64,
    )

    features = pure_features(embeddings, boxes)

    assert features[6] == pytest.approx(1.0, abs=1e-12)
    assert features[4] == pytest.approx(0.0)
    assert features[5] == pytest.approx(0.0)


def test_geometry_features_are_translation_and_uniform_scale_invariant():
    embeddings, boxes = _synthetic_set(k=8)
    original = pure_features(embeddings, boxes)
    shifted_scaled = pure_features(embeddings, boxes * 3.5 + [17.0, -9.0, 17.0, -9.0])

    np.testing.assert_allclose(shifted_scaled[7:], original[7:], rtol=1e-12, atol=1e-12)


def test_iou_threshold_is_strictly_greater_than_one_half():
    embeddings = np.eye(5, dtype=np.float64)
    boxes = np.asarray(
        [
            [0.0, 0.0, 2.0, 2.0],
            [0.0, 0.0, 1.0, 2.0],  # IoU with the first box is exactly 0.5.
            [10.0, 0.0, 12.0, 2.0],
            [20.0, 0.0, 22.0, 2.0],
            [30.0, 0.0, 32.0, 2.0],
        ]
    )

    features = pure_features(embeddings, boxes)

    assert features[7] == pytest.approx(0.5 / 10.0)
    assert features[9] == 0.0


def test_rejects_small_sets_shape_mismatch_zero_and_nonfinite_crops():
    embeddings, boxes = _synthetic_set(k=5)
    with pytest.raises(ValueError, match="K >= 5"):
        pure_features(embeddings[:4], boxes[:4])
    with pytest.raises(ValueError, match="K="):
        pure_features(embeddings, boxes[:4])

    zero = embeddings.copy()
    zero[2] = 0.0
    with pytest.raises(ValueError, match="zero vector"):
        pure_features(zero, boxes)

    nonfinite = embeddings.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pure_features(nonfinite, boxes)


@pytest.mark.parametrize(
    "bad_box",
    [
        [0.0, 0.0, 0.0, 2.0],
        [0.0, 0.0, 2.0, 0.0],
        [3.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, np.inf, 1.0],
    ],
)
def test_rejects_invalid_or_nonfinite_positive_area_boxes(bad_box):
    embeddings, boxes = _synthetic_set(k=5)
    boxes[0] = bad_box

    with pytest.raises(ValueError):
        pure_features(embeddings, boxes)
