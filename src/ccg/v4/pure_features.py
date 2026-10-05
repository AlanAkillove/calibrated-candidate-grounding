"""Strictly query-independent candidate-set features for V4.

``pure_features`` is a deterministic function of an unordered set of crop
embeddings and their ``xyxy`` boxes.  Its signature deliberately has no query,
scorer, score/rank, winner, target identity, ground-truth, label, or objectness
input.  Given the same candidate set, it therefore returns the same 12-vector
regardless of which query is being considered upstream.

This is functional independence conditional on the supplied candidate set.  It
does not establish end-to-end statistical independence when an upstream
controlled-set constructor may use a target or expression to choose that set.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PURE_FEATURE_NAMES", "pure_features"]

PURE_FEATURE_NAMES: tuple[str, ...] = (
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


def _unit_rows_float64(embeddings: np.ndarray) -> np.ndarray:
    """Convert rows to float64 unit vectors, rejecting zero/non-finite rows."""
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"crop_embeddings must have shape [K, D], got {values.shape}")
    if values.shape[1] < 1:
        raise ValueError("crop_embeddings must have D >= 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("crop_embeddings must be finite")

    # Scale before taking the norm so every finite nonzero float64 row can be
    # normalised without overflowing its sum of squares or underflowing it.
    scale = np.max(np.abs(values), axis=1)
    if np.any(scale == 0.0):
        raise ValueError("crop_embeddings must not contain a zero vector")
    scaled = values / scale[:, None]
    norm = np.sqrt(np.sum(scaled * scaled, axis=1))
    if not np.all(np.isfinite(norm)) or np.any(norm == 0.0):
        raise ValueError("crop_embeddings could not be normalised in float64")
    unit = scaled / norm[:, None]
    if not np.all(np.isfinite(unit)):
        raise ValueError("normalised crop_embeddings contain a non-finite value")
    return unit


def _pair_iou(boxes: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    """Compute xyxy IoU for the requested unordered pairs."""
    left_top = np.maximum(boxes[pair_i, :2], boxes[pair_j, :2])
    right_bottom = np.minimum(boxes[pair_i, 2:], boxes[pair_j, 2:])
    intersection_wh = np.maximum(right_bottom - left_top, 0.0)
    intersection = intersection_wh[:, 0] * intersection_wh[:, 1]

    wh = boxes[:, 2:] - boxes[:, :2]
    area = wh[:, 0] * wh[:, 1]
    union = area[pair_i] + area[pair_j] - intersection
    if (
        not np.all(np.isfinite(intersection))
        or not np.all(np.isfinite(union))
        or np.any(union <= 0.0)
    ):
        raise ValueError("candidate box areas and pairwise unions must be finite and positive")
    iou = intersection / union
    if not np.all(np.isfinite(iou)):
        raise ValueError("pairwise box IoU contains a non-finite value")
    # The mathematical result is in [0, 1]; clamp only floating-point drift.
    return np.clip(iou, 0.0, 1.0)


def pure_features(crop_embeddings: np.ndarray, candidate_boxes: np.ndarray) -> np.ndarray:
    """Return the fixed 12-D ``V_pure`` vector for one unordered candidate set.

    Parameters
    ----------
    crop_embeddings
        ``[K, D]`` candidate crop embeddings.  Rows are re-normalized in
        float64; every row must be finite and nonzero.
    candidate_boxes
        ``[K, 4]`` finite ``xyxy`` boxes with strictly positive width and
        height.  Boxes and embeddings must describe the same K candidates.

    Returns
    -------
    numpy.ndarray
        A finite float64 vector in :data:`PURE_FEATURE_NAMES` order.  All pair
        features use exactly the ``i < j`` pairs, with no diagonal entries.
        Per-candidate nearest-neighbor cosine excludes each candidate itself.

    Definitions
    -----------
    K must be at least 5.  Let ``Z`` be the float64-renormalized crop rows,
    ``C = Z Z.T``, ``P = {C[i,j] : i < j}``, and let ``N_i = max_{j != i} C[i,j]``.

    * ``cos_mean``, ``cos_std``: mean and population standard deviation of P
      (``ddof=0``); ``cos_q50`` and ``cos_q90`` are NumPy linear quantiles.
    * ``nn_cos_mean`` and ``nn_cos_max``: mean and maximum of all ``N_i``.
    * ``spectral_effective_rank_normalized``: compute eigenvalues of the unit
      vector Gram matrix C, clip negative eigenvalues to zero, set
      ``p_l = lambda_l / sum(lambda)``, and return
      ``exp(-sum_l p_l log(p_l)) / min(K, D)`` (zero-probability terms are
      omitted).
    * ``pair_iou_mean``, ``pair_iou_q90``: mean and linear 90th percentile of
      pairwise box IoUs over i < j.  ``pair_iou_fraction_gt_05`` is the
      fraction strictly greater than 0.5.
    * ``center_distance_mean_normalized``: mean Euclidean distance between all
      i < j box centers divided by the diagonal of the smallest xyxy box
      containing the full candidate set.
    * ``log_area_std``: population standard deviation (``ddof=0``) of the
      natural log of positive box areas.

    No candidate is truncated, sorted by query score, or selected as a winner.
    In particular, the normalized spectral effective rank is not a query-rank
    statistic.  ``nn_cos_mean`` is the predeclared primary ambiguity measure.
    """
    unit = _unit_rows_float64(crop_embeddings)
    boxes = np.asarray(candidate_boxes, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1:] != (4,):
        raise ValueError(f"candidate_boxes must have shape [K, 4], got {boxes.shape}")
    if not np.all(np.isfinite(boxes)):
        raise ValueError("candidate_boxes must be finite")

    k, d = unit.shape
    if k < 5:
        raise ValueError(f"pure_features requires K >= 5, got K={k}")
    if boxes.shape[0] != k:
        raise ValueError(
            f"candidate_boxes has K={boxes.shape[0]} but crop_embeddings has K={k}"
        )

    wh = boxes[:, 2:] - boxes[:, :2]
    if not np.all(np.isfinite(wh)) or np.any(wh <= 0.0):
        raise ValueError("candidate_boxes must have finite, strictly positive width and height")
    areas = wh[:, 0] * wh[:, 1]
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("candidate box areas must be finite and positive in float64")

    pair_i, pair_j = np.triu_indices(k, k=1)
    gram = unit @ unit.T
    if not np.all(np.isfinite(gram)):
        raise ValueError("crop cosine matrix contains a non-finite value")
    # Unit-vector cosine is bounded mathematically; clip roundoff at endpoints.
    cosine = np.clip(gram, -1.0, 1.0)
    pair_cos = cosine[pair_i, pair_j]
    cos_q50, cos_q90 = np.quantile(pair_cos, [0.50, 0.90], method="linear")

    without_self = cosine.copy()
    np.fill_diagonal(without_self, -np.inf)
    nearest_cos = np.max(without_self, axis=1)

    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    eigenvalue_sum = float(np.sum(eigenvalues))
    if not np.isfinite(eigenvalue_sum) or eigenvalue_sum <= 0.0:
        raise ValueError("crop Gram matrix must have positive finite spectral mass")
    probabilities = eigenvalues / eigenvalue_sum
    positive_probabilities = probabilities > 0.0
    spectral_entropy = -float(
        np.sum(probabilities[positive_probabilities] * np.log(probabilities[positive_probabilities]))
    )
    spectral_effective_rank_normalized = np.exp(spectral_entropy) / float(min(k, d))

    pair_iou = _pair_iou(boxes, pair_i, pair_j)
    pair_iou_q90 = float(np.quantile(pair_iou, 0.90, method="linear"))
    pair_iou_fraction_gt_05 = float(np.mean(pair_iou > 0.5))

    centers = boxes[:, :2] + wh * 0.5
    center_delta = centers[pair_i] - centers[pair_j]
    if not np.all(np.isfinite(center_delta)):
        raise ValueError("candidate box center differences must be finite")
    pair_center_distances = np.hypot(center_delta[:, 0], center_delta[:, 1])
    set_extent = np.max(boxes[:, 2:], axis=0) - np.min(boxes[:, :2], axis=0)
    set_diagonal = float(np.hypot(set_extent[0], set_extent[1]))
    if (
        not np.all(np.isfinite(pair_center_distances))
        or not np.isfinite(set_diagonal)
        or set_diagonal <= 0.0
    ):
        raise ValueError("candidate center distances and set bounding diagonal must be finite")
    center_distance_mean_normalized = float(np.mean(pair_center_distances) / set_diagonal)

    log_area_std = float(np.std(np.log(areas), ddof=0))
    result = np.asarray(
        [
            np.mean(pair_cos),
            np.std(pair_cos, ddof=0),
            cos_q50,
            cos_q90,
            np.mean(nearest_cos),
            np.max(nearest_cos),
            spectral_effective_rank_normalized,
            np.mean(pair_iou),
            pair_iou_q90,
            pair_iou_fraction_gt_05,
            center_distance_mean_normalized,
            log_area_std,
        ],
        dtype=np.float64,
    )
    if result.shape != (len(PURE_FEATURE_NAMES),) or not np.all(np.isfinite(result)):
        raise ValueError("V_pure computation produced a malformed or non-finite feature vector")
    return result
