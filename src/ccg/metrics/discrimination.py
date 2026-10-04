"""Discrimination metrics of the correctness event (Phase 0A.1).

These are companions of the calibration / selective metrics: they measure how
well a confidence score separates correct from incorrect predictions. For
fixed class-conditional score distributions, AUROC is invariant to the positive
rate; changes in the score distributions of correct and incorrect samples can
still change AUROC across settings. AUPRC's chance level is the positive
prevalence and is therefore reported next to it.

Contract functions
------------------
* :func:`auroc_correct` -- area under the ROC curve of ``1[correct]`` vs
  confidence, computed with the tie-correct Mann-Whitney average-rank formula
  (identical to :func:`sklearn.metrics.roc_auc_score`).
* :func:`auprc_correct` -- average precision of the same event, computed with
  the scikit-learn threshold/tie convention (all samples tied at a threshold
  enter the operating point together).

Degenerate inputs
-----------------
A *single-class* correctness vector (all correct or all wrong) or an *empty*
input has no ROC / PR curve; both functions return ``float("nan")`` rather
than raising, so a pipeline can keep the row and mark the metric undefined.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "auprc_correct",
    "auroc_correct",
]


def _coerce(
    confidence: Any,
    correctness: Any,
) -> Tuple[Optional[NDArray], Optional[NDArray]]:
    """Validate aligned ``(confidence, correctness)`` arrays.

    Returns ``(None, None)`` for an empty input (the caller maps that to
    ``nan``); otherwise finite FP64 confidence and binary 0/1 correctness.
    """
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    corr = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if conf.shape[0] != corr.shape[0]:
        raise ValueError(
            f"`confidence` ({conf.shape[0]}) and `correctness` ({corr.shape[0]}) "
            "differ in length"
        )
    if conf.shape[0] == 0:
        return None, None
    if not np.all(np.isfinite(conf)):
        raise ValueError("`confidence` contains NaN or inf values")
    if not np.all(np.isfinite(corr)):
        raise ValueError("`correctness` contains NaN or inf values")
    if np.any((corr != 0.0) & (corr != 1.0)):
        raise ValueError("`correctness` must be binary (0/1 or bool)")
    return conf, corr


def _average_ranks(values: NDArray) -> NDArray:
    """1-based ranks with ties replaced by their mean rank (Mann-Whitney)."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    new_group = np.empty(values.shape[0], dtype=bool)
    new_group[0] = True
    if values.shape[0] > 1:
        new_group[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = np.cumsum(new_group) - 1
    n_groups = int(group_id[-1]) + 1
    sums = np.bincount(group_id, weights=ranks, minlength=n_groups)
    counts = np.bincount(group_id, minlength=n_groups).astype(np.float64)
    mean_ranks = (sums / counts)[group_id]
    out = np.empty(values.shape[0], dtype=np.float64)
    out[order] = mean_ranks
    return out


def auroc_correct(confidence: Any, correctness: Any) -> float:
    """AUROC of ``confidence`` separating correct (1) from incorrect (0).

    ``1.0`` = every correct prediction is more confident than every incorrect
    one, ``0.5`` = uninformative, ``0.0`` = perfectly inverted.  Computed with
    the tie-correct Mann-Whitney average-rank statistic, exactly matching
    :func:`sklearn.metrics.roc_auc_score`.

    Returns ``float("nan")`` when the input is empty or when ``correctness``
    contains a single class (no positive or no negative sample), where the ROC
    curve is undefined.
    """
    conf, corr = _coerce(confidence, correctness)
    if conf is None or corr is None:
        return float("nan")
    n_pos = float(np.sum(corr))
    n_neg = float(corr.shape[0]) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return float("nan")
    ranks = _average_ranks(conf)
    sum_ranks_pos = float(np.sum(ranks[corr == 1.0]))
    return float((sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def auprc_correct(confidence: Any, correctness: Any) -> float:
    """Average precision of ``confidence`` for the correct (1) event.

    ``1.0`` when every correct prediction is ranked above every incorrect one;
    the chance level equals the positive prevalence.  Ties follow the
    scikit-learn convention: samples sharing a score enter the operating point
    together (the curve is evaluated at distinct score thresholds), so the
    result matches :func:`sklearn.metrics.average_precision_score`.

    Returns ``float("nan")`` when the input is empty or when ``correctness``
    contains a single class (no positive or no negative sample).
    """
    conf, corr = _coerce(confidence, correctness)
    if conf is None or corr is None:
        return float("nan")
    n_pos = float(np.sum(corr))
    if n_pos == 0.0 or n_pos == float(corr.shape[0]):
        return float("nan")

    order = np.argsort(-conf, kind="stable")
    corr_sorted = corr[order]
    conf_sorted = conf[order]
    tps = np.cumsum(corr_sorted)
    fps = np.cumsum(1.0 - corr_sorted)

    # last index of each distinct-score group = the scikit-learn thresholds
    n = corr_sorted.shape[0]
    change = np.empty(n - 1, dtype=bool) if n > 1 else np.empty(0, dtype=bool)
    if n > 1:
        change = conf_sorted[:-1] != conf_sorted[1:]
    threshold_idxs = np.concatenate((np.nonzero(change)[0], [n - 1])).astype(np.int64)

    precision = tps[threshold_idxs] / (tps[threshold_idxs] + fps[threshold_idxs])
    recall = tps[threshold_idxs] / tps[-1]
    # precision_recall_curve appends the (recall=0, precision=1) anchor point
    precision = np.concatenate((precision[::-1], [1.0]))
    recall = np.concatenate((recall[::-1], [0.0]))
    return float(-np.sum(np.diff(recall) * precision[:-1]))
