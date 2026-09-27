"""Target-absence / NONE-decision metrics (Phase 2, section 21 of the protocol).

Contract functions: :func:`auroc`, :func:`auprc`, :func:`fpr_at_tpr`,
:func:`none_precision`, :func:`none_recall`, :func:`none_f1`,
:func:`false_selection_rate`.

Two different input conventions are used:

*Score-based* (does the model separate target-present from target-absent?):
``scores_pos_neg`` is a ``[N]`` array of "is the target present" scores
(higher = more likely present; a max-softmax probability or ``1 - P(NONE)`` both
work) and ``labels`` is the matching ``[N]`` binary ground truth with ``1`` =
target present, ``0`` = target absent.

*Decision-based* (what did the system actually output?): ``pred_none`` /
``true_none`` are aligned boolean arrays and ``pred_index`` is the emitted
candidate index with ``-1`` meaning "NONE / abstain".

AUROC / AUPRC use :mod:`scikit-learn` when it is installed and fall back to pure
numpy rank-based implementations (tie-correct Mann-Whitney U for AUROC,
sum-of-precision-increments for AUPRC) otherwise; both paths are exact, so the
fallback is not an approximation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ccg.metrics._validation import (
    as_int_array,
    require_same_length,
    to_numpy_1d,
    to_probability_1d,
)

try:  # pragma: no cover - depends on the environment
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    HAS_SKLEARN = True
except Exception:  # pragma: no cover - scikit-learn missing / broken import
    average_precision_score = None  # type: ignore[assignment]
    roc_auc_score = None  # type: ignore[assignment]
    roc_curve = None  # type: ignore[assignment]
    HAS_SKLEARN = False

__all__ = [
    "auprc",
    "auroc",
    "false_selection_rate",
    "fpr_at_tpr",
    "none_f1",
    "none_precision",
    "none_recall",
    "roc_points",
]


def _binary_inputs(scores: Any, labels: Any) -> tuple[NDArray, NDArray]:
    """Validate score/label pairs and return ``(scores, labels.astype(float))``."""
    s = to_numpy_1d(scores, "scores_pos_neg")
    y = to_numpy_1d(labels, "labels", dtype=np.float64)
    if np.any(~np.isfinite(s)):
        raise ValueError("`scores_pos_neg` contains NaN or inf values")
    if np.any((y != 0.0) & (y != 1.0)):
        raise ValueError("`labels` must be binary with 1 = target present and 0 = target absent")
    require_same_length(s, y, "scores_pos_neg", "labels")
    n_pos, n_neg = float(np.sum(y)), float(np.sum(1.0 - y))
    if n_pos == 0.0 or n_neg == 0.0:
        raise ValueError("`labels` must contain at least one positive and one negative sample")
    return s, y


def _average_ranks(values: NDArray) -> NDArray:
    """1-based ranks with ties replaced by their mean rank (used by Mann-Whitney)."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    # collapse equal values onto their mean rank
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


def _auroc_numpy(scores: NDArray, labels: NDArray) -> float:
    """Exact tie-correct AUROC via the Mann-Whitney U statistic (pure numpy)."""
    ranks = _average_ranks(scores)
    n_pos = float(np.sum(labels))
    n_neg = float(labels.shape[0]) - n_pos
    sum_ranks_pos = float(np.sum(ranks[labels == 1.0]))
    return (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _roc_points(labels: NDArray, scores: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """ROC operating points sorted by decreasing score (numpy ``roc_curve`` clone)."""
    desc = np.argsort(-scores, kind="stable")
    scores_sorted = scores[desc]
    labels_sorted = labels[desc]
    # one operating point per distinct score, taken *after* every sample that
    # reaches it (the scikit-learn convention: score >= threshold)
    n_points = scores_sorted.shape[0]
    keep = np.empty(n_points, dtype=bool)
    keep[-1] = True
    if n_points > 1:
        keep[:-1] = scores_sorted[:-1] != scores_sorted[1:]
    thresholds = scores_sorted[keep]
    tps = np.cumsum(labels_sorted)[keep]
    fps = np.cumsum(1.0 - labels_sorted)[keep]
    n_pos = float(labels_sorted.sum())
    n_neg = float(labels_sorted.shape[0]) - n_pos
    tpr = np.concatenate(([0.0], tps / n_pos))
    fpr = np.concatenate(([0.0], fps / n_neg))
    thresholds = np.concatenate(([thresholds[0] + 1.0], thresholds))
    return fpr, tpr, thresholds


def roc_points(scores: Any, labels: Any) -> tuple[NDArray, NDArray, NDArray]:
    """Return ``(fpr, tpr, thresholds)`` of the binary ROC curve (pure numpy)."""
    s, y = _binary_inputs(scores, labels)
    return _roc_points(y, s)


def _auprc_numpy(scores: NDArray, labels: NDArray) -> float:
    """Average precision: sum of precision increments over decreasing score."""
    desc = np.argsort(-scores, kind="stable")
    labels_sorted = labels[desc]
    tps = np.cumsum(labels_sorted)
    precision = tps / np.arange(1, labels_sorted.shape[0] + 1, dtype=np.float64)
    recall = tps / float(labels_sorted.sum())
    delta_recall = np.diff(np.concatenate(([0.0], recall)))
    return float(np.sum(precision * delta_recall))


def auroc(scores_pos_neg: Any, labels: Any) -> float:
    """Area under the ROC curve for "target present vs target absent" detection.

    ``1.0`` = perfectly separable (all positives above all negatives), ``0.5`` =
    uninformative, ``0.0`` = perfectly inverted. Ties are handled exactly.
    """
    s, y = _binary_inputs(scores_pos_neg, labels)
    if HAS_SKLEARN:  # pragma: no cover - environment dependent
        return float(roc_auc_score(y, s))
    return _auroc_numpy(s, y)


def auprc(scores: Any, labels: Any) -> float:
    """Area under the precision-recall curve (= average precision).

    ``1.0`` when every positive is ranked above every negative; the chance level
    equals the positive prevalence, which makes this the preferred summary when
    the target-absent class is rare.
    """
    s, y = _binary_inputs(scores, labels)
    if HAS_SKLEARN:  # pragma: no cover - environment dependent
        return float(average_precision_score(y, s))
    return _auprc_numpy(s, y)


def fpr_at_tpr(scores: Any, labels: Any, tpr: float = 0.95) -> float:
    """Smallest false-positive rate reached at a true-positive rate of ``tpr``.

    Computed from the empirical ROC curve without interpolation: the returned
    value is ``min{fpr : tpr >= requested}``, i.e. an achievable operating point
    at the chosen abstention threshold. Lower is better.
    """
    level = float(tpr)
    if not np.isfinite(level) or level <= 0.0 or level > 1.0:
        raise ValueError(f"`tpr` must be in (0, 1], got {tpr}")
    s, y = _binary_inputs(scores, labels)
    fpr, tpr_curve, _thr = _roc_points(y, s)
    hits = np.flatnonzero(tpr_curve >= level)
    if hits.size == 0:  # pragma: no cover - last point is always tpr == 1
        return 1.0
    return float(fpr[hits[0]])


def _none_labels(pred_none: Any, true_none: Any) -> tuple[NDArray, NDArray]:
    pred = to_numpy_1d(pred_none, "pred_none")
    true = to_numpy_1d(true_none, "true_none")
    if np.any((pred != 0) & (pred != 1)) or np.any((true != 0) & (true != 1)):
        raise ValueError("`pred_none` / `true_none` must be boolean 0/1 arrays")
    require_same_length(pred, true, "pred_none", "true_none")
    return pred.astype(bool), true.astype(bool)


def none_precision(pred_none: Any, true_none: Any) -> float:
    """Fraction of emitted NONE decisions that were correct.

    Returns ``0.0`` when the model never predicts NONE (undefined 0/0 follows the
    scikit-learn ``zero_division=0`` convention).
    """
    pred, true = _none_labels(pred_none, true_none)
    n_pred = int(np.sum(pred))
    if n_pred == 0:
        return 0.0
    return float(np.sum(pred & true) / n_pred)


def none_recall(pred_none: Any, true_none: Any) -> float:
    """Fraction of truly absent targets that the model flagged as NONE.

    Returns ``0.0`` when no sample has an absent target.
    """
    pred, true = _none_labels(pred_none, true_none)
    n_true = int(np.sum(true))
    if n_true == 0:
        return 0.0
    return float(np.sum(pred & true) / n_true)


def none_f1(pred_none: Any, true_none: Any) -> float:
    """Harmonic mean of :func:`none_precision` and :func:`none_recall` (``0.0`` if both 0)."""
    precision = none_precision(pred_none, true_none)
    recall = none_recall(pred_none, true_none)
    if precision + recall <= 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def false_selection_rate(
    pred_index: Any,
    target_index: Any,
    target_present: Any,
) -> float:
    """Share of target-absent cases where the model still selects a candidate.

    ``pred_index`` is the emitted candidate index (``-1`` = NONE/abstain) and
    ``target_present`` is the boolean ground-truth flag. ``target_present`` is
    authoritative: rows are classified by it, and ``target_index`` is only
    checked for the "declared present but index == -1" inconsistency. Formally:

        mean over rows with ``target_present == False`` of ``pred_index != -1``

    ``0.0`` is the ideal value (the system always abstains when the target is
    missing); ``1.0`` means it never abstains, which is the risk profile of a
    plain top-1 grounding model in a visual-agent setting.
    """
    pred = as_int_array(pred_index, "pred_index")
    tgt = as_int_array(target_index, "target_index")
    present_arr = to_probability_1d(target_present, "target_present")
    present = present_arr > 0.5
    require_same_length(pred, tgt, "pred_index", "target_index")
    require_same_length(pred, present, "pred_index", "target_present")
    inconsistent = present & (tgt < 0)
    if bool(np.any(inconsistent)):
        raise ValueError(
            f"{int(np.sum(inconsistent))} row(s) are target_present but have target_index < 0"
        )
    n_absent = int(np.sum(~present))
    if n_absent == 0:
        raise ValueError("false_selection_rate needs at least one target-absent sample")
    return float(np.sum((~present) & (pred != -1)) / n_absent)
