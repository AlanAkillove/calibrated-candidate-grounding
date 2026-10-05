"""V4 practical-utility metrics for fixed prediction rows.

This module is deliberately independent of the V3 public API.  It contains
only deterministic metric helpers: callers provide the already-saved labels,
confidence scores, group masks, and (for a shared-image bootstrap draw) the
integer row multiplicities induced by that image draw.

The V4 AURC integrates fractional-boundary selective risk continuously over
coverage.  Equal-score observations are treated as a uniformly randomized
tie group, so the integral is invariant to input row order.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_COVERAGES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
AURC_VERSION = "AURC_V4_CONTINUOUS_FRACTIONAL_TIES"
_PAIR_STATES = ("loss", "tie", "win")


def _coerce(
    scores: Any,
    correctness: Any,
    multiplicities: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confidence = np.asarray(scores, dtype=np.float64).reshape(-1)
    raw_labels = np.asarray(correctness, dtype=np.float64).reshape(-1)
    if confidence.shape != raw_labels.shape:
        raise ValueError("scores and correctness must have equal lengths")
    if not np.isfinite(confidence).all():
        raise ValueError("scores must be finite")
    if not np.isfinite(raw_labels).all() or np.any(~np.isin(raw_labels, (0.0, 1.0))):
        raise ValueError("correctness must contain only finite binary 0/1 values")
    labels = raw_labels.astype(bool)
    if multiplicities is None:
        weights = np.ones(confidence.size, dtype=np.int64)
    else:
        raw_weights = np.asarray(multiplicities)
        if raw_weights.shape != confidence.shape:
            raise ValueError("integer multiplicities must align with scores")
        if not np.issubdtype(raw_weights.dtype, np.number):
            raise ValueError("multiplicities must be finite nonnegative integers")
        as_float = raw_weights.astype(np.float64)
        if (
            not np.isfinite(as_float).all()
            or np.any(as_float < 0)
            or np.any(as_float != np.floor(as_float))
            or np.any(as_float > np.iinfo(np.int64).max)
        ):
            raise ValueError("multiplicities must be finite nonnegative integers")
        weights = as_float.astype(np.int64)
    return confidence, labels, weights


def _score_groups(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return descending distinct scores and weighted row/error counts."""
    active = weights > 0
    if not np.any(active):
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    order = np.argsort(-scores[active], kind="stable")
    s = scores[active][order]
    y = labels[active][order]
    w = weights[active][order].astype(np.float64)
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    group_n = np.add.reduceat(w, starts)
    group_errors = np.add.reduceat(w * (~y), starts)
    return s[starts], group_n, group_errors


def fractional_risk_curve(
    scores: Any,
    correctness: Any,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    *,
    multiplicities: Any | None = None,
) -> dict[str, Any]:
    """Evaluate fractional-boundary risk at requested coverages.

    The zero-coverage risk is undefined.  Its right limit is the error rate
    inside the highest-confidence tie group and is returned in an explicitly
    tagged field for plotting the curve near zero.
    """
    confidence, labels, weights = _coerce(scores, correctness, multiplicities)
    levels = np.asarray(tuple(coverages), dtype=np.float64)
    if levels.ndim != 1 or not np.isfinite(levels).all() or np.any((levels <= 0) | (levels > 1)):
        raise ValueError("coverages must be finite values in (0, 1]")
    if np.unique(levels).size != levels.size:
        raise ValueError("coverages must not contain duplicates")
    _, group_n, group_errors = _score_groups(confidence, labels, weights)
    total = float(group_n.sum())
    right_limit = float(group_errors[0] / group_n[0]) if total > 0 else float("nan")
    risks: dict[str, float | None] = {}
    if total == 0:
        risks = {str(float(level)): None for level in levels}
    else:
        cumulative_n = np.cumsum(group_n)
        cumulative_errors = np.cumsum(group_errors)
        for level in levels:
            budget = float(level * total)
            group = int(np.searchsorted(cumulative_n, budget, side="left"))
            before_n = float(cumulative_n[group - 1]) if group else 0.0
            before_errors = float(cumulative_errors[group - 1]) if group else 0.0
            fraction = (budget - before_n) / float(group_n[group])
            risk = (before_errors + fraction * float(group_errors[group])) / budget
            risks[str(float(level))] = float(risk)
    return {
        "coverage_zero": {
            "risk": None,
            "status": "UNDEFINED_AT_ZERO",
            "right_limit": right_limit if np.isfinite(right_limit) else None,
            "right_limit_tag": "RISK_COVERAGE_ZERO_RIGHT_LIMIT",
        },
        "risks": risks,
        "n_effective": int(total),
    }


def aurc_v4(
    scores: Any,
    correctness: Any,
    *,
    multiplicities: Any | None = None,
) -> float:
    """Continuous AURC with fractional interpolation within every score tie.

    For each tie group of weighted size ``g`` and errors ``h``, let ``m`` and
    ``e`` be the number of rows and errors above it and ``p=h/g``.  Its exact
    area is ``(p*g + (e-p*m)*log((m+g)/m))/N`` when ``m>0`` and ``p*g/N`` for
    the first group.  The empty bootstrap draw is undefined and returns NaN.
    """
    confidence, labels, weights = _coerce(scores, correctness, multiplicities)
    _, group_n, group_errors = _score_groups(confidence, labels, weights)
    total = float(group_n.sum())
    if total == 0:
        return float("nan")
    count_before = 0.0
    errors_before = 0.0
    area = 0.0
    for size, errors in zip(group_n, group_errors, strict=True):
        proportion = float(errors / size)
        if count_before == 0.0:
            group_area = proportion * float(size)
        else:
            group_area = proportion * float(size) + (
                errors_before - proportion * count_before
            ) * float(np.log((count_before + size) / count_before))
        area += group_area
        count_before += float(size)
        errors_before += float(errors)
    return float(area / total)


def oracle_aurc_v4(accuracy: float) -> float:
    """Oracle AURC given accuracy, using the frozen V4 definition.

    ``(1-a) + a*log(a)`` is evaluated with the explicit endpoint limits 1 at
    zero accuracy and 0 at perfect accuracy.
    """
    value = float(accuracy)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("accuracy must be finite and lie in [0, 1]")
    if value == 0.0:
        return 1.0
    if value == 1.0:
        return 0.0
    return float((1.0 - value) + value * np.log(value))


def e_aurc_v4(
    scores: Any,
    correctness: Any,
    *,
    multiplicities: Any | None = None,
) -> float:
    """V4 excess AURC; this definition remains dependent on base accuracy."""
    _, labels, weights = _coerce(scores, correctness, multiplicities)
    total = int(weights.sum())
    if total == 0:
        return float("nan")
    accuracy = float(np.dot(weights, labels.astype(np.int64)) / total)
    return float(aurc_v4(scores, correctness, multiplicities=weights) - oracle_aurc_v4(accuracy))


def weighted_auroc(
    scores: Any,
    correctness: Any,
    *,
    multiplicities: Any | None = None,
) -> float:
    """Correctness AUROC from exact positive-negative pair credits.

    A pair scores one for a correct row above an incorrect row, one-half for a
    tie, and zero for a loss.  Single-class and empty inputs are undefined.
    """
    return float(pairwise_credit_counts(scores, correctness, multiplicities=multiplicities)["auroc"])


def pairwise_credit_counts(
    scores: Any,
    correctness: Any,
    *,
    multiplicities: Any | None = None,
) -> dict[str, int | float | None]:
    """Count all weighted positive-negative wins, ties, and losses in O(n log n)."""
    confidence, labels, weights = _coerce(scores, correctness, multiplicities)
    active = weights > 0
    if not np.any(active):
        return {
            "n_positive": 0,
            "n_negative": 0,
            "pair_count": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "credit": 0.0,
            "auroc": None,
        }
    positive_count = int(weights[labels].sum())
    negative_count = int(weights[~labels].sum())
    pairs = positive_count * negative_count
    if pairs == 0:
        return {
            "n_positive": positive_count,
            "n_negative": negative_count,
            "pair_count": 0,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "credit": None,
            "auroc": None,
        }

    order = np.argsort(-confidence[active], kind="stable")
    sorted_scores = confidence[active][order]
    sorted_labels = labels[active][order]
    sorted_weights = weights[active][order].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
    pos_by_group = np.add.reduceat(sorted_weights * sorted_labels, starts)
    neg_by_group = np.add.reduceat(sorted_weights * ~sorted_labels, starts)
    neg_before = np.r_[0, np.cumsum(neg_by_group, dtype=np.int64)[:-1]]
    neg_after = np.cumsum(neg_by_group[::-1], dtype=np.int64)[::-1] - neg_by_group
    wins = int(np.dot(pos_by_group, neg_after))
    ties = int(np.dot(pos_by_group, neg_by_group))
    losses = int(np.dot(pos_by_group, neg_before))
    credit = float(wins + 0.5 * ties)
    if wins + ties + losses != pairs:
        raise AssertionError("weighted pair credit counts do not sum to all positive-negative pairs")
    return {
        "n_positive": positive_count,
        "n_negative": negative_count,
        "pair_count": pairs,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "credit": credit,
        "auroc": credit / pairs,
    }


def pairwise_transition_counts(
    full_scores: Any,
    sq_scores: Any,
    correctness: Any,
    *,
    multiplicities: Any | None = None,
    block_size: int = 128,
) -> dict[str, Any]:
    """Exactly count SQ-to-Full pair transitions using bounded memory.

    The standard AUROC pair universe contains every correct-incorrect row
    pair, including expressions from the same image and from different images.
    ``multiplicities`` must be the row counts from one shared image-cluster
    draw; pair multiplicity is the product of its endpoint row counts.  The
    positive rows are traversed in blocks so the temporary comparison arrays
    have size at most ``block_size * n_negative``.
    """
    full, labels, weights = _coerce(full_scores, correctness, multiplicities)
    sq, sq_labels, sq_weights = _coerce(sq_scores, correctness, multiplicities)
    if sq.shape != full.shape or not np.array_equal(labels, sq_labels) or not np.array_equal(weights, sq_weights):
        raise ValueError("Full and S+Q scores must share labels and row multiplicities")
    block = int(block_size)
    if block <= 0:
        raise ValueError("block_size must be a positive integer")
    positive = np.flatnonzero(labels & (weights > 0))
    negative = np.flatnonzero((~labels) & (weights > 0))
    n_positive = int(weights[positive].sum())
    n_negative = int(weights[negative].sum())
    pair_count = n_positive * n_negative

    # Rows index S+Q states; columns index Full states.  State order is loss,
    # tie, win, matching the exact Mann-Whitney pair-credit convention.
    matrix = np.zeros((3, 3), dtype=np.int64)
    for start in range(0, positive.size, block):
        rows = positive[start : start + block]
        full_state = np.sign(full[rows, None] - full[negative][None, :]).astype(np.int8)
        sq_state = np.sign(sq[rows, None] - sq[negative][None, :]).astype(np.int8)
        pair_weights = weights[rows, None] * weights[negative][None, :]
        flat_cells = ((sq_state + 1) * 3 + (full_state + 1)).reshape(-1)
        flat_weights = pair_weights.reshape(-1)
        np.add.at(matrix.reshape(-1), flat_cells, flat_weights)
    if int(matrix.sum()) != pair_count:
        raise AssertionError("transition matrix does not cover the complete pair universe")

    state_credits = np.asarray([0.0, 0.5, 1.0])
    sq_credit = float(np.dot(matrix.sum(axis=1), state_credits))
    full_credit = float(np.dot(matrix.sum(axis=0), state_credits))
    delta_credit = full_credit - sq_credit
    transition_map = {
        f"{_PAIR_STATES[row]}_to_{_PAIR_STATES[column]}": int(matrix[row, column])
        for row in range(3)
        for column in range(3)
    }
    tie_changes = {
        "loss_to_tie": transition_map["loss_to_tie"],
        "tie_to_win": transition_map["tie_to_win"],
        "win_to_tie": transition_map["win_to_tie"],
        "tie_to_loss": transition_map["tie_to_loss"],
    }

    def model_counts(row_axis: bool) -> dict[str, int | float | None]:
        marginal = matrix.sum(axis=1 if row_axis else 0)
        wins, ties, losses = (int(value) for value in marginal[[2, 1, 0]])
        credit = float(wins + 0.5 * ties)
        return {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "credit": credit if pair_count else None,
            "auroc": credit / pair_count if pair_count else None,
        }

    return {
        "pair_scope": "all_correct_incorrect_expression_pairs_including_within_and_between_image",
        "n_positive": n_positive,
        "n_negative": n_negative,
        "pair_count": pair_count,
        "state_order": list(_PAIR_STATES),
        "models": {"S+Q": model_counts(True), "Full": model_counts(False)},
        "transition_matrix": {
            _PAIR_STATES[row]: {
                _PAIR_STATES[column]: int(matrix[row, column]) for column in range(3)
            }
            for row in range(3)
        },
        "transitions": transition_map,
        "regained_loss_to_win": transition_map["loss_to_win"],
        "regressed_win_to_loss": transition_map["win_to_loss"],
        "tie_changes": tie_changes,
        "net_credit_full_minus_sq": delta_credit if pair_count else None,
        "net_credit_per_10000_pairs": delta_credit / pair_count * 10000.0 if pair_count else None,
    }


def practical_profile(
    scores: Any,
    correctness: Any,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    multiplicities: Any | None = None,
) -> dict[str, Any]:
    """Return the absolute AURC, AUROC, and fixed-coverage risk profile."""
    confidence, labels, weights = _coerce(scores, correctness, multiplicities)
    total = int(weights.sum())
    positives = int(weights[labels].sum())
    negatives = total - positives
    auc_counts = pairwise_credit_counts(confidence, labels, multiplicities=weights)
    curve = fractional_risk_curve(
        confidence, labels, coverages, multiplicities=weights
    )
    aurc = aurc_v4(confidence, labels, multiplicities=weights)
    accuracy = float(positives / total) if total else float("nan")
    oracle = oracle_aurc_v4(accuracy) if total else float("nan")
    return {
        "n_effective": total,
        "n_correct": positives,
        "n_incorrect": negatives,
        "accuracy": accuracy if np.isfinite(accuracy) else None,
        "auroc_correct": auc_counts["auroc"],
        "pair_counts": auc_counts,
        "aurc_version": AURC_VERSION,
        "aurc_v4": aurc if np.isfinite(aurc) else None,
        "oracle_aurc_v4": oracle if np.isfinite(oracle) else None,
        "e_aurc_v4": (aurc - oracle) if np.isfinite(aurc) and np.isfinite(oracle) else None,
        "aurc_right_limit_at_zero": curve["coverage_zero"],
        "risk_by_coverage": curve["risks"],
    }


def paired_aurc_effect(
    full_scores: Any,
    sq_scores: Any,
    correctness: Any,
    *,
    coverages: Sequence[float] = DEFAULT_COVERAGES,
    multiplicities: Any | None = None,
) -> dict[str, Any]:
    """Return Full-minus-S+Q AURC and AUROC effects on shared labels."""
    full, labels, weights = _coerce(full_scores, correctness, multiplicities)
    sq, sq_labels, sq_weights = _coerce(sq_scores, correctness, multiplicities)
    if sq.shape != full.shape or not np.array_equal(labels, sq_labels) or not np.array_equal(weights, sq_weights):
        raise ValueError("Full and S+Q scores must share labels and row multiplicities")
    profile_full = practical_profile(full, labels, coverages=coverages, multiplicities=weights)
    profile_sq = practical_profile(sq, labels, coverages=coverages, multiplicities=weights)
    full_aurc, sq_aurc = profile_full["aurc_v4"], profile_sq["aurc_v4"]
    full_eaurc, sq_eaurc = profile_full["e_aurc_v4"], profile_sq["e_aurc_v4"]
    full_auc, sq_auc = profile_full["auroc_correct"], profile_sq["auroc_correct"]
    delta_aurc = float(full_aurc - sq_aurc) if full_aurc is not None and sq_aurc is not None else None
    delta_eaurc = float(full_eaurc - sq_eaurc) if full_eaurc is not None and sq_eaurc is not None else None
    if delta_aurc is not None and delta_eaurc is not None and not np.isclose(
        delta_aurc, delta_eaurc, atol=1e-12, rtol=0
    ):
        raise AssertionError("same-label Delta E-AURC must equal Delta AURC")
    return {
        "estimator": "paired Full-minus-S+Q on the same labels and rows",
        "Full": profile_full,
        "S+Q": profile_sq,
        "delta_aurc_full_minus_sq": delta_aurc,
        "delta_e_aurc_v4_full_minus_sq": delta_eaurc,
        "delta_auroc_full_minus_sq": (
            float(full_auc - sq_auc) if full_auc is not None and sq_auc is not None else None
        ),
    }


def high_low_auroc_effect(
    full_scores: Any,
    sq_scores: Any,
    correctness: Any,
    high_mask: Any,
    *,
    multiplicities: Any | None = None,
) -> dict[str, Any]:
    """Compute the high-minus-low difference in paired AUROC effects.

    The caller owns the group definition and supplies the frozen high-group
    mask.  Each group must contain both correctness classes for a defined
    AUROC; otherwise its AUROCs and the interaction remain undefined.
    """
    full, labels, weights = _coerce(full_scores, correctness, multiplicities)
    sq, sq_labels, sq_weights = _coerce(sq_scores, correctness, multiplicities)
    high = np.asarray(high_mask, dtype=bool).reshape(-1)
    if high.shape != full.shape:
        raise ValueError("high_mask must align with score rows")
    if sq.shape != full.shape or not np.array_equal(labels, sq_labels) or not np.array_equal(weights, sq_weights):
        raise ValueError("Full and S+Q scores must share labels and row multiplicities")
    result: dict[str, Any] = {"group_definition": "caller_supplied_high_mask; low is its complement"}
    effects: dict[str, float | None] = {}
    for name, mask in (("high", high), ("low", ~high)):
        group_weights = weights * mask
        full_counts = pairwise_credit_counts(full, labels, multiplicities=group_weights)
        sq_counts = pairwise_credit_counts(sq, labels, multiplicities=group_weights)
        full_auc, sq_auc = full_counts["auroc"], sq_counts["auroc"]
        delta = float(full_auc - sq_auc) if full_auc is not None and sq_auc is not None else None
        effects[name] = delta
        result[name] = {
            "n_effective": int(group_weights.sum()),
            "pair_count": int(full_counts["pair_count"]),
            "Full": full_counts,
            "S+Q": sq_counts,
            "delta_auroc_full_minus_sq": delta,
        }
    high_effect, low_effect = effects["high"], effects["low"]
    result["high_minus_low_delta_auroc"] = (
        float(high_effect - low_effect)
        if high_effect is not None and low_effect is not None
        else None
    )
    return result


def image_draw_row_multiplicities(
    image_ids: Any,
    sampled_images: Any,
) -> np.ndarray:
    """Convert a sampled image-id sequence to one multiplicity per input row.

    This helper preserves every expression belonging to each sampled image.
    It does not resample expressions independently.
    """
    images = np.asarray(image_ids).reshape(-1)
    draws = np.asarray(sampled_images).reshape(-1)
    if not images.size:
        raise ValueError("image_ids must be nonempty")
    if not np.isin(draws, np.unique(images)).all():
        raise ValueError("sampled_images contains an image absent from image_ids")
    unique_images, inverse = np.unique(images, return_inverse=True)
    draw_counts = np.asarray([np.count_nonzero(draws == image) for image in unique_images], dtype=np.int64)
    return draw_counts[inverse]

