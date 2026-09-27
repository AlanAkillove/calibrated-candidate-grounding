"""Plan-audit statistics: the frozen ten-function contract (multi-agent brief).

Every function answers one question of the proposal / candidate-set audit
(protocol sections 4-6) and is **pure numpy, deterministic, and free of silent
filtering**: a pool that is too small yields ``available=False`` *and* a real
:func:`remaining_candidate_count`; nothing is padded, nothing is dropped, and no
denominator is ever invented.  Unknown categories are ``-1`` and excluded from
same-category counts rather than being folded into a known class.

Contract note
-------------
These ten signatures are frozen by the multi-agent brief (2026-09-27) and
supersede the earlier draft of this module (``max_iou_per_object`` /
``recall_summary`` / ``expression_recall`` / ``same_category_availability`` /
``aggregate_same_category`` / ``natural_omission_rates`` / ``audit_image`` and
the dict-returning variants of the names kept below).  Callers of the draft -
``ccg.data.audit_report`` and ``scripts/run_proposal_audit.py`` - must migrate
to :func:`proposal_recall`, :func:`target_availability`,
:func:`redundancy_stats`, :func:`assign_gt_category` /
:func:`same_category_counts` and :func:`natural_omission_rate`; that migration
belongs to those modules' owner (this file is the frozen side of the contract).

Deliberately different from :func:`ccg.data.proposals.proposal_recall` (which
takes per-example ``gt_boxes`` / ``bank_boxes`` sequences and returns the
expression-level hit rate): this module's :func:`proposal_recall` is the
*GT-object level* view - every ground-truth object of one image counts once and
is recalled iff at least one proposal overlaps it at ``iou_thresh``.

Documented empty-input conventions (explicit, never silent)
-----------------------------------------------------------
* :func:`assign_gt_category` on an empty proposal set returns empty arrays;
  with no GT objects every proposal is ``-1`` with best IoU ``0.0``.
* :func:`proposal_recall` with no GT objects is vacuously covered (``1.0``);
  with GT objects but no proposals it is ``0.0``.
* :func:`natural_omission_rate` / :func:`aggregate_availability` over an empty
  population return ``0.0`` (an empty audit population is a caller bug; it must
  surface in the surrounding counts, not crash a reporting script).
* ``wilson_ci(0, 0)`` is ``(0.0, 1.0)`` - no observations, no information.
* :func:`iou_quantiles` of an empty sample is ``nan`` for every requested ``q``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .proposals import DEFAULT_IOU_THRESH, iou_matrix

__all__ = [
    "DEFAULT_IOU_THRESH",
    "DEFAULT_KS",
    "assign_gt_category",
    "proposal_recall",
    "target_availability",
    "aggregate_availability",
    "remaining_candidate_count",
    "redundancy_stats",
    "same_category_counts",
    "natural_omission_rate",
    "wilson_ci",
    "iou_quantiles",
]

#: The Phase 0 cardinality grid (protocol ``Ks_test``).
DEFAULT_KS: Tuple[int, ...] = (5, 10, 20, 50)

_UNKNOWN_CATEGORY = -1


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _as_boxes(boxes: Any, name: str = "boxes") -> np.ndarray:
    """``[N,4]`` float32 view of ``boxes`` (a single ``[4]`` row is accepted)."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"{name} must have shape [N,4], got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _check_thresh(thresh: Any, name: str = "iou_thresh") -> float:
    value = float(thresh)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {thresh!r}")
    return value


def _check_ks(Ks: Sequence[int]) -> List[int]:
    ks = [int(k) for k in Ks]
    for k in ks:
        if k < 1:
            raise ValueError(f"candidate-set sizes must be >= 1, got K={k}")
    return ks


def _blocked_indices(
    n: int, target_idx: Any, to_remove_idx: Any, *, name: str
) -> Tuple[set, Any]:
    """``{target} U to_remove_idx`` as a set of validated row indices.

    ``target_idx is None`` means "natural miss, there is no target" (the
    :func:`ccg.data.proposals.assign_target` convention); ``None`` for
    ``to_remove_idx`` is an empty removal.  Out-of-range indices raise - a stale
    index would silently corrupt every count downstream.
    """
    blocked: set = set()
    if to_remove_idx is not None:
        for raw in np.asarray(to_remove_idx, dtype=np.int64).reshape(-1):
            index = int(raw)
            if not 0 <= index < n:
                raise ValueError(f"{name}: to_remove_idx {index} out of range [0,{n})")
            blocked.add(index)
    if target_idx is not None:
        index = int(target_idx)
        if not 0 <= index < n:
            raise ValueError(f"{name}: target_idx {index} out of range [0,{n})")
        blocked.add(index)
    return blocked, (None if target_idx is None else int(target_idx))


# ---------------------------------------------------------------------------
# GT-object categories and recall
# ---------------------------------------------------------------------------
def assign_gt_category(
    boxes: Any,
    gt_boxes: Any,
    gt_categories: Any,
    iou_thresh: float = DEFAULT_IOU_THRESH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-proposal COCO category from the highest-IoU GT object.

    Each proposal inherits the category of its best-overlapping GT object, but
    only when that overlap is ``>= iou_thresh``; otherwise the category stays
    ``-1`` ("unknown"), which keeps the audit honest about class-agnostic
    proposals that landed on background.  IoU ties resolve to the lowest GT row
    (``argmax`` behaviour - deterministic for a sorted annotation list).

    Returns
    -------
    (categories, best_ious)
        ``categories`` ``int64[N]`` (``-1`` = unknown), ``best_ious``
        ``float32[N]`` - the best GT IoU *even for unknown proposals*, so the
        near-miss distribution stays auditable.
    """
    proposals = _as_boxes(boxes, "boxes")
    gt = _as_boxes(gt_boxes, "gt_boxes")
    cats = np.asarray(gt_categories, dtype=np.int64).reshape(-1)
    if gt.shape[0] != cats.size:
        raise ValueError(f"gt_boxes ({gt.shape[0]}) and gt_categories ({cats.size}) misaligned")
    thresh = _check_thresh(iou_thresh, "iou_thresh")

    n = int(proposals.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
    if gt.shape[0] == 0:
        return (
            np.full(n, _UNKNOWN_CATEGORY, dtype=np.int64),
            np.zeros(n, dtype=np.float32),
        )
    ious = iou_matrix(proposals, gt)  # [N, M]
    best_gt = ious.argmax(axis=1)
    best_ious = ious[np.arange(n), best_gt].astype(np.float32)
    categories = np.where(best_ious >= thresh, cats[best_gt], _UNKNOWN_CATEGORY).astype(np.int64)
    return categories, best_ious


def proposal_recall(prop_boxes: Any, gt_boxes: Any, iou_thresh: float) -> float:
    """GT-object-level recall of a proposal set (the "AR@K as hit rate" view).

    ``recall = |{gt : max_i IoU(p_i, gt) >= iou_thresh}| / |gt|``.  The
    boundary is inclusive: ``IoU == iou_thresh`` counts as covered, exactly as
    :func:`ccg.data.proposals.assign_target` accepts it.

    Empty conventions: no GT objects -> ``1.0`` (vacuously covered); GT objects
    but zero proposals -> ``0.0`` (none of the real objects can be hit).
    """
    proposals = _as_boxes(prop_boxes, "prop_boxes")
    gt = _as_boxes(gt_boxes, "gt_boxes")
    thresh = _check_thresh(iou_thresh, "iou_thresh")
    if gt.shape[0] == 0:
        return 1.0
    if proposals.shape[0] == 0:
        return 0.0
    per_gt_max = iou_matrix(gt, proposals).max(axis=1)  # [M]
    return float(np.mean(per_gt_max >= thresh))


# ---------------------------------------------------------------------------
# candidate availability after target-equivalent removal
# ---------------------------------------------------------------------------
def target_availability(
    prop_boxes: Any,
    target_idx: Any,
    to_remove_idx: Any,
    Ks: Sequence[int] = DEFAULT_KS,
) -> Dict[int, bool]:
    """Which ``K``-way candidate sets can still be built for one example.

    A proposal is a *valid distractor* iff it is neither the target nor one of
    the target's equivalents (``to_remove_idx`` - they would be a second correct
    answer).  A ``K``-set needs the target plus ``K - 1`` distractors, so

        ``available[K] = num_valid_distractors >= K - 1``

    with the real count always recoverable through
    :func:`remaining_candidate_count` / :func:`aggregate_availability`.
    ``target_idx=None`` (natural miss) counts *every* non-removed proposal as a
    distractor - the availability of the pool is a fact about the pool, reported
    separately from the miss.
    """
    boxes = _as_boxes(prop_boxes, "prop_boxes")
    n = int(boxes.shape[0])
    ks = _check_ks(Ks)
    blocked, _ = _blocked_indices(n, target_idx, to_remove_idx, name="target_availability")
    valid_distractors = n - len(blocked)
    return {k: bool(valid_distractors >= k - 1) for k in ks}


def aggregate_availability(
    per_image_avail: List[dict],
    Ks: Sequence[int] = DEFAULT_KS,
) -> Dict[int, float]:
    """``P(available)`` per ``K`` over per-image :func:`target_availability` rows.

    ``per_image_avail`` is the list of per-image availability dicts (int keys as
    returned by :func:`target_availability`; string keys are accepted too, so a
    JSON round-trip cannot silently zero the rate).  A missing ``K`` counts as
    ``False`` for that image - counting it as ``True`` would fabricate capacity
    that was never measured.  An empty population returns ``0.0`` for every
    ``K`` (documented convention: the emptiness must be visible in the
    surrounding ``num_*`` counts).
    """
    entries = list(per_image_avail)
    ks = _check_ks(Ks)
    if not entries:
        return {k: 0.0 for k in ks}
    for position, row in enumerate(entries):
        if not isinstance(row, Mapping):
            raise TypeError(f"per_image_avail[{position}] must be a mapping, got {type(row).__name__}")
    out: Dict[int, float] = {}
    for k in ks:
        hits = 0
        for row in entries:
            if k in row:
                hits += bool(row[k])
            else:
                hits += bool(row.get(str(k), False))
        out[k] = hits / len(entries)
    return out


def remaining_candidate_count(prop_boxes: Any, target_idx: Any, to_remove_idx: Any) -> int:
    """Proposals left once the target's equivalents are deleted (no floor at 0).

    ``= N - |{target} U to_remove_idx|``.  With ``target_idx=None`` (natural
    miss) the target part simply vanishes.  The count is returned as measured -
    it is never clamped to ``K`` or padded, so "pool too small" is visible.
    """
    boxes = _as_boxes(prop_boxes, "prop_boxes")
    n = int(boxes.shape[0])
    blocked, _ = _blocked_indices(n, target_idx, to_remove_idx, name="remaining_candidate_count")
    return int(n - len(blocked))


# ---------------------------------------------------------------------------
# redundancy / same-category distractors
# ---------------------------------------------------------------------------
def redundancy_stats(
    prop_boxes: Any,
    thresh_mid: float = 0.7,
    thresh_high: float = 0.9,
) -> dict:
    """Fraction of proposal *pairs* above the mid / high IoU levels.

    All ``N*(N-1)/2`` unordered pairs are counted (self pairs excluded;
    ``N=128`` gives 8,128 pairs, so the quadratic IoU matrix is irrelevant
    here).  Comparisons are strict (``>``) so a "pair equal to the threshold"
    is not counted as redundant.  Fewer than two proposals returns
    ``n_pairs = 0`` with zero fractions rather than ``nan`` - an audit row must
    always be joinable.

    Returns
    -------
    ``{"frac_pairs_gt_mid": float, "frac_pairs_gt_high": float, "n_pairs": int}``
    """
    boxes = _as_boxes(prop_boxes, "prop_boxes")
    mid = _check_thresh(thresh_mid, "thresh_mid")
    high = _check_thresh(thresh_high, "thresh_high")
    n = int(boxes.shape[0])
    if n < 2:
        return {"frac_pairs_gt_mid": 0.0, "frac_pairs_gt_high": 0.0, "n_pairs": 0}
    ious = iou_matrix(boxes, boxes)
    pair_ious = ious[np.triu_indices(n, k=1)]
    n_pairs = int(pair_ious.size)
    return {
        "frac_pairs_gt_mid": float(np.count_nonzero(pair_ious > mid) / n_pairs),
        "frac_pairs_gt_high": float(np.count_nonzero(pair_ious > high) / n_pairs),
        "n_pairs": n_pairs,
    }


def same_category_counts(
    prop_categories: Any,
    target_idx: Any,
    to_remove_idx: Any,
) -> int:
    """Number of valid (non-removed) distractors sharing the target's category.

    The target's category is read off its *proposal* (``prop_categories`` is
    whatever :func:`assign_gt_category` returned, ``-1`` = unknown).  Unknown
    categories never match - not even against each other - and the target /
    ``to_remove_idx`` rows are excluded, so the count is exactly the supply of
    same-category hard negatives.  ``target_idx=None`` or an unknown target
    category returns ``0`` ("no matchable category", which the caller must
    report as such, not as "no hard negatives exist" silently folded in).
    """
    cats = np.asarray(prop_categories, dtype=np.int64).reshape(-1)
    n = int(cats.size)
    blocked, target = _blocked_indices(n, target_idx, to_remove_idx, name="same_category_counts")
    if target is None:
        return 0
    target_category = int(cats[target])
    if target_category == _UNKNOWN_CATEGORY:
        return 0
    keep = np.ones(n, dtype=bool)
    if blocked:
        keep[np.asarray(sorted(blocked), dtype=np.int64)] = False
    return int(np.count_nonzero(keep & (cats == target_category)))


# ---------------------------------------------------------------------------
# natural omission, confidence intervals, quantiles
# ---------------------------------------------------------------------------
def natural_omission_rate(max_ious: Any, iou_thresh: float = DEFAULT_IOU_THRESH) -> float:
    """``P(max_i IoU(p_i, b*) < iou_thresh)`` - the natural-miss fraction.

    Strictly below the threshold (``IoU == iou_thresh`` is *covered*, matching
    :func:`proposal_recall` and ``assign_target``).  An empty sample returns
    ``0.0`` (documented convention), not ``nan``.
    """
    ious = np.asarray(max_ious, dtype=np.float64).reshape(-1)
    thresh = _check_thresh(iou_thresh, "iou_thresh")
    if ious.size == 0:
        return 0.0
    return float(np.mean(ious < thresh))


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for ``k / n`` successes (95% at ``z=1.96``).

    Preferred over the normal approximation for the audit's binomial rates
    because it stays inside ``[0, 1]`` and behaves at ``k = 0`` / ``k = n``,
    which is exactly where recall/availability numbers live.  The interval is
    clamped to ``[0, 1]`` and its endpoints are *exact*: ``k = 0`` yields
    ``low = 0.0`` and ``k = n`` yields ``high = 1.0`` (the floating-point form
    of the closed expression lands on ``0.9999999999999999`` otherwise).
    ``n = 0`` returns ``(0.0, 1.0)`` (no observations, no information) and
    invalid counts raise.
    """
    successes = int(k)
    total = int(n)
    z_value = float(z)
    if total < 0:
        raise ValueError(f"n must be >= 0, got {n!r}")
    if not 0 <= successes <= total:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k!r}, n={n!r}")
    if z_value <= 0.0:
        raise ValueError(f"z must be positive, got {z!r}")
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    z2 = z_value * z_value
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z_value * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    # boundary counts pin the mathematically exact endpoints; round-off in the
    # closed expression must not leak into them
    if successes == 0:
        low = 0.0
    if successes == total:
        high = 1.0
    return (low, high)


def iou_quantiles(values: Any, qs: Sequence[float] = (0.25, 0.5, 0.75)) -> dict:
    """Linear-interpolation quantiles of an IoU sample, keyed by ``q``.

    ``{q: float}`` for every requested ``q`` (``numpy.quantile`` semantics, so
    ``[0, 0.5, 1]`` at ``q=0.25`` gives ``0.25``).  An empty sample returns
    ``nan`` for every ``q`` - unlike the rate functions, a quantile of nothing
    has no defensible ``0.0`` reading and must stay visibly undefined.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    out: Dict[float, float] = {}
    for raw_q in qs:
        q = float(raw_q)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {raw_q!r}")
        out[q] = float(np.quantile(arr, q)) if arr.size else float("nan")
    return out
