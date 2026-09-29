"""Phase 1E feasibility audit: can the frozen pipeline run on FineCops-Ref at all?

This module owns the F2/F3 measurement and the F4 decision rule.  It is pure
numpy - the driver hands it proposal banks produced by the *frozen* RPN
(``N = 64``, ``ccg.data.rpn``), and it returns per-expression rows plus the
aggregate rates the pre-registered engineering criteria are evaluated on.

Why an engineering gate comes before any science
------------------------------------------------
RefCOCO+ was audited the same way (Amendment A2/A3, ``results/proposal_audit``).
If the class-agnostic COCO-trained RPN cannot even propose the FineCops target,
a low semantic-increment number would be uninterpretable: the experiment would
be measuring ``COCO -> GQA`` perception shift, not candidate-semantic
reliability.  So the criteria below are frozen *before* the audit runs:

* ``target proposal recall@0.5 >= 90%`` **and** ``K=5 availability >= 90%``
  -> ``EXTERNAL GO``;
* ``recall@0.5 < 80%`` -> ``EXTERNAL STOP`` (uninterpretable, report only);
* in between -> ``ENGINEERING GRAY ZONE`` (report, do not touch the generator).

The proposal generator is never adapted to FineCops: no detector swap, no
fine-tuning, no ``N`` re-selection (instruction sections 6/7).  The same rule
applies to the *candidate regime*: ``same-name`` hard cohorts are adopted as
primary only if they clear the same 90% availability bar the random cohort has
to clear, otherwise FineCops' official difficulty level becomes primary
(instruction sections 10/11) and the same-name subset stays a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..data.audit import (
    assign_gt_category,
    proposal_recall,
    redundancy_stats,
    same_category_counts,
    wilson_ci,
)
from ..data.proposals import DEFAULT_IOU_THRESH, assign_target
from ..data.types import ProposalBank
from .finecops import FineCopsExpression

__all__ = [
    "N_PROPOSALS",
    "PRIMARY_K",
    "SECONDARY_K",
    "RECALL_IOUS",
    "GO_RECALL_MIN",
    "GO_AVAILABILITY_MIN",
    "STOP_RECALL_MAX",
    "SAME_CATEGORY_PRIMARY_MIN_AVAILABILITY",
    "SAME_CATEGORY_DIAGNOSTIC_MIN_AVAILABILITY",
    "MIN_GROUP_FOR_TUPLE_TYPE",
    "INVALID_CROP_MIN_SIDE",
    "UNKNOWN_NAME_CODE",
    "ProposalAuditRow",
    "audit_expression",
    "audit_cohort",
    "summarise_rows",
    "engineering_verdict",
    "candidate_regime_decision",
    "build_name_code_table",
    "graph_object_boxes",
    "_iou_tag",
]

#: Frozen proposal budget of the RefCOCO+ pipeline (Amendment A2: N = 64).
N_PROPOSALS = 64
#: External primary cardinality (instruction section 9): K = 5 only.
PRIMARY_K = 5
#: Reported for description, never as an additional primary claim (section 9).
SECONDARY_K = 10
RECALL_IOUS: Tuple[float, ...] = (0.5, 0.7)

# --- pre-registered engineering criteria (instruction section 6) ------------
GO_RECALL_MIN = 0.90
GO_AVAILABILITY_MIN = 0.90
STOP_RECALL_MAX = 0.80
#: Same 90% bar as the random cohort: a primary regime must be *constructible*,
#: not merely occasionally constructible.  Fixed for consistency with the
#: RefCOCO+ rule, not tuned to any observed FineCops number.
SAME_CATEGORY_PRIMARY_MIN_AVAILABILITY = 0.90
#: Below the primary bar but large enough to be reported as a diagnostic cohort.
SAME_CATEGORY_DIAGNOSTIC_MIN_AVAILABILITY = 0.50
#: Instruction section 22: report a tuple type only when it has this many rows.
MIN_GROUP_FOR_TUPLE_TYPE = 300
#: A proposal thinner than this cannot yield a usable CLIP crop.
INVALID_CROP_MIN_SIDE = 4.0
#: ``assign_gt_category``'s "no GT object reached the IoU threshold" code.
UNKNOWN_NAME_CODE = -1


def _iou_tag(threshold: float) -> str:
    """IoU level -> the two-digit suffix used in every key (``0.5 -> "05"``).

    Only one-decimal levels are supported, so ``recall_at_05`` / ``recall_at_07``
    cannot silently collide with a truncated ``0.95 -> "10"``.
    """
    value = float(threshold)
    if abs(value * 10 - round(value * 10)) > 1e-9:
        raise ValueError(f"IoU level {threshold!r} must be a multiple of 0.1")
    return f"{int(round(value * 10)):02d}"


# ---------------------------------------------------------------------------
# per-expression audit
# ---------------------------------------------------------------------------
@dataclass
class ProposalAuditRow:
    """One (expression, frozen ``N=64`` bank) measurement."""

    expr_id: int
    image_id: int
    level: int
    tuple_type: str
    target_name: str
    n_proposals: int
    target_best_iou: float
    target_present: bool
    n_equivalent: int
    n_valid_distractors: int
    n_invalid_crop: int
    n_same_name_distractors: int
    available: Dict[int, bool] = field(default_factory=dict)
    same_name_available: Dict[int, bool] = field(default_factory=dict)
    object_recall: Dict[float, float] = field(default_factory=dict)
    redundancy: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "expr_id": self.expr_id,
            "image_id": self.image_id,
            "level": self.level,
            "tuple_type": self.tuple_type,
            "target_name": self.target_name,
            "n_proposals": self.n_proposals,
            "target_best_iou": round(float(self.target_best_iou), 6),
            "target_present_at_05": int(self.target_present),
            "n_equivalent": self.n_equivalent,
            "n_valid_distractors": self.n_valid_distractors,
            "n_invalid_crop": self.n_invalid_crop,
            "n_same_name_distractors": self.n_same_name_distractors,
        }
        for k, flag in sorted(self.available.items()):
            row[f"available_K{k}"] = int(bool(flag))
        for k, flag in sorted(self.same_name_available.items()):
            row[f"same_name_available_K{k}"] = int(bool(flag))
        for iou, rate in sorted(self.object_recall.items()):
            row[f"object_recall_at_{_iou_tag(iou)}"] = round(float(rate), 6)
        row["frac_pairs_iou_gt_07"] = round(float(self.redundancy.get("frac_pairs_gt_mid", 0.0)), 6)
        return row


def graph_object_boxes(
    objects: Mapping[str, Mapping[str, Any]], name_codes: Mapping[str, int]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """``(boxes xyxy, name codes, object ids)`` of one scene graph, id-sorted."""
    ids = sorted(objects.keys())
    boxes = np.zeros((len(ids), 4), dtype=np.float32)
    codes = np.full(len(ids), UNKNOWN_NAME_CODE, dtype=np.int64)
    for row, oid in enumerate(ids):
        o = objects[oid]
        boxes[row] = (o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"])
        codes[row] = int(name_codes.get(str(o.get("name")), UNKNOWN_NAME_CODE))
    return boxes, codes, ids


def build_name_code_table(
    expressions: Sequence[FineCopsExpression],
    names_by_image: Mapping[str, Iterable[str]],
) -> Dict[str, int]:
    """Stable ``GQA object name -> int`` codes over the audited images.

    The codes only make :func:`ccg.data.audit.assign_gt_category` usable: a code
    is *exact name identity*, which is the same equivalence relation FineCops'
    own level definition uses.  No hypernym, no COCO vocabulary, no human
    grouping is introduced.
    """
    del expressions  # kept for signature symmetry; the table is name-driven
    names: set[str] = set()
    for rows in names_by_image.values():
        names.update(str(n) for n in rows)
    return {name: code for code, name in enumerate(sorted(names))}


def _invalid_crop_count(bank: ProposalBank, image_w: int, image_h: int) -> int:
    """Proposals that cannot produce a usable crop (degenerate or off-frame)."""
    boxes = np.asarray(bank.boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return 0
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    degenerate = (w < INVALID_CROP_MIN_SIDE) | (h < INVALID_CROP_MIN_SIDE)
    off_frame = (boxes[:, 0] < -1) | (boxes[:, 1] < -1) | (boxes[:, 2] > image_w + 1) | (
        boxes[:, 3] > image_h + 1
    )
    return int(np.count_nonzero(degenerate | off_frame))


def audit_expression(
    expr: FineCopsExpression,
    bank: ProposalBank,
    *,
    gt_object_boxes_xyxy: np.ndarray,
    gt_object_name_codes: np.ndarray,
    iou_thresh: float = DEFAULT_IOU_THRESH,
    Ks: Sequence[int] = (PRIMARY_K, SECONDARY_K),
    recall_ious: Sequence[float] = RECALL_IOUS,
) -> ProposalAuditRow:
    """Measure one expression against the frozen bank.

    Target assignment, equivalent removal and same-category supply reuse the
    RefCOCO+ audit primitives verbatim, so "available" means exactly the same
    thing on both datasets.  ``gt_object_name_codes`` encodes the GQA object
    name of every graph object, so ``same_category_counts`` counts *same-name*
    distractors - the external analogue of the A8 same-category cohort.
    """
    ks = [int(k) for k in Ks]
    assignment = assign_target(bank, expr.gt_box, iou_thresh=iou_thresh)
    categories, _best = assign_gt_category(
        bank.boxes, gt_object_boxes_xyxy, gt_object_name_codes, iou_thresh=iou_thresh
    )
    n_same = same_category_counts(
        categories, assignment.target_proposal_idx, assignment.to_remove
    )
    pool = int(bank.boxes.shape[0]) - (
        (0 if assignment.target_proposal_idx is None else 1) + int(assignment.to_remove.size)
    )
    available = {
        k: bool(assignment.target_proposal_idx is not None and pool >= k - 1) for k in ks
    }
    same_available = {k: bool(available[k] and n_same >= k - 1) for k in ks}
    object_recall = {
        float(t): (
            proposal_recall(bank.boxes, gt_object_boxes_xyxy, t)
            if gt_object_boxes_xyxy.shape[0]
            else float("nan")
        )
        for t in recall_ious
    }
    return ProposalAuditRow(
        expr_id=expr.expr_id,
        image_id=expr.image_id,
        level=expr.level,
        tuple_type=expr.tuple_type,
        target_name=expr.target_name or "",
        n_proposals=int(bank.boxes.shape[0]),
        target_best_iou=float(assignment.best_iou),
        target_present=bool(assignment.target_proposal_idx is not None),
        n_equivalent=int(assignment.num_equivalent),
        n_valid_distractors=max(0, pool),
        n_invalid_crop=_invalid_crop_count(bank, expr.width, expr.height),
        n_same_name_distractors=int(n_same),
        available=available,
        same_name_available=same_available,
        object_recall=object_recall,
        redundancy=redundancy_stats(bank.boxes),
    )


def audit_cohort(
    rows: Sequence[ProposalAuditRow],
) -> List[Dict[str, Any]]:
    """Serialise audit rows (the body of ``rpn_audit.csv``)."""
    return [r.to_row() for r in rows]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _rate(values: Sequence[bool] | Sequence[int]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def summarise_rows(
    rows: Sequence[ProposalAuditRow],
    *,
    Ks: Sequence[int] = (PRIMARY_K, SECONDARY_K),
    recall_ious: Sequence[float] = RECALL_IOUS,
) -> Dict[str, Any]:
    """Aggregate an audit into the numbers the F4 decision consumes.

    Rates are computed over *expressions* (the unit a candidate set is built
    for); image-level clustering is not needed here because no significance
    claim is made at the feasibility stage, but Wilson intervals are reported
    so the sampling noise of the audit subset stays visible.
    """
    ks = [int(k) for k in Ks]
    if not rows:
        return {"n_rows": 0}
    covered = [r for r in rows if r.target_present]
    out: Dict[str, Any] = {
        "n_rows": len(rows),
        "n_images": len({r.image_id for r in rows}),
        "n_names": len({r.target_name for r in rows if r.target_name}),
        "mean_bank_size": round(float(np.mean([r.n_proposals for r in rows])), 3),
        "min_bank_size": int(min(r.n_proposals for r in rows)),
        "target_max_iou": {
            "mean": round(float(np.mean([r.target_best_iou for r in rows])), 6),
            "p10": float(np.quantile([r.target_best_iou for r in rows], 0.10)),
            "p25": float(np.quantile([r.target_best_iou for r in rows], 0.25)),
            "median": float(np.median([r.target_best_iou for r in rows])),
            "p75": float(np.quantile([r.target_best_iou for r in rows], 0.75)),
            "p90": float(np.quantile([r.target_best_iou for r in rows], 0.90)),
        },
        "n_target_present": len(covered),
        "natural_omission_at_05": round(1.0 - _rate([r.target_present for r in rows]), 6),
        "mean_n_equivalent": round(float(np.mean([r.n_equivalent for r in rows])), 4),
        "mean_n_valid_distractors": round(
            float(np.mean([r.n_valid_distractors for r in rows])), 3
        ),
        "invalid_crop_rate": round(
            float(
                np.sum([r.n_invalid_crop for r in rows])
                / max(1, int(np.sum([r.n_proposals for r in rows])))
            ),
            6,
        ),
        "redundancy_frac_pairs_gt_07": round(
            float(np.mean([r.redundancy.get("frac_pairs_gt_mid", 0.0) for r in rows])), 6
        ),
    }
    for t in recall_ious:
        key = f"recall_at_{_iou_tag(t)}"
        hits = _rate([r.target_best_iou >= float(t) for r in rows])
        lo, hi = wilson_ci(int(round(hits * len(rows))), len(rows))
        out[key] = {"rate": round(hits, 6), "wilson_low": round(lo, 6), "wilson_high": round(hi, 6)}
        obj = [r.object_recall.get(float(t)) for r in rows]
        obj = [v for v in obj if v is not None and np.isfinite(v)]
        out[f"gt_object_{key}"] = round(float(np.mean(obj)), 6) if obj else float("nan")
    for k in ks:
        hits = _rate([r.available.get(k, False) for r in rows])
        lo, hi = wilson_ci(int(round(hits * len(rows))), len(rows))
        same = _rate([r.same_name_available.get(k, False) for r in rows])
        slo, shi = wilson_ci(int(round(same * len(rows))), len(rows))
        out[f"k{k}_availability"] = {
            "rate": round(hits, 6),
            "wilson_low": round(lo, 6),
            "wilson_high": round(hi, 6),
            "same_name_rate": round(same, 6),
            "same_name_wilson_low": round(slo, 6),
            "same_name_wilson_high": round(shi, 6),
        }
    same_supply = [r.n_same_name_distractors for r in rows]
    out["same_name_distractors"] = {
        "mean": round(float(np.mean(same_supply)), 4),
        "median": float(np.median(same_supply)),
        "p90": float(np.quantile(same_supply, 0.90)),
        "max": int(max(same_supply)),
        "share_ge_1": round(_rate([s >= 1 for s in same_supply]), 6),
        "share_ge_4": round(_rate([s >= PRIMARY_K - 1 for s in same_supply]), 6),
        "share_ge_9": round(_rate([s >= SECONDARY_K - 1 for s in same_supply]), 6),
    }
    out["by_level"] = {}
    for level in sorted({r.level for r in rows}):
        sub = [r for r in rows if r.level == level]
        out["by_level"][str(level)] = {
            "n_rows": len(sub),
            "n_images": len({r.image_id for r in sub}),
            "recall_at_05": _rate([r.target_best_iou >= 0.5 for r in sub]),
            "recall_at_07": _rate([r.target_best_iou >= 0.7 for r in sub]),
            f"k{PRIMARY_K}_availability": _rate([r.available.get(PRIMARY_K, False) for r in sub]),
            f"k{PRIMARY_K}_same_name_availability": _rate(
                [r.same_name_available.get(PRIMARY_K, False) for r in sub]
            ),
            "same_name_distractors_mean": round(
                float(np.mean([r.n_same_name_distractors for r in sub])), 4
            ),
        }
    out["by_tuple_type"] = {}
    for tt in sorted({r.tuple_type for r in rows}):
        sub = [r for r in rows if r.tuple_type == tt]
        out["by_tuple_type"][tt] = {
            "n_rows": len(sub),
            "reportable": bool(len(sub) >= MIN_GROUP_FOR_TUPLE_TYPE),
            "recall_at_05": round(_rate([r.target_best_iou >= 0.5 for r in sub]), 6),
            f"k{PRIMARY_K}_availability": round(
                _rate([r.available.get(PRIMARY_K, False) for r in sub]), 6
            ),
            f"k{PRIMARY_K}_same_name_availability": round(
                _rate([r.same_name_available.get(PRIMARY_K, False) for r in sub]), 6
            ),
        }
    return out


# ---------------------------------------------------------------------------
# decisions (F4)
# ---------------------------------------------------------------------------
def engineering_verdict(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the frozen section-6 criteria to one audit summary."""
    recall = float(summary.get("recall_at_05", {}).get("rate", float("nan")))
    k5 = float(summary.get(f"k{PRIMARY_K}_availability", {}).get("rate", float("nan")))
    conditions = {
        f"recall_at_05_ge_{GO_RECALL_MIN:.2f}": bool(recall >= GO_RECALL_MIN),
        f"k{PRIMARY_K}_availability_ge_{GO_AVAILABILITY_MIN:.2f}": bool(k5 >= GO_AVAILABILITY_MIN),
    }
    if not np.isfinite(recall) or not np.isfinite(k5):
        verdict = "AUDIT INCOMPLETE"
    elif all(conditions.values()):
        verdict = "EXTERNAL GO"
    elif recall < STOP_RECALL_MAX:
        verdict = "EXTERNAL STOP"
    else:
        verdict = "ENGINEERING GRAY ZONE"
    return {
        "criteria": {
            "go_recall_min": GO_RECALL_MIN,
            "go_availability_min": GO_AVAILABILITY_MIN,
            "stop_recall_max": STOP_RECALL_MAX,
            "n_proposals": N_PROPOSALS,
            "primary_k": PRIMARY_K,
        },
        "measured": {"recall_at_05": recall, f"k{PRIMARY_K}_availability": k5},
        "conditions": conditions,
        "verdict": verdict,
        "note": (
            "proposal-generator changes are forbidden as a response to this "
            "verdict (instruction sections 6/7)"
        ),
    }


def candidate_regime_decision(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Choose the external primary candidate regime (instruction sections 10/11).

    ``same_category_primary`` requires GQA same-name distractors for the primary
    K at the same 90% constructibility bar the random cohort must clear.  If it
    fails but the subset is still large, that subset is kept as a *diagnostic*
    and the official difficulty level carries the primary analysis; below the
    diagnostic bar even that is dropped rather than padded with a fabricated
    category mapping.
    """
    k5 = summary.get(f"k{PRIMARY_K}_availability", {}) or {}
    same = float(k5.get("same_name_rate", float("nan")))
    same_wilson_low = float(k5.get("same_name_wilson_low", float("nan")))
    supply = summary.get("same_name_distractors", {}) or {}
    name_resolution = summary.get("n_names", 0)
    if same >= SAME_CATEGORY_PRIMARY_MIN_AVAILABILITY:
        branch = "same_category_primary"
    elif same >= SAME_CATEGORY_DIAGNOSTIC_MIN_AVAILABILITY:
        branch = "level_primary_same_category_diagnostic"
    else:
        branch = "level_primary_only"
    return {
        "thresholds": {
            "primary_min_availability": SAME_CATEGORY_PRIMARY_MIN_AVAILABILITY,
            "diagnostic_min_availability": SAME_CATEGORY_DIAGNOSTIC_MIN_AVAILABILITY,
        },
        "measured": {
            "same_name_k5_availability": same,
            "same_name_k5_availability_wilson_low": same_wilson_low,
            "share_rows_ge_4_same_name_distractors": supply.get("share_ge_4"),
            "n_distinct_target_names": name_resolution,
        },
        "branch": branch,
        "primary_hard_regime": {
            "same_category_primary": "GQA same-name K=5 candidates",
            "level_primary_same_category_diagnostic": "FineCops official level (2/3 vs 1), random K=5",
            "level_primary_only": "FineCops official level (2/3 vs 1), random K=5",
        }[branch],
        "secondary": (
            "level + tuple_type stratification"
            if branch != "same_category_primary"
            else "level + tuple_type stratification on the same-name cohort"
        ),
        "rationale": (
            "same-name distractor supply decides constructibility; no COCO "
            "category mapping is invented for a GQA benchmark"
        ),
    }
