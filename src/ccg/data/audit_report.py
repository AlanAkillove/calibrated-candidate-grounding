"""Aggregation and artifacts for the proposal audit (pure numpy + stdlib).

The audit driver (:mod:`scripts.run_proposal_audit`) produces one *image
record* per (image, truncation ``N``):

* per-GT-object max IoUs of the top-N bank plus the GT-object recall counts
  returned by :func:`ccg.data.audit.proposal_recall` (``gt_recall_counts``);
* the bank's pairwise redundancy (``redundancy_stats``: ``n_pairs`` +
  ``frac_pairs_gt_mid`` / ``frac_pairs_gt_high``);
* one entry per referring expression (``expressions``): max IoU of the
  RefCOCO+ target, natural-miss flag, remaining-candidate count after
  target-equivalent removal (``remaining_candidate_count``), same-category
  distractor count (``same_category_counts``) and the per-K availability flags
  returned by :func:`ccg.data.audit.target_availability`.

This module turns those records into the committed artifacts, without a GPU,
a model or the on-disk caches:

    results/proposal_audit/
        summary.json                       (written by the script itself)
        recall_by_N.csv
        candidate_availability_by_N.csv
        same_category_availability.csv
        natural_omission.csv
        proposal_iou_statistics.csv
        figures/*.png                      (render_figures, --figures)

Statistics follow the frozen ten-function contract of :mod:`ccg.data.audit`
(2026-09-27); this module re-exports its :func:`wilson_ci` instead of defining
its own, and inherits its explicit empty-input conventions:

* an empty population reports ``0.0`` (with ``num_* == 0`` and ``wilson_ci``'s
  ``(0.0, 1.0)``) - the emptiness must be visible in the surrounding counts,
  exactly as the frozen module documents;
* quantiles of an empty sample stay ``nan`` (``iou_quantiles`` semantics);
* every rate ships with its counts, so no denominator is ever invented.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .audit import (
    DEFAULT_KS,
    aggregate_availability,
    iou_quantiles,
    natural_omission_rate,
    wilson_ci,
)

__all__ = [
    "wilson_ci",
    "DEFAULT_K_LEVELS",
    "DEFAULT_SAME_CATEGORY_LEVELS",
    "DEFAULT_RECALL_THRESHOLDS",
    "group_refs_by_image",
    "make_expression_record",
    "make_image_record",
    "aggregate_audit",
    "make_jsonable",
    "RECALL_FIELDS",
    "CANDIDATE_AVAILABILITY_FIELDS",
    "SAME_CATEGORY_FIELDS",
    "NATURAL_OMISSION_FIELDS",
    "IOU_STATISTICS_FIELDS",
    "recall_rows",
    "candidate_availability_rows",
    "same_category_rows",
    "natural_omission_rows",
    "iou_statistics_rows",
    "write_csv_artifacts",
    "write_summary_json",
    "render_figures",
]

#: The Phase 0 cardinality grid - single source of truth is the frozen contract.
DEFAULT_K_LEVELS: tuple[int, ...] = tuple(int(k) for k in DEFAULT_KS)
DEFAULT_SAME_CATEGORY_LEVELS: tuple[int, ...] = (1, 2, 4, 9)
DEFAULT_RECALL_THRESHOLDS: tuple[float, ...] = (0.5, 0.7)

_NAN = float("nan")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def group_refs_by_image(examples: Iterable[Any]) -> Dict[int, List[Any]]:
    """``image_id -> [ReferringExample, ...]`` sorted by ``ref_id`` (deterministic)."""
    grouped: Dict[int, List[Any]] = {}
    for example in examples:
        grouped.setdefault(int(example.image_id), []).append(example)
    for bucket in grouped.values():
        bucket.sort(key=lambda ex: int(ex.ref_id))
    return grouped


# ---------------------------------------------------------------------------
# record builders (what the script produces per image and per N)
# ---------------------------------------------------------------------------
def make_expression_record(
    ref_id: int,
    max_iou: float,
    is_miss: bool,
    num_remaining_candidates: int,
    num_equivalent_removed: int,
    num_same_category_distractors: int,
    available: Mapping[Any, bool],
    target_category_unknown: bool = False,
) -> Dict[str, Any]:
    """Flatten one expression's audit into the aggregation schema.

    ``num_remaining_candidates`` comes from
    :func:`ccg.data.audit.remaining_candidate_count`, ``available`` from
    :func:`ccg.data.audit.target_availability` (``{K: bool}``) and
    ``num_same_category_distractors`` from
    :func:`ccg.data.audit.same_category_counts`.

    The frozen contract counts the pool identically in both functions -
    ``N - |{target} U to_remove|`` - so the remaining candidates *are* the valid
    distractors and ``available[K]`` must equal
    ``num_remaining_candidates >= K - 1``; the flags are re-checked below so a
    drifted upstream rule cannot pass silently.
    """
    max_iou = float(max_iou)
    if not 0.0 <= max_iou <= 1.0:
        raise ValueError(f"ref {ref_id}: max_iou {max_iou} outside [0, 1]")
    miss = bool(is_miss)
    remaining = int(num_remaining_candidates)
    removed = int(num_equivalent_removed)
    distractors = remaining  # frozen: remaining == N - |{target} U to_remove|
    if remaining < 0 or removed < 0:
        raise ValueError(
            f"ref {ref_id}: inconsistent counts remaining={remaining}, removed={removed}, "
            f"is_miss={miss}"
        )
    if miss and removed != 0:
        raise ValueError(
            f"ref {ref_id}: natural miss cannot remove equivalents (removed={removed})"
        )
    num_same = int(num_same_category_distractors)
    unknown = bool(target_category_unknown)
    if miss:
        if num_same != 0 or unknown:
            raise ValueError(
                f"ref {ref_id}: natural miss cannot carry same-category counts "
                f"(count={num_same}, unknown={unknown})"
            )
    if unknown and num_same != 0:
        raise ValueError(
            f"ref {ref_id}: unknown target category cannot carry same-category "
            f"distractors (count={num_same})"
        )
    if num_same > distractors:
        raise ValueError(
            f"ref {ref_id}: same_category {num_same} > valid distractors {distractors}"
        )
    if not available:
        raise ValueError(f"ref {ref_id}: availability flags missing")
    flags: Dict[str, bool] = {}
    for raw_k, raw_v in dict(available).items():
        k = int(raw_k)
        if k < 1:
            raise ValueError(f"ref {ref_id}: availability K={k} must be >= 1")
        expected = distractors >= k - 1
        if bool(raw_v) != expected:
            raise ValueError(
                f"ref {ref_id}: availability[K={k}]={bool(raw_v)} contradicts distractors="
                f"{distractors} (expected {expected}) - upstream rule drifted"
            )
        flags[str(k)] = bool(raw_v)
    return {
        "ref_id": int(ref_id),
        "max_iou": max_iou,
        "is_miss": miss,
        "num_remaining_candidates": remaining,
        "num_equivalent_removed": removed,
        "num_same_category_distractors": num_same,
        "target_category_unknown": unknown,
        "available": flags,
    }


def make_image_record(
    image_id: int,
    N_cap: int,
    num_proposals: int,
    gt_max_ious: Sequence[float],
    gt_recall_counts: Mapping[str, Sequence[int]],
    redundancy: Mapping[str, Any],
    expressions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """One JSON-serialisable record: an image audited at truncation ``N_cap``.

    ``gt_max_ious`` are the per-GT-object max IoUs (quantiles / omission use
    them); ``gt_recall_counts`` maps the recall threshold label (``"0.5"``,
    ``"0.7"``) to ``[num_recalled, num_units]`` as measured by
    :func:`ccg.data.audit.proposal_recall` - the two views are cross-checked so
    the aggregated counts and the raw IoU sample cannot disagree.  ``redundancy``
    is the :func:`ccg.data.audit.redundancy_stats` dict.
    """
    N_cap = int(N_cap)
    num_proposals = int(num_proposals)
    if N_cap <= 0 or num_proposals <= 0:
        raise ValueError(f"image {image_id}: N_cap={N_cap} / num_proposals={num_proposals} <= 0")
    ious = [float(v) for v in gt_max_ious]
    for value in ious:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"image {image_id}: gt max IoU {value} outside [0, 1]")
    num_gt = len(ious)
    counts: Dict[str, Tuple[int, int]] = {}
    for label, pair in dict(gt_recall_counts).items():
        recalled, units = (int(v) for v in pair)
        if units != num_gt:
            raise ValueError(
                f"image {image_id}: gt_recall_counts[{label!r}] units {units} != "
                f"{num_gt} GT objects"
            )
        if not 0 <= recalled <= units:
            raise ValueError(
                f"image {image_id}: gt_recall_counts[{label!r}] recalled {recalled} "
                f"outside [0, {units}]"
            )
        threshold = float(label)
        if units and recalled != int(np.count_nonzero(np.asarray(ious) >= threshold)):
            raise ValueError(
                f"image {image_id}: gt_recall_counts[{label!r}] = {recalled} disagrees "
                "with the per-object max-IoU sample"
            )
        counts[str(label)] = (recalled, units)
    n_pairs = int(redundancy.get("n_pairs", 0))
    frac_mid = float(redundancy.get("frac_pairs_gt_mid", 0.0))
    frac_high = float(redundancy.get("frac_pairs_gt_high", 0.0))
    for name, value in (("frac_pairs_gt_mid", frac_mid), ("frac_pairs_gt_high", frac_high)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"image {image_id}: {name}={value} outside [0, 1]")
    return {
        "image_id": int(image_id),
        "N_cap": N_cap,
        "num_proposals": num_proposals,
        "num_gt_objects": num_gt,
        "gt_max_ious": ious,
        "gt_recall_counts": {label: [k, m] for label, (k, m) in counts.items()},
        "redundancy": {
            "n_pairs": n_pairs,
            "frac_pairs_gt_mid": frac_mid,
            "frac_pairs_gt_high": frac_high,
        },
        "expressions": [dict(e) for e in expressions],
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate_audit(
    records: Sequence[Mapping[str, Any]],
    *,
    k_levels: Sequence[int] = DEFAULT_K_LEVELS,
    same_category_levels: Sequence[int] = DEFAULT_SAME_CATEGORY_LEVELS,
    recall_thresholds: Sequence[float] = DEFAULT_RECALL_THRESHOLDS,
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """Aggregate per-image records into the ``per_N`` report blocks.

    Records are grouped by ``N_cap``; every level gets its own denominators
    (proposal counts, expressions, GT objects).  Returns a JSON-serialisable
    dict (string keys everywhere) consumed by the CSV row builders.
    """
    record_list = [dict(r) for r in records]
    if not record_list:
        raise ValueError("aggregate_audit() called with no image records")
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in record_list:
        grouped.setdefault(int(record["N_cap"]), []).append(record)

    ks = [int(k) for k in k_levels]
    same_levels = [int(m) for m in same_category_levels]
    labels = [f"{float(t):g}" for t in recall_thresholds]
    per_n: Dict[str, Any] = {}
    for n_cap in sorted(grouped):
        group = grouped[n_cap]
        gt_values = np.asarray([v for r in group for v in r["gt_max_ious"]], dtype=np.float64)
        exprs = [e for r in group for e in r["expressions"]]
        expr_values = np.asarray([float(e["max_iou"]) for e in exprs], dtype=np.float64)
        remaining = np.asarray(
            [float(e["num_remaining_candidates"]) for e in exprs], dtype=np.float64
        )

        gt_recall = {
            label: (
                int(sum(int(r["gt_recall_counts"][label][0]) for r in group)),
                int(sum(int(r["gt_recall_counts"][label][1]) for r in group)),
            )
            for label in labels
        }
        ref_recall = {
            label: (
                int(np.count_nonzero(expr_values >= float(label))),
                int(expr_values.size),
            )
            for label in labels
        }

        block: Dict[str, Any] = {
            "num_images": len(group),
            "num_images_short_bank": int(
                sum(1 for r in group if int(r["num_proposals"]) < n_cap)
            ),
            "num_proposals_total": int(sum(int(r["num_proposals"]) for r in group)),
            "num_expressions": len(exprs),
            "num_gt_objects": int(gt_values.size),
            "recall": {
                "gt_object": _recall_block(gt_recall, labels),
                "ref_target": _recall_block(ref_recall, labels),
            },
            "candidate_availability": {},
            "same_category": {"num_target_category_unknown": 0},
            "natural_omission": {
                "expression": _omission_block(expr_values, iou_thresh, "expression"),
                "gt_object": _omission_block(gt_values, iou_thresh, "coco_gt_object"),
            },
            "iou_stats": {
                "target_max_iou": _quantile_block(expr_values),
                "gt_max_iou": _quantile_block(gt_values),
                "remaining_count": _quantile_block(remaining),
                "redundancy": _redundancy_block(group),
            },
        }

        if exprs:
            fracs = aggregate_availability([e["available"] for e in exprs], ks)
            for k in ks:
                available_flags = [bool(e["available"].get(str(k), e["available"].get(k, False)))
                                   for e in exprs]
                got = int(sum(available_flags))
                low, high = wilson_ci(got, len(exprs))
                block["candidate_availability"][str(k)] = {
                    "K": k,
                    "num_expressions": len(exprs),
                    "num_available": got,
                    "frac_available": float(fracs[k]),
                    "ci_low": low,
                    "ci_high": high,
                }
            same_values = np.asarray(
                [int(e["num_same_category_distractors"]) for e in exprs], dtype=np.int64
            )
            block["same_category"]["num_target_category_unknown"] = int(
                sum(1 for e in exprs if bool(e["target_category_unknown"]))
            )
            for m in same_levels:
                got = int(np.count_nonzero(same_values >= m))
                low, high = wilson_ci(got, same_values.size)
                block["same_category"][str(m)] = {
                    "threshold": m,
                    "num_expressions": int(same_values.size),
                    "num_available": got,
                    "frac": float(got / same_values.size),
                    "ci_low": low,
                    "ci_high": high,
                }
        else:
            for k in ks:
                block["candidate_availability"][str(k)] = _empty_rate(
                    "K", k, "num_available", "frac_available"
                )
            for m in same_levels:
                block["same_category"][str(m)] = _empty_rate(
                    "threshold", m, "num_available", "frac"
                )

        per_n[str(int(n_cap))] = block

    return {"levels": sorted(int(k) for k in grouped), "per_N": per_n}


def _recall_block(
    counts: Mapping[str, Tuple[int, int]], labels: Sequence[str]
) -> Dict[str, Any]:
    num_units = next((units for _, units in counts.values()), 0)
    block: Dict[str, Any] = {"num_units": int(num_units), "by_threshold": {}}
    for label in labels:
        recalled, units = counts[label]
        low, high = wilson_ci(recalled, units)  # frozen convention: (0, 1) when units == 0
        block["by_threshold"][label] = {
            "threshold": float(label),
            "num_units": int(units),
            "num_recalled": int(recalled),
            "recall": float(recalled / units) if units else 0.0,
            "ci_low": low,
            "ci_high": high,
        }
    return block


def _omission_block(values: np.ndarray, iou_thresh: float, level: str) -> Dict[str, Any]:
    units = int(values.size)
    omitted = int(np.count_nonzero(values < float(iou_thresh)))
    low, high = wilson_ci(omitted, units)
    return {
        "level": level,
        "iou_thresh": float(iou_thresh),
        "num_units": units,
        "num_omitted": omitted,
        "rate": float(natural_omission_rate(values, iou_thresh=float(iou_thresh))),
        "ci_low": low,
        "ci_high": high,
        "mean_max_iou": float(values.mean()) if units else _NAN,
        "quantiles_max_iou": {
            f"q{int(q * 100)}": float(v)
            for q, v in iou_quantiles(values, (0.1, 0.25, 0.5, 0.75, 0.9)).items()
        },
    }


def _quantile_block(values: np.ndarray) -> Dict[str, Any]:
    quantiles = iou_quantiles(values, (0.25, 0.5, 0.75))
    return {
        "num_values": int(values.size),
        "mean": float(values.mean()) if values.size else _NAN,
        "p25": float(quantiles[0.25]),
        "median": float(quantiles[0.5]),
        "p75": float(quantiles[0.75]),
    }


def _redundancy_block(group: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n_pairs = int(sum(int(r["redundancy"]["n_pairs"]) for r in group))
    if n_pairs:
        frac_mid = float(
            sum(float(r["redundancy"]["frac_pairs_gt_mid"]) * int(r["redundancy"]["n_pairs"])
                for r in group) / n_pairs
        )
        frac_high = float(
            sum(float(r["redundancy"]["frac_pairs_gt_high"]) * int(r["redundancy"]["n_pairs"])
                for r in group) / n_pairs
        )
    else:
        frac_mid = frac_high = 0.0  # frozen redundancy_stats convention for n < 2
    return {
        "n_pairs": n_pairs,
        "frac_gt_0.7": frac_mid,
        "frac_gt_0.9": frac_high,
    }


def _empty_rate(unit_key: str, unit: int, count_key: str, rate_key: str) -> Dict[str, Any]:
    low, high = wilson_ci(0, 0)  # frozen convention: no observations, no information
    return {
        unit_key: int(unit),
        "num_expressions": 0,
        count_key: 0,
        rate_key: 0.0,
        "ci_low": low,
        "ci_high": high,
    }


# ---------------------------------------------------------------------------
# CSV row builders
# ---------------------------------------------------------------------------
RECALL_FIELDS = [
    "N",
    "num_gt_objects",
    "num_gt_recalled@0.5",
    "gt_object_recall@0.5",
    "gt_object_recall@0.5_ci_low",
    "gt_object_recall@0.5_ci_high",
    "num_gt_recalled@0.7",
    "gt_object_recall@0.7",
    "gt_object_recall@0.7_ci_low",
    "gt_object_recall@0.7_ci_high",
    "num_ref_targets",
    "num_ref_target_recalled@0.5",
    "ref_target_recall@0.5",
    "ref_target_recall@0.5_ci_low",
    "ref_target_recall@0.5_ci_high",
    "num_ref_target_recalled@0.7",
    "ref_target_recall@0.7",
    "ref_target_recall@0.7_ci_low",
    "ref_target_recall@0.7_ci_high",
]
CANDIDATE_AVAILABILITY_FIELDS = [
    "N",
    "K",
    "num_expressions",
    "num_available",
    "frac_available",
    "ci_low",
    "ci_high",
]
SAME_CATEGORY_FIELDS = [
    "N",
    "threshold",
    "num_expressions",
    "num_available",
    "frac",
    "ci_low",
    "ci_high",
    "num_target_category_unknown",
]
NATURAL_OMISSION_FIELDS = ["N", "level", "num_units", "num_omitted", "rate", "ci_low", "ci_high"]
IOU_STATISTICS_FIELDS = [
    "N",
    "target_max_iou_p25",
    "target_max_iou_median",
    "target_max_iou_p75",
    "gt_max_iou_p25",
    "gt_max_iou_median",
    "gt_max_iou_p75",
    "redundancy_frac_gt_0.7",
    "redundancy_frac_gt_0.9",
    "remaining_count_median",
]


def _level_keys(agg: Mapping[str, Any]) -> List[str]:
    return sorted((str(k) for k in agg["per_N"]), key=int)


def recall_rows(agg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in _level_keys(agg):
        level = agg["per_N"][key]
        gt = level["recall"]["gt_object"]
        ref = level["recall"]["ref_target"]
        row: Dict[str, Any] = {"N": int(key), "num_gt_objects": int(gt["num_units"])}
        for label in ("0.5", "0.7"):
            entry = gt["by_threshold"][label]
            row[f"num_gt_recalled@{label}"] = entry["num_recalled"]
            row[f"gt_object_recall@{label}"] = entry["recall"]
            row[f"gt_object_recall@{label}_ci_low"] = entry["ci_low"]
            row[f"gt_object_recall@{label}_ci_high"] = entry["ci_high"]
        row["num_ref_targets"] = int(ref["num_units"])
        for label in ("0.5", "0.7"):
            entry = ref["by_threshold"][label]
            row[f"num_ref_target_recalled@{label}"] = entry["num_recalled"]
            row[f"ref_target_recall@{label}"] = entry["recall"]
            row[f"ref_target_recall@{label}_ci_low"] = entry["ci_low"]
            row[f"ref_target_recall@{label}_ci_high"] = entry["ci_high"]
        rows.append(row)
    return rows


def candidate_availability_rows(agg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in _level_keys(agg):
        block = agg["per_N"][key]["candidate_availability"]
        for k in sorted(block, key=int):
            entry = block[k]
            rows.append(
                {
                    "N": int(key),
                    "K": int(entry["K"]),
                    "num_expressions": entry["num_expressions"],
                    "num_available": entry["num_available"],
                    "frac_available": entry["frac_available"],
                    "ci_low": entry["ci_low"],
                    "ci_high": entry["ci_high"],
                }
            )
    return rows


def same_category_rows(agg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in _level_keys(agg):
        level = agg["per_N"][key]["same_category"]
        unknown = int(level["num_target_category_unknown"])
        for threshold in sorted((k for k in level if k.isdigit()), key=int):
            entry = level[threshold]
            rows.append(
                {
                    "N": int(key),
                    "threshold": int(entry["threshold"]),
                    "num_expressions": entry["num_expressions"],
                    "num_available": entry["num_available"],
                    "frac": entry["frac"],
                    "ci_low": entry["ci_low"],
                    "ci_high": entry["ci_high"],
                    "num_target_category_unknown": unknown,
                }
            )
    return rows


def natural_omission_rows(agg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in _level_keys(agg):
        omission = agg["per_N"][key]["natural_omission"]
        for level_name in ("expression", "gt_object"):
            entry = omission[level_name]
            rows.append(
                {
                    "N": int(key),
                    "level": level_name,
                    "num_units": entry["num_units"],
                    "num_omitted": entry["num_omitted"],
                    "rate": entry["rate"],
                    "ci_low": entry["ci_low"],
                    "ci_high": entry["ci_high"],
                }
            )
    return rows


def iou_statistics_rows(agg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in _level_keys(agg):
        stats = agg["per_N"][key]["iou_stats"]
        target = stats["target_max_iou"]
        gt = stats["gt_max_iou"]
        redundancy = stats["redundancy"]
        remaining = stats["remaining_count"]
        rows.append(
            {
                "N": int(key),
                "target_max_iou_p25": target["p25"],
                "target_max_iou_median": target["median"],
                "target_max_iou_p75": target["p75"],
                "gt_max_iou_p25": gt["p25"],
                "gt_max_iou_median": gt["median"],
                "gt_max_iou_p75": gt["p75"],
                "redundancy_frac_gt_0.7": redundancy["frac_gt_0.7"],
                "redundancy_frac_gt_0.9": redundancy["frac_gt_0.9"],
                "remaining_count_median": remaining["median"],
            }
        )
    return rows


#: ``filename -> (header fields, row builder)`` - the committed CSV artifacts.
CSV_ARTIFACTS: tuple[tuple[str, List[str], Any], ...] = (
    ("recall_by_N.csv", RECALL_FIELDS, recall_rows),
    ("candidate_availability_by_N.csv", CANDIDATE_AVAILABILITY_FIELDS, candidate_availability_rows),
    ("same_category_availability.csv", SAME_CATEGORY_FIELDS, same_category_rows),
    ("natural_omission.csv", NATURAL_OMISSION_FIELDS, natural_omission_rows),
    ("proposal_iou_statistics.csv", IOU_STATISTICS_FIELDS, iou_statistics_rows),
)


def write_csv_artifacts(out_dir: str | Path, agg: Mapping[str, Any]) -> Dict[str, Path]:
    """Write the five CSV artifacts into ``out_dir``; returns ``name -> path``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for name, fields, builder in CSV_ARTIFACTS:
        path = out_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in builder(agg):
                writer.writerow(row)
        written[name] = path
    return written


# ---------------------------------------------------------------------------
# JSON / figures
# ---------------------------------------------------------------------------
def make_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/scalar values into strict JSON types.

    Non-finite floats become ``None`` so the artifacts stay valid JSON
    (``json.dump`` would otherwise emit the non-standard ``NaN`` literal).
    """
    if isinstance(obj, Mapping):
        return {str(k): make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return make_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_summary_json(path: str | Path, summary: Mapping[str, Any]) -> Path:
    """Write ``summary.json`` (numpy-safe, deterministic formatting)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_jsonable(summary), handle, indent=2, ensure_ascii=False, default=float)
    return path


def render_figures(
    records: Sequence[Mapping[str, Any]],
    agg: Mapping[str, Any],
    fig_dir: str | Path,
    *,
    dpi: int = 110,
) -> Dict[str, Path]:
    """Diagnostic figures (plain matplotlib, no styling): returns ``name -> path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    levels = [int(n) for n in agg["levels"]]
    per_n = agg["per_N"]
    written: Dict[str, Path] = {}

    def _save(fig, name: str) -> None:
        path = fig_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=int(dpi))
        plt.close(fig)
        written[name] = path

    def _expr_values(n: int) -> List[float]:
        return [
            float(e["max_iou"]) for r in records if int(r["N_cap"]) == n for e in r["expressions"]
        ]

    # 1) expression max-IoU histograms, one step curve per N
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0.0, 1.0, 21)
    for n in levels:
        values = _expr_values(n)
        if values:
            ax.hist(values, bins=bins, histtype="step", lw=1.6, label=f"N={n}")
    ax.set_xlabel("max IoU(target, top-N bank)")
    ax.set_ylabel("num expressions")
    ax.set_title("RefCOCO+ target max-IoU")
    ax.legend()
    _save(fig, "expression_max_iou_histogram.png")

    # 2) valid candidate count histograms (remaining candidates after removal)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    max_n = max(levels) if levels else 1
    bins = np.arange(0, max_n + 2) - 0.5
    for n in levels:
        values = [
            int(e["num_remaining_candidates"])
            for r in records
            if int(r["N_cap"]) == n
            for e in r["expressions"]
        ]
        if values:
            ax.hist(values, bins=bins, histtype="step", lw=1.6, label=f"N={n}")
    ax.set_xlabel("remaining valid candidates (target-equivalents removed)")
    ax.set_ylabel("num expressions")
    ax.set_title("valid candidate count")
    ax.legend()
    _save(fig, "remaining_candidate_count_histogram.png")

    # 3) candidate availability vs N, one line per K
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    k_levels = sorted(
        {int(k) for level in per_n.values() for k in level["candidate_availability"]}
    )
    xs = levels
    for k in k_levels:
        ys = [
            per_n[str(n)]["candidate_availability"][str(k)]["frac_available"] for n in levels
        ]
        ax.plot(xs, ys, marker="o", lw=1.5, label=f"K={k}")
    ax.set_xlabel("bank truncation N")
    ax.set_ylabel("P(>= K valid candidates)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(xs)
    ax.set_title("candidate availability vs N")
    ax.legend()
    _save(fig, "candidate_availability_vs_N.png")

    # 4) same-category availability bars
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    thresholds = sorted(
        {int(k) for level in per_n.values() for k in level["same_category"] if str(k).isdigit()}
    )
    width = 0.8 / max(1, len(levels))
    positions = np.arange(len(thresholds), dtype=np.float64)
    for offset, n in enumerate(levels):
        block = per_n[str(n)]["same_category"]
        ys = [float(block[str(t)]["frac"]) for t in thresholds]
        ax.bar(positions + offset * width - 0.4 + width / 2, ys, width=width, label=f"N={n}")
    ax.set_xticks(positions)
    ax.set_xticklabels([f">={t}" for t in thresholds])
    ax.set_xlabel("same-category distractors required")
    ax.set_ylabel("fraction of expressions")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("same-category availability")
    ax.legend()
    _save(fig, "same_category_availability.png")

    # 5) natural omission rate with 95% Wilson CI
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(levels), dtype=np.float64)
    for level_name, marker in (("expression", "o"), ("gt_object", "s")):
        rates = [float(per_n[str(n)]["natural_omission"][level_name]["rate"]) for n in levels]
        lows = [float(per_n[str(n)]["natural_omission"][level_name]["ci_low"]) for n in levels]
        highs = [float(per_n[str(n)]["natural_omission"][level_name]["ci_high"]) for n in levels]
        lower = [r - lo for r, lo in zip(rates, lows)]
        upper = [hi - r for r, hi in zip(rates, highs)]
        ax.errorbar(xs, rates, yerr=[lower, upper], marker=marker, capsize=4, lw=1.5, label=level_name)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"N={n}" for n in levels])
    ax.set_ylabel("P(max IoU < 0.5)  (95% Wilson CI)")
    ax.set_title("natural omission")
    ax.legend()
    _save(fig, "natural_omission_ci.png")

    return written
