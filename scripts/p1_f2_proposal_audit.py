"""V2-P1 P1-F2: DETR-R50 proposal feasibility audit + gate.

Strictly re-uses the *frozen* V1 proposal-audit definition.  The only change
relative to :mod:`scripts.run_proposal_audit` is the **source of the proposal
bank**: instead of running the RPN we read the frozen, query-independent
DETR-R50 ``N = 64`` bank written by ``scripts/p1_extract_detr_proposals.py``.
Everything else is byte-for-byte the V1 pipeline:

  * the same frozen image subset (``data/audit_subset.csv``, 1500 images);
  * the same RefCOCO+ region loading (``load_refs_pickle`` + ``load_instances_json``
    + ``refs_by_image`` -> :class:`ReferringExample`, one per region);
  * the same COCO GT index (``instances_train2014.json``, ``iscrowd`` excluded);
  * the same frozen per-image primitives (``audit_level`` -> the ``ccg.data.audit``
    / ``ccg.data.proposals`` ten-function contract);
  * the same aggregation / CSV artifacts (``ccg.data.audit_report``).

Target criterion is unchanged: a proposal matches the RefCOCO+ target when its
IoU with the highest-IoU GT object is >= ``0.5``; the same-category assignment is
proposal -> highest-IoU COCO GT object only if IoU >= ``0.5``.  We do NOT touch
``N``, the IoU threshold, the detector, or add NMS (protocol section 9-10).

The gate is on **primary = DETR ref-target recall @ 0.5**:

    >= 0.95            FULL
    [0.90, 0.95)       GRAY
    [0.80, 0.90)       LIMITED (discuss before scoring)
    <  0.80            STOP

plus a hard sub-gate on ``candidate_availability[K = 50] >= 0.90`` (below that the
C1 K5 -> K50 main experiment cannot run).

It also writes a proposal-distribution comparison (``candidate_distribution.csv``)
between the frozen RPN (Proposal-A) and DETR (Proposal-B) and loads the RPN
reference numbers from the frozen ``results/proposal_audit/summary.json`` rather
than hard-coding them.

    conda activate deepminer
    python scripts/p1_f2_proposal_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# frozen V1 pipeline pieces - never re-implemented here
import run_proposal_audit as v1  # noqa: E402
from ccg.data.audit_report import (  # noqa: E402
    DEFAULT_K_LEVELS,
    DEFAULT_SAME_CATEGORY_LEVELS,
    aggregate_audit,
    group_refs_by_image,
    render_figures,
    write_csv_artifacts,
    write_summary_json,
)
from ccg.data.bank import bank_attrs, image_ids, read_bank_image  # noqa: E402
from ccg.data.audit import redundancy_stats  # noqa: E402

__all__ = ["DEFAULT_N", "run", "build_parser", "main"]

DEFAULT_N = 64
DETR_BANK = _ROOT / "cache" / "proposals_detr_r50.h5"
RPN_BANK = _ROOT / "cache" / "proposals.h5"
SUBSET = _ROOT / "data" / "audit_subset.csv"
OUT_DIR = _ROOT / "results" / "v2_proposal_robustness" / "p1_detr_r50"
RPN_REFERENCE = _ROOT / "results" / "proposal_audit" / "summary.json"

# the truncation level audited: DETR stores exactly 64 proposals per image
_LEVELS: tuple[int, ...] = (DEFAULT_N,)


def _read_bank_at(bank: Path, image_id: int, n: int):
    """Return the first ``n`` (sorted) proposals of one image, or ``None``."""
    try:
        boxes, objectness = read_bank_image(bank, int(image_id))
    except KeyError:
        return None
    k = int(min(n, boxes.shape[0]))
    if k == 0:
        return None
    return np.asarray(boxes[:k], dtype=np.float32), np.asarray(objectness[:k], dtype=np.float32)


def _distributions(bank: Path, ids: Sequence[int], n: int) -> Dict[str, float]:
    """Pooled per-proposal distribution stats over ``ids`` (diagnostic only)."""
    areas: List[float] = []
    aspects: List[float] = []
    scores: List[float] = []
    top1: List[float] = []
    red_high: List[float] = []
    for image_id in ids:
        got = _read_bank_at(bank, image_id, n)
        if got is None:
            continue
        boxes, objectness = got
        w = np.clip(boxes[:, 2] - boxes[:, 0], 0.0, None)
        h = np.clip(boxes[:, 3] - boxes[:, 1], 0.0, None)
        areas.extend((w * h).tolist())
        aspects.extend((np.where(h > 0, w / np.maximum(h, 1e-6), np.nan)).tolist())
        scores.extend(objectness.tolist())
        top1.append(float(objectness[0]))
        red_high.append(float(redundancy_stats(boxes)["frac_pairs_gt_high"]))
    return {
        "mean_area_px2": float(np.mean(areas)) if areas else float("nan"),
        "median_area_px2": float(np.median(areas)) if areas else float("nan"),
        "mean_aspect_ratio": float(np.nanmean(aspects)) if aspects else float("nan"),
        "median_aspect_ratio": float(np.nanmedian(aspects)) if aspects else float("nan"),
        "mean_proposal_score": float(np.mean(scores)) if scores else float("nan"),
        "mean_top1_proposal_score": float(np.mean(top1)) if top1 else float("nan"),
        "mean_redundancy_frac_gt_0p9": float(np.mean(red_high)) if red_high else float("nan"),
    }


def _gate_block(per_n: Dict[str, Any], n: int) -> Dict[str, Any]:
    level = per_n[str(n)]
    ref = level["recall"]["ref_target"]["by_threshold"]
    recall05 = float(ref["0.5"]["recall"])
    recall07 = float(ref["0.7"]["recall"])
    avail = level["candidate_availability"].get("50")
    avail50 = float(avail["frac_available"]) if avail else float("nan")
    if recall05 >= 0.95:
        verdict = "FULL"
    elif recall05 >= 0.90:
        verdict = "GRAY"
    elif recall05 >= 0.80:
        verdict = "LIMITED"
    else:
        verdict = "STOP"
    subgate_ok = bool(np.isfinite(avail50) and avail50 >= 0.90)
    if verdict in ("FULL", "GRAY", "LIMITED") and not subgate_ok:
        verdict = "STOP"
    return {
        "n": n,
        "primary_metric": "ref_target_recall@0.5",
        "ref_target_recall@0.5": recall05,
        "ref_target_recall@0.5_ci": [ref["0.5"]["ci_low"], ref["0.5"]["ci_high"]],
        "ref_target_recall@0.7": recall07,
        "candidate_availability_K50_frac": avail50,
        "candidate_availability_K50_gate": ">= 0.90",
        "candidate_availability_K50_ok": subgate_ok,
        "verdict": verdict,
        "thresholds": {"FULL": ">=0.95", "GRAY": "[0.90,0.95)", "LIMITED": "[0.80,0.90)", "STOP": "<0.80"},
    }


def _rpn_reference(path: Path, n: int) -> Dict[str, Any]:
    """Load the frozen RPN per_N reference (never hard-coded)."""
    if not Path(path).exists():
        return {"available": False, "path": str(path)}
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    level = (summary.get("per_N") or {}).get(str(n))
    if not level:
        return {"available": False, "path": str(path), "reason": f"per_N['{n}'] absent"}
    ref = level["recall"]["ref_target"]["by_threshold"]
    gt = level["recall"]["gt_object"]["by_threshold"]
    avail = level["candidate_availability"].get("50", {})
    same4 = level["same_category"].get("4", {})
    return {
        "available": True,
        "path": str(path),
        "ref_target_recall@0.5": float(ref["0.5"]["recall"]),
        "ref_target_recall@0.7": float(ref["0.7"]["recall"]),
        "gt_object_recall@0.5": float(gt["0.5"]["recall"]),
        "candidate_availability_K50_frac": float(avail.get("frac_available", float("nan"))),
        "same_category_ge4_frac": float(same4.get("frac", float("nan"))),
        "num_expressions": int(level["num_expressions"]),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    subset_path = Path(args.subset)
    detr_bank = Path(args.detr_bank)
    rpn_bank = Path(args.rpn_bank)
    out_dir = Path(args.out)
    n = int(args.n)
    iou_thresh = float(args.iou_thresh)

    print(f"[p1-f2] subset      : {subset_path}")
    rows = v1.load_subset_rows(subset_path)
    ids = [int(r["image_id"]) for r in rows]
    print(f"[p1-f2] rows        : {len(rows)}")

    print(f"[p1-f2] DETR bank   : {detr_bank}")
    attrs = bank_attrs(detr_bank)
    if not attrs:
        raise RuntimeError(f"{detr_bank} has no bank attrs; run p1_extract_detr_proposals.py first")
    bank_ids = set(int(i) for i in image_ids(detr_bank).tolist())
    print(
        f"[p1-f2] DETR bank   : {len(bank_ids)} images; model={attrs.get('model_name')} "
        f"N={attrs.get('top_n')} query_independent={attrs.get('query_independent')}"
    )

    print(f"[p1-f2] refcoco+    : {args.refcoco_pickle}")
    refs = v1.load_refcoco_refs(Path(args.refcoco_pickle), Path(args.refcoco_instances))
    refs_by_image = group_refs_by_image(refs)
    print(f"[p1-f2] refs        : {len(refs)} regions on {len(refs_by_image)} images")

    print(f"[p1-f2] gt          : {args.coco_annotations}")
    gt_index = v1.load_gt_index(Path(args.coco_annotations), ids)
    print(f"[p1-f2] gt index    : {len(gt_index)} images (subset-restricted)")

    records: List[Dict[str, Any]] = []
    n_missing = n_empty = n_analyzed = 0
    for image_id in ids:
        if image_id not in bank_ids:
            n_missing += 1
            continue
        got = _read_bank_at(detr_bank, image_id, n)
        if got is None:
            n_empty += 1
            continue
        boxes, objectness = got
        gt = gt_index.objects(image_id).non_crowd()
        records.append(
            v1.audit_level(
                image_id=image_id,
                bank_boxes=boxes,
                bank_scores=objectness,
                gt_boxes=gt.boxes,
                gt_categories=gt.categories,
                refs=refs_by_image.get(image_id, []),
                n_cap=n,
                iou_thresh=iou_thresh,
                k_levels=DEFAULT_K_LEVELS,
            )
        )
        n_analyzed += 1

    if not records:
        raise RuntimeError("no DETR-bank image could be audited; is the bank populated for the subset?")

    agg = aggregate_audit(
        records,
        k_levels=DEFAULT_K_LEVELS,
        same_category_levels=DEFAULT_SAME_CATEGORY_LEVELS,
        iou_thresh=iou_thresh,
    )
    gate = _gate_block(agg["per_N"], n)
    rpn_ref = _rpn_reference(Path(args.rpn_reference), n)

    print(f"[p1-f2] analyzed    : {n_analyzed} (missing_in_bank={n_missing}, empty={n_empty})")
    print(
        f"[p1-f2] DETR recall@0.5 = {gate['ref_target_recall@0.5']:.4f} "
        f"(@0.7 {gate['ref_target_recall@0.7']:.4f}) | K50 avail = {gate['candidate_availability_K50_frac']:.4f}"
    )
    print(f"[p1-f2] GATE VERDICT: {gate['verdict']}")

    write_csv_artifacts(out_dir, agg)
    if args.figures and records:
        render_figures(records, agg, out_dir / "figures")

    # distribution shift: DETR vs frozen RPN over the same subset (diagnostic only)
    detr_dist = _distributions(detr_bank, ids, n)
    rpn_dist = _distributions(rpn_bank, ids, n) if rpn_bank.exists() else {}
    dist_rows = [
        {"metric": key, "detr_r50": detr_dist.get(key), "rpn": rpn_dist.get(key),
         "detr_minus_rpn": (detr_dist.get(key) - rpn_dist.get(key))
         if (key in detr_dist and key in rpn_dist) else None}
        for key in detr_dist
    ]
    for key, value in (
        ("ref_target_recall@0.5", (gate["ref_target_recall@0.5"], rpn_ref.get("ref_target_recall@0.5"))),
        ("ref_target_recall@0.7", (gate["ref_target_recall@0.7"], rpn_ref.get("ref_target_recall@0.7"))),
        ("candidate_availability_K50_frac",
         (gate["candidate_availability_K50_frac"], rpn_ref.get("candidate_availability_K50_frac"))),
        ("same_category_ge4_frac",
         (float(agg["per_N"][str(n)]["same_category"].get("4", {}).get("frac", float("nan"))),
          rpn_ref.get("same_category_ge4_frac"))),
    ):
        d, r = value
        dist_rows.append({"metric": key, "detr_r50": d, "rpn": r,
                          "detr_minus_rpn": (d - r) if (r is not None and np.isfinite(d)) else None})
    import csv as _csv
    dist_path = out_dir / "candidate_distribution.csv"
    with dist_path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=["metric", "detr_r50", "rpn", "detr_minus_rpn"])
        writer.writeheader()
        for row in dist_rows:
            writer.writerow(row)
    print(f"[p1-f2] wrote       : {dist_path}")

    summary: Dict[str, Any] = {
        "task": "p1_f2_proposal_audit",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "subset": str(subset_path), "detr_bank": str(detr_bank), "rpn_bank": str(rpn_bank),
            "out_dir": str(out_dir), "n": n, "iou_thresh": iou_thresh,
            "K_levels": list(DEFAULT_K_LEVELS), "same_category_levels": list(DEFAULT_SAME_CATEGORY_LEVELS),
            "coco_annotations": str(args.coco_annotations),
            "refcoco_pickle": str(args.refcoco_pickle), "refcoco_instances": str(args.refcoco_instances),
            "target_criterion": "proposal IoU with highest-IoU GT target >= 0.5 (unchanged from V1)",
            "same_category_assignment": "proposal -> highest-IoU COCO GT object only if IoU >= 0.5 (unchanged)",
            "reuses": "scripts.run_proposal_audit primitives (V1 authoritative proposal audit)",
        },
        "detr_bank_attrs": {
            "model_name": attrs.get("model_name"), "weights": attrs.get("weights"),
            "checkpoint_revision": attrs.get("checkpoint_revision"),
            "checkpoint_sha256": str(attrs.get("checkpoint_sha256"))[:16] if attrs.get("checkpoint_sha256") else None,
            "top_n": attrs.get("top_n"), "query_independent": attrs.get("query_independent"),
            "n_images_in_bank": len(bank_ids),
        },
        "images": {"subset_rows": len(rows), "analyzed": n_analyzed,
                   "missing_in_detr_bank": n_missing, "empty_bank": n_empty},
        "gate": gate,
        "rpn_frozen_reference": rpn_ref,
        "per_N": agg["per_N"],
        "runtime_seconds": float(time.perf_counter() - started),
        "verdict": gate["verdict"],
    }
    summary_path = write_summary_json(out_dir / "proposal_summary.json", summary)
    print(f"[p1-f2] summary     : {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = v1.build_parser().parse_args([])
    p.add_argument("--subset", type=Path, default=SUBSET)
    p.add_argument("--detr-bank", type=Path, default=DETR_BANK)
    p.add_argument("--rpn-bank", type=Path, default=RPN_BANK)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--iou-thresh", type=float, default=0.5)
    p.add_argument("--coco-annotations", type=Path, default=defaults.coco_annotations)
    p.add_argument("--refcoco-pickle", type=Path, default=defaults.refcoco_pickle)
    p.add_argument("--refcoco-instances", type=Path, default=defaults.refcoco_instances)
    p.add_argument("--rpn-reference", type=Path, default=RPN_REFERENCE)
    p.add_argument("--figures", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    verdict = summary["gate"]["verdict"]
    # FULL / GRAY proceed; LIMITED is a soft-stop for discussion; STOP is a hard failure.
    return 0 if verdict in ("FULL", "GRAY") else 1


if __name__ == "__main__":
    sys.exit(main())
