"""Full-data natural omission counts for the ``bank-v1`` proposal bank.

**Counting only - no presence model, no training, no candidate construction.**
For every RefCOCO+ region (``refs(unc).p``) the target box is joined from
``instances.json`` through ``ann_id`` (``xywh -> xyxy``) and compared with the
image's bank rows::

    max_iou(ref) = max_i IoU(p_i, b*)        # N = 64, frozen bank
    miss(ref)    = max_iou < iou_thresh      # primary threshold 0.5

Three denominators (frozen counting rules of the multi-agent brief, section 27)

* ``expression``     one unit per **sentence**: ``sum len(sentences)``; a miss
  region marks *all* of its sentences as missed;
* ``unique_object``  one unit per distinct ``ann_id`` (target object);
* ``image``          images with ``>= 1`` miss region over images holding
  ``>= 1`` scored region.

Every unit is reported per split (train / val / testA / testB) with a rate and
a Wilson 95% interval (:func:`ccg.data.audit.wilson_ci`).  Sensitivity rows at
IoU 0.3 / 0.7 are emitted as well and flagged ``is_primary=False`` - the
primary decision boundary stays 0.5.

Honesty rules
-------------
* The bank group of a ref's image is read through the frozen layout
  (``image_{image_id}``); an absent group is recorded under
  ``missing_bank_images`` (with its ref count) and **excluded from every
  denominator** - never silently skipped, never assumed hit or miss.
* A ref whose ``ann_id`` cannot be joined, or whose bank group is corrupt /
  empty, is recorded under ``unresolved_refs`` / ``unscorable_bank_images``.
* Also reported: the all-data quantiles of ``max_iou`` (p10 / p25 / p50 /
  p75 / p90) and the bank's root attrs (provenance cross-check).

Outputs (``--out`` directory)::

    summary.json               full report (counts, CI, missing lists, quantiles)
    by_split.csv               primary threshold rows: split,unit,num_total,
                               num_miss,rate,ci_low,ci_high
    threshold_sensitivity.csv  IoU 0.3/0.5/0.7 rows, is_primary flags them

Usage (repo root, conda env ``deepminer``)::

    python scripts/natural_omission_full.py \
        --bank cache/proposals.h5 \
        --refs data/raw/refcoco+/refcoco+/refs(unc).p \
        --instances data/raw/refcoco+/refcoco+/instances.json \
        --out results/natural_omission_full
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from ccg.data.audit import iou_quantiles, wilson_ci  # noqa: E402  (path fixed above)
from ccg.data.bank import bank_attrs, bank_group_name  # noqa: E402
from ccg.data.proposals import iou_matrix, xywh_to_xyxy  # noqa: E402
from ccg.data.refcoco import load_instances_json, load_refs_pickle  # noqa: E402
from ccg.data.splits import (  # noqa: E402
    SPLIT_TEST_A,
    SPLIT_TEST_B,
    SPLIT_TRAIN,
    SPLIT_VAL,
    normalize_split_name,
)

__all__ = [
    "SPLITS",
    "THRESHOLDS",
    "PRIMARY_THRESHOLD",
    "UNITS",
    "UNIT_DESCRIPTIONS",
    "CSV_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "score_refs",
    "split_counts",
    "counts_to_rows",
    "quantile_summary",
    "build_summary",
    "write_outputs",
    "build_parser",
    "run",
    "main",
]

#: The four RefCOCO+ splits (image-level, disjoint).
SPLITS: Tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST_A, SPLIT_TEST_B)
#: Sensitivity grid; only 0.5 is the primary boundary.
THRESHOLDS: Tuple[float, ...] = (0.3, 0.5, 0.7)
PRIMARY_THRESHOLD = 0.5
UNITS: Tuple[str, ...] = ("expression", "unique_object", "image")

UNIT_DESCRIPTIONS: Dict[str, str] = {
    "expression": (
        "one unit per sentence (sum of len(sentences)); a miss region marks all of its "
        "sentences as missed"
    ),
    "unique_object": "one unit per distinct ann_id (target object) within the split",
    "image": (
        "images with >= 1 miss region, over images with >= 1 scored region in the split"
    ),
}

#: Frozen column order of ``by_split.csv`` (multi-agent brief).
CSV_COLUMNS: Tuple[str, ...] = ("split", "unit", "num_total", "num_miss", "rate", "ci_low", "ci_high")
SENSITIVITY_COLUMNS: Tuple[str, ...] = ("iou_threshold",) + CSV_COLUMNS + ("is_primary",)

_QUANTILE_NAMES: Tuple[Tuple[str, float], ...] = (
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
)


def _threshold_key(threshold: float) -> str:
    return f"{float(threshold):g}"


# ---------------------------------------------------------------------------
# per-ref scoring against the bank
# ---------------------------------------------------------------------------
def score_refs(
    refs: Sequence[Mapping[str, Any]],
    instances: Mapping[str, dict],
    bank_path: str | Path,
    progress: bool = True,
) -> Dict[str, Any]:
    """One ``max_iou`` per Joirable ref; everything else is recorded verbatim.

    Returns ``{"per_ref", "missing_bank_images", "unscorable_bank_images",
    "unresolved_refs"}``.  ``per_ref`` rows carry ``ref_id / image_id / split /
    ann_id / n_sentences / max_iou``.  A ref never disappears: images without a
    bank group are counted per image, corrupt groups and unjoinable
    ``ann_id``s are listed.  ``progress=True`` shows a tqdm bar over the refs
    on a TTY (auto-disabled otherwise via ``disable=None``); an empty ``refs``
    sequence shows no bar at all.
    """
    import h5py

    per_ref: List[Dict[str, Any]] = []
    missing_counts: Dict[int, int] = {}
    unscorable: Dict[int, str] = {}
    unscorable_counts: Dict[int, int] = {}
    unresolved: List[Dict[str, Any]] = []
    boxes_cache: Dict[int, Tuple[Optional[np.ndarray], str]] = {}

    with h5py.File(Path(bank_path), "r") as handle:

        def boxes_for(image_id: int) -> Tuple[Optional[np.ndarray], str]:
            """``(boxes, status)`` with status in {ok, missing, unscorable}."""
            cached = boxes_cache.get(image_id)
            if cached is not None:
                return cached
            name = bank_group_name(image_id)
            if name not in handle:
                result: Tuple[Optional[np.ndarray], str] = (None, "missing")
            else:
                group = handle[name]
                if "boxes" not in group or "objectness" not in group:
                    unscorable.setdefault(
                        image_id, "bank group corrupt (missing boxes/objectness)"
                    )
                    result = (None, "unscorable")
                else:
                    boxes = np.asarray(group["boxes"], dtype=np.float32).reshape(-1, 4)
                    if boxes.shape[0] == 0:
                        unscorable.setdefault(image_id, "bank group is empty")
                        result = (None, "unscorable")
                    else:
                        result = (boxes, "ok")
            boxes_cache[image_id] = result
            return result

        source: Iterable[Mapping[str, Any]] = refs
        if progress and len(refs):
            source = tqdm(refs, total=len(refs), desc="omission", unit="ref", disable=None)
        for position, record in enumerate(source):
            image_id = int(record["image_id"])
            ann_id = int(record["ann_id"])
            split = normalize_split_name(record.get("split"))
            n_sentences = len(record["sentences"])
            ref_id = record.get("ref_id")
            ref_id = int(ref_id) if ref_id is not None else -1

            info = instances.get(str(ann_id))
            if info is None:
                unresolved.append(
                    {
                        "ref_id": ref_id,
                        "image_id": image_id,
                        "split": split,
                        "ann_id": ann_id,
                        "reason": f"ann_id {ann_id} not in instances json",
                    }
                )
                continue
            target_xywh = np.asarray(info["bbox"], dtype=np.float64).reshape(-1)
            if target_xywh.size != 4 or not (target_xywh[2] > 0 and target_xywh[3] > 0):
                unresolved.append(
                    {
                        "ref_id": ref_id,
                        "image_id": image_id,
                        "split": split,
                        "ann_id": ann_id,
                        "reason": f"degenerate target box {target_xywh.tolist()}",
                    }
                )
                continue

            boxes, status = boxes_for(image_id)
            if status != "ok":
                if status == "missing":
                    missing_counts[image_id] = missing_counts.get(image_id, 0) + 1
                else:
                    unscorable_counts[image_id] = unscorable_counts.get(image_id, 0) + 1
                continue

            target = xywh_to_xyxy(target_xywh.astype(np.float32))
            max_iou = float(iou_matrix(target.reshape(1, 4), boxes).max())
            per_ref.append(
                {
                    "ref_id": ref_id,
                    "image_id": image_id,
                    "split": split,
                    "ann_id": ann_id,
                    "n_sentences": int(n_sentences),
                    "max_iou": max_iou,
                }
            )

    missing_list = [
        {"image_id": image_id, "n_refs": count}
        for image_id, count in sorted(missing_counts.items())
    ]
    unscorable_list = [
        {"image_id": image_id, "reason": reason, "n_refs": unscorable_counts.get(image_id, 0)}
        for image_id, reason in sorted(unscorable.items())
    ]
    return {
        "per_ref": per_ref,
        "missing_bank_images": missing_list,
        "unscorable_bank_images": unscorable_list,
        "unresolved_refs": unresolved,
    }


# ---------------------------------------------------------------------------
# counting
# ---------------------------------------------------------------------------
def split_counts(
    per_ref: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float] = THRESHOLDS,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, Any]:
    """Three denominators per split per threshold.

    Returns ``{"counts": {t_key: {split: {unit: (num_total, num_miss)}}},
    "other_splits": {label: n_refs}, "anomalies": [...]}``.  A same-``ann_id``
    inconsistency (theoretically impossible: one ann_id -> one box -> one
    image -> one bank) is reported in ``anomalies`` instead of being averaged
    away.
    """
    counts: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {}
    anomalies: set = set()
    for threshold in thresholds:
        per_split: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for split in splits:
            subset = [row for row in per_ref if row["split"] == split]

            expression_total = int(sum(row["n_sentences"] for row in subset))
            expression_miss = int(
                sum(row["n_sentences"] for row in subset if row["max_iou"] < threshold)
            )

            objects: Dict[int, bool] = {}
            for row in subset:
                is_miss = bool(row["max_iou"] < threshold)
                previous = objects.get(int(row["ann_id"]))
                if previous is None:
                    objects[int(row["ann_id"])] = is_miss
                elif previous != is_miss:
                    anomalies.add(
                        f"ann_id {row['ann_id']} has inconsistent miss judgments across refs"
                    )

            images: Dict[int, bool] = {}
            for row in subset:
                image_id = int(row["image_id"])
                images[image_id] = bool(
                    images.get(image_id, False) or (row["max_iou"] < threshold)
                )

            per_split[split] = {
                "expression": (expression_total, expression_miss),
                "unique_object": (len(objects), int(sum(objects.values()))),
                "image": (len(images), int(sum(images.values()))),
            }
        counts[_threshold_key(threshold)] = per_split

    others = Counter(
        str(row["split"]) for row in per_ref if row["split"] not in set(splits)
    )
    return {"counts": counts, "other_splits": dict(sorted(others.items())), "anomalies": sorted(anomalies)}


def counts_to_rows(
    counts: Mapping[str, Mapping[str, Mapping[str, Tuple[int, int]]]],
    threshold: float,
    splits: Sequence[str] = SPLITS,
    *,
    is_primary: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Long-form rows for one threshold, in frozen column order."""
    key = _threshold_key(threshold)
    if key not in counts:
        raise KeyError(f"threshold {threshold} not in the computed counts (have {sorted(counts)})")
    rows: List[Dict[str, Any]] = []
    for split in splits:
        for unit in UNITS:
            if split not in counts[key] or unit not in counts[key][split]:
                raise KeyError(f"counts[{key}][{split}][{unit}] missing")
            num_total, num_miss = counts[key][split][unit]
            num_total = int(num_total)
            num_miss = int(num_miss)
            rate = float(num_miss / num_total) if num_total else 0.0
            ci_low, ci_high = wilson_ci(num_miss, num_total)
            row: Dict[str, Any] = {
                "split": split,
                "unit": unit,
                "num_total": num_total,
                "num_miss": num_miss,
                "rate": rate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
            }
            if is_primary is not None:
                row["is_primary"] = bool(is_primary)
            rows.append(row)
    return rows


def quantile_summary(per_ref: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """p10/p25/p50/p75/p90 (+ mean/n) of ``max_iou`` over every scored ref."""
    ious = [float(row["max_iou"]) for row in per_ref]
    if not ious:
        return {"n": 0, "mean": None, **{name: None for name, _ in _QUANTILE_NAMES}}
    quantiles = iou_quantiles(ious, [q for _, q in _QUANTILE_NAMES])
    out: Dict[str, Any] = {"n": len(ious), "mean": float(np.mean(ious))}
    for name, q in _QUANTILE_NAMES:
        out[name] = float(quantiles[q])
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_summary(
    scoring: Mapping[str, Any],
    counts_blob: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    bank_meta: Mapping[str, Any],
    n_refs_total: int,
    runtime_seconds: float,
) -> Dict[str, Any]:
    counts = counts_blob["counts"]
    by_split = counts_to_rows(counts, PRIMARY_THRESHOLD)
    sensitivity: List[Dict[str, Any]] = []
    for threshold in THRESHOLDS:
        for row in counts_to_rows(
            counts, threshold, is_primary=(threshold == PRIMARY_THRESHOLD)
        ):
            row["iou_threshold"] = float(threshold)
            sensitivity.append(row)
    missing = list(scoring["missing_bank_images"])
    return {
        "task": "natural_omission_full",
        "counting_only": True,
        "primary_iou_threshold": PRIMARY_THRESHOLD,
        "iou_thresholds": [float(t) for t in THRESHOLDS],
        "units": dict(UNIT_DESCRIPTIONS),
        "config": {
            "bank": str(args.bank),
            "refs": str(args.refs),
            "instances": str(args.instances),
            "out_dir": str(args.out),
        },
        "bank_attrs": {str(key): value for key, value in dict(bank_meta).items()},
        "n_refs_total": int(n_refs_total),
        "n_refs_scored": len(scoring["per_ref"]),
        "n_refs_missing_bank_image": int(sum(item["n_refs"] for item in missing)),
        "n_refs_unscorable": int(
            sum(item.get("n_refs", 0) for item in scoring["unscorable_bank_images"])
        ),
        "missing_bank_images": missing,
        "unscorable_bank_images": list(scoring["unscorable_bank_images"]),
        "unresolved_refs": list(scoring["unresolved_refs"]),
        "other_splits": dict(counts_blob["other_splits"]),
        "anomalies": list(counts_blob["anomalies"]),
        "by_split": by_split,
        "threshold_sensitivity": sensitivity,
        "max_iou_quantiles": quantile_summary(scoring["per_ref"]),
        "runtime_seconds": float(runtime_seconds),
        "notes": [
            "Counting only: no presence model is trained or evaluated here; see the "
            "frozen counting rules of the multi-agent brief (section 27).",
            "Primary boundary IoU=0.5; the 0.3/0.7 rows are sensitivity analysis only "
            "(is_primary=False).",
            "Refs of images without a bank group are excluded from every denominator "
            "and listed under missing_bank_images (never assumed hit or miss).",
            "A miss at the region (ref) level marks all of the region's sentences as "
            "missed; unique_object deduplicates by ann_id; image counts an image once.",
        ],
    }


def write_outputs(summary: Dict[str, Any], out_dir: str | Path) -> Dict[str, Path]:
    """Write ``summary.json`` + ``by_split.csv`` + ``threshold_sensitivity.csv``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_split_path = out_dir / "by_split.csv"
    with by_split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in summary["by_split"]:
            writer.writerow({key: row[key] for key in CSV_COLUMNS})

    sensitivity_path = out_dir / "threshold_sensitivity.csv"
    with sensitivity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SENSITIVITY_COLUMNS))
        writer.writeheader()
        for row in summary["threshold_sensitivity"]:
            writer.writerow({key: row[key] for key in SENSITIVITY_COLUMNS})

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    return {"summary": summary_path, "by_split": by_split_path, "sensitivity": sensitivity_path}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    """Load inputs, score every ref, count, write the artefacts; returns summary."""
    started = time.perf_counter()
    bank_path = Path(args.bank)
    refs_path = Path(args.refs)
    instances_path = Path(args.instances)
    out_dir = Path(args.out)

    if not bank_path.exists():
        raise FileNotFoundError(
            f"proposal bank {bank_path} not found - run scripts/extract_proposals.py first"
        )
    print(f"[natural-omission] bank        : {bank_path}")
    bank_meta = bank_attrs(bank_path)
    print(
        f"[natural-omission] bank attrs  : schema={bank_meta.get('schema_version')} "
        f"top_n={bank_meta.get('top_n')} n_images={bank_meta.get('n_images')}"
    )

    print(f"[natural-omission] refs        : {refs_path}")
    refs = load_refs_pickle(refs_path)
    instances = load_instances_json(instances_path)
    print(f"[natural-omission] refs        : {len(refs)} regions, {len(instances)} instance boxes")

    scoring = score_refs(refs, instances, bank_path)
    print(
        f"[natural-omission] scored      : {len(scoring['per_ref'])} refs "
        f"(missing_bank_images={len(scoring['missing_bank_images'])} "
        f"unscorable={len(scoring['unscorable_bank_images'])} "
        f"unresolved={len(scoring['unresolved_refs'])})"
    )

    counts_blob = split_counts(scoring["per_ref"])
    summary = build_summary(
        scoring,
        counts_blob,
        args,
        bank_meta=bank_meta,
        n_refs_total=len(refs),
        runtime_seconds=time.perf_counter() - started,
    )

    written = write_outputs(summary, out_dir)
    print(f"[natural-omission] summary     : {written['summary']}")
    print(f"[natural-omission] by_split    : {written['by_split']}")
    print(f"[natural-omission] sensitivity : {written['sensitivity']}")
    for row in summary["by_split"]:
        print(
            f"[natural-omission] {row['split']:<5} {row['unit']:<13} "
            f"miss={row['num_miss']}/{row['num_total']} rate={row['rate']:.6f} "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]"
        )
    print(
        f"[natural-omission] max_iou     : "
        + ", ".join(
            f"{name}={value:.4f}" if value is not None else f"{name}=n/a"
            for name, value in summary["max_iou_quantiles"].items()
            if name != "n" and name != "mean"
        )
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bank", type=Path, default=_REPO_ROOT / "cache" / "proposals.h5", help="bank-v1 h5"
    )
    parser.add_argument(
        "--refs",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "refs(unc).p",
        help="RefCOCO+ refs(unc).p",
    )
    parser.add_argument(
        "--instances",
        type=Path,
        default=_REPO_ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "instances.json",
        help="COCO-style instances.json joining ann_id -> bbox",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "results" / "natural_omission_full",
        help="output directory",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[natural-omission] FATAL: {exc}")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
