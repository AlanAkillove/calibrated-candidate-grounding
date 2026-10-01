"""D1 — Reviewed RefCOCO+ Annotation Robustness (V2-D, stage 1).

Question the stage answers (before any model is re-run):

    Can a *human-reviewed* RefCOCO+ annotation source be obtained and mapped
    back to the exact image_id / ref_id / sentence_id / target-ann identity of
    the V1 archive - and how much of the original archive survives the review?

Stage-1 pipeline (no model is trained or re-scored anywhere):

1. **Source acquisition audit** - fetch the reviewed annotation files
   (``refcocos_annotation_reviewed/refcoco+_{val,testA,testB}_reviewed.json``)
   from the HuggingFace dataset ``JierunChen/Ref-L4`` (paper: arXiv 2406.16866),
   verify each file against its frozen sha256, and record source / revision /
   license / download checksum / annotation schema in ``source_manifest.json``.
   The files live in the dedicated namespace ``data/reviewed_refcocoplus/raw/``
   and the original ``data/raw/refcoco+`` archive is *never* touched.
2. **Identity mapping** - map every original val/testA/testB sentence to its
   reviewed row via (image_id, ann_id) groups aligned by in-ref sent_id order,
   with a hard caption-text check; emits ``mapping_audit.csv`` with one row per
   original sentence and a status in ``{UNCHANGED, REMOVED, UNMATCHED}``
   (this source supports no other labels; CORRECTED_BOX / CORRECTED_TEXT /
   AMBIGUOUS / INVALID do not exist in it and are *declared unsupported*,
   never invented).
3. **Proposal compatibility audit** - for every kept (UNCHANGED) ref the target
   IoU mapping is *recomputed* from the frozen bank (``cache/proposals.h5``,
   N=64) and the instances join table via
   :func:`ccg.data.proposals.assign_target`, and verified against the frozen
   manifest row (the reviewed source leaves every box untouched, so the two
   must agree exactly; any disagreement is a hard error).  Reports target
   recall@0.5, K=5/10/20/50 availability and same-category availability, then
   applies the proposal gate (>=0.95 CONTINUE / 0.90-0.95 REVIEWED-PROPOSAL
   GRAY / <0.90 STOP).
4. **Feasibility + D1_clean + sample-size gate** - ``D1_clean`` = UNCHANGED and
   text/box-verified and target-present (recomputed IoU assignment >= 0.5).
   Sample-size gate on the clean test expressions: >=3000 expressions and
   >=500 images -> FULL robustness audit; 1000-2999 -> LIMITED; <1000 ->
   UNDERPOWERED stop.  Emits ``feasibility.json`` and ``clean_manifest.csv``.
5. **Removed/kept V1 diagnostics** - joins the REMOVED rows back to the frozen
   per-sentence V1 predictions and reports their accuracy / confidence / error
   rate next to the kept rows (a pure read-only diagnostic: did the original
   annotation noise inflate the measured model error?).

Artifacts land in ``results/v2_data_robustness/d1_reviewed_annotations/``:

    source_manifest.json, mapping_audit.csv, proposal_audit.json,
    feasibility.json, clean_manifest.csv, removed_ambiguous_audit.csv,
    removed_ambiguous_summary.json, figures/, metadata.json

Stage 2 (C1 cardinality + C4 hard-semantic replication with frozen weights) is
*not* part of this script; it is only run when both stage-1 gates pass.

Usage
-----
    python -u scripts/run_d1_reviewed.py
    python -u scripts/run_d1_reviewed.py --force-download   # re-fetch source files

0 new training parameters; every scored artifact read here is frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ccg.data.audit import (  # noqa: E402
    assign_gt_category,
    same_category_counts,
    target_availability,
)
from ccg.data.bank import read_bank_image  # noqa: E402
from ccg.data.coco import build_index as build_coco_index  # noqa: E402
from ccg.data.manifests import ManifestFile  # noqa: E402
from ccg.data.proposals import ProposalBank, assign_target, xywh_to_xyxy  # noqa: E402
from ccg.data.refcoco import load_instances_json, load_refs_pickle  # noqa: E402

__all__ = [
    "build_mapping",
    "fetch_reviewed_sources",
    "main",
    "mapping_statistics",
    "proposal_compatibility_audit",
    "run_d1_stage1",
    "size_gate",
]

# ---------------------------------------------------------------------------
# constants (frozen)
# ---------------------------------------------------------------------------
RAW_DIR = _REPO_ROOT / "data" / "reviewed_refcocoplus" / "raw"
OUT_DIR = _REPO_ROOT / "results" / "v2_data_robustness" / "d1_reviewed_annotations"

REFS_PATH = _REPO_ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "refs(unc).p"
INSTANCES_PATH = _REPO_ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "instances.json"
COCO_GT_PATH = _REPO_ROOT / "data" / "raw" / "annotations" / "instances_train2014.json"
BANK_PATH = _REPO_ROOT / "cache" / "proposals.h5"
MANIFEST_DIR = _REPO_ROOT / "cache" / "manifests"
NPZ_TEMPLATE = "results/phase0b_independent/seed_{seed}/per_sentence_predictions.npz"

HF_ENDPOINT = "https://hf-mirror.com"
HF_REPO = "JierunChen/Ref-L4"
HF_REVISION_AT_AUDIT = "d62c4f4e5f3a639b34adab34e5c8ffeb39f168c1"
HF_DIR = "refcocos_annotation_reviewed"
PAPER = (
    "arXiv:2406.16866 (Chen et al., 2024, 'Revisiting Referring Expression "
    "Comprehension Evaluation in the Era of Large Multimodal Models')"
)
LICENSE_NOTE = (
    "Dataset card: CC BY-NC 4.0; the RefCOCO+ portion carries its original "
    "Apache-2.0 terms (per the per-file 'licenses' field)."
)
#: reviewed file name -> frozen sha256 of the audited download
EXPECTED_SHA256: Dict[str, str] = {
    "refcoco+_val_reviewed.json": "c61080db850708f343f9bef5f408bc7c358a364d09f9fd843765a6ab0ceb262f",
    "refcoco+_testA_reviewed.json": "9e03439fc8ed189d169d65d6cb6ad455c7798b6de4109a4ebabcf7067b4987f8",
    "refcoco+_testB_reviewed.json": "1057944a4a1cc0ce2b8b8ac693be2144f736ee6bba640a656459bbbdc2fdb76f",
}
FILE_NAMES: Tuple[str, ...] = (
    "refcoco+_val_reviewed.json",
    "refcoco+_testA_reviewed.json",
    "refcoco+_testB_reviewed.json",
)
SPLITS: Tuple[str, ...] = ("val", "testA", "testB")
TEST_SPLITS: Tuple[str, ...] = ("testA", "testB")
KS: Tuple[int, ...] = (5, 10, 20, 50)
IOU_THRESH = 0.5
SEEDS: Tuple[int, ...] = (1, 2, 3)
BANK_TOP_N = 64

#: sample-size gate (protocol section 8)
SIZE_GATE_FULL_EXPRS = 3000
SIZE_GATE_FULL_IMAGES = 500
SIZE_GATE_LIMITED_EXPRS = 1000
#: proposal gate (protocol section 10)
PROPOSAL_GATE_CONTINUE = 0.95
PROPOSAL_GATE_GRAY = 0.90

DOWNLOAD_CHUNK = 1 << 20
DOWNLOAD_RETRIES = 3


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [D1] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_split(name: Any) -> str:
    lowered = str(name).strip().lower()
    return {"train": "train", "val": "val", "testa": "testA", "testb": "testB"}.get(
        lowered, lowered
    )


def _split_of_reviewed_file(name: str) -> str:
    """``refcoco+_testA_reviewed.json`` -> ``testA``."""
    return name.replace("refcoco+_", "").replace("_reviewed.json", "")


# ---------------------------------------------------------------------------
# 1. source acquisition audit
# ---------------------------------------------------------------------------
def _download(urls: Sequence[str], dest: Path) -> int:
    import urllib.request

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ccg-d1-audit/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
                size = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    size += len(chunk)
            return size
        except Exception as exc:  # noqa: BLE001 - retried below, re-raised when exhausted
            last_error = exc
    raise RuntimeError(f"could not download from {list(urls)}: {last_error}")


def fetch_reviewed_sources(*, force: bool = False, raw_dir: Path = RAW_DIR) -> Dict[str, Any]:
    """Fetch + verify the reviewed sources; returns the ``source_manifest`` payload."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = f"{HF_ENDPOINT}/datasets/{HF_REPO}/resolve/main/{HF_DIR}"
    files: Dict[str, Any] = {}
    for name in FILE_NAMES:
        dest = raw_dir / name
        expected = EXPECTED_SHA256[name]
        urls = [f"{base}/{name.replace('+', '%2B')}", f"{base}/{name}"]
        status = "prefetched_verified"
        if force or not dest.exists():
            status = "downloaded"
            log(f"downloading {name} ...")
            size = _download(urls, dest)
            log(f"  downloaded {size} bytes")
        digest = sha256_file(dest)
        if digest != expected:
            raise RuntimeError(
                f"{name}: sha256 {digest} does not match the audited checksum {expected} - "
                "the source is not the audited artifact; refusing to continue"
            )
        files[name] = {
            "urls_tried": urls,
            "bytes": int(dest.stat().st_size),
            "sha256": digest,
            "status": status,
        }
        log(f"source ok: {name} ({status}, sha256={digest[:16]}...)")
    return {
        "artifact": "d1_source_manifest",
        "source": {
            "dataset": f"HuggingFace datasets/{HF_REPO}",
            "directory": HF_DIR,
            "paper": PAPER,
            "revision_at_audit": HF_REVISION_AT_AUDIT,
            "license": LICENSE_NOTE,
            "endpoint": HF_ENDPOINT,
        },
        "annotation_schema": {
            "format": "COCO-like, one record per (image, expression) pair",
            "images[]": {
                "id": "review-local row id",
                "original_id": "COCO image id (matches the V1 archive)",
                "caption": "expression text (matches the V1 'sent' field)",
                "tokens_negative": "negative spans of the expression",
            },
            "annotations[]": {
                "id": "review-local annotation id",
                "image_id": "review-local image row id",
                "original_id": "original COCO instance id (= V1 ref ann_id)",
                "caption_quality": "1 = expression kept by the human review, 0 = excluded",
                "tokens_positive": "positive spans of the expression",
                "bbox": "xywh, verified equal to instances.json (zero box modifications)",
            },
            "extra_fields_ignored": ["model prediction columns", "categories"],
        },
        "files": files,
        "frozen_namespace": {
            "raw_dir": str(raw_dir.relative_to(_REPO_ROOT)),
            "note": "dedicated namespace; data/raw/refcoco+ is never modified",
        },
        "expected_sha256": dict(EXPECTED_SHA256),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# 2. identity mapping
# ---------------------------------------------------------------------------
def _load_reviewed_groups(path: Path) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """``(image_id, ann_id) -> [(caption, quality, bbox), ...]`` in file order."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in payload["images"]}
    groups: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for annotation in payload["annotations"]:
        image = images[int(annotation["image_id"])]
        key = (int(image["original_id"]), int(annotation["original_id"]))
        groups.setdefault(key, []).append(
            {
                "caption": str(image["caption"]),
                "quality": int(annotation["caption_quality"]),
                "bbox": [float(value) for value in annotation["bbox"]],
            }
        )
    return groups


def load_reviewed_by_split(raw_dir: Path = RAW_DIR) -> Dict[str, Dict[Tuple[int, int], List[Dict[str, Any]]]]:
    out: Dict[str, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {}
    for name in FILE_NAMES:
        out[_split_of_reviewed_file(name)] = _load_reviewed_groups(Path(raw_dir) / name)
    return out


def build_mapping(
    refs_records: Sequence[Mapping[str, Any]],
    reviewed_by_split: Mapping[str, Mapping[Tuple[int, int], Sequence[Mapping[str, Any]]]],
    instance_bbox: Mapping[int, Sequence[float]],
    *,
    box_atol: float = 1e-6,
) -> List[Dict[str, Any]]:
    """One row per original val/testA/testB sentence with mapping status.

    The (image_id, ann_id) group key is the join; inside a group the original
    sentences (in-ref sent_id ascending) are aligned positionally with the
    reviewed rows (file order) and the caption text is asserted equal - a group
    whose size or text sequence disagrees is reported UNMATCHED, never
    force-aligned.
    """
    rows: List[Dict[str, Any]] = []
    for record in refs_records:
        split = _normalize_split(record.get("split"))
        if split not in SPLITS:
            continue
        image_id = int(record["image_id"])
        ann_id = int(record["ann_id"])
        ref_id = int(record["ref_id"])
        sentences = sorted(record["sentences"], key=lambda item: int(item["sent_id"]))
        group = reviewed_by_split.get(split, {}).get((image_id, ann_id), [])
        gt_bbox = instance_bbox.get(ann_id)
        text_match = len(group) == len(sentences) and all(
            str(sentence["sent"]) == str(entry["caption"])
            for sentence, entry in zip(sentences, group)
        )
        group_ok = bool(group) and text_match
        for position, sentence in enumerate(sentences):
            entry = group[position] if group_ok else None
            box_match = False
            if entry is not None and gt_bbox is not None:
                box_match = bool(
                    np.allclose(
                        np.asarray(entry["bbox"], dtype=np.float64),
                        np.asarray(gt_bbox, dtype=np.float64),
                        atol=box_atol,
                    )
                )
            if entry is None:
                status = "UNMATCHED"
                quality: Optional[int] = None
            else:
                quality = int(entry["quality"])
                status = "UNCHANGED" if quality == 1 else "REMOVED"
            rows.append(
                {
                    "split": split,
                    "sent_id": int(sentence["sent_id"]),
                    "ref_id": ref_id,
                    "image_id": image_id,
                    "ann_id": ann_id,
                    "sentence": str(sentence["sent"]),
                    "group_pos": int(position),
                    "group_size": int(len(sentences)),
                    "reviewed_rows": int(len(group)),
                    "text_match": bool(text_match),
                    "box_match": bool(box_match),
                    "caption_quality": quality,
                    "status": status,
                }
            )
    rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["sent_id"]))
    return rows


def mapping_statistics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for split in list(SPLITS) + ["all"]:
        selected = [row for row in rows if split == "all" or row["split"] == split]
        if not selected:
            continue
        counts: Dict[str, int] = {}
        for row in selected:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        kept = [row for row in selected if row["status"] == "UNCHANGED"]
        verified = [row for row in kept if row["text_match"] and row["box_match"]]
        stats[split] = {
            "total_expressions": int(len(selected)),
            "mapped_expressions": int(len(selected) - counts.get("UNMATCHED", 0)),
            "unmatched_expressions": int(counts.get("UNMATCHED", 0)),
            "unchanged": int(counts.get("UNCHANGED", 0)),
            "removed": int(counts.get("REMOVED", 0)),
            "unchanged_verified": int(len(verified)),
            "images_total": int(len({row["image_id"] for row in selected})),
            "images_unchanged": int(len({row["image_id"] for row in kept})),
            "box_mismatch_rows": int(sum(not row["box_match"] for row in kept)),
            "text_mismatch_rows": int(sum(not row["text_match"] for row in selected)),
        }
    return stats


# ---------------------------------------------------------------------------
# 3. gates
# ---------------------------------------------------------------------------
def size_gate(clean_test_expressions: int, clean_test_images: int) -> Dict[str, Any]:
    if (
        clean_test_expressions >= SIZE_GATE_FULL_EXPRS
        and clean_test_images >= SIZE_GATE_FULL_IMAGES
    ):
        verdict, proceed = "FULL_ROBUSTNESS_AUDIT", True
    elif clean_test_expressions >= SIZE_GATE_LIMITED_EXPRS:
        verdict, proceed = "LIMITED_ROBUSTNESS_AUDIT", True
    else:
        verdict, proceed = "UNDERPOWERED", False
    return {
        "criterion": (
            f"clean test expressions >= {SIZE_GATE_FULL_EXPRS} and images >= "
            f"{SIZE_GATE_FULL_IMAGES} -> FULL; {SIZE_GATE_LIMITED_EXPRS}-"
            f"{SIZE_GATE_FULL_EXPRS - 1} -> LIMITED; < {SIZE_GATE_LIMITED_EXPRS} -> UNDERPOWERED"
        ),
        "clean_test_expressions": int(clean_test_expressions),
        "clean_test_images": int(clean_test_images),
        "verdict": verdict,
        "proceed": bool(proceed),
    }


def proposal_gate(recall: float) -> Dict[str, Any]:
    if recall >= PROPOSAL_GATE_CONTINUE:
        verdict = "CONTINUE"
    elif recall >= PROPOSAL_GATE_GRAY:
        verdict = "REVIEWED-PROPOSAL GRAY"
    else:
        verdict = "STOP"
    return {
        "criterion": (
            f"target recall@0.5 >= {PROPOSAL_GATE_CONTINUE} -> CONTINUE; "
            f"{PROPOSAL_GATE_GRAY}-{PROPOSAL_GATE_CONTINUE} -> REVIEWED-PROPOSAL GRAY; "
            f"< {PROPOSAL_GATE_GRAY} -> STOP"
        ),
        "recall_at_0.5": float(recall),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 4. proposal compatibility audit (recomputed IoU target mapping)
# ---------------------------------------------------------------------------
def _manifest_refs(split: str, manifest_dir: Path) -> Dict[int, Mapping[str, Any]]:
    manifest = ManifestFile.load(Path(manifest_dir) / f"random_{split}.jsonl")
    return {int(entry.ref_id): entry for entry in manifest.entries}


def proposal_compatibility_audit(
    kept_rows: Sequence[Mapping[str, Any]],
    instance_bbox: Mapping[int, Sequence[float]],
    *,
    bank_path: Path = BANK_PATH,
    coco_gt_path: Path = COCO_GT_PATH,
    manifest_dir: Path = MANIFEST_DIR,
) -> Tuple[Dict[str, Any], Dict[str, set], Dict[str, List[float]], Dict[str, List[int]]]:
    """Recompute the target IoU mapping of every kept ref against the frozen bank.

    Returns ``(report, matchable_by_split, ious_by_split, sames_by_split)``
    where ``matchable_by_split`` holds the ref_ids whose recomputed target
    assignment is present (best IoU >= 0.5) and the two last maps carry the
    per-ref figure data of the audit.
    """
    refs_by_split: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    for split in SPLITS:
        seen: Dict[int, Mapping[str, Any]] = {}
        for row in kept_rows:
            if row["split"] == split:
                seen.setdefault(int(row["ref_id"]), row)
        if seen:
            refs_by_split[split] = seen
    needed_images = sorted(
        {int(row["image_id"]) for split in refs_by_split.values() for row in split.values()}
    )
    log(
        f"proposal audit: {sum(len(v) for v in refs_by_split.values())} kept refs, "
        f"{len(needed_images)} images"
    )
    payload = json.loads(Path(coco_gt_path).read_text(encoding="utf-8"))
    gt_index = build_coco_index(payload, source=str(coco_gt_path), only_image_ids=needed_images)
    del payload

    bank_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    per_split: Dict[str, Any] = {}
    matchable_by_split: Dict[str, set] = {}
    mismatches = {"target_index": 0, "target_max_iou": 0, "n_same_category": 0}
    max_iou_delta = 0.0
    ious_by_split: Dict[str, List[float]] = {}
    sames_by_split: Dict[str, List[int]] = {}
    for split, refs in refs_by_split.items():
        manifest_refs = _manifest_refs(split, manifest_dir)
        n = len(refs)
        recall_hits = 0
        avail_counts = {k: 0 for k in KS}
        n_same_values: List[int] = []
        best_ious: List[float] = []
        matchable: set = set()
        for ref_id, row in refs.items():
            image_id = int(row["image_id"])
            if image_id not in bank_cache:
                boxes, objectness = read_bank_image(bank_path, image_id)
                bank_cache[image_id] = (
                    np.ascontiguousarray(np.asarray(boxes, dtype=np.float32)[:BANK_TOP_N]),
                    np.ascontiguousarray(np.asarray(objectness, dtype=np.float32)[:BANK_TOP_N]),
                )
            boxes64, obj64 = bank_cache[image_id]
            bbox = instance_bbox[int(row["ann_id"])]
            gt_box = xywh_to_xyxy(np.asarray(bbox, dtype=np.float32))
            bank = ProposalBank(image_id=image_id, boxes=boxes64, objectness=obj64)
            assignment = assign_target(bank, gt_box, iou_thresh=IOU_THRESH)
            best_iou = float(assignment.best_iou)
            best_ious.append(best_iou)
            if best_iou >= IOU_THRESH:
                recall_hits += 1
                matchable.add(int(ref_id))
            availability = target_availability(
                bank.boxes, assignment.target_proposal_idx, assignment.to_remove, Ks=KS
            )
            for k in KS:
                avail_counts[k] += int(bool(availability[k]))
            gt_objects = gt_index.objects(image_id)
            categories, _ = assign_gt_category(
                bank.boxes, gt_objects.boxes, gt_objects.categories, iou_thresh=IOU_THRESH
            )
            n_same = int(
                same_category_counts(
                    categories, assignment.target_proposal_idx, assignment.to_remove
                )
            )
            n_same_values.append(n_same)
            entry = manifest_refs.get(int(ref_id))
            if entry is None:
                raise RuntimeError(f"ref {ref_id} is not in the frozen random_{split} manifest")
            recomputed_idx = (
                -1 if assignment.target_proposal_idx is None else int(assignment.target_proposal_idx)
            )
            frozen_idx = -1 if entry.target_index is None else int(entry.target_index)
            if recomputed_idx != frozen_idx:
                mismatches["target_index"] += 1
            delta = abs(float(entry.target_max_iou) - best_iou)
            max_iou_delta = max(max_iou_delta, delta)
            if delta > 1e-5:
                mismatches["target_max_iou"] += 1
            if int(entry.n_same_category_available) != n_same:
                mismatches["n_same_category"] += 1
        same_arr = np.asarray(n_same_values, dtype=np.int64)
        ious_by_split[split] = best_ious
        sames_by_split[split] = n_same_values
        matchable_by_split[split] = matchable
        per_split[split] = {
            "n_refs": int(n),
            "target_recall_at_0.5": float(recall_hits / n) if n else None,
            "natural_miss_refs": int(n - recall_hits),
            "k_availability_refs": {str(k): int(avail_counts[k]) for k in KS},
            "k_availability_rate": {
                str(k): float(avail_counts[k] / n) if n else None for k in KS
            },
            "same_category": {
                "min": int(same_arr.min()) if n else None,
                "mean": float(same_arr.mean()) if n else None,
                "median": float(np.median(same_arr)) if n else None,
                "max": int(same_arr.max()) if n else None,
                "ge_4_refs": int((same_arr >= 4).sum()) if n else None,
                "ge_8_refs": int((same_arr >= 8).sum()) if n else None,
            },
        }
    pooled_refs = sum(per_split[s]["n_refs"] for s in per_split)
    pooled_hits = sum(
        per_split[s]["target_recall_at_0.5"] * per_split[s]["n_refs"] for s in per_split
    )
    pooled_recall = float(pooled_hits / pooled_refs) if pooled_refs else 0.0
    report = {
        "artifact": "d1_proposal_audit",
        "audited_population": "kept (UNCHANGED) refs of the reviewed source",
        "bank": {
            "path": str(Path(bank_path).relative_to(_REPO_ROOT)),
            "proposals_per_image": BANK_TOP_N,
            "iou_thresh": IOU_THRESH,
            "frozen": True,
        },
        "recomputed_target_mapping": True,
        "box_changed_by_review": False,
        "note": (
            "the reviewed source leaves every target box untouched (verified against "
            "instances.json), so the recomputed IoU mapping must equal the frozen "
            "manifest row; mismatches are reported below and are a hard error"
        ),
        "per_split": per_split,
        "pooled": {
            "n_refs": int(pooled_refs),
            "target_recall_at_0.5": pooled_recall,
            "k_availability_rate": {
                str(k): float(
                    sum(per_split[s]["k_availability_refs"][str(k)] for s in per_split)
                    / pooled_refs
                )
                if pooled_refs
                else None
                for k in KS
            },
        },
        "manifest_consistency": {
            "target_index_mismatches": int(mismatches["target_index"]),
            "target_max_iou_mismatches": int(mismatches["target_max_iou"]),
            "target_max_iou_max_abs_delta": float(max_iou_delta),
            "n_same_category_mismatches": int(mismatches["n_same_category"]),
        },
        "proposal_gate": proposal_gate(pooled_recall),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return report, matchable_by_split, ious_by_split, sames_by_split


# ---------------------------------------------------------------------------
# 5. removed/kept V1 diagnostics (protocol section 19)
# ---------------------------------------------------------------------------
def _npz_sentence_set() -> Dict[int, bool]:
    data = np.load(_REPO_ROOT / NPZ_TEMPLATE.format(seed=SEEDS[0]), allow_pickle=False)
    return {int(value): True for value in np.asarray(data["sentence_ids"], dtype=np.int64).tolist()}


def removed_v1_audit(
    mapping_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Accuracy / confidence / error rate of REMOVED vs KEPT rows on frozen V1 scores."""
    seed_payloads = []
    for seed in SEEDS:
        data = np.load(_REPO_ROOT / NPZ_TEMPLATE.format(seed=seed), allow_pickle=False)
        seed_payloads.append(
            {
                "sent_ids": np.asarray(data["sentence_ids"], dtype=np.int64),
                "correct": {k: np.asarray(data[f"correct_K{k}"], dtype=np.float64) for k in KS},
                "conf": {
                    k: np.asarray(data[f"confidence_global_T_corrected_K{k}"], dtype=np.float64)
                    for k in KS
                },
            }
        )
    index_of = {int(v): pos for pos, v in enumerate(seed_payloads[0]["sent_ids"].tolist())}
    rows_out: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"per_k": {}, "not_evaluated": {}}
    for k in KS:
        for subset in ("REMOVED", "UNCHANGED"):
            for split in SPLITS:
                sent_ids = [
                    int(row["sent_id"])
                    for row in mapping_rows
                    if row["split"] == split
                    and row["status"] == subset
                    and int(row["sent_id"]) in index_of
                ]
                if not sent_ids:
                    continue
                positions = np.asarray([index_of[sid] for sid in sent_ids], dtype=np.int64)
                correct_stack = np.stack(
                    [payload["correct"][k][positions] for payload in seed_payloads]
                )
                conf_stack = np.stack(
                    [payload["conf"][k][positions] for payload in seed_payloads]
                )
                correct_mean = float(correct_stack.mean())
                rows_out.append(
                    {
                        "split": split,
                        "subset": subset,
                        "K": int(k),
                        "n_evaluated": int(len(sent_ids)),
                        "accuracy": correct_mean,
                        "mean_confidence_global_T_corrected": float(conf_stack.mean()),
                        "error_rate": float(1.0 - correct_mean),
                    }
                )
        removed_all = [r for r in rows_out if r["K"] == k and r["subset"] == "REMOVED"]
        kept_all = [r for r in rows_out if r["K"] == k and r["subset"] == "UNCHANGED"]
        if removed_all and kept_all:
            n_removed = sum(r["n_evaluated"] for r in removed_all)
            n_kept = sum(r["n_evaluated"] for r in kept_all)
            err_removed = sum(r["error_rate"] * r["n_evaluated"] for r in removed_all)
            err_kept = sum(r["error_rate"] * r["n_evaluated"] for r in kept_all)
            total_err = err_removed + err_kept
            summary["per_k"][str(k)] = {
                "n_removed_evaluated": int(n_removed),
                "n_kept_evaluated": int(n_kept),
                "error_rate_removed": float(err_removed / n_removed) if n_removed else None,
                "error_rate_kept": float(err_kept / n_kept) if n_kept else None,
                "share_of_total_errors_from_removed": (
                    float(err_removed / total_err) if total_err else None
                ),
                "share_of_evaluated_rows_removed": float(n_removed / (n_removed + n_kept)),
            }
    summary["not_evaluated"] = {
        "removed_rows_without_v1": {
            split: int(
                sum(
                    1
                    for row in mapping_rows
                    if row["split"] == split
                    and row["status"] == "REMOVED"
                    and int(row["sent_id"]) not in index_of
                )
            )
            for split in SPLITS
        }
    }
    return rows_out, summary


# ---------------------------------------------------------------------------
# artifact writers
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def write_mapping_audit(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    _write_csv(
        path,
        rows,
        [
            "split",
            "sent_id",
            "ref_id",
            "image_id",
            "ann_id",
            "sentence",
            "group_pos",
            "group_size",
            "reviewed_rows",
            "text_match",
            "box_match",
            "caption_quality",
            "status",
        ],
    )


def build_clean_manifest(
    mapping_rows: Sequence[Mapping[str, Any]],
    matchable_by_split: Mapping[str, set],
    npz_by_sent: Mapping[int, bool],
) -> List[Dict[str, Any]]:
    clean: List[Dict[str, Any]] = []
    for row in mapping_rows:
        if row["status"] != "UNCHANGED" or not (row["text_match"] and row["box_match"]):
            continue
        if int(row["ref_id"]) not in matchable_by_split.get(row["split"], set()):
            continue
        clean.append(
            {
                "split": row["split"],
                "sent_id": row["sent_id"],
                "ref_id": row["ref_id"],
                "image_id": row["image_id"],
                "ann_id": row["ann_id"],
                "group_pos": row["group_pos"],
                "group_size": row["group_size"],
                "in_phase0b_cohort": bool(npz_by_sent.get(int(row["sent_id"]), False)),
            }
        )
    return clean


def write_clean_manifest(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    _write_csv(
        path,
        rows,
        [
            "split",
            "sent_id",
            "ref_id",
            "image_id",
            "ann_id",
            "group_pos",
            "group_size",
            "in_phase0b_cohort",
        ],
    )


def write_removed_audit(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    _write_csv(
        path,
        rows,
        [
            "split",
            "subset",
            "K",
            "n_evaluated",
            "accuracy",
            "mean_confidence_global_T_corrected",
            "error_rate",
        ],
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def write_figures(
    stats: Mapping[str, Any],
    ious_by_split: Mapping[str, Sequence[float]],
    sames_by_split: Mapping[str, Sequence[int]],
    out_dir: Path,
) -> None:
    figures = Path(out_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    splits = [split for split in SPLITS if split in stats]

    kept = [stats[split]["unchanged"] for split in splits]
    removed = [stats[split]["removed"] for split in splits]
    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, kept, label="UNCHANGED")
    ax.bar(x, removed, bottom=kept, label="REMOVED")
    ax.set_xticks(x, splits)
    ax.set_ylabel("expressions")
    ax.set_title("D1 mapping status by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "mapping_status_by_split.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    iou_values = np.concatenate(
        [np.asarray(ious_by_split.get(s, []), dtype=np.float64) for s in splits]
    )
    ax.hist(iou_values, bins=50, range=(0.0, 1.0))
    ax.axvline(IOU_THRESH, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("recomputed best target IoU")
    ax.set_ylabel("refs")
    ax.set_title("Reviewed target vs frozen N=64 bank")
    fig.tight_layout()
    fig.savefig(figures / "target_iou_hist.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    same_values = np.concatenate(
        [np.asarray(sames_by_split.get(s, []), dtype=np.int64) for s in splits]
    )
    ax.hist(same_values, bins=np.arange(0, int(same_values.max()) + 2) - 0.5)
    ax.set_xlabel("same-category distractors available")
    ax.set_ylabel("refs")
    ax.set_title("Same-category availability (reviewed target identity)")
    fig.tight_layout()
    fig.savefig(figures / "same_category_hist.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _frozen_input_hashes() -> Dict[str, str]:
    paths = {
        "data/raw/refcoco+/refcoco+/refs(unc).p": REFS_PATH,
        "data/raw/refcoco+/refcoco+/instances.json": INSTANCES_PATH,
        "data/raw/annotations/instances_train2014.json": COCO_GT_PATH,
        "cache/proposals.h5": BANK_PATH,
    }
    for name in (
        "random_val.jsonl",
        "random_testA.jsonl",
        "random_testB.jsonl",
        "same_category_val.jsonl",
        "same_category_testA.jsonl",
        "same_category_testB.jsonl",
    ):
        paths[f"cache/manifests/{name}"] = MANIFEST_DIR / name
    for seed in SEEDS:
        paths[f"results/phase0b_independent/seed_{seed}/per_sentence_predictions.npz"] = (
            _REPO_ROOT / NPZ_TEMPLATE.format(seed=seed)
        )
    return {name: sha256_file(path) for name, path in paths.items()}


def run_d1_stage1(*, force_download: bool = False, out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("stage 1 start")
    source_manifest = fetch_reviewed_sources(force=force_download)
    _write_json(out_dir / "source_manifest.json", source_manifest)

    refs_records = load_refs_pickle(REFS_PATH)
    instance_table = load_instances_json(INSTANCES_PATH)
    instance_bbox = {int(key): value["bbox"] for key, value in instance_table.items()}
    del instance_table
    reviewed_by_split = load_reviewed_by_split()
    log(
        "reviewed groups loaded: "
        f"{[(split, len(groups)) for split, groups in reviewed_by_split.items()]}"
    )

    mapping_rows = build_mapping(refs_records, reviewed_by_split, instance_bbox)
    write_mapping_audit(mapping_rows, out_dir / "mapping_audit.csv")
    stats = mapping_statistics(mapping_rows)
    log(
        "mapping rows: "
        f"{len(mapping_rows)}; kept/removed per split: "
        f"{ {s: (stats[s]['unchanged'], stats[s]['removed']) for s in SPLITS} }"
    )

    kept_rows = [row for row in mapping_rows if row["status"] == "UNCHANGED"]
    proposal_audit, matchable_by_split, ious_by_split, sames_by_split = (
        proposal_compatibility_audit(kept_rows, instance_bbox)
    )
    _write_json(out_dir / "proposal_audit.json", proposal_audit)
    log(
        f"proposal gate: {proposal_audit['proposal_gate']['verdict']} "
        f"(recall@0.5={proposal_audit['proposal_gate']['recall_at_0.5']:.6f})"
    )
    consistency = proposal_audit["manifest_consistency"]
    if any(
        consistency[key]
        for key in ("target_index_mismatches", "target_max_iou_mismatches", "n_same_category_mismatches")
    ):
        raise RuntimeError(f"proposal audit found manifest inconsistencies: {consistency}")

    clean_rows = [
        row
        for row in mapping_rows
        if row["status"] == "UNCHANGED"
        and row["text_match"]
        and row["box_match"]
        and int(row["ref_id"]) in matchable_by_split.get(row["split"], set())
    ]
    npz_by_sent = _npz_sentence_set()
    clean_test = [row for row in clean_rows if row["split"] in TEST_SPLITS]
    gate_size = size_gate(
        clean_test_expressions=len(clean_test),
        clean_test_images=len({row["image_id"] for row in clean_test}),
    )
    clean_stats = {
        split: {
            "n_expressions": int(sum(1 for row in clean_rows if row["split"] == split)),
            "n_images": int(
                len({row["image_id"] for row in clean_rows if row["split"] == split})
            ),
            "n_in_phase0b_cohort": int(
                sum(
                    1
                    for row in clean_rows
                    if row["split"] == split and npz_by_sent.get(int(row["sent_id"]), False)
                )
            ),
        }
        for split in SPLITS
    }
    feasibility = {
        "artifact": "d1_feasibility",
        "splits": stats,
        "reviewed_subset_note": (
            "The reviewed source covers the full val+testA+testB archives 1:1 "
            f"({sum(stats[s]['total_expressions'] for s in SPLITS)} sentences); train is "
            "not covered by the source and is not used by D1. Nothing is extrapolated "
            "beyond the reviewed splits."
        ),
        "status_taxonomy_supported": {
            "UNCHANGED": "caption_quality == 1",
            "REMOVED": "caption_quality == 0",
            "UNMATCHED": "no reviewed row (expected 0)",
            "CORRECTED_BOX": "not supported by this source (zero box modifications: verified)",
            "CORRECTED_TEXT": "not supported by this source (captions match the V1 'sent' text)",
            "AMBIGUOUS": "not supported by this source",
            "INVALID": "not supported by this source",
        },
        "d1_clean_definition": [
            "status == UNCHANGED (kept by the human review)",
            "mapping verified (group text sequence equal + target box equal to instances.json)",
            "target identity determined (ann_id present in the instances join table)",
            "target-present and proposal-matchable (recomputed IoU target assignment >= 0.5)",
        ],
        "d1_clean": {
            **clean_stats,
            "test_pooled": {
                "n_expressions": int(len(clean_test)),
                "n_images": int(len({row["image_id"] for row in clean_test})),
            },
        },
        "sample_size_gate": gate_size,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out_dir / "feasibility.json", feasibility)
    log(
        f"sample-size gate: {gate_size['verdict']} "
        f"(exprs={gate_size['clean_test_expressions']}, images={gate_size['clean_test_images']})"
    )

    if gate_size["proceed"]:
        clean_manifest = build_clean_manifest(mapping_rows, matchable_by_split, npz_by_sent)
        write_clean_manifest(clean_manifest, out_dir / "clean_manifest.csv")
        log(f"clean manifest written: {len(clean_manifest)} rows")
    else:
        log("sample-size gate says UNDERPOWERED - clean manifest + C1/C4 are stopped")

    removed_rows, removed_summary = removed_v1_audit(mapping_rows)
    write_removed_audit(removed_rows, out_dir / "removed_ambiguous_audit.csv")
    _write_json(out_dir / "removed_ambiguous_summary.json", removed_summary)
    log("removed/kept V1 diagnostics written")

    write_figures(stats, ious_by_split, sames_by_split, out_dir)

    metadata = {
        "artifact": "d1_metadata",
        "stage": "feasibility + mapping + proposal compatibility (no model run)",
        "new_training_parameters": 0,
        "runtime_sec": float(time.time() - started),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "frozen_inputs_sha256": _frozen_input_hashes(),
        "reviewed_source_sha256": dict(EXPECTED_SHA256),
        "gates": {
            "sample_size": gate_size,
            "proposal": proposal_audit["proposal_gate"],
        },
    }
    _write_json(out_dir / "metadata.json", metadata)
    log(f"stage 1 done in {metadata['runtime_sec']:.1f}s -> {out_dir}")
    return {
        "feasibility": feasibility,
        "proposal_audit": proposal_audit,
        "removed_summary": removed_summary,
        "metadata": metadata,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"artifact dir (default: {OUT_DIR})")
    parser.add_argument("--force-download", action="store_true", help="re-fetch the reviewed source files")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run_d1_stage1(force_download=bool(args.force_download), out_dir=args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
