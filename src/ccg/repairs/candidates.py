"""Versioned audit of proposal-category mismatch and true-target-category candidates.

This module reads the frozen RefCOCO+ annotations, proposal banks, manifests and
predictions. It never writes to the existing cache or result trees. The candidate
correction is emitted only under ``results/research_repair_v1/candidates`` by the
dedicated script.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccg.repairs.atomic import AtomicReplaceError, replace_with_retry
from ccg.data import bank as bank_io
from ccg.data.audit import assign_gt_category
from ccg.data.coco import CocoIndex, load_coco_index
from ccg.data.manifests import ManifestEntry, ManifestFile
from ccg.data.proposals import iou_matrix, xywh_to_xyxy
from ccg.data.refcoco import load_instances_json, load_refs_pickle
from ccg.data.splits import normalize_split_name

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "research_repair_v1" / "candidates"
REFS_PATH = ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "refs(unc).p"
INSTANCES_PATH = ROOT / "data" / "raw" / "refcoco+" / "refcoco+" / "instances.json"
COCO_PATH = ROOT / "data" / "raw" / "annotations" / "instances_train2014.json"
PRED_DIR = ROOT / "results" / "v2_proposal_robustness" / "predictions"
PHASE1F_DIR = ROOT / "results" / "phase1f_hard_semantic"
COCO_IOU_THRESHOLD = 0.5
TARGET_EQ_IOU_THRESHOLD = 0.5
FAMILIES = ("RPN", "DETR")
SEEDS = (1, 2, 3)
VARIANT = "true_target_category_v1"
SUPPLEMENTAL_MANIFEST_PATH = OUT_DIR / "supplemental_input_manifest.json"
BASE_MANIFEST_PATH = ROOT / "results" / "research_repair_v1" / "input_manifest.json"


class CandidateAuditError(RuntimeError):
    """A frozen input or candidate identity invariant failed."""


def candidate_audit_run_dir(run_id: str, *, base_dir: Path = OUT_DIR) -> Path:
    """Resolve a fresh, contained audit directory without overwriting prior outputs."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", str(run_id)):
        raise ValueError("run_id must be 1-64 ASCII letters, digits, underscores, or hyphens")
    base = Path(base_dir).resolve()
    output = (base / f"audit_run_{run_id}").resolve()
    try:
        output.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"candidate audit output escapes its base directory: {output}") from exc
    if output == base:
        raise ValueError("candidate audit run directory must be a child of the candidate output root")
    if output.exists():
        raise FileExistsError(f"candidate audit run directory already exists; refusing overwrite: {output}")
    return output


def write_current_audit_pointer(
    summary_path: Path, *, base_dir: Path = OUT_DIR, repo_root: Path = ROOT
) -> Dict[str, Any]:
    """Atomically point readers at one fully completed, hash-pinned candidate audit."""
    base = Path(base_dir).resolve()
    root = Path(repo_root).resolve()
    summary_file = Path(summary_path).resolve()
    try:
        summary_file.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"candidate summary escapes its output root: {summary_file}") from exc
    if summary_file.name != "summary.json" or summary_file.parent.parent != base:
        raise ValueError("current audit summary must be summary.json in a direct audit_run_<ID> child")
    if not summary_file.parent.name.startswith("audit_run_"):
        raise ValueError("current audit summary is not inside audit_run_<ID>")
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    if payload.get("schema") != "research-repair-v1-candidate-summary-v1":
        raise CandidateAuditError("candidate summary schema mismatch; current pointer not updated")
    if payload.get("status") not in ("AUDIT_COMPLETE_MISMATCH_FOUND", "AUDIT_COMPLETE_NO_MISMATCH"):
        raise CandidateAuditError("candidate audit is not complete; current pointer not updated")
    summary_rel = summary_file.relative_to(root).as_posix()
    pointer = {
        "schema": "research-repair-v1-current-audit-pointer-v1",
        "status": payload["status"],
        "audit_run_id": summary_file.parent.name[len("audit_run_"):],
        "summary_path": summary_rel,
        "summary_sha256": _sha256(summary_file),
    }
    _write_json(base / "current_audit.json", pointer)
    return pointer


@dataclass(frozen=True)
class CellSpec:
    """One already-scored hard or dose candidate cell."""

    name: str
    k: int
    same_count: int
    kind: str


CELL_SPECS: Mapping[str, CellSpec] = {
    "hard5": CellSpec("hard5", 5, 4, "hard"),
    "hard10": CellSpec("hard10", 10, 9, "hard"),
    "expb_m0": CellSpec("expb_m0", 10, 0, "dose"),
    "expb_m2": CellSpec("expb_m2", 10, 2, "dose"),
    "expb_m4": CellSpec("expb_m4", 10, 4, "dose"),
    "expb_m8": CellSpec("expb_m8", 10, 8, "dose"),
}

FAMILY_CELLS: Mapping[str, Tuple[str, ...]] = {
    # RPN is the Phase-1F hard/dose reference and has all frozen cells.
    "RPN": tuple(CELL_SPECS),
    # The proposal-family replication has a frozen DETR hard-K5 cell only.
    "DETR": ("hard5",),
}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [repair-candidates] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_supplemental_input_manifest(
    *, manifest_path: Path = SUPPLEMENTAL_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Snapshot the one required COCO GT input omitted from the prepared baseline.

    Existing snapshots are immutable: a later invocation verifies them rather than
    refreshing their hashes. The ``inputs`` entries use the root baseline schema.
    """
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        return verify_supplemental_input_manifest(manifest_path=manifest_path)
    if not BASE_MANIFEST_PATH.exists():
        raise CandidateAuditError(f"baseline input manifest missing: {BASE_MANIFEST_PATH}")
    if not COCO_PATH.is_file():
        raise FileNotFoundError(f"supplemental COCO GT file missing: {COCO_PATH}")
    baseline = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    baseline_paths = {str(item["path"]) for item in baseline.get("inputs", [])}
    coco_rel = COCO_PATH.relative_to(ROOT).as_posix()
    if coco_rel in baseline_paths:
        raise CandidateAuditError("COCO GT is already in the 794-file baseline; refusing a redundant supplement")
    payload = {
        "schema": "research-repair-input-manifest-v1",
        "purpose": "supplemental inputs omitted from the prepared 794-file baseline",
        "baseline_manifest": BASE_MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "baseline_manifest_sha256": _sha256(BASE_MANIFEST_PATH),
        "baseline_n_files": int(baseline.get("n_files", -1)),
        "source_path_gap": {
            "prepared_expected_path": "data/raw/mscoco/annotations/instances_train2014.json",
            "prepared_expected_path_exists": bool(
                (ROOT / "data/raw/mscoco/annotations/instances_train2014.json").exists()
            ),
            "actual_source_path": coco_rel,
            "reason": "the candidate audit loads the repository's actual COCO GT path; it was not covered by the prepared baseline",
        },
        "inputs": [{
            "path": coco_rel,
            "size_bytes": int(COCO_PATH.stat().st_size),
            "sha256": _sha256(COCO_PATH),
        }],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents an accidental replacement if another process races.
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return verify_supplemental_input_manifest(manifest_path=manifest_path)


def verify_supplemental_input_manifest(
    *, manifest_path: Path = SUPPLEMENTAL_MANIFEST_PATH,
) -> Dict[str, Any]:
    """Verify the immutable supplemental snapshot and baseline-manifest anchor."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise CandidateAuditError(
            "supplemental_input_manifest.json must be created before loading COCO GT"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "research-repair-input-manifest-v1":
        raise CandidateAuditError("supplemental input manifest schema mismatch")
    if payload.get("baseline_manifest_sha256") != _sha256(BASE_MANIFEST_PATH):
        raise CandidateAuditError("baseline input manifest changed after supplemental snapshot")
    mismatches = []
    for item in payload.get("inputs", []):
        path = ROOT / str(item["path"])
        if not path.is_file():
            mismatches.append({"path": item["path"], "reason": "missing"})
            continue
        actual_size = int(path.stat().st_size)
        actual_hash = _sha256(path)
        if actual_size != int(item["size_bytes"]) or actual_hash != item["sha256"]:
            mismatches.append({
                "path": item["path"], "reason": "identity_changed",
                "expected_size_bytes": int(item["size_bytes"]), "actual_size_bytes": actual_size,
                "expected_sha256": item["sha256"], "actual_sha256": actual_hash,
            })
    if mismatches:
        raise CandidateAuditError(f"supplemental input verification failed: {mismatches}")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Only remove the uniquely named temporary file owned by this call.
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    # Exhausted replace retries deliberately retain the complete temp for recovery.
    replace_with_retry(temp, path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=float, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Write a versioned candidate array archive atomically inside its fresh run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"._{uuid.uuid4().hex[:12]}.tmp")
    try:
        with temp.open("xb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    replace_with_retry(temp, path)


def _prediction_path(family: str, job: str, seed: int) -> Path:
    return PRED_DIR / f"p1__{family}__{job}__b3_seed{seed}.npz"


def load_prediction_identity(family: str, job: str) -> Dict[str, np.ndarray]:
    """Load and cross-check identity columns for a frozen job across B3 seeds."""
    fields = ("sentence_id", "ref_id", "image_id")
    reference: Optional[Dict[str, np.ndarray]] = None
    for seed in SEEDS:
        path = _prediction_path(family, job, seed)
        if not path.exists():
            raise FileNotFoundError(f"missing frozen prediction: {path}")
        with np.load(path, allow_pickle=False) as archive:
            current = {name: np.asarray(archive[name], dtype=np.int64) for name in fields}
        n = current["sentence_id"].size
        if any(value.shape != (n,) for value in current.values()):
            raise CandidateAuditError(f"{family}/{job}/seed{seed}: malformed identity columns")
        if np.unique(current["sentence_id"]).size != n:
            raise CandidateAuditError(f"{family}/{job}/seed{seed}: sentence_id is not unique")
        if reference is None:
            reference = current
        else:
            for name in fields:
                if not np.array_equal(reference[name], current[name]):
                    raise CandidateAuditError(
                        f"{family}/{job}: seed{seed} {name} rows do not match seed1"
                    )
    assert reference is not None
    order = np.argsort(reference["sentence_id"], kind="stable")
    return {name: values[order] for name, values in reference.items()}


def load_test_manifests(family: str) -> Tuple[Dict[int, ManifestEntry], Dict[int, ManifestEntry]]:
    """Return random and same-category entries for the frozen testA/testB split."""
    root = ROOT / ("cache/manifests" if family == "RPN" else "cache/manifests_detr/manifests")
    tables: List[Dict[int, ManifestEntry]] = []
    for regime in ("random", "same_category"):
        table: Dict[int, ManifestEntry] = {}
        for split in ("testA", "testB"):
            path = root / f"{regime}_{split}.jsonl"
            manifest = ManifestFile.load(path)
            for entry in manifest.entries:
                if int(entry.ref_id) in table:
                    raise CandidateAuditError(f"duplicate ref_id={entry.ref_id} across test splits")
                table[int(entry.ref_id)] = entry
        tables.append(table)
    return tables[0], tables[1]


def build_ref_annotation_index() -> Dict[int, Dict[str, int]]:
    """Join UNC ``ref_id`` to its referring ``ann_id`` and COCO category."""
    refs = load_refs_pickle(REFS_PATH)
    instances = load_instances_json(INSTANCES_PATH)
    out: Dict[int, Dict[str, int]] = {}
    for record in refs:
        ref_id = int(record["ref_id"])
        ann_id = int(record["ann_id"])
        image_id = int(record["image_id"])
        split = normalize_split_name(record["split"])
        ann = instances.get(str(ann_id))
        if ann is None:
            raise CandidateAuditError(f"ref {ref_id}: ann_id={ann_id} missing from instances.json")
        if int(ann["image_id"]) != image_id:
            raise CandidateAuditError(
                f"ref {ref_id}: ann_id={ann_id} belongs to image {ann['image_id']}, not {image_id}"
            )
        row = {
            "ann_id": ann_id,
            "image_id": image_id,
            "category_id": int(ann["category_id"]),
            "split": split,
        }
        previous = out.setdefault(ref_id, row)
        if previous != row:
            raise CandidateAuditError(f"ref_id={ref_id}: conflicting referring annotation mapping")
    return out


def assign_proposals_to_objects(
    boxes: Any,
    gt_boxes: Any,
    gt_categories: Any,
    gt_object_ids: Any,
    *,
    iou_threshold: float = COCO_IOU_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """Return frozen category assignment plus its deterministic GT object identity."""
    proposal_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    object_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    object_categories = np.asarray(gt_categories, dtype=np.int64).reshape(-1)
    object_ids = np.asarray(gt_object_ids, dtype=np.int64).reshape(-1)
    if object_boxes.shape[0] != object_categories.size or object_ids.size != object_categories.size:
        raise ValueError("COCO GT boxes/categories/object IDs are misaligned")
    assigned_categories, best_iou = assign_gt_category(
        proposal_boxes, object_boxes, object_categories, iou_thresh=iou_threshold
    )
    if object_boxes.shape[0] == 0:
        best_object = np.full(proposal_boxes.shape[0], -1, dtype=np.int64)
    else:
        overlaps = iou_matrix(proposal_boxes, object_boxes)
        best_gt = overlaps.argmax(axis=1)
        best_object = object_ids[best_gt].astype(np.int64)
        best_object = np.where(best_iou >= float(iou_threshold), best_object, -1).astype(np.int64)
    return {
        "category_id": np.asarray(assigned_categories, dtype=np.int64),
        "object_id": best_object,
        "best_iou": np.asarray(best_iou, dtype=np.float64),
    }


def build_versioned_candidate_indices(
    *,
    target_index: int,
    valid_indices: Any,
    proposal_categories: Any,
    true_target_category: int,
    old_hard_order: Any,
    spec: CellSpec,
) -> Optional[np.ndarray]:
    """Build one corrected cell, preserving the frozen remainder ordering.

    The corrected same-category prefix is ascending bank index, as in manifests-v1.
    Fillers retain the old same-category manifest's frozen random remainder ordering,
    with proposals newly promoted to the true target category removed from that tail.
    """
    target = int(target_index)
    valid = np.asarray(valid_indices, dtype=np.int64).reshape(-1)
    cats = np.asarray(proposal_categories, dtype=np.int64).reshape(-1)
    old_order = np.asarray(old_hard_order, dtype=np.int64).reshape(-1)
    if target in set(valid.tolist()):
        raise ValueError("target_index must not occur in valid distractor indices")
    if np.unique(valid).size != valid.size or np.unique(old_order).size != old_order.size:
        raise ValueError("valid indices and frozen remainder order must be unique")
    if np.any(valid < 0) or np.any(valid >= cats.size):
        raise ValueError("valid candidate index is outside proposal-category vector")
    if not set(old_order.tolist()).issubset(set(valid.tolist())):
        raise ValueError("old same-category ordering contains an invalid candidate")
    same = np.sort(valid[cats[valid] == int(true_target_category)])
    rest = old_order[~np.isin(old_order, same)]
    if same.size < spec.same_count or rest.size < spec.k - 1 - spec.same_count:
        return None
    if spec.kind == "hard":
        distractors = same[: spec.k - 1]
    elif spec.kind == "dose":
        n_hard = int(spec.same_count)
        n_fill = int(spec.k - 1 - n_hard)
        distractors = np.concatenate([same[:n_hard], rest[:n_fill]])
    else:  # pragma: no cover - specs are frozen constants
        raise ValueError(f"unknown candidate cell kind {spec.kind!r}")
    result = np.concatenate([np.asarray([target], dtype=np.int64), distractors.astype(np.int64)])
    if result.shape != (spec.k,) or np.unique(result).size != spec.k:
        raise CandidateAuditError(f"{spec.name}: candidate construction is not a unique K-set")
    if result[0] != target:
        raise CandidateAuditError(f"{spec.name}: target is not in canonical slot zero")
    return result


def original_candidate_indices(entry: ManifestEntry, spec: CellSpec) -> Optional[np.ndarray]:
    """Reconstruct one historical cell from its immutable same-category manifest."""
    if entry.target_index is None:
        return None
    if spec.kind == "hard":
        if int(entry.n_same_category_available) < spec.same_count:
            return None
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        distractors = order[: spec.k - 1]
    else:
        if int(entry.n_same_category_available) < 8:
            return None
        order = np.asarray(entry.distractor_order, dtype=np.int64)
        n_same = int(entry.n_same_category_available)
        n_fill = spec.k - 1 - spec.same_count
        distractors = np.concatenate([order[: spec.same_count], order[n_same : n_same + n_fill]])
    result = np.concatenate([[int(entry.target_index)], np.asarray(distractors, dtype=np.int64)])
    if result.shape != (spec.k,) or np.unique(result).size != spec.k:
        raise CandidateAuditError(f"{spec.name}: frozen old candidate cell is malformed")
    return result


def availability_intersection_counts(
    historical_ids: Iterable[int], corrected_ids: Iterable[int]
) -> Dict[str, Any]:
    """Count old/new availability and attrition on an identified source cohort."""
    old = {int(value) for value in historical_ids}
    new = {int(value) for value in corrected_ids}
    overlap = old & new
    return {
        "historical_old_available_rows": len(old),
        "corrected_true_category_available_rows": len(new),
        "old_new_common_rows": len(overlap),
        "old_only_rows_lost_under_corrected_category": len(old - new),
        "corrected_only_rows_not_in_historical_cohort": len(new - old),
        "old_common_retention": len(overlap) / len(old) if old else None,
        "new_common_retention": len(overlap) / len(new) if new else None,
        "old_new_common_sentence_ids": sorted(overlap),
    }


def dose_common_cohort(
    cell_sentence_ids: Mapping[str, Iterable[int]],
    cells: Sequence[str] = ("expb_m0", "expb_m2", "expb_m4", "expb_m8"),
) -> List[int]:
    """Return sorted sentence IDs available at every requested fixed-K dose level."""
    if not cells:
        return []
    sets = [{int(value) for value in cell_sentence_ids.get(cell, ())} for cell in cells]
    return sorted(set.intersection(*sets))


def _quantiles(values: Sequence[float], prefix: str) -> Dict[str, Optional[float]]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{suffix}": None for suffix in ("mean", "p10", "median", "p90")}
    qs = np.quantile(arr, [0.10, 0.50, 0.90])
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_p10": float(qs[0]),
        f"{prefix}_median": float(qs[1]),
        f"{prefix}_p90": float(qs[2]),
    }


def candidate_geometry(
    candidate_indices: Any,
    *,
    boxes: Any,
    objectness: Any,
    proposal_object_ids: Any,
    proposal_best_iou: Any,
    proposal_categories: Any,
    target_index: int,
    true_target_category_id: int,
    true_target_ann_id: int,
    target_proposal_object_id: int,
    target_gt_box: Any,
    image_area: float,
) -> Dict[str, Any]:
    """Per-expression candidate/object, objectness, area, IoU and duplication stats."""
    indices = np.asarray(candidate_indices, dtype=np.int64).reshape(-1)
    proposal_boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    obj = np.asarray(objectness, dtype=np.float64).reshape(-1)
    object_ids = np.asarray(proposal_object_ids, dtype=np.int64).reshape(-1)
    best_iou = np.asarray(proposal_best_iou, dtype=np.float64).reshape(-1)
    categories = np.asarray(proposal_categories, dtype=np.int64).reshape(-1)
    if any(a.size != proposal_boxes.shape[0] for a in (obj, object_ids, best_iou, categories)):
        raise ValueError("proposal geometry inputs are misaligned")
    if np.any(indices < 0) or np.any(indices >= proposal_boxes.shape[0]):
        raise ValueError("candidate index is outside proposal bank")
    selected_boxes = proposal_boxes[indices]
    selected_object_ids = object_ids[indices]
    matched_mask = selected_object_ids >= 0
    counts = Counter(int(value) for value in selected_object_ids[matched_mask].tolist())
    distinct_ids = sorted(counts)
    pair_duplicates = int(sum(count * (count - 1) // 2 for count in counts.values()))
    duplicate_excess = int(matched_mask.sum() - len(distinct_ids))
    gt_target_ious = iou_matrix(
        selected_boxes,
        np.asarray(target_gt_box, dtype=np.float32).reshape(1, 4),
    )[:, 0].astype(np.float64)
    pair_iou = iou_matrix(selected_boxes, selected_boxes)
    upper = pair_iou[np.triu_indices(indices.size, k=1)]
    areas = np.asarray(
        np.maximum(selected_boxes[:, 2] - selected_boxes[:, 0], 0.0)
        * np.maximum(selected_boxes[:, 3] - selected_boxes[:, 1], 0.0),
        dtype=np.float64,
    )
    out: Dict[str, Any] = {
        "n_proposals": int(indices.size),
        "n_matched_proposals": int(matched_mask.sum()),
        "n_unmatched_to_coco_object_iou_ge_0_5": int((~matched_mask).sum()),
        "n_distinct_matched_gt_objects": int(len(distinct_ids)),
        "duplicate_excess_proposals": duplicate_excess,
        "duplicate_gt_object_pairs": pair_duplicates,
        "n_proposals_assigned_true_target_object": int(np.count_nonzero(
            selected_object_ids == int(true_target_ann_id))),
        "n_proposals_assigned_target_proposal_gt_object": int(np.count_nonzero(
            selected_object_ids == int(target_proposal_object_id)
        )) if target_proposal_object_id >= 0 else 0,
        "n_proposal_iou_pairs_ge_0_5": int(np.count_nonzero(upper >= 0.5)),
        "proposal_iou_pair_count": int(upper.size),
        "n_proposals_assigned_true_target_category": int(np.count_nonzero(
            categories[indices] == int(true_target_category_id)
        )),
        "n_proposals_assigned_target_proposal_category": int(np.count_nonzero(
            categories[indices] == int(categories[int(target_index)])
        )),
        "unique_gt_object_ids_json": json.dumps(distinct_ids, separators=(",", ":")),
        "proposals_per_gt_object_json": json.dumps(
            {str(k): int(counts[k]) for k in distinct_ids}, separators=(",", ":")
        ),
    }
    out.update(_quantiles(obj[indices].tolist(), "objectness"))
    out.update(_quantiles(areas.tolist(), "area_px2"))
    out.update(_quantiles((areas / max(float(image_area), 1.0)).tolist(), "area_fraction"))
    out.update(_quantiles(best_iou[indices].tolist(), "max_iou_to_coco_object"))
    out.update(_quantiles(gt_target_ious.tolist(), "iou_to_true_target_object"))
    out["geometric_duplicate_pair_fraction"] = (
        float(np.count_nonzero(upper >= 0.5) / upper.size) if upper.size else 0.0
    )
    out["gt_object_duplicate_fraction"] = (
        float(duplicate_excess / matched_mask.sum()) if matched_mask.any() else None
    )
    return out


def summarize_geometry_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-expression geometry with explicit proposal vs object counts."""
    if not rows:
        return {"n_expressions": 0, "n_proposals": 0, "n_distinct_gt_object_references": 0}
    scalar_fields = (
        "n_proposals", "n_matched_proposals", "n_unmatched_to_coco_object_iou_ge_0_5",
        "n_distinct_matched_gt_objects", "duplicate_excess_proposals",
        "duplicate_gt_object_pairs", "n_proposals_assigned_true_target_object",
        "n_proposals_assigned_target_proposal_gt_object", "n_proposal_iou_pairs_ge_0_5",
    )
    out: Dict[str, Any] = {
        "n_expressions": int(len(rows)),
        "n_images": int(len({int(row["image_id"]) for row in rows})),
        "n_distinct_gt_object_references": int(len({
            (int(row["image_id"]), int(obj_id))
            for row in rows
            for obj_id in json.loads(str(row["unique_gt_object_ids_json"]))
        })),
    }
    for field in scalar_fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        out[f"{field}_sum"] = float(values.sum())
        out[f"{field}_mean_per_expression"] = float(values.mean())
        out[f"{field}_median_per_expression"] = float(np.median(values))
    # Proposal-level properties are summarized as the mean over expressed cells and as
    # a pooled proposal-weighted estimate; both are useful when cell sizes vary.
    per_row_n = np.asarray([int(row["n_proposals"]) for row in rows], dtype=np.int64)
    out["total_proposals"] = int(per_row_n.sum())
    for field in (
        "objectness_mean", "objectness_p10", "objectness_median", "objectness_p90",
        "area_px2_mean", "area_px2_p10", "area_px2_median", "area_px2_p90",
        "area_fraction_mean", "area_fraction_p10", "area_fraction_median", "area_fraction_p90",
        "max_iou_to_coco_object_mean", "max_iou_to_coco_object_p10",
        "max_iou_to_coco_object_median", "max_iou_to_coco_object_p90",
        "iou_to_true_target_object_mean", "iou_to_true_target_object_p10",
        "iou_to_true_target_object_median", "iou_to_true_target_object_p90",
    ):
        values = [row.get(field) for row in rows if row.get(field) is not None]
        out[f"{field}_mean_over_expressions"] = float(np.mean(values)) if values else None
    out["mean_geometric_duplicate_pair_fraction"] = float(np.mean(
        [float(row["geometric_duplicate_pair_fraction"]) for row in rows]
    ))
    dup = [row.get("gt_object_duplicate_fraction") for row in rows
           if row.get("gt_object_duplicate_fraction") is not None]
    out["mean_gt_object_duplicate_fraction"] = float(np.mean(dup)) if dup else None
    return out


def _cohort_mask_for_cell(entry: ManifestEntry, spec: CellSpec) -> bool:
    if entry.target_index is None:
        return False
    if spec.kind == "hard":
        return bool(entry.eligible.get(spec.k, False) and
                    int(entry.n_same_category_available) >= spec.same_count)
    # Historical dose cells shared the frozen same8 cohort at K10.
    return bool(entry.eligible.get(10, False) and int(entry.n_same_category_available) >= 8)


def _build_cell_candidate(
    *,
    entry: ManifestEntry,
    random_entry: ManifestEntry,
    proposal_categories: np.ndarray,
    true_target_category: int,
    spec: CellSpec,
    corrected: bool,
) -> Optional[np.ndarray]:
    if entry.target_index is None or random_entry.target_index is None:
        return None
    if int(entry.target_index) != int(random_entry.target_index):
        raise CandidateAuditError(f"ref {entry.ref_id}: random/hard target proposal mismatch")
    # The pool is the same valid pool in both frozen regimes. We reconstruct it from the
    # random manifest, whose ordering contains each valid distractor exactly once.
    valid = np.asarray(random_entry.distractor_order, dtype=np.int64)
    if corrected:
        old_n_same = int(entry.n_same_category_available)
        old_order = np.asarray(entry.distractor_order, dtype=np.int64)[old_n_same:]
        return build_versioned_candidate_indices(
            target_index=int(entry.target_index), valid_indices=valid,
            proposal_categories=proposal_categories,
            true_target_category=int(true_target_category), old_hard_order=old_order,
            spec=spec,
        )
    return original_candidate_indices(entry, spec)


def _candidate_indices_npz_path(family: str, cell: str) -> Path:
    return PHASE1F_DIR / "manifests" / f"{cell}_candidates.npz"


def _build_historical_candidate_sources(
    *,
    family: str,
    row_context: Mapping[Tuple[str, int], Sequence[Mapping[str, Any]]],
    hard_identity: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Recover each actually evaluated hard/dose candidate cohort and its old sets."""
    rows_by_k = {
        k: {int(row["sentence_id"]): row for row in row_context[(family, k)]}
        for k in (5, 10)
    }
    sources: List[Dict[str, Any]] = []
    if family == "RPN":
        for cell in CELL_SPECS:
            path = _candidate_indices_npz_path(family, cell)
            with np.load(path, allow_pickle=False) as archive:
                ids = np.asarray(archive["sentence_id"], dtype=np.int64)
                refs = np.asarray(archive["ref_id"], dtype=np.int64)
                images = np.asarray(archive["image_id"], dtype=np.int64)
                candidates = np.asarray(archive["candidate_indices"], dtype=np.int64)
                targets = np.asarray(archive["target_index"], dtype=np.int64)
            spec = CELL_SPECS[cell]
            k = 5 if cell == "hard5" else 10
            if candidates.shape != (ids.size, spec.k):
                raise CandidateAuditError(f"{cell}: malformed frozen candidate array shape")
            source_rows: List[Dict[str, Any]] = []
            seen: set[int] = set()
            for sid, ref, image, indices, target in zip(ids, refs, images, candidates, targets):
                sid_i, ref_i, image_i = int(sid), int(ref), int(image)
                if sid_i in seen:
                    raise CandidateAuditError(f"{cell}: duplicate archived sentence_id={sid_i}")
                seen.add(sid_i)
                row = rows_by_k[k].get(sid_i)
                if row is None:
                    raise CandidateAuditError(f"{cell}: archived sentence_id={sid_i} absent from frozen identities")
                if int(row["ref_id"]) != ref_i or int(row["image_id"]) != image_i:
                    raise CandidateAuditError(f"{cell}/sentence{sid_i}: source ref/image identity mismatch")
                if int(row["target_index"]) != int(target):
                    raise CandidateAuditError(f"{cell}/sentence{sid_i}: archived target proposal mismatch")
                expected = original_candidate_indices(row["hard_entry"], spec)
                if expected is None or not np.array_equal(indices, expected):
                    raise CandidateAuditError(f"{cell}/sentence{sid_i}: candidate archive differs from frozen manifest")
                source_rows.append({**row, "old_candidate_indices": indices.astype(np.int64)})
            sources.append({
                "family": family, "cell": cell, "source": "phase1f_frozen_candidate_archive",
                "source_path": path.relative_to(ROOT).as_posix(), "source_rows": source_rows,
            })

    # V2-P has a separate frozen hard-K5 evaluated cohort for both proposal families.
    # Its candidate indices are reconstructed from the immutable same-category manifest.
    if family in FAMILIES:
        spec = CELL_SPECS["hard5"]
        source_rows = []
        for sid, ref, image in zip(
            hard_identity["sentence_id"], hard_identity["ref_id"], hard_identity["image_id"]
        ):
            sid_i, ref_i, image_i = int(sid), int(ref), int(image)
            row = rows_by_k[5].get(sid_i)
            if row is None:
                raise CandidateAuditError(f"p1 hard5: sentence_id={sid_i} absent from random-K5 identity")
            if int(row["ref_id"]) != ref_i or int(row["image_id"]) != image_i:
                raise CandidateAuditError(f"p1 hard5/sentence{sid_i}: source ref/image identity mismatch")
            candidate = original_candidate_indices(row["hard_entry"], spec)
            if candidate is None:
                raise CandidateAuditError(f"p1 hard5/sentence{sid_i}: frozen manifest has no hard-K5 candidate")
            source_rows.append({**row, "old_candidate_indices": candidate})
        sources.append({
            "family": family, "cell": "hard5", "source": "v2_p1_frozen_hard_k5_predictions",
            "source_path": str(_prediction_path(family, "hard_k5", 1).relative_to(ROOT).as_posix()),
            "source_rows": source_rows,
        })
    return sources


def _verify_phase1f_candidate_artifacts(family: str) -> Dict[str, Any]:
    """Check RPN's archived candidate arrays against the frozen manifests."""
    if family != "RPN":
        return {"applicable": False, "reason": "DETR candidate lists are frozen in manifests"}
    checks: Dict[str, Any] = {}
    for cell in FAMILY_CELLS[family]:
        path = _candidate_indices_npz_path(family, cell)
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray(archive["sentence_id"], dtype=np.int64)
            refs = np.asarray(archive["ref_id"], dtype=np.int64)
            images = np.asarray(archive["image_id"], dtype=np.int64)
            candidates = np.asarray(archive["candidate_indices"], dtype=np.int64)
            targets = np.asarray(archive["target_index"], dtype=np.int64)
        if candidates.ndim != 2 or candidates.shape[0] != ids.size:
            raise CandidateAuditError(f"{cell}: malformed frozen candidate_indices array")
        by_id = {int(sid): pos for pos, sid in enumerate(ids.tolist())}
        checks[cell] = {
            "rows": int(ids.size), "unique_sentence_id": int(len(by_id)) == int(ids.size),
            "target_first": bool(np.all(candidates[:, 0] == targets)),
            "candidate_k": int(candidates.shape[1]),
        }
        if not checks[cell]["unique_sentence_id"] or not checks[cell]["target_first"]:
            raise CandidateAuditError(f"{cell}: archived identities/target slot violate invariant")
        # The archive is a frozen artifact. Full candidate equality is checked after the
        # cell tables have been built, when the manifest-derived rows are available.
        checks[cell]["identity_sha256"] = hashlib.sha256(
            np.column_stack([ids, refs, images]).astype("<i8", copy=False).tobytes()
        ).hexdigest()
    return checks


def run_candidate_audit(*, out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Run the authorized CPU audit and save new artifacts under ``out_dir`` only."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    supplemental_manifest = verify_supplemental_input_manifest(
        manifest_path=SUPPLEMENTAL_MANIFEST_PATH
    )
    _log("loading frozen prediction identities and RefCOCO+ annotation joins")
    identities: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    hard_identities: Dict[str, Dict[str, np.ndarray]] = {}
    for family in FAMILIES:
        identities[(family, 5)] = load_prediction_identity(family, "random_k5")
        identities[(family, 10)] = load_prediction_identity(family, "random_k10")
        hard_identities[family] = load_prediction_identity(family, "hard_k5")
    ref_info = build_ref_annotation_index()
    ref_by_family: Dict[str, Dict[int, ManifestEntry]] = {}
    hard_by_family: Dict[str, Dict[int, ManifestEntry]] = {}
    row_context: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    needed_images: set[int] = set()
    for family in FAMILIES:
        random_entries, hard_entries = load_test_manifests(family)
        ref_by_family[family] = random_entries
        hard_by_family[family] = hard_entries
        for k in (5, 10):
            identity = identities[(family, k)]
            rows: List[Dict[str, Any]] = []
            for sid, ref, image in zip(identity["sentence_id"], identity["ref_id"], identity["image_id"]):
                sid_i, ref_i, image_i = int(sid), int(ref), int(image)
                meta = ref_info.get(ref_i)
                if meta is None:
                    raise CandidateAuditError(f"{family} sentence_id={sid_i}: ref_id={ref_i} missing")
                if meta["image_id"] != image_i:
                    raise CandidateAuditError(
                        f"{family} sentence_id={sid_i}: prediction/ref image ids disagree"
                    )
                if meta["split"] not in ("testA", "testB"):
                    raise CandidateAuditError(f"{family} sentence_id={sid_i}: non-test split {meta['split']}")
                re = random_entries.get(ref_i)
                he = hard_entries.get(ref_i)
                if re is None or he is None:
                    raise CandidateAuditError(f"{family} sentence_id={sid_i}: frozen test manifest missing ref")
                if int(re.image_id) != image_i or int(he.image_id) != image_i:
                    raise CandidateAuditError(f"{family} sentence_id={sid_i}: manifest image id mismatch")
                if re.target_index != he.target_index:
                    raise CandidateAuditError(f"{family} ref_id={ref_i}: random/hard target index differs")
                rows.append({
                    "sentence_id": sid_i, "ref_id": ref_i, "image_id": image_i,
                    "eval_split": meta["split"], "ann_id": meta["ann_id"],
                    "true_target_category_id": meta["category_id"],
                    "target_index": int(re.target_index) if re.target_index is not None else -1,
                    "random_entry": re, "hard_entry": he,
                })
                needed_images.add(image_i)
            row_context[(family, k)] = rows
        _log(
            f"{family}: random K5={len(row_context[(family, 5)])}, "
            f"random K10={len(row_context[(family, 10)])}, "
            f"V2 hard K5={len(hard_identities[family]['sentence_id'])}"
        )

    # The COCO index is pruned to the pooled-test images represented in the frozen rows.
    _log(f"loading COCO objects for {len(needed_images)} audited images")
    coco = load_coco_index([COCO_PATH], only_image_ids=sorted(needed_images))
    categories = dict(coco.category_names)
    target_objects: Dict[int, Dict[str, Any]] = {}
    for image_id in sorted(needed_images):
        objects = coco.objects(image_id)
        for pos, ann_id in enumerate(objects.object_ids.tolist()):
            target_objects[int(ann_id)] = {
                "image_id": int(image_id), "category_id": int(objects.categories[pos]),
                "bbox_xyxy": objects.boxes[pos].astype(np.float64),
            }
    # Confirm RefCOCO+'s ann_id/category mapping agrees with the source COCO annotation.
    for key, meta in ref_info.items():
        if int(meta["image_id"]) not in needed_images:
            continue
        target = target_objects.get(int(meta["ann_id"]))
        if target is None:
            raise CandidateAuditError(
                f"ref_id={key}: ann_id={meta['ann_id']} absent from source COCO objects"
            )
        if int(target["image_id"]) != int(meta["image_id"]):
            raise CandidateAuditError(f"ref_id={key}: target ann_id resolves to another image")
        if int(target["category_id"]) != int(meta["category_id"]):
            raise CandidateAuditError(
                f"ref_id={key}: category disagreement RefCOCO+={meta['category_id']} "
                f"COCO={target['category_id']}"
            )

    # Read each proposal bank once per image and retain only its small box/objectness
    # arrays plus GT assignment vectors. K remains the fixed bank dimension (64).
    proposal_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for family in FAMILIES:
        bank_path = ROOT / ("cache/proposals.h5" if family == "RPN" else "cache/proposals_detr_r50.h5")
        image_ids = sorted({int(row["image_id"]) for k in (5, 10) for row in row_context[(family, k)]})
        for image_id in image_ids:
            boxes, objectness = bank_io.read_bank_image(bank_path, image_id)
            gt = coco.objects(image_id)
            assignment = assign_proposals_to_objects(
                boxes, gt.boxes, gt.categories, gt.object_ids,
                iou_threshold=COCO_IOU_THRESHOLD,
            )
            # Category assignment must match the repository's frozen primitive exactly.
            cats_frozen, ious_frozen = assign_gt_category(
                boxes, gt.boxes, gt.categories, iou_thresh=COCO_IOU_THRESHOLD
            )
            if not np.array_equal(assignment["category_id"], cats_frozen) or not np.array_equal(
                assignment["best_iou"], np.asarray(ious_frozen, dtype=np.float64)
            ):
                raise CandidateAuditError(f"{family}/image{image_id}: category assignment drift")
            proposal_cache[(family, image_id)] = {
                "boxes": np.asarray(boxes, dtype=np.float32),
                "objectness": np.asarray(objectness, dtype=np.float64),
                **assignment,
                "image_area": float(coco.image(image_id).width * coco.image(image_id).height),
            }
        _log(f"{family}: assigned COCO categories for {len(image_ids)} proposal banks")

    mismatch_rows: List[Dict[str, Any]] = []
    mismatches_by_family: Dict[str, Dict[int, bool]] = {}
    target_proposal_category_by_family: Dict[str, Dict[int, int]] = {}
    for family in FAMILIES:
        per_sid: Dict[int, bool] = {}
        assigned_by_sid: Dict[int, int] = {}
        for row in row_context[(family, 5)]:
            image_id = int(row["image_id"])
            target_index = int(row["target_index"])
            info = proposal_cache[(family, image_id)]
            gt = target_objects[int(row["ann_id"])]
            target_box = np.asarray(gt["bbox_xyxy"], dtype=np.float32).reshape(1, 4)
            target_prop_iou = float(iou_matrix(info["boxes"][target_index:target_index + 1], target_box)[0, 0])
            assigned_cat = int(info["category_id"][target_index])
            assigned_obj = int(info["object_id"][target_index])
            best_iou = float(info["best_iou"][target_index])
            true_cat = int(row["true_target_category_id"])
            mismatch = bool(assigned_cat != true_cat)
            per_sid[int(row["sentence_id"])] = mismatch
            assigned_by_sid[int(row["sentence_id"])] = assigned_cat
            mismatch_rows.append({
                "family": family, "sentence_id": int(row["sentence_id"]),
                "ref_id": int(row["ref_id"]), "image_id": image_id,
                "eval_split": str(row["eval_split"]), "ann_id": int(row["ann_id"]),
                "true_target_category_id": true_cat,
                "true_target_category_name": categories.get(true_cat, ""),
                "target_proposal_index": target_index,
                "target_proposal_assigned_category_id": assigned_cat,
                "target_proposal_assigned_category_name": categories.get(assigned_cat, "UNKNOWN/-1"),
                "target_proposal_assigned_gt_object_id": assigned_obj,
                "target_proposal_max_iou_to_coco_gt": best_iou,
                "target_proposal_iou_to_referring_ann": target_prop_iou,
                "category_mismatch": int(mismatch),
                "mismatch_type": (
                    "unknown_target_proposal_category" if assigned_cat < 0
                    else "known_category_disagreement" if mismatch else "match"
                ),
            })
        mismatches_by_family[family] = per_sid
        target_proposal_category_by_family[family] = assigned_by_sid

    _log("summarizing expression/category/image/object mismatch tables")
    mismatch_by_category: List[Dict[str, Any]] = []
    category_counts: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    category_confusion: Dict[Tuple[str, int, int, int], int] = Counter()
    image_counts: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    object_counts: Dict[Tuple[str, int, int], Counter] = defaultdict(Counter)
    for row in mismatch_rows:
        family, true_cat = str(row["family"]), int(row["true_target_category_id"])
        assigned_cat = int(row["target_proposal_assigned_category_id"])
        mismatch = int(row["category_mismatch"])
        cat = category_counts[(family, true_cat)]
        cat["n_expressions"] += 1
        cat["n_mismatches"] += mismatch
        cat["n_unknown_proposal_category"] += int(assigned_cat < 0)
        category_confusion[(family, true_cat, assigned_cat,
                            int(row["target_proposal_assigned_gt_object_id"]))] += 1
        ic = image_counts[(family, int(row["image_id"]))]
        ic["n_expressions"] += 1
        ic["n_mismatches"] += mismatch
        if "ann_ids" not in ic:
            ic["ann_ids"] = {int(row["ann_id"])}
        else:
            ic["ann_ids"].add(int(row["ann_id"]))
        oc = object_counts[(family, int(row["image_id"]), int(row["ann_id"]))]
        oc["n_expressions"] += 1
        if "n_refs" not in oc:
            oc["n_refs"] = {int(row["ref_id"])}
        else:
            oc["n_refs"].add(int(row["ref_id"]))
        oc["n_mismatches"] += mismatch
        oc[f"proposal_category_{assigned_cat}"] += 1
    for (family, category_id), count in sorted(category_counts.items()):
        mismatch_by_category.append({
            "family": family, "true_target_category_id": category_id,
            "true_target_category_name": categories.get(category_id, ""),
            "n_expressions": int(count["n_expressions"]),
            "n_mismatches": int(count["n_mismatches"]),
            "mismatch_rate": float(count["n_mismatches"] / count["n_expressions"]),
            "n_unknown_target_proposal_category": int(count["n_unknown_proposal_category"]),
        })
    image_rows = []
    for (family, image_id), count in sorted(image_counts.items()):
        image_rows.append({
            "family": family, "image_id": image_id,
            "n_expressions": int(count["n_expressions"]),
            "n_mismatches": int(count["n_mismatches"]),
            "n_distinct_target_objects": int(len(count["ann_ids"])),
        })
    object_rows = []
    for (family, image_id, ann_id), count in sorted(object_counts.items()):
        obj = target_objects[ann_id]
        object_rows.append({
            "family": family, "image_id": image_id, "ann_id": ann_id,
            "true_target_category_id": int(obj["category_id"]),
            "true_target_category_name": categories.get(int(obj["category_id"]), ""),
            "n_refs": int(len(count["n_refs"])), "n_expressions": int(count["n_expressions"]),
            "n_mismatches": int(count["n_mismatches"]),
            "mismatch_rate": float(count["n_mismatches"] / count["n_expressions"]),
            "proposal_category_counts_json": json.dumps(
                {key.removeprefix("proposal_category_"): int(value)
                 for key, value in count.items() if key.startswith("proposal_category_")},
                separators=(",", ":"),
            ),
        })
    confusion_rows = [{
        "family": family, "true_target_category_id": true_cat,
        "true_target_category_name": categories.get(true_cat, ""),
        "target_proposal_category_id": assigned_cat,
        "target_proposal_category_name": categories.get(assigned_cat, "UNKNOWN/-1"),
        "target_proposal_assigned_gt_object_id": object_id,
        "n_expressions": int(count),
    } for (family, true_cat, assigned_cat, object_id), count in sorted(category_confusion.items())]

    # If no mismatch is observed, record the audit and skip candidate reconstruction.
    mismatch_total = int(sum(int(row["category_mismatch"]) for row in mismatch_rows))
    mismatch_by_fam = {
        family: {
            "n_expressions": int(sum(1 for row in mismatch_rows if row["family"] == family)),
            "n_mismatches": int(sum(int(row["category_mismatch"]) for row in mismatch_rows
                                     if row["family"] == family)),
            "mismatch_rate": float(np.mean([int(row["category_mismatch"]) for row in mismatch_rows
                                            if row["family"] == family])),
            "n_unknown_target_proposal_category": int(sum(
                int(row["mismatch_type"] == "unknown_target_proposal_category")
                for row in mismatch_rows if row["family"] == family
            )),
        } for family in FAMILIES
    }

    availability_rows: List[Dict[str, Any]] = []
    geometry_summary_rows: List[Dict[str, Any]] = []
    geometry_expression_rows: List[Dict[str, Any]] = []
    sensitivity_paired_by_cell: Dict[Tuple[str, str], Dict[int, Dict[str, Any]]] = {}
    candidate_npz_payload: Dict[str, np.ndarray] = {}
    candidate_checks: Dict[str, Any] = {}
    corrected_constructed = mismatch_total > 0
    if corrected_constructed:
        _log("mismatch detected; building versioned true-target-category candidates")
        for family in FAMILIES:
            for cell in FAMILY_CELLS[family]:
                spec = CELL_SPECS[cell]
                k_cohort = 5 if cell == "hard5" else 10
                rows = row_context[(family, k_cohort)]
                old_avail_ids: List[int] = []
                new_avail_ids: List[int] = []
                paired_rows: List[Dict[str, Any]] = []
                old_candidates: Dict[int, np.ndarray] = {}
                new_candidates: Dict[int, np.ndarray] = {}
                for row in rows:
                    he: ManifestEntry = row["hard_entry"]
                    re: ManifestEntry = row["random_entry"]
                    sid = int(row["sentence_id"])
                    old_mask = _cohort_mask_for_cell(he, spec)
                    old_candidate = _build_cell_candidate(
                        entry=he, random_entry=re,
                        proposal_categories=proposal_cache[(family, int(row["image_id"]))]["category_id"],
                        true_target_category=int(row["true_target_category_id"]),
                        spec=spec, corrected=False,
                    )
                    if old_mask:
                        if old_candidate is None:
                            raise CandidateAuditError(f"{family}/{cell}/{sid}: old cohort candidate missing")
                        old_avail_ids.append(sid)
                        old_candidates[sid] = old_candidate
                    new_candidate: Optional[np.ndarray] = None
                    if he.target_index is not None and re.target_index is not None:
                        info = proposal_cache[(family, int(row["image_id"]))]
                        # Valid pool from the frozen random manifest; candidate order is not
                        # taken from the target-category label used by the old samecat cell.
                        new_candidate = _build_cell_candidate(
                            entry=he, random_entry=re,
                            proposal_categories=info["category_id"],
                            true_target_category=int(row["true_target_category_id"]),
                            spec=spec, corrected=True,
                        )
                    if new_candidate is not None:
                        new_avail_ids.append(sid)
                    if old_mask and new_candidate is not None:
                        if old_candidate is None:
                            raise CandidateAuditError(f"{family}/{cell}/{sid}: old candidate missing on overlap")
                        paired_rows.append({
                            **row, "old_candidate_indices": old_candidate,
                            "candidate_indices": new_candidate,
                        })
                old_set, new_set = set(old_avail_ids), set(new_avail_ids)
                intersection = old_set & new_set
                cell_name = f"{family}/{cell}"
                availability_rows.append({
                    "family": family, "cell": cell, "k": int(spec.k),
                    "cell_kind": spec.kind, "same_category_proposals_required": int(spec.same_count),
                    "evaluation_universe_rows": int(len(rows)),
                    "historical_old_available_rows": int(len(old_set)),
                    "corrected_true_category_available_rows": int(len(new_set)),
                    "old_new_common_rows": int(len(intersection)),
                    "old_only_rows_lost_under_corrected_category": int(len(old_set - new_set)),
                    "corrected_only_rows_not_in_historical_cohort": int(len(new_set - old_set)),
                    "old_common_retention": float(len(intersection) / len(old_set)) if old_set else None,
                    "new_common_retention": float(len(intersection) / len(new_set)) if new_set else None,
                    "new_variant": VARIANT,
                    "comparison_cohort": "old_new_common_availability_intersection",
                })
                if cell == "expb_m8":
                    candidate_checks[f"{family}/dose_common_m8"] = {
                        "old_available": int(len(old_set)), "new_available": int(len(new_set)),
                        "intersection": int(len(intersection)),
                    }
                paired_rows.sort(key=lambda row: int(row["sentence_id"]))
                sensitivity_paired_by_cell[(family, cell)] = {
                    int(row["sentence_id"]): row for row in paired_rows
                }
                if paired_rows:
                    prefix = f"{family}__{cell}"
                    candidate_npz_payload[f"{prefix}__sentence_id"] = np.asarray(
                        [row["sentence_id"] for row in paired_rows], dtype=np.int64
                    )
                    candidate_npz_payload[f"{prefix}__ref_id"] = np.asarray(
                        [row["ref_id"] for row in paired_rows], dtype=np.int64
                    )
                    candidate_npz_payload[f"{prefix}__image_id"] = np.asarray(
                        [row["image_id"] for row in paired_rows], dtype=np.int64
                    )
                    candidate_npz_payload[f"{prefix}__ann_id"] = np.asarray(
                        [row["ann_id"] for row in paired_rows], dtype=np.int64
                    )
                    candidate_npz_payload[f"{prefix}__old_indices"] = np.stack(
                        [row["old_candidate_indices"] for row in paired_rows]
                    ).astype(np.int64)
                    candidate_npz_payload[f"{prefix}__true_target_category_v1_indices"] = np.stack(
                        [row["candidate_indices"] for row in paired_rows]
                    ).astype(np.int64)
                # Candidate geometry is compared only on the old/new paired intersection.
                for variant_field in ("old_candidate_indices", "candidate_indices"):
                    variant_name = "old_proposal_category" if variant_field == "old_candidate_indices" else VARIANT
                    row_metrics: List[Dict[str, Any]] = []
                    for row in paired_rows:
                        info = proposal_cache[(family, int(row["image_id"]))]
                        true_obj = target_objects[int(row["ann_id"])]
                        target_index = int(row["target_index"])
                        target_prop_object = int(info["object_id"][target_index])
                        geom = candidate_geometry(
                            row[variant_field], boxes=info["boxes"], objectness=info["objectness"],
                            proposal_object_ids=info["object_id"], proposal_best_iou=info["best_iou"],
                            proposal_categories=info["category_id"], target_index=target_index,
                            true_target_category_id=int(row["true_target_category_id"]),
                            true_target_ann_id=int(row["ann_id"]),
                            target_proposal_object_id=target_prop_object,
                            target_gt_box=true_obj["bbox_xyxy"], image_area=info["image_area"],
                        )
                        detail = {
                            "family": family, "cell": cell, "variant": variant_name,
                            "sentence_id": int(row["sentence_id"]), "ref_id": int(row["ref_id"]),
                            "image_id": int(row["image_id"]), "ann_id": int(row["ann_id"]),
                            "true_target_category_id": int(row["true_target_category_id"]),
                            **geom,
                        }
                        row_metrics.append(detail)
                        geometry_expression_rows.append(detail)
                    summary = summarize_geometry_rows(row_metrics)
                    geometry_summary_rows.append({
                        "family": family, "cell": cell, "variant": variant_name,
                        "n_proposals_per_candidate_set": int(spec.k),
                        "comparison_cohort": "manifest_candidate_supply_old_new_intersection",
                        **summary,
                    })
        # A nested dose comparison should use one cohort available at all four old/new levels.
        for family in FAMILIES:
            if family != "RPN":
                continue
            dose_sets: Dict[str, set[int]] = {}
            for cell in ("expb_m0", "expb_m2", "expb_m4", "expb_m8"):
                prefix = f"{family}__{cell}__sentence_id"
                dose_sets[cell] = set(np.asarray(candidate_npz_payload.get(prefix, []), dtype=np.int64).tolist())
            common_dose = dose_common_cohort(dose_sets)
            candidate_checks[f"{family}/all_dose_intersection"] = {
                "rows": int(len(common_dose)),
                "definition": "P1 K10 identity rows old/new candidate-supply available at every m in {0,2,4,8}; measured source cohorts are separately enumerated",
            }

    # Always audit the original evaluated hard/dose candidate sources, including when
    # mismatch==0. These source cohorts differ (Phase1F versus V2-P hard-K5), so they
    # remain separately identified rather than being collapsed to one common cohort.
    historical_sources: List[Dict[str, Any]] = []
    source_mismatch_coverage: List[Dict[str, Any]] = []
    source_candidate_availability: List[Dict[str, Any]] = []
    source_availability_summary: List[Dict[str, Any]] = []
    baseline_geometry_summary_rows: List[Dict[str, Any]] = []
    baseline_geometry_expression_rows: List[Dict[str, Any]] = []
    sensitivity_geometry_summary_rows: List[Dict[str, Any]] = []
    sensitivity_geometry_expression_rows: List[Dict[str, Any]] = []
    for family in FAMILIES:
        historical_sources.extend(_build_historical_candidate_sources(
            family=family, row_context=row_context, hard_identity=hard_identities[family]
        ))
    for source_record in historical_sources:
        family = str(source_record["family"])
        cell = str(source_record["cell"])
        source_name = str(source_record["source"])
        spec = CELL_SPECS[cell]
        source_rows = list(source_record["source_rows"])
        paired_rows = sensitivity_paired_by_cell.get((family, cell), {})
        base_metrics: List[Dict[str, Any]] = []
        sensitivity_metrics: List[Dict[str, Any]] = []
        new_available_ids: set[int] = set()
        mismatched_source_ids: List[int] = []
        source_prefix = f"{family}__{source_name}__{cell}"
        for row in source_rows:
            sid = int(row["sentence_id"])
            if bool(mismatches_by_family[family].get(sid, False)):
                mismatched_source_ids.append(sid)
            info = proposal_cache[(family, int(row["image_id"]))]
            ann_id = int(row["ann_id"])
            true_obj = target_objects[ann_id]
            target_index = int(row["target_index"])
            target_prop_object = int(info["object_id"][target_index])
            old_indices = np.asarray(row["old_candidate_indices"], dtype=np.int64)
            old_metric = candidate_geometry(
                old_indices, boxes=info["boxes"], objectness=info["objectness"],
                proposal_object_ids=info["object_id"], proposal_best_iou=info["best_iou"],
                proposal_categories=info["category_id"], target_index=target_index,
                true_target_category_id=int(row["true_target_category_id"]),
                true_target_ann_id=ann_id, target_proposal_object_id=target_prop_object,
                target_gt_box=true_obj["bbox_xyxy"], image_area=info["image_area"],
            )
            baseline_detail = {
                "family": family, "cell": cell, "candidate_source": source_name,
                "source_path": source_record["source_path"], "variant": "original_frozen_candidate_set",
                "sentence_id": sid, "ref_id": int(row["ref_id"]),
                "image_id": int(row["image_id"]), "ann_id": ann_id,
                "true_target_category_id": int(row["true_target_category_id"]),
                **old_metric,
            }
            base_metrics.append(baseline_detail)
            baseline_geometry_expression_rows.append(baseline_detail)

            corrected_row = paired_rows.get(sid)
            corrected_indices = None
            if corrected_constructed and corrected_row is not None:
                # A source candidate must match the independently reconstructed old set.
                if not np.array_equal(old_indices, corrected_row["old_candidate_indices"]):
                    raise CandidateAuditError(
                        f"{family}/{source_name}/{cell}/sentence{sid}: source old set differs from candidate manifest"
                    )
                corrected_indices = np.asarray(corrected_row["candidate_indices"], dtype=np.int64)
                new_available_ids.add(sid)
                new_metric = candidate_geometry(
                    corrected_indices, boxes=info["boxes"], objectness=info["objectness"],
                    proposal_object_ids=info["object_id"], proposal_best_iou=info["best_iou"],
                    proposal_categories=info["category_id"], target_index=target_index,
                    true_target_category_id=int(row["true_target_category_id"]),
                    true_target_ann_id=ann_id, target_proposal_object_id=target_prop_object,
                    target_gt_box=true_obj["bbox_xyxy"], image_area=info["image_area"],
                )
                sensitivity_detail = {
                    "family": family, "cell": cell, "candidate_source": source_name,
                    "source_path": source_record["source_path"],
                    "variant": VARIANT, "sentence_id": sid,
                    "ref_id": int(row["ref_id"]), "image_id": int(row["image_id"]),
                    "ann_id": ann_id,
                    "true_target_category_id": int(row["true_target_category_id"]),
                    **new_metric,
                }
                sensitivity_metrics.append(sensitivity_detail)
                sensitivity_geometry_expression_rows.append(sensitivity_detail)
                source_candidate_availability.append({
                    "family": family, "cell": cell, "candidate_source": source_name,
                    "source_path": source_record["source_path"], "sentence_id": sid,
                    "ref_id": int(row["ref_id"]), "image_id": int(row["image_id"]),
                    "ann_id": ann_id, "true_target_category_id": int(row["true_target_category_id"]),
                    "historical_candidate_available": 1, "corrected_candidate_available": 1,
                    "in_old_new_common_intersection": 1,
                })
            else:
                source_candidate_availability.append({
                    "family": family, "cell": cell, "candidate_source": source_name,
                    "source_path": source_record["source_path"], "sentence_id": sid,
                    "ref_id": int(row["ref_id"]), "image_id": int(row["image_id"]),
                    "ann_id": ann_id, "true_target_category_id": int(row["true_target_category_id"]),
                    "historical_candidate_available": 1,
                    "corrected_candidate_available": 0 if corrected_constructed else None,
                    "in_old_new_common_intersection": 0 if corrected_constructed else None,
                })

        baseline_geometry_summary_rows.append({
            "family": family, "cell": cell, "candidate_source": source_name,
            "source_path": source_record["source_path"], "variant": "original_frozen_candidate_set",
            "n_proposals_per_candidate_set": int(spec.k),
            "comparison_cohort": "historical_evaluated_source_cohort",
            **summarize_geometry_rows(base_metrics),
        })
        if corrected_constructed:
            availability_counts = availability_intersection_counts(
                [int(row["sentence_id"]) for row in source_rows], new_available_ids
            )
            source_availability_summary.append({
                "family": family, "cell": cell, "candidate_source": source_name,
                "source_path": source_record["source_path"], "k": int(spec.k),
                "cell_kind": spec.kind,
                "historical_evaluated_rows": int(len(source_rows)),
                **{key: value for key, value in availability_counts.items()
                   if key != "old_new_common_sentence_ids"},
                "historical_rows_lost_under_corrected_category": int(len(source_rows) - len(new_available_ids)),
                "historical_retention": float(len(new_available_ids) / len(source_rows)) if source_rows else None,
                "mismatched_target_proposal_rows": int(len(mismatched_source_ids)),
                "candidate_variant": VARIANT,
                "comparison_cohort": "historical_source_old_new_common_intersection",
            })
            if sensitivity_metrics:
                sensitivity_geometry_summary_rows.append({
                    "family": family, "cell": cell, "candidate_source": source_name,
                    "source_path": source_record["source_path"], "variant": VARIANT,
                    "n_proposals_per_candidate_set": int(spec.k),
                    "comparison_cohort": "historical_source_old_new_common_intersection",
                    **summarize_geometry_rows(sensitivity_metrics),
                })
            if len(new_available_ids) > 0:
                selected_source_rows = [row for row in source_rows if int(row["sentence_id"]) in new_available_ids]
                candidate_npz_payload[f"{source_prefix}__sentence_id"] = np.asarray(
                    [row["sentence_id"] for row in selected_source_rows], dtype=np.int64
                )
                candidate_npz_payload[f"{source_prefix}__ref_id"] = np.asarray(
                    [row["ref_id"] for row in selected_source_rows], dtype=np.int64
                )
                candidate_npz_payload[f"{source_prefix}__image_id"] = np.asarray(
                    [row["image_id"] for row in selected_source_rows], dtype=np.int64
                )
                candidate_npz_payload[f"{source_prefix}__ann_id"] = np.asarray(
                    [row["ann_id"] for row in selected_source_rows], dtype=np.int64
                )
                candidate_npz_payload[f"{source_prefix}__old_indices"] = np.stack([
                    np.asarray(row["old_candidate_indices"], dtype=np.int64)
                    for row in selected_source_rows
                ])
                candidate_npz_payload[f"{source_prefix}__true_target_category_v1_indices"] = np.stack([
                    np.asarray(paired_rows[int(row["sentence_id"])]["candidate_indices"], dtype=np.int64)
                    for row in selected_source_rows
                ])
        source_mismatch_coverage.append({
            "family": family, "cell": cell, "candidate_source": source_name,
            "source_path": source_record["source_path"],
            "source_rows": int(len(source_rows)),
            "source_sentence_ids_unique": int(len({int(row["sentence_id"]) for row in source_rows})) == len(source_rows),
            "source_rows_covered_by_category_audit": int(sum(
                int(row["sentence_id"]) in mismatches_by_family[family] for row in source_rows
            )),
            "target_proposal_mismatches": int(len(mismatched_source_ids)),
            "target_proposal_mismatch_rate": float(len(mismatched_source_ids) / len(source_rows)) if source_rows else None,
            "measured_cell_applicable": True,
        })

    candidate_checks["historical_candidate_sources"] = source_mismatch_coverage

    # Metadata and outputs are committed only inside the designated repair directory.
    _write_csv(out_dir / "mismatch_by_expression.csv", mismatch_rows, [
        "family", "sentence_id", "ref_id", "image_id", "eval_split", "ann_id",
        "true_target_category_id", "true_target_category_name", "target_proposal_index",
        "target_proposal_assigned_category_id", "target_proposal_assigned_category_name",
        "target_proposal_assigned_gt_object_id", "target_proposal_max_iou_to_coco_gt",
        "target_proposal_iou_to_referring_ann", "category_mismatch", "mismatch_type",
    ])
    _write_csv(out_dir / "mismatch_by_category.csv", mismatch_by_category, [
        "family", "true_target_category_id", "true_target_category_name", "n_expressions",
        "n_mismatches", "mismatch_rate", "n_unknown_target_proposal_category",
    ])
    _write_csv(out_dir / "mismatch_category_confusion.csv", confusion_rows, [
        "family", "true_target_category_id", "true_target_category_name",
        "target_proposal_category_id", "target_proposal_category_name",
        "target_proposal_assigned_gt_object_id", "n_expressions",
    ])
    _write_csv(out_dir / "mismatch_by_image.csv", image_rows, [
        "family", "image_id", "n_expressions", "n_mismatches", "n_distinct_target_objects",
    ])
    _write_csv(out_dir / "mismatch_by_object.csv", object_rows, [
        "family", "image_id", "ann_id", "true_target_category_id", "true_target_category_name",
        "n_refs", "n_expressions", "n_mismatches", "mismatch_rate", "proposal_category_counts_json",
    ])
    if corrected_constructed:
        _write_csv(out_dir / "availability_intersection.csv", availability_rows, [
            "family", "cell", "k", "cell_kind", "same_category_proposals_required",
            "evaluation_universe_rows", "historical_old_available_rows",
            "corrected_true_category_available_rows", "old_new_common_rows",
            "old_only_rows_lost_under_corrected_category", "corrected_only_rows_not_in_historical_cohort",
            "old_common_retention", "new_common_retention", "new_variant", "comparison_cohort",
        ])
        _write_csv(out_dir / "candidate_geometry_supply_intersection_summary.csv", geometry_summary_rows, [
            "family", "cell", "variant", "n_proposals_per_candidate_set", "comparison_cohort",
            "n_expressions", "n_images", "n_distinct_gt_object_references", "total_proposals",
            "n_proposals_sum", "n_proposals_mean_per_expression", "n_proposals_median_per_expression",
            "n_matched_proposals_sum", "n_matched_proposals_mean_per_expression",
            "n_unmatched_to_coco_object_iou_ge_0_5_sum",
            "n_unmatched_to_coco_object_iou_ge_0_5_mean_per_expression",
            "n_distinct_matched_gt_objects_sum", "n_distinct_matched_gt_objects_mean_per_expression",
            "n_distinct_matched_gt_objects_median_per_expression", "duplicate_excess_proposals_sum",
            "duplicate_excess_proposals_mean_per_expression", "duplicate_gt_object_pairs_sum",
            "duplicate_gt_object_pairs_mean_per_expression", "n_proposals_assigned_true_target_object_sum",
            "n_proposals_assigned_target_proposal_gt_object_sum", "n_proposal_iou_pairs_ge_0_5_sum",
            "objectness_mean_mean_over_expressions", "objectness_p10_mean_over_expressions",
            "objectness_median_mean_over_expressions", "objectness_p90_mean_over_expressions",
            "area_px2_mean_mean_over_expressions", "area_px2_p10_mean_over_expressions",
            "area_px2_median_mean_over_expressions", "area_px2_p90_mean_over_expressions",
            "area_fraction_mean_mean_over_expressions", "area_fraction_p10_mean_over_expressions",
            "area_fraction_median_mean_over_expressions", "area_fraction_p90_mean_over_expressions",
            "max_iou_to_coco_object_mean_mean_over_expressions",
            "max_iou_to_coco_object_p10_mean_over_expressions",
            "max_iou_to_coco_object_median_mean_over_expressions",
            "max_iou_to_coco_object_p90_mean_over_expressions",
            "iou_to_true_target_object_mean_mean_over_expressions",
            "iou_to_true_target_object_p10_mean_over_expressions",
            "iou_to_true_target_object_median_mean_over_expressions",
            "iou_to_true_target_object_p90_mean_over_expressions",
            "mean_geometric_duplicate_pair_fraction", "mean_gt_object_duplicate_fraction",
        ])
        _write_csv(out_dir / "candidate_geometry_supply_intersection_per_expression.csv", geometry_expression_rows, [
            "family", "cell", "variant", "sentence_id", "ref_id", "image_id", "ann_id",
            "true_target_category_id", "n_proposals", "n_matched_proposals",
            "n_unmatched_to_coco_object_iou_ge_0_5", "n_distinct_matched_gt_objects",
            "duplicate_excess_proposals", "duplicate_gt_object_pairs",
            "n_proposals_assigned_true_target_object", "n_proposals_assigned_target_proposal_gt_object",
            "n_proposal_iou_pairs_ge_0_5", "proposal_iou_pair_count",
            "objectness_mean", "objectness_p10", "objectness_median", "objectness_p90",
            "area_px2_mean", "area_px2_p10", "area_px2_median", "area_px2_p90",
            "area_fraction_mean", "area_fraction_p10", "area_fraction_median", "area_fraction_p90",
            "max_iou_to_coco_object_mean", "max_iou_to_coco_object_p10",
            "max_iou_to_coco_object_median", "max_iou_to_coco_object_p90",
            "iou_to_true_target_object_mean", "iou_to_true_target_object_p10",
            "iou_to_true_target_object_median", "iou_to_true_target_object_p90",
            "geometric_duplicate_pair_fraction", "gt_object_duplicate_fraction",
            "unique_gt_object_ids_json", "proposals_per_gt_object_json",
        ])
        candidate_npz_path = out_dir / "candidate_sets_true_target_category_v1.npz"
        _write_npz(candidate_npz_path, candidate_npz_payload)
    else:
        candidate_npz_path = None

    # Stable source-level reports preserve the historical cohorts separately. The
    # baseline geometry artifacts are required even when no correction is warranted.
    geometry_summary_columns = [
        "family", "cell", "candidate_source", "source_path", "variant",
        "n_proposals_per_candidate_set", "comparison_cohort",
    ] + sorted({
        key for row in baseline_geometry_summary_rows for key in row
        if key not in {
            "family", "cell", "candidate_source", "source_path", "variant",
            "n_proposals_per_candidate_set", "comparison_cohort",
        }
    })
    _write_csv(out_dir / "candidate_geometry_historical_source_summary.csv", baseline_geometry_summary_rows,
               geometry_summary_columns)
    sensitivity_summary_columns = [
        "family", "cell", "candidate_source", "source_path", "variant",
        "n_proposals_per_candidate_set", "comparison_cohort",
    ] + sorted({
        key for row in sensitivity_geometry_summary_rows for key in row
        if key not in {
            "family", "cell", "candidate_source", "source_path", "variant",
            "n_proposals_per_candidate_set", "comparison_cohort",
        }
    })
    if corrected_constructed:
        _write_csv(out_dir / "candidate_geometry_sensitivity_summary.csv",
                   sensitivity_geometry_summary_rows, sensitivity_summary_columns)
        _write_csv(out_dir / "candidate_geometry_sensitivity_per_expression.csv",
                   sensitivity_geometry_expression_rows,
                   [
                       "family", "cell", "candidate_source", "source_path", "variant",
                       "sentence_id", "ref_id", "image_id", "ann_id",
                   ] + sorted({
                       key for row in sensitivity_geometry_expression_rows for key in row
                       if key not in {
                           "family", "cell", "candidate_source", "source_path", "variant",
                           "sentence_id", "ref_id", "image_id", "ann_id",
                       }
                   }))
        all_availability_rows = [
            {
                **row,
                "candidate_source": "manifest_candidate_supply_universe",
                "source_path": "cache/manifests*/same_category_test{A,B}.jsonl",
            }
            for row in availability_rows
        ] + source_availability_summary
        availability_columns = [
            "family", "cell", "candidate_source", "source_path", "k", "cell_kind",
            "same_category_proposals_required", "evaluation_universe_rows",
            "historical_old_available_rows", "historical_evaluated_rows",
            "historical_candidate_available_rows", "corrected_true_category_available_rows",
            "old_new_common_rows", "old_only_rows_lost_under_corrected_category",
            "corrected_only_rows_not_in_historical_cohort",
            "historical_rows_lost_under_corrected_category", "old_common_retention",
            "new_common_retention", "historical_retention", "mismatched_target_proposal_rows",
            "old_variant", "new_variant", "candidate_variant", "comparison_cohort",
        ]
        _write_csv(out_dir / "availability_intersection.csv", all_availability_rows,
                   availability_columns)
    _write_csv(out_dir / "candidate_geometry_historical_source_per_expression.csv",
               baseline_geometry_expression_rows,
               [
                   "family", "cell", "candidate_source", "source_path", "variant",
                   "sentence_id", "ref_id", "image_id", "ann_id",
               ] + sorted({
                   key for row in baseline_geometry_expression_rows for key in row
                   if key not in {
                       "family", "cell", "candidate_source", "source_path", "variant",
                       "sentence_id", "ref_id", "image_id", "ann_id",
                   }
               }))
    _write_csv(out_dir / "source_cohort_mismatch_coverage.csv", source_mismatch_coverage, [
        "family", "cell", "candidate_source", "source_path", "source_rows",
        "source_sentence_ids_unique", "source_rows_covered_by_category_audit",
        "target_proposal_mismatches", "target_proposal_mismatch_rate",
        "measured_cell_applicable",
    ])
    _write_csv(out_dir / "source_candidate_availability.csv", source_candidate_availability, [
        "family", "cell", "candidate_source", "source_path", "sentence_id", "ref_id",
        "image_id", "ann_id", "true_target_category_id", "historical_candidate_available",
        "corrected_candidate_available", "in_old_new_common_intersection",
    ])

    summary = {
        "schema": "research-repair-v1-candidate-summary-v1",
        "artifact": "research_repair_v1_candidate_category_audit",
        "audit_run_id": out_dir.name[len("audit_run_"):] if out_dir.name.startswith("audit_run_") else None,
        "protocol": "Research Repair v1 C",
        "status": "AUDIT_COMPLETE_MISMATCH_FOUND" if mismatch_total else "AUDIT_COMPLETE_NO_MISMATCH",
        "candidate_variant": VARIANT if corrected_constructed else None,
        "candidate_variant_constructed": bool(corrected_constructed),
        "candidate_variant_reason": (
            "at least one target proposal category differs from its RefCOCO+ ann_id category"
            if corrected_constructed else "mismatch count is zero; protocol says no construction"
        ),
        "inputs": {
            "refs": str(REFS_PATH.relative_to(ROOT).as_posix()),
            "instances": str(INSTANCES_PATH.relative_to(ROOT).as_posix()),
            "coco_gt": str(COCO_PATH.relative_to(ROOT).as_posix()),
            "coco_gt_sha256": supplemental_manifest["inputs"][0]["sha256"],
            "baseline_n_files": int(supplemental_manifest["baseline_n_files"]),
            "supplemental_manifest": str(SUPPLEMENTAL_MANIFEST_PATH.relative_to(ROOT).as_posix()),
            "prepare_path_gap": supplemental_manifest["source_path_gap"],
            "iou_threshold": COCO_IOU_THRESHOLD,
            "target_equivalence_iou_threshold": TARGET_EQ_IOU_THRESHOLD,
            "proposal_category_assignment": "best-IoU COCO object over the full frozen GT object list, including crowd rows; category is UNKNOWN/-1 below IoU 0.5",
            "candidate_families": list(FAMILIES),
            "candidate_cells": {k: list(v) for k, v in FAMILY_CELLS.items()},
            "frozen_seed_identity_checked": list(SEEDS),
            "new_feature_extraction": 0,
            "new_training": 0,
            "new_model_forward": 0,
            "GPU_used": False,
        },
        "mismatch": {
            "total_expressions": int(len(mismatch_rows)),
            "total_mismatches": mismatch_total,
            "overall_rate": float(mismatch_total / len(mismatch_rows)) if mismatch_rows else None,
            "by_family": mismatch_by_fam,
            "summary_dimensions": ["expression", "true_category", "image", "target_GT_object_ann_id", "proposal_assigned_category/object"],
            "unmatched_wording": "proposal not assigned to a COCO object at IoU >= 0.5; never described as nothing",
        },
        "availability": availability_rows,
        "historical_candidate_source_scope": {
            "family_cells": {family: list(cells) for family, cells in FAMILY_CELLS.items()},
            "sources": source_mismatch_coverage,
            "source_cohorts_kept_separate": True,
            "mismatch_audit_universe": "frozen P1 random K5 rows; all frozen hard/dose source cohort sentence_id/ref_id/image_id rows verified as subsets and summarized in source_cohort_mismatch_coverage.csv",
        },
        "candidate_geometry": {
            "baseline_historical_sources": baseline_geometry_summary_rows,
            "corrected_common_intersections": sensitivity_geometry_summary_rows,
            "unmatched_wording": "proposal not assigned to a COCO object at IoU >= 0.5; never described as nothing",
        },
        "candidate_checks": candidate_checks,
        "output_files": {
            "expression_audit": "mismatch_by_expression.csv",
            "category_summary": "mismatch_by_category.csv",
            "category_confusion": "mismatch_category_confusion.csv",
            "image_summary": "mismatch_by_image.csv",
            "object_summary": "mismatch_by_object.csv",
            "availability_intersection": "availability_intersection.csv" if corrected_constructed else None,
            "candidate_source_availability": "source_candidate_availability.csv",
            "source_mismatch_coverage": "source_cohort_mismatch_coverage.csv",
            "candidate_geometry_summary": "candidate_geometry_historical_source_summary.csv",
            "candidate_geometry_per_expression": "candidate_geometry_historical_source_per_expression.csv",
            "candidate_geometry_supply_intersection_summary": "candidate_geometry_supply_intersection_summary.csv" if corrected_constructed else None,
            "candidate_geometry_supply_intersection_per_expression": "candidate_geometry_supply_intersection_per_expression.csv" if corrected_constructed else None,
            "candidate_geometry_sensitivity_summary": "candidate_geometry_sensitivity_summary.csv" if corrected_constructed else None,
            "candidate_geometry_sensitivity_per_expression": "candidate_geometry_sensitivity_per_expression.csv" if corrected_constructed else None,
            "versioned_candidate_indices": str(candidate_npz_path.relative_to(ROOT).as_posix()) if candidate_npz_path else None,
        },
        "output_artifacts": {
            "versioned_candidate_indices": (
                {
                    "path": str(candidate_npz_path.relative_to(ROOT).as_posix()),
                    "size_bytes": int(candidate_npz_path.stat().st_size),
                    "sha256": _sha256(candidate_npz_path),
                }
                if candidate_npz_path is not None else None
            ),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json(out_dir / "summary.json", summary)
    _log(f"completed: mismatches={mismatch_total}/{len(mismatch_rows)}; corrected construction={corrected_constructed}")
    return summary


def _verify_phase1f_candidate_artifact_for_cell(
    cell: str, paired_rows: Sequence[Mapping[str, Any]], old_candidates: Mapping[int, np.ndarray]
) -> Dict[str, Any]:
    """Verify the same-category manifest reconstructs the saved RPN row arrays."""
    path = _candidate_indices_npz_path("RPN", cell)
    with np.load(path, allow_pickle=False) as archive:
        ids = np.asarray(archive["sentence_id"], dtype=np.int64)
        refs = np.asarray(archive["ref_id"], dtype=np.int64)
        images = np.asarray(archive["image_id"], dtype=np.int64)
        candidates = np.asarray(archive["candidate_indices"], dtype=np.int64)
        targets = np.asarray(archive["target_index"], dtype=np.int64)
    positions = {int(sid): i for i, sid in enumerate(ids.tolist())}
    if len(positions) != ids.size:
        raise CandidateAuditError(f"{cell}: archived sentence_id is not unique")
    n_checked = 0
    for sid, expected in old_candidates.items():
        pos = positions.get(int(sid))
        if pos is None:
            # Dose uses the frozen same8 cohort; old hard5/hard10 use their supply cohort.
            raise CandidateAuditError(f"{cell}: manifest-eligible sentence_id={sid} absent from archive")
        if not np.array_equal(candidates[pos], np.asarray(expected, dtype=np.int64)):
            raise CandidateAuditError(f"{cell}/sentence{sid}: manifest-derived old indices differ")
        if int(candidates[pos, 0]) != int(targets[pos]):
            raise CandidateAuditError(f"{cell}/sentence{sid}: target slot is not canonical slot zero")
        n_checked += 1
    # Also check exact archive row identities against all candidate rows in paired data.
    paired_ids = {int(row["sentence_id"]) for row in paired_rows}
    if not paired_ids.issubset(positions):
        raise CandidateAuditError(f"{cell}: paired candidate rows missing from frozen archive")
    return {"applicable": True, "rows_checked": n_checked, "archived_rows": int(ids.size),
            "candidate_indices_match": True}


__all__ = [
    "CandidateAuditError", "CELL_SPECS", "FAMILIES", "FAMILY_CELLS", "VARIANT",
    "candidate_audit_run_dir",
    "write_current_audit_pointer",
    "assign_proposals_to_objects", "build_versioned_candidate_indices", "candidate_geometry",
    "load_prediction_identity", "original_candidate_indices", "run_candidate_audit",
    "summarize_geometry_rows", "prepare_supplemental_input_manifest",
    "verify_supplemental_input_manifest", "availability_intersection_counts",
    "dose_common_cohort",
]
